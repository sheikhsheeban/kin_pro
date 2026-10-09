from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from kin.app import KinApp
from kin.audio.pipeline import Engines
from kin.audio.stt import SttLine
from kin.config import KinConfig, load_config
from kin.core.bus import EventBus
from kin.core.events import Event

SR = 16000
FRAME = 512
WAKE_MARKER = 0.4242  # a frame containing this sample value makes FakeWake fire

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def speech(n_frames: int = 1, amp: float = 0.3) -> list[np.ndarray]:
    t = np.arange(n_frames * FRAME) / SR
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return list(x.reshape(n_frames, FRAME))


def silence(n_frames: int = 1) -> list[np.ndarray]:
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n_frames)]


def wake_frames(n_frames: int = 4) -> list[np.ndarray]:
    frames = speech(n_frames)
    frames[-1] = frames[-1].copy()
    frames[-1][0] = WAKE_MARKER
    return frames


class FakeVAD:
    def speech_prob(self, frame: np.ndarray) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > 0.05 else 0.0

    def reset(self) -> None:
        pass


class FakeWake:
    def __init__(self) -> None:
        self.resets = 0

    def process(self, frame: np.ndarray) -> float | None:
        return 0.99 if np.any(np.isclose(frame, WAKE_MARKER, atol=1e-6)) else 0.01

    def reset(self) -> None:
        self.resets += 1


class FakeSTT:
    """Emits a partial line per feed and a final line on stop."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.samples = 0
        self.fed: list[np.ndarray] = []
        self._cb: Callable[[SttLine], None] | None = None
        self.voiced = False
        self.closed = False

    def start(self, on_line: Callable[[SttLine], None]) -> None:
        self._cb = on_line
        self.samples = 0
        self.voiced = False

    def feed(self, samples: np.ndarray) -> None:
        self.samples += len(samples)
        self.fed.append(samples.copy())
        self.voiced = self.voiced or bool(np.abs(samples).max(initial=0) > 0.05)
        if self._cb and self.voiced:  # like a real STT, silence yields no text
            self._cb(SttLine(id=1, text=f"{self.name} heard {self.samples}", final=False))

    def stop(self) -> None:
        if self._cb and self.voiced:
            self._cb(SttLine(id=1, text=f"{self.name} heard {self.samples}", final=True))
        self._cb = None

    def close(self) -> None:
        self.closed = True


class FakeInputStream:
    """Mimics sounddevice.InputStream; tests push frames into the callback."""

    def __init__(self, *, samplerate: int, blocksize: int, device, callback) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.callback = callback
        self.active = False
        self.closed = False
        self.closed_at: float | None = None

    def start(self) -> None:
        self.active = True

    def abort(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False
        self.closed = True
        self.closed_at = time.monotonic()

    def push(self, frames: list[np.ndarray]) -> None:
        for f in frames:
            if self.active and not self.closed:
                self.callback(f.reshape(-1, 1), len(f), None, None)


class Recorder:
    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        bus.subscribe(Event, self.events.append, name="test.recorder")

    def of(self, cls: type) -> list:
        return [e for e in self.events if isinstance(e, cls)]


class Harness:
    def __init__(self, app: KinApp, streams: list[FakeInputStream], rec: Recorder, engines: dict) -> None:
        self.app = app
        self.streams = streams
        self.rec = rec
        self.engines = engines

    @property
    def stream(self) -> FakeInputStream:
        return self.streams[-1]

    @property
    def bus(self) -> EventBus:
        return self.app.bus

    @property
    def hw(self):
        return self.app.hardware

    async def settle(self, timeout: float = 2.0) -> None:
        """Wait until the perception worker has drained its queue and events are delivered."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            p = self.app.pipeline
            if p is None or p._q.empty():
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.05)


def fake_engines_factory(store: dict) -> Callable[[KinConfig], Engines]:
    def factory(cfg: KinConfig) -> Engines:
        wake = FakeWake()
        passive = FakeSTT("tiny")

        def make_query() -> FakeSTT:
            store["query"] = FakeSTT("small")
            return store["query"]

        store.update(wake=wake, passive=passive)
        return Engines(FakeVAD(), wake, passive, make_query, lazy_query=cfg.stt.lazy_load_query)

    return factory


def build_fake_app(profile: str = "laptop", query_timeout_s: float | None = None, brain_factory=None) -> Harness:
    """A KinApp with fake mic stream and fake models. Not started yet."""
    cfg = load_config(profile)
    cfg.vad.min_silence_ms = 96  # 3 frames, keeps tests quick
    if query_timeout_s is not None:
        cfg.stt.query_timeout_s = query_timeout_s
    streams: list[FakeInputStream] = []

    def stream_factory(**kw) -> FakeInputStream:
        s = FakeInputStream(**kw)
        streams.append(s)
        return s

    engines: dict = {}
    bus = EventBus()
    rec = Recorder(bus)
    app = KinApp(
        cfg,
        bus=bus,
        engines_factory=fake_engines_factory(engines),
        stream_factory=stream_factory,
        device_namer=lambda _d: "fake mic",
        brain_factory=brain_factory,
        gateway_factory=None,
    )
    return Harness(app, streams, rec, engines)


@pytest.fixture
async def make_harness():
    created: list[Harness] = []

    async def _make(profile: str = "laptop", **kwargs) -> Harness:
        h = build_fake_app(profile, **kwargs)
        await h.app.start()
        created.append(h)
        await h.settle()
        return h

    yield _make
    for h in created:
        await h.app.stop()


@pytest.fixture
async def harness(make_harness):
    return await make_harness()


def models_available() -> bool:
    return (MODELS / "silero_vad.onnx").exists() and any(MODELS.glob("**/tiny-streaming-en/*/encoder.ort"))


# --- Phase 2 fakes ----------------------------------------------------------------------

import re as _re
import secrets as _secrets
import threading as _threading

from kin.brain.llm import LLMBackend, LlmChunk, LlmSession

_STOP = set("i i'm im am is are was were a an the at to of on in for and or my me you your what when where who how do does did will with about".split())


def _words(text: str) -> list[str]:
    return [w for w in _re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP]


class FakeEmbedder:
    """Bag-of-words hashing embedder: shared content words give high cosine similarity."""

    name = "fake-bow"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        import zlib

        v = np.zeros(self.dim, np.float32)
        for w in _words(text):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        v[self.dim - 1] += 0.05  # never a zero vector
        return v / np.linalg.norm(v)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_document(self, text: str) -> np.ndarray:
        return self._vec(text)

    def close(self) -> None:
        pass


class FakeLLM(LLMBackend):
    """A scriptable stand-in for Gemma 4 E2B.

    In a session with tools, the user's question is routed by `fake_route` (a few obvious
    patterns -> a tool call, greetings -> a direct answer, anything else -> recall). Tool results
    and tool-less sessions are answered from the best-overlapping memory line, citing it.

    `script` fixes the text of every non-routing answer. `turns` scripts the session exactly: each
    item is a ToolCall (the model calls a tool) or a string (the model answers), consumed in order.
    The fixed warm-up exchange never consumes a scripted turn.
    """

    name = "fake-llm"

    def __init__(self, script: str | None = None, delay_s: float = 0.0, turns: list | None = None) -> None:
        super().__init__(context_tokens=2048)
        self.script = script
        self.delay_s = delay_s
        self.turns = list(turns) if turns is not None else None
        self.calls: list[tuple[str, str]] = []
        self.tools_offered: list[list[str]] = []
        self.sessions = 0
        self.cancelled = False
        self.extract_replies: list[str] = []  # scripted raw replies to extraction prompts, consumed in order
        self.dedup_answer = "ADD"  # what the ADD/UPDATE/DUPLICATE/CONTRADICTS question gets
        self.keep_answer = "yes"  # what the per-episode "anything worth keeping?" gate gets
        self.extractions = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 1

    def session(self, system, *, tools=(), max_output_tokens):
        self.sessions += 1
        return _FakeSession(self, system, [t.name for t in tools])

    @staticmethod
    def _answer(prompt: str) -> str:
        return _FakeLLMAnswers.answer(prompt)

    @property
    def queries(self) -> list[str]:
        """The user messages the model saw (no warm-ups, no tool results, no background extraction)."""
        from kin.brain.extract import DEDUP_SYSTEM, EXTRACT_SYSTEM, KEEP_SYSTEM
        from kin.brain.orchestrator import WARMUP_MESSAGE

        return [m for system, m in self.calls if m != WARMUP_MESSAGE and not m.startswith("TOOL RESULT:")
                and system not in (EXTRACT_SYSTEM, DEDUP_SYSTEM, KEEP_SYSTEM)]


def fake_route(query: str):
    """What a sensible model does with an obvious request (the real model is tested in test_tool_calls_models.py)."""
    from kin.brain.llm import ToolCall

    q = query.strip()
    low = q.lower().rstrip("?.!")
    if _re.match(r"^(hi|hello|hey|thanks|thank you|how are you|who are you|good (morning|evening|night))\b", low):
        return None
    m = _re.match(r"^(?:please\s+)?remind me to (.+)$", q, _re.I)
    if m:
        return ToolCall("add_task", {"text": m.group(1).rstrip(".?!"), "due": None})
    m = _re.match(r"^(?:please\s+)?(?:add|put) (.+?) (?:to|on) my (?:\w+ )?(?:list|to-?do list)$", low)
    if m:
        return ToolCall("add_task", {"text": m.group(1)})
    if _re.search(r"\b(on my list|my reminders|my tasks)\b", low):
        return ToolCall("list_tasks", {})
    m = _re.match(r"^(?:mark|tick off) (.+?)(?: as done)?$", low)
    if m:
        return ToolCall("complete_task", {"text": m.group(1)})
    if _re.match(r"^(that'?s|that is) (wrong|not true|not right)|^i never said that", low):
        return ToolCall("retract_fact", {"which": None})
    m = _re.match(r"^(?:no|nope|actually)[,.!]?\s+(.+)$", q, _re.I)
    if m:
        return ToolCall("correct_fact", {"correction": q, "which": None})
    if _re.search(r"\bwhat did you remember (today|yesterday)\b", low):
        return ToolCall("review_memory", {"day": "yesterday" if "yesterday" in low else "today"})
    m = _re.match(r"^remember that (.+)$", q, _re.I)
    if m:
        return ToolCall("remember", {"text": m.group(1)})
    if _re.search(r"\b(weather|umbrella|rain)\b", low):
        return ToolCall("lookup", {"kind": "weather", "query": "", "when": None})
    return ToolCall("recall", {"question": q})


def fake_extract(prompt: str) -> str:
    """What a sensible extractor returns: one fact per line with a plan / fact / idea cue, text as heard."""
    import json

    from kin.brain.names import extract_names

    convo = prompt.split("Now the real conversation.", 1)[-1]  # not the few-shot examples
    router = StubRouter()
    kinds = {"commitment": "plan", "fact_about_person": "fact", "idea": "idea"}
    out = []
    for n, text in _re.findall(r"^\[(\d+)\] \S+ \S+ \w+: (.+)$", convo, flags=_re.M):
        tag = router.classify(text)
        if tag.label in kinds or _re.search(r"\b(password|code|number)\b", text, _re.I):
            out.append({"text": text, "kind": kinds.get(tag.label, "note"), "subject": None,
                        "people": sorted(extract_names(text)), "event_time": None, "lines": [int(n)]})
    return json.dumps(out)


class _FakeSession(LlmSession):
    def __init__(self, backend: FakeLLM, system: str, tools: list[str]) -> None:
        super().__init__(backend)
        self.system = system
        self.tools = tools
        self.prompt = ""
        backend.tools_offered.append(tools)

    def send(self, message, *, cancel=None):
        import time as _t

        from kin.brain.llm import LlmStats, ToolCall, ToolResult
        from kin.brain.orchestrator import WARMUP_MESSAGE

        b: FakeLLM = self.backend
        text_in = f"TOOL RESULT: {message.content}" if isinstance(message, ToolResult) else message
        b.calls.append((self.system, text_in))
        b.last_stats = LlmStats(prompt_tokens=b.count_tokens(self.system + text_in), ttft_s=0.01)
        if message == WARMUP_MESSAGE:
            yield LlmChunk(text="Ready.")
            return
        from kin.brain.extract import DEDUP_SYSTEM, EXTRACT_SYSTEM, KEEP_SYSTEM

        if self.system == KEEP_SYSTEM:  # the per-episode gate before extraction
            yield LlmChunk(text=b.keep_answer)
            return
        if self.system == EXTRACT_SYSTEM:  # background fact extraction for one episode
            b.extractions += 1
            if not message.startswith("That was not valid"):
                self.prompt = message
            yield LlmChunk(text=b.extract_replies.pop(0) if b.extract_replies else fake_extract(self.prompt))
            return
        if self.system == DEDUP_SYSTEM:
            yield LlmChunk(text=b.dedup_answer)
            return
        if isinstance(message, ToolResult):
            self.prompt = message.content
        elif not message.startswith("That tool call was invalid"):
            self.prompt = message
        if b.turns:
            item = b.turns.pop(0)
        elif self.tools and not isinstance(message, ToolResult) and "\nMemories:\n" in message:
            item = b.script if b.script is not None else b._answer(self.prompt)  # facts given up front: answer
        elif self.tools and not isinstance(message, ToolResult):
            item = fake_route(message) or (b.script if b.script is not None else "Hello! Nice to talk to you.")
        else:
            item = b.script if b.script is not None else b._answer(self.prompt)
        if isinstance(item, ToolCall):
            yield LlmChunk(tool_call=item)
            return
        for word in item.split(" "):
            if cancel is not None and cancel.is_set():
                b.cancelled = True
                b.last_stats.stop_reason = "cancelled"
                return
            if b.delay_s:
                _t.sleep(b.delay_s)
            b.last_stats.output_tokens += 1
            yield LlmChunk(text=word + " ")
        b.last_stats.duration_s = 0.1


class StubRouter:
    """Stands in for the semantic router in fast tests (the fake embedder has no semantics).

    Sentences are tagged by a few obvious cues; sentences in `unsure` go to pending.
    The real router is tested with EmbeddingGemma in test_semantic_router.py.
    """

    ready = True

    def __init__(self, unsure: set[str] | None = None) -> None:
        from kin.brain.semantic_router import SemanticRouter

        self._pre = SemanticRouter({}, min_words=3).prefilter
        self.unsure = unsure or set()

    async def fit(self, *_args, **_kw) -> None:
        pass

    def prefilter(self, text: str):
        return self._pre(text)

    def classify(self, text: str, vector=None):
        from kin.brain.semantic_router import Tag

        pre = self._pre(text)
        if pre is not None:
            return pre
        low = text.lower()
        if text.strip() in self.unsure:
            return Tag("chatter", 0.5, 0.01, False)
        if _re.search(r"\b(meeting|appointment|tomorrow|at \d|\d ?pm|need to|have to|remind me)\b", low):
            return Tag("commitment", 0.8, 0.2, True)
        if _re.search(r"\b(allergic|loves|birthday|vegan|vegetarian|lives in|works at|is a)\b", low):
            return Tag("fact_about_person", 0.8, 0.2, True)
        if _re.search(r"\b(what if|we could|idea)\b", low):
            return Tag("idea", 0.8, 0.2, True)
        return Tag("chatter", 0.8, 0.2, True)


class FakeLookupClient:
    """Stands in for the gateway connection. Records every request; answers with canned results."""

    def __init__(self, result=None) -> None:
        from kin.gateway.providers.fake import FakeWeather

        self.requests: list[tuple[str, dict]] = []
        self.offline = False
        self.names: list[str] = []
        self._weather = FakeWeather()
        self._result = result

    def lookup(self, kind: str, params: dict):
        import json as _json

        from kin.gateway import schemas
        from kin.gateway.http import Exchange
        from kin.tools.lookup_client import LookupResult

        from pydantic import ValidationError

        self.requests.append((kind, params))
        try:
            req = schemas.LOOKUP_ADAPTER.validate_python({"kind": kind, "params": params})
        except ValidationError:  # like the real gateway: refused and audited, never a crash
            return LookupResult(False, "refused", "invalid", audit={"id": "fake1234", "kind": kind, "status": "refused"})
        audit = {"id": "fake1234", "kind": kind, "request": schemas.canonical(req), "upstream": [], "provider": "fake",
                 "response_bytes": 640, "reply_bytes": 300, "duration_ms": 5, "scrubbed": 0}
        if self._result is not None:
            return self._result
        if self.offline:
            return LookupResult(False, "refused", "offline", audit={**audit, "status": "refused"})
        ex = Exchange()
        data = self._weather.fetch(req.params, None, ex) if kind == "weather" else {"query": params, "results": []}
        audit["upstream"] = ex.requests
        assert _json.dumps(data)
        return LookupResult(True, "ok", None, data, audit={**audit, "status": "ok"})

    def set_names(self, salt: bytes, hashes: list[str]) -> dict:
        self.names = hashes
        return {"ok": True}

    def set_offline(self, offline: bool) -> dict:
        self.offline = offline
        return {"ok": True}

    def audit(self, limit: int = 100) -> list:
        return []

    def close(self) -> None:
        pass


class _FakeLLMAnswers:
    @staticmethod
    def answer(prompt: str) -> str:
        question = prompt.rsplit("Question:", 1)[-1]
        if "Memories:" not in prompt:
            return "Hello! Nice to talk to you."
        q = set(_words(question))
        best, best_score = None, 0
        for mid, text in _re.findall(r"^\[#(\d+)\] \([^)]*\) (.+)$", prompt, flags=_re.M):
            score = len(q & set(_words(text)))
            if score > best_score:
                best, best_score = (mid, text), score
        if best is None:
            return "I don't remember that."
        return f"You told me: {best[1].rstrip('.')} [#{best[0]}]."


class FakeTTS:
    sample_rate = 16000

    def __init__(self) -> None:
        self.said: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.said.append(text)
        return np.zeros(160, np.float32)


def make_fake_brain_factory(
    tmp_path: Path,
    llm: FakeLLM | None = None,
    tts: FakeTTS | None = None,
    store: dict | None = None,
    lookup_client=None,
    router=None,
    shortcut: bool = True,
    key: str | None = None,
):
    from kin.brain.orchestrator import Brain
    from kin.io.tts import NullSink, Speaker
    from kin.memory.service import MemoryService
    from kin.memory.store import MemoryStore
    from kin.tools.reminders import ReminderScheduler
    from kin.tools.tasks import TaskService

    key = key or _secrets.token_hex(32)
    store = store if store is not None else {}

    async def factory(cfg, bus, monitor):
        if not shortcut:  # tests of the tool-call path itself: never answer in one call
            cfg.memory.retrieval.strong_score = float("inf")
        memory = MemoryService(bus, lambda: FakeEmbedder(cfg.memory.dim), lambda dim: MemoryStore(tmp_path / "kin.db", key, dim),
                               retrieval=cfg.memory.retrieval)
        with monitor.measure("embedder + memory"):
            await memory.start()
        sink = NullSink()
        tasks = TaskService(bus, memory)
        brain = Brain(
            cfg, bus,
            llm_factory=lambda: llm or FakeLLM(),
            memory=memory,
            speaker=Speaker(tts, sink),
            router=router or StubRouter(),
            lookup_client=lookup_client,
            tasks=tasks,
        )
        brain.reminders = ReminderScheduler(bus, tasks, brain.announce, check_s=0.05)
        await brain.load_llm()
        await brain.start()
        store.update(brain=brain, sink=sink, key=key, db=tmp_path / "kin.db", tasks=tasks)
        return brain

    return factory
