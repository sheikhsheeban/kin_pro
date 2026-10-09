"""Phase 2 acceptance with the real models: Gemma 4 E2B (LiteRT-LM), EmbeddingGemma, SQLCipher, Piper.

The end-to-end test speaks into the (fake) microphone with Piper-generated speech,
held in RAM only, so the whole path runs: audio -> VAD -> Moonshine -> router ->
encrypted memory -> retrieval -> Gemma -> gate -> Piper.
Skipped when the models have not been downloaded.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import numpy as np
import pytest

from kin.app import KinApp
from kin.audio import build_engines
from kin.brain import build_brain
from kin.config import load_config
from kin.core.bus import EventBus
from kin.core.events import ActivationEvent, AttendingEnded, MemoryStored, ReplyCompleted, TranscriptEvent
from tests.conftest import FRAME, MODELS, FakeInputStream, Recorder

LLM = MODELS / "llm" / "gemma-4-E2B-it.litertlm"
GGUF = MODELS / "llm" / "Qwen3.5-2B-Q4_K_M.gguf"
PRIYA = "I'm meeting Priya at 4 pm at the library cafe."

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not LLM.exists(), reason="Gemma 4 E2B not downloaded (uv run kin download-models)"),
]

RECALL_SYSTEM = None


def _recall_prompt(question: str, memories: list[tuple[int, str]]) -> tuple[str, str]:
    from kin.brain.prompt import load_prompt

    lines = "\n".join(f"[#{i}] (Wed 01 Oct, 09:12) {t}" for i, t in memories)
    return load_prompt("recall_system"), f"Now: Wednesday 01 October 2026, 14:00\nMemories:\n{lines}\n\nQuestion: {question}"


MEMS = [(7, PRIYA), (9, "Buy oat milk on the way home."), (12, "Rahul is allergic to peanuts.")]


@pytest.fixture  # per test: never hold a Gemma engine (~2.6 GB commit) while other tests load the full stack
def litert():
    from kin.brain.llm import LiteRTBackend

    cfg = load_config("laptop")
    llm = LiteRTBackend(LLM, context_tokens=2048, threads=4, cache_dir=cfg.model_path("llm/cache"))
    yield llm
    llm.close()


def test_litert_recall_cites_and_abstains(litert):
    from kin.brain.gate import Gate

    gate = Gate(0.5)
    answer = litert.generate(*_recall_prompt("When am I meeting Priya?", MEMS), max_output_tokens=80)
    print(f"\nanswer: {answer!r} · {litert.last_stats}")
    verdict = gate.check(answer.strip(), {7, 9, 12})
    assert verdict.speakable and 7 in verdict.citations and "4" in answer
    assert litert.last_stats.ttft_s is not None and litert.last_stats.output_tokens > 3

    unknown = litert.generate(*_recall_prompt("What is my sister's birthday?", MEMS), max_output_tokens=40)
    print(f"unknown: {unknown!r}")
    assert gate.check(unknown.strip(), {7, 9, 12}).reason == "abstention"


def test_litert_cancel_stops_generation(litert):
    cancel = threading.Event()
    chunks = 0
    for _ in litert.stream("You are verbose.", "Count from 1 to 200 in words.", max_output_tokens=400, cancel=cancel):
        chunks += 1
        if chunks == 3:
            cancel.set()
    assert litert.last_stats.stop_reason == "cancelled" and chunks <= 4


def test_litert_context_cap_is_enforced(litert):
    assert litert.context_tokens == 2048
    assert litert.count_tokens("hello world") in range(1, 5)


@pytest.mark.skipif(not GGUF.exists(), reason="GGUF fallback not downloaded")
def test_llamacpp_fallback_recall():
    pytest.importorskip("llama_cpp")
    from kin.brain.gate import Gate
    from kin.brain.llm import LlamaCppBackend

    llm = LlamaCppBackend(GGUF, context_tokens=2048, threads=4)
    try:
        answer = llm.generate(*_recall_prompt("When am I meeting Priya?", MEMS), max_output_tokens=120)
        print(f"\nfallback answer: {answer!r}")
        assert "<think>" not in answer
        assert 7 in Gate(0.5).check(answer.strip(), {7, 9, 12}).citations
    finally:
        llm.close()


def test_embeddinggemma_separates_relevant_from_unrelated():
    from kin.memory.embed import GemmaEmbedder, find_embedding_model

    cfg = load_config("laptop")
    emb = GemmaEmbedder(find_embedding_model(cfg.model_path("embedding")), "q4", 256)
    docs = np.stack([emb.embed_document(t) for _, t in MEMS])
    assert docs.shape == (3, 256) and np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-4)
    priya = docs @ emb.embed_query("When am I meeting Priya?")
    rahul = docs @ emb.embed_query("Does Rahul have any allergies?")
    unknown = docs @ emb.embed_query("What is my sister's birthday?")
    print(f"\npriya {priya.round(3)} rahul {rahul.round(3)} unknown {unknown.round(3)}")
    floor = 0.5  # raw cosine separation (the gate now thresholds the hybrid confidence, memory.retrieval.answer_floor)
    assert priya.argmax() == 0 and priya.max() >= floor
    assert rahul.argmax() == 2 and rahul.max() >= floor
    assert unknown.max() < floor
    emb.close()


# --- full stack ---------------------------------------------------------------------------


@pytest.fixture
async def real_kin(tmp_path, monkeypatch):
    monkeypatch.setenv("KIN_DB_KEY", "ab" * 32)
    cfg = load_config("laptop")
    cfg.data_dir = tmp_path
    cfg.tts.sink = "null"  # synthesise for real, but do not play through the laptop speakers
    cfg.tts.ack_delay_s = 30  # keep the timing measurement about the answer
    streams: list[FakeInputStream] = []

    def stream_factory(**kw):
        streams.append(FakeInputStream(**kw))
        return streams[-1]

    bus = EventBus()
    rec = Recorder(bus)
    app = KinApp(
        cfg, bus=bus, engines_factory=build_engines, stream_factory=stream_factory,
        device_namer=lambda _d: "piper voice", brain_factory=build_brain, gateway_factory=None,
    )
    await app.start()
    await app.wait_brain()
    assert app.brain_status == "ready", app.brain_status
    yield app, streams, rec
    await app.stop()


def _speech(app, text: str) -> list[np.ndarray]:
    from scipy.signal import resample_poly

    tts = app.brain.speaker.tts
    audio = tts.synthesize(text)
    x = resample_poly(audio, 16000, tts.sample_rate).astype(np.float32) * 0.8
    pad = np.zeros(8000, np.float32)
    x = np.concatenate([pad, x, pad, pad])
    n = len(x) // FRAME
    return list(x[: n * FRAME].reshape(n, FRAME))


async def _play(stream: FakeInputStream, frames: list[np.ndarray]) -> None:
    def run() -> None:
        t0 = time.monotonic()
        for i, f in enumerate(frames):
            stream.push([f])
            delay = t0 + (i + 1) * FRAME / 16000 - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    await asyncio.to_thread(run)


async def test_spoken_memory_then_spoken_question_end_to_end(real_kin):
    """Acceptance: say the Priya sentence, later ask about it; Kin answers aloud citing the memory."""
    app, streams, rec = real_kin
    await _play(streams[-1], _speech(app, "I'm meeting Priya at 4 pm at the library cafe."))
    await asyncio.sleep(1.0)
    await app.brain.drain()
    await app.brain.flush()  # facts are extracted when the conversation ends (two-tier memory, Prompt 2)
    heard = [e.text for e in rec.of(TranscriptEvent) if e.final]
    print(f"\nheard: {heard} -> tags {[(t.label, t.score, t.margin, t.decision) for t in app.brain.recent_tags]}")
    stored = next(iter(rec.of(MemoryStored)), None)
    assert stored is not None, f"nothing stored from {heard}"
    assert "priya" in stored.text.lower() and stored.kind in ("plan", "commitment")

    # Moonshine mishears this synthetic voice now and then ("When am I needing prayer?"; measured standalone, outside
    # Kin). STT accuracy is measured by scripts/eval_spoken.py; this test checks the path, so a misheard question is
    # asked again, as a person would, up to three times.
    for _attempt in range(3):
        app.hardware.simulate_button(True)
        await app.bus.wait_for(ActivationEvent, timeout=2)
        await _play(streams[-1], _speech(app, "When am I meeting Priya?"))
        app.hardware.simulate_button(False)
        await app.bus.wait_for(AttendingEnded, timeout=5)
        reply = await app.bus.wait_for(ReplyCompleted, timeout=40)
        print(f"query: {reply.query!r}\nreply: {reply.text!r} outcome={reply.outcome} cites={reply.citations}"
              f" ttfa={reply.time_to_first_audio_ms}ms llm_ttft={reply.llm_ttft_ms}ms tok/s={reply.tokens_per_s}")
        if "priya" in reply.query.lower() and "meet" in reply.query.lower():
            break
    assert "priya" in reply.query.lower(), f"the question was misheard three times: {reply.query!r}"
    assert reply.outcome == "answered"
    assert str(stored.id) in reply.citations.split(",")
    # Moonshine writes the time either way ("4 pm" or "four p m"); the answer repeats what was stored.
    assert ("4" in reply.text or "four" in reply.text.lower()) and "[#" not in reply.text
    assert reply.time_to_first_audio_ms is not None


async def test_real_stack_abstains_on_unknown_and_works_offline(real_kin, monkeypatch):
    app, _streams, _rec = real_kin

    def refuse(*_a, **_k):
        raise OSError("network disabled for this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    await app.brain.memory.remember(PRIYA, "commitment", 0.9)
    await app.brain.memory.remember("Rahul is allergic to peanuts.", "fact_about_person", 0.85)

    known = await app.brain.handle_query("Is Rahul allergic to anything?")
    print(f"\nknown: {known.text!r} ({known.outcome}, cites {known.citations})")
    assert known.outcome == "answered" and "peanut" in known.text.lower()

    unknown = await app.brain.handle_query("What is my sister's birthday?")
    print(f"unknown: {unknown.text!r} ({unknown.outcome}), closest #{unknown.closest_id} sim {unknown.closest_similarity}")
    assert unknown.outcome == "abstained" and "remember" in unknown.text.lower()


async def test_component_ram_is_reported(real_kin):
    app, _s, _r = real_kin
    stats = app.system_stats()
    print("\n" + "\n".join(f"  {k:32s} {v['ram_mb']:7.1f} MB  {v['load_s']:5.2f}s" for k, v in stats["components"].items()))
    print(f"  total RSS {stats['rss_mb']:.0f} MB")
    assert {"llm", "tts", "embedder + memory", "speech (VAD, wake word, STT)"} <= set(stats["components"])


def test_pi5_profile_stays_under_ram_target(monkeypatch):
    """Pi 5 mode: every model loaded (incl. lazy Moonshine Small) plus one full query stays under 3.2 GB.

    Measured in a fresh process so models held by other tests cannot inflate the number.
    """
    import json
    import subprocess
    import sys

    from tests.conftest import ROOT

    env = {**__import__("os").environ, "KIN_DB_KEY": "cd" * 32}
    out = subprocess.run([sys.executable, "-m", "tests.ram_probe", "pi5"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=600)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, out.stderr[-2000:]
    r = json.loads(line[7:])
    print(f"\npi5 fresh process: RSS {r['rss_mb']} MB (unique {r['uss_mb']} MB) · reply {r['reply']!r}")
    for k, v in r["components"].items():
        print(f"  {k:32s} {v['ram_mb']:7.1f} MB  {v['load_s']:5.2f}s")
    assert r["outcome"] == "answered"
    assert r["rss_mb"] < load_config("pi5").runtime.ram_target_mb


async def test_wake_phrase_heard_in_speech_activates_when_the_wake_model_does_not(real_kin):
    """A voice the acoustic wake model scores under its threshold: the transcript fallback still activates Kin."""
    app, streams, rec = real_kin
    app.cfg.wake.threshold = 1.01  # the acoustic model can never fire
    await app.brain.memory.remember(PRIYA, "commitment", 0.9)
    await _play(streams[-1], _speech(app, "Hey Jarvis, when am I meeting Priya?"))
    act = await app.bus.wait_for(ActivationEvent, timeout=10)
    reply = await app.bus.wait_for(ReplyCompleted, timeout=40)
    print(f"\nactivated by {act.source}; query {reply.query!r} -> {reply.text!r}")
    assert act.source == "wake_phrase"
    assert "priya" in reply.query.lower() and "jarvis" not in reply.query.lower()
    assert reply.outcome == "answered" and ("4" in reply.text or "four" in reply.text.lower())
