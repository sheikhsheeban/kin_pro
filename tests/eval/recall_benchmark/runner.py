"""Run the recall benchmark through Kin's real memory pipeline, without audio.

Write path: each line of the week goes through `Brain._ingest` at its own timestamp (filler is
dropped; every other line becomes an utterance in an episode). At the end of each simulated day the
finished episodes are extracted into facts by Gemma 4 E2B (`Brain.extract_ready`), as Kin does when
idle. A line counts as kept when it is evidence for a fact.
Read path: each question goes through `Brain.handle_query` (tool call -> recall -> gate -> answer),
plus a plain `MemoryService.retrieve(question, 5)` to measure retrieval on its own. The reply's "why"
record (confidence, facts offered, shortcut or not) is kept with each question.

The harness only observes and drives; it does not change runtime behaviour. It pins the clock by
swapping a fixed clock into the modules that read wall time (memory service/store, prompt,
orchestrator), waits for the warm session between questions (an idle user), and stops the
background episode/extraction loops so episodes are extracted at deterministic points. LLM decoding is greedy
already (LiteRT-LM top_k=1, temperature 0; llama.cpp temperature 0).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import time as _time
from datetime import datetime
from pathlib import Path

from tests.eval.recall_benchmark import ASK_AT, Line, Question, load_questions, load_week
from tests.eval.recall_benchmark.scoring import QuestionResult, as_dict, check, failure_reason, percentile, summarise

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
TOP_K = 5

JUDGE_SYSTEM = "You grade answers. Reply with only yes or no."
JUDGE_PROMPT = (
    "Question: {q}\nReference answer: {ref}\nAssistant's answer: {ans}\n"
    "Does the assistant's answer give the same key facts as the reference, without contradicting it? "
    "Answering 'I don't remember' is no."
)


# --- fixed clock -----------------------------------------------------------------------------------


class FixedClock:
    def __init__(self, at: datetime) -> None:
        self.t = at.timestamp()

    def set(self, at: datetime) -> None:
        self.t = at.timestamp()


class _TimeProxy:
    """Stands in for the `time` module inside Kin modules: wall time is fixed, everything else is real."""

    def __init__(self, clock: FixedClock) -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.t

    def __getattr__(self, name):
        return getattr(_time, name)


def _datetime_proxy(clock: FixedClock):
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(clock.t, tz)

    return _FixedDatetime


@contextlib.contextmanager
def fixed_clock(clock: FixedClock):
    import kin.brain.orchestrator as orch
    import kin.brain.prompt as prompt
    import kin.memory.service as service
    import kin.memory.store as store
    import kin.tools.tasks as tasks

    proxy = _TimeProxy(clock)
    saved = [(m, "time", m.time) for m in (orch, prompt, service, store, tasks)] + [(orch, "datetime", orch.datetime)]
    try:
        for m, name, _ in saved:
            setattr(m, name, proxy if name == "time" else _datetime_proxy(clock))
        yield clock
    finally:
        for m, name, value in saved:
            setattr(m, name, value)


# --- provenance ------------------------------------------------------------------------------------


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(["git", "status", "--porcelain", "kin"], cwd=ROOT, capture_output=True, text=True).stdout
            return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def code_hash() -> str:
    """sha256 over kin/**/*.py, *.txt, *.yaml: identifies the runtime under test even without git."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "kin").rglob("*")):
        if p.suffix in (".py", ".txt", ".yaml") and "__pycache__" not in p.parts:
            h.update(p.relative_to(ROOT).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


# --- building Kin ----------------------------------------------------------------------------------


async def build_real(tmp: Path, overrides: dict | None = None):
    from kin.brain import build_brain
    from kin.config import load_config
    from kin.core.bus import EventBus
    from kin.core.metrics import ResourceMonitor

    os.environ["KIN_DB_KEY"] = secrets.token_hex(32)
    cfg = load_config("laptop")
    cfg.data_dir = tmp
    cfg.tts.enabled = False  # answers are text; TTS adds nothing to recall
    cfg.tts.sink = "null"
    cfg.tts.ack_delay_s = 600
    cfg.gateway.enabled = False  # no network: a lookup call is refused, as in offline mode
    for k, v in (overrides or {}).items():  # memory.retrieval settings under test (e.g. answer_floor=0 to calibrate)
        if k == "answer_model":  # llm.answer_model: a larger model writes the recall answers (Prompt 5, B7)
            cfg.llm.answer_model = v
        elif k in ("answer_facts", "list_facts"):
            setattr(cfg.memory, k, v)
        else:
            setattr(cfg.memory.retrieval, k, v)
    cache = ROOT / "data" / "cache"
    if cache.exists():  # router example vectors (from the config file, never user data)
        shutil.copytree(cache, tmp / "data" / "cache", dirs_exist_ok=True)
    bus = EventBus()
    await bus.start()
    brain = await build_brain(cfg, bus, ResourceMonitor())
    return brain, bus


async def build_fake(tmp: Path):
    """The fast-suite fakes (bag-of-words embedder, cue-word router, overlap LLM): plumbing only."""
    from kin.brain.orchestrator import Brain
    from kin.config import load_config
    from kin.core.bus import EventBus
    from kin.io.tts import NullSink, Speaker
    from kin.memory.service import MemoryService
    from kin.memory.store import MemoryStore
    from kin.tools.tasks import TaskService
    from tests.conftest import FakeEmbedder, FakeLLM, StubRouter

    cfg = load_config("laptop")
    cfg.data_dir = tmp
    cfg.llm.warm_session = False
    key = secrets.token_hex(32)
    bus = EventBus()
    await bus.start()
    memory = MemoryService(bus, lambda: FakeEmbedder(cfg.memory.dim), lambda dim: MemoryStore(tmp / "kin.db", key, dim))
    await memory.start()
    brain = Brain(cfg, bus, llm_factory=FakeLLM, memory=memory, speaker=Speaker(None, NullSink()),
                  router=StubRouter(), tasks=TaskService(bus, memory))
    await brain.load_llm()
    await brain.start()
    return brain, bus


# --- the run ---------------------------------------------------------------------------------------


class _TagRecorder:
    """Wraps Brain._publish_tag to see every passive decision (first decision and final one)."""

    def __init__(self, brain) -> None:
        self.events: list[tuple[str, str, str, str, float, float]] = []
        original = brain._publish_tag

        def record(text, tag, decision):
            self.events.append((text, decision, tag.label, tag.source, tag.score, tag.margin))
            return original(text, tag, decision)

        brain._publish_tag = record


async def _wait_warm(brain) -> None:
    t = brain._warm_task
    if t is not None and not t.done():
        await asyncio.gather(t, return_exceptions=True)


class _ExtractionRecorder:
    def __init__(self, brain) -> None:
        self.events = []
        original = brain._log_extraction

        def record(e):
            self.events.append(e)
            return original(e)

        brain._log_extraction = record


def _speaker(ln: Line, speakers: str) -> str | None:
    if speakers == "oracle":  # ground truth, as if Kin could identify voices
        return "owner" if ln.speaker == "me" else "other"
    return None  # what Kin really has: no speaker identification


async def ingest_week(brain, clock: FixedClock, lines: list[Line], progress=None, speakers: str = "none") -> tuple[list[dict], list[dict]]:
    rec = _TagRecorder(brain)
    ext = _ExtractionRecorder(brain)
    rows: list[dict] = []
    for i, ln in enumerate(lines):
        clock.set(ln.when)
        t0 = _time.perf_counter()
        await brain._ingest(ln.text, ts=ln.when.timestamp(), speaker=_speaker(ln, speakers), stt_model="benchmark-transcript")
        ms = (_time.perf_counter() - t0) * 1000
        text, decision, label, source, score, margin = rec.events[-1]
        assert text == ln.text, (text, ln.text)
        rows.append({"id": ln.id, "keep": ln.keep, "speaker": ln.speaker, "text": ln.text, "ingest_ms": round(ms, 1),
                     "first_decision": decision, "label": label, "source": source, "score": score, "margin": margin})
        last_of_day = i + 1 == len(lines) or lines[i + 1].when.date() != ln.when.date()
        if last_of_day:
            clock.set(ln.when.replace(hour=23, minute=30))
            t1 = _time.perf_counter()
            n = len(ext.events)
            await brain.close_episodes(force=True)
            done = await brain.extract_ready(wait_idle=False)
            if progress:
                ev = ext.events[n:]
                facts = sum(e.added + e.updated for e in ev)
                progress(f"  day {ln.day}: {done} episodes extracted ({sum(e.outcome == 'skipped' for e in ev)} skipped by the pre-filter, "
                         f"{sum(e.outcome == 'failed' for e in ev)} failed) -> {facts} facts in {_time.perf_counter() - t1:.1f} s")
    await brain.drain()
    # kept = the line is evidence for some fact (active or superseded)
    utt_line = {u.id: by for u in await brain.memory.utterances(None, 100_000) if (by := _line_id(lines, u.text))}
    facts = await brain.memory.recent(limit=100_000, status=None)
    cited = {utt_line[e] for f in facts for e in f.evidence if e in utt_line}
    for r in rows:
        r["decision"] = "stored" if r["id"] in cited else "dropped"
    episodes = [{"episode": e.episode_id, "lines": e.lines, "outcome": e.outcome, "added": e.added, "updated": e.updated,
                 "duplicates": e.duplicates, "ms": e.duration_ms, "llm_calls": e.llm_calls, "error": e.error}
                for e in ext.events]
    return rows, episodes


async def replay_week(brain, clock: FixedClock, lines: list[Line], saved: dict, speakers: str = "none") -> tuple[list[dict], list[dict]]:
    """Answer-stage runs: every line is heard as usual (utterances, vectors), but no episode is extracted;
    the facts are loaded exactly as a saved run extracted them (text, people, times, evidence, supersede
    chains). Extraction is deterministic and untouched by retrieval work, so this skips ~20 min per variant."""
    import numpy as _np

    rec = _TagRecorder(brain)
    rows: list[dict] = []
    for ln in lines:
        clock.set(ln.when)
        t0 = _time.perf_counter()
        await brain._ingest(ln.text, ts=ln.when.timestamp(), speaker=_speaker(ln, speakers), stt_model="benchmark-transcript")
        text, decision, label, source, score, margin = rec.events[-1]
        rows.append({"id": ln.id, "keep": ln.keep, "speaker": ln.speaker, "text": ln.text,
                     "ingest_ms": round((_time.perf_counter() - t0) * 1000, 1), "first_decision": decision, "label": label,
                     "source": source, "score": score, "margin": margin})
    await brain.drain()
    week = {ln.id: ln for ln in lines}
    utt = {u.text: u.id for u in await brain.memory.utterances(None, 100_000)}
    new_id: dict[int, int] = {}
    for f in sorted(saved["facts"], key=lambda f: f["id"]):
        ids = [i for i in f["lines"] if i in week]
        heard = max((week[i].when.timestamp() for i in ids), default=lines[-1].when.timestamp())
        evidence = [utt[week[i].text] for i in ids if week[i].text in utt]
        event = datetime.fromisoformat(f["event"]).timestamp() if f.get("event") else None
        for name in {*f["people"], *([f["subject"]] if f["subject"] not in (None, "owner") else [])}:
            if name and name[:1].isupper() and len(name.split()) <= 2:
                await brain.memory.add_person(name)
        vec = await brain.memory.embed_document(f["text"])
        fact = await brain.memory.run(lambda s, f=f, vec=vec, heard=heard, event=event, evidence=evidence: s.add_fact(
            f["text"], f["kind"], _np.asarray(vec), confidence=0.8, source=f.get("source", "extracted"), subject=f["subject"],
            people=f["people"], event_ts=event, heard_ts=heard, evidence=evidence))
        new_id[f["id"]] = fact.id
    for f in saved["facts"]:
        if f["status"] != "active" and f.get("superseded_by") in new_id:
            await brain.memory.supersede(new_id[f["id"]], new_id[f["superseded_by"]], f["status"])
    await brain.sync_names()
    cited = {i for f in saved["facts"] for i in f["lines"]}
    for r in rows:
        r["decision"] = "stored" if r["id"] in cited else "dropped"
    return rows, list(saved.get("episodes", []))


def _line_id(lines: list[Line], text: str) -> str | None:
    for ln in lines:
        if ln.text == text:
            return ln.id
    return None


async def fact_lines(brain, lines: list[Line]) -> tuple[dict[int, set[str]], list[dict]]:
    """For every fact: the week's line ids it rests on (its evidence; legacy facts by exact text)."""
    by_text = {ln.text: ln.id for ln in lines}
    utt_line = {u.id: by_text[u.text] for u in await brain.memory.utterances(None, 100_000) if u.text in by_text}
    out: dict[int, set[str]] = {}
    dump = []
    for f in await brain.memory.recent(limit=100_000, status=None):
        ids = {utt_line[e] for e in f.evidence if e in utt_line} or ({by_text[f.text]} if f.text in by_text else set())
        out[f.id] = ids
        dump.append({"id": f.id, "text": f.text, "kind": f.kind, "subject": f.subject, "people": list(f.people),
                     "event": datetime.fromtimestamp(f.event_ts).isoformat(timespec="minutes") if f.event_ts else None,
                     "status": f.status, "superseded_by": f.superseded_by, "source": f.source, "lines": sorted(ids)})
    return out, dump


async def ask(brain, clock: FixedClock, q: Question, line_of_memory: dict[int, set[str]], floor: float, judge: bool) -> QuestionResult:
    clock.set(ASK_AT)
    hits = await brain.memory.retrieve(q.q, TOP_K)
    retrieved = [{"memory": h.fact.id, "lines": sorted(line_of_memory.get(h.fact.id, ())), "sim": h.similarity,
                  "score": h.score, "text": h.fact.text} for h in hits]
    await _wait_warm(brain)
    reply = await brain.handle_query(q.q)
    c = check(q, reply.text, reply.outcome)
    stored_lines = set().union(*line_of_memory.values()) if line_of_memory else set()
    got = {lid for x in retrieved for lid in x["lines"]}
    r = QuestionResult(
        id=q.id, cat=q.cat, q=q.q, reference=q.answer or ("I don't remember." if q.abstain else ""),
        spoken=reply.text, outcome=reply.outcome, path=reply.path, citations=reply.citations,
        blocked_sentences=reply.blocked_sentences, total_ms=reply.total_ms, first_audio_ms=reply.time_to_first_audio_ms,
        top_similarity=hits[0].similarity if hits else None,
        top_score=max((h.score for h in hits if h.score is not None), default=None),
        explain=getattr(reply, "explain", "") or "", retrieved=retrieved, support=list(q.support),
        support_stored=[s for s in q.support if s in stored_lines],
        support_retrieved=[s for s in q.support if s in got],
        check=c.__dict__,
    )
    if judge and not q.abstain and brain.llm is not None:
        prompt = JUDGE_PROMPT.format(q=q.q, ref=q.answer, ans=reply.text or "(silence)")
        out = await brain._llm_worker.run(brain.llm.generate, JUDGE_SYSTEM, prompt, max_output_tokens=3)
        r.judge = "yes" if out.strip().lower().startswith("yes") else "no"
    r.failure = failure_reason(q, r, floor)
    return r


async def run_variant(variant: str, subset: str = "full", *, engines: str = "real", judge: bool = False,
                      only: set[str] | None = None, progress=print, speakers: str = "none", facts_from: Path | None = None,
                      overrides: dict | None = None) -> dict:
    lines = load_week(variant)
    questions = load_questions(subset)
    if only:
        questions = [q for q in questions if q.id in only]
    qmap = {q.id: q for q in questions}
    tmp = Path(tempfile.mkdtemp(prefix=f"kin-eval-{variant}-"))
    clock = FixedClock(lines[0].when)
    started = datetime.now()
    brain = bus = None
    try:
        with fixed_clock(clock):
            brain, bus = await (build_real(tmp, overrides) if engines == "real" else build_fake(tmp))
            for name in ("_extract_task", "_episode_task"):  # episodes are closed and extracted per day instead
                task = getattr(brain, name)
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    setattr(brain, name, None)
            progress(f"[{variant}] ingesting {len(lines)} lines ({brain.llm_name}, {brain.memory.embedder_name}, speakers: {speakers})")
            if facts_from is not None:
                import json as _json

                saved = _json.loads(Path(facts_from).read_text(encoding="utf-8"))
                assert saved["meta"]["variant"] == variant, (facts_from, variant)
                progress(f"[{variant}] facts as extracted in {Path(facts_from).name} (answer stage only)")
                write, episodes = await replay_week(brain, clock, lines, saved, speakers)
            else:
                write, episodes = await ingest_week(brain, clock, lines, progress, speakers)
            line_of_memory, facts_dump = await fact_lines(brain, lines)
            memories_before = await brain.memory.count()
            floor = brain.cfg.memory.retrieval.answer_floor if hasattr(brain.cfg.memory, "retrieval") else brain.cfg.memory.similarity_floor
            results: list[QuestionResult] = []
            for i, q in enumerate(questions, 1):
                r = await ask(brain, clock, q, line_of_memory, floor, judge)
                results.append(r)
                mark = "ok " if r.check["correct"] else "BAD"
                progress(f"  [{variant} {i:>3}/{len(questions)}] {mark} {q.id} {r.total_ms:>6} ms  {q.q!r} -> {r.spoken!r}"
                         + (f"  ({r.failure})" if r.failure else ""))
            memories_after = await brain.memory.count()
            summary = summarise(write, results, qmap)
            done = [e["ms"] for e in episodes if e["outcome"] == "done"]
            summary["extraction"] = {
                "episodes": len(episodes),
                "extracted": len(done),
                "skipped_by_prefilter": sum(e["outcome"] == "skipped" for e in episodes),
                "failed": sum(e["outcome"] == "failed" for e in episodes),
                "facts_active": memories_before,
                "facts_superseded": sum(f["status"] == "superseded" for f in facts_dump),
                "updates": sum(e["updated"] for e in episodes),
                "duplicates": sum(e["duplicates"] for e in episodes),
                "median_ms_per_episode": percentile(done, 50),
                "p95_ms_per_episode": percentile(done, 95),
                "mean_lines_per_episode": round(sum(e["lines"] for e in episodes) / len(episodes), 2) if episodes else None,
                "llm_calls": sum(e["llm_calls"] for e in episodes),
            }
            meta = {
                "variant": variant,
                "subset": subset if not only else f"{subset}:{','.join(sorted(only))}",
                "engines": engines,
                "llm": brain.llm_name,
                "embedder": brain.memory.embedder_name,
                "decoding": "greedy (LiteRT-LM top_k=1 temperature=0)" if engines == "real" else "fake",
                "ask_at": ASK_AT.isoformat(),
                "similarity_floor": floor,  # the gate's threshold (v5: answer_floor on the hybrid confidence)
                "retrieval": brain.cfg.memory.retrieval.model_dump() if hasattr(brain.cfg.memory, "retrieval") else None,
                "router": {"accept_score": brain.cfg.router.accept_score, "accept_margin": brain.cfg.router.accept_margin},
                "top_k": brain.cfg.memory.top_k,
                "judge": judge,
                "git": git_hash(),
                "code_hash": code_hash(),
                "started": started.isoformat(timespec="seconds"),
                "duration_s": round((datetime.now() - started).total_seconds(), 1),
                "speakers": speakers,
                "facts_from": Path(facts_from).name if facts_from else None,
                "overrides": overrides or {},
                "memories_after_week": memories_before,
                "memories_after_questions": memories_after,  # > before if a question made Kin store something
            }
            return {"meta": meta, "summary": summary, "write": write, "episodes": episodes, "facts": facts_dump,
                    "questions": [as_dict(r) for r in results]}
    finally:
        if brain is not None:
            await brain.stop()
        if bus is not None:
            await bus.stop()
        shutil.rmtree(tmp, ignore_errors=True)
