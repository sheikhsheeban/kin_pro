"""Spoken end-to-end recall set: 40 scenarios synthesised with Piper, through the real ears and brain.

Each scenario speaks the lines a question needs (plus one line of chatter) into the passive pipeline,
then asks the question with the button held, exactly as a user would: Piper -> 16 kHz -> FakeInputStream
-> Silero VAD -> Moonshine -> ingest -> episode extraction -> hybrid recall -> answer. Three voices
(en_US lessac, amy, ryan) at three speeds (0.9x, 1.0x, 1.2x) rotate across utterances.

Audio exists only in RAM: synthesised into numpy arrays, pushed as frames, never written anywhere (the
privacy audit hook would refuse an audio file write anyway; CLAUDE.md).

Questions whose answers depend on the scripted week's dates (time-scoped ones, or weekday answers) are
left out: here every line is heard at the real current time.

  uv run python scripts/eval_spoken.py            # 40 scenarios (~20 min)
  uv run python scripts/eval_spoken.py --n 8      # a quick look
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.eval.recall_benchmark import load_questions, load_week  # noqa: E402
from tests.eval.recall_benchmark.scoring import check  # noqa: E402

VOICES = ["tts/en_US-lessac-medium.onnx", "tts/en_US-amy-medium.onnx", "tts/en_US-ryan-medium.onnx"]
SPEEDS = [1.0, 0.9, 1.2]  # Piper length_scale: 0.9 is faster speech, 1.2 slower
FRAME = 512
_DATEY = re.compile(r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|yesterday|today", re.I)


def scenarios(n: int = 40) -> list[dict]:
    week = {ln.id: ln for ln in load_week("clean")}
    chatter = [ln.text for ln in week.values() if not ln.keep and len(ln.text.split()) >= 4]
    qs = load_questions("test") + load_questions("full")  # held-out questions first
    ok_cats = ("single_fact", "lookup", "split", "who_said", "multi_fact", "latest_version")
    answerable = [q for q in qs if q.cat in ok_cats and not any(_DATEY.search(a) for g in q.expect for a in g)
                  and len(q.support) <= 4]
    abstain = [q for q in qs if q.abstain]
    by_cat: dict[str, list] = {}
    for q in answerable:
        by_cat.setdefault(q.cat, []).append(q)
    mixed = []  # round-robin over question types, so every type is spoken
    while any(by_cat.values()) and len(mixed) < n - n // 4:
        for cat in ok_cats:
            if by_cat.get(cat) and len(mixed) < n - n // 4:
                mixed.append(by_cat[cat].pop(0))
    picked = mixed + abstain[: n // 4]
    out = []
    for i, q in enumerate(picked):
        lines = [week[s].text for s in sorted(q.support, key=lambda s: week[s].when)]
        lines.insert(len(lines) // 2, chatter[i % len(chatter)])
        out.append({"q": q, "lines": lines})
    return out


class Voices:
    def __init__(self, cfg) -> None:
        from piper import PiperVoice

        self.voices = [PiperVoice.load(str(cfg.model_path(v))) for v in VOICES]
        self.k = 0

    def speak(self, text: str) -> tuple[list[np.ndarray], str]:
        """One utterance as 16 kHz frames (in memory only), rotating voice and speed."""
        from piper import SynthesisConfig
        from scipy.signal import resample_poly

        v = self.voices[self.k % len(self.voices)]
        speed = SPEEDS[(self.k // len(self.voices)) % len(SPEEDS)]
        self.k += 1
        audio = np.concatenate([c.audio_float_array for c in v.synthesize(text, syn_config=SynthesisConfig(length_scale=speed))])
        x = resample_poly(audio, 16000, int(v.config.sample_rate)).astype(np.float32) * 0.8
        pad = np.zeros(8000, np.float32)
        x = np.concatenate([pad, x, pad, pad])
        n = len(x) // FRAME
        return list(x[: n * FRAME].reshape(n, FRAME)), f"{Path(VOICES[self.voices.index(v)]).stem}@{speed}"


async def play(stream, frames: list[np.ndarray]) -> None:
    def run() -> None:
        t0 = time.monotonic()
        for i, f in enumerate(frames):
            stream.push([f])
            delay = t0 + (i + 1) * FRAME / 16000 - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    await asyncio.to_thread(run)


async def main(args) -> dict:
    import secrets

    from kin.app import KinApp
    from kin.audio import build_engines
    from kin.brain import build_brain
    from kin.config import load_config
    from kin.core.bus import EventBus
    from kin.core.events import ActivationEvent, AttendingEnded, ReplyCompleted, TranscriptEvent
    from tests.conftest import FakeInputStream

    os.environ["KIN_DB_KEY"] = secrets.token_hex(32)
    cfg = load_config("laptop")
    cfg.data_dir = Path(tempfile.mkdtemp(prefix="kin-spoken-"))
    cfg.tts.sink = "null"
    cfg.tts.ack_delay_s = 600
    cfg.gateway.enabled = False
    streams: list = []

    def stream_factory(**kw):
        streams.append(FakeInputStream(**kw))
        return streams[-1]

    bus = EventBus()
    heard: list[str] = []
    bus.subscribe(TranscriptEvent, lambda e: heard.append(e.text) if e.final else None, name="eval.heard")
    app = KinApp(cfg, bus=bus, engines_factory=build_engines, stream_factory=stream_factory,
                 device_namer=lambda _d: "piper", brain_factory=build_brain, gateway_factory=None)
    await app.start()
    await app.wait_brain()
    voices = Voices(cfg)
    rows = []
    try:
        for i, sc in enumerate(scenarios(args.n), 1):
            q = sc["q"]
            for line in sc["lines"]:
                frames, _ = voices.speak(line)
                await play(streams[-1], frames)
                await asyncio.sleep(0.3)
            await asyncio.sleep(1.0)
            await app.brain.drain()
            await app.brain.flush()
            heard.clear()
            app.hardware.simulate_button(True)
            await bus.wait_for(ActivationEvent, timeout=5)
            frames, voice = voices.speak(q.q)
            await play(streams[-1], frames)
            app.hardware.simulate_button(False)
            await bus.wait_for(AttendingEnded, timeout=10)
            reply = await bus.wait_for(ReplyCompleted, timeout=60)
            c = check(q, reply.text, reply.outcome)
            rows.append({"id": q.id, "cat": q.cat, "abstain": q.abstain, "q": q.q, "heard_q": reply.query, "voice": voice,
                         "said": reply.text, "correct": c.correct, "false": c.false_answer, "abstained": c.abstained,
                         "first_ms": reply.time_to_first_audio_ms, "path": reply.path})
            print(f"  [{i:>2}] {'ok ' if c.correct else 'BAD'} {voice:28s} {q.q!r} heard {reply.query!r} -> {reply.text[:90]!r}",
                  flush=True)
    finally:
        await app.stop()
    ans = [r for r in rows if not r["abstain"]]
    una = [r for r in rows if r["abstain"]]
    first = [r["first_ms"] for r in rows if r["first_ms"] is not None]
    by_voice: dict[str, list[bool]] = {}
    for r in rows:
        by_voice.setdefault(r["voice"].split("@")[0], []).append(r["correct"])
    out = {
        "scenarios": len(rows),
        "answerable_accuracy": round(100 * sum(r["correct"] for r in ans) / len(ans), 1) if ans else None,
        "correct_abstention": round(100 * sum(r["correct"] for r in una) / len(una), 1) if una else None,
        "false_answers": sum(r["false"] for r in rows),
        "false_answer_rate": round(100 * sum(r["false"] for r in rows) / len(rows), 1),
        "first_audio_ms_median": statistics.median(first) if first else None,
        "by_voice": {k: round(100 * sum(v) / len(v), 1) for k, v in by_voice.items()},
        "question_transcribed_exactly": sum(r["heard_q"].strip().lower().rstrip("?.") == r["q"].lower().rstrip("?.") for r in rows),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": out, "rows": rows, "at": datetime.now().isoformat()}, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path)
    logging.basicConfig(level=logging.WARNING)
    from scripts.eval_recall import keep_awake

    keep_awake()
    asyncio.run(main(ap.parse_args()))
