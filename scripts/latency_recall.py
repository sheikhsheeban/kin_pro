"""Recall-query latency with the real models, laptop or Pi 5 profile (no audio, TTS off).

Loads the facts of a saved benchmark run into a fresh encrypted store, then asks a fixed set of recall
questions, each after the warm session is ready (an idle user), several rounds. Reports the median
time to the first answer sentence and to the full answer, LLM calls, the path taken, and the retrieval
time. Works on the code before hybrid recall too (for before/after numbers).

Usage:
  uv run python scripts/latency_recall.py --profile laptop
  uv run python scripts/latency_recall.py --profile pi5          # Pi limits: 4 threads, 2K context, 600-token prompts
  uv run python scripts/latency_recall.py --profile pi5 --set rerank=true
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = [
    "What's my gym locker code?",
    "Where is Anil's wedding?",
    "When is Rahul's birthday?",
    "Who is my mentor?",
    "What did Neha tell me yesterday?",
    "When is my dentist appointment?",
    "What do I know about Kavya?",
    "What's the office wifi password?",
]


async def main(args) -> dict:
    from kin.brain import build_brain
    from kin.config import load_config
    from kin.core.bus import EventBus
    from kin.core.metrics import ResourceMonitor
    from tests.eval.recall_benchmark import ASK_AT, load_week
    from tests.eval.recall_benchmark.runner import FixedClock, _wait_warm, fixed_clock

    res = json.loads(args.facts.read_text(encoding="utf-8"))
    week = {ln.id: ln for ln in load_week(res["meta"]["variant"])}
    tmp = Path(tempfile.mkdtemp(prefix="kin-latency-"))
    os.environ["KIN_DB_KEY"] = secrets.token_hex(32)
    cfg = load_config(args.profile)
    cfg.data_dir = tmp
    cfg.tts.enabled = False
    cfg.tts.sink = "null"
    cfg.tts.ack_delay_s = 600
    cfg.gateway.enabled = False
    cfg.extract.enabled = False
    for kv in args.set:
        k, v = kv.split("=", 1)
        target = cfg.memory.retrieval if hasattr(cfg.memory, "retrieval") else cfg.memory
        setattr(target, k, {"true": True, "false": False}.get(v, float(v) if v.replace(".", "", 1).isdigit() else v))
    if args.answer_model:
        cfg.llm.answer_model = args.answer_model
    cache = ROOT / "data" / "cache"
    if cache.exists():
        shutil.copytree(cache, tmp / "data" / "cache", dirs_exist_ok=True)
    clock = FixedClock(ASK_AT)
    bus = EventBus()
    await bus.start()
    rows = []
    with fixed_clock(clock):
        brain = await build_brain(cfg, bus, ResourceMonitor())
        try:
            for f in sorted(res["facts"], key=lambda f: f["id"]):
                if f["status"] != "active":
                    continue
                ts = [week[i].when.timestamp() for i in f["lines"] if i in week]
                await brain.memory.remember(
                    f["text"], f["kind"], 0.8, source="extracted", ts=max(ts) if ts else ASK_AT.timestamp() - 86400,
                    subject=f["subject"], people=f["people"],
                    event_ts=datetime.fromisoformat(f["event"]).timestamp() if f.get("event") else None)
            for p in {n for f in res["facts"] for n in f["people"] if n[:1].isupper()}:
                await brain.memory.add_person(p)
            for rnd in range(args.rounds):
                for q in QUESTIONS:
                    await _wait_warm(brain)
                    t0 = time.perf_counter()
                    hits = await brain.memory.retrieve(q, cfg.memory.top_k)
                    retrieval_ms = (time.perf_counter() - t0) * 1000
                    reply = await brain.handle_query(q)
                    first = reply.time_to_first_audio_ms
                    rows.append({"round": rnd, "q": q, "total_ms": reply.total_ms, "first_ms": first, "llm_calls": reply.llm_calls,
                                 "path": reply.path, "retrieval_ms": round(retrieval_ms, 1), "text": reply.text,
                                 "prompt_tokens": reply.prompt_tokens,
                                 "rss_mb": round(__import__("psutil").Process().memory_info().rss / 1e6)})
                    print(f"  {reply.total_ms:6d} ms  first {first}  calls {reply.llm_calls}  {reply.path:28s} {q} -> {reply.text[:80]}",
                          flush=True)
        finally:
            await brain.stop()
            await bus.stop()
            shutil.rmtree(tmp, ignore_errors=True)
    med = lambda xs: round(statistics.median(xs), 1) if xs else None  # noqa: E731
    summary = {
        "profile": args.profile, "settings": args.set, "answer_model": args.answer_model, "questions": len(rows),
        "rss_mb_peak": max(r["rss_mb"] for r in rows),
        "total_ms_median": med([r["total_ms"] for r in rows]),
        "first_sentence_ms_median": med([r["first_ms"] for r in rows if r["first_ms"] is not None]),
        "retrieval_ms_median": med([r["retrieval_ms"] for r in rows]),
        "llm_calls_mean": round(statistics.mean(r["llm_calls"] for r in rows), 2),
        "one_call_share": round(sum(r["llm_calls"] == 1 for r in rows) / len(rows), 2),
        "by_path": {p: med([r["total_ms"] for r in rows if r["path"] == p]) for p in sorted({r["path"] for r in rows})},
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="laptop", choices=["laptop", "pi5"])
    ap.add_argument("--facts", type=Path, default=ROOT / "tests" / "eval" / "results" / "recall-clean-full-20261005-213653-nogit.json")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--set", nargs="*", default=[], help="memory.retrieval overrides, e.g. rerank=true strong_score=2")
    ap.add_argument("--answer-model", default=None, help="llm.answer_model, e.g. llm/gemma-4-E4B-it.litertlm")
    ap.add_argument("--out", type=Path)
    logging.basicConfig(level=logging.WARNING)
    from scripts.eval_recall import keep_awake  # noqa: E402

    keep_awake()
    asyncio.run(main(ap.parse_args()))
