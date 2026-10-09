"""Choose `answer_floor` (and check `utterance_floor`) from a calibration run of the recall benchmark.

A calibration run asks every question with `answer_floor=0`, so every question reaches the LLM and its
real answer is known. Raising the floor only changes questions whose confidence falls under it: those
abstain, or quote the closest raw line if it reaches `utterance_floor`. So every threshold can be
replayed exactly from one run. Raw lines are scored with the same recall code on the same week.

Picks the floor with the most correct answers while hallucinations stay under the limit (ties: the
highest such floor, the most cautious). Two definitions are reported:
  made up   answering a question nothing in the week answers (the default for the limit)
  all false every false answer, including a wrong fact given for an answerable question; most of
            those come with a high confidence (the wrong one of two similar facts), so no retrieval
            threshold removes them

Usage:
  uv run python scripts/eval_recall.py --facts-from <clean>.json <noisy>.json --set answer_floor=0 --out runs/
  uv run python scripts/calibrate_floor.py runs/recall-clean-*.json runs/recall-noisy-*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kin.brain.gate import ABSTAIN_REPLY  # noqa: E402
from tests.eval.recall_benchmark import ASK_AT, load_questions  # noqa: E402
from tests.eval.recall_benchmark.scoring import check  # noqa: E402


async def best_lines(run: dict) -> dict[str, tuple[float, str, float]]:
    """{question id: (line score, line text, heard ts)} for the closest raw line of the week."""
    from tests.eval.recall_benchmark.offline import build
    from tests.eval.recall_benchmark.runner import FixedClock, fixed_clock

    tmp = Path(run["_path"])
    svc, _, _ = await build(tmp, 256, rerank=bool((run["meta"].get("retrieval") or {}).get("rerank")))
    out = {}
    try:
        with fixed_clock(FixedClock(ASK_AT)):
            for q in load_questions():
                plan = (await svc.recall(q.q, 5)).plan
                lines = await svc.recall_lines(plan, 1)
                if lines:
                    out[q.id] = (lines[0].score, lines[0].line.text, lines[0].line.ts)
    finally:
        await svc.close()
    return out


def heard_when(ts: float) -> str:
    from kin.brain.orchestrator import heard_when as hw

    return hw(ts, ASK_AT)


def simulate(run: dict, lines: dict, floor: float, ufloor: float) -> dict:
    qs = {q.id: q for q in load_questions()}
    correct = false = false_unans = abstained = quoted = 0
    for r in run["questions"]:
        q = qs[r["id"]]
        conf = None
        if r.get("explain"):
            conf = json.loads(r["explain"])["decision"]["confidence"]
        if conf is None:
            conf = r.get("top_score")
        if conf is not None and conf >= floor:
            c = r["check"]
        else:
            line = lines.get(q.id)
            if line and line[0] >= ufloor:
                text = f"I'm not sure, but {heard_when(line[2])} I heard: “{line[1]}”"
                c = check(q, text, "answered").__dict__
                quoted += 1
            else:
                c = check(q, ABSTAIN_REPLY, "abstained").__dict__
        correct += c["correct"]
        false += c["false_answer"]
        false_unans += c["false_answer"] and q.abstain
        abstained += c["abstained"]
    n = len(run["questions"])
    return {"floor": floor, "utterance_floor": ufloor, "correct": correct, "accuracy": round(100 * correct / n, 1),
            "false_answers": false, "hallucination_rate": round(100 * false / n, 1), "false_on_unanswerable": false_unans,
            "abstained": abstained, "quoted": quoted}


async def main(args) -> int:
    runs = []
    for p in args.runs:
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_path"] = str(p)
        runs.append(r)
    lines = {r["meta"]["variant"]: await best_lines(r) for r in runs}
    floors = [round(x * 0.025, 3) for x in range(0, 41)]
    best = None
    total = sum(len(r["questions"]) for r in runs)
    print("floor  " + "  ".join(f"{r['meta']['variant']:>34}" for r in runs) + "   combined: correct, made up, all false")
    for f in floors:
        sims = [simulate(r, lines[r["meta"]["variant"]], f, args.utterance_floor) for r in runs]
        corr = sum(s["correct"] for s in sims)
        made_up = 100 * sum(s["false_on_unanswerable"] for s in sims) / total
        all_false = 100 * sum(s["false_answers"] for s in sims) / total
        rate = all_false if args.define == "all" else made_up
        print(f"{f:5.3f}  " + "  ".join(f"correct {s['correct']:3d} false {s['false_answers']:2d} ({s['false_on_unanswerable']} unans) q{s['quoted']:2d}"
                                       for s in sims) + f"   {corr:3d}  {made_up:4.1f}%  {all_false:4.1f}%")
        if rate < args.max_hallucination and (best is None or corr >= best[1]):
            best = (f, corr, rate, sims)
    if best is None:
        print(f"\nno floor keeps hallucinations under {args.max_hallucination}%")
        floor_min = min(floors, key=lambda f: (sum(simulate(r, lines[r['meta']['variant']], f, args.utterance_floor)["false_answers"] for r in runs), -f))
        print(f"fewest false answers at floor {floor_min}")
    else:
        print(f"\nchosen answer_floor {best[0]}: {best[1]} correct, hallucinations {best[2]:.1f}%")
        for s in best[3]:
            print("  ", s)
    if args.out:
        args.out.write_text(json.dumps({"best": best and best[:3], "at": datetime.now().isoformat()}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--utterance-floor", type=float, default=0.8)
    ap.add_argument("--max-hallucination", type=float, default=2.0, help="percent of all questions")
    ap.add_argument("--define", choices=["made_up", "all"], default="made_up", help="what counts as a hallucination")
    ap.add_argument("--out", type=Path)
    sys.exit(asyncio.run(main(ap.parse_args())))
