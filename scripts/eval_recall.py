"""Kin recall benchmark: a scripted week of overheard speech, then ~100 questions about it.

Injects tests/eval/recall_benchmark/week.txt (and the ASR-noisy copy) straight into the memory
pipeline at fixed timestamps (no audio), asks every question in questions.yaml through the real
query path, and reports:

  write     keep/drop accuracy of the passive router (+ LLM tagger for unsure lines)
  retrieve  recall@5 of the supporting lines for the plain question
  answer    accuracy by rules (+ optional Gemma judge), correct abstention rate, false answers
  latency   median and p95 per question (the warm session is ready, as for an idle user)

Results go to tests/eval/results/recall-<variant>-<subset>-<stamp>-<git>.json.

Usage:
  uv run python scripts/eval_recall.py                       # clean + noisy, all questions (~1 h on a laptop)
  uv run python scripts/eval_recall.py --variant clean       # one variant
  uv run python scripts/eval_recall.py --subset smoke        # the 20-question smoke subset
  uv run python scripts/eval_recall.py --judge               # also grade answers with local Gemma (yes/no)
  uv run python scripts/eval_recall.py --only r01,r02        # just these question ids
  uv run python scripts/eval_recall.py --engines fake        # fast plumbing check with the test fakes
  uv run python scripts/eval_recall.py --report a.json b.json  # table + worst failures from saved results
  uv run python scripts/eval_recall.py --facts-from a.json b.json   # answer stage only: facts as a saved run extracted them
  uv run python scripts/eval_recall.py --set answer_floor=0 rerank=true # memory.retrieval overrides
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.eval.recall_benchmark import load_questions  # noqa: E402
from tests.eval.recall_benchmark.runner import run_variant  # noqa: E402
from tests.eval.recall_benchmark.scoring import QuestionResult, as_dict, check, failure_reason, summarise  # noqa: E402

RESULTS = ROOT / "tests" / "eval" / "results"


def table(results: list[dict]) -> str:
    rows = [
        ("Write: keep/drop accuracy", "write", "keep_drop_accuracy", "%"),
        ("Write: keep-lines stored", "write", "keep_recall", "%"),
        ("Write: chatter not stored", "write", "drop_precision", "%"),
        ("Retrieval recall@5 (supporting lines)", "retrieval", "recall_at_5", ""),
        ("Retrieval recall@5, stored lines only", "retrieval", "recall_at_5_of_stored", ""),
        ("Answer accuracy (rules), all", "answers", "accuracy", "%"),
        ("Answer accuracy (rules), answerable", "answers", "answerable_accuracy", "%"),
        ("Answer accuracy (Gemma judge), answerable", "answers", "judge_accuracy", "%"),
        ("Correct abstention rate", "answers", "correct_abstention_rate", "%"),
        ("False answers (total)", "answers", "false_answers", ""),
        ("  on unanswerable questions", "answers", "false_answers_on_unanswerable", ""),
        ("  wrong facts on answerable", "answers", "false_answers_on_answerable", ""),
        ("Wrong abstentions (answer was there)", "answers", "wrong_abstentions", ""),
        ("Raw citation marker spoken", "answers", "raw_marker_spoken", ""),
        ("Latency median (ms)", "latency_ms", "median", ""),
        ("Latency p95 (ms)", "latency_ms", "p95", ""),
        ("  first audio median (ms)", "latency_ms", "first_audio_median", ""),
        ("  one-call shortcut median (ms)", "latency_ms", "shortcut_median", ""),
        ("  tool-call recall median (ms)", "latency_ms", "tool_recall_median", ""),
        ("Answered by the one-call shortcut", "paths", "shortcut", ""),
        ("Transcript fallback used (correct)", "paths", "transcript_fallback", ""),
    ]
    head = "| Metric | " + " | ".join(r["meta"]["variant"] for r in results) + " |"
    out = [head, "|---|" + "---|" * len(results)]
    for label, sec, key, unit in rows:
        vals = []
        for r in results:
            v = r["summary"].get(sec, {}).get(key)
            if key == "transcript_fallback" and v is not None:
                v = f"{v} ({r['summary']['paths'].get('transcript_fallback_correct', 0)})"
            vals.append("-" if v is None else f"{v}{unit}")
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1))


def _pct_ci(k: int, n: int) -> str:
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{lo}-{hi}] (n={n})"


def splits(results: list[dict]) -> str:
    """Dev (tuned on) and held-out test, with 95% confidence intervals, against the Prompt 5 targets."""
    test_ids = {q.id for q in load_questions("test")}
    out = ["| Metric (95% CI) | " + " | ".join(f"{r['meta']['variant']} {sp}" for r in results for sp in ("dev", "test")) + " |",
           "|---|" + "---|" * (2 * len(results))]
    rows = {"Answerable accuracy": [], "False answers (of all)": [], 'Correct "I don\'t remember"': []}
    for r in results:
        for sp in ("dev", "test"):
            qs = [q for q in r["questions"] if (q["id"] in test_ids) == (sp == "test")]
            ans = [q for q in qs if q["cat"] != "abstain"]
            una = [q for q in qs if q["cat"] == "abstain"]
            rows["Answerable accuracy"].append(_pct_ci(sum(q["check"]["correct"] for q in ans), len(ans)))
            rows["False answers (of all)"].append(_pct_ci(sum(q["check"]["false_answer"] for q in qs), len(qs)))
            rows['Correct "I don\'t remember"'].append(_pct_ci(sum(q["check"]["correct"] for q in una), len(una)))
    for label, vals in rows.items():
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def categories(results: list[dict]) -> str:
    """Answer accuracy per question type, one column per run."""
    cats = list(dict.fromkeys(c for r in results for c in r["summary"]["answers"]["by_category"]))
    out = ["| Question type (n) | " + " | ".join(r["meta"]["variant"] for r in results) + " |", "|---|" + "---|" * len(results)]
    for c in cats:
        n = next(r["summary"]["answers"]["by_category"][c]["n"] for r in results if c in r["summary"]["answers"]["by_category"])
        vals = [f"{r['summary']['answers']['by_category'].get(c, {}).get('accuracy', '-')}%" for r in results]
        out.append(f"| {c} ({n}) | " + " | ".join(vals) + " |")
    return "\n".join(out)


def _severity(failure: str) -> int:
    return 4 if failure.startswith("false answer") else 3 if failure.startswith("retrieved but wrong") else 2


def worst(results: list[dict], n: int = 10) -> str:
    """The questions that failed worst: in more variants first, then hallucinations before misses."""
    fails: dict[str, list[tuple[str, dict]]] = {}
    for r in results:
        for q in r["questions"]:
            if q["failure"]:
                fails.setdefault(q["id"], []).append((r["meta"]["variant"], q))
    ranked = sorted(fails.items(), key=lambda kv: (-len(kv[1]), -max(_severity(q["failure"]) for _, q in kv[1]), kv[0]))
    picked, per_cat = [], {}
    for qid, items in ranked:  # at most two per category, so the list shows different failure modes
        cat = items[0][1]["cat"]
        if per_cat.get(cat, 0) < 2:
            per_cat[cat] = per_cat.get(cat, 0) + 1
            picked.append((qid, items))
    out = ["| # | Question | Variant(s) | Kin said | Reason |", "|---|---|---|---|---|"]
    for i, (qid, items) in enumerate(picked[:n], 1):
        v, q = items[0]
        said = q["spoken"].replace("|", "/") or "(nothing)"
        reasons = "; ".join(sorted({f"{q2['failure']}" + ("" if len(items) == 1 else f" ({v2})") for v2, q2 in items}))
        out.append(f"| {i} | {qid}: {q['q']} | {', '.join(v2 for v2, _ in items)} | {said} | {reasons} |")
    return "\n".join(out)


def rescore(res: dict) -> dict:
    """Recompute failure reasons with the current scoring code (saved runs keep every raw field)."""
    questions = {q.id: q for q in load_questions("all")}
    rows = []
    for d in res["questions"]:
        q = questions[d["id"]]
        r = QuestionResult(**d)
        r.check = check(q, r.spoken, r.outcome).__dict__
        r.failure = failure_reason(q, r, res["meta"]["similarity_floor"])  # the gate's threshold for that run
        rows.append(r)
    res["questions"] = [as_dict(r) for r in rows]
    res["summary"] = summarise(res["write"], rows, questions)
    return res


def keep_awake() -> None:
    """A run takes about an hour: ask Windows not to sleep meanwhile (the request ends with the process). An idle
    sleep in the middle of a question shows up as hours of "latency" and a timeout answer."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["clean", "noisy", "both"], default="both")
    ap.add_argument("--subset", choices=["full", "dev", "test", "all", "smoke"], default="full",
                    help="full/dev: the 99 tuned-on questions; test: the 152 held out; all: both")
    ap.add_argument("--engines", choices=["real", "fake"], default="real")
    ap.add_argument("--speakers", choices=["none", "oracle"], default="none",
                    help="none: what Kin has (no speaker identification); oracle: ground-truth owner/other labels")
    ap.add_argument("--judge", action="store_true", help="also ask local Gemma whether each answer matches the reference")
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--report", nargs="+", type=Path, help="print the table and worst failures for saved result files")
    ap.add_argument("--facts-from", nargs="+", type=Path, default=[],
                    help="saved runs (one per variant): load their facts instead of extracting the week again")
    ap.add_argument("--set", nargs="*", default=[], help="memory.retrieval overrides, e.g. answer_floor=0 rerank=true")
    args = ap.parse_args()
    if args.report:
        saved = [rescore(json.loads(p.read_text(encoding="utf-8"))) for p in args.report]
        print(table(saved) + "\n\n" + categories(saved) + "\n\n" + splits(saved) + "\n\n" + worst(saved))
        return 0
    logging.basicConfig(level=logging.WARNING)
    keep_awake()

    variants = ["clean", "noisy"] if args.variant == "both" else [args.variant]
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    progress = (lambda *_: None) if args.quiet else (lambda s: print(s, flush=True))
    results = []
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    facts_from = {json.loads(p.read_text(encoding="utf-8"))["meta"]["variant"]: p for p in args.facts_from}
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = {"true": True, "false": False}.get(v.lower(), v)
        if isinstance(overrides[k], str):
            try:
                overrides[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                pass
    for v in variants:
        res = asyncio.run(run_variant(v, args.subset, engines=args.engines, judge=args.judge, only=only, progress=progress,
                                      speakers=args.speakers, facts_from=facts_from.get(v), overrides=overrides))
        results.append(res)
        tag = ("-fake" if args.engines == "fake" else "") + ("-oracle" if args.speakers == "oracle" else "") \
            + ("-replay" if v in facts_from else "")
        name = f"recall-{v}-{args.subset}{tag}-{stamp}-{res['meta']['git']}.json"
        path = args.out / name
        path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print()
    print(table(results))
    print()
    print(categories(results))
    print()
    print(splits(results))
    print()
    print(worst(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
