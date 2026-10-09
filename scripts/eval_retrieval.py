"""Retrieval-only benchmark (about a minute): rebuild the facts of a saved recall run and re-ask every question.

No LLM: it measures what reaches the answer prompt (recall@5 of the supporting lines) and how well the
confidence score separates answerable from unanswerable questions (the abstention threshold).

Usage:
  uv run python scripts/eval_retrieval.py                                 # newest clean + noisy runs, 256-d
  uv run python scripts/eval_retrieval.py --dim 256 768                   # compare embedding sizes
  uv run python scripts/eval_retrieval.py --rerank                        # with the cross-encoder
  uv run python scripts/eval_retrieval.py --before                        # also the retrieval Kin had before
  uv run python scripts/eval_retrieval.py --sweep                         # threshold table for the gate
  uv run python scripts/eval_retrieval.py --variants "" "person_mode=boost" "lexical_weight=0.5,alt_weight=0"
  uv run python scripts/eval_retrieval.py --results tests/eval/results/recall-clean-*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.eval.recall_benchmark.offline import build, evaluate, evaluate_before, summary, sweep  # noqa: E402

RESULTS = ROOT / "tests" / "eval" / "results"


def newest(variant: str) -> Path:
    files = sorted((p for p in RESULTS.glob(f"recall-{variant}-full-*.json") if "oracle" not in p.name and "fake" not in p.name),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no saved full {variant} run in {RESULTS}; run scripts/eval_recall.py first")
    return files[-1]


async def run(args) -> list[dict]:
    out = []
    for path in args.results or [newest("clean"), newest("noisy")]:
        for dim in args.dim:
            svc, lines_of, res = await build(path, dim, rerank=args.rerank)
            try:
                configs = [("hybrid" + ("+rerank" if args.rerank else "") + (f" {v}" if v else ""), v) for v in args.variants]
                if args.before:
                    configs.insert(0, ("before", "before"))
                base = svc.cfg
                for name, mode in configs:
                    if mode not in (None, "before"):
                        svc.cfg = base.model_copy(update=_parse(mode))
                    t0 = time.perf_counter()
                    rows = await (evaluate_before(svc, lines_of) if mode == "before" else evaluate(svc, lines_of))
                    svc.cfg = base
                    s = summary(rows) | {"variant": res["meta"]["variant"], "dim": dim, "retrieval": name,
                                         "ms_per_question": round((time.perf_counter() - t0) * 1000 / len(rows), 1)}
                    if args.sweep:
                        s["sweep"] = sweep(rows, [round(0.30 + 0.025 * i, 3) for i in range(25)])
                    if args.dump:
                        s["rows"] = [{"id": r.q.id, "cat": r.q.cat, "abstain": r.q.abstain, "conf": r.confidence,
                                      "support": list(r.q.support), "got": r.support_retrieved, "line_conf": r.line_conf,
                                      "line_has_support": r.line_has_support, "plan": r.plan,
                                      "top": [(g["id"], round(g["cos"], 3), g["score"], g.get("rerank"), g["text"][:90]) for g in r.retrieved],
                                      "hits": [{k: g.get(k) for k in ("id", "lines", "cos", "score", "rerank", "coverage", "rrf")} for g in r.retrieved]}
                                     for r in rows]
                    out.append(s)
                    print(f"{s['variant']:5} {dim:4}-d {name:40} recall@5 {s['recall_at_5']:.3f}  any@5 {s['any_support_at_5']:.3f}  "
                          f"top1 {s['top1_support']:.3f}  abstain AUC {s['abstain_auc']}  {s['ms_per_question']} ms/q", flush=True)
            finally:
                await svc.close()
    return out


def _parse(spec: str) -> dict:
    out = {}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=", 1)
        out[k.strip()] = v if not v.replace(".", "", 1).replace("-", "", 1).isdigit() else float(v) if "." in v else int(v)
        if v in ("true", "false"):
            out[k.strip()] = v == "true"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", type=Path)
    ap.add_argument("--dim", nargs="+", type=int, default=[256])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--before", action="store_true", help="also score the pre-hybrid retrieval on the same facts")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--variants", nargs="+", default=[""], help="retrieval settings to compare, e.g. 'person_mode=boost,person_boost=0.5'")
    ap.add_argument("--dump", type=Path, help="write every row and summary to this JSON file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = asyncio.run(run(args))
    for s in out:
        print(f"\n{s['variant']} {s['dim']}-d {s['retrieval']}: by category {s['by_category']}")
        if "sweep" in s:
            print("  t      answerable kept (with support)   unanswerable passed")
            for x in s["sweep"]:
                print(f"  {x['t']:.3f}  {x['answerable_kept']:3d} ({x['answerable_kept_with_support']:3d})"
                      f"                  {x['unanswerable_passed']:3d}")
    if args.dump:
        args.dump.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
