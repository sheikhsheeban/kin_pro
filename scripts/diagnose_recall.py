"""Where do recall answers fail? (Prompt 5, Part A)

  attribution   every wrong answer of a saved run, classified:
                  a  never stored         no stored fact rests on a supporting line
                  b  not in top-k         stored, but no offered fact rests on a supporting line
                  c  model abstained      offered, but Kin said it doesn't remember
                  d  model answered wrong offered, but the answer is wrong
                  e  gate blocked         offered, a sentence was blocked and the answer came out wrong
                  u  unanswerable         a false answer to a question nothing in the week answers
  extraction    precision / recall of a saved run's facts against gold_facts.yaml (split facts and
                noisy text separately)
  oracle        the answer step alone: the model gets exactly the gold supporting facts (unanswerable
                questions: the 3 nearest unrelated gold facts), through the real prompt and gate.
                This is the ceiling of the answer step for a model.

  uv run python scripts/diagnose_recall.py attribution tests/eval/results/recall-clean-....json ...
  uv run python scripts/diagnose_recall.py extraction tests/eval/results/recall-clean-....json ...
  uv run python scripts/diagnose_recall.py oracle --model litert:llm/gemma-4-E2B-it.litertlm
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tests.eval.recall_benchmark import ASK_AT, HERE, load_questions, load_week  # noqa: E402
from tests.eval.recall_benchmark.scoring import check  # noqa: E402

GOLD = HERE / "gold_facts.yaml"


def load_gold() -> list[dict]:
    out = []
    for g in yaml.safe_load(GOLD.read_text(encoding="utf-8"))["facts"]:
        out.append({"status": "active", "split": False, **g, "lines": [str(x) for x in g["lines"]]})
    return out


# --- A1: attribution -----------------------------------------------------------------------------------


def attribute(run: dict) -> dict:
    qs = {q.id: q for q in load_questions("all")}
    lines_of = {f["id"]: set(f["lines"]) for f in run["facts"]}
    counts: dict[str, int] = {}
    rows = []
    for r in run["questions"]:
        if r["check"]["correct"]:
            continue
        q = qs[r["id"]]
        if q.abstain:
            cat = "u"
        else:
            sup = set(q.support)
            offered = [u["id"] for u in json.loads(r["explain"])["used"]] if r.get("explain") else []
            offered_lines = set().union(*(lines_of.get(i, set()) for i in offered)) if offered else set()
            stored = any(sup & ls for ls in lines_of.values())
            if not stored:
                cat = "a"
            elif not sup & offered_lines:
                cat = "b"
            elif r["blocked_sentences"]:
                cat = "e"
            elif r["check"]["abstained"]:
                cat = "c"
            else:
                cat = "d"
        counts[cat] = counts.get(cat, 0) + 1
        rows.append((cat, r["id"], q.cat, r["q"], r["spoken"][:110]))
    return {"counts": counts, "rows": sorted(rows)}


# --- A2: extraction quality ----------------------------------------------------------------------------

_KEY = re.compile(r"\d[\d,:.]*\d|\d|[A-Z][a-z]+")


def _keys(text: str) -> set[str]:
    stop = {"The", "User", "Someone", "Mum", "She", "He", "They", "On", "In", "At", "My", "It", "This", "That", "And",
            "Let", "Next", "Your", "January", "February", "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    return {k.replace(",", "").lower() for k in _KEY.findall(text) if k not in stop}


def matches(fact: dict, gold: dict) -> bool:
    """Same source line(s), and the gold fact's names and numbers are mostly there (heard spellings allowed)."""
    from rapidfuzz import fuzz

    if not set(fact["lines"]) & set(gold["lines"]):
        return False
    gk, fk = _keys(gold["text"]), _keys(fact["text"])
    if gk:
        hit = sum(any(fuzz.ratio(g, f) >= 80 for f in fk) for g in gk)
        if hit / len(gk) < 0.5:
            return False
    return fuzz.token_set_ratio(gold["text"].lower(), fact["text"].lower()) >= 50


def extraction(run: dict) -> dict:
    gold = load_gold()
    keep = {ln.id for ln in load_week(run["meta"]["variant"]) if ln.keep}
    facts = run["facts"]
    found = {g["id"]: [f["id"] for f in facts if matches(f, g)] for g in gold}
    matched_facts = {fid for ids in found.values() for fid in ids}
    chatter = [f for f in facts if f["lines"] and not set(f["lines"]) & keep]
    split = [g for g in gold if g.get("split")]
    single = [g for g in gold if not g.get("split")]
    rec = lambda gs: round(sum(bool(found[g["id"]]) for g in gs) / len(gs), 3)  # noqa: E731
    return {
        "facts_stored": len(facts), "gold": len(gold),
        "precision": round(len(matched_facts) / len(facts), 3) if facts else None,
        "recall": rec(gold), "recall_single_line": rec(single), "recall_split": rec(split),
        "facts_from_chatter_only": len(chatter),
        "missed": [(g["id"], g["text"]) for g in gold if not found[g["id"]]],
        "unmatched_facts": [(f["id"], f["text"]) for f in facts if f["id"] not in matched_facts][:40],
    }


# --- A3: the answer oracle -----------------------------------------------------------------------------


def _hits(gold: list[dict], week: dict):
    from kin.memory.store import Fact, Hit

    out = []
    for i, g in enumerate(gold, 1):
        heard = max(week[x].when.timestamp() for x in g["lines"] if x in week)
        f = Fact(int(g["id"][1:]), g["text"], "fact", None, (), None, heard, g["status"], None, 0.9, "extracted", ())
        out.append(Hit(f, 1.0 - i * 0.01, score=1.0))
    return out


def oracle(model: str, threads: int | None, limit: int | None = None) -> dict:
    """The answer step alone, on gold facts. `model` is backend:path, e.g. litert:llm/gemma-4-E4B-it.litertlm."""
    from rapidfuzz import fuzz

    from kin.brain.gate import ABSTAIN_REPLY, Gate
    from kin.brain.llm import LiteRTBackend, LlamaCppBackend
    from kin.brain.prompt import PromptBuilder
    from kin.brain.text import SentenceSplitter
    from kin.config import load_config

    cfg = load_config("laptop")
    kind, path = model.split(":", 1)
    t0 = time.perf_counter()
    if kind == "litert":
        llm = LiteRTBackend(cfg.model_path(path), context_tokens=4096, threads=threads, cache_dir=cfg.model_path("llm/cache"))
    else:
        llm = LlamaCppBackend(cfg.model_path(path), context_tokens=4096, threads=threads)
    load_s = time.perf_counter() - t0
    import psutil

    rss = psutil.Process().memory_info().rss / 1e6
    builder = PromptBuilder(llm.count_tokens, 1000)
    gate = Gate(0.0)
    gold = load_gold()
    week = {ln.id: ln for ln in load_week("clean")}
    results, lat, first = [], [], []
    questions = load_questions()[:limit] if limit else load_questions()
    for q in questions:
        if q.abstain:
            facts = sorted((g for g in gold if g["status"] == "active"),
                           key=lambda g: -fuzz.token_set_ratio(q.q.lower(), g["text"].lower()))[:3]
        else:
            facts = [g for g in gold if set(g["lines"]) & set(q.support)]
            facts.sort(key=lambda g: g["status"] != "active")
        hits = _hits(facts, week)
        prompt = builder.recall(q.q, hits, ASK_AT.timestamp())
        session = llm.session(prompt.system, max_output_tokens=160)
        t = time.perf_counter()
        out, t_first = [], None
        splitter = SentenceSplitter()
        spoken, cited = [], False
        for chunk in session.send(prompt.user):
            if chunk.text:
                t_first = t_first or time.perf_counter()
                for s in splitter.feed(chunk.text):
                    out.append(s)
        rest = splitter.flush()
        if rest:
            out.append(rest)
        session.close()
        for s in out:
            v = gate.check(s, set(prompt.memory_ids))
            if v.speakable:
                spoken.append(v.spoken)
                cited |= v.reason == "cited"
        text = " ".join(spoken) or ABSTAIN_REPLY
        c = check(q, text, "answered" if cited else "abstained")
        lat.append((time.perf_counter() - t) * 1000)
        first.append(((t_first or time.perf_counter()) - t) * 1000)
        results.append({"id": q.id, "cat": q.cat, "abstain": q.abstain, "correct": c.correct, "false": c.false_answer,
                        "abstained": c.abstained, "said": text, "raw": " ".join(out)})
        print(f"  {'ok ' if c.correct else 'BAD'} {q.id} {lat[-1]:6.0f} ms  {q.q!r} -> {text[:100]!r}", flush=True)
    llm.close()
    ans = [r for r in results if not r["abstain"]]
    una = [r for r in results if r["abstain"]]
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["cat"], []).append(r["correct"])
    return {
        "model": model, "load_s": round(load_s, 1), "rss_mb_after_load": round(rss),
        "answerable_accuracy": round(100 * sum(r["correct"] for r in ans) / len(ans), 1),
        "abstain_correct": round(100 * sum(r["correct"] for r in una) / len(una), 1) if una else None,
        "false_answers": sum(r["false"] for r in results),
        "wrong_abstentions_answerable": sum(r["abstained"] for r in ans),
        "latency_ms_median": round(statistics.median(lat)), "first_sentence_ms_median": round(statistics.median(first)),
        "by_category": {k: round(100 * sum(v) / len(v), 1) for k, v in by_cat.items()},
        "rows": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["attribution", "extraction", "oracle"])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--model", default="litert:llm/gemma-4-E2B-it.litertlm")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.what == "oracle":
        res = oracle(args.model, args.threads, args.limit)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    else:
        res = {}
        for p in args.runs:
            run = json.loads(p.read_text(encoding="utf-8"))
            r = attribute(run) if args.what == "attribution" else extraction(run)
            res[run["meta"]["variant"]] = r
            print(f"== {run['meta']['variant']} ({p.name})")
            if args.what == "attribution":
                print("  counts:", dict(sorted(r["counts"].items())))
                for row in r["rows"]:
                    print("   ", *row)
            else:
                print(json.dumps({k: v for k, v in r.items() if k not in ("missed", "unmatched_facts")}, indent=1))
                print("  missed gold facts:")
                for m in r["missed"]:
                    print("   ", *m)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
