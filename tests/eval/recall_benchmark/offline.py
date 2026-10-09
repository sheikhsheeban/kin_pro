"""Retrieval-only evaluation in a minute, from a saved benchmark run (no LLM).

A full benchmark run spends most of its time extracting facts with Gemma. Its JSON keeps every fact
it extracted (text, subject, people, event time, status, the lines behind it), so the fact store can be
rebuilt exactly and every question asked again through `MemoryService.recall` with any retrieval
settings, embedding size or reranker. Used to tune fusion weights and to calibrate the abstention
threshold before spending an hour on the full run.

Rebuild: each fact is re-embedded (EmbeddingGemma, document prompt) and stored with heard_ts = the
time of its last supporting line (as extraction does); superseded facts keep their chain; people come
from the facts; the week's non-filler lines become utterances with vectors (for the transcript fallback).
"""

from __future__ import annotations

import json
import secrets
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tests.eval.recall_benchmark import ASK_AT, Question, load_questions, load_week


@dataclass
class Row:
    q: Question
    retrieved: list[dict]  # [{id, lines, cos, score, rerank, text}]
    support_retrieved: list[str]
    confidence: float | None
    line_conf: float | None  # best transcript line
    line_has_support: bool
    plan: dict
    filtered: list[dict]


async def build(result_file: Path, dim: int, *, rerank: bool = False, tmp: Path | None = None):
    """A MemoryService holding the facts of a saved run (and the week's lines). Returns (service, fact-id -> line ids)."""
    from kin.brain.semantic_router import is_filler
    from kin.config import load_config
    from kin.core.bus import EventBus
    from kin.memory.embed import GemmaEmbedder, find_embedding_model
    from kin.memory.rerank import CrossEncoder, find_reranker
    from kin.memory.service import MemoryService
    from kin.memory.store import MemoryStore

    res = json.loads(Path(result_file).read_text(encoding="utf-8"))
    variant = res["meta"]["variant"]
    week = {ln.id: ln for ln in load_week(variant)}
    cfg = load_config("laptop")
    tmp = tmp or Path(tempfile.mkdtemp(prefix="kin-offline-"))
    key = secrets.token_hex(32)
    model = find_reranker(cfg.model_path(cfg.memory.retrieval.rerank_model))
    svc = MemoryService(
        EventBus(), lambda: GemmaEmbedder(find_embedding_model(cfg.model_path("embedding")), "q4", dim),
        lambda d: MemoryStore(tmp / "kin.db", key, d),
        retrieval=cfg.memory.retrieval.model_copy(update={"rerank": rerank}),
        reranker_factory=(lambda: CrossEncoder(model, threads=cfg.runtime.threads)) if model else None,
    )
    await svc.start()
    facts = sorted(res["facts"], key=lambda f: f["id"])
    new_id: dict[int, int] = {}
    lines_of: dict[int, list[str]] = {}

    def load(store):
        for ln in sorted(week.values(), key=lambda x: x.when):
            if not is_filler(ln.text):
                store.add_utterance(ln.text, ln.when.timestamp(), vector=svc.embedder.embed_document(ln.text), gap_s=90, max_s=600)
        for f in facts:
            ts = [week[i].when.timestamp() for i in f["lines"] if i in week]
            heard = max(ts) if ts else ASK_AT.timestamp() - 86400
            event = datetime.fromisoformat(f["event"]).timestamp() if f.get("event") else None
            for name in {*f["people"], *([f["subject"]] if f["subject"] not in (None, "owner") else [])}:
                if name and name[0].isupper() and len(name.split()) <= 2:
                    store.add_person(name)
            nf = store.add_fact(f["text"], f["kind"], svc.embedder.embed_document(f["text"]), confidence=0.8,
                                source=f.get("source", "extracted"), subject=f["subject"], people=f["people"],
                                event_ts=event, heard_ts=heard)
            new_id[f["id"]] = nf.id
            lines_of[nf.id] = list(f["lines"])
        for f in facts:
            if f["status"] != "active" and f.get("superseded_by") in new_id:
                store.supersede(new_id[f["id"]], new_id[f["superseded_by"]], f["status"])

    await svc.run(load)
    return svc, lines_of, res


async def evaluate(svc, lines_of: dict[int, list[str]], questions: list[Question] | None = None, k: int = 5) -> list[Row]:
    from tests.eval.recall_benchmark.runner import FixedClock, fixed_clock

    questions = questions or load_questions()
    week_text = {ln.text: ln.id for v in ("clean", "noisy") for ln in load_week(v)}
    rows = []
    with fixed_clock(FixedClock(ASK_AT)):
        for q in questions:
            r = await svc.recall(q.q, k)
            r.lines = await svc.recall_lines(r.plan, 3)  # always, to calibrate the transcript fallback
            got = [{"id": h.fact.id, "lines": lines_of.get(h.fact.id, []), "cos": h.similarity, "score": h.score,
                    "rerank": h.detail.get("rerank"), "coverage": h.detail.get("coverage"), "rrf": h.detail.get("rrf"),
                    "text": h.fact.text}
                   for h in r.hits]
            lines = {x for g in got for x in g["lines"]}
            best_line = r.lines[0] if r.lines else None
            rows.append(Row(q, got, [s for s in q.support if s in lines], r.confidence,
                            best_line.score if best_line else None,
                            bool(best_line and week_text.get(best_line.line.text) in q.support),
                            r.plan.describe(), [s.detail() for s in r.filtered[:5]]))
    return rows


async def evaluate_before(svc, lines_of: dict[int, list[str]], questions: list[Question] | None = None, k: int = 5) -> list[Row]:
    """The retrieval Kin had before hybrid recall (cosine top-k + 2 keyword + 3 same-day places, cosine floor)."""
    from kin.memory.people import NameIndex
    from kin.memory.timescope import day_window

    questions = questions or load_questions()

    def one(store, q: Question):
        text, _ = NameIndex(store.people(), svc.name_match).normalise_text(q.q)
        return store.search(svc.embedder.embed_query(text), k, text, window=day_window(text, ASK_AT))

    rows = []
    for q in questions:
        hits = await svc.run(lambda store, q=q: one(store, q))
        got = [{"id": h.fact.id, "lines": lines_of.get(h.fact.id, []), "cos": h.similarity, "score": h.similarity,
                "text": h.fact.text} for h in hits]
        lines = {x for g in got for x in g["lines"]}
        rows.append(Row(q, got, [s for s in q.support if s in lines], hits[0].similarity if hits else None, None, False, {}, []))
    return rows


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def summary(rows: list[Row]) -> dict:
    ans = [r for r in rows if not r.q.abstain]
    una = [r for r in rows if r.q.abstain]
    sup = [r for r in ans if r.q.support]
    by_cat: dict[str, list[float]] = {}
    for r in sup:
        by_cat.setdefault(r.q.cat, []).append(len(r.support_retrieved) / len(r.q.support))
    pos = [r.confidence or 0.0 for r in ans]
    neg = [r.confidence or 0.0 for r in una]
    return {
        "recall_at_5": round(statistics.mean(len(r.support_retrieved) / len(r.q.support) for r in sup), 3),
        "any_support_at_5": round(sum(bool(r.support_retrieved) for r in sup) / len(sup), 3),
        "top1_support": round(sum(bool(r.retrieved and set(r.retrieved[0]["lines"]) & set(r.q.support)) for r in sup) / len(sup), 3),
        "by_category": {c: round(statistics.mean(v), 3) for c, v in sorted(by_cat.items())},
        "abstain_auc": auc(pos, neg),
        "answerable_conf_median": round(statistics.median(pos), 3),
        "unanswerable_conf_max": round(max(neg), 3) if neg else None,
    }


def sweep(rows: list[Row], thresholds: list[float]) -> list[dict]:
    """For each threshold: answerable questions that still reach the LLM, unanswerable ones that would too."""
    ans = [r for r in rows if not r.q.abstain]
    una = [r for r in rows if r.q.abstain]
    out = []
    for t in thresholds:
        keep_a = [r for r in ans if (r.confidence or 0) >= t]
        out.append({
            "t": t,
            "answerable_kept": len(keep_a),
            "answerable_kept_with_support": sum(bool(r.support_retrieved) for r in keep_a),
            "unanswerable_passed": sum((r.confidence or 0) >= t for r in una),
        })
    return out
