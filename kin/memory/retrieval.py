"""Hybrid ranking of facts for a question: pure functions, no I/O (the store gathers candidates).

  1. Filters with backfill. Facts that pass every filter the question implies (its time window, the
     people it names) rank first; facts failing one filter come next, then those failing both. With
     enough matches this is a hard filter; with few, the best other facts fill the remaining places
     and are marked as backfill. (A strict person filter lost answers: facts heard without a known
     speaker say "Someone said: my new flat is in Dharampeth", not "Sam".) Superseded facts take
     part only when the question asks about history.
  2. Reciprocal Rank Fusion over ranked lists: dense (EmbeddingGemma cosine) and lexical (FTS5 BM25)
     for the question, and again for the model's rewording if there is one:
         rrf(f) = sum_i  w_i / (k + rank_i(f))        k = 60
  3. Time boost, multiplicative: questions about the future prefer upcoming events and demote ones
     already past; otherwise a mild decay by when the fact was heard.

Every candidate keeps its scores and, if it was left out, the reason (time, person, superseded),
for the "Why this answer?" panel.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from kin.memory.query import QueryPlan
from kin.memory.store import Fact, Hit


@dataclass(frozen=True)
class FusionWeights:
    rrf_k: float = 60.0
    dense: float = 1.0
    lexical: float = 1.0
    alt: float = 0.5  # lists from the model's rewording count this much of the question's own
    future_boost: float = 0.3  # "when is ...?": upcoming events x1.3
    past_event_penalty: float = 0.2  # "when is ...?": events already past x0.8
    recency_boost: float = 0.1  # otherwise: just heard x1.1, decaying with the half-life
    recency_half_life_days: float = 7.0
    person_mode: str = "tier"  # tier: facts about the named person rank first (backfill); boost: x(1 + person_boost)
    person_boost: float = 0.3


@dataclass
class Scored:
    fact: Fact
    cos: float = 0.0  # best cosine over the question (and its rewording)
    ranks: dict[str, int] = field(default_factory=dict)  # list name -> 1-based rank, e.g. {"dense": 1, "lexical": 3}
    rrf: float = 0.0
    boost: float = 1.0
    score: float = 0.0  # rrf * boost (the fused score)
    rerank: float | None = None  # cross-encoder probability, if the reranker ran
    coverage: float = 0.0  # share of the question's content words found in the fact
    misses: tuple[str, ...] = ()  # filters it fails ("time", "person"): ranked after full matches (backfill)
    why_out: str | None = None  # why it was left out entirely (superseded) or None

    def detail(self) -> dict:
        return {
            "id": self.fact.id, "text": self.fact.text, "status": self.fact.status, "cos": round(self.cos, 4),
            "ranks": self.ranks, "rrf": round(self.rrf, 5), "boost": round(self.boost, 3), "score": round(self.score, 5),
            "rerank": None if self.rerank is None else round(self.rerank, 4), "coverage": round(self.coverage, 3),
            "backfill": list(self.misses), "filtered": self.why_out,
        }

    def hit(self, confidence: float) -> Hit:
        return Hit(self.fact, round(self.cos, 4), score=round(confidence, 5), detail=self.detail())


# --- filters --------------------------------------------------------------------------------------


def _word(name: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", re.I)


def mentions(fact: Fact, people: tuple[str, ...] | list[str], aliases: dict[str, tuple[str, ...]] | None = None) -> bool:
    """Is any of these people the fact's subject, or named in its text? (Not its `people` list: E2B sometimes
    copies every known name into it, e.g. "The team offsite is in Lonavala" with seven people.)"""
    if not people:
        return True
    subject = (fact.subject or "").lower()
    for p in people:
        spellings = (p, *((aliases or {}).get(p, ())))
        if any(s.lower() == subject for s in spellings) or any(_word(s).search(fact.text) for s in spellings):
            return True
    return False


def in_window(fact: Fact, window: tuple[float, float] | None, time_field: str) -> bool:
    if window is None:
        return True
    a, b = window
    heard = a <= fact.heard_ts < b
    event = fact.event_ts is not None and a <= fact.event_ts < b
    if time_field == "heard":
        return heard
    if time_field == "event":
        return event
    return heard or event


def filter_reason(fact: Fact, plan: QueryPlan, aliases: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Every filter this fact fails: 'superseded', 'time', 'person' (empty: it is eligible)."""
    out = []
    if fact.status != "active" and not plan.history:
        out.append("superseded" if fact.status == "superseded" else fact.status)
    if not in_window(fact, plan.window, plan.time_field):
        out.append("time")
    if not mentions(fact, plan.people, aliases):
        out.append("person")
    return out


# --- fusion -------------------------------------------------------------------------------------------


def rrf(lists: list[tuple[str, list[int], float]], k: float = 60.0) -> tuple[dict[int, float], dict[int, dict[str, int]]]:
    """Reciprocal Rank Fusion. `lists` holds (name, ids best first, weight). Returns scores and each id's ranks."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ids, weight in lists:
        for r, i in enumerate(dict.fromkeys(ids), 1):
            scores[i] = scores.get(i, 0.0) + weight / (k + r)
            ranks.setdefault(i, {})[name] = r
    return scores, ranks


def time_boost(fact: Fact, plan: QueryPlan, now: float, w: FusionWeights) -> float:
    if plan.future and fact.event_ts is not None:
        return 1.0 + w.future_boost if fact.event_ts >= now - 3600 else 1.0 - w.past_event_penalty
    age_days = max(0.0, (now - fact.heard_ts) / 86400)
    return 1.0 + w.recency_boost * math.pow(0.5, age_days / max(w.recency_half_life_days, 1e-6))


_CONTENT = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or of to in on at for with about from by is am are was were be been do does did done have has had "
    "i me my mine you your we us our he she it they them their his her its what whats when where who whom which why how "
    "this that these those there here will would can could should shall may might must tell know remember told said say "
    "says any anything something much many some s t ll re ve d get got going go".split()
)


def content_words(text: str) -> set[str]:
    return {w.rstrip("s") if len(w) > 3 else w for w in _CONTENT.findall(text.lower()) if w not in _STOP}


def coverage(question_words: set[str], fact: Fact) -> float:
    if not question_words:
        return 0.0
    words = content_words(" ".join([fact.text, fact.subject or "", *fact.people]))
    return len(question_words & words) / len(question_words)


@dataclass
class Fused:
    ranked: list[Scored]  # eligible (or relaxed) candidates, best first
    filtered: list[Scored]  # left out (superseded), with the reason
    relaxed: list[str]  # filters no candidate passed (the ranking then ignores them)


def fuse(
    plan: QueryPlan,
    facts: dict[int, Fact],
    cos: dict[str, dict[int, float]],
    lexical: dict[str, list[int]],
    now: float,
    w: FusionWeights = FusionWeights(),
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> Fused:
    """Rank `facts` (the candidate pool) for the question.

    `cos` maps a query name ("q", "alt") to {fact id: cosine}; `lexical` maps it to fact ids in BM25 order.
    Ranks are taken within the eligible set, so a filtered question ranks its own candidates 1..n.
    """
    reasons = {i: filter_reason(f, plan, aliases) for i, f in facts.items()}
    person_tier = w.person_mode == "tier"
    # superseded facts are left out (unless the question asks about history); time and person only reorder
    eligible = [i for i in facts if "superseded" not in reasons[i] and "retracted" not in reasons[i]]
    eligible_set = set(eligible)
    relaxed = [r for r in ("time", "person") if any(r in reasons[i] for i in eligible)
               and all(r in reasons[i] for i in eligible)]  # filters nothing passed

    lists: list[tuple[str, list[int], float]] = []
    for name, sims in cos.items():
        weight = w.dense * (1.0 if name == "q" else w.alt)
        lists.append((f"dense:{name}" if name != "q" else "dense",
                      sorted((i for i in sims if i in eligible_set), key=lambda i: -sims[i]), weight))
    for name, ids in lexical.items():
        weight = w.lexical * (1.0 if name == "q" else w.alt)
        lists.append((f"lexical:{name}" if name != "q" else "lexical", [i for i in ids if i in eligible_set], weight))
    scores, ranks = rrf(lists, w.rrf_k)

    qwords = content_words(plan.search)
    out: list[Scored] = []
    for i in eligible:
        f = facts[i]
        s = Scored(f, cos=max((c.get(i, 0.0) for c in cos.values()), default=0.0), ranks=ranks.get(i, {}),
                   rrf=scores.get(i, 0.0), boost=time_boost(f, plan, now, w), coverage=coverage(qwords, f),
                   misses=tuple(reasons[i]))
        if not person_tier and plan.people:
            s.boost *= 1.0 if "person" in reasons[i] else 1.0 + w.person_boost
        s.score = s.rrf * s.boost
        out.append(s)
    out.sort(key=lambda s: (len([m for m in s.misses if person_tier or m != "person"]), -s.score, -s.cos, -s.fact.heard_ts))
    filtered = []
    for i, f in facts.items():
        if i in eligible_set:
            continue
        r = reasons[i]
        filtered.append(Scored(f, cos=max((c.get(i, 0.0) for c in cos.values()), default=0.0), why_out=", ".join(r)))
    filtered.sort(key=lambda s: -s.cos)
    return Fused(out, filtered, relaxed)


def max_rrf(n_lists: float, w: FusionWeights) -> float:
    """The fused score of a fact ranked first in every list (for normalising to 0..1)."""
    return n_lists / (w.rrf_k + 1)


# --- conflicts ----------------------------------------------------------------------------------------

_NUMBERS = re.compile(r"\d+(?:[:.]\d+)?")


def conflicts(hits: list[Hit], vectors: dict[int, np.ndarray], *, similarity: float = 0.8) -> list[tuple[int, int]]:
    """Pairs of active facts (older id first) that look like two versions of one thing but disagree:
    the same subject or a shared person, close in meaning, and a different time or different numbers.
    Neither superseded the other, so the answer should give both, with when each was heard."""
    pairs = []
    act = [h.fact for h in hits if h.fact.status == "active"]
    for x in range(len(act)):
        for y in range(x + 1, len(act)):
            a, b = sorted((act[x], act[y]), key=lambda f: (f.heard_ts, f.id))
            same = (a.subject and a.subject == b.subject and a.subject != "owner") or set(a.people) & set(b.people)
            if not same or a.id not in vectors or b.id not in vectors:
                continue
            if float(np.dot(vectors[a.id], vectors[b.id])) < similarity:
                continue
            differ = (a.event_ts is not None and b.event_ts is not None and abs(a.event_ts - b.event_ts) >= 60) or (
                set(_NUMBERS.findall(a.text)) != set(_NUMBERS.findall(b.text)))
            if differ:
                pairs.append((a.id, b.id))
    return pairs
