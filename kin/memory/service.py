"""Async facade over the embedder and the encrypted store.

Both live on one `ModelWorker` thread: SQLite connections are thread-bound
and the embedder is not re-entrant. Every mutation of facts is announced on the bus.

Recall (`recall`) is hybrid: the question is parsed for a time window and known people (query.py),
candidates come from vectors, BM25 keywords, the window and the people (store.py), and they are fused
with Reciprocal Rank Fusion, filtered and time-boosted (retrieval.py), optionally reranked by a small
cross-encoder (rerank.py). The result carries every score and every filtered fact with its reason.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import numpy as np

from kin.config import RetrievalConfig
from kin.core.bus import EventBus
from kin.core.events import FactChanged, FactSuperseded, MemoriesDeleted, MemoryStored
from kin.core.workers import ModelWorker
from kin.memory.embed import Embedder
from kin.memory.people import NameIndex
from kin.memory.query import QueryPlan, understand
from kin.memory.retrieval import FusionWeights, Scored, conflicts, content_words, fuse, mentions, rrf
from kin.memory.store import Change, Episode, Fact, Hit, MemoryStore, Person, Utterance  # noqa: F401

log = logging.getLogger(__name__)

DUPLICATE_WINDOW_S = 600.0
T = TypeVar("T")


@dataclass(frozen=True)
class LineHit:
    """A raw transcript line offered as evidence when no fact answers the question."""

    line: Utterance
    similarity: float
    score: float  # the confidence compared with `utterance_floor`

    def to_dict(self) -> dict:
        return {"id": self.line.id, "text": self.line.text, "ts": self.line.ts, "cos": round(self.similarity, 4),
                "score": round(self.score, 4)}


@dataclass
class Recall:
    """Everything one recall decided: the plan, the facts offered (best first), the ones filtered out and why."""

    plan: QueryPlan
    hits: list[Hit]
    filtered: list[Scored] = field(default_factory=list)
    relaxed: list[str] = field(default_factory=list)  # filters dropped because nothing passed them
    conflicts: list[tuple[int, int]] = field(default_factory=list)
    lines: list[LineHit] = field(default_factory=list)  # transcript fallback (see `recall(lines=True)`)
    ms: float = 0.0
    spellings: dict[str, tuple[str, ...]] = field(default_factory=dict)  # every known person: name, aliases, heard here

    @property
    def confidence(self) -> float | None:
        """The best confidence among the facts offered (the model sees all of them, so any can answer)."""
        return max((h.score for h in self.hits if h.score is not None), default=None)

    def explain(self, limit: int = 8) -> dict:
        return {
            "plan": self.plan.describe(),
            "used": [h.detail | {"confidence": h.score} for h in self.hits],
            "filtered": [s.detail() for s in self.filtered[:limit]],
            "relaxed": self.relaxed,
            "conflicts": [list(p) for p in self.conflicts],
            "lines": [x.to_dict() for x in self.lines],
            "ms": round(self.ms, 1),
        }


class MemoryService:
    def __init__(
        self,
        bus: EventBus,
        embedder_factory: Callable[[], Embedder],
        store_factory: Callable[[int], MemoryStore],
        *,
        name_match: float = 80.0,
        retrieval: RetrievalConfig | None = None,
        reranker_factory: Callable[[], object] | None = None,
        only_about_me: bool = False,
    ) -> None:
        self._bus = bus
        self._embedder_factory = embedder_factory
        self._store_factory = store_factory
        self._reranker_factory = reranker_factory
        self._worker = ModelWorker("memory")
        self.name_match = name_match
        self.cfg = retrieval or RetrievalConfig()
        self.embedder: Embedder | None = None
        self.store: MemoryStore | None = None
        self.reranker = None
        self.reembedded = 0  # vectors computed at start (a new embedding size, or lines from before schema v5)
        self._only_about_me = only_about_me

    @property
    def weights(self) -> FusionWeights:
        c = self.cfg
        return FusionWeights(c.rrf_k, c.dense_weight, c.lexical_weight, c.alt_weight, c.future_boost, c.past_event_penalty,
                             c.recency_boost, c.recency_half_life_days, c.person_mode, c.person_boost)

    async def start(self) -> None:
        def load() -> None:
            self.embedder = self._embedder_factory()
            self.store = self._store_factory(self.embedder.dim)
            if self.cfg.rerank and self._reranker_factory is not None:
                self.reranker = self._reranker_factory()
            self.reembedded = self._fill_vectors()
            saved = self.store.get_meta("only_about_me")  # the UI setting, kept in the encrypted database
            if saved is not None:
                self._only_about_me = saved == "1"

        await self._worker.run(load)

    def _fill_vectors(self) -> int:
        """Embed facts and lines that have no vector: after a change of embedding size, or lines kept
        before schema v5. Runs once at start; a few hundred lines take seconds."""
        n = 0
        for table in ("facts", "utterances"):
            missing = self.store.missing_vectors(table)
            for rowid, text in missing:
                self.store.set_vector(table, rowid, self.embedder.embed_document(text))
            n += len(missing)
        if n:
            log.info("embedded %d facts and lines that had no %d-d vector", n, self.embedder.dim)
        return n

    async def close(self) -> None:
        def unload() -> None:
            if self.store is not None:
                self.store.close()
            if self.embedder is not None:
                self.embedder.close()
            if self.reranker is not None:
                self.reranker.close()

        await self._worker.run(unload)
        self._worker.shutdown()

    # --- facts: writes -----------------------------------------------------------------------

    async def remember(
        self,
        text: str,
        kind: str,
        confidence: float,
        *,
        source: str = "explicit",
        ts: float | None = None,
        vector: np.ndarray | None = None,
        subject: str | None = None,
        people: list[str] | tuple[str, ...] = (),
        event_ts: float | None = None,
        evidence: list[int] | tuple[int, ...] = (),
        episode_id: int | None = None,
    ) -> Fact | None:
        """Store one fact (None if the same text was stored in the last 10 minutes).
        `vector` reuses a document embedding already computed."""

        def add() -> Fact | None:
            now = time.time() if ts is None else ts
            if self.store.exists_recent(text, now - DUPLICATE_WINDOW_S):
                return None
            vec = vector if vector is not None else self.embedder.embed_document(text)
            return self.store.add_fact(
                text, kind, vec, confidence=confidence, source=source, subject=subject, people=people,
                event_ts=event_ts, heard_ts=now, evidence=evidence, episode_id=episode_id,
            )

        fact = await self._worker.run(add)
        if fact is not None:
            self._announce(fact)
        return fact

    def _announce(self, f: Fact) -> None:
        self._bus.publish(
            MemoryStored(id=f.id, text=f.text, kind=f.kind, confidence=f.confidence, memory_ts=f.heard_ts,
                         source=f.source, subject=f.subject, event_ts=f.event_ts, evidence=",".join(map(str, f.evidence)))
        )

    async def supersede(self, old_id: int, new_id: int, status: str = "superseded") -> bool:
        ok = await self._worker.run(lambda: self.store.supersede(old_id, new_id, status))
        if ok:
            self._bus.publish(FactSuperseded(old_id=old_id, new_id=new_id, status=status))  # type: ignore[arg-type]
        return ok

    async def add_evidence(self, fact_id: int, utterance_ids: list[int]) -> None:
        await self._worker.run(lambda: self.store.add_evidence(fact_id, utterance_ids))

    async def similar_active(
        self, vector: np.ndarray, k: int, *, subject: str | None, people: list[str] | tuple[str, ...], text: str | None = None
    ) -> list[Hit]:
        """Active facts close to a new one that share its subject or one of its people (or have no subject yet)."""

        def find() -> list[Hit]:
            hits = self.store.search(vector, max(k * 4, 20), text)
            names = {p.lower() for p in people} | ({subject.lower()} if subject else set())
            out = []
            for h in hits:
                f = h.fact
                theirs = {p.lower() for p in f.people} | ({f.subject.lower()} if f.subject else set())
                if not names or not theirs or names & theirs or (subject or "") == (f.subject or ""):
                    out.append(h)
            return out[:k]

        return await self._worker.run(find)

    # --- deletes (the user's forget; superseding never deletes) ----------------------------------

    async def delete(self, ids: list[int], reason: str = "user_delete") -> list[int]:
        deleted = await self._worker.run(lambda: self.store.delete(ids))
        self._announce_delete(deleted, reason)
        return deleted

    async def forget_last(self, minutes: float, reason: str) -> list[int]:
        since = time.time() - minutes * 60
        deleted = await self._worker.run(lambda: self.store.delete_since(since))
        self._announce_delete(deleted, reason)
        return deleted

    async def ids_since(self, minutes: float) -> list[int]:
        since = time.time() - minutes * 60
        return await self._worker.run(lambda: self.store.ids_since(since))

    async def forget_ids(self, ids: list[int], *, pending_since_minutes: float | None, reason: str) -> list[int]:
        """Delete exactly these facts (the ones the user confirmed), plus the raw lines heard in the same span."""

        def drop() -> list[int]:
            deleted = self.store.delete(ids)
            if pending_since_minutes is not None:
                self.store.delete_utterances_since(time.time() - pending_since_minutes * 60)
            return deleted

        deleted = await self._worker.run(drop)
        self._announce_delete(deleted, reason)
        return deleted

    async def forget_most_recent(self, reason: str) -> list[int]:
        def drop() -> list[int]:
            latest = self.store.recent(limit=1)
            return self.store.delete([latest[0].id]) if latest else []

        deleted = await self._worker.run(drop)
        self._announce_delete(deleted, reason)
        return deleted

    def _announce_delete(self, ids: list[int], reason: str) -> None:
        self._bus.publish(MemoriesDeleted(count=len(ids), reason=reason, ids=",".join(map(str, ids))))  # type: ignore[arg-type]

    # --- utterances & episodes ---------------------------------------------------------------------

    async def hear(
        self, text: str, ts: float, *, speaker: str | None, stt_model: str | None, worth: int, gap_s: float, max_s: float,
        vector: np.ndarray | None = None,
    ) -> tuple[Utterance, list[Episode]]:
        """Store one heard line. `vector` (the document embedding the router already computed) is kept for the
        transcript fallback; without one the line is embedded here."""

        def add() -> tuple[Utterance, list[Episode]]:
            vec = vector if vector is not None else self.embedder.embed_document(text)
            return self.store.add_utterance(text, ts, speaker=speaker, stt_model=stt_model, worth=worth, gap_s=gap_s,
                                            max_s=max_s, vector=vec)

        return await self._worker.run(add)

    async def close_idle_episodes(self, now: float, gap_s: float, max_s: float, *, force: bool = False) -> list[Episode]:
        return await self._worker.run(lambda: self.store.close_idle_episodes(now, gap_s, max_s, force=force))

    async def ready_episodes(self, now: float) -> list[Episode]:
        return await self._worker.run(lambda: self.store.ready_episodes(now))

    async def episode_lines(self, eid: int) -> list[Utterance]:
        return await self._worker.run(lambda: self.store.episode_lines(eid))

    async def discard_partial_extraction(self, eid: int) -> list[int]:
        """An episode that is extracted again (ready or failed) but already has facts was interrupted half way
        (the process died while its facts were being stored): drop them so the new run does not duplicate them."""
        ids = await self._worker.run(lambda: self.store.episode_fact_ids(eid))
        return await self.delete(ids, reason="resume_extraction") if ids else []

    async def mark_episode(self, eid: int, status: str, *, error: str | None = None, next_try_ts: float | None = None) -> None:
        await self._worker.run(lambda: self.store.mark_episode(eid, status, error=error, next_try_ts=next_try_ts))

    async def episode_counts(self) -> dict[str, int]:
        return await self._worker.run(lambda: self.store.episode_counts())

    async def purge_utterances(self, retention_days: float, now: float | None = None) -> int:
        before = (time.time() if now is None else now) - retention_days * 86400
        n = await self._worker.run(lambda: self.store.purge_utterances(before))
        if n:
            log.info("retention: purged %d raw transcript lines older than %.0f days", n, retention_days)
        return n

    async def utterances(self, ids: list[int] | None = None, limit: int = 200) -> list[Utterance]:
        return await self._worker.run(lambda: self.store.utterances(ids, limit))

    async def search_utterances(self, text: str, limit: int = 20) -> list[Utterance]:
        return await self._worker.run(lambda: self.store.search_utterances(text, limit))

    # --- people ------------------------------------------------------------------------------------

    async def people(self) -> list[Person]:
        return await self._worker.run(lambda: self.store.people())

    async def name_index(self) -> NameIndex:
        return NameIndex(await self.people(), self.name_match)

    async def add_person(self, name: str, aliases: list[str] | tuple[str, ...] = (), source: str = "extracted") -> Person:
        return await self._worker.run(lambda: self.store.add_person(name, aliases, source))

    async def rename_person(self, pid: int, name: str) -> Person | None:
        return await self._worker.run(lambda: self.store.rename_person(pid, name))

    async def set_aliases(self, pid: int, aliases: list[str]) -> Person | None:
        return await self._worker.run(lambda: self.store.set_aliases(pid, aliases))

    async def delete_person(self, pid: int) -> bool:
        return await self._worker.run(lambda: self.store.delete_person(pid))

    # --- embeddings ----------------------------------------------------------------------------------

    async def embed_document(self, text: str) -> np.ndarray:
        return await self._worker.run(lambda: self.embedder.embed_document(text))

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return await self._worker.run(lambda: np.stack([self.embedder.embed_document(t) for t in texts]))

    @property
    def embedder_name(self) -> str:
        return getattr(self.embedder, "name", "unknown")

    # --- reads -------------------------------------------------------------------------------------

    async def retrieve(self, query: str, k: int, *, alt: str | None = None) -> list[Hit]:
        """Active facts for a question, best first (see `recall` for the whole decision)."""
        return (await self.recall(query, k, alt=alt)).hits

    async def recall(self, query: str, k: int, *, alt: str | None = None, lines: bool = False) -> Recall:
        """Hybrid recall. `alt` is the model's rewording of the question (searched too, fused with less weight).
        `lines`: if no fact reaches `answer_floor`, also search the raw transcript (the "heard on" fallback)."""

        def run() -> Recall:
            t0 = time.perf_counter()
            r, queries = self._recall(query, k, alt)
            if lines and (r.confidence is None or r.confidence < self.cfg.answer_floor):
                r.lines = self._recall_lines(r.plan, 3, queries)
            r.ms = (time.perf_counter() - t0) * 1000
            return r

        return await self._worker.run(run)

    async def scope_facts(self, plan: QueryPlan, limit: int) -> list[Fact]:
        """Active facts about the people the question names and/or inside its time window: pinned first, then
        newest. The deterministic list behind "what do I know about Rahul" and "what did I plan yesterday"."""
        from kin.memory.retrieval import in_window

        def run() -> list[Fact]:
            idx = NameIndex(self.store.people(), self.name_match)
            aliases = {p.name: p.aliases for p in idx.people}
            if plan.people:
                spellings = [s for p in plan.people for s in (p, *aliases.get(p, ()))]
                facts = [self.store.get(i) for i in self.store.ids_about(spellings)]
                facts = [f for f in facts if f is not None and f.status == "active" and mentions(f, plan.people, aliases)]
            else:
                facts = self.store.recent(limit=2000)
            if plan.window:
                facts = [f for f in facts if in_window(f, plan.window, plan.time_field)]
            facts.sort(key=lambda f: (not f.pinned, -f.heard_ts))
            return facts[:limit]

        return await self._worker.run(run)

    async def recall_lines(self, plan: QueryPlan, n: int = 3) -> list[LineHit]:
        """The transcript search alone, whatever the facts say (UI, evaluation)."""
        return await self._worker.run(lambda: self._recall_lines(plan, n))

    def _plan(self, query: str, alt: str | None) -> tuple[QueryPlan, NameIndex]:
        idx = NameIndex(self.store.people(), self.name_match)
        return understand(query, datetime.fromtimestamp(time.time()), idx).with_alt(alt), idx

    def _queries(self, plan: QueryPlan) -> dict[str, tuple[np.ndarray, str]]:
        queries = {"q": (self.embedder.embed_query(plan.search), plan.search)}
        if plan.alt:
            queries["alt"] = (self.embedder.embed_query(plan.alt), plan.alt)
        if (heard := plan.search_original) and heard != plan.search:
            queries["heard"] = (self.embedder.embed_query(heard), heard)  # fused at the rewording's weight
        return queries

    def _recall(self, query: str, k: int, alt: str | None) -> tuple[Recall, dict]:
        c = self.cfg
        plan, idx = self._plan(query, alt)
        queries = self._queries(plan)
        facts, cos, lexical = self.store.candidates(queries, c.candidates, history=plan.history, window=plan.window,
                                                    people=plan.people)
        fused = fuse(plan, facts, cos, lexical, time.time(), self.weights, {p.name: p.aliases for p in idx.people})
        ranked = fused.ranked
        if self.reranker is not None and ranked:
            top = ranked[: c.rerank_top]
            for s, p in zip(top, self.reranker.scores(plan.search, [s.fact.text for s in top])):
                s.rerank = p
            tiered = c.person_mode == "tier"
            ranked = sorted(top, key=lambda s: (len([m for m in s.misses if tiered or m != "person"]), -(s.rerank or 0.0), -s.score))                 + ranked[c.rerank_top :]
        hits = [s.hit(self._confidence(s.cos, s.coverage, s.rerank)) for s in ranked[:k]]
        if plan.people:  # pinned facts about a person the question names are always offered
            shown = {h.fact.id for h in hits}
            aliases = {p.name: p.aliases for p in idx.people}
            for s in ranked[k:]:
                if s.fact.pinned and s.fact.id not in shown and mentions(s.fact, plan.people, aliases) and len(hits) < k + 3:
                    s.why_out = None
                    hit = s.hit(self._confidence(s.cos, s.coverage, s.rerank))
                    hit.detail["pinned"] = True
                    hits.append(hit)
        pairs = conflicts(hits, self.store.vectors([h.fact.id for h in hits]), similarity=c.conflict_similarity)
        spellings = {p.name: (p.name, *p.aliases, *(k for k, v in plan.fixes.items() if v == p.name)) for p in idx.people}
        return Recall(plan, hits, fused.filtered, fused.relaxed, pairs, spellings=spellings), queries

    def _confidence(self, cos: float, coverage: float, rerank: float | None) -> float:
        """The number the gate thresholds: how sure we are that this fact (or line) answers the question."""
        mode = self.cfg.confidence
        if mode == "rerank" and rerank is not None:
            return rerank
        if mode == "cos":
            return cos
        return cos + self.cfg.coverage_weight * coverage

    def _recall_lines(self, plan: QueryPlan, n: int, queries: dict | None = None) -> list[LineHit]:
        """Transcript fallback: raw lines ranked like facts (dense + BM25, RRF), inside the time window if any."""
        c = self.cfg
        queries = queries or self._queries(plan)
        lines, cos, lexical = self.store.utterance_candidates(queries, c.candidates, plan.window)
        if not lines:
            return []
        lists = [("dense" if name == "q" else f"dense:{name}", sorted(sims, key=lambda i: -sims[i]),
                  c.dense_weight * (1.0 if name == "q" else c.alt_weight)) for name, sims in cos.items()]
        lists += [("lexical" if name == "q" else f"lexical:{name}", ids, c.lexical_weight * (1.0 if name == "q" else c.alt_weight))
                  for name, ids in lexical.items()]
        scores, _ = rrf(lists, c.rrf_k)
        best = sorted(lines, key=lambda i: -scores.get(i, 0.0))[: max(n, c.rerank_top if self.reranker else n)]
        probs = self.reranker.scores(plan.search, [lines[i].text for i in best]) if self.reranker is not None else [None] * len(best)
        qwords = content_words(plan.search)
        out = []
        for i, p in zip(best, probs):
            sim = max((x.get(i, 0.0) for x in cos.values()), default=0.0)
            cov = len(qwords & content_words(lines[i].text)) / len(qwords) if qwords else 0.0
            out.append(LineHit(lines[i], sim, self._confidence(sim, cov, p)))
        out.sort(key=lambda x: -x.score)
        return out[:n]

    # --- corrections, review, consent (Prompt 4) ------------------------------------------------------------

    def _changed(self, action: str, fact_id: int, actor: str, other_id: int | None = None, text: str = "") -> None:
        self._bus.publish(FactChanged(action=action, fact_id=fact_id, other_id=other_id, actor=actor, text=text))  # type: ignore[arg-type]

    async def correct(self, fid: int, new_text: str, *, actor: str = "voice", event_ts: float | None = None) -> Fact | None:
        """A corrected version replaces the fact: a new active fact, the old one superseded (both kept, logged)."""
        old = await self.get(fid)
        if old is None or old.status != "active":
            return None
        from kin.brain.corrections import corrected_event

        event = event_ts if event_ts is not None else corrected_event(old.event_ts, new_text)

        def write() -> Fact:
            vec = self.embedder.embed_document(new_text)
            new = self.store.add_fact(new_text, old.kind, vec, confidence=1.0, source="explicit", subject=old.subject,
                                      people=old.people, event_ts=event, evidence=old.evidence, episode_id=old.episode_id,
                                      actor=actor)
            self.store.supersede(fid, new.id, actor=actor, action="correct")
            if old.pinned:
                self.store.set_pinned(new.id, True, actor)
            return self.store.get(new.id)

        new = await self._worker.run(write)
        self._announce(new)
        self._bus.publish(FactSuperseded(old_id=fid, new_id=new.id, status="superseded"))
        self._changed("correct", fid, actor, new.id, new_text)
        return new

    async def retract(self, fid: int, *, actor: str = "voice") -> bool:
        ok = await self._worker.run(lambda: self.store.retract(fid, actor))
        if ok:
            self._changed("retract", fid, actor)
        return ok

    async def restore(self, fid: int, *, actor: str = "ui") -> bool:
        ok = await self._worker.run(lambda: self.store.restore(fid, actor))
        if ok:
            self._changed("restore", fid, actor)
        return ok

    async def pin(self, fid: int, pinned: bool = True, *, actor: str = "ui") -> bool:
        ok = await self._worker.run(lambda: self.store.set_pinned(fid, pinned, actor))
        if ok:
            self._changed("pin" if pinned else "unpin", fid, actor)
        return ok

    async def merge(self, keep_id: int, other_id: int, text: str | None = None, *, actor: str = "ui") -> Fact | None:
        def run() -> Fact | None:
            vec = self.embedder.embed_document(text) if text else None
            return self.store.merge(keep_id, other_id, text, vec, actor)

        fact = await self._worker.run(run)
        if fact is not None:
            self._bus.publish(FactSuperseded(old_id=other_id, new_id=keep_id, status="superseded"))
            self._changed("merge", keep_id, actor, other_id, fact.text)
        return fact

    async def changes(self, since: float, until: float | None = None) -> list[Change]:
        return await self._worker.run(lambda: self.store.changes(since, until))

    async def fact_changes(self, fid: int) -> list[Change]:
        return await self._worker.run(lambda: self.store.fact_changes(fid))

    async def review(self, day: datetime | None = None) -> dict:
        """"What I remembered today": facts added that day (still active or not), and every change made to one."""
        start = (day or datetime.fromtimestamp(time.time())).replace(hour=0, minute=0, second=0, microsecond=0)
        a, b = start.timestamp(), start.timestamp() + 86400

        def run() -> dict:
            changes = self.store.changes(a, b)
            added_ids = [c.fact_id for c in changes if c.action == "add"]
            facts = {f.id: f for f in (self.store.get(i) for i in dict.fromkeys(added_ids)) if f is not None}
            return {
                "day": start.date().isoformat(),
                "added": [f.to_dict() for f in facts.values()],
                "changes": [c.to_dict() for c in changes if c.action != "add"],
                "still_active": sum(f.status == "active" for f in facts.values()),
            }

        return await self._worker.run(run)

    async def about(self, name: str) -> tuple[list[str], list[int], list[int]]:
        """Spellings of a person (canonical name + aliases), and the facts and raw lines that name them."""

        def run():
            idx = NameIndex(self.store.people(), self.name_match)
            canon = idx.canonical(name) or name.strip()
            p = self.store.get_person(canon)
            spellings = [canon, *(p.aliases if p else ())]
            return spellings, self.store.ids_about(spellings), self.store.utterance_ids_about(spellings)

        return await self._worker.run(run)

    async def forget_person(self, name: str, reason: str = "forget_command") -> tuple[list[int], int]:
        """Delete every fact and raw line naming this person, and the person. Only after the user confirmed."""
        spellings, fids, uids = await self.about(name)

        def run() -> tuple[list[int], int]:
            deleted = self.store.delete(fids)
            n = self.store.delete_utterances(uids)
            p = self.store.get_person(spellings[0])
            if p is not None:
                self.store.delete_person(p.id)
            return deleted, n

        deleted, lines = await self._worker.run(run)
        self._announce_delete(deleted, reason)
        return deleted, lines

    @property
    def only_about_me(self) -> bool:
        return self._only_about_me

    async def set_only_about_me(self, value: bool) -> None:
        self._only_about_me = bool(value)
        await self._worker.run(lambda: self.store.set_meta("only_about_me", "1" if value else "0"))

    # --- maintenance --------------------------------------------------------------------------------------

    async def maintain(self, utterance_days: float, fact_days: float | None, now: float | None = None) -> dict:
        """Retention purges, FTS optimize, VACUUM, integrity check. The report is kept in the database."""
        now = time.time() if now is None else now

        def run() -> dict:
            t0 = time.perf_counter()
            report = {"ts": now, "purged_lines": self.store.purge_utterances(now - utterance_days * 86400)}
            report["purged_facts"] = len(self.store.purge_facts(now - fact_days * 86400)) if fact_days else 0
            report |= self.store.optimize()
            report["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self.store.set_meta("last_maintenance", json.dumps(report))
            return report

        report = await self._worker.run(run)
        log.info("maintenance: %s", report)
        return report

    async def health(self) -> dict:
        def run() -> dict:
            h = self.store.health()
            last = self.store.get_meta("last_maintenance")
            h["last_maintenance"] = json.loads(last) if last else None
            h["only_about_me"] = self._only_about_me
            return h

        return await self._worker.run(run)

    async def backup(self, folder: Path, passphrase: str) -> Path:
        from kin.memory.backup import write_backup

        return await self._worker.run(lambda: write_backup(self.store, Path(folder), passphrase))

    async def demo_reset(self) -> dict:
        """Wipe the demo database and load the demo week again (refuses a database that is not a demo one)."""
        from kin.memory import demo_week

        return await self._worker.run(lambda: demo_week.seed(self.store, self.embedder.embed_document))

    async def recent(self, limit: int = 100, contains: str | None = None, status: str | None = "active") -> list[Fact]:
        return await self._worker.run(lambda: self.store.recent(limit=limit, contains=contains, status=status))

    async def get(self, fid: int) -> Fact | None:
        return await self._worker.run(lambda: self.store.get(fid))

    async def history(self, fid: int) -> list[Fact]:
        return await self._worker.run(lambda: self.store.history(fid))

    async def count(self, status: str | None = "active") -> int:
        return await self._worker.run(lambda: self.store.count(status))

    async def texts(self) -> list[str]:
        return await self._worker.run(lambda: [m.text for m in self.store.recent(limit=100_000)])

    async def run(self, fn: Callable[[MemoryStore], T]) -> T:
        """Run any store operation on the memory worker (tasks and reminders share the encrypted DB)."""
        return await self._worker.run(lambda: fn(self.store))
