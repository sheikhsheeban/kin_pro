"""Orchestrator: turns transcripts into memories and queries into grounded, spoken answers.

Passive path (overheard speech), two tiers:
  final line -> filler ("okay", "yeah") is dropped; everything else is stored as an utterance
             -> grouped into an episode (new one after 90 s of silence or 10 min)
             -> the semantic router only pre-filters: an episode where every line is confidently
                chatter is skipped without the LLM
  finished episode -> Gemma 4 E2B extracts self-contained facts (JSON, validated, one retry), in the
             background, only while no question runs -> names normalised (rapidfuzz) -> each fact is
             ADDed, or UPDATEs / CONTRADICTS an active fact (the old one is superseded, never
             deleted), or is a DUPLICATE (the old fact gains the new evidence)

Active path (a question), in this order:
  1. safety guard  "forget the last 10 minutes" / "forget that" / "cancel the reminder ..." are
                   recognised by anchored regexes only, and never run without "confirm".
  2. hybrid recall runs at once, while the session is taken (memory/service.py: time and people from
                   the words, dense + BM25 fused with RRF, filters, optional reranker). If the best
                   confidence reaches `strong_score` and the words ask about the user's life (not a
                   command, not the weather), the facts go straight into the first message: one LLM
                   call answers (the strong-match shortcut, no separate tool-choice call).
  3. otherwise one Gemma 4 E2B call with tools (tools_schema.py). The model either answers directly
                   (chit-chat) or calls one tool: recall, remember, add_task, list_tasks,
                   complete_task, cancel_reminder, lookup. Calls are schema-validated; an invalid
                   call gets one correction, then falls back to recall.
  4. execute the tool. recall (re-run with the model's rewording fused in) and lookup feed their
                   result back and E2B writes the spoken answer (second call) through the citation
                   gate; task tools answer with a fixed sentence. Under `answer_floor` Kin abstains,
                   or quotes the closest raw line it heard ("I'm not sure, but on Saturday I heard ...").
  At most two tool calls per query. Lookups go only through the gateway (LED blue, "I checked ...").
  Each reply carries a "why" record: the plan, the facts offered with their scores, the ones filtered.

Every use of the LLM goes through one priority queue: questions first, background work
(pending tagging, warming the next session) after. The fixed system prompt + tool definitions
are prefilled into a spare session while idle, so a question only prefills its own tokens.

Runs on the asyncio loop. Blocking work goes to the LLM, memory, gateway and TTS workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from kin.brain import lookup as lk
from kin.brain.extract import ExtractedFact, ExtractionError, Extractor
from kin.brain.commands import SafeCommand, clean_query, confirmation_answer, parse_safe_command
from kin.brain.gate import ABSTAIN_REPLY, BLOCKED_REPLY, ERROR_REPLY, TIMEOUT_REPLY, Gate, strip_citations
from kin.brain.llm import LLMBackend, LlmSession, ToolCall, ToolResult
from kin.brain.llm_queue import ACTIVE, BACKGROUND, LlmQueue
from kin.brain.names import extract_names
from kin.brain.prompt import BuiltPrompt, PromptBuilder, format_memory
from kin.brain.semantic_router import SemanticRouter, Tag, fact_kind, is_filler
from kin.brain.text import SentenceSplitter
from kin.brain.tools_schema import TOOLS, InvalidCall, ValidCall, correction_message, system_prompt, validate
from kin.config import KinConfig
from kin.core.bus import EventBus
from kin.core.events import (
    ActivationEvent,
    AttendingEnded,
    BrainState,
    BrainStateChanged,
    ConfirmationRequested,
    ConfirmationResolved,
    EpisodeExtracted,
    Event,
    LookupPerformed,
    MemoryStored,
    PassiveTagged,
    QueryReceived,
    ReplyCompleted,
    ReplySentence,
    TranscriptEvent,
)
from kin.core.workers import ModelWorker
from kin.gateway.scrub import name_hash
from kin.io.tts import Speaker, Turn
from kin.memory.retrieval import mentions
from kin.memory.service import MemoryService, Recall
from kin.memory.store import WORTH_CHATTER, WORTH_LIKELY, WORTH_UNSURE, Episode, Fact, Hit
from kin.tools.lookup_client import LookupClient, LookupResult
from kin.tools.tasks import TaskService
from kin.tools.timeparse import describe, parse_due, parse_when

log = logging.getLogger(__name__)

LOOKUP_CONTEXT_FLOOR = 0.25  # memories this similar are offered as context (not a gate) to a lookup answer
MAX_TOOL_CALLS = 2
MAX_ANSWER_SENTENCES = 3
# The router over-predicts "commitment" for small talk ("let's get lunch", "I keep forgetting to drink water"),
# always with a margin near 0; real plans kept in the benchmark had >= 0.06 unless they named a person or place.
SMALL_TALK_MARGIN = 0.03  # a recall answer is spoken: more than three sentences are cut
LIST_RULE = ("Say what these memories tell about it in at most three short sentences, most important first; end each "
             "sentence with the [#N] of every memory it uses.")
RECALL_DONE = "Recall results for the user's question (recall was already called for you; do not call a tool):"
# The "else" clause is needed: without it E2B answered 3-4 unanswerable questions per run from the nearest
# fact ("Sam's birthday is on the 22nd of October", which is Rahul's). With it, misheard names made it abstain
# ("Raul's birthday ..." for "When is Rahul's birthday?"); the prompt now says which spellings are the same person.
# "One sentence unless asked for several things" made answers incomplete ("Aunty Shobha comes on the 10th" for
# "how long is she staying?"), so up to three sentences, best match first.
ANSWER_RULE = ("Answer in at most three short sentences from these memories only, best match first; end each sentence "
               "with the [#N] of every memory it uses. Do not say when something was heard unless the question asks or "
               "two memories disagree. If they do not answer it, say: I don't remember that.")
# The shortcut is only for questions about the user's life: never a command or an online lookup.
_COMMAND = re.compile(r"^(please\s+)?(remind me to|remember( that)?\b|add |put |mark |tick |cancel |forget |delete |set an? |"
                      r"note that|save )", re.I)
_ONLINE = re.compile(r"\b(weather|rain\w*|umbrella|forecast|temperature|sunny|snow|news|headlines?|look (it |that )?up|"
                     r"search|online|google|wikipedia|internet|stock|score)\b", re.I)
_QUESTION = re.compile(r"^(who|whom|whose|what|what's|whats|when|where|which|why|how|is|are|am|was|were|do|does|did|"
                       r"can|could|will|would|have|has|had|tell me|remind me (what|when|where|who|how|of))\b", re.I)
# Filler E2B adds once heard times are shown ("This information was heard on Monday.", "That is what you
# mentioned.", "This is the only information I have on that."): nothing in it answers, so it is not spoken.
_WHEN_WORD = r"(on|at|in|from|yesterday|today|last|earlier)\b"
_PROVENANCE = re.compile(
    r"^(this|that|it)( \w+)? (was|is) (heard|mentioned|shared|recorded|noted|saved|stored|said)( to you)? " + _WHEN_WORD
    + r"|^i (heard|learned|noted) (this|that|it) " + _WHEN_WORD
    + r"|^(that|this) is (what|all) (you|i) (mentioned|said|told me|know|remember|heard)\b"
    + r"|^this is the only (information|memory|thing) i (have|know|remember)\b", re.I)
CONFIRM_WINDOW_S = 10.0
WARMUP_MESSAGE = "Session start. Reply with one word: Ready."


@dataclass
class _Reply:
    """Mutable record of one reply while it is being produced."""

    query_id: str
    query: str
    started: float = field(default_factory=time.monotonic)
    spoken: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    blocked: int = 0
    answered: bool = False
    abstained: bool = False
    looked_up: list[str] = field(default_factory=list)  # kinds that returned data
    path: list[str] = field(default_factory=list)
    llm_calls: int = 0
    answer_sentences: int = 0  # cited sentences spoken (capped at MAX_ANSWER_SENTENCES)
    recall: Recall | None = None  # the retrieval behind the answer (for "why this answer?")
    shortcut: bool = False
    heard: list[int] = field(default_factory=list)  # raw lines quoted by the transcript fallback
    lookup_facts: str = ""  # what a lookup returned this turn: its numbers are grounded too


@dataclass
class _Pass:
    timed_out: bool = False
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class _ToolOutcome:
    """What executing one tool produced: either a final spoken outcome, or a result for the model."""

    content: str | None = None  # fed back to the model (None: the tool already answered)
    allowed: set[int] = field(default_factory=set)
    prompt: BuiltPrompt | None = None  # recall: the answer prompt (for llm.answer_model)
    web_ok: bool = False
    outcome: str | None = None  # set when the tool answered by itself


@dataclass
class _PendingConfirm:
    confirm_id: str
    cmd: SafeCommand
    ids: list[int]
    task_id: int | None
    what: str  # the spoken description of the thing to delete
    expires: float
    timer: asyncio.Task | None = None


class Brain:
    def __init__(
        self,
        cfg: KinConfig,
        bus: EventBus,
        *,
        llm_factory: Callable[[], LLMBackend],
        memory: MemoryService,
        speaker: Speaker,
        router: SemanticRouter,
        lookup_client: LookupClient | None = None,
        tasks: TaskService | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.memory = memory
        self.speaker = speaker
        self.router = router
        self.extractor = Extractor(lambda: self.llm, cfg.extract.max_output_tokens)
        self.lookup_client = lookup_client
        self.tasks = tasks
        self.gate = Gate(cfg.memory.retrieval.answer_floor)
        self._llm_factory = llm_factory
        self._llm_worker = ModelWorker("llm")
        self._net_worker = ModelWorker("gateway-client")
        self.llm: LLMBackend | None = None
        self.builder: PromptBuilder | None = None
        self.queue = LlmQueue()
        self.system = system_prompt()
        self.state = BrainState.IDLE
        self.last_reply: ReplyCompleted | None = None
        self._query_parts: list[str] = []
        self._task: asyncio.Task | None = None
        self._turn: Turn | None = None
        self._cancel = threading.Event()
        self._sub = None
        self._ack_audio = None
        self._ingesting: set[asyncio.Task] = set()
        self._voice = asyncio.Lock()  # one speaker: replies and reminders take turns
        self._queries_active = 0
        self._last_activity = time.monotonic()
        self._spare: LlmSession | None = None  # a session whose system prompt + tools are already prefilled
        self._warm_task: asyncio.Task | None = None
        self._extract_task: asyncio.Task | None = None
        self._episode_task: asyncio.Task | None = None
        self._extract_wake = asyncio.Event()
        self._extract_lock = asyncio.Lock()  # one extraction run at a time (the loop, or a flush)
        self.extraction_log: deque[EpisodeExtracted] = deque(maxlen=50)
        self._confirm: _PendingConfirm | None = None
        self._names: set[str] = set()
        self._name_salt = os.urandom(16)
        self.recent_tags: deque[PassiveTagged] = deque(maxlen=20)
        self.pending_count = 0  # episodes waiting for extraction
        self.reminders = None  # ReminderScheduler, wired by build_brain
        self._last_cited: list[int] = []  # facts the last answer cited: what "that's wrong" refers to
        self.answer_llm: LLMBackend | None = None  # llm.answer_model, if configured
        self.last_maintenance: dict | None = None

    # --- lifecycle -------------------------------------------------------------------

    async def load_llm(self) -> None:
        self.llm = await self._llm_worker.run(self._llm_factory)
        self.builder = PromptBuilder(self.llm.count_tokens, self.cfg.memory.prompt_token_budget)
        if self.cfg.llm.answer_model and self.answer_llm is None:  # optional larger model for recall answers
            from kin.brain.llm import LiteRTBackend

            path = self.cfg.model_path(self.cfg.llm.answer_model)
            self.answer_llm = await self._llm_worker.run(lambda: LiteRTBackend(
                path, context_tokens=self.cfg.llm.context_tokens, threads=self.cfg.runtime.threads,
                cache_dir=self.cfg.model_path("llm/cache")))

    async def start(self) -> None:
        if self.cfg.tts.ack_text and self.speaker.enabled:
            self._ack_audio = await self.speaker.precache(self.cfg.tts.ack_text)
        if not self.router.ready:
            await self.router.fit(self.memory.embed_documents, self.memory.embedder_name, self.cfg.data_path("data/cache"))
        # One subscription for everything, so query lines and AttendingEnded stay in publish order.
        self._sub = self.bus.subscribe(Event, self._on_event, name="brain")
        await self.sync_names()
        await self._update_pending()
        self.last_maintenance = (await self.memory.health()).get("last_maintenance")  # survives restarts
        if self.reminders is not None:
            self.reminders.start()
        self._episode_task = asyncio.create_task(self._episode_loop(), name="brain.episodes")
        if self.cfg.extract.enabled:
            self._extract_task = asyncio.create_task(self._extract_loop(), name="brain.extract")
            if self.pending_count:
                self._extract_wake.set()
        self._schedule_warm()

    async def stop(self) -> None:
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
        if self.reminders is not None:
            await self.reminders.stop()
        self._interrupt()
        for t in (self._extract_task, self._episode_task, self._warm_task, self._confirm.timer if self._confirm else None):
            if t is not None:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
        await self.drain()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._spare is not None:
            spare, self._spare = self._spare, None
            await self._llm_worker.run(spare.close)
        if self.llm is not None:
            await self._llm_worker.run(self.llm.close)
        if self.answer_llm is not None:
            await self._llm_worker.run(self.answer_llm.close)
        self._llm_worker.shutdown()
        if self.lookup_client is not None:
            await self._net_worker.run(self.lookup_client.close)
        self._net_worker.shutdown()
        await self.memory.close()
        self.speaker.close()

    @property
    def llm_name(self) -> str | None:
        return self.llm.name if self.llm else None

    # --- event routing -------------------------------------------------------------------

    async def _on_event(self, e: Event) -> None:
        if isinstance(e, TranscriptEvent) and e.final:
            if e.mode == "query":
                self._query_parts.append(e.text)
            elif self.state is not BrainState.SPEAKING:  # never memorise Kin's own voice
                self._background(self._ingest(e.text, speaker=e.speaker, stt_model=e.stt_model), "brain.ingest")
        elif isinstance(e, ActivationEvent):
            self._last_activity = time.monotonic()
            self._interrupt()  # barge-in: a new activation stops the current reply
            self._query_parts.clear()
        elif isinstance(e, AttendingEnded):
            text = " ".join(self._query_parts).strip()
            self._query_parts.clear()
            if text:
                self._task = asyncio.create_task(self.handle_query(text), name="brain.query")
                # A question ends the conversation before it: read it for facts right after the answer
                # (extraction waits until the question is done) instead of after 90 s of silence.
                self._background(self._close_before_question(), "brain.close_episode")
        elif isinstance(e, MemoryStored):
            if extract_names(e.text, {self.cfg.user.home_city}) - self._names:
                self._background(self.sync_names(), "brain.names")

    def _background(self, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._ingesting.add(task)
        task.add_done_callback(self._ingesting.discard)

    async def drain(self) -> None:
        """Wait for in-flight background work (ingestion, name sync)."""
        while self._ingesting:
            await asyncio.gather(*list(self._ingesting), return_exceptions=True)

    async def sync_names(self) -> None:
        """Send the gateway salted hashes of every person name in memory, so it can scrub them."""
        names: set[str] = {p.name for p in await self.memory.people()}
        names |= {a for p in await self.memory.people() for a in p.aliases}
        for text in await self.memory.texts():
            names |= extract_names(text, {self.cfg.user.home_city})
        self._names = names
        if self.lookup_client is None:
            return
        hashes = sorted(name_hash(self._name_salt, n) for n in names)
        try:
            await self._net_worker.run(self.lookup_client.set_names, self._name_salt, hashes)
        except Exception as exc:  # gateway down: lookups will fail closed anyway
            log.warning("could not send name list to gateway: %s", exc)

    # --- passive: utterances -> episodes -> background fact extraction ---------------------------

    async def _ingest(
        self, text: str, *, ts: float | None = None, speaker: str | None = None, stt_model: str | None = None
    ) -> None:
        """One final overheard line: dropped if filler, else kept as an utterance in the current episode."""
        text = text.strip()
        if not text:
            return
        if is_filler(text):
            self._publish_tag(text, Tag("chatter", 1.0, 1.0, True, "rule"), "dropped")
            return
        tag = self.router.prefilter(text)  # very short lines: chatter for the pre-filter, still kept as context
        vector = None
        if tag is None:
            vector = await self.memory.embed_document(text)
            tag = self.router.classify(text, vector)
        worth = self._worth(text, tag)
        x = self.cfg.extract
        _, closed = await self.memory.hear(  # the router's vector is kept for the transcript fallback
            text, time.time() if ts is None else ts, speaker=speaker, stt_model=stt_model, worth=worth,
            gap_s=x.episode_gap_s, max_s=x.episode_max_s, vector=vector,
        )
        if closed:
            await self._episodes_closed(closed)
        self._publish_tag(text, tag, "queued")

    def _worth(self, text: str, tag: Tag) -> int:
        """Pre-filter level of one line: 2 likely worth keeping (the router says plan / fact / idea, or the line
        holds a number or a name: "The door code is 4512." can look like chatter to the k-NN); 1 unsure;
        0 confident chatter."""
        if tag.label != "chatter" or any(c.isdigit() for c in text) or extract_names(text, {self.cfg.user.home_city}):
            return WORTH_LIKELY
        return WORTH_CHATTER if tag.accepted else WORTH_UNSURE

    def _publish_tag(self, text: str, tag: Tag, decision: str) -> None:
        event = PassiveTagged(
            text=text, label=tag.label, score=tag.score, margin=tag.margin, source=tag.source,  # type: ignore[arg-type]
            decision=decision, pending=self.pending_count,  # type: ignore[arg-type]
        )
        self.recent_tags.append(event)
        self.bus.publish(event)

    async def _episodes_closed(self, closed: list[Episode]) -> None:
        for ep in closed:
            if ep.status == "skipped":
                self._log_extraction(EpisodeExtracted(episode_id=ep.id, lines=ep.lines, outcome="skipped"))
        await self._update_pending()
        if any(ep.status == "ready" for ep in closed):
            self._extract_wake.set()

    async def _update_pending(self) -> None:
        counts = await self.memory.episode_counts()
        self.pending_count = counts.get("ready", 0) + counts.get("failed", 0)

    async def close_episodes(self, now: float | None = None, *, force: bool = False) -> list[Episode]:
        """Close episodes that went quiet (or all open ones, with force)."""
        x = self.cfg.extract
        closed = await self.memory.close_idle_episodes(time.time() if now is None else now, x.episode_gap_s, x.episode_max_s, force=force)
        if closed:
            await self._episodes_closed(closed)
        return closed

    async def _close_before_question(self) -> None:
        me = asyncio.current_task()
        others = [t for t in self._ingesting if t is not me]
        await asyncio.gather(*others, return_exceptions=True)  # the last overheard lines are stored first
        await self.close_episodes(force=True)

    async def flush(self) -> int:
        """Close the open episode and extract everything waiting now (tests, benchmarks, shutdown)."""
        await self.drain()
        await self.close_episodes(force=True)
        return await self.extract_ready(wait_idle=False)

    async def _episode_loop(self) -> None:
        """Close episodes after the silence gap even when nothing else is heard; run the nightly maintenance."""
        check_s = max(1.0, min(10.0, self.cfg.extract.episode_gap_s / 3))
        while True:
            try:
                await self.close_episodes()
                if self._maintenance_due():
                    await self.maintain()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("episode housekeeping failed")
            await asyncio.sleep(check_s)

    def _maintenance_due(self, now: datetime | None = None, plugged: bool | None = None) -> bool:
        """Once a night after `hour`, or as soon as the device is on mains power; never more often than
        `min_interval_h`; never while a question runs."""
        m = self.cfg.memory.maintenance
        if not m.enabled or self._queries_active or not self.queue.idle.is_set():
            return False
        now = now or datetime.now()
        last = (self.last_maintenance or {}).get("ts", 0.0)
        if now.timestamp() - last < m.min_interval_h * 3600:
            return False
        if plugged is None and m.when_charging:
            plugged = _on_mains_power()
        return now.hour >= m.hour or bool(m.when_charging and plugged)

    async def maintain(self) -> dict:
        """Retention purges (lines, and facts if a fact retention is set), FTS optimize, VACUUM, integrity check."""
        m = self.cfg.memory
        report = await self.memory.maintain(m.utterance_retention_days, m.fact_retention_days)
        self.last_maintenance = report
        return report

    async def _extract_loop(self) -> None:
        while True:
            await self._extract_wake.wait()
            self._extract_wake.clear()
            try:
                await self.extract_ready()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("fact extraction failed")

    async def extract_ready(self, *, wait_idle: bool = True) -> int:
        """Extract facts from every finished episode, one at a time, only while no question is running."""
        async with self._extract_lock:
            return await self._extract_all(wait_idle)

    async def _extract_all(self, wait_idle: bool) -> int:
        done = 0
        while self.llm is not None:
            ready = await self.memory.ready_episodes(time.time())
            if not ready:
                break
            if wait_idle:
                await self._wait_idle()
            outcome = await self._extract_episode(ready[0], wait_idle=wait_idle)
            if outcome is None:
                continue  # a question slipped in: let it go first
            done += 1
            await self._update_pending()
        await self._update_pending()
        return done

    async def _extract_episode(self, ep: Episode, *, wait_idle: bool) -> EpisodeExtracted | None:
        lines = await self.memory.episode_lines(ep.id)
        if not lines:
            await self.memory.mark_episode(ep.id, "skipped")
            return self._log_extraction(EpisodeExtracted(episode_id=ep.id, lines=0, outcome="skipped"))
        stale = await self.memory.discard_partial_extraction(ep.id)
        if stale:
            log.info("episode %d: an interrupted extraction left %d facts; extracting it again", ep.id, len(stale))
        people = [p.name for p in await self.memory.people()]
        t0 = time.monotonic()
        async with self.queue.slot(BACKGROUND):
            if wait_idle and (self._queries_active or self.queue.active_busy):
                return None
            try:
                # Only an episode the router thought was probably chatter gets the cheap yes/no gate first.
                # (Asked of every episode, it said no to "I lent Sam my blue umbrella.": 14 of 79 keep-lines lost.)
                if ep.worth < WORTH_LIKELY and not await self._llm_worker.run(self.extractor.worth, lines):
                    await self.memory.mark_episode(ep.id, "skipped")
                    await self._update_pending()
                    return self._log_extraction(EpisodeExtracted(
                        episode_id=ep.id, lines=len(lines), outcome="skipped", llm_calls=1,
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    ))
                # Prompt 5 tried the 3 lines before the episode as context plus merge rules (B5); it changed how E2B
                # phrased facts and lowered answer accuracy on the benchmark, so extraction sees the episode only.
                result = await self._llm_worker.run(self.extractor.extract, lines, datetime.now(), people)
            except Exception as exc:  # invalid JSON twice, or the model failed: keep the episode for later
                if not isinstance(exc, ExtractionError):
                    log.exception("extraction of episode %d failed", ep.id)
                retry = time.time() + self.cfg.extract.retry_failed_after_s
                await self.memory.mark_episode(ep.id, "failed", error=str(exc)[:300], next_try_ts=retry)
                await self._update_pending()
                return self._log_extraction(EpisodeExtracted(
                    episode_id=ep.id, lines=len(lines), outcome="failed", error=str(exc)[:300],
                    duration_ms=int((time.monotonic() - t0) * 1000), llm_calls=3,
                ))
        counts = {"ADD": 0, "UPDATE": 0, "CONTRADICTS": 0, "DUPLICATE": 0, "SKIPPED": 0}
        calls = result.llm_calls + (ep.worth < WORTH_LIKELY)  # + the yes/no gate, if it was asked
        for fact in result.facts:
            verb, used = await self._store_fact(fact, ep.id)
            counts[verb] += 1
            calls += used
        await self.memory.mark_episode(ep.id, "done")
        await self._update_pending()
        return self._log_extraction(EpisodeExtracted(
            episode_id=ep.id, lines=len(lines), outcome="done", added=counts["ADD"],
            updated=counts["UPDATE"] + counts["CONTRADICTS"], duplicates=counts["DUPLICATE"],
            duration_ms=int((time.monotonic() - t0) * 1000), llm_calls=calls,
        ))

    def _log_extraction(self, e: EpisodeExtracted) -> EpisodeExtracted:
        self.extraction_log.append(e)
        self.bus.publish(e)
        return e

    async def _store_fact(self, f: ExtractedFact, episode_id: int) -> tuple[str, int]:
        """Normalise names, then ADD it, UPDATE / CONTRADICT a close active fact, or note a DUPLICATE."""
        idx = await self.memory.name_index()
        text, fixes = idx.normalise_text(f.text)
        people, more = idx.normalise_names(f.people)
        subject = f.subject
        if subject not in (None, "owner"):
            subject = idx.canonical(subject) or subject
        if self.memory.only_about_me and subject not in (None, "", "owner"):
            return "SKIPPED", 0  # the owner chose to keep only facts about themselves
        for heard, canon in {**fixes, **more}.items():
            await self.memory.add_person(canon, [heard])  # the misheard spelling becomes an alias
        for name in {*people, *([subject] if subject not in (None, "owner") else [])}:
            await self.memory.add_person(name)
        vector = await self.memory.embed_document(text)
        if f.small_talk:  # E2B copied a line of small talk; the router agrees it is chatter: not a memory
            tag = self.router.classify(text, vector)
            if tag.label == "chatter" or (tag.label == "commitment" and tag.margin < SMALL_TALK_MARGIN):
                return "SKIPPED", 0
        x = self.cfg.extract
        close = [
            h for h in await self.memory.similar_active(vector, x.dedup_candidates, subject=subject, people=people, text=text)
            if h.similarity >= x.dedup_similarity
            or (subject and h.fact.subject == subject and f.event_ts and h.fact.event_ts and abs(h.fact.event_ts - f.event_ts) < 60)
        ]
        verb, old, calls = "ADD", None, 0
        if close:
            async with self.queue.slot(BACKGROUND):
                verb, old = await self._llm_worker.run(self.extractor.classify, text, close)
            calls = 1
        if verb == "DUPLICATE" and old is not None:
            await self.memory.add_evidence(old, f.evidence)
            return verb, calls
        fact = await self.memory.remember(
            text, f.kind, 0.8, source="extracted", ts=f.heard_ts, vector=vector, subject=subject, people=people,
            event_ts=f.event_ts, evidence=f.evidence, episode_id=episode_id,
        )
        if fact is None:
            return "DUPLICATE", calls
        if verb in ("UPDATE", "CONTRADICTS") and old is not None:
            await self.memory.supersede(old, fact.id)
        return (verb if old is not None else "ADD"), calls

    async def _wait_idle(self) -> None:
        idle_s = self.cfg.extract.idle_s
        while True:
            await self.queue.idle.wait()
            remaining = idle_s - (time.monotonic() - self._last_activity)
            if self._queries_active == 0 and self.queue.idle.is_set() and remaining <= 0:
                return
            await asyncio.sleep(max(remaining, 0.05))

    # --- warm session ----------------------------------------------------------------------

    def _schedule_warm(self) -> None:
        if not self.cfg.llm.warm_session or self.llm is None or self._spare is not None:
            return
        if self._warm_task is None or self._warm_task.done():
            self._warm_task = asyncio.create_task(self._warm(), name="brain.warm")

    async def _warm(self) -> None:
        """Prefill the fixed system prompt + tools into a spare session, at background priority.

        LiteRT-LM 0.17 cannot clone a session or share a KV prefix, and a conversation is never
        reused across questions (forgotten memories must not linger in its KV cache). So a fresh
        session is primed with one fixed exchange while Kin is idle (~6 s on the laptop).
        A warm-up is never cancelled: LiteRT-LM cannot interrupt a prefill (cancel_process only
        stops decoding, and leaves the conversation unusable), so a question that arrives meanwhile
        waits for the rest of the prefill and then uses the primed session, which is always sooner
        than prefilling a new one.
        """
        session: LlmSession | None = None
        try:
            async with self.queue.slot(BACKGROUND):
                if self._spare is not None or self.llm is None:
                    return
                session = await self._llm_worker.run(
                    self.llm.session, self.system, tools=TOOLS, max_output_tokens=self.cfg.llm.max_output_tokens
                )

                def prime() -> None:
                    for _ in session.send(WARMUP_MESSAGE):
                        pass

                await self._llm_worker.run(prime)
                self._spare, session = session, None  # before releasing the slot: a waiting question takes it
        except Exception:
            log.exception("session warm-up failed")
        finally:
            if session is not None:
                await self._llm_worker.run(session.close)

    async def _take_session(self) -> LlmSession:
        spare, self._spare = self._spare, None
        if spare is not None:
            return spare
        return await self._llm_worker.run(
            self.llm.session, self.system, tools=TOOLS, max_output_tokens=self.cfg.llm.max_output_tokens
        )

    # --- active queries ------------------------------------------------------------------

    async def handle_query(self, text: str) -> ReplyCompleted:
        self._queries_active += 1
        self._last_activity = time.monotonic()
        try:
            async with self._voice:
                return await self._handle_query(text)
        finally:
            self._queries_active -= 1
            self._last_activity = time.monotonic()

    async def _handle_query(self, text: str) -> ReplyCompleted:
        query = clean_query(text)
        reply = _Reply(uuid.uuid4().hex[:8], query)
        self.bus.publish(QueryReceived(query_id=reply.query_id, text=query))
        self._set_state(BrainState.THINKING)
        self._cancel = threading.Event()
        turn = self._turn = self.speaker.turn(on_audio=lambda: self._set_state(BrainState.SPEAKING))
        extra: dict = {}
        try:
            outcome = await self._answer_confirmation(turn, reply) if self._confirm else None
            if outcome is None:
                cmd = parse_safe_command(query)
                if cmd is not None:
                    reply.path.append(f"safety guard: {cmd.kind}")
                    outcome = await self._request_confirmation(turn, reply, cmd)
                elif self.llm is None:
                    outcome = "error"
                    await self._say(turn, reply, "I'm still waking up. Ask me again in a moment.")
                else:
                    outcome, extra = await self._agent(turn, reply)
        except asyncio.CancelledError:
            outcome = "interrupted"
        except Exception:
            log.exception("query failed")
            outcome = "error"
            await self._say(turn, reply, ERROR_REPLY)
        if turn.cancelled:
            outcome = "interrupted"
        else:
            await turn.finish()
        done = ReplyCompleted(
            query_id=reply.query_id,
            query=query,
            text=" ".join(reply.spoken),
            outcome=outcome,  # type: ignore[arg-type]
            citations=",".join(map(str, dict.fromkeys(reply.citations))),
            blocked_sentences=reply.blocked,
            time_to_first_audio_ms=int((turn.first_audio_at - reply.started) * 1000) if turn.first_audio_at else None,
            total_ms=int((time.monotonic() - reply.started) * 1000),
            path=" → ".join(reply.path) or ("answered directly" if reply.llm_calls else ""),
            llm_calls=reply.llm_calls,
            explain=self._explain(reply),
            **extra,
        )
        self.last_reply = done
        if done.citations and outcome == "answered":
            self._last_cited = [int(i) for i in done.citations.split(",") if i]
        self.bus.publish(done)
        if self._turn is turn:
            self._turn = None
            self._set_state(BrainState.IDLE)
        return done

    async def _agent(self, turn: Turn, reply: _Reply) -> tuple[str, dict]:
        """One E2B call with tools; at most two tool calls; every answer through the gate."""
        speculative = asyncio.create_task(self.memory.recall(reply.query, self.cfg.memory.top_k, lines=True))
        ack = asyncio.create_task(self._ack_later(turn, reply))
        try:
            async with self.queue.slot(ACTIVE):
                session = await self._take_session()
                try:
                    return await self._agent_loop(turn, reply, session, speculative)
                finally:
                    await self._llm_worker.run(session.close)
        finally:
            ack.cancel()
            speculative.cancel()
            self._schedule_warm()

    async def _agent_loop(self, turn: Turn, reply: _Reply, session: LlmSession, speculative: asyncio.Task) -> tuple[str, dict]:
        rec: Recall = await speculative
        reply.recall = rec
        hits = rec.hits
        extra = _closest(hits)
        # Honesty for direct answers: if memory clearly holds something relevant, a reply that skips
        # recall must cite (it cannot, with no memories offered), so it is blocked and recall runs instead.
        strong = self.gate.retrieval_ok(rec.confidence)
        allowed: set[int] = set()
        web_ok = False
        tool_calls = 0
        retried = False

        if (scope := _list_scope(reply.query, rec)) is not None:
            return await self._list_answer(turn, reply, session, rec, scope), extra | self._llm_metrics()
        if self._shortcut_ok(reply.query, rec):
            # Strong match: the facts go into the first message of the warm session, so one call answers.
            # The tools stay available; if the model still calls one, it is handled as usual below.
            reply.shortcut = True
            reply.path.append("shortcut: recall")
            prompt = await self._llm_worker.run(self.builder.recall, reply.query, self._offer(rec), None, rec.conflicts,
                                                rec.plan.fixes)
            allowed = set(prompt.memory_ids)
            if self.answer_llm is not None:
                result = await self._answer_pass(turn, reply, prompt)
            else:
                message = f"{RECALL_DONE}\n{prompt.user}\n\n{ANSWER_RULE}"
                result = await self._model_pass(turn, reply, session, message, allowed, web_ok=False)
        else:
            result = await self._model_pass(turn, reply, session, reply.query, allowed, web_ok=False, require=strong)
        while True:
            if result.timed_out:
                if not reply.spoken:
                    await self._say(turn, reply, TIMEOUT_REPLY)
                return "timeout", extra | self._llm_metrics()
            if not result.calls:
                break
            try:
                call = validate(result.calls[0])
            except InvalidCall as exc:
                log.info("invalid tool call %r: %s", result.calls[0].name, exc)
                if not retried:
                    retried = True
                    reply.path.append("retry")
                    result = await self._model_pass(turn, reply, session, correction_message(str(exc)), allowed, web_ok=web_ok)
                    continue
                reply.path.append("fallback: recall")
                return await self._fallback_recall(turn, reply, rec), extra | self._llm_metrics()
            if tool_calls >= MAX_TOOL_CALLS:
                log.info("tool call limit reached; ignoring %s", call.name)
                break
            tool_calls += 1
            reply.path.append(call.label)
            done = await self._execute(turn, reply, call, rec)
            if done.outcome is not None:
                return done.outcome, extra | self._llm_metrics()
            allowed |= done.allowed
            web_ok = web_ok or done.web_ok
            if self.answer_llm is not None and done.prompt is not None:  # the larger model writes the recall answer
                result = await self._answer_pass(turn, reply, done.prompt)
                continue
            result = await self._model_pass(
                turn, reply, session, ToolResult(call.name, done.content or ""), allowed, web_ok=web_ok, require=True
            )

        if not reply.spoken:
            if tool_calls == 0 and reply.blocked and strong and not reply.shortcut:
                reply.path.append("fallback: recall")  # answered about the user without recalling
                return await self._fallback_recall(turn, reply, rec), extra | self._llm_metrics()
            await self._say(turn, reply, BLOCKED_REPLY)
            return "abstained", extra | self._llm_metrics()
        if tool_calls == 0 and not reply.shortcut:
            return ("abstained" if reply.abstained else "answered" if reply.answered else "chit_chat"), extra | self._llm_metrics()
        return ("answered" if reply.answered else "abstained"), extra | self._llm_metrics()

    async def _answer_pass(self, turn: Turn, reply: _Reply, prompt: BuiltPrompt) -> _Pass:
        """A recall answer from llm.answer_model, in a tool-less session (its result never calls tools)."""
        reply.llm_calls += 1
        reply.path.append("answer: " + self.answer_llm.name)
        session = await self._llm_worker.run(self.answer_llm.session, prompt.system,
                                             max_output_tokens=self.cfg.llm.max_output_tokens)
        try:
            return await self._run_pass(turn, reply, session, f"{prompt.user}\n\n{ANSWER_RULE}", set(prompt.memory_ids),
                                        web_ok=False, require=True)
        finally:
            await self._llm_worker.run(session.close)

    async def _model_pass(
        self, turn: Turn, reply: _Reply, session: LlmSession, message: str | ToolResult, allowed: set[int],
        *, web_ok: bool, require: bool = True,
    ) -> _Pass:
        reply.llm_calls += 1
        return await self._run_pass(turn, reply, session, message, allowed, web_ok=web_ok, require=require, capture_calls=True)

    def _offer(self, rec: Recall) -> list[Hit]:
        """The facts the answer prompt sees: the best `answer_facts` (a 2B model picks the right one far more often
        from 3 than from 5: Prompt 5 oracle), plus any fact pinned for a person the question names."""
        n = self.cfg.memory.answer_facts
        return rec.hits[:n] + [h for h in rec.hits[n:] if h.detail.get("pinned")]

    def _shortcut_ok(self, query: str, rec: Recall) -> bool:
        conf = rec.confidence
        if conf is None or conf < self.cfg.memory.retrieval.strong_score:
            return False
        q = query.strip()
        return not _COMMAND.search(q) and not _ONLINE.search(q) and bool(_QUESTION.search(q) or q.endswith("?"))

    async def _list_answer(self, turn: Turn, reply: _Reply, session: LlmSession, rec: Recall, scope: str) -> str:
        """"What do I know about Rahul?", "what did I plan yesterday?": the matching active facts, listed
        deterministically (pinned first, newest first), summarised by the model in up to three cited sentences.
        If the summary yields nothing citable, Kin reads the list."""
        facts = await self.memory.scope_facts(rec.plan, self.cfg.memory.list_facts)
        reply.path.append(f"list: {scope}")
        if not facts:
            await self._say(turn, reply, ABSTAIN_REPLY)
            return "abstained"
        reply.shortcut = True
        hits = [Hit(f, 1.0, score=1.0, detail={"id": f.id, "text": f.text, "status": f.status, "listed": True}) for f in facts]
        prompt = await self._llm_worker.run(self.builder.recall, reply.query, hits, None, rec.conflicts, rec.plan.fixes)
        message = f"{RECALL_DONE}\n{prompt.user}\n\n{LIST_RULE}"
        result = await self._model_pass(turn, reply, session, message, set(prompt.memory_ids), web_ok=False)
        if not result.timed_out and not reply.answered:
            reply.path.append("read list")
            for f in facts[:MAX_ANSWER_SENTENCES]:
                await self._say(turn, reply, _you(f.text), [f.id])
            reply.answered = True
        return "answered" if reply.answered else "abstained"

    async def _fallback_recall(self, turn: Turn, reply: _Reply, rec: Recall) -> str:
        """Plain recall without tools (a fresh, tool-less session): used when tool calling fails."""
        if not self.gate.retrieval_ok(rec.confidence):
            return await self._abstain_or_heard(turn, reply, rec)
        prompt = await self._llm_worker.run(self.builder.recall, reply.query, self._offer(rec), None, rec.conflicts,
                                            rec.plan.fixes)
        reply.llm_calls += 1
        result = await self._single_pass(turn, reply, prompt, require_citations=True)
        if result.timed_out:
            if not reply.spoken:
                await self._say(turn, reply, TIMEOUT_REPLY)
            return "timeout"
        if not reply.spoken:
            await self._say(turn, reply, BLOCKED_REPLY)
            return "abstained"
        return "answered" if reply.answered else "abstained"

    # --- tools ------------------------------------------------------------------------------

    async def _execute(self, turn: Turn, reply: _Reply, call: ValidCall, rec: Recall) -> _ToolOutcome:
        a = call.args
        if call.name == "recall":
            return await self._tool_recall(turn, reply, a["question"], rec)
        if call.name == "lookup":
            return await self._tool_lookup(turn, reply, a, rec.hits)
        if call.name == "cancel_reminder":
            return _ToolOutcome(outcome=await self._request_confirmation(turn, reply, SafeCommand("cancel_reminder", text=a["text"])))
        if call.name in ("correct_fact", "retract_fact"):
            msg = await self._tool_correct(call.name, a.get("correction"), a.get("which"), reply)
            await self._say(turn, reply, msg)
            return _ToolOutcome(outcome="command")
        if call.name == "review_memory":
            msg = await self._tool_review(a.get("day"))
            await self._say(turn, reply, msg)
            return _ToolOutcome(outcome="command")
        if call.name == "remember":
            msg = await self._tool_remember(a["text"])
        else:
            msg = await self._task_tool(call)
        await self._say(turn, reply, msg)
        return _ToolOutcome(outcome="command")

    async def _tool_recall(self, turn: Turn, reply: _Reply, question: str, speculative: Recall) -> _ToolOutcome:
        """The model's rewording is a second query, fused (RRF) with the user's own words."""
        rec = speculative
        if question.strip().lower() != reply.query.strip().lower():
            rec = await self.memory.recall(reply.query, self.cfg.memory.top_k, alt=question, lines=True)
        reply.recall = rec
        if not self.gate.retrieval_ok(rec.confidence):
            return _ToolOutcome(outcome=await self._abstain_or_heard(turn, reply, rec))
        prompt = await self._llm_worker.run(self.builder.recall, reply.query, self._offer(rec), None, rec.conflicts,
                                            rec.plan.fixes)
        return _ToolOutcome(content=f"{prompt.user}\n\n{ANSWER_RULE}", allowed=set(prompt.memory_ids), prompt=prompt)

    async def _abstain_or_heard(self, turn: Turn, reply: _Reply, rec: Recall) -> str:
        """No fact is close enough. If a raw line Kin heard is, quote it word for word, saying when it was
        heard and that Kin is not sure; otherwise abstain."""
        floor = self.cfg.memory.retrieval.utterance_floor
        best = next((x for x in rec.lines if x.score >= floor), None)
        if best is None:
            await self._say(turn, reply, ABSTAIN_REPLY)
            return "abstained"
        reply.path.append("fallback: transcript")
        reply.heard.append(best.line.id)
        when = heard_when(best.line.ts, datetime.now())
        await self._say(turn, reply, f"I'm not sure, but {when} I heard: “{best.line.text.strip()}”")
        return "answered"

    async def _tool_lookup(self, turn: Turn, reply: _Reply, args: dict, hits: list[Hit]) -> _ToolOutcome:
        kind, params = lk.to_request(args, home_city=self.cfg.user.home_city, names=self._names)
        context = [h for h in hits if h.similarity >= LOOKUP_CONTEXT_FLOOR]
        memories = "\n".join(format_memory(h) for h in context) or "none relevant"
        now = datetime.now().strftime("%A %d %B %Y, %H:%M")
        result = await self._call_gateway(kind, params)
        if result.ok and result.data is not None:
            reply.looked_up.append(kind)
            await self._say(turn, reply, lk.CHECKED[kind])
            facts, web_ok = lk.render(kind, result.data), True
            reply.lookup_facts += facts
        else:
            # Nothing came back, so there is nothing for the model to reason about: asked to "just remind
            # the user of the plan", it still made weather decisions up. Say what failed, then the most
            # relevant memory word for word (cited), and stop.
            await self._say(turn, reply, lk.failure_sentence(kind, result.reason))
            best = context[0] if context else None  # quoted verbatim, so the lookup-context floor is enough
            if best is None:
                return _ToolOutcome(outcome="abstained")
            reply.answered = True
            await self._say(turn, reply, f"Here's what I remember: {best.fact.text.rstrip('.')}.", [best.fact.id])
            return _ToolOutcome(outcome="answered")
        content = f"{facts}\nNow: {now}\nThe user's memories:\n{memories}\nQuestion: {reply.query}"
        return _ToolOutcome(content=content, allowed={h.fact.id for h in context}, web_ok=web_ok)

    async def _correction_target(self, about: str | None) -> Fact | None:
        """The memory a correction is about: one cited in the last answer (the one closest to `about` if it
        cited several); with no recent answer, a strong recall match for `about`."""
        from rapidfuzz import fuzz

        facts = [f for f in [await self.memory.get(i) for i in self._last_cited] if f is not None and f.status == "active"]
        if facts:
            if len(facts) == 1 or not about:
                return facts[0]
            return max(facts, key=lambda f: fuzz.token_set_ratio(about.lower(), f.text.lower()))
        if about:
            rec = await self.memory.recall(about, 3)
            if rec.hits and (rec.confidence or 0) >= self.cfg.memory.retrieval.strong_score:
                return rec.hits[0].fact
        return None

    async def _tool_correct(self, name: str, correction: str | None, which: str | None, reply: _Reply) -> str:
        """Voice corrections act on what Kin just said. Nothing is deleted: the old version is superseded
        (corrected) or retracted, logged in the fact history, and can be restored from the Memory tab."""
        reply.path.append("correction")
        target = await self._correction_target(which or (correction if name == "correct_fact" else None))
        if target is None:
            return "Which memory do you mean? Ask me about it first, then tell me what's wrong."
        if name == "retract_fact":
            await self.memory.retract(target.id, actor="voice")
            self._last_cited = [i for i in self._last_cited if i != target.id]
            return f"Okay, I won't use that any more: {target.text.rstrip('.')}."
        from kin.brain.corrections import corrected_text

        new_text = corrected_text(target.text, correction or "")
        if new_text is None:
            return "Tell me the whole corrected sentence, for example: the dinner is at 8:30."
        new = await self.memory.correct(target.id, new_text, actor="voice")
        if new is None:
            return "I couldn't change that memory."
        self._last_cited = [new.id if i == target.id else i for i in self._last_cited]
        return f"Got it, I've corrected it: {new.text.rstrip('.')}."

    async def _tool_review(self, day: str | None) -> str:
        when = datetime.now() - (timedelta(days=1) if day == "yesterday" else timedelta())
        r = await self.memory.review(when)
        label = "yesterday" if day == "yesterday" else "today"
        added = [f for f in r["added"] if f["status"] == "active"]
        if not added and not r["changes"]:
            return f"I didn't remember anything new {label}."
        parts = [f"{len(added)} new {_plural(len(added), 'memory', 'memories')} {label}"]
        if r["changes"]:
            parts.append(f"{len(r['changes'])} {_plural(len(r['changes']), 'change', 'changes')}")
        lead = "I have " + " and ".join(parts) + "."
        shown = "; ".join(f["text"].rstrip(".") for f in added[:3])
        more = " The Memory tab lists the rest." if len(added) > 3 else ""
        return f"{lead} {shown}.{more}" if shown else lead

    async def _tool_remember(self, text: str) -> str:
        """Explicit "remember that ...": stored at once (no extraction, no LLM), said by the owner."""
        idx = await self.memory.name_index()
        sentence, _ = idx.normalise_text(_sentence(text.strip().rstrip(".")) + ".")
        vector = await self.memory.embed_document(sentence)
        kind = fact_kind(self.router.classify(sentence, vector).label)
        people = sorted(extract_names(sentence, {self.cfg.user.home_city}))
        first = sentence.split()[0].rstrip("'s,") if sentence.split() else ""
        subject = first if first in people else "owner"
        for name in people:
            await self.memory.add_person(name)
        mem = await self.memory.remember(sentence, kind, 1.0, source="explicit", vector=vector, subject=subject, people=people)
        return f"Okay, I'll remember that. That's memory {mem.id}." if mem else "I already have that."

    async def _task_tool(self, call: ValidCall) -> str:
        if self.tasks is None:
            return "Tasks and reminders aren't available right now."
        now = datetime.now()
        a = call.args
        if call.name == "add_task":
            when = parse_due(a.get("due"), now)
            inline = parse_when(a["text"], now)  # "email the professor at 5" with due=null
            what = _clean_task(inline.rest if inline.due is not None else a["text"])
            if when.due is None:
                when = inline
            if not what:
                return "What should I remind you about?"
            if when.due is None:
                if a.get("due"):
                    await self.tasks.add(_sentence(what))
                    return f"I added {what} to your tasks. I didn't catch a time, so I won't remind you about it."
                task = await self.tasks.add(_sentence(what))
                return f"Added to your list: {task.text.rstrip('.')}."
            await self.tasks.add(_sentence(what), when.due.timestamp())
            return f"Okay, I'll remind you to {what} {describe(when.due, now, relative=when.relative)}."
        if call.name == "list_tasks":
            open_tasks = await self.tasks.open()
            if not open_tasks:
                return "Your list is empty, and there are no reminders."
            parts = []
            for t in open_tasks[:5]:
                due = f", {describe(datetime.fromtimestamp(t.due_ts), now)}" if t.due_ts else ""
                parts.append(f"{t.text.rstrip('.')}{due}")
            more = f" and {len(open_tasks) - 5} more" if len(open_tasks) > 5 else ""
            return f"You have {len(open_tasks)} {_plural(len(open_tasks), 'item', 'items')}: " + "; ".join(parts) + more + "."
        # complete_task: only a task that is really on the list ("I finished the book" is not one)
        task = await self.tasks.find(a["text"])
        if task is None:
            return "That isn't on your list, so I didn't mark anything as done."
        await self.tasks.complete(task.id)
        return f"Nice. I ticked off {task.text.rstrip('.')}."

    async def _call_gateway(self, kind: str, params: dict) -> LookupResult:
        if self.lookup_client is None or not self.cfg.gateway.enabled:
            return LookupResult(False, "refused", "gateway_unavailable")
        self._set_state(BrainState.LOOKUP)
        try:
            result = await self._net_worker.run(self.lookup_client.lookup, kind, params)
        finally:
            self._set_state(BrainState.THINKING)  # the turn flips it to SPEAKING whenever audio plays
        a = result.audit or {}
        self.bus.publish(
            LookupPerformed(
                request_id=str(a.get("id", "")),
                kind=a.get("kind", kind),
                request=a.get("request"),
                upstream="\n".join(a.get("upstream") or []),
                status=result.status,
                reason=result.reason,
                provider=a.get("provider"),
                response_bytes=int(a.get("response_bytes") or 0),
                reply_bytes=int(a.get("reply_bytes") or 0),
                duration_ms=int(a.get("duration_ms") or 0),
                scrubbed=int(a.get("scrubbed") or 0),
            )
        )
        return result

    # --- destructive commands: always confirmed ------------------------------------------------

    async def _request_confirmation(self, turn: Turn, reply: _Reply, cmd: SafeCommand) -> str:
        self._clear_confirmation("cancelled", "replaced by a new request")
        ids: list[int] = []
        task_id = None
        if cmd.kind == "forget_span":
            minutes = cmd.minutes or 10
            ids = await self.memory.ids_since(minutes)
            span = _describe_minutes(minutes)
            if not ids:
                await self.memory.forget_ids([], pending_since_minutes=minutes, reason="forget_command")
                await self._say(turn, reply, f"There's nothing from the last {span} to forget.")
                return "command"
            what = f"{len(ids)} {_plural(len(ids), 'memory', 'memories')} from the last {span}"
            prompt = f"That's {what}. Say confirm to delete."
        elif cmd.kind == "forget_last":
            latest = await self.memory.recent(limit=1)
            if not latest:
                await self._say(turn, reply, "There's nothing to forget.")
                return "command"
            ids = [latest[0].id]
            what = "your last memory"
            prompt = f"That's your last memory: {latest[0].text.rstrip('.')}. Say confirm to delete."
        elif cmd.kind == "forget_person":
            spellings, ids, lines = await self.memory.about(cmd.text or "")
            who = spellings[0]
            if not ids and not lines:
                await self._say(turn, reply, f"I don't have anything about {who}.")
                return "command"
            what = f"{len(ids)} {_plural(len(ids), 'memory', 'memories')} and {len(lines)} heard {_plural(len(lines), 'line', 'lines')} about {who}"
            prompt = f"That's {what}. Say confirm to delete them."
        else:
            task = await self.tasks.find(cmd.text or "") if self.tasks else None
            if task is None or task.kind != "reminder":
                await self._say(turn, reply, "I couldn't find that reminder, so I didn't cancel anything.")
                return "command"
            task_id = task.id
            what = f"the reminder to {task.text.rstrip('.').lower()}"
            prompt = f"Cancel {what}? Say confirm."
        pc = _PendingConfirm(uuid.uuid4().hex[:8], cmd, ids, task_id, what, time.time() + CONFIRM_WINDOW_S)
        pc.timer = asyncio.create_task(self._expire_confirmation(pc), name="brain.confirm")
        self._confirm = pc
        self.bus.publish(
            ConfirmationRequested(confirm_id=pc.confirm_id, action=cmd.kind, summary=prompt, count=len(ids) or 1, expires_ts=pc.expires)
        )
        await self._say(turn, reply, prompt)
        return "command"

    async def _answer_confirmation(self, turn: Turn, reply: _Reply) -> str | None:
        """The next query after a confirmation request: "confirm" runs it, "no" cancels it,
        anything else cancels it and is handled as a normal query (returns None)."""
        pc = self._confirm
        if pc is None or time.time() > pc.expires:
            return None
        answer = confirmation_answer(reply.query)
        if answer is None:
            self._clear_confirmation("cancelled", "another request")
            return None
        reply.path.append("confirmation")
        msg = await self._resolve(pc, answer)
        await self._say(turn, reply, msg)
        return "command"

    async def resolve_confirmation(self, confirm_id: str, accept: bool) -> bool:
        """From the UI's Confirm / Cancel buttons. Speaks the result like any other reply."""
        pc = self._confirm
        if pc is None or pc.confirm_id != confirm_id or time.time() > pc.expires:
            return False
        msg = await self._resolve(pc, accept)
        await self.announce(msg)
        return True

    async def _resolve(self, pc: _PendingConfirm, accept: bool) -> str:
        self._confirm = None
        if pc.timer is not None:
            pc.timer.cancel()
        if not accept:
            self.bus.publish(ConfirmationResolved(confirm_id=pc.confirm_id, outcome="cancelled"))
            return f"Okay, I kept {pc.what}." if pc.cmd.kind != "cancel_reminder" else "Okay, the reminder stays."
        if pc.cmd.kind == "cancel_reminder":
            await self.tasks.delete(pc.task_id)
            msg = f"Okay, I cancelled {pc.what}."
        elif pc.cmd.kind == "forget_person":
            deleted, lines = await self.memory.forget_person(pc.cmd.text or "")
            self._last_cited = []
            msg = f"Done. I forgot {len(deleted)} {_plural(len(deleted), 'memory', 'memories')} and {lines} heard {_plural(lines, 'line', 'lines')}."
        else:
            minutes = pc.cmd.minutes if pc.cmd.kind == "forget_span" else None
            deleted = await self.memory.forget_ids(pc.ids, pending_since_minutes=minutes, reason="forget_command")
            msg = (f"Done. I forgot {len(deleted)} {_plural(len(deleted), 'memory', 'memories')} from the last "
                   f"{_describe_minutes(minutes)}." if minutes else "Done, I forgot that.")
        self.bus.publish(ConfirmationResolved(confirm_id=pc.confirm_id, outcome="confirmed", detail=msg))
        return msg

    async def _expire_confirmation(self, pc: _PendingConfirm) -> None:
        await asyncio.sleep(max(0.0, pc.expires - time.time()))
        if self._confirm is pc:
            self._confirm = None
            self.bus.publish(ConfirmationResolved(confirm_id=pc.confirm_id, outcome="expired", detail="nothing was deleted"))

    def _clear_confirmation(self, outcome: str, detail: str) -> None:
        pc, self._confirm = self._confirm, None
        if pc is not None:
            if pc.timer is not None:
                pc.timer.cancel()
            self.bus.publish(ConfirmationResolved(confirm_id=pc.confirm_id, outcome=outcome, detail=detail))  # type: ignore[arg-type]

    @property
    def pending_confirmation(self) -> dict | None:
        pc = self._confirm
        if pc is None or time.time() > pc.expires:
            return None
        return {"confirm_id": pc.confirm_id, "action": pc.cmd.kind, "what": pc.what, "expires_ts": pc.expires}

    # --- speaking unprompted -------------------------------------------------------------------

    async def announce(self, text: str) -> None:
        """Say something unprompted (a reminder), after any reply in progress."""
        async with self._voice:
            turn = self._turn = self.speaker.turn(on_audio=lambda: self._set_state(BrainState.SPEAKING))
            try:
                await turn.say(text)
                await turn.finish()
            finally:
                if self._turn is turn:
                    self._turn = None
                    self._set_state(BrainState.IDLE)

    # --- streaming ---------------------------------------------------------------------------

    async def _single_pass(self, turn: Turn, reply: _Reply, prompt: BuiltPrompt, *, require_citations: bool) -> _Pass:
        session = await self._llm_worker.run(
            self.llm.session, prompt.system, max_output_tokens=self.cfg.llm.max_output_tokens
        )
        try:
            return await self._run_pass(
                turn, reply, session, prompt.user, set(prompt.memory_ids), web_ok=False, require=require_citations
            )
        finally:
            await self._llm_worker.run(session.close)

    async def _run_pass(
        self,
        turn: Turn,
        reply: _Reply,
        session: LlmSession,
        message: str | ToolResult,
        allowed: set[int],
        *,
        web_ok: bool,
        require: bool = True,
        capture_calls: bool = False,
    ) -> _Pass:
        """Stream one model turn through splitter -> gate -> TTS, with the generation timeout."""
        result = _Pass()
        splitter = SentenceSplitter()

        async def consume() -> None:
            async for chunk in self._llm_worker.stream(session.send, message, cancel=self._cancel):
                if chunk.tool_call is not None:
                    if capture_calls:
                        result.calls.append(chunk.tool_call)
                    continue
                for sentence in splitter.feed(chunk.text):
                    await self._emit_sentence(turn, reply, sentence, allowed, require, web_ok)
            rest = splitter.flush()
            if rest:
                await self._emit_sentence(turn, reply, rest, allowed, require, web_ok)

        task = asyncio.create_task(consume())
        done, _ = await asyncio.wait({task}, timeout=self.cfg.llm.timeout_s)
        if not done:
            # Signal the model first: the stream only finishes once the worker stops generating.
            self._cancel.set()
            task.cancel()
            result.timed_out = True
        try:
            await task
        except asyncio.CancelledError:
            if not result.timed_out:
                raise
        if result.timed_out:
            self._cancel = threading.Event()  # a fresh flag for anything after the timeout
        return result

    async def _emit_sentence(
        self, turn: Turn, reply: _Reply, sentence: str, allowed: set[int], require: bool, web_ok: bool = False
    ) -> None:
        if turn.cancelled:
            return
        if require and _PROVENANCE.search(strip_citations(sentence)):
            log.info("dropped a provenance-only sentence: %r", sentence)
            return
        if require:
            verdict = self.gate.check(sentence, allowed, web_ok=web_ok)
            if not verdict.speakable:
                reply.blocked += 1
                log.info("gate blocked sentence (%s)", verdict.reason)
                return
            if verdict.reason == "cited" and (who := self._wrong_person(reply, verdict.spoken, verdict.citations)):
                reply.blocked += 1
                log.info("gate blocked sentence (it is about %s, the memories it cites are about someone else)", who)
                return
            if verdict.reason == "cited" and not verdict.web and (
                    n := self._ungrounded_number(reply, verdict.spoken, verdict.citations)):  # a lookup brings its own numbers
                reply.blocked += 1
                log.info("gate blocked sentence (%s is in none of the memories it cites)", n)
                return
            if verdict.reason == "abstention" and reply.answered:
                # "Your manager is Ajay [#30]. I don't remember who your manager is on Friday.": once the question
                # is answered from memory, a trailing "I don't remember ..." only confuses.
                log.info("dropped an abstention after a cited answer: %r", verdict.spoken)
                return
            if verdict.reason == "cited":
                if reply.answer_sentences >= MAX_ANSWER_SENTENCES:
                    log.info("answer cut after %d sentences", MAX_ANSWER_SENTENCES)
                    return
                reply.answer_sentences += 1
            text, cites = verdict.spoken, list(verdict.citations)
            reply.answered |= verdict.reason == "cited"
            reply.abstained |= verdict.reason == "abstention"
        else:
            text, cites = strip_citations(sentence), []
        await self._say(turn, reply, text, cites)

    @staticmethod
    def _wrong_person(reply: _Reply, text: str, cites: tuple[int, ...]) -> str | None:
        """Attribution check: a sentence naming a person from the question ("Sam's birthday is on the 22nd [#40]")
        whose cited memories name only other known people (#40: "Rahul's birthday ...") is not spoken. A memory
        naming nobody ("Someone said: my new flat is in Dharampeth") does not block: the speaker was unknown."""
        rec = reply.recall
        if rec is None or not rec.plan.people or not cites:
            return None
        facts = {h.fact.id: h.fact for h in rec.hits}
        cited = [facts[i] for i in cites if i in facts]
        if not cited:
            return None
        for person in rec.plan.people:
            mine = rec.spellings.get(person, (person,))
            if not any(re.search(r"(?<![A-Za-z])" + re.escape(s) + r"(?![A-Za-z])", text, re.I) for s in mine):
                continue
            if any(mentions(f, (person,), {person: mine}) for f in cited):
                continue
            others = [p for p in rec.spellings if p != person]
            if others and all(mentions(f, others, rec.spellings) for f in cited):
                return person
        return None

    @staticmethod
    def _ungrounded_number(reply: _Reply, text: str, cites: tuple[int, ...]) -> str | None:
        """Numbers check: every number in a memory answer must come from a cited memory (its text, when it
        happens, when it was heard; 17:00 may be said as 5), this year, or the question. Stops "Your yoga class
        is on Monday at 09:00" citing "Classes start on the 12th of January" (09:00 was the time now)."""
        rec = reply.recall
        if rec is None:
            return None
        facts = {h.fact.id: h.fact for h in rec.hits}
        cited = [facts[i] for i in cites if i in facts]
        if not cited:
            return None
        # A sentence citing a memory may carry the forecast's numbers ("rain at 80% around 4 pm [#7]").
        known: set[int] = {datetime.now().year} | _numbers(reply.query) | _numbers(reply.lookup_facts)
        for f in cited:
            known |= _numbers(f.text)
            # Months are said as words, so their numbers are not taken: September (9) let "09:00" through.
            if f.event_ts:  # when it happens: the day and the clock time
                t = datetime.fromtimestamp(f.event_ts)
                known |= {t.year, t.day, t.hour, t.hour % 12 or 12, t.minute}
            h = datetime.fromtimestamp(f.heard_ts)  # when it was heard: the day only ("on 28 September you said")
            known |= {h.year, h.day}
        for n in _numbers(text):
            if n not in known:
                return str(n)
        return None

    async def _say(self, turn: Turn, reply: _Reply, text: str, citations: list[int] | None = None) -> None:
        if turn.cancelled or not text:
            return
        index = len(reply.spoken)
        reply.spoken.append(text)
        reply.citations += citations or []
        self.bus.publish(
            ReplySentence(query_id=reply.query_id, index=index, text=text, citations=",".join(map(str, citations or [])))
        )
        await turn.say(text)

    async def _ack_later(self, turn: Turn, reply: _Reply) -> None:
        await asyncio.sleep(self.cfg.tts.ack_delay_s)
        if not reply.spoken and not turn.cancelled and self._ack_audio is not None:
            turn.play_cached(self._ack_audio, "ack")

    def _explain(self, reply: _Reply) -> str:
        """The "Why this answer?" record: the plan, the facts offered and their scores, what was filtered and why."""
        rec = reply.recall
        if rec is None:
            return ""
        r = self.cfg.memory.retrieval
        cited = set(reply.citations)
        d = rec.explain()
        for u in d["used"]:
            u["cited"] = u["id"] in cited
        d["decision"] = {
            "confidence": None if rec.confidence is None else round(rec.confidence, 4),
            "answer_floor": r.answer_floor, "strong_score": r.strong_score, "utterance_floor": r.utterance_floor,
            "shortcut": reply.shortcut, "heard_lines": reply.heard, "confidence_mode": r.confidence,
            "reranker": self.memory.reranker is not None,
        }
        return json.dumps(d, ensure_ascii=False)

    def _llm_metrics(self) -> dict:
        s = self.llm.last_stats if self.llm else None
        return {
            "backend": self.llm_name,
            "prompt_tokens": s.prompt_tokens if s else None,
            "output_tokens": s.output_tokens if s else None,
            "tokens_per_s": s.tokens_per_s if s else None,
            "llm_ttft_ms": int(s.ttft_s * 1000) if s and s.ttft_s is not None else None,
        }

    # --- state -------------------------------------------------------------------------

    def _interrupt(self) -> None:
        self._cancel.set()
        if self._turn is not None:
            self._turn.cancel()
            self._turn = None
            self._set_state(BrainState.IDLE)

    def _set_state(self, state: BrainState) -> None:
        if state is not self.state:
            self.state = state
            self.bus.publish(BrainStateChanged(state=state))


def _closest(hits: list[Hit]) -> dict:
    best = hits[0] if hits else None
    return {
        "closest_id": best.fact.id if best else None,
        "closest_text": best.fact.text if best else None,
        "closest_similarity": best.similarity if best else None,
    }


def _numbers(text: str) -> set[int]:
    """Whole numbers in a sentence: "1,600" -> 1600, "8:30" -> 8 and 30, "22nd" -> 22, "98220 41377" -> both."""
    return {int(x) for x in re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d{3}\b)", "", text))}


_LIST_PERSON = re.compile(r"^(?:what|anything)\s+(?:do|did)\s+(?:i|you)\s+know\s+about\b|^tell\s+me\s+(?:everything\s+)?about\b"
                          r"|^what\s+have\s+i\s+(?:said|told you|heard)\s+about\b", re.I)
_LIST_TIME = re.compile(r"^what(?:'s| is| are)?\s+(?:my\s+)?(?:plans?|on|happening|coming\s+up)\b"
                        r"|^what\s+did\s+i\s+(?:plan|do|decide)\b|^what\s+have\s+i\s+got\b|^what\s+do\s+i\s+have\b", re.I)


def _list_scope(query: str, rec: Recall) -> str | None:
    """A question that asks for everything about a person or a time span (not one detail)."""
    q = query.strip()
    if rec.plan.people and _LIST_PERSON.search(q):
        return "person"
    if rec.plan.window and _LIST_TIME.search(q):
        return "time"
    return None


def _you(text: str) -> str:
    """A stored fact read back to its owner: "The user's code ..." -> "Your code ..."."""
    t = re.sub(r"\bthe user's\b", "your", text, flags=re.I)
    t = re.sub(r"\bthe user\b", "you", t, flags=re.I)
    return t[:1].upper() + t[1:]


def _on_mains_power() -> bool:
    """True on mains power, or with no battery to ask (a Pi on its supply, a desktop)."""
    try:
        import psutil

        b = psutil.sensors_battery()
    except Exception:
        return True
    return True if b is None else bool(b.power_plugged)


def heard_when(ts: float, now: datetime) -> str:
    """'today at 10:05', 'yesterday at 18:30', 'on Saturday at 10:05', 'on 28 September at 09:00'."""
    t = datetime.fromtimestamp(ts)
    days = (now.date() - t.date()).days
    clock = t.strftime("%H:%M")
    if days <= 0:
        return f"today at {clock}"
    if days == 1:
        return f"yesterday at {clock}"
    if days < 7:
        return f"on {t.strftime('%A')} at {clock}"
    return f"on {t.day} {t.strftime('%B')} at {clock}"


def _clean_task(text: str) -> str:
    t = text.strip().strip(",.").strip()
    for prefix in ("to ", "that i need to ", "that i have to ", "about ", "i need to ", "i have to ", "remind me to "):
        if t.lower().startswith(prefix):
            t = t[len(prefix) :]
    return t.strip()


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _describe_minutes(minutes: float) -> str:
    if minutes >= 60 and minutes % 60 == 0:
        h = int(minutes // 60)
        return "hour" if h == 1 else f"{h} hours"
    if minutes < 1:
        s = int(round(minutes * 60))
        return f"{s} seconds"
    m = int(minutes) if minutes == int(minutes) else round(minutes, 1)
    return "minute" if m == 1 else f"{m} minutes"
