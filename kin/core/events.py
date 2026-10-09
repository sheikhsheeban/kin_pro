"""Typed events that travel on the bus.

Privacy rule: events carry derived scalars and text only. Raw audio samples
never go on the bus (see tests/test_privacy.py::test_events_carry_no_audio).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class PipelineState(str, Enum):
    """What Kin is doing right now. Drives the LED colour (with the stream state)."""

    IDLE = "idle"  # mic stream closed (muted or not started)
    LISTENING = "listening"  # passive capture
    ATTENDING = "attending"  # activated, capturing a query
    THINKING = "thinking"  # retrieving and reasoning
    SPEAKING = "speaking"  # Kin is talking; the mic is ducked so Kin never transcribes itself
    LOOKUP = "lookup"  # online factual lookup in progress (Phase 3)


class BrainState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    LOOKUP = "lookup"  # waiting on the gateway (LED blue)
    SPEAKING = "speaking"


MemoryKind = Literal["commitment", "fact_about_person", "idea", "note"]
ReplyOutcome = Literal["answered", "abstained", "timeout", "interrupted", "error", "command", "chit_chat"]


class LedColor(str, Enum):
    OFF = "off"
    GREEN = "green"  # listening
    WHITE_PULSE = "white_pulse"  # attending
    AMBER = "amber"  # thinking
    BLUE = "blue"  # online lookup


ActivationSource = Literal["wake_word", "wake_phrase", "button"]  # wake_phrase: the phrase found in a transcribed line
TranscriptMode = Literal["passive", "query"]


@dataclass(frozen=True, kw_only=True)
class Event:
    ts: float = field(default_factory=time.time)

    @property
    def type(self) -> str:
        return type(self).__name__

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        d["type"] = self.type
        return d


# --- hardware -------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MuteChanged(Event):
    muted: bool


@dataclass(frozen=True, kw_only=True)
class ButtonChanged(Event):
    pressed: bool


@dataclass(frozen=True, kw_only=True)
class OfflineChanged(Event):
    offline: bool


@dataclass(frozen=True, kw_only=True)
class LedChanged(Event):
    color: LedColor


# --- audio ----------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class StreamStateChanged(Event):
    """Emitted after the real input stream was opened or closed."""

    open: bool
    device: str | None = None
    sample_rate: int | None = None
    error: str | None = None


@dataclass(frozen=True, kw_only=True)
class AudioLevel(Event):
    """Mic level for the meter. A single loudness number, never samples."""

    rms_db: float


@dataclass(frozen=True, kw_only=True)
class WakeLevel(Event):
    """The wake word model's highest score over the last ~67 ms (sent with AudioLevel, 15 Hz)."""

    score: float
    threshold: float


@dataclass(frozen=True, kw_only=True)
class SpeechStarted(Event):
    pass


@dataclass(frozen=True, kw_only=True)
class SpeechEnded(Event):
    duration_s: float
    wake_max: float | None = None  # highest wake word score during this utterance (None: no wake word model)


@dataclass(frozen=True, kw_only=True)
class ActivationEvent(Event):
    """Wake word or push-to-talk. Both sources produce exactly this event."""

    source: ActivationSource
    score: float | None = None


@dataclass(frozen=True, kw_only=True)
class AttendingEnded(Event):
    reason: Literal["query_complete", "button_released", "timeout", "stream_closed"]


@dataclass(frozen=True, kw_only=True)
class TranscriptEvent(Event):
    line_id: str
    text: str
    final: bool
    mode: TranscriptMode
    latency_ms: int | None = None
    stt_model: str | None = None  # e.g. moonshine-small-streaming
    speaker: Literal["owner", "other"] | None = None  # only from real speaker identification; never guessed


@dataclass(frozen=True, kw_only=True)
class PipelineStateChanged(Event):
    state: PipelineState


# --- app ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ProfileChanged(Event):
    profile: str
    loading: bool = False
    error: str | None = None


# --- brain & memory (Phase 2) -------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BrainStateChanged(Event):
    state: BrainState


@dataclass(frozen=True, kw_only=True)
class MemoryStored(Event):
    id: int
    text: str
    kind: str
    confidence: float
    memory_ts: float
    source: str = "extracted"  # extracted | explicit | legacy
    subject: str | None = None
    event_ts: float | None = None
    evidence: str = ""  # comma-separated utterance ids


@dataclass(frozen=True, kw_only=True)
class FactSuperseded(Event):
    """A newer fact replaced an older one (kept, marked superseded or retracted, and linked)."""

    old_id: int
    new_id: int
    status: Literal["superseded", "retracted"]


@dataclass(frozen=True, kw_only=True)
class FactChanged(Event):
    """A fact was corrected, retracted, restored, pinned or merged (by voice or in the UI)."""

    action: Literal["correct", "retract", "restore", "pin", "unpin", "merge"]
    fact_id: int
    other_id: int | None = None  # the corrected version / the merged fact
    actor: str = "ui"
    text: str = ""


@dataclass(frozen=True, kw_only=True)
class EpisodeExtracted(Event):
    """Background extraction finished one episode (a conversation)."""

    episode_id: int
    lines: int
    outcome: Literal["done", "skipped", "failed"]
    added: int = 0
    updated: int = 0
    duplicates: int = 0
    duration_ms: int = 0
    llm_calls: int = 0
    error: str | None = None


@dataclass(frozen=True, kw_only=True)
class MemoriesDeleted(Event):
    count: int
    reason: Literal["user_delete", "forget_command", "forget_button", "resume_extraction"]
    ids: str  # comma-separated memory ids


@dataclass(frozen=True, kw_only=True)
class QueryReceived(Event):
    query_id: str
    text: str


@dataclass(frozen=True, kw_only=True)
class ReplySentence(Event):
    """One sentence of a reply, emitted only after it passed the gate and was sent to TTS."""

    query_id: str
    index: int
    text: str
    citations: str  # comma-separated memory ids cited by this sentence


@dataclass(frozen=True, kw_only=True)
class ReplyCompleted(Event):
    query_id: str
    query: str
    text: str
    outcome: ReplyOutcome
    citations: str = ""
    closest_id: int | None = None
    closest_text: str | None = None
    closest_similarity: float | None = None
    blocked_sentences: int = 0
    backend: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_s: float | None = None
    llm_ttft_ms: int | None = None
    time_to_first_audio_ms: int | None = None  # query complete -> first answer audio
    total_ms: int | None = None
    path: str = ""  # how it was answered: "tool: recall", "tool: lookup (weather)", "answered directly", ...
    llm_calls: int = 0  # model passes for this query (1 for chit-chat, 2 for a tool query)
    explain: str = ""  # JSON for "Why this answer?": query plan, facts offered with scores, filtered facts and why


# --- online boundary & tools (Phase 3) -------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class LookupPerformed(Event):
    """One gateway call, accepted or refused, exactly as the gateway audited it."""

    request_id: str
    kind: str | None
    request: str | None  # the exact JSON the gateway accepted, e.g. {"kind":"weather","params":{...}}
    upstream: str  # newline-separated requests that went to the internet
    status: str  # ok | refused | error
    reason: str | None = None
    provider: str | None = None
    response_bytes: int = 0
    reply_bytes: int = 0
    duration_ms: int = 0
    scrubbed: int = 0


@dataclass(frozen=True, kw_only=True)
class GatewayStatus(Event):
    running: bool
    offline: bool
    detail: str | None = None


@dataclass(frozen=True, kw_only=True)
class TaskChanged(Event):
    action: Literal["added", "completed", "deleted", "fired", "expired"]
    task_id: int
    text: str
    kind: str
    due_ts: float | None = None


@dataclass(frozen=True, kw_only=True)
class Announcement(Event):
    """Something Kin says on its own initiative (a reminder)."""

    text: str
    kind: Literal["reminder"]


# --- routing (intent refactor) -----------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ConfirmationRequested(Event):
    """A destructive action waits for the user to say "confirm" (or press Confirm in the UI)."""

    confirm_id: str
    action: Literal["forget_span", "forget_last", "cancel_reminder", "forget_person"]
    summary: str  # what Kin said, e.g. "That's 3 memories from the last 10 minutes. Say confirm to delete."
    count: int
    expires_ts: float


@dataclass(frozen=True, kw_only=True)
class ConfirmationResolved(Event):
    confirm_id: str
    outcome: Literal["confirmed", "cancelled", "expired"]
    detail: str = ""


@dataclass(frozen=True, kw_only=True)
class PassiveTagged(Event):
    """What the passive router decided for one overheard line."""

    text: str
    label: str
    score: float
    margin: float
    source: Literal["semantic", "rule", "llm"]
    decision: Literal["stored", "dropped", "pending", "queued"]  # queued: kept as an utterance, facts come later
    pending: int = 0  # episodes waiting for extraction after this decision
