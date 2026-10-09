"""Perception pipeline: frames -> ring buffer -> wake word (every frame) and VAD -> STT -> events.

All model inference runs on one dedicated worker thread, which owns every
piece of pipeline state. Mic frames and control commands (button, stream
opened/closed) arrive on the same FIFO queue, so they are handled strictly
in order and nothing needs a lock. Results go out as bus events.

Modes:
  idle       mic stream closed; nothing runs
  listening  passive capture with Moonshine Tiny, gated by VAD
  attending  after an activation (wake word or button); speech goes to Moonshine Small

The wake word model sees every frame while the stream is open, whatever VAD says and even
while Kin is speaking: its rolling feature window (~1.3 s) must hold continuous, current
audio, and openWakeWord zeroes its first 5 predictions after a reset. So it is reset only
when the stream closes (mute). Detections count only while listening, not ducked, and
outside the cooldown.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kin.audio.ring_buffer import RingBuffer
from kin.audio.stt import SttEngine, SttLine
from kin.audio.vad import VoiceActivityDetector
from kin.audio.wake import WakeWordDetector, spoken_wake
from kin.config import KinConfig
from kin.core.events import (
    ActivationEvent,
    ActivationSource,
    AttendingEnded,
    AudioLevel,
    Event,
    PipelineState,
    PipelineStateChanged,
    SpeechEnded,
    SpeechStarted,
    TranscriptEvent,
    TranscriptMode,
    WakeLevel,
)

log = logging.getLogger(__name__)

VAD_RELEASE_MARGIN = 0.15  # hysteresis: speech ends below threshold - margin
MAX_QUERY_SECONDS = 15.0
# After Kin stops speaking, its own last words are still in the wake model's ~1.3 s window.
UNDUCK_WAKE_HOLDOFF_S = 1.3


class Engines:
    """The models the pipeline uses. The query STT may be loaded lazily (pi5 profile)."""

    def __init__(
        self,
        vad: VoiceActivityDetector,
        wake: WakeWordDetector | None,
        passive_stt: SttEngine,
        query_stt_factory: Callable[[], SttEngine],
        lazy_query: bool = False,
    ) -> None:
        self.vad = vad
        self.wake = wake
        self.passive_stt = passive_stt
        self._query_factory = query_stt_factory
        self._query: SttEngine | None = None if lazy_query else query_stt_factory()

    @property
    def query_loaded(self) -> bool:
        return self._query is not None

    @property
    def query_stt(self) -> SttEngine:
        if self._query is None:
            t0 = time.monotonic()
            self._query = self._query_factory()
            log.info("lazy-loaded query STT in %.2fs", time.monotonic() - t0)
        return self._query

    def close(self) -> None:
        for engine in (self.passive_stt, self._query):
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    log.exception("closing STT engine failed")


@dataclass(frozen=True)
class _Cmd:
    name: str
    arg: object = None


_STOP = _Cmd("stop")


class PerceptionPipeline:
    def __init__(self, cfg: KinConfig, engines: Engines, ring: RingBuffer, emit: Callable[[Event], None]) -> None:
        self.cfg = cfg
        self.engines = engines
        self.ring = ring
        self._emit = emit
        self._q: queue.SimpleQueue[np.ndarray | _Cmd] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

        sr = cfg.audio.sample_rate
        self._frame = cfg.audio.frame_samples
        self._min_speech_frames = max(1, math.ceil(cfg.vad.min_speech_ms * sr / 1000 / self._frame))
        self._min_silence_frames = max(1, math.ceil(cfg.vad.min_silence_ms * sr / 1000 / self._frame))
        self._preroll = int(cfg.audio.preroll_ms * sr / 1000)
        self._level_period = 1.0 / cfg.audio.level_meter_hz if cfg.audio.level_meter_hz > 0 else math.inf

        # --- worker-thread state (only touched on the worker thread) ---
        self._mode = PipelineState.IDLE
        self._speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._speech_started_at = 0.0
        self._stt: SttEngine | None = None
        self._stt_mode: TranscriptMode = "passive"
        self._fed_until = 0
        self._utterance = 0
        self._ptt = False
        self._attend_source: ActivationSource | None = None
        self._attend_started = 0.0
        self._attend_from = 0
        self._query_has_text = False
        self._wake_cooldown_until = 0.0
        self._activating = False  # an activation is finalising the passive line: that line must not activate again
        self._wake_peak = 0.0  # highest wake score since the last WakeLevel event
        self._utterance_wake_max: float | None = None  # highest wake score during the current speech
        self._last_level = 0.0
        self._ducked = False

    # --- public API (any thread) -----------------------------------------

    @property
    def state(self) -> PipelineState:
        return self._mode

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="kin-perception", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker and free the models. Safe to call from any thread.

        Frames still queued are discarded. Models are freed only after the
        worker has exited, never while it might still be using them.
        """
        self._stopping.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            self._q.put(_STOP)
            thread.join(timeout)
            if thread.is_alive():
                log.error("perception worker did not stop within %.0fs; leaking its models", timeout)
                return
        self.engines.close()

    def submit_frame(self, frame: np.ndarray) -> None:
        """Called from the audio callback thread. Must stay cheap."""
        self._q.put(frame)

    def stream_opened(self) -> None:
        self._q.put(_Cmd("stream_opened"))

    def stream_closed(self) -> None:
        self._q.put(_Cmd("stream_closed"))

    def button(self, pressed: bool) -> None:
        self._q.put(_Cmd("button", pressed))

    def duck(self, ducked: bool) -> None:
        """While Kin speaks, ignore the mic so Kin never transcribes or wakes on its own voice."""
        self._q.put(_Cmd("duck", ducked))

    # --- worker ----------------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                item = None
            if item is _STOP:
                self._stop_stt()
                return
            if self._stopping.is_set():
                continue  # drain the backlog without processing it
            try:
                if isinstance(item, np.ndarray):
                    if self._mode is not PipelineState.IDLE:
                        self._process_frame(item)
                elif isinstance(item, _Cmd):
                    self._handle_cmd(item)
                self._tick()
            except Exception:
                log.exception("perception pipeline error")

    def _handle_cmd(self, cmd: _Cmd) -> None:
        if cmd.name == "stream_opened":
            self._reset_audio_state()
            self._set_mode(PipelineState.LISTENING)
        elif cmd.name == "stream_closed":
            if self._mode is PipelineState.ATTENDING:
                self._end_attending("stream_closed")
            self._stop_stt()
            self._ptt = False
            self._reset_audio_state()
            self.ring.clear()
            if self.engines.wake is not None:
                self.engines.wake.reset()  # muted: drop the audio openWakeWord keeps in its own buffers
            self._set_mode(PipelineState.IDLE)
        elif cmd.name == "duck":
            if bool(cmd.arg) == self._ducked:
                return
            self._ducked = bool(cmd.arg)
            if self._mode is not PipelineState.ATTENDING:
                self._stop_stt()
            self._reset_audio_state()
            if not self._ducked:  # Kin's last words must not wake it
                self._wake_cooldown_until = max(self._wake_cooldown_until, time.monotonic() + UNDUCK_WAKE_HOLDOFF_S)
        elif cmd.name == "button":
            if cmd.arg:
                if self._mode is PipelineState.IDLE:
                    return  # muted: there is nothing to listen with
                if self._ducked:  # barge-in: the user wants to talk over Kin
                    self._ducked = False
                    self._reset_audio_state()
                self._ptt = True
                self._activate("button", None)
            elif self._ptt:
                self._ptt = False
                if self._mode is PipelineState.ATTENDING:
                    self._end_attending("button_released")

    def _process_frame(self, frame: np.ndarray) -> None:
        end = self.ring.write(frame)
        now = time.monotonic()

        score = self._feed_wake(frame)  # every frame, before VAD and even while ducked

        if now - self._last_level >= self._level_period:
            self._last_level = now
            rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
            self._emit(AudioLevel(rms_db=round(20 * math.log10(rms + 1e-9), 1)))
            if self.engines.wake is not None:
                self._emit(WakeLevel(score=round(self._wake_peak, 3), threshold=self.cfg.wake.threshold))
                self._wake_peak = 0.0

        if self._ducked and not self._ptt:
            return

        if score is not None and self._wake_allowed(score):
            self._wake_cooldown_until = now + self.cfg.wake.cooldown_s
            self._activate("wake_word", round(score, 3))

        prob = self.engines.vad.speech_prob(frame)
        thr = self.cfg.vad.threshold

        if not self._speaking:
            self._speech_run = self._speech_run + 1 if prob >= thr else 0
            if self._speech_run >= self._min_speech_frames:
                self._speaking = True
                self._silence_run = 0
                self._speech_started_at = now
                onset = end - self._speech_run * len(frame) - self._preroll
                if self._mode is PipelineState.ATTENDING:
                    onset = max(onset, self._attend_from)
                self._fed_until = max(onset, self.ring.start, self._fed_until)
                self._utterance_wake_max = score
                self._emit(SpeechStarted())
        else:
            self._silence_run = self._silence_run + 1 if prob < thr - VAD_RELEASE_MARGIN else 0

        if self._speaking and score is not None:
            self._utterance_wake_max = max(self._utterance_wake_max or 0.0, score)

        if self._speaking or self._ptt:
            self._ensure_stt()
            self._feed_stt(end)

        if self._speaking and self._silence_run >= self._min_silence_frames:
            self._speaking = False
            self._speech_run = 0
            wake_max = None if self.engines.wake is None else round(self._utterance_wake_max or 0.0, 3)
            self._utterance_wake_max = None
            self._emit(SpeechEnded(duration_s=round(now - self._speech_started_at, 2), wake_max=wake_max))
            if not self._ptt:
                self._stop_stt()
                if self._mode is PipelineState.ATTENDING and self._query_has_text:
                    self._end_attending("query_complete")

    def _tick(self) -> None:
        if self._mode is not PipelineState.ATTENDING or self._ptt:
            return
        waited = time.monotonic() - self._attend_started
        if (not self._query_has_text and not self._speaking and waited > self.cfg.stt.query_timeout_s) or (
            waited > MAX_QUERY_SECONDS
        ):
            self._end_attending("timeout")

    # --- wake word & activation ---------------------------------------------

    def _feed_wake(self, frame: np.ndarray) -> float | None:
        """Give the wake model this frame. Returns the best score computed from it (one per 80 ms), if any."""
        wake = self.engines.wake
        if wake is None:
            return None
        score = wake.process(frame)
        if score is not None:
            self._wake_peak = max(self._wake_peak, score)
        return score

    def _wake_allowed(self, score: float) -> bool:
        return (
            score >= self.cfg.wake.threshold
            and self._mode is PipelineState.LISTENING
            and not self._ducked
            and time.monotonic() >= self._wake_cooldown_until
        )

    def _activate(self, source: ActivationSource, score: float | None) -> None:
        """Wake word and button both end up here and emit the same ActivationEvent."""
        if self._mode is PipelineState.ATTENDING:
            return
        self._activating = True
        try:
            self._stop_stt()  # finalise any passive line in progress
        finally:
            self._activating = False
        self._attend_source = source
        self._attend_started = time.monotonic()
        self._attend_from = self.ring.end
        self._fed_until = self.ring.end
        self._query_has_text = False
        self._set_mode(PipelineState.ATTENDING)
        self._emit(ActivationEvent(source=source, score=score))
        if self._speaking or self._ptt:
            self._ensure_stt()

    def _end_attending(self, reason: str) -> None:
        self._stop_stt()
        self._attend_source = None
        self._emit(AttendingEnded(reason=reason))  # type: ignore[arg-type]
        self._set_mode(PipelineState.LISTENING)
        self._wake_cooldown_until = time.monotonic() + self.cfg.wake.cooldown_s

    # --- STT -------------------------------------------------------------

    def _ensure_stt(self) -> None:
        if self._stt is not None:
            return
        attending = self._mode is PipelineState.ATTENDING
        self._stt_mode = "query" if attending else "passive"
        self._stt = self.engines.query_stt if attending else self.engines.passive_stt
        self._utterance += 1
        utt, mode, model = self._utterance, self._stt_mode, getattr(self._stt, "name", None)
        self._stt.start(lambda line: self._on_line(utt, mode, line, model))
        self._fed_until = max(self._fed_until, self.ring.start)

    def _feed_stt(self, end: int) -> None:
        if self._stt is None or end <= self._fed_until:
            return
        audio = self.ring.read(self._fed_until, end)
        self._stt.feed(audio)
        # Transcribed audio is wiped from the buffer straight away.
        self.ring.zero(self._fed_until, end)
        self._fed_until = end

    def _stop_stt(self) -> None:
        stt, self._stt = self._stt, None
        if stt is not None:
            stt.stop()

    def _on_line(self, utterance: int, mode: TranscriptMode, line: SttLine, model: str | None = None) -> None:
        if mode == "query" and line.text:
            self._query_has_text = True
        if mode == "passive" and line.final and self._wake_from_text(line, model):
            return
        self._emit(
            TranscriptEvent(
                line_id=f"{utterance}-{line.id}",
                text=line.text,
                final=line.final,
                mode=mode,
                latency_ms=line.latency_ms,
                stt_model=model,
                speaker=line.speaker,
            )
        )

    def _wake_from_text(self, line: SttLine, model: str | None) -> bool:
        """The acoustic wake model can score a voice under its threshold; the passive line then reads "Hey Jarvis,
        who is my manager?". Treat it as an activation: the rest of the line is the question, or, if the line was
        only the wake phrase, Kin attends to what is said next. The line is not stored as overheard speech."""
        if not self.cfg.wake.enabled or self._mode is not PipelineState.LISTENING or self._ducked or self._activating:
            return False
        rest = spoken_wake(line.text, self.cfg.wake.phrase)
        if rest is None:
            return False
        if not rest.strip(" .!?"):
            self._activate("wake_phrase", None)  # "Hey Jarvis." -> listen for the question
            return True
        self._attend_source = "wake_phrase"
        self._set_mode(PipelineState.ATTENDING)
        self._emit(ActivationEvent(source="wake_phrase", score=None))
        self._emit(TranscriptEvent(line_id=f"wake-{self._utterance}-{line.id}", text=rest, final=True, mode="query",
                                   latency_ms=line.latency_ms, stt_model=model, speaker=line.speaker))
        self._end_attending("query_complete")
        return True

    # --- helpers -----------------------------------------------------------

    def _reset_audio_state(self) -> None:
        self._speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self.engines.vad.reset()  # the wake model is deliberately not reset: see the module docstring
        self._fed_until = self.ring.end

    def _set_mode(self, mode: PipelineState) -> None:
        if mode is not self._mode:
            self._mode = mode
            self._emit(PipelineStateChanged(state=mode))
