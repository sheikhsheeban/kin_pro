"""Wake word: the model sees every frame, is never reset during normal operation, and detections are
ignored while Kin speaks. The real-model test feeds synthetic "Hey Jarvis" speech (Piper, in RAM)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import kin.audio.pipeline as pipeline_mod
from kin.audio.pipeline import Engines, PerceptionPipeline
from kin.audio.ring_buffer import RingBuffer
from kin.audio.wake import OpenWakeWordDetector
from kin.config import load_config
from kin.core.events import ActivationEvent, AttendingEnded, SpeechEnded, WakeLevel
from tests.conftest import WAKE_MARKER, FakeSTT, FakeVAD, silence, speech, wake_frames


class RecordingWake:
    """FakeWake that also counts every frame it is given."""

    def __init__(self, score: float = 0.99) -> None:
        self.frames = 0
        self.resets = 0
        self.score = score

    def process(self, frame: np.ndarray) -> float | None:
        self.frames += 1
        return self.score if np.any(np.isclose(frame, WAKE_MARKER, atol=1e-6)) else 0.01

    def reset(self) -> None:
        self.resets += 1


class Run:
    """A started pipeline with fakes; `feed` returns once every queued frame has been processed."""

    def __init__(self, wake: RecordingWake, **cfg_changes) -> None:
        self.cfg = load_config("laptop")
        self.cfg.vad.min_silence_ms = 96
        for key, value in cfg_changes.items():
            setattr(self.cfg.wake, key, value)
        self.wake = wake
        self.events: list = []
        self._lock = threading.Lock()
        self.p = PerceptionPipeline(self.cfg, Engines(FakeVAD(), wake, FakeSTT("tiny"), lambda: FakeSTT("small")),
                                    RingBuffer(30, 16000), self._emit)
        self.p.start()
        self.p.stream_opened()

    def _emit(self, e) -> None:
        with self._lock:
            self.events.append(e)

    def feed(self, frames: list[np.ndarray]) -> None:
        target = self.wake.frames + len(frames)
        for f in frames:
            self.p.submit_frame(f)
        deadline = time.monotonic() + 5
        while self.wake.frames < target and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.02)  # let the last frame's events out

    def cmd(self, fn, *args) -> None:
        fn(*args)
        time.sleep(0.05)

    def of(self, cls) -> list:
        with self._lock:
            return [e for e in self.events if isinstance(e, cls)]

    def stop(self) -> None:
        self.p.stop()


@pytest.fixture
def run():
    made: list[Run] = []

    def make(wake=None, **cfg_changes) -> Run:
        r = Run(wake or RecordingWake(), **cfg_changes)
        made.append(r)
        return r

    yield make
    for r in made:
        r.stop()


def test_wake_model_sees_every_frame_in_silence_speech_and_while_ducked(run):
    r = run()
    r.feed(silence(20))
    r.cmd(r.p.duck, True)
    r.feed(speech(10))
    r.cmd(r.p.duck, False)
    r.feed(silence(10))
    assert r.wake.frames == 40, "every frame reaches the wake model, not only those VAD calls speech"


def test_no_reset_between_utterances_activations_and_replies(run, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "UNDUCK_WAKE_HOLDOFF_S", 0.0)
    r = run(cooldown_s=0.0)
    for _ in range(3):
        r.feed(speech(10) + silence(6))  # an overheard sentence
        r.feed(wake_frames(4) + speech(6) + silence(6))  # "Hey Jarvis, …" and the question
        r.cmd(r.p.duck, True)  # Kin answers
        r.feed(speech(5))
        r.cmd(r.p.duck, False)
    assert len([a for a in r.of(ActivationEvent) if a.source == "wake_word"]) == 3
    assert r.wake.resets == 0, "openWakeWord zeroes its first 5 predictions after a reset"
    r.cmd(r.p.stream_closed)  # mute: the one place its buffers are cleared
    assert r.wake.resets == 1


def test_detection_ignored_while_ducked_then_holdoff_then_works(run, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "UNDUCK_WAKE_HOLDOFF_S", 0.3)
    r = run()
    r.cmd(r.p.duck, True)
    r.feed(wake_frames(4) + silence(4))  # Kin says something that scores high: ignored
    assert not r.of(ActivationEvent)
    r.cmd(r.p.duck, False)
    r.feed(wake_frames(4) + silence(4))  # inside the hold-off after Kin stops talking
    assert not r.of(ActivationEvent)
    time.sleep(0.35)
    r.feed(wake_frames(4))
    assert [a.source for a in r.of(ActivationEvent)] == ["wake_word"]


def test_cooldown_after_attending_is_kept(run):
    r = run()  # cooldown_s = 2.0 from the profile
    r.feed(wake_frames(4) + speech(4) + silence(6))
    assert len(r.of(AttendingEnded)) == 1
    r.feed(wake_frames(4) + silence(4))  # straight away: within the 2 s cooldown
    assert len(r.of(ActivationEvent)) == 1


def test_threshold_is_configurable(run):
    r = run(RecordingWake(score=0.7), threshold=0.8)
    r.feed(wake_frames(4) + silence(4))
    assert not r.of(ActivationEvent)
    r2 = run(RecordingWake(score=0.7), threshold=0.6)
    r2.feed(wake_frames(4) + silence(4))
    assert [a.score for a in r2.of(ActivationEvent)] == [0.7]
    with pytest.raises(ValueError):
        load_config("laptop", {"wake": {"threshold": 1.5}})


def test_live_score_and_utterance_max_are_published(run):
    r = run(threshold=0.995)  # high enough that the marker does not activate
    r.feed(silence(10) + wake_frames(4) + silence(6))
    time.sleep(0.1)  # the meter publishes at most 15 times a second; frames here arrive faster than real time
    r.feed(silence(1))
    levels = r.of(WakeLevel)
    assert levels and all(lv.threshold == 0.995 for lv in levels)
    assert max(lv.score for lv in levels) == pytest.approx(0.99)
    (ended,) = r.of(SpeechEnded)
    assert ended.wake_max == pytest.approx(0.99)


def test_512_frames_become_1280_sample_int16_chunks():
    seen: list[np.ndarray] = []

    class Model:
        def predict(self, chunk):
            seen.append(chunk.copy())
            return {"hey_jarvis": 0.25}

    det = object.__new__(OpenWakeWordDetector)
    det._model, det._key, det._pending = Model(), "hey_jarvis", np.zeros(0, np.int16)
    x = np.linspace(-1.2, 1.2, 512 * 5).astype(np.float32)  # includes values outside [-1, 1]
    scores = [det.process(x[i : i + 512]) for i in range(0, len(x), 512)]
    assert [len(c) for c in seen] == [1280, 1280] and all(c.dtype == np.int16 for c in seen)
    assert scores == [None, None, 0.25, None, 0.25]  # a score once a full 80 ms chunk is ready (3 x 512 >= 1280)
    expected = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    assert np.array_equal(np.concatenate(seen), expected), "no sample lost, scaled by 32767, clipped"
    assert seen[0][0] == -32767 and seen[1][-1] == 32767


# --- real models: synthetic "Hey Jarvis" through the real VAD + openWakeWord -----------------------------


@pytest.mark.models
def test_real_hey_jarvis_activates_after_silence():
    from tests.wake_audio import TEN_TRIES, available, frames, room_noise, say

    cfg = load_config("laptop")
    if not available() or not cfg.model_path(cfg.wake.model).exists():
        pytest.skip("Piper voice or openWakeWord model not downloaded")
    from kin.audio import build_engines

    real = build_engines(cfg)
    real.passive_stt.close()
    resets = []
    real_reset = real.wake.reset
    real.wake.reset = lambda: (resets.append(1), real_reset())
    events: list = []
    p = PerceptionPipeline(cfg, Engines(real.vad, real.wake, FakeSTT("tiny"), lambda: FakeSTT("small")),
                           RingBuffer(30, 16000), events.append)
    p.start()
    p.stream_opened()
    try:
        stream = frames(room_noise(3 * 16000, 0.002, 1))  # three seconds of a quiet room
        clip = say(TEN_TRIES[0])  # "Hey Jarvis, what time is it?"
        stream += frames(clip + room_noise(len(clip), 0.002, 2)) + frames(room_noise(16000, 0.002, 3))
        for f in stream:
            p.submit_frame(f)
        deadline = time.monotonic() + 30
        while not any(isinstance(e, ActivationEvent) for e in events) and time.monotonic() < deadline:
            time.sleep(0.05)
        acts = [e for e in events if isinstance(e, ActivationEvent)]
        assert acts and acts[0].source == "wake_word" and acts[0].score >= cfg.wake.threshold, acts
        assert resets == [], "no reset during normal operation"
    finally:
        p.stop()
