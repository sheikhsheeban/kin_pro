"""Integration tests with the real models (Silero VAD, openWakeWord, Moonshine v2).

Speech comes from the sample recording bundled with moonshine-voice, read
into memory only. Skipped when ./models has not been downloaded.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import numpy as np
import pytest

from kin.app import KinApp
from kin.audio import build_engines
from kin.config import load_config
from kin.core.bus import EventBus
from kin.core.events import ActivationEvent, AttendingEnded, SpeechStarted, TranscriptEvent
from tests.conftest import FRAME, FakeInputStream, Recorder, models_available
from tests.test_privacy import audio_files_written, snapshot_files

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not models_available(), reason="models not downloaded (uv run kin download-models)"),
]


@pytest.fixture(scope="module")
def sample_speech() -> np.ndarray:
    """~6 s of read speech at 16 kHz, in RAM."""
    from moonshine_voice import get_assets_path, load_wav_file
    from scipy.signal import resample_poly

    audio, sr = load_wav_file(os.path.join(get_assets_path(), "two_cities.wav"))
    x = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    return x[: 16000 * 6]


def frames_of(x: np.ndarray) -> list[np.ndarray]:
    n = len(x) // FRAME
    return list(x[: n * FRAME].reshape(n, FRAME))


def test_silero_vad_separates_speech_from_silence(sample_speech):
    from kin.audio.vad import SileroVAD

    cfg = load_config("laptop")
    vad = SileroVAD(cfg.model_path(cfg.vad.model))
    quiet = [vad.speech_prob(f) for f in frames_of(np.zeros(16000, np.float32))]
    vad.reset()
    talk = [vad.speech_prob(f) for f in frames_of(sample_speech)]
    assert max(quiet) < 0.2
    assert np.mean(np.array(talk) > 0.5) > 0.4


def test_openwakeword_quiet_on_silence_and_non_wake_speech(sample_speech):
    from kin.audio.wake import OpenWakeWordDetector

    cfg = load_config("laptop")
    w = OpenWakeWordDetector(
        cfg.model_path(cfg.wake.model), cfg.model_path(cfg.wake.melspec_model), cfg.model_path(cfg.wake.embedding_model)
    )
    scores = [s for f in frames_of(np.zeros(16000 * 2, np.float32)) + frames_of(sample_speech) if (s := w.process(f)) is not None]
    assert scores and max(scores) < cfg.wake.threshold


class RealtimeFeeder(threading.Thread):
    """Plays frames into a stream callback at real-time pace, like a microphone."""

    def __init__(self, stream: FakeInputStream, frames: list[np.ndarray]) -> None:
        super().__init__(daemon=True)
        self.stream, self.frames = stream, frames
        self.t0 = 0.0

    def run(self) -> None:
        self.t0 = time.monotonic()
        for i, f in enumerate(self.frames):
            self.stream.push([f])
            delay = self.t0 + (i + 1) * FRAME / 16000 - time.monotonic()
            if delay > 0:
                time.sleep(delay)


@pytest.fixture
async def real_app():
    cfg = load_config("laptop")
    streams: list[FakeInputStream] = []

    def factory(**kw):
        streams.append(FakeInputStream(**kw))
        return streams[-1]

    bus = EventBus()
    rec = Recorder(bus)
    app = KinApp(
        cfg, bus=bus, engines_factory=build_engines, stream_factory=factory,
        device_namer=lambda _d: "sample", brain_factory=None, gateway_factory=None,
    )
    await app.start()
    yield app, streams, rec
    await app.stop()


async def test_live_transcript_within_one_second_and_no_audio_on_disk(real_app, sample_speech):
    app, streams, rec = real_app
    before = snapshot_files()
    silence = np.zeros(16000, np.float32)
    feeder = RealtimeFeeder(streams[-1], frames_of(np.concatenate([silence, sample_speech, silence, silence])))
    feeder.start()

    started = await app.bus.wait_for(SpeechStarted, timeout=5)
    first = await app.bus.wait_for(TranscriptEvent, timeout=5)
    latency = first.ts - started.ts
    print(f"\nfirst transcript {latency * 1000:.0f} ms after speech onset: {first.text!r}")
    assert latency < 1.0, f"first transcript took {latency:.2f}s"

    await asyncio.to_thread(feeder.join, 15)
    await asyncio.sleep(1.0)
    finals = [e for e in rec.of(TranscriptEvent) if e.final]
    text = " ".join(e.text for e in finals).lower()
    print(f"final passive transcript: {text!r}")
    assert all(e.mode == "passive" for e in finals)
    assert "best of times" in text or "worst of times" in text
    assert audio_files_written(before) == []


async def test_push_to_talk_query_uses_small_model(real_app, sample_speech):
    app, streams, rec = real_app
    app.hardware.simulate_button(True)
    await app.bus.wait_for(ActivationEvent, timeout=2)
    feeder = RealtimeFeeder(streams[-1], frames_of(sample_speech[: 16000 * 3]))
    feeder.start()
    await asyncio.to_thread(feeder.join, 10)
    app.hardware.simulate_button(False)
    await app.bus.wait_for(AttendingEnded, timeout=5)
    await asyncio.sleep(0.3)
    query = [e for e in rec.of(TranscriptEvent) if e.mode == "query" and e.final]
    print(f"\nquery transcript: {[e.text for e in query]}")
    assert query and "best" in " ".join(e.text for e in query).lower()
