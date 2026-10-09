"""Pipeline behaviour with fake engines: transcripts, mute, activation, LED."""

from __future__ import annotations

import time

import numpy as np

from kin.core.events import (
    ActivationEvent,
    AttendingEnded,
    AudioLevel,
    LedChanged,
    LedColor,
    PipelineState,
    SpeechEnded,
    SpeechStarted,
    StreamStateChanged,
    TranscriptEvent,
)
from tests.conftest import silence, speech, wake_frames


async def test_starts_listening_with_green_led(harness):
    assert harness.app.stream_open
    assert harness.app.state is PipelineState.LISTENING
    assert harness.hw.led is LedColor.GREEN


async def test_speech_produces_passive_transcript(harness):
    harness.stream.push(silence(5) + speech(10) + silence(5))
    await harness.settle()
    lines = harness.rec.of(TranscriptEvent)
    assert lines, "expected transcript events"
    assert all(e.mode == "passive" for e in lines)
    assert any(not e.final for e in lines) and lines[-1].final
    assert len(harness.rec.of(SpeechStarted)) == 1 and len(harness.rec.of(SpeechEnded)) == 1
    assert harness.rec.of(AudioLevel)


async def test_silence_produces_no_transcript(harness):
    harness.stream.push(silence(40))
    await harness.settle()
    assert not harness.rec.of(TranscriptEvent)
    assert not harness.rec.of(SpeechStarted)


async def test_preroll_is_given_to_stt(harness):
    harness.stream.push(silence(20) + speech(10) + silence(5))
    await harness.settle()
    fed = sum(len(a) for a in harness.engines["passive"].fed)
    # 10 speech frames + 300 ms preroll + trailing silence until VAD release
    assert fed >= 10 * 512 + 4800


async def test_transcribed_audio_is_zeroed_in_ring_buffer(harness):
    harness.stream.push(speech(10) + silence(5))
    await harness.settle()
    assert not harness.app.ring.read(0).any(), "transcribed frames must be wiped from RAM"


async def test_mute_closes_stream_within_200ms(harness):
    stream = harness.stream
    t0 = time.monotonic()
    harness.hw.simulate_mute(True)
    closed = await harness.bus.wait_for(StreamStateChanged, lambda e: not e.open, timeout=1)
    elapsed = time.monotonic() - t0
    assert closed.open is False
    assert stream.closed and not stream.active, "mute must close the stream, not drop frames"
    assert stream.closed_at - t0 < 0.2 and elapsed < 0.2
    assert not harness.app.stream_open
    await harness.settle()
    assert harness.hw.led is LedColor.OFF
    assert harness.app.state is PipelineState.IDLE


async def test_transcript_stops_when_muted_and_resumes_on_unmute(harness):
    harness.hw.simulate_mute(True)
    await harness.settle()
    harness.rec.events.clear()
    harness.streams[0].push(speech(20) + silence(5))  # a closed stream delivers nothing
    await harness.settle()
    assert not harness.rec.of(TranscriptEvent)
    assert not harness.app.ring.read(0).any(), "ring buffer is wiped on mute"

    harness.hw.simulate_mute(False)
    await harness.settle()
    assert len(harness.streams) == 2 and harness.app.stream_open
    assert harness.hw.led is LedColor.GREEN
    harness.stream.push(speech(10) + silence(5))
    await harness.settle()
    assert harness.rec.of(TranscriptEvent)


async def test_capture_callback_ignores_frames_after_close(harness):
    stream = harness.stream
    callback = stream.callback
    harness.hw.simulate_mute(True)
    await harness.settle()
    harness.rec.events.clear()
    callback(np.full((512, 1), 0.3, dtype=np.float32), 512, None, None)  # late PortAudio callback
    await harness.settle()
    assert not harness.rec.of(AudioLevel)


async def test_wake_word_and_button_fire_the_same_activation_event(harness):
    harness.stream.push(wake_frames(4))
    act = await harness.bus.wait_for(ActivationEvent, timeout=2)
    assert act.source == "wake_word" and act.score is not None
    await harness.settle()
    assert harness.hw.led is LedColor.WHITE_PULSE
    harness.stream.push(speech(6) + silence(5))
    await harness.bus.wait_for(AttendingEnded, timeout=2)
    await harness.settle()
    assert harness.hw.led is LedColor.GREEN

    harness.hw.simulate_button(True)
    act2 = await harness.bus.wait_for(ActivationEvent, timeout=2)
    assert type(act2) is type(act) and act2.source == "button"
    harness.hw.simulate_button(False)
    ended = await harness.bus.wait_for(AttendingEnded, timeout=2)
    assert ended.reason == "button_released"


async def test_wake_word_query_goes_to_query_stt(harness):
    harness.stream.push(wake_frames(4))
    await harness.bus.wait_for(ActivationEvent, timeout=2)
    harness.stream.push(speech(8) + silence(5))
    ended = await harness.bus.wait_for(AttendingEnded, timeout=2)
    assert ended.reason == "query_complete"
    await harness.settle()
    query = [e for e in harness.rec.of(TranscriptEvent) if e.mode == "query"]
    assert query and query[-1].final and query[-1].text.startswith("small")
    assert harness.app.state is PipelineState.LISTENING


async def test_hold_to_talk_transcribes_while_held_even_in_pauses(harness):
    harness.hw.simulate_button(True)
    await harness.bus.wait_for(ActivationEvent, timeout=2)
    harness.stream.push(speech(4) + silence(10) + speech(4))
    await harness.settle()
    assert harness.app.state is PipelineState.ATTENDING, "a pause must not end push-to-talk"
    harness.hw.simulate_button(False)
    await harness.bus.wait_for(AttendingEnded, timeout=2)
    await harness.settle()
    finals = [e for e in harness.rec.of(TranscriptEvent) if e.mode == "query" and e.final]
    assert len(finals) == 1
    assert harness.engines["query"].samples == 18 * 512


async def test_button_while_muted_does_nothing(harness):
    harness.hw.simulate_mute(True)
    await harness.settle()
    harness.hw.simulate_button(True)
    await harness.settle()
    assert not harness.rec.of(ActivationEvent)
    assert harness.hw.led is LedColor.OFF


async def test_attending_times_out_without_speech(make_harness):
    h = await make_harness(query_timeout_s=0.2)
    h.stream.push(wake_frames(4) + silence(5))
    await h.bus.wait_for(ActivationEvent, timeout=2)
    ended = await h.bus.wait_for(AttendingEnded, timeout=2)
    assert ended.reason == "timeout"
    await h.settle()
    assert h.hw.led is LedColor.GREEN


async def test_led_follows_real_pipeline_state(harness):
    assert [e.color for e in harness.rec.of(LedChanged)] == [LedColor.GREEN]  # startup: stream opened
    harness.rec.events.clear()
    harness.stream.push(wake_frames(4))
    await harness.bus.wait_for(ActivationEvent, timeout=2)
    harness.stream.push(speech(4) + silence(5))
    await harness.bus.wait_for(LedChanged, lambda e: e.color is LedColor.GREEN, timeout=2)
    harness.hw.simulate_mute(True)
    await harness.settle()
    colors = [e.color for e in harness.rec.of(LedChanged)]
    assert colors == [LedColor.WHITE_PULSE, LedColor.GREEN, LedColor.OFF], colors


async def test_led_never_shows_green_while_stream_closed(harness):
    """Even if a late 'listening' state arrives after mute, the LED stays off."""
    harness.stream.push(wake_frames(4))
    await harness.bus.wait_for(ActivationEvent, timeout=2)
    harness.stream.push(speech(4) + silence(5))
    await harness.bus.wait_for(AttendingEnded, timeout=2)
    harness.hw.simulate_mute(True)  # races with the pipeline's LISTENING state change
    await harness.settle()
    assert harness.hw.led is LedColor.OFF
    after_mute = [e for e in harness.rec.events if isinstance(e, (LedChanged, StreamStateChanged))]
    closed_at = next(i for i, e in enumerate(after_mute) if isinstance(e, StreamStateChanged) and not e.open)
    assert all(e.color is LedColor.OFF for e in after_mute[closed_at:] if isinstance(e, LedChanged))


async def test_pi5_profile_lazy_loads_query_stt(make_harness):
    h = await make_harness("pi5")
    assert not h.app.pipeline.engines.query_loaded
    h.hw.simulate_button(True)
    await h.bus.wait_for(ActivationEvent, timeout=2)
    await h.settle()
    assert h.app.pipeline.engines.query_loaded


async def test_profile_switch_hot_swaps_pipeline(harness):
    await harness.app.switch_profile("pi5")
    assert harness.app.cfg.profile == "pi5" and harness.app.cfg.runtime.threads == 4
    assert harness.streams[0].closed and harness.app.stream_open
    await harness.settle()
    assert harness.hw.led is LedColor.GREEN


def test_stop_with_backlog_is_fast_and_frees_models_only_after_worker_exits():
    import threading

    from kin.audio.pipeline import Engines, PerceptionPipeline
    from kin.audio.ring_buffer import RingBuffer
    from kin.config import load_config
    from tests.conftest import FakeSTT, FakeVAD

    class SlowSTT(FakeSTT):
        def feed(self, samples):
            time.sleep(0.05)
            super().feed(samples)

        def close(self):
            assert not any(t.name == "kin-perception" and t.is_alive() for t in threading.enumerate())
            super().close()

    cfg = load_config("laptop")
    stt = SlowSTT("tiny")
    p = PerceptionPipeline(cfg, Engines(FakeVAD(), None, stt, lambda: SlowSTT("small")), RingBuffer(30, 16000), lambda e: None)
    p.start()
    p.stream_opened()
    for f in speech(300):  # ~15 s of work queued
        p.submit_frame(f)
    time.sleep(0.2)
    t0 = time.monotonic()
    p.stop()
    assert time.monotonic() - t0 < 1.0
    assert stt.closed
