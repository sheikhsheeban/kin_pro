"""Real microphone: the PortAudio stream really opens, delivers 16 kHz frames, and mute closes it fast.

Skipped automatically on machines without an input device.
"""

from __future__ import annotations

import threading
import time

import pytest

from kin.audio.capture import MicCapture
from kin.config import load_config


def _has_input() -> bool:
    try:
        import sounddevice as sd

        sd.check_input_settings(samplerate=16000, channels=1, dtype="float32")
        return True
    except Exception:
        return False


pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not _has_input(), reason="no microphone available")]


def test_real_mic_delivers_frames_and_mute_closes_within_200ms():
    cfg = load_config("laptop")
    frames: list[int] = []
    got = threading.Event()

    def on_frame(f):
        frames.append(len(f))
        if len(frames) >= 5:
            got.set()

    cap = MicCapture(cfg.audio, on_frame)
    cap.open()
    try:
        assert cap.is_open
        assert got.wait(3), "no audio frames from the microphone"
        assert set(frames) == {cfg.audio.frame_samples}
    finally:
        t0 = time.monotonic()
        cap.close()
        elapsed = time.monotonic() - t0
    assert not cap.is_open
    n = len(frames)
    time.sleep(0.2)
    assert len(frames) == n, "frames arrived after the stream was closed"
    print(f"\nreal mic stream closed in {elapsed * 1000:.0f} ms")
    assert elapsed < 0.2
