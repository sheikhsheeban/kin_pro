"""Synthetic "Hey Jarvis" speech for wake word tests, generated in RAM with the Piper voice Kin already ships.

Nothing here is ever written to disk (CLAUDE.md: no audio in test fixtures). openWakeWord's stock
models were trained on synthetic TTS speech, so Piper audio is a fair, repeatable stand-in for a voice.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from tests.conftest import MODELS

SR = 16000
VOICE = MODELS / "tts" / "en_US-lessac-medium.onnx"


@dataclass(frozen=True)
class Try:
    """One attempt: how it is said and how loud the room is."""

    text: str
    length_scale: float = 1.0  # >1 slower
    noise_scale: float = 0.667
    gain: float = 0.5  # peak level at the mic, 0..1
    snr_db: float | None = None  # background noise; None = quiet room
    seed: int = 0


# Ten different tries: speed, voice variation, distance (level) and background noise.
TEN_TRIES: tuple[Try, ...] = (
    Try("Hey Jarvis, what time is it?"),
    Try("Hey Jarvis, when am I meeting Priya?", length_scale=1.15, seed=1),
    Try("Hey Jarvis.", length_scale=0.9, gain=0.35, seed=2),
    Try("Hey Jarvis, do I need an umbrella today?", noise_scale=0.9, seed=3),
    Try("Hey Jarvis, what's on my list?", gain=0.2, seed=4),  # further away
    Try("Hey Jarvis, remind me to call mum.", length_scale=1.25, snr_db=20, seed=5),
    Try("Hey Jarvis, how are you?", noise_scale=0.4, snr_db=15, seed=6),
    Try("Hey Jarvis!", length_scale=0.85, gain=0.7, seed=7),
    Try("Hey Jarvis, forget that.", gain=0.12, seed=8),  # quiet, far side of the room
    Try("Hey Jarvis, what did Rahul say yesterday?", noise_scale=0.8, snr_db=12, seed=9),
)


def available() -> bool:
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return VOICE.exists()


@lru_cache(maxsize=1)
def _voice():
    from piper import PiperVoice

    return PiperVoice.load(str(VOICE))


def say(t: Try) -> np.ndarray:
    """The utterance as 16 kHz float32, peak-normalised to `t.gain`, with optional room noise."""
    from math import gcd

    from piper import SynthesisConfig
    from scipy.signal import resample_poly

    voice = _voice()
    cfg = SynthesisConfig(length_scale=t.length_scale, noise_scale=t.noise_scale, noise_w_scale=0.8)
    audio = np.concatenate([c.audio_float_array for c in voice.synthesize(t.text, syn_config=cfg)]).astype(np.float32)
    src = int(voice.config.sample_rate)
    g = gcd(src, SR)
    audio = resample_poly(audio, SR // g, src // g).astype(np.float32)
    audio *= t.gain / max(1e-6, float(np.abs(audio).max()))
    if t.snr_db is not None:
        audio = audio + room_noise(len(audio), rms_of(audio) / (10 ** (t.snr_db / 20)), t.seed)
    return np.clip(audio, -1, 1).astype(np.float32)


def rms_of(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def room_noise(n: int, rms: float, seed: int = 0) -> np.ndarray:
    """Pinkish noise (a fan, a fridge, a street outside) at the given RMS level."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n + 64)
    pink = np.convolve(white, np.ones(64) / 64, mode="valid")[:n] + 0.3 * white[:n]
    return (pink / (rms_of(pink) or 1.0) * rms).astype(np.float32)


def frames(audio: np.ndarray, size: int = 512) -> list[np.ndarray]:
    pad = (-len(audio)) % size
    audio = np.concatenate([audio, np.zeros(pad, np.float32)])
    return list(audio.reshape(-1, size))
