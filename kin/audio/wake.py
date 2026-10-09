"""Wake word detection with openWakeWord (ONNX, CPU only).

Uses a stock model ("Hey Jarvis") until a custom "Hey Kin" model is trained.
A detection publishes the same `ActivationEvent` as the push-to-talk button.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class WakeWordDetector(Protocol):
    def process(self, frame: np.ndarray) -> float | None:
        """Feed 16 kHz float32 audio. Returns a score when one is computed."""
        ...

    def reset(self) -> None: ...


class OpenWakeWordDetector:
    CHUNK = 1280  # 80 ms, openWakeWord's native step

    def __init__(self, model_path: Path, melspec_path: Path, embedding_path: Path, threads: int | None = None) -> None:
        from openwakeword.model import Model

        for p in (model_path, melspec_path, embedding_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"openWakeWord model not found at {p}. Run: uv run kin download-models")
        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            melspec_model_path=str(melspec_path),
            embedding_model_path=str(embedding_path),
            ncpu=threads or 1,
        )
        self._key = next(iter(self._model.models))
        self._pending = np.zeros(0, dtype=np.int16)

    def process(self, frame: np.ndarray) -> float | None:
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        self._pending = np.concatenate([self._pending, pcm16])
        score: float | None = None
        while len(self._pending) >= self.CHUNK:
            chunk, self._pending = self._pending[: self.CHUNK], self._pending[self.CHUNK :]
            s = float(self._model.predict(chunk)[self._key])
            score = s if score is None else max(score, s)
        return score

    def reset(self) -> None:
        self._model.reset()
        self._pending = np.zeros(0, dtype=np.int16)


_GREETINGS = {"hey", "hi", "hay", "he", "hei", "hai", "a", "ay", "eh", "ok", "okay", "oh", "hello"}


def spoken_wake(text: str, phrase: str) -> str | None:
    """If a transcribed line starts with the wake phrase ("Hey Jarvis, who is my manager?"), the rest of the line
    ("who is my manager?", possibly empty); otherwise None. A fallback for when the acoustic model scored the
    phrase under its threshold (another voice or microphone) and the line went to passive transcription.

    The name may be misheard ("A Jarvis", "Hey Arviz"), but must come first, or second after a greeting:
    "I told Jarvis" or "Hey Travis" do not wake Kin."""
    from rapidfuzz import fuzz

    words = phrase.lower().split()
    if not words:
        return None
    name, greetings = words[-1], _GREETINGS | set(words[:-1])
    raw = text.split()
    toks = ["".join(c for c in w.lower() if c.isalnum()) for w in raw]

    def is_name(tok: str) -> bool:
        if len(tok) < 3:
            return False
        probe = tok if tok[0] == name[0] else name[0] + tok
        return fuzz.ratio(probe, name) >= 80

    for i in range(min(2, len(toks))):
        if is_name(toks[i]) and (i == 0 or toks[0] in greetings):
            return " ".join(raw[i + 1:]).lstrip(" ,.!?;:-").strip()
    return None
