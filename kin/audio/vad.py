"""Voice activity detection.

Silero VAD v5 run directly through onnxruntime. We avoid the `silero-vad`
pip package because it pulls in PyTorch, which does not fit the Pi budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class VoiceActivityDetector(Protocol):
    def speech_prob(self, frame: np.ndarray) -> float:
        """Probability that a 512-sample, 16 kHz frame contains speech."""
        ...

    def reset(self) -> None: ...


class SileroVAD:
    SAMPLE_RATE = 16000
    WINDOW = 512
    CONTEXT = 64  # v5 expects the last 64 samples of the previous window prepended

    def __init__(self, model_path: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Silero VAD model not found at {model_path}. Run: uv run kin download-models")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(self.SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def speech_prob(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.WINDOW:
            raise ValueError(f"SileroVAD expects {self.WINDOW}-sample frames, got {x.shape[1]}")
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -self.CONTEXT :]
        return float(out[0][0])
