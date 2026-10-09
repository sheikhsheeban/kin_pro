"""Fixed-size RAM ring buffer of mono float32 PCM.

The buffer is preallocated once and overwritten continuously. It has no
save, dump or serialise method on purpose: audio only ever lives here.
Positions are absolute sample indices since the buffer was created, so
readers can ask for "samples N..M" without caring about wrap-around.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, seconds: float, sample_rate: int) -> None:
        self.capacity = int(seconds * sample_rate)
        self.sample_rate = sample_rate
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._end = 0  # absolute index one past the newest sample
        self._lock = threading.Lock()

    @property
    def end(self) -> int:
        return self._end

    @property
    def start(self) -> int:
        """Oldest absolute index still held."""
        return max(0, self._end - self.capacity)

    def write(self, samples: np.ndarray) -> int:
        """Append samples, overwriting the oldest. Returns the new end index."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = len(samples)
        with self._lock:
            if n >= self.capacity:
                samples = samples[-self.capacity :]
                self._end += n - self.capacity
                n = self.capacity
            pos = self._end % self.capacity
            first = min(n, self.capacity - pos)
            self._buf[pos : pos + first] = samples[:first]
            if first < n:
                self._buf[: n - first] = samples[first:]
            self._end += n
            return self._end

    def read(self, start: int, end: int | None = None) -> np.ndarray:
        """Copy of samples [start, end). Clamped to what is still held."""
        with self._lock:
            end = self._end if end is None else min(end, self._end)
            start = max(start, self.start)
            if end <= start:
                return np.zeros(0, dtype=np.float32)
            idx = np.arange(start, end) % self.capacity
            return self._buf[idx].copy()

    def zero(self, start: int, end: int) -> None:
        """Overwrite [start, end) with silence (used once audio is transcribed)."""
        with self._lock:
            start = max(start, self.start)
            end = min(end, self._end)
            if end > start:
                self._buf[np.arange(start, end) % self.capacity] = 0.0

    def clear(self) -> None:
        """Wipe all audio (e.g. on mute)."""
        with self._lock:
            self._buf.fill(0.0)
