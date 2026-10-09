"""RAM and CPU accounting for the System tab.

Per-component RAM is the process RSS growth while that component loaded
(components load one after another so the deltas do not overlap). It is an
estimate: memory-mapped weights are counted once they are touched.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import psutil


class ResourceMonitor:
    def __init__(self, ram_target_mb: int | None = None) -> None:
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # prime the CPU counter
        self.components: dict[str, dict[str, float]] = {}
        self.ram_target_mb = ram_target_mb

    def rss_mb(self) -> float:
        return self._proc.memory_info().rss / 2**20

    @contextmanager
    def measure(self, name: str):
        before, t0 = self.rss_mb(), time.monotonic()
        try:
            yield
        finally:
            self.components[name] = {
                "ram_mb": round(max(0.0, self.rss_mb() - before), 1),
                "load_s": round(time.monotonic() - t0, 2),
            }

    def forget(self, name: str) -> None:
        self.components.pop(name, None)

    def snapshot(self) -> dict:
        return {
            "rss_mb": round(self.rss_mb(), 1),
            "cpu_percent": round(self._proc.cpu_percent(None), 1),
            "system_total_mb": round(psutil.virtual_memory().total / 2**20),
            "ram_target_mb": self.ram_target_mb,
            "components": self.components,
        }
