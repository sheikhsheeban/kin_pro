"""Load the full Kin stack for a profile in a fresh process, answer one query, print RAM as JSON.

Run by tests/test_brain_models.py::test_pi5_profile_stays_under_ram_target in a subprocess,
so models loaded by other tests in the pytest process cannot distort the measurement.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from kin.app import KinApp
from kin.audio import build_engines
from kin.brain import build_brain
from kin.config import load_config
from kin.core.bus import EventBus


class _NoMic:
    active = closed = False

    def __init__(self, **_kw) -> None:
        pass

    def start(self) -> None:
        self.active = True

    def abort(self) -> None:
        self.active = False

    def close(self) -> None:
        self.closed = True


async def main(profile: str) -> dict:
    cfg = load_config(profile)
    cfg.data_dir = Path(tempfile.mkdtemp(prefix="kin-ram-"))
    cfg.tts.sink = "null"
    cfg.gateway.enabled = False  # a separate process with its own (small) footprint
    app = KinApp(cfg, bus=EventBus(), engines_factory=build_engines, stream_factory=_NoMic,
                 device_namer=lambda _d: "none", brain_factory=build_brain)
    await app.start()
    try:
        await app.wait_brain()
        app.pipeline.engines.query_stt  # noqa: B018 - count the lazily loaded model too
        await app.brain.memory.remember("I'm meeting Priya at 4 pm at the library cafe.", "commitment", 0.9)
        reply = await app.brain.handle_query("When am I meeting Priya?")
        info = app.monitor._proc.memory_full_info()
        return {
            "rss_mb": round(info.rss / 2**20),
            "uss_mb": round(info.uss / 2**20),
            "outcome": reply.outcome,
            "reply": reply.text,
            "components": dict(app.monitor.components),  # copied: stop() clears it
        }
    finally:
        await app.stop()


if __name__ == "__main__":
    print("RESULT " + json.dumps(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "pi5"))))
