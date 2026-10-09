"""Child process for the crash-safety tests: does one kind of work on a database and stops at a precise point
(prints AT and sleeps), where the test kills it with SIGKILL / TerminateProcess.

  python -m tests.crash_child <mode> <db> <key> [<folder>]
  modes: write, backup-copy, backup-file, extract, migrate
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np


def at() -> None:
    print("AT", flush=True)
    time.sleep(120)  # killed here
    os._exit(3)


def write(db: Path, key: str) -> None:
    from kin.memory.store import MemoryStore

    s = MemoryStore(db, key, 16)
    for i in range(3):
        s.add_fact(f"Committed fact number {i}.", "fact", np.ones(16, np.float32), confidence=0.9)
    real_log = s._log

    def log_then_die(*a, **k):  # inside add_fact's transaction: the fact row, vector and FTS row are written
        real_log(*a, **k)
        at()

    s._log = log_then_die
    s.add_fact("Half-written fact.", "fact", np.ones(16, np.float32), confidence=0.9)


def backup(db: Path, key: str, folder: Path, where: str) -> None:
    import kin.memory.backup as bk
    from kin.memory.store import MemoryStore

    s = MemoryStore(db, key, 16)
    if where == "copy":
        real = bk._check_copy
        bk._check_copy = lambda *a: (real(*a), at())
    else:
        os.fsync = lambda _fd: at()  # the .part file is written, not yet renamed to .kinbak
    bk.write_backup(s, folder, "lantern river 42 pebbles")


def extract(db: Path, key: str) -> None:
    import json

    from kin.brain.orchestrator import Brain
    from tests.conftest import FakeLLM, build_fake_app, make_fake_brain_factory

    llm = FakeLLM()
    llm.extract_replies = [json.dumps([
        {"text": "Rahul is allergic to peanuts.", "kind": "fact", "subject": "Rahul", "people": ["Rahul"], "event_time": None, "lines": [1]},
        {"text": "The door code is 4512.", "kind": "fact", "subject": None, "people": [], "event_time": None, "lines": [2]},
    ])]
    real = Brain._store_fact
    stored = []

    async def store_then_die(self, fact, eid):
        out = await real(self, fact, eid)
        stored.append(out)
        if len(stored) == 1:  # one fact stored, the episode not yet marked done
            at()
        return out

    Brain._store_fact = store_then_die

    async def main() -> None:
        h = build_fake_app(brain_factory=make_fake_brain_factory(db.parent, llm=llm, key=key))
        await h.app.start()
        await h.app.wait_brain()
        brain = h.app.brain
        await brain._ingest("Rahul's allergic to peanuts.", ts=time.time() - 300)
        await brain._ingest("The door code is 4512.", ts=time.time() - 290)
        await brain.flush()

    asyncio.run(main())


def migrate(db: Path, key: str) -> None:
    from kin.memory.store import MemoryStore

    real = MemoryStore._to_v6

    def step_then_die(self, stored_dim):
        real(self, stored_dim)
        at()

    MemoryStore._to_v6 = step_then_die
    MemoryStore(db, key, 256)


if __name__ == "__main__":
    mode, db, key = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if mode == "write":
        write(db, key)
    elif mode.startswith("backup-"):
        backup(db, key, Path(sys.argv[4]), mode.split("-", 1)[1])
    elif mode == "extract":
        extract(db, key)
    elif mode == "migrate":
        migrate(db, key)
    print("DONE", flush=True)  # never reached when killed
