"""One queue for every use of the LLM, with priorities.

Active queries (the user is waiting) go before background work (tagging overheard
speech, pre-warming the next session). A holder keeps the LLM for a whole job (all
passes of one query), so passes of different jobs never interleave in the KV cache.

Background work is done in small steps (one tagging question, one warm-up), so a question
waits for at most one of them. Steps are not interrupted: LiteRT-LM cannot cancel a prefill.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

ACTIVE = 0
BACKGROUND = 1


class LlmQueue:
    def __init__(self) -> None:
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()
        self._holder: int | None = None  # priority of the current holder
        self._active_waiting = 0
        self.idle = asyncio.Event()  # set when no active job holds or waits for the LLM
        self.idle.set()

    @property
    def active_busy(self) -> bool:
        return self._active_waiting > 0 or self._holder == ACTIVE

    @property
    def queued(self) -> int:
        return len(self._waiters)

    @asynccontextmanager
    async def slot(self, priority: int) -> AsyncIterator[None]:
        if priority == ACTIVE:
            self._active_waiting += 1
            self._update_idle()
        try:
            if self._holder is None and not self._waiters:
                self._holder = priority
            else:
                fut = asyncio.get_running_loop().create_future()
                heapq.heappush(self._waiters, (priority, next(self._seq), fut))
                try:
                    await fut
                except asyncio.CancelledError:
                    if fut.done() and not fut.cancelled():
                        self._release()  # we were handed the slot just as we were cancelled
                    else:
                        self._waiters = [w for w in self._waiters if w[2] is not fut]
                        heapq.heapify(self._waiters)
                    raise
        finally:
            if priority == ACTIVE:
                self._active_waiting -= 1
                self._update_idle()
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        while self._waiters:
            priority, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                self._holder = priority
                fut.set_result(None)
                break
        else:
            self._holder = None
        self._update_idle()

    def _update_idle(self) -> None:
        if self.active_busy:
            self.idle.clear()
        else:
            self.idle.set()
