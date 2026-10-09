"""Single-thread executors for blocking model work.

Each heavy component (LLM, memory, TTS) owns one worker thread. Native model
handles are created and used on that thread only, calls are serialised in
submission order, and the asyncio loop never blocks on inference.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")

_DONE = object()


class ModelWorker:
    def __init__(self, name: str) -> None:
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"kin-{name}")

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    async def stream(self, gen_fn: Callable[..., Iterator[T]], *args: Any, **kwargs: Any) -> AsyncIterator[T]:
        """Run a blocking generator on the worker thread and yield its items asynchronously."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()

        def pump() -> None:
            try:
                for item in gen_fn(*args, **kwargs):
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the loop side
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _DONE)

        fut = loop.run_in_executor(self._executor, pump)
        try:
            while True:
                item = await q.get()
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            await asyncio.shield(fut)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
