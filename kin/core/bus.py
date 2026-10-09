"""Asyncio event bus with typed events.

Each subscriber gets its own queue and dispatch task, so a slow consumer
(e.g. a lagging browser tab) never blocks the audio pipeline or other
subscribers. Events published to one subscriber arrive in publish order.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from kin.core.events import Event

log = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)
Handler = Callable[[Any], Awaitable[None] | None]


@dataclass(eq=False)
class Subscription:
    event_type: type[Event]
    handler: Handler
    name: str
    queue: asyncio.Queue[Event] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        for sub in self._subs:
            self._start_task(sub)

    async def stop(self) -> None:
        self._closed = True
        tasks = [s.task for s in self._subs if s.task]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def subscribe(self, event_type: type[E], handler: Callable[[E], Any], name: str | None = None) -> Subscription:
        """Call `handler` for every event that is an instance of `event_type`.

        Subscribe to `Event` to receive everything.
        """
        sub = Subscription(event_type, handler, name or getattr(handler, "__qualname__", repr(handler)))
        self._subs.append(sub)
        if self._loop is not None:
            self._start_task(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)
        if sub.task:
            sub.task.cancel()

    def publish(self, event: Event) -> None:
        """Publish from the event loop thread. Never blocks."""
        if self._closed:
            return
        for sub in self._subs:
            if isinstance(event, sub.event_type):
                sub.queue.put_nowait(event)

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from any thread (audio callback, perception worker)."""
        loop = self._loop
        if loop is None or loop.is_closed() or self._closed:
            return
        try:
            loop.call_soon_threadsafe(self.publish, event)
        except RuntimeError:  # loop shutting down
            pass

    async def wait_for(
        self,
        event_type: type[E],
        predicate: Callable[[E], bool] = lambda _e: True,
        timeout: float | None = 5.0,
    ) -> E:
        """Wait for the next matching event (mainly for tests and orchestration)."""
        fut: asyncio.Future[E] = asyncio.get_running_loop().create_future()

        def on_event(e: E) -> None:
            if not fut.done() and predicate(e):
                fut.set_result(e)

        sub = self.subscribe(event_type, on_event, name=f"wait_for:{event_type.__name__}")
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.unsubscribe(sub)

    def _start_task(self, sub: Subscription) -> None:
        if sub.task is None or sub.task.done():
            sub.task = asyncio.get_running_loop().create_task(self._dispatch(sub), name=f"bus:{sub.name}")

    async def _dispatch(self, sub: Subscription) -> None:
        while True:
            event = await sub.queue.get()
            try:
                result = sub.handler(event)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("bus subscriber %s failed on %s", sub.name, event.type)
