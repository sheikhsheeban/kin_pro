"""Reminder scheduler: speaks reminders when they fall due.

A reminder is marked fired *before* it is spoken, so a crash mid-sentence can
never make it repeat forever. Reminders missed while Kin was off are spoken on
start-up if they are within `missed_grace_h`, otherwise expired silently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from kin.core.bus import EventBus
from kin.core.events import Announcement
from kin.tools.tasks import TaskService

log = logging.getLogger(__name__)


class ReminderScheduler:
    def __init__(
        self,
        bus: EventBus,
        tasks: TaskService,
        speak: Callable[[str], Awaitable[None]],
        *,
        check_s: float = 1.0,
        missed_grace_h: float = 12.0,
    ) -> None:
        self._bus = bus
        self._tasks = tasks
        self._speak = speak
        self.check_s = check_s
        self.grace_s = missed_grace_h * 3600
        self._task: asyncio.Task | None = None
        self._first = True

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="reminders")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def tick(self) -> int:
        """Fire everything that is due. Returns how many reminders were spoken."""
        now = time.time()
        spoken = 0
        for task in await self._tasks.due(now):
            late = now - (task.due_ts or now)
            if late > self.grace_s:
                await self._tasks.mark_fired(task, expired=True)
                continue
            await self._tasks.mark_fired(task)
            text = f"{'Earlier reminder' if self._first and late > 60 else 'Reminder'}: {task.text}."
            self._bus.publish(Announcement(text=text, kind="reminder"))
            try:
                await self._speak(text)
            except Exception:
                log.exception("could not speak reminder %s", task.id)
            spoken += 1
        self._first = False
        return spoken

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("reminder check failed")
            await asyncio.sleep(self.check_s)
