"""Tasks and reminders, stored locally in the encrypted memory database.

A reminder is a task with a due time. Everything runs on the memory worker,
and every change is announced on the bus for the UI.
"""

from __future__ import annotations

import re
import time

from kin.core.bus import EventBus
from kin.core.events import TaskChanged
from kin.memory.service import MemoryService
from kin.memory.store import Task

_STOP = frozenset("the a an to my of for and on in at with".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP}


class TaskService:
    def __init__(self, bus: EventBus, memory: MemoryService) -> None:
        self._bus = bus
        self._memory = memory

    async def add(self, text: str, due_ts: float | None = None) -> Task:
        kind = "reminder" if due_ts is not None else "task"
        task = await self._memory.run(lambda s: s.add_task(text, kind, due_ts))
        self._publish("added", task)
        return task

    async def open(self) -> list[Task]:
        return await self._memory.run(lambda s: s.open_tasks())

    async def find(self, text: str) -> Task | None:
        """Best open task by word overlap ("mark the professor email as done")."""
        want = _words(text)
        best, score = None, 0.0
        for t in await self.open():
            have = _words(t.text)
            overlap = len(want & have) / max(1, len(have))
            if overlap > score:
                best, score = t, overlap
        return best if score >= 0.5 else None

    async def complete(self, task_id: int) -> Task | None:
        task = await self._memory.run(lambda s: s.get_task(task_id))
        if task and await self._memory.run(lambda s: s.complete_task(task_id)):
            self._publish("completed", task)
            return task
        return None

    async def delete(self, task_id: int) -> bool:
        task = await self._memory.run(lambda s: s.get_task(task_id))
        if task and await self._memory.run(lambda s: s.delete_task(task_id)):
            self._publish("deleted", task)
            return True
        return False

    async def due(self, now: float | None = None) -> list[Task]:
        return await self._memory.run(lambda s: s.due_reminders(time.time() if now is None else now))

    async def mark_fired(self, task: Task, expired: bool = False) -> None:
        await self._memory.run(lambda s: s.mark_fired(task.id, time.time()))
        self._publish("expired" if expired else "fired", task)

    def _publish(self, action: str, task: Task) -> None:
        self._bus.publish(TaskChanged(action=action, task_id=task.id, text=task.text, kind=task.kind, due_ts=task.due_ts))  # type: ignore[arg-type]
