"""Prompt building under a hard token budget.

The system prompt is fixed (so it can be KV-cached in Phase 4). Retrieved
memories are added best-first until the budget is reached; prefill time on
the Pi is roughly linear in prompt length (architecture §8).

Each memory is one line: its id, then what a 2B model needs to answer about time and change
(what it is about, when it happens, when it was heard, whether it was replaced), then the text:
    [#12] (about Priya; happens Wed 30 Sep, 17:00; heard Tue 29 Sep, 08:45) Priya and you meet ...
Two active memories that disagree (neither replaced the other) get a note telling the model to say both.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import resources

from kin.memory.store import Hit

MAX_MEMORY_CHARS = 240


def load_prompt(name: str) -> str:
    return resources.files("kin.brain.prompts").joinpath(f"{name}.txt").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class BuiltPrompt:
    system: str
    user: str
    memory_ids: tuple[int, ...]
    tokens: int


def _when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%a %d %b, %H:%M")


def format_memory(hit: Hit) -> str:
    """[#12] (about Priya; happens Wed 30 Sep, 16:00; heard Tue 29 Sep, 08:45) Priya and you meet ..."""
    f = hit.fact
    text = f.text
    if len(text) > MAX_MEMORY_CHARS:
        text = text[: MAX_MEMORY_CHARS - 1].rstrip() + "…"
    if f.source == "legacy":
        return f"[#{f.id}] ({_when(f.heard_ts)}) {text}"  # old memories: exactly as before
    meta = []
    # "about X" only when the text does not name X already: on misheard names ("about Raul") it made E2B
    # refuse a question about "Rahul" even with the answer in the text
    if f.subject and f.subject != "owner" and f.subject.lower() not in text.lower():
        meta.append(f"about {f.subject.replace('(', '').replace(')', '')}")
    if f.event_ts:
        meta.append(f"happens {_when(f.event_ts)}")
    meta.append(f"heard {_when(f.heard_ts)}")
    if f.status != "active":
        meta.append(f"OUTDATED, replaced by #{f.superseded_by}" if f.superseded_by else "OUTDATED")
    return f"[#{f.id}] ({'; '.join(meta)}) {text}"


def conflict_note(pairs: Sequence[tuple[int, int]], ids: Sequence[int]) -> str:
    shown = [(a, b) for a, b in pairs if a in ids and b in ids]
    if not shown:
        return ""
    lines = [f"[#{a}] and [#{b}] disagree and neither replaced the other: give both, each with when it was heard, "
             f"older first, citing both." for a, b in shown]
    return "\nNote: " + "\nNote: ".join(lines)


class PromptBuilder:
    def __init__(self, count_tokens: Callable[[str], int], budget: int) -> None:
        self._count = count_tokens
        self.budget = budget
        self.recall_system = load_prompt("recall_system")

    def recall(
        self, question: str, hits: Sequence[Hit], now: float | None = None, conflicts: Sequence[tuple[int, int]] = (),
        names: dict[str, str] | None = None,
    ) -> BuiltPrompt:
        """Best memory first. `conflicts`: pairs of offered ids that disagree (see memory.retrieval.conflicts).
        `names`: {spelling in the question: spelling in memory} for names heard differently ("Rahul" / "Raul")."""
        now_line = f"Now: {datetime.fromtimestamp(now or time.time()).strftime('%A %d %B %Y, %H:%M')}"
        same = "; ".join(f"{a} in the question is {b} in the memories" for a, b in (names or {}).items() if a != b)
        tail = f"\n\nQuestion: {question}" + (f"\n(Same person, heard differently: {same}.)" if same else "")
        used = self._count(self.recall_system) + self._count(now_line + "\nMemories:\n" + tail)
        lines: list[str] = []
        ids: list[int] = []
        for hit in hits:
            line = format_memory(hit)
            cost = self._count(line + "\n")
            if lines and used + cost > self.budget:
                break  # always keep at least the best memory
            lines.append(line)
            ids.append(hit.fact.id)
            used += cost
        note = conflict_note(conflicts, ids)
        used += self._count(note) if note else 0
        user = f"{now_line}\nMemories:\n" + "\n".join(lines) + note + tail
        return BuiltPrompt(self.recall_system, user, tuple(ids), used)
