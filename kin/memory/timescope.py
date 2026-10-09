"""Day expressions in questions -> a time window (the retrieval Kin had before hybrid recall).

Superseded by `kin.memory.query`, which covers more phrasings and also names, the future and history.
Kept unchanged as the reference for the "before" column of scripts/eval_retrieval.py.

So "what did Mum tell me last Monday?" can find what was heard that Monday (embeddings know nothing about dates).

Past-tense questions ("did", "said", "told", ...) look back to the most recent such day; others
("when is my meeting on Friday?") look ahead to the next one. Facts heard OR happening in the
window are candidates (`MemoryStore.search(..., window=...)`).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_PAST = re.compile(r"\b(did|was|were|said|told|tell me|met|had|heard|planned|mentioned|booked|submitted|spent|bought|gave|paid)\b", re.I)
_DAY = re.compile(r"\b(?:(last|this|next|on)\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.I)


def _day(d: datetime) -> tuple[float, float]:
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def day_window(question: str, now: datetime) -> tuple[float, float] | None:
    q = question.lower()
    if re.search(r"\byesterday\b", q):
        return _day(now - timedelta(days=1))
    if re.search(r"\b(today|this morning|this afternoon|tonight|earlier)\b", q):
        return _day(now)
    if re.search(r"\blast week\b", q):
        start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), now.timestamp()
    if re.search(r"\b(this|next) week\b", q):
        return now.timestamp(), (now + timedelta(days=7)).timestamp()
    m = _DAY.search(q)
    if not m:
        return None
    which, wd = (m.group(1) or "").lower(), _WEEKDAYS.index(m.group(2).lower())
    past = which == "last" or (which != "next" and bool(_PAST.search(q)))
    if past:
        back = (now.weekday() - wd) % 7 or 7
        return _day(now - timedelta(days=back))
    ahead = (wd - now.weekday()) % 7 or 7
    return _day(now + timedelta(days=ahead))
