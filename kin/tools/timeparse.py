"""Parse spoken times for reminders: "at 5", "at 5:30 pm", "in 10 minutes", "tomorrow at 9", "on Friday".

Returns the due datetime and the text with the time phrase removed, so
"email the professor at 5" -> (17:00 today, "email the professor").

Ambiguous clock times without am/pm: 1-6 o'clock means afternoon/evening
(nobody means 5 am by "at 5"); otherwise the next occurrence is used.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import NamedTuple


class When(NamedTuple):
    due: datetime | None
    rest: str  # the text with the time phrase removed
    relative: bool = False  # "in 10 minutes" (confirm relatively) vs "at 5" (confirm with the clock time)

_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty five": 45, "fifty": 50, "sixty": 60, "a": 1, "an": 1, "a couple of": 2, "a few": 3, "half an": 0.5,
}
_NUMW = "|".join(sorted(map(re.escape, _NUM), key=len, reverse=True))
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PARTS = {"morning": 9, "afternoon": 15, "evening": 18, "tonight": 20, "night": 20, "noon": 12, "midday": 12, "lunchtime": 12}

_IN = re.compile(rf"\b(?:in|after)\s+(\d+(?:\.\d+)?|{_NUMW})\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", re.I)
_HALF_HOUR = re.compile(r"\bin\s+half\s+an\s+hour\b", re.I)
_AT = re.compile(
    rf"\b(?:at|by)\s+(\d{{1,2}}|{_NUMW})(?:[:.](\d{{2}}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?|o'?clock)?(?=\W|$)", re.I
)
_AT_WORD = re.compile(r"\b(?:at\s+)?(noon|midday|midnight|lunchtime)\b", re.I)
_DAY = re.compile(rf"\b(today|tonight|tomorrow|(?:on\s+)?(?:next\s+)?({'|'.join(_WEEKDAYS)}))\b", re.I)
_PART = re.compile(r"\b(?:this|in the|tomorrow)\s+(morning|afternoon|evening|night)\b", re.I)


def _num(token: str) -> float:
    t = token.lower()
    return float(t) if re.fullmatch(r"\d+(?:\.\d+)?", t) else float(_NUM[t])


def _strip(text: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(text[last:a])
        last = max(last, b)
    out.append(text[last:])
    cleaned = re.sub(r"\s{2,}", " ", "".join(out))
    return re.sub(r"\s+([,.!?])", r"\1", cleaned).strip(" ,.")


def parse_when(text: str, now: datetime) -> When:
    spans: list[tuple[int, int]] = []

    m = _HALF_HOUR.search(text)
    if m:
        return When(now + timedelta(minutes=30), _strip(text, [m.span()]), True)
    m = _IN.search(text)
    if m:
        n, unit = _num(m.group(1)), m.group(2).lower()
        secs = n * (1 if unit.startswith("s") else 60 if unit.startswith("m") else 3600)
        return When(now + timedelta(seconds=secs), _strip(text, [m.span()]), True)

    day_offset: int | None = None
    hint_pm: bool | None = None
    dm = _DAY.search(text)
    if dm:
        spans.append(dm.span())
        word = dm.group(1).lower()
        if word == "tomorrow":
            day_offset = 1
        elif word in ("today", "tonight"):
            day_offset = 0
            hint_pm = True if word == "tonight" else None
        else:
            wd = _WEEKDAYS.index(dm.group(2).lower())
            day_offset = (wd - now.weekday()) % 7 or 7

    hour: int | None = None
    minute = 0
    am = _AT.search(text)
    if am:
        spans.append(am.span())
        hour = int(_num(am.group(1)))
        minute = int(am.group(2) or 0)
        suffix = (am.group(3) or "").lower().replace(".", "").replace(" ", "")
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        elif suffix in ("", "o'clock", "oclock") and hour < 12:
            pm = _PART.search(text)
            if pm and pm.group(1).lower() in ("afternoon", "evening", "night") or hint_pm:
                hour += 12
            elif not (pm and pm.group(1).lower() == "morning") and 1 <= hour <= 6:
                hour += 12
            elif not pm and day_offset in (None, 0) and now.replace(hour=hour, minute=minute) <= now:
                hour += 12  # "at 9" said at 10 am means 9 pm
    else:
        wm = _AT_WORD.search(text)
        if wm:
            spans.append(wm.span())
            hour = 0 if wm.group(1).lower() == "midnight" else 12
            if hour == 0 and day_offset is None:
                day_offset = 1
    pm = _PART.search(text)
    if pm:
        spans.append(pm.span())
        if hour is None:
            hour = _PARTS[pm.group(1).lower()]
            if pm.group(0).lower().startswith("tomorrow"):
                day_offset = 1
    if hour is None and dm and dm.group(1).lower() == "tonight":
        hour = 20
    if hour is None and day_offset is not None:
        hour = 9  # "tomorrow" alone: morning
    if hour is None:
        return When(None, text.strip(" ,."))

    hour = min(hour, 23)
    due = (now + timedelta(days=day_offset or 0)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_offset is None and due <= now:
        due += timedelta(days=1)
    return When(due, _strip(text, spans))


def describe(due: datetime, now: datetime, relative: bool = False) -> str:
    """'at 5 PM today', 'at 5:30 PM tomorrow', 'in 10 minutes', 'at 9 AM on Friday'.

    Relative wording only when the user spoke relatively ("in 10 minutes").
    """
    delta = (due - now).total_seconds()
    if relative and delta < 3600:
        mins = max(1, round(delta / 60))
        if delta < 90:
            return f"in {max(1, round(delta))} seconds"
        return f"in {mins} minutes"
    t = due.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")
    days = (due.date() - now.date()).days
    if days == 0:
        return f"at {t} today"
    if days == 1:
        return f"at {t} tomorrow"
    return f"at {t} on {due.strftime('%A')}"


def parse_due(due: str | None, now: datetime) -> When:
    """The `due` argument of an add_task tool call -> When (rest is unused here).

    Spoken forms ("at 5", "in 10 minutes", "tomorrow at 9") go through `parse_when`, which
    knows that "at 5" means 5 PM; anything else ("17:00", "October 12", "next week at noon")
    goes to dateparser, relative to `now` in local time and preferring the future.
    """
    text = (due or "").strip()
    if not text or text.lower() in ("none", "null", "no", "never"):
        return When(None, "")
    spoken = parse_when(text, now)
    if spoken.due is not None:
        return When(spoken.due, "", spoken.relative)
    if re.fullmatch(r"\d{1,2}", text):  # a bare hour, as in due="5"
        return When(parse_when(f"at {text}", now).due, "", False)
    import dateparser

    parsed = dateparser.parse(
        text, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now, "RETURN_AS_TIMEZONE_AWARE": False}
    )
    if parsed is None:
        return When(None, "")
    if parsed.time() == datetime.min.time() and not re.search(r"\d:\d|(?:am|pm|midnight)", text, re.I):
        parsed = parsed.replace(hour=9)  # a date without a time: morning, like "on Friday"
    return When(parsed, "", False) if parsed > now else When(None, "")
