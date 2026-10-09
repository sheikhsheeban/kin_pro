"""Turning a spoken correction into the corrected fact, without asking the LLM to rewrite anything.

  "no, it's 5 pm"                 + "Priya and you meet at the library cafe at 4 pm."  -> "... at 5 pm."
  "Rahul isn't vegan any more"    -> used as the new fact as it is (a whole sentence)
  "it's 18,500"                   + "The rent goes up to 18,000 rupees."               -> "... 18,500 rupees."
  "no"                            -> None: Kin asks for the whole corrected sentence

A changed clock time also moves the fact's event time (same day, new time).
"""

from __future__ import annotations

import re
from datetime import datetime

_LEAD = re.compile(r"^(?:no|nope|wrong|actually|sorry|correction|that'?s wrong|that is wrong|not quite|it was|it'?s|it is|"
                   r"make it|change it to|i meant|i said)[\s,.!:-]*", re.I)
_TIME = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?|o'?clock)?(?=\W|$)", re.I)
# one clock time in a fact: "4 pm", "4:30 p.m.", "16:00", or "at 4" (the "at" is kept)
_CLOCK_IN_TEXT = re.compile(r"(?P<at>\bat\s+)?\b(?P<t>\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?![a-z])"
                            r"|\d{1,2}[:.]\d{2}\b|(?<=at\s)\d{1,2}\b(?!\s*(?:st|nd|rd|th|[:.]\d)))", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_VERBISH = re.compile(r"\b(is|isn'?t|are|aren'?t|was|wasn'?t|were|has|hasn'?t|have|had|will|won'?t|does|doesn'?t|did|"
                      r"meet|meets|moved|lives|works|likes|loves|got|gets|starts|ends)\b", re.I)


def strip_lead(correction: str) -> str:
    t = correction.strip().strip(".!")
    for _ in range(3):  # "no, actually, it's 5 pm"
        t2 = _LEAD.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    return t


def _sentence(text: str) -> str:
    t = text.strip().rstrip(".!") + "."
    return t[0].upper() + t[1:]


def corrected_text(old: str, correction: str) -> str | None:
    """The fact as corrected, or None if the correction is too short to apply safely."""
    c = strip_lead(correction)
    if not c:
        return None
    if len(c.split()) >= 3 and _VERBISH.search(c):
        return _sentence(c)  # a whole new statement ("Rahul isn't vegan any more")
    if (m := _TIME.fullmatch(c.strip())) and (m.group(3) or m.group(2)) and _CLOCK_IN_TEXT.search(old):
        new = _CLOCK_IN_TEXT.sub(lambda x: (x.group("at") or "") + c.strip(), old, count=1)
        return _sentence(new) if new != old else None
    nums = _NUMBER.findall(old)
    if _NUMBER.fullmatch(c.split()[0]) and len(set(nums)) == 1:
        return _sentence(old.replace(nums[0], c.split()[0]))
    return None


def corrected_event(old_event_ts: float | None, new_text: str) -> float | None:
    """Same day, the clock time the corrected text gives (if it gives one); else the old event time."""
    if old_event_ts is None:
        return None
    m = re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", new_text, re.I) or re.search(r"\b(\d{1,2})[:.](\d{2})\b", new_text)
    if not m:
        return old_event_ts
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    ampm = (m.group(3) if m.lastindex and m.lastindex >= 3 else "") or ""
    if ampm.lower().startswith("p") and hour < 12:
        hour += 12
    if ampm.lower().startswith("a") and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return old_event_ts
    return datetime.fromtimestamp(old_event_ts).replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()
