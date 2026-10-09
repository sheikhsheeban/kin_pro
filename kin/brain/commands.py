"""Safety guard: the only regex intent left. Destructive commands are recognised here, never by the LLM.

  "forget the last ten minutes" / "forget the last half an hour"  -> forget_span
  "forget that" / "forget what I just said"                        -> forget_last
  "cancel the reminder about the oven"                             -> cancel_reminder
  "forget everything about Rahul"                                  -> forget_person

Patterns are anchored to the start of the cleaned query (after the wake word and an optional
"please" / "can you"), and anything negated or habitual is refused: "don't forget the last
5 minutes of the meeting" and "I always forget what we said in the last hour" are not commands.
None of these execute here: the orchestrator asks for confirmation first.

Everything else goes to Gemma 4 E2B with tools (see tools_schema.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_WAKE_PREFIX = re.compile(r"^\s*(?:(?:hey|hi|ok|okay)[\s,]+)?(?:jarvis|kin)\b[\s,.!?]*", re.I)
_LEAD_HEY = re.compile(r"^\s*(?:hey|ok|okay)[\s,]+(?=\w)", re.I)

# --- durations -------------------------------------------------------------------------

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_FUZZY = {"a": 1, "an": 1, "a couple of": 2, "couple of": 2, "a few": 3, "few": 3}
_NUM_WORD = "|".join(sorted([*_ONES, *_TENS], key=len, reverse=True))
_QTY = (
    rf"(?:\d+(?:\.\d+)?|(?:{'|'.join(_TENS)})(?:[\s-](?:{'|'.join(k for k in _ONES if k not in ('zero', 'ten'))}))?"
    rf"|{_NUM_WORD}|a couple of|couple of|a few|few|an|a)"
)
_UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)"
_DURATION = re.compile(
    rf"^(?:(?P<half>half an? hour|half hour)|(?P<qty>{_QTY})\s*(?P<and_half>and a half\s+)?(?P<unit>{_UNIT})"
    rf"(?:\s+and a half)?|(?P<bare>{_UNIT}))$",
    re.I,
)


def _number(token: str) -> float:
    t = token.lower().strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    if t in _FUZZY:
        return float(_FUZZY[t])
    parts = re.split(r"[\s-]+", t)
    total = 0
    for p in parts:
        if p in _TENS:
            total += _TENS[p]
        elif p in _ONES:
            total += _ONES[p]
        else:
            raise ValueError(f"not a number: {token!r}")
    return float(total)


def parse_duration(text: str) -> float | None:
    """Minutes in a spoken duration: "ten minutes", "half an hour", "an hour", "eleven minutes",
    "90 mins", "1.5 hours", "an hour and a half", "twenty five minutes". None if it is not one."""
    m = _DURATION.match(text.strip().rstrip(".!?").strip())
    if not m:
        return None
    if m.group("half"):
        return 30.0
    if m.group("bare"):  # "the last hour"
        qty, unit = 1.0, m.group("bare")
    else:
        qty, unit = _number(m.group("qty")), m.group("unit")
    u = unit.lower()
    per = 1 / 60 if u.startswith("s") else 60.0 if u.startswith("h") else 1.0
    minutes = qty * per
    if m.group("and_half") or text.strip().rstrip(".!?").lower().endswith("and a half"):
        minutes += 0.5 * per
    return minutes if minutes > 0 else None


# --- the guard -------------------------------------------------------------------------

SafeKind = Literal["forget_span", "forget_last", "cancel_reminder", "forget_person"]


@dataclass(frozen=True)
class SafeCommand:
    kind: SafeKind
    minutes: float | None = None
    text: str | None = None


_POLITE = r"(?:(?:please|can you|could you|would you|will you|kin|now)[\s,]+)*"
# Negated or habitual: these talk *about* forgetting, they do not ask Kin to forget.
_NEGATED = re.compile(
    r"\b(?:don'?t|do not|never|always|won'?t|will not|didn'?t|did not|can'?t|cannot|shouldn'?t|mustn'?t|not)\b"
    r"|\bi\s+(?:forget|forgot|keep forgetting|tend to forget|often forget|sometimes forget)\b",
    re.I,
)
_FORGET_SPAN = re.compile(
    rf"^{_POLITE}(?:forget|delete|erase|wipe)\s+(?:everything|all|what\s+(?:i|we)\s+said|anything)?\s*"
    rf"(?:from\s+|in\s+|over\s+|during\s+)?(?:the\s+)?(?:last|past)\s+(?P<dur>.+?)$",
    re.I,
)
_FORGET_LAST = re.compile(
    rf"^{_POLITE}forget\s+(?:that|this|it|the last thing(?:\s+i\s+said)?|what\s+i\s+just\s+said|my last memory)$",
    re.I,
)
_CANCEL = re.compile(
    rf"^{_POLITE}(?:cancel|delete|remove)\s+(?:the\s+|my\s+)?reminder\s*(?:to|about|for|that)?\s+(?P<what>.+)$",
    re.I,
)


_FORGET_PERSON = re.compile(
    rf"^{_POLITE}(?:forget|delete|erase|wipe)\s+(?:everything|all|anything|all (?:the )?memories|what you know)\s+"
    rf"(?:about|of|on|regarding)\s+(?P<who>[A-Za-z][A-Za-z' -]{{1,40}})$",
    re.I,
)


def clean_query(text: str) -> str:
    """Drop a leading wake word ("hey Kin", "okay Jarvis") or a bare "hey" that leaked into the transcript."""
    t = _WAKE_PREFIX.sub("", text).strip()
    return _LEAD_HEY.sub("", t).strip()


def parse_safe_command(text: str) -> SafeCommand | None:
    """A destructive command, or None. Always run on the cleaned query; never executes anything."""
    t = clean_query(text).strip().rstrip(".!?").strip()
    if not t or _NEGATED.search(t):
        return None
    m = _FORGET_PERSON.match(t)
    if m:
        who = re.sub(r"^(?:my|the)\s+", "", m.group("who").strip(), flags=re.I)
        return SafeCommand("forget_person", text=who) if who else None
    m = _FORGET_SPAN.match(t)
    if m:
        minutes = parse_duration(m.group("dur"))
        return SafeCommand("forget_span", minutes=minutes) if minutes else None
    if _FORGET_LAST.match(t):
        return SafeCommand("forget_last")
    m = _CANCEL.match(t)
    if m:
        return SafeCommand("cancel_reminder", text=m.group("what").strip())
    return None


_YES = re.compile(r"^(?:yes|yeah|yep|confirm(?:ed)?|do it|go ahead|delete (?:it|them)|yes,? (?:confirm|delete|please)|ok(?:ay)?,? confirm)$", re.I)
_NO = re.compile(r"^(?:no|nope|cancel|stop|don'?t|keep (?:it|them)|never ?mind)$", re.I)


def confirmation_answer(text: str) -> bool | None:
    """True for "confirm" / "yes", False for "no" / "cancel", None for anything else."""
    t = clean_query(text).strip().rstrip(".!?").strip()
    t = re.sub(r"\s+please$", "", t, flags=re.I)
    if _YES.match(t):
        return True
    if _NO.match(t):
        return False
    return None
