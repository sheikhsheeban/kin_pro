"""Find person names in memories, so the gateway can scrub them from lookups.

Conservative on purpose: a capitalised word counts as a person only in a
person-shaped context. That keeps places like "London" out ("I live in London"),
so they are never scrubbed from weather lookups.
"""

from __future__ import annotations

import re

_NOT_NAMES = frozenset(
    "I The A An This That These Those It He She They We You My Your His Her Our Their What When Where Who Why How "
    "Which Do Does Did Is Are Was Were Can Could Will Would Should Okay Yes No So And But Or If Then Also Just Maybe "
    "Today Tomorrow Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June "
    "July August September October November December Hey Hi Hello Please Let Remember Don't".split()
)

_CAP = r"([A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20})?)"
# "The Library is closed" is not a person: no determiner directly before or at the start of the name.
_DETERMINERS = frozenset({"The", "A", "An", "My", "Our", "This", "That", "Your", "Their", "His", "Her"})
_NO_DET = "".join(f"(?<!{d} )" for d in sorted(_DETERMINERS | {d.lower() for d in _DETERMINERS}))
_PATTERNS = [
    # after social verbs and "with": "meeting Priya", "coffee with Priya Shah", "Call Sam"
    re.compile(
        r"\b(?i:with|meet|meeting|met|call|calling|called|email|emailing|text|texting|message|tell|told|ask|asked|"
        r"see|seeing|saw|visit|visiting|invite|invited|thank|remind|for|and)\s+" + _CAP
    ),
    # subject of an attribute: "Rahul is allergic", "Priya loves jazz"
    re.compile(
        _NO_DET + _CAP
        + r"\s+(?:is|was|has|had|likes|loves|hates|lives|works|prefers|wants|studies|moved|said|says|can't|doesn't)\b"
    ),
    # possessive: "Priya's birthday"
    re.compile(_NO_DET + _CAP + r"'s\b"),
]


def extract_names(text: str, exclude: set[str] = frozenset()) -> set[str]:
    excl = {e.lower() for e in exclude}
    found: set[str] = set()
    for pattern in _PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1)
            if name.split()[0] in _DETERMINERS:  # "The Library is closed"
                continue
            parts = [p for p in name.split() if p not in _NOT_NAMES]
            if not parts:
                continue
            for candidate in {" ".join(parts), *parts}:
                if candidate.lower() not in excl:
                    found.add(candidate)
    return found
