"""Known people and misheard names: "Pria" in a new fact becomes "Priya" if Priya is known.

Matching uses rapidfuzz's ratio against every known name and alias. The default threshold (80)
merges Pria/Priya (89) and Raul/Rahul (89) but keeps Priya and Priyanka (77) apart. Each spelling
that was corrected is recorded as an alias, so the people list in the UI shows what was heard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from kin.brain.names import _NOT_NAMES
from kin.memory.store import Person

_CAPWORD = re.compile(r"\b([A-Z][a-z]{2,20})(?='s\b|\b)")
_NOT = {w.lower() for w in _NOT_NAMES} | {"owner", "user", "someone", "dr", "mr", "mrs", "ms", "mum", "mom", "dad"}


@dataclass
class NameIndex:
    people: list[Person]
    threshold: float = 80.0
    _variants: dict[str, str] = field(default_factory=dict)  # lower spelling -> canonical name

    def __post_init__(self) -> None:
        for p in self.people:
            self._variants[p.name.lower()] = p.name
            for a in p.aliases:
                self._variants.setdefault(a.lower(), p.name)

    def canonical(self, name: str) -> str | None:
        """The known person this spelling refers to, or None."""
        low = name.strip().lower()
        if not low or low in _NOT:
            return None
        if low in self._variants:
            return self._variants[low]
        if len(low) < 3:
            return None
        best, score = None, 0.0
        for variant, canon in self._variants.items():
            s = fuzz.ratio(low, variant)
            if s > score:
                best, score = canon, s
        return best if score >= self.threshold else None

    def normalise_text(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace near-miss spellings of known names. Returns the text and {heard: canonical}."""
        fixes: dict[str, str] = {}

        def sub(m: re.Match) -> str:
            word = m.group(1)
            canon = self.canonical(word)
            if canon is None or canon == word or " " in canon:
                return word
            fixes[word] = canon
            return canon

        return _CAPWORD.sub(sub, text), fixes

    def normalise_names(self, names: list[str]) -> tuple[list[str], dict[str, str]]:
        out, fixes = [], {}
        for n in names:
            canon = self.canonical(n)
            if canon is not None and canon != n:
                fixes[n] = canon
            out.append(canon or n)
        return list(dict.fromkeys(out)), fixes
