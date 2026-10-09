"""Incremental sentence splitting for streaming LLM output into TTS."""

from __future__ import annotations

import re

_ABBREVIATIONS = ("mr.", "mrs.", "ms.", "dr.", "st.", "a.m.", "p.m.", "e.g.", "i.e.", "vs.", "etc.", "no.")
# A sentence ends at . ! ? (plus closing quotes/brackets) followed by whitespace and then a
# character that is not "[" -- a citation like "... cafe. [#7]" belongs to the sentence before it.
_BOUNDARY = re.compile(r"[.!?][\"')]*(?:\s*\[\s*#?\s*\d[^\]]*\])*\s+(?=[^\s\[])")


class SentenceSplitter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        start = 0
        for m in _BOUNDARY.finditer(self._buf):
            candidate = self._buf[start : m.end()].strip()
            if candidate.lower().endswith(_ABBREVIATIONS) or _ends_with_initial(candidate):
                continue
            out.append(candidate)
            start = m.end()
        self._buf = self._buf[start:]
        return out

    def flush(self) -> str | None:
        rest, self._buf = self._buf.strip(), ""
        return rest or None


def _ends_with_initial(s: str) -> bool:
    # "J. K. Rowling" style initials should not end a sentence
    return bool(re.search(r"(^|\s)[A-Z]\.$", s))
