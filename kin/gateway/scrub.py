"""Scrub personal names from lookup params, without the gateway ever learning the names.

The brain sends a salted, keyed hash of every person name it finds in the
memory store. The gateway hashes each word (and adjacent word pair) of the
params with the same salt and drops matches. The gateway is the only process
that talks to the internet, so it never holds names in plain text.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading

_WORD = re.compile(r"[\w']+")


def name_hash(salt: bytes, name: str) -> str:
    return hmac.new(salt, name.strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()[:24]


class NameScrubber:
    def __init__(self) -> None:
        self._salt = b""
        self._hashes: frozenset[str] = frozenset()
        self._lock = threading.Lock()

    def update(self, salt: bytes, hashes: list[str]) -> None:
        with self._lock:
            self._salt, self._hashes = salt, frozenset(hashes[:5000])

    @property
    def size(self) -> int:
        return len(self._hashes)

    def scrub(self, text: str) -> tuple[str, int]:
        """Remove every name (single word or two-word span) found in the denylist."""
        with self._lock:
            salt, hashes = self._salt, self._hashes
        if not hashes:
            return text, 0
        words = list(_WORD.finditer(text))
        drop: set[int] = set()
        for i, w in enumerate(words):
            if name_hash(salt, w.group()) in hashes:
                drop.add(i)
            if i + 1 < len(words) and name_hash(salt, f"{w.group()} {words[i + 1].group()}") in hashes:
                drop.update({i, i + 1})
        if not drop:
            return text, 0
        out, last = [], 0
        for i, w in enumerate(words):
            if i in drop:
                out.append(text[last : w.start()])
                last = w.end()
        out.append(text[last:])
        cleaned = re.sub(r"\s{2,}", " ", "".join(out)).strip(" ,.'")
        return cleaned, len(drop)
