"""Confidence gate: nothing reaches the speaker unless it is grounded.

Checks, in order:
  1. Retrieval floor: for recall, the best memory must be at least `similarity_floor`
     similar to the question, otherwise Kin abstains (and the UI shows the closest match).
  2. Per-sentence sources: each sentence must cite a memory id that was actually
     retrieved ([#7]) or a lookup that actually returned data this turn ([web]),
     or be an honest abstention. Anything else is blocked.
  3. Timeout: generation is cancelled after `timeout_s` (enforced by the orchestrator).

Sentences are checked as they stream, before TTS, so a blocked sentence is never spoken.
Small models are sloppy with the format ("[#web]", "[#7, [web]]"); all of those parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# A bracket group (one level of nesting allowed) whose every token is a source marker.
_GROUP = re.compile(r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]")
_TOKEN = re.compile(r"#?\s*(\d+|web|online|lookup|weather|search|news)\b", re.I)
_WEB_WORDS = {"web", "online", "lookup", "weather", "search", "news"}
_ABSTAIN = re.compile(
    r"\b(i (don'?t|do not) (remember|recall|know)|i'?m not sure|i have no (memory|record)|"
    r"i (can'?t|cannot|couldn'?t) (check|look|get|find))\b",
    re.I,
)

ABSTAIN_REPLY = "I don't remember that."
BLOCKED_REPLY = "I'm not sure. I couldn't find that in what I remember."
TIMEOUT_REPLY = "Sorry, that is taking too long, so I'll stop there."
ERROR_REPLY = "Sorry, something went wrong while I was thinking."


@dataclass(frozen=True)
class Sources:
    ids: tuple[int, ...] = ()
    web: bool = False

    def __bool__(self) -> bool:
        return bool(self.ids) or self.web


@dataclass(frozen=True)
class Verdict:
    speakable: bool
    spoken: str  # what TTS says: source markers removed
    citations: tuple[int, ...]
    reason: Literal["cited", "abstention", "uncited", "unknown_citation", "empty"]
    web: bool = False


def _is_source_group(group: str) -> bool:
    inner = re.sub(r"[\[\]]", " ", group)
    leftover = _TOKEN.sub("", inner)
    return bool(_TOKEN.search(inner)) and not re.sub(r"[\s,;&]|and", "", leftover)


def sources_in(sentence: str) -> Sources:
    ids: list[int] = []
    web = False
    for m in _GROUP.finditer(sentence):
        if not _is_source_group(m.group()):
            continue
        for t in _TOKEN.findall(m.group()):
            if t.isdigit():
                ids.append(int(t))
            elif t.lower() in _WEB_WORDS:
                web = True
    return Sources(tuple(dict.fromkeys(ids)), web)


def citations_in(sentence: str) -> tuple[int, ...]:
    return sources_in(sentence).ids


_EMPTY_MARKER = re.compile(r"\[\s*#?\s*\]")  # "I don't remember that. [#]": a marker with no id, never spoken


def strip_citations(sentence: str) -> str:
    s = _EMPTY_MARKER.sub("", sentence)
    s = _GROUP.sub(lambda m: "" if _is_source_group(m.group()) else m.group(), s)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def is_abstention(sentence: str) -> bool:
    return bool(_ABSTAIN.search(sentence))


class Gate:
    def __init__(self, similarity_floor: float) -> None:
        self.similarity_floor = similarity_floor

    def retrieval_ok(self, best_similarity: float | None) -> bool:
        return best_similarity is not None and best_similarity >= self.similarity_floor

    def check(self, sentence: str, allowed_ids: set[int], *, web_ok: bool = False) -> Verdict:
        spoken = strip_citations(sentence)
        if not spoken:
            return Verdict(False, "", (), "empty")
        src = sources_in(sentence)
        if (src.ids and not set(src.ids) <= allowed_ids) or (src.web and not web_ok):
            return Verdict(False, spoken, src.ids, "unknown_citation", src.web)
        if src:
            return Verdict(True, spoken, src.ids, "cited", src.web)
        if is_abstention(sentence):
            return Verdict(True, spoken, (), "abstention")
        return Verdict(False, spoken, (), "uncited")
