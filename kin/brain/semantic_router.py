"""Cheap pre-filter for overheard speech with the already-loaded EmbeddingGemma (no extra model).

Every final passive line that is not filler is embedded once and compared with labelled example
sentences (config/router_examples.yaml):

  per label, the mean cosine similarity of its k nearest examples
  label  = the best label;  score = its similarity;  margin = best - runner-up

The router no longer decides what is stored (facts are extracted per episode by the LLM, see
`kin.brain.extract`). A line it accepts as chatter (score >= accept_score and margin >=
accept_margin) does not count towards its episode; an episode in which every line is chatter
is skipped without calling the LLM. The labels also give "remember that ..." facts their kind.

Example vectors are computed once per (examples, embedder) and cached on disk: they come
from the config file, never from user speech.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

log = logging.getLogger(__name__)

Label = Literal["commitment", "fact_about_person", "idea", "note", "chatter"]
LABELS: tuple[str, ...] = ("commitment", "fact_about_person", "idea", "chatter")
STORABLE: frozenset[str] = frozenset({"commitment", "fact_about_person", "idea", "note"})

_FILLER = re.compile(r"^(ok(ay)?|yeah|yes|no|nope|um+|uh+|hmm+|hm+|thanks?( you)?|hi|hello|hey|bye|right|sure|cool|wow|oh)\b")


@dataclass(frozen=True)
class Tag:
    label: str
    score: float
    margin: float
    accepted: bool
    source: Literal["semantic", "rule", "llm"] = "semantic"

    @property
    def storable(self) -> bool:
        return self.accepted and self.label in STORABLE


def load_examples(path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [label for label in LABELS if not data.get(label)]
    if missing:
        raise ValueError(f"{path}: no examples for {missing}")
    unknown = set(data) - set(LABELS)
    if unknown:
        raise ValueError(f"{path}: unknown labels {sorted(unknown)}")
    return {label: [str(s).strip() for s in data[label] if str(s).strip()] for label in LABELS}


class SemanticRouter:
    def __init__(
        self,
        examples: dict[str, list[str]],
        *,
        k: int = 5,
        accept_score: float = 0.70,
        accept_margin: float = 0.08,
        min_words: int = 3,
    ) -> None:
        self.examples = examples
        self.k = k
        self.accept_score = accept_score
        self.accept_margin = accept_margin
        self.min_words = min_words
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None  # (n_examples, dim), rows L2-normalised
        self._owner: np.ndarray | None = None  # label index per row

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def fingerprint(self, embedder_name: str) -> str:
        blob = json.dumps({"embedder": embedder_name, "examples": self.examples}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    async def fit(
        self,
        embed_many: Callable[[list[str]], Awaitable[np.ndarray]],
        embedder_name: str,
        cache_dir: Path | None = None,
    ) -> None:
        """Embed the examples (or load the cached vectors for this exact example set and embedder)."""
        texts = [t for label in LABELS for t in self.examples[label]]
        owner = np.array([i for i, label in enumerate(LABELS) for _ in self.examples[label]])
        cache = Path(cache_dir) / f"router-examples-{self.fingerprint(embedder_name)}.npy" if cache_dir else None
        vectors = None
        if cache is not None and cache.exists():
            try:
                vectors = np.load(cache)
                if vectors.shape[0] != len(texts):
                    vectors = None
            except Exception:  # corrupt cache: recompute
                vectors = None
        if vectors is None:
            vectors = np.asarray(await embed_many(texts), dtype=np.float32)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._matrix = vectors / np.where(norms == 0, 1, norms)
        self._owner = owner
        self._labels = list(LABELS)

    def prefilter(self, text: str) -> Tag | None:
        """Lines too short or pure filler are chatter without embedding them."""
        low = text.strip().lower()
        words = re.findall(r"[\w']+", low)
        if len(words) < self.min_words or (_FILLER.match(low) and len(words) <= 4):
            return Tag("chatter", 1.0, 1.0, True, "rule")
        return None

    def scores(self, vector: np.ndarray) -> dict[str, float]:
        assert self._matrix is not None and self._owner is not None, "call fit() first"
        v = np.asarray(vector, dtype=np.float32)
        v = v / (np.linalg.norm(v) or 1.0)
        sims = self._matrix @ v
        out: dict[str, float] = {}
        for i, label in enumerate(self._labels):
            mine = np.sort(sims[self._owner == i])[::-1][: self.k]
            out[label] = float(mine.mean())
        return out

    def classify(self, text: str, vector: np.ndarray) -> Tag:
        pre = self.prefilter(text)
        if pre is not None:
            return pre
        ranked = sorted(self.scores(vector).items(), key=lambda kv: kv[1], reverse=True)
        (label, score), (_, second) = ranked[0], ranked[1]
        margin = score - second
        accepted = score >= self.accept_score and margin >= self.accept_margin
        return Tag(label, round(score, 4), round(margin, 4), accepted)


def is_filler(text: str) -> bool:
    """Pure filler ("okay", "yeah sure", "thanks") is not even kept as an utterance."""
    low = text.strip().lower()
    words = re.findall(r"[\w']+", low)
    return not words or (bool(_FILLER.match(low)) and len(words) <= 4)


_KIND_OF_LABEL = {"commitment": "plan", "fact_about_person": "fact", "idea": "idea"}


def fact_kind(label: str) -> str:
    """A fact kind (plan | fact | idea | note) from a router label."""
    return _KIND_OF_LABEL.get(label, "note")
