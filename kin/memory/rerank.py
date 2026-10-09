"""Optional cross-encoder reranker (ms-marco-MiniLM-L-6-v2, ONNX, int8) over the fused top 20.

Off by default (`memory.retrieval.rerank`). A cross-encoder reads the question and one fact together,
so it can tell "When is Neha's birthday?" from a fact about Rahul's birthday, which neither cosine
nor BM25 can. The tokenizer is BERT's uncased WordPiece, written out here (no `tokenizers` dependency).

Download: `uv run kin download-models --reranker` (about 23 MB).
"""

from __future__ import annotations

import logging
import math
import platform
import unicodedata
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "ms-marco-MiniLM-L-6-v2"
MODEL_FILES = ("model_qint8_arm64.onnx", "model_quint8_avx2.onnx", "model.onnx")


def find_reranker(reranker_dir: Path) -> Path | None:
    """The ONNX file to use under models/reranker/ (int8 for this CPU if present), or None."""
    base = Path(reranker_dir) / MODEL_NAME
    if not (base / "vocab.txt").exists():
        return None
    arm = platform.machine().lower() in ("arm64", "aarch64")
    order = [MODEL_FILES[0], MODEL_FILES[2]] if arm else [MODEL_FILES[1], MODEL_FILES[2]]
    for name in order:
        if (base / name).exists():
            return base / name
    return None


class WordPiece:
    """BERT uncased tokenizer: lower-case, strip accents, split punctuation, greedy longest-match WordPiece."""

    def __init__(self, vocab_file: Path) -> None:
        self.vocab = {t: i for i, t in enumerate(Path(vocab_file).read_text(encoding="utf-8").splitlines())}
        self.cls, self.sep, self.unk, self.pad = (self.vocab[t] for t in ("[CLS]", "[SEP]", "[UNK]", "[PAD]"))

    @staticmethod
    def _basic(text: str) -> list[str]:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        out: list[str] = []
        for word in text.split():
            cur = ""
            for c in word:
                cat = unicodedata.category(c)
                if cat.startswith("P") or (33 <= ord(c) <= 47 or 58 <= ord(c) <= 64 or 91 <= ord(c) <= 96 or 123 <= ord(c) <= 126):
                    if cur:
                        out.append(cur)
                        cur = ""
                    out.append(c)
                elif cat.startswith("C"):
                    continue
                else:
                    cur += c
            if cur:
                out.append(cur)
        return out

    def ids(self, text: str) -> list[int]:
        out: list[int] = []
        for tok in self._basic(text):
            if len(tok) > 100:
                out.append(self.unk)
                continue
            start, pieces = 0, []
            while start < len(tok):
                end = len(tok)
                cur = None
                while start < end:
                    piece = ("##" if start else "") + tok[start:end]
                    if piece in self.vocab:
                        cur = self.vocab[piece]
                        break
                    end -= 1
                if cur is None:
                    pieces = [self.unk]
                    break
                pieces.append(cur)
                start = end
            out.extend(pieces)
        return out

    def pair(self, a: str, b: str, max_len: int = 256) -> tuple[list[int], list[int]]:
        x, y = self.ids(a), self.ids(b)
        while len(x) + len(y) > max_len - 3:  # longest-first truncation
            (x if len(x) > len(y) else y).pop()
        ids = [self.cls, *x, self.sep, *y, self.sep]
        types = [0] * (len(x) + 2) + [1] * (len(y) + 1)
        return ids, types


class CrossEncoder:
    def __init__(self, model_file: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_file), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        self.tok = WordPiece(Path(model_file).parent / "vocab.txt")
        self.name = f"{MODEL_NAME}/{Path(model_file).stem}"

    def scores(self, query: str, docs: list[str]) -> list[float]:
        """Relevance of each doc to the query, as a probability (sigmoid of the logit)."""
        if not docs:
            return []
        pairs = [self.tok.pair(query, d) for d in docs]
        width = max(len(ids) for ids, _ in pairs)
        ids = np.zeros((len(pairs), width), np.int64)
        types = np.zeros_like(ids)
        mask = np.zeros_like(ids)
        for r, (i, t) in enumerate(pairs):
            ids[r, : len(i)] = i
            types[r, : len(t)] = t
            mask[r, : len(i)] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        logits = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0].reshape(-1)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]

    def close(self) -> None:
        self._session = None
