"""Unit tests: gate, sentence splitter, prompt budget, LLM selection, LLM priority queue."""

from __future__ import annotations

import time

import pytest

from kin.brain.gate import Gate, citations_in, strip_citations
from kin.brain.prompt import PromptBuilder
from kin.brain.text import SentenceSplitter
from kin.memory.store import Fact, Hit
from tests.conftest import FakeLLM

# --- gate ----------------------------------------------------------------------------------


def test_gate_rules():
    g = Gate(0.5)
    assert g.check("You meet Priya at 4 pm [#7].", {7}).speakable
    assert g.check("You meet Priya at 4 pm.", {7}).reason == "uncited"
    assert g.check("You meet Priya [#8].", {7}).reason == "unknown_citation"
    assert g.check("You meet Priya [#7, #8].", {7}).reason == "unknown_citation"
    assert g.check("I don't remember that.", set()).reason == "abstention"
    assert g.check("[#7]", {7}).reason == "empty"
    assert g.retrieval_ok(0.5) and not g.retrieval_ok(0.49) and not g.retrieval_ok(None)


def test_citation_parsing_and_stripping():
    assert citations_in("A [#3] b [#4, #5] c [12].") == (3, 4, 5, 12)
    assert strip_citations("You meet Priya at 4 pm [#7].") == "You meet Priya at 4 pm."


# --- splitter ------------------------------------------------------------------------------


def test_splitter_streams_sentences_and_keeps_trailing_citations():
    s = SentenceSplitter()
    out = []
    for chunk in ["You are meet", "ing Priya at 4 p.m. at the ", "library cafe. [#7] ", "Dr. Rao will join [#9]. Bye"]:
        out += s.feed(chunk)
    assert out == ["You are meeting Priya at 4 p.m. at the library cafe. [#7]", "Dr. Rao will join [#9]."]
    assert s.flush() == "Bye"


# --- prompt budget ----------------------------------------------------------------------------


def _hit(i: int, text: str, sim: float = 0.9) -> Hit:
    return Hit(Fact(i, text, "plan", None, (), None, time.time(), "active", None, 0.9, "extracted", ()), sim)


def test_prompt_respects_token_budget_but_keeps_best_memory():
    llm = FakeLLM()
    hits = [_hit(i, "word " * 50) for i in range(1, 6)]
    small = PromptBuilder(llm.count_tokens, budget=120).recall("When?", hits)
    big = PromptBuilder(llm.count_tokens, budget=5000).recall("When?", hits)
    assert small.memory_ids == (1,)  # the best memory is always included
    assert big.memory_ids == (1, 2, 3, 4, 5)
    assert "[#1]" in small.user and "Question: When?" in small.user and "Now:" in small.user
    assert "[#N]" in small.system and "I don't remember that." in small.system
    assert "[#7]" not in small.system  # a concrete example id gets copied verbatim by small models


# --- LLM selection -------------------------------------------------------------------------


def test_create_llm_falls_back_when_primary_missing(monkeypatch, tmp_path):
    import kin.brain.llm as llm_mod
    from kin.config import load_config

    cfg = load_config("laptop")
    cfg.models_dir = tmp_path
    built = []

    class FakeLlama(FakeLLM):
        def __init__(self, path, **kw):
            super().__init__()
            built.append(path)

    monkeypatch.setattr(llm_mod, "LlamaCppBackend", FakeLlama)
    backend = llm_mod.create_llm(cfg)  # the .litertlm file does not exist in tmp_path
    assert isinstance(backend, FakeLlama) and built[0].name == "Qwen3.5-2B-Q4_K_M.gguf"

    cfg.llm.fallback_backend = "none"
    with pytest.raises(FileNotFoundError):
        llm_mod.create_llm(cfg)


def test_llm_stats_tokens_per_second():
    from kin.brain.llm import LlmStats

    assert LlmStats(output_tokens=21, ttft_s=1.0, duration_s=3.0).tokens_per_s == 10.5
    assert LlmStats(output_tokens=1, ttft_s=1.0, duration_s=1.0).tokens_per_s is None
