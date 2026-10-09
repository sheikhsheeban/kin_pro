"""Semantic router: k-NN over labelled examples with EmbeddingGemma.

Fast tests use hand-made vectors for the maths, the thresholds and the cache. The model tests
embed the real example file and a held-out labelled set (tests/router_labelled.py, 42 lines).
"""

from __future__ import annotations

import numpy as np
import pytest

from kin.brain.semantic_router import LABELS, SemanticRouter, Tag, load_examples
from kin.config import CONFIG_DIR, load_config
from tests.conftest import MODELS
from tests.router_labelled import LABELLED

# --- fast: maths, thresholds, cache ---------------------------------------------------------------------


def _toy_examples() -> dict[str, list[str]]:
    return {label: [f"{label} {i}" for i in range(6)] for label in LABELS}


def _toy_vectors(texts: list[str]) -> np.ndarray:
    """Each label owns one axis; examples sit near it with a little noise."""
    rng = np.random.default_rng(0)
    out = []
    for t in texts:
        v = np.zeros(8, np.float32)
        v[LABELS.index(t.split()[0])] = 1.0
        v += rng.normal(0, 0.05, 8).astype(np.float32)
        out.append(v)
    return np.stack(out)


async def _fitted(tmp_path=None, **kw) -> SemanticRouter:
    r = SemanticRouter(_toy_examples(), **kw)

    async def embed(texts):
        return _toy_vectors(texts)

    await r.fit(embed, "toy", tmp_path)
    return r


def _axis(i: int, j: int | None = None, w: float = 0.0) -> np.ndarray:
    v = np.zeros(8, np.float32)
    v[i] = 1.0
    if j is not None:
        v[j] = w
    return v


async def test_clear_line_is_accepted_with_score_and_margin():
    r = await _fitted()
    tag = r.classify("I'm meeting Sam at 3 tomorrow.", _axis(LABELS.index("commitment")))
    assert tag.label == "commitment" and tag.accepted and tag.source == "semantic"
    assert tag.score > 0.9 and tag.margin > 0.5 and tag.storable


async def test_ambiguous_line_is_not_accepted():
    r = await _fitted()
    c, f = LABELS.index("commitment"), LABELS.index("fact_about_person")
    tag = r.classify("Something between a plan and a fact.", _axis(c, f, 0.95))
    assert not tag.accepted and tag.margin < 0.08 and not tag.storable


async def test_thresholds_are_configurable():
    strict = await _fitted(accept_score=0.99, accept_margin=0.08)
    loose = await _fitted(accept_score=0.5, accept_margin=0.01)
    v = _axis(LABELS.index("idea"), LABELS.index("chatter"), 0.6)
    assert not strict.classify("We could build a treehouse one day.", v).accepted
    assert loose.classify("We could build a treehouse one day.", v).accepted


async def test_chatter_is_accepted_but_not_storable_and_filler_skips_embedding():
    r = await _fitted()
    tag = r.classify("That film was rather long, honestly.", _axis(LABELS.index("chatter")))
    assert tag.label == "chatter" and tag.accepted and not tag.storable
    assert r.prefilter("okay sure") == Tag("chatter", 1.0, 1.0, True, "rule")
    assert r.prefilter("Rahul is allergic to peanuts.") is None


async def test_example_vectors_are_cached_per_example_set_and_embedder(tmp_path):
    calls = []

    async def embed(texts):
        calls.append(len(texts))
        return _toy_vectors(texts)

    a = SemanticRouter(_toy_examples())
    await a.fit(embed, "toy", tmp_path)
    b = SemanticRouter(_toy_examples())
    await b.fit(embed, "toy", tmp_path)
    assert calls == [24], "the second start-up loads the cached vectors"
    c = SemanticRouter(_toy_examples())
    await c.fit(embed, "another-embedder", tmp_path)
    assert calls == [24, 24], "a different embedder never reuses vectors"
    assert len(list(tmp_path.glob("router-examples-*.npy"))) == 2


def test_example_file_is_complete():
    ex = load_examples(CONFIG_DIR / "router_examples.yaml")
    assert set(ex) == set(LABELS)
    assert all(20 <= len(v) <= 30 for v in ex.values()), {k: len(v) for k, v in ex.items()}
    held_out = {t for t, _ in LABELLED}
    assert not held_out & {t for v in ex.values() for t in v}, "test sentences must not be examples"
    assert len(LABELLED) >= 40


def test_profiles_carry_thresholds():
    for profile in ("laptop", "pi5"):
        r = load_config(profile).router
        assert 0 < r.accept_score < 1 and 0 < r.accept_margin < 1
    assert load_config("pi5").extract.idle_s == 60
    assert load_config("laptop").extract.idle_s == 0
    assert load_config("pi5").stt.passive.arch == "tiny-streaming"


# --- real EmbeddingGemma -------------------------------------------------------------------------------

EMBED_DIR = MODELS / "embedding"


@pytest.fixture(scope="module")
def real():
    if not any(EMBED_DIR.glob("**/embeddinggemma-300m/tokenizer.bin")):
        pytest.skip("EmbeddingGemma not downloaded")
    import asyncio

    from kin.memory.embed import GemmaEmbedder, find_embedding_model

    emb = GemmaEmbedder(find_embedding_model(EMBED_DIR), "q4", 256)
    router = SemanticRouter(load_examples(CONFIG_DIR / "router_examples.yaml"))

    async def embed(texts):
        return np.stack([emb.embed_document(t) for t in texts])

    asyncio.run(router.fit(embed, emb.name))
    rows = []
    for text, want in LABELLED:
        ranked = sorted(router.scores(emb.embed_document(text)).items(), key=lambda kv: kv[1], reverse=True)
        rows.append((text, want, ranked[0][0], ranked[0][1], ranked[0][1] - ranked[1][1]))
    yield router, emb, rows
    emb.close()


def _report(rows, score, margin) -> tuple[int, float]:
    accepted = [r for r in rows if r[3] >= score and r[4] >= margin]
    acc = sum(r[1] == r[2] for r in accepted) / len(accepted) if accepted else 1.0
    return len(accepted), acc


@pytest.mark.models
@pytest.mark.parametrize("profile", ["laptop", "pi5"])
def test_accuracy_on_accepted_items_with_profile_thresholds(real, profile):
    _, _, rows = real
    r = load_config(profile).router
    n, acc = _report(rows, r.accept_score, r.accept_margin)
    top1 = sum(row[1] == row[2] for row in rows) / len(rows)
    print(f"\n{profile}: accepted {n}/{len(rows)} at {acc:.0%} accuracy (score>={r.accept_score}, margin>={r.accept_margin}); "
          f"top-1 over all {top1:.0%}")
    for spec in [(0.70, 0.08), (0.60, 0.06), (0.55, 0.06), (0.50, 0.04)]:
        print("   score>={} margin>={}: accepted {}/{} at {:.0%}".format(*spec, *_report(rows, *spec)[:1], len(rows), _report(rows, *spec)[1]))
    assert acc >= 0.85
    assert n >= len(rows) // 2, "most clear lines should not need the LLM"


@pytest.mark.models
def test_low_confidence_items_go_to_pending(real):
    router, emb, rows = real
    unsure = [r for r in rows if r[3] < router.accept_score or r[4] < router.accept_margin]
    assert unsure, "the default (spec) thresholds leave some lines for the LLM"
    for text, *_ in unsure:
        tag = router.classify(text, emb.embed_document(text))
        assert not tag.accepted and not tag.storable  # -> the pending table, not stored, not dropped


@pytest.mark.models
def test_old_hard_cases(real):
    router, emb, _ = real
    for text in ["The wifi password is tiger42.", "I parked on level three near the blue pillar."]:
        tag = router.classify(text, emb.embed_document(text))
        print(f"\n{text!r}: {tag}")
        # Either a confident storable label, or unsure (the E2B tagger decides). Never confidently dropped.
        assert tag.storable or not tag.accepted
