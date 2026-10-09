"""Recall benchmark (tests/eval/recall_benchmark): data integrity, scoring rules, and runs.

Fast (normal pytest): the dataset is well-formed, the noisy copy is up to date, the scoring rules
work, and the whole harness runs end to end with the test fakes.
Models (normal pytest when models are downloaded): the 20-question smoke subset through the real
pipeline, with floors just under the recorded baseline so regressions show up.
Slow: the full benchmark, clean and noisy. Run it explicitly:
    uv run pytest tests/eval -m slow -s        (or: uv run python scripts/eval_recall.py)
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tests.conftest import MODELS
from tests.eval.recall_benchmark import WEEK_NOISY, load_questions, load_week, noise
from tests.eval.recall_benchmark.runner import ROOT, run_variant
from tests.eval.recall_benchmark.scoring import check, contains, normalise, raw_marker

HAVE_MODELS = (MODELS / "llm" / "gemma-4-E2B-it.litertlm").exists()
RESULTS = ROOT / "tests" / "eval" / "results"

# --- the dataset ----------------------------------------------------------------------------------


def test_week_has_enough_realistic_lines():
    lines = load_week("clean")
    assert len(lines) >= 150
    assert len({ln.when.date() for ln in lines}) == 7
    speakers = Counter(ln.speaker for ln in lines)
    assert speakers["me"] >= 60 and len(speakers) >= 8, speakers  # the owner and other people
    keep = sum(ln.keep for ln in lines)
    assert 0.35 < keep / len(lines) < 0.65  # lots of chatter that should not be stored
    assert len({ln.text for ln in lines}) == len(lines), "lines are matched to memories by text"


def test_questions_cover_every_category_and_point_at_real_lines():
    qs = load_questions()
    ids = {ln.id for ln in load_week("clean")}
    cats = Counter(q.cat for q in qs)
    assert len(qs) >= 80
    assert cats["abstain"] >= 15
    for cat in ("single_fact", "lookup", "time_scoped", "latest_version", "who_said", "multi_fact", "split", "real_failure"):
        assert cats[cat] >= 3, cat
    assert len({q.id for q in qs}) == len(qs)
    for q in qs:
        assert set(q.support) <= ids, (q.id, q.support)
        assert q.abstain != bool(q.expect), f"{q.id}: either abstain or expect"
        assert q.abstain or q.support, q.id
        assert 1 <= q.needed <= len(q.expect) or q.abstain
    smoke = load_questions("smoke")
    assert len(smoke) == 20 and {"abstain", "real_failure"} <= {q.cat for q in smoke}


def test_noisy_copy_is_generated_and_about_ten_percent_wer():
    assert WEEK_NOISY.read_text(encoding="utf-8") == noise.build(), "run: python -m tests.eval.recall_benchmark.noise"
    clean, noisy = load_week("clean"), load_week("noisy")
    assert [(c.id, c.when, c.keep) for c in clean] == [(n.id, n.when, n.keep) for n in noisy]
    errors = words = 0
    for c, n in zip(clean, noisy):
        e, w = noise.word_error_rate(c.text, n.text)
        errors, words = errors + e, words + w
    assert 0.07 <= errors / words <= 0.14
    assert any("Pria" in n.text for n in noisy) and any("Raul" in n.text for n in noisy)


# --- scoring rules ----------------------------------------------------------------------------------


def test_normalise_numbers_codes_and_times():
    assert normalise("It's 1,850 rupees") == "it's 1850 rupees"
    assert normalise("Saffron2026") == "saffron 2026"
    assert normalise("at five PM") == "at 5 pm"
    assert contains("Your code is 4471.", "4471") and not contains("Your code is 44712.", "4471")
    assert contains("She was promoted.", "promot")
    assert contains("At 5pm.", "5") and not contains("At 15:00.", "5")
    assert contains("Coach B2", "b2")


def test_check_answer_abstention_and_false_answers():
    by_id = {q.id: q for q in load_questions()}
    assert check(by_id["n01"], "Your gym locker code is 4471.", "answered").correct
    wrong = check(by_id["n01"], "Your gym locker code is 4417.", "answered")
    assert not wrong.correct and wrong.false_answer
    shy = check(by_id["n01"], "I don't remember that.", "abstained")
    assert not shy.correct and shy.wrong_abstention and not shy.false_answer
    assert check(by_id["a02"], "I don't remember that.", "abstained").correct
    halluc = check(by_id["a02"], "Your Amazon password is bluefin88.", "answered")
    assert not halluc.correct and halluc.false_answer
    assert not check(by_id["w01"], "You're vegan, so no paneer.", "answered").correct  # wrong speaker
    assert check(by_id["w02"], "I don't remember that.", "abstained").correct  # honest partial answer
    assert check(by_id["m01"], "Rahul is vegan and his birthday is on 22 October.", "answered").correct
    assert not check(by_id["m01"], "Rahul is vegan.", "answered").correct  # needs two facts


def test_raw_marker_matches_the_real_failure():
    assert raw_marker("I don't remember that. [#]")
    assert raw_marker("Priyanka is your mentor [#3].")
    assert not raw_marker("Priyanka is your mentor.")


# --- runs -------------------------------------------------------------------------------------------


async def test_harness_runs_end_to_end_with_fakes():
    """Plumbing only: the fake embedder/router/LLM have no semantics, so the numbers mean nothing."""
    res = await run_variant("clean", "smoke", engines="fake", progress=lambda *_: None)
    s = res["summary"]
    assert s["write"]["lines"] == len(load_week("clean"))
    assert s["answers"]["questions"] == 20
    assert res["meta"]["code_hash"] and res["meta"]["git"]
    assert all(r["failure"] is None or r["failure"] for r in res["questions"])
    json.dumps(res)  # serialisable


# Floors a little under the recorded baseline (tests/eval/results): they catch regressions, not targets.
# Baseline 2026-10-05 (clean): keep/drop 89.4%, smoke accuracy 45-50%, smoke abstention 3/4.
SMOKE_FLOORS = {"keep_drop_accuracy": 85.0, "accuracy": 35.0, "correct_abstention_rate": 50.0}


@pytest.mark.models
@pytest.mark.skipif(not HAVE_MODELS, reason="models not downloaded")
async def test_smoke_subset_real_models():
    res = await run_variant("clean", "smoke", progress=print)
    s = res["summary"]
    print(json.dumps(s, indent=1))
    assert s["answers"]["questions"] == 20
    assert s["write"]["keep_drop_accuracy"] >= SMOKE_FLOORS["keep_drop_accuracy"]
    assert s["answers"]["accuracy"] >= SMOKE_FLOORS["accuracy"]
    assert s["answers"]["correct_abstention_rate"] >= SMOKE_FLOORS["correct_abstention_rate"]


@pytest.mark.models
@pytest.mark.slow
@pytest.mark.skipif(not HAVE_MODELS, reason="models not downloaded")
@pytest.mark.parametrize("variant", ["clean", "noisy"])
async def test_full_benchmark(variant, request):
    if "slow" not in (request.config.option.markexpr or ""):
        pytest.skip("the full benchmark takes ~30 min per variant: run with -m slow")
    from scripts.eval_recall import table  # noqa: PLC0415

    res = await run_variant(variant, "full", progress=print)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"recall-{variant}-full-pytest-{res['meta']['started'].replace(':', '')}-{res['meta']['git']}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(table([res]))
    assert res["summary"]["answers"]["questions"] >= 80
