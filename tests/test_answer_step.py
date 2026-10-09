"""Prompt 5: small-talk filter, heard-spelling search, the 3-fact answer prompt with examples, and the list route
for "what do I know about X"."""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from kin.brain.extract import ExtractedFact, is_small_talk_text
from kin.brain.prompt import load_prompt
from kin.memory.people import NameIndex
from kin.memory.query import understand
from kin.memory.store import Person, Utterance
from tests.conftest import FakeLLM, build_fake_app, make_fake_brain_factory

NOW = datetime(2026, 10, 5, 9, 0)


def _u(text: str, worth: int = 0) -> Utterance:
    return Utterance(1, NOW.timestamp(), text, None, 1, None, worth)


def _f(text: str, event: float | None = None) -> ExtractedFact:
    return ExtractedFact(text, "note", None, [], event, [1])


@pytest.mark.parametrize("fact, line, small", [
    ("Let's get lunch.", "I'm starving, let's get lunch.", True),
    ("Do you want to come over for dinner sometime?", "Do you want to come over for dinner sometime?", True),
    ("Someone said: That's amazing, congratulations!", "That's amazing, congratulations!", True),
    ("Someone said: I'm allergic to peanuts.", "I'm allergic to peanuts.", True),  # text alone cannot tell
    ("The gym locker code is 4471.", "My new locker code at the gym is 4471.", False),  # a number
    ("Priyanka is the senior data scientist.", "She's the senior data scientist on the third floor.", False),  # a name
    ("Nice to have a holiday for Gandhi Jayanti.", "Nice to have a holiday for Gandhi Jayanti.", False),  # a name: kept
])
def test_small_talk_text_rule(fact, line, small):
    assert is_small_talk_text(_f(fact), [_u(line)]) is small
    assert not is_small_talk_text(_f(fact, event=NOW.timestamp()), [_u(line)]), "a fact with a time is never small talk"


async def test_router_has_the_last_word_on_small_talk(tmp_path):
    """The text rule flags "Someone is allergic to peanuts" too; the router keeps it (not chatter)."""
    from tests.conftest import StubRouter

    llm = FakeLLM()
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, None, {}, router=StubRouter()))
    await h.app.start()
    await h.app.wait_brain()
    brain = h.app.brain
    try:
        llm.extract_replies = [json.dumps([
            {"text": "Let's grab some lunch.", "kind": "note", "subject": None, "people": [], "event_time": None, "lines": [1]},
            {"text": "Someone said: I'm allergic to peanuts.", "kind": "fact", "subject": None, "people": [], "event_time": None, "lines": [2]},
        ])]
        await brain._ingest("Let's grab some lunch.", ts=time.time() - 300)
        await brain._ingest("I'm allergic to peanuts.", ts=time.time() - 290)
        await brain.flush()
        assert [f.text for f in await brain.memory.recent()] == ["Someone said: I'm allergic to peanuts."]
    finally:
        await h.app.stop()


def test_a_misheard_question_is_searched_both_ways():
    idx = NameIndex([Person(1, "Rahul")])
    p = understand("What did Raul say on Friday?", NOW, idx)
    assert p.search == "What did Rahul say?" and p.search_original.startswith("What did Raul say")
    assert understand("What did Rahul say?", NOW, idx).search_original is None, "nothing changed: one query"


def test_answer_prompts_carry_worked_examples_with_unusable_ids():
    for name in ("agent_system", "recall_system"):
        p = load_prompt(name)
        assert "[#901]" in p and "I don't remember that." in p and "OUTDATED, replaced by #904" in p
        assert "[#7]" not in p


async def _brain(tmp_path, llm):
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, None, {}))
    h.app.cfg.memory.retrieval.answer_floor = 0.0
    h.app.cfg.memory.retrieval.strong_score = 0.0
    await h.app.start()
    await h.app.wait_brain()
    return h, h.app.brain


async def test_the_answer_prompt_offers_at_most_answer_facts(tmp_path):
    llm = FakeLLM()
    h, brain = await _brain(tmp_path, llm)
    try:
        for i in range(6):
            await brain.memory.remember(f"The locker code number {i} is 44{i}{i}.", "fact", 0.9)
        await brain.handle_query("What is the locker code?")
        sent = [m for _, m in llm.calls if "Question: What is the locker code?" in m][0]
        assert sum(line.startswith("[#") for line in sent.splitlines()) == brain.cfg.memory.answer_facts == 5
    finally:
        await h.app.stop()


async def test_what_do_i_know_about_someone_lists_their_facts_newest_first(tmp_path):
    summary = "Rahul is vegan [#1]. His birthday is on the 22nd of October [#2]."
    llm = FakeLLM(script=summary)
    h, brain = await _brain(tmp_path, llm)
    mem = brain.memory
    try:
        await mem.add_person("Rahul")
        await mem.remember("Rahul is vegan.", "fact", 0.9, ts=time.time() - 7200, subject="Rahul")
        await mem.remember("Rahul's birthday is on the 22nd of October.", "fact", 0.9, ts=time.time() - 3600, subject="Rahul")
        await mem.remember("The door code is 4512.", "fact", 0.9)
        reply = await brain.handle_query("What do I know about Rahul?")
        assert reply.path == "list: person" and reply.llm_calls == 1 and reply.text == summary.replace(" [#1]", "").replace(" [#2]", "")
        sent = [m for _, m in llm.calls if "Question: What do I know about Rahul?" in m][0]
        assert sent.index("[#2]") < sent.index("[#1]") and "[#3]" not in sent, "newest first; only facts about Rahul"
    finally:
        await h.app.stop()


async def test_the_list_is_read_back_when_the_summary_cites_nothing(tmp_path):
    llm = FakeLLM(script="I don't remember that.")
    h, brain = await _brain(tmp_path, llm)
    mem = brain.memory
    try:
        await mem.add_person("Rahul")
        await mem.remember("The user lent Rahul 500 rupees.", "fact", 0.9, subject="Rahul")
        reply = await brain.handle_query("Tell me about Rahul")
        assert reply.path == "list: person → read list" and reply.outcome == "answered"
        assert "You lent Rahul 500 rupees." in reply.text and reply.citations == "1"
    finally:
        await h.app.stop()
