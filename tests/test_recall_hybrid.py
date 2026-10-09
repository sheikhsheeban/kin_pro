"""Hybrid recall: query understanding, fusion, filters, conflicts, the gate, the one-call shortcut,
the transcript fallback and the "why this answer" record. Fast: fake embedder and LLM, real encrypted store."""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime

import numpy as np
import pytest

from kin.brain.gate import Gate
from kin.brain.prompt import PromptBuilder, format_memory
from kin.memory.people import NameIndex
from kin.memory.query import QueryPlan, time_window, understand
from kin.memory.retrieval import FusionWeights, conflicts, fuse, in_window, mentions, rrf
from kin.memory.store import Fact, Hit, MemoryStore, Person
from tests.conftest import FakeEmbedder, FakeLLM, build_fake_app, make_fake_brain_factory

NOW = datetime(2026, 10, 5, 9, 0)  # Monday, as in the recall benchmark
DAY = 86400.0


def _day(w) -> str:
    return datetime.fromtimestamp(w[0]).strftime("%a %d %b")


def _fact(i: int, text: str, *, heard: float | None = None, event: float | None = None, subject: str | None = None,
          people: tuple[str, ...] = (), status: str = "active", superseded_by: int | None = None) -> Fact:
    return Fact(i, text, "fact", subject, people, event, heard if heard is not None else NOW.timestamp() - DAY, status,
                superseded_by, 0.8, "extracted", ())


# --- time parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize("question, day, past", [
    ("What did I plan yesterday?", "Sun 04 Oct", True),
    ("What's on tomorrow?", "Tue 06 Oct", False),
    ("What did Mum tell me last Monday?", "Mon 28 Sep", True),
    ("Who did I meet at the cafe on Wednesday?", "Wed 30 Sep", True),  # past tense looks back
    ("When is my meeting on Friday?", "Fri 09 Oct", False),  # otherwise ahead
    ("What am I doing next Wednesday?", "Wed 07 Oct", False),
    ("What did Priya say 3 days ago?", "Fri 02 Oct", True),
    ("What's happening on 14 November?", "Sat 14 Nov", False),
    ("What's happening on November 14th?", "Sat 14 Nov", False),
    ("What did I do on the 2nd?", "Fri 02 Oct", True),
    ("What do I have this weekend?", "Sat 10 Oct", False),
    ("What did we do last weekend?", "Sat 03 Oct", True),
    ("What's due in 2 days?", "Wed 07 Oct", False),
])
def test_time_expressions_become_windows(question, day, past):
    window, label, is_past = time_window(question, NOW)
    assert window is not None and _day(window) == day, label
    assert is_past == past


def test_week_and_month_windows():
    w, _, _ = time_window("What did I spend money on last week?", NOW)
    assert datetime.fromtimestamp(w[0]) == datetime(2026, 9, 28) and w[1] == NOW.timestamp()
    w, _, _ = time_window("What's coming up later in October?", NOW)
    assert w == (NOW.timestamp(), datetime(2026, 11, 1).timestamp())
    w, _, _ = time_window("What did I buy in September?", NOW)
    assert w == (datetime(2026, 9, 1).timestamp(), datetime(2026, 10, 1).timestamp())
    w, _, _ = time_window("What's next week looking like?", NOW)
    assert w == (datetime(2026, 10, 12).timestamp(), datetime(2026, 10, 19).timestamp())


@pytest.mark.parametrize("question", [
    "How much will the rent be from November?",  # the month is what was said, not when to look
    "When is my dentist appointment?",
    "What may happen if it rains?",
    "Who is Priyanka?",
])
def test_no_window_without_a_time_scope(question):
    assert time_window(question, NOW)[0] is None


def test_future_and_history_flags():
    assert understand("When is Rahul's birthday?", NOW).future
    assert understand("What's coming up?", NOW).future
    assert not understand("What did Rahul say?", NOW).future
    assert understand("What time was the meeting originally?", NOW).history
    assert not understand("When is the meeting?", NOW).history
    p = understand("What's on tomorrow?", NOW)
    assert p.time_field == "event" and p.future, "ahead: when things happen"
    assert understand("What did I plan yesterday?", NOW).time_field == "either", "behind: heard or happened then"


# --- names --------------------------------------------------------------------------------------------


def test_people_known_alias_fuzzy_lowercase_and_unknown():
    idx = NameIndex([Person(1, "Priya", ("Pria",)), Person(2, "Priyanka"), Person(3, "Rahul")])
    p = understand("Where is Pria moving?", NOW, idx)
    assert p.people == ("Priya",) and p.text == "Where is Priya moving?" and p.fixes == {"Pria": "Priya"}
    assert understand("What did priyanka say?", NOW, idx).people == ("Priyanka",), "lower case from the STT"
    assert understand("What does Raul eat?", NOW, idx).people == ("Rahul",), "a near-miss spelling"
    assert "Priya" not in understand("Who is Priyanka?", NOW, idx).people, "Priya and Priyanka stay apart"
    unknown = understand("Who is Rohan?", NOW, idx)
    assert unknown.people == () and unknown.unknown_names == ("Rohan",)
    assert understand("What did Mum tell me on Friday?", NOW, idx).unknown_names == (), "relations are not names"


# --- fusion -------------------------------------------------------------------------------------------


def test_rrf_sums_weighted_reciprocal_ranks():
    scores, ranks = rrf([("dense", [1, 2, 3], 1.0), ("lexical", [3, 1], 1.0), ("dense:alt", [2], 0.5)], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62 + 0.5 / 61)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert ranks[3] == {"dense": 3, "lexical": 1}
    assert max(scores, key=scores.get) == 1, "first in one list and second in another beats one list alone"


def test_fusion_lexical_rescues_a_code_the_vectors_missed():
    plan = understand("What's my locker code?", NOW)
    facts = {i: _fact(i, t) for i, t in [(1, "My gym locker code is 4471."), (2, "The gym is closed on Sunday."),
                                          (3, "The locker at work is blue."), (4, "Gym class at 7.")]}
    cos = {"q": {1: 0.40, 2: 0.62, 3: 0.55, 4: 0.58}}  # the vectors prefer anything about the gym
    lexical = {"q": [1, 3]}  # BM25: "locker" and "code"
    fused = fuse(plan, facts, cos, lexical, NOW.timestamp(), FusionWeights(recency_boost=0))
    assert fused.ranked[0].fact.id == 1 and fused.ranked[0].ranks == {"dense": 4, "lexical": 1}


def test_time_filter_ranks_the_named_day_first_with_backfill():
    plan = understand("What did I plan yesterday?", NOW)
    yday, older = NOW.timestamp() - DAY + 3600, NOW.timestamp() - 5 * DAY
    facts = {1: _fact(1, "Join the badminton club.", heard=yday), 2: _fact(2, "Plan a trip to Goa.", heard=older),
             3: _fact(3, "Buy a gift for Dad.", heard=yday)}
    cos = {"q": {1: 0.3, 2: 0.9, 3: 0.2}}
    fused = fuse(plan, facts, cos, {"q": []}, NOW.timestamp())
    assert [s.fact.id for s in fused.ranked] == [1, 3, 2], "both of yesterday's facts first, despite a better cosine"
    assert fused.ranked[2].misses == ("time",) and fused.ranked[2].detail()["backfill"] == ["time"]
    assert not fused.relaxed


def test_superseded_facts_are_left_out_unless_the_question_asks_about_history():
    facts = {1: _fact(1, "Meet Priya at 4.", status="superseded", superseded_by=2), 2: _fact(2, "Meet Priya at 5.")}
    cos = {"q": {1: 0.9, 2: 0.8}}
    now = NOW.timestamp()
    fused = fuse(understand("When do I meet Priya?", NOW), facts, cos, {"q": []}, now)
    assert [s.fact.id for s in fused.ranked] == [2]
    assert [(s.fact.id, s.why_out) for s in fused.filtered] == [(1, "superseded")]
    hist = fuse(understand("What time was it before it changed?", NOW), facts, cos, {"q": []}, now)
    assert {s.fact.id for s in hist.ranked} == {1, 2} and not hist.filtered


def test_person_matching_uses_subject_and_text_not_the_people_list():
    stuffed = _fact(1, "The team offsite is in Lonavala.", people=("Priya", "Rahul", "Sam"))
    assert not mentions(stuffed, ("Priya",)), "E2B sometimes copies every known name into `people`"
    assert mentions(_fact(2, "Someone said: lunch with Priya.", subject=None), ("Priya",))
    assert mentions(_fact(3, "Moving in December.", subject="Priya"), ("Priya",))
    assert not mentions(_fact(4, "Priyanka is a data scientist."), ("Priya",)), "whole words only"
    assert mentions(_fact(5, "Pria is moving."), ("Priya",), {"Priya": ("Pria",)})


def test_person_boost_and_filters_relax_when_nothing_passes():
    plan = understand("Where is Sam's new flat?", NOW, NameIndex([Person(1, "Sam")]))
    facts = {1: _fact(1, "Someone said: my new flat is in Dharampeth."), 2: _fact(2, "Sam borrowed the umbrella.")}
    cos = {"q": {1: 0.6, 2: 0.4}}
    boosted = fuse(plan, facts, cos, {"q": []}, NOW.timestamp(), FusionWeights(person_mode="boost", person_boost=0.0))
    assert boosted.ranked[0].fact.id == 1, "with no extra weight the better match still wins"
    tiered = fuse(plan, facts, cos, {"q": []}, NOW.timestamp(), FusionWeights(person_mode="tier"))
    assert tiered.ranked[0].fact.id == 2 and tiered.ranked[1].misses == ("person",)
    nobody = fuse(understand("What did Sam say on Friday?", NOW, NameIndex([Person(1, "Sam")])), {1: facts[1]},
                  {"q": {1: 0.5}}, {"q": []}, NOW.timestamp())
    assert set(nobody.relaxed) == {"time", "person"} and nobody.ranked, "nothing passed: still searched, marked relaxed"


def test_future_questions_prefer_upcoming_events():
    plan = understand("When is my dentist appointment?", NOW)
    past = _fact(1, "Dentist appointment Friday 11:30.", event=NOW.timestamp() - 3 * DAY)
    ahead = _fact(2, "Dentist appointment Wednesday at 6.", event=NOW.timestamp() + 2 * DAY)
    fused = fuse(plan, {1: past, 2: ahead}, {"q": {1: 0.8, 2: 0.8}}, {"q": [1, 2]}, NOW.timestamp())
    assert fused.ranked[0].fact.id == 2 and fused.ranked[0].boost > 1 > fused.ranked[1].boost


def test_in_window_fields():
    a, b = NOW.timestamp(), NOW.timestamp() + DAY
    f = _fact(1, "x", heard=a - DAY, event=a + 3600)
    assert in_window(f, (a, b), "event") and in_window(f, (a, b), "either") and not in_window(f, (a, b), "heard")


# --- conflicts ------------------------------------------------------------------------------------------


def test_disagreeing_active_facts_are_paired():
    v = np.ones(4, np.float32) / 2
    f1 = _fact(12, "Meet Priya at 4 pm on Wednesday.", subject="Priya", event=NOW.timestamp() + 16 * 3600, heard=NOW.timestamp() - 2 * DAY)
    f2 = _fact(15, "Meet Priya at 5 pm on Wednesday.", subject="Priya", event=NOW.timestamp() + 17 * 3600, heard=NOW.timestamp() - DAY)
    f3 = _fact(16, "Rahul is vegan.", subject="Rahul")
    hits = [Hit(f2, 0.9), Hit(f1, 0.8), Hit(f3, 0.5)]
    assert conflicts(hits, {12: v, 15: v, 16: v}) == [(12, 15)], "older first; Rahul shares no subject"
    old = Fact(*[*f1.__dict__.values()][:7], "superseded", 15, 0.8, "extracted", ())
    assert conflicts([Hit(f2, 0.9), Hit(old, 0.8)], {12: v, 15: v}) == [], "a replaced fact is not a conflict"
    far = np.array([1, 0, 0, 0], np.float32), np.array([0, 1, 0, 0], np.float32)
    assert conflicts(hits, {12: far[0], 15: far[1], 16: v}) == [], "not about the same thing"


def test_prompt_lists_subject_times_status_and_the_conflict():
    f1 = _fact(12, "Meet Priya at 4 pm.", subject="Priya", event=NOW.timestamp() + 7 * 3600, heard=NOW.timestamp() - 2 * DAY)
    f2 = _fact(15, "Meet Priya at 5 pm.", subject="Priya", event=NOW.timestamp() + 8 * 3600, heard=NOW.timestamp() - DAY)
    assert format_memory(Hit(f1, 0.9)) == "[#12] (happens Mon 05 Oct, 16:00; heard Sat 03 Oct, 09:00) Meet Priya at 4 pm."
    dad = _fact(7, "Your father's birthday is on the 9th.", subject="Dad", heard=NOW.timestamp() - DAY)
    assert format_memory(Hit(dad, 0.9)).startswith("[#7] (about Dad; heard"), "the subject when the text doesn't name it"
    hinted = PromptBuilder(lambda t: len(t.split()), 1000).recall("When is Rahul's birthday?", [Hit(dad, 0.9)], NOW.timestamp(),
                                                                names={"Rahul": "Raul"})
    assert "Rahul in the question is Raul in the memories" in hinted.user
    old = _fact(3, "Old plan.", status="superseded", superseded_by=15)
    assert "OUTDATED, replaced by #15" in format_memory(Hit(old, 0.5))
    built = PromptBuilder(lambda t: len(t.split()), 1000).recall("When do I meet Priya?", [Hit(f2, 0.9), Hit(f1, 0.8)],
                                                               NOW.timestamp(), conflicts=[(12, 15)])
    assert built.user.index("[#15]") < built.user.index("[#12] ("), "best first"
    assert "Note: [#12] and [#15] disagree" in built.user and built.memory_ids == (15, 12)
    assert "[#N][#M]" in built.system and "[#12]" not in built.system


def test_gate_accepts_a_sentence_citing_two_offered_facts():
    g = Gate(0.5)
    v = g.check("On Saturday you said 4 pm, later 5 pm [#12][#15].", {12, 15})
    assert v.speakable and v.citations == (12, 15) and v.spoken == "On Saturday you said 4 pm, later 5 pm."
    assert g.check("You said 4 pm, later 5 pm [#12][#16].", {12, 15}).reason == "unknown_citation"
    assert g.check("I'm not sure.", {12}).reason == "abstention"
    empty = g.check("I do not remember who Rohan is [#].", {12})
    assert empty.reason == "abstention" and empty.spoken == "I do not remember who Rohan is.", "no marker is read aloud"


# --- store: candidates, utterance vectors, a new embedding size -----------------------------------------


def test_candidates_include_window_and_people_and_skip_superseded(tmp_path):
    emb = FakeEmbedder(16)
    s = MemoryStore(tmp_path / "kin.db", secrets.token_hex(32), 16)
    try:
        t = NOW.timestamp()
        a = s.add_fact("Priya moves to Bangalore in December.", "fact", emb.embed_document("Priya moves"), confidence=0.8,
                       subject="Priya", heard_ts=t - 3 * DAY)
        b = s.add_fact("The water is off on Tuesday.", "fact", emb.embed_document("water off"), confidence=0.8,
                       event_ts=t + DAY + 3600, heard_ts=t - DAY)
        c = s.add_fact("Old plan.", "fact", emb.embed_document("old plan"), confidence=0.8, heard_ts=t - DAY)
        s.supersede(c.id, a.id)
        q = {"q": (emb.embed_query("zzz unrelated"), "zzz unrelated")}
        facts, cos, lex = s.candidates(q, 1, window=(t + DAY, t + 2 * DAY), people=("Priya",))
        assert {a.id, b.id} <= set(facts) and c.id not in facts
        assert set(cos["q"]) == set(facts), "every candidate gets its cosine"
        facts, _, _ = s.candidates(q, 5, history=True)
        assert c.id in facts
    finally:
        s.close()


def test_forgetting_lines_also_deletes_their_vectors(tmp_path):
    emb = FakeEmbedder(16)
    s = MemoryStore(tmp_path / "kin.db", secrets.token_hex(32), 16)
    try:
        now = time.time()
        u, _ = s.add_utterance("The door code is 4512.", now - 30, vector=emb.embed_document("door code 4512"))
        s.add_utterance("Old line.", now - 20 * DAY, vector=emb.embed_document("old line"))
        assert s._db.execute("SELECT count(*) FROM utterance_vec").fetchone()[0] == 2
        lines, cos, lex = s.utterance_candidates({"q": (emb.embed_query("door code"), "door code")}, 5)
        assert u.id in lines and lex["q"][0] == u.id and cos["q"][u.id] > 0.5
        s.purge_utterances(now - 14 * DAY)
        s.delete_since(now - 60)
        assert s._db.execute("SELECT count(*) FROM utterance_vec").fetchone()[0] == 0, "vectors go with the lines"
    finally:
        s.close()


async def test_a_new_embedding_size_reembeds_every_fact_and_line(tmp_path):
    from kin.core.bus import EventBus
    from kin.memory.service import MemoryService

    key = secrets.token_hex(32)
    db = tmp_path / "kin.db"
    s = MemoryStore(db, key, 8)
    f = s.add_fact("Rahul is vegan.", "fact", FakeEmbedder(8).embed_document("Rahul is vegan."), confidence=0.8)
    s.add_utterance("Rahul is vegan now.", time.time(), vector=FakeEmbedder(8).embed_document("Rahul is vegan now."))
    s.close()
    with pytest.raises(RuntimeError, match="8-d"):
        MemoryStore(db, key, 16)
    svc = MemoryService(EventBus(), lambda: FakeEmbedder(16), lambda dim: MemoryStore(db, key, dim, reembed=True))
    await svc.start()
    try:
        assert svc.reembedded == 2 and svc.store.rebuilt_from_dim == 8
        hits = await svc.retrieve("Is Rahul vegan?", 3)
        assert hits and hits[0].fact.id == f.id and hits[0].similarity > 0.5
    finally:
        await svc.close()


# --- the brain: shortcut, fallback, cap, explanation -------------------------------------------------------


async def _kin(tmp_path, llm: FakeLLM | None = None, **retrieval):
    store: dict = {}
    llm = llm or FakeLLM()
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, None, store))
    for k, v in retrieval.items():
        setattr(h.app.cfg.memory.retrieval, k, v)
    await h.app.start()
    await h.app.wait_brain()
    return h, h.app.brain, llm


async def test_strong_match_is_answered_in_one_call_with_the_facts(tmp_path):
    h, brain, llm = await _kin(tmp_path, answer_floor=0.3, strong_score=0.6)
    try:
        await brain.memory.remember("I'm meeting Priya at 4 pm at the library cafe.", "plan", 0.9)
        reply = await brain.handle_query("When am I meeting Priya?")
        assert reply.path == "shortcut: recall" and reply.llm_calls == 1 and reply.outcome == "answered"
        assert "4 pm" in reply.text and reply.citations == "1"
        first = [m for _, m in llm.calls if "When am I meeting Priya?" in m][0]
        assert "Memories:" in first and "[#1]" in first, "the facts are in the first and only message"
        why = json.loads(reply.explain)
        assert why["decision"]["shortcut"] and why["decision"]["confidence"] >= 0.6
        assert why["used"][0]["id"] == 1 and why["used"][0]["cited"] and "dense" in why["used"][0]["ranks"]
    finally:
        await h.app.stop()


@pytest.mark.parametrize("query, path", [
    ("Remind me to call Priya at 4 pm", "tool: add_task"),  # a command, even though memory matches it
    ("Do I need an umbrella for meeting Priya at 4 pm?", "tool: lookup (weather)"),  # needs the forecast
])
async def test_commands_and_lookups_never_take_the_shortcut(tmp_path, query, path):
    h, brain, llm = await _kin(tmp_path, answer_floor=0.3, strong_score=0.3)
    try:
        await brain.memory.remember("I'm meeting Priya at 4 pm at the library cafe.", "plan", 0.9)
        reply = await brain.handle_query(query)
        assert reply.path.startswith(path) and "shortcut" not in reply.path
    finally:
        await h.app.stop()


async def test_weak_facts_abstain_or_quote_a_heard_line(tmp_path):
    h, brain, llm = await _kin(tmp_path, answer_floor=0.95, strong_score=2.0, utterance_floor=0.5)
    try:
        await brain.memory.remember("The office wifi password is Saffron2026.", "fact", 0.9)
        await brain._ingest("Kavya said the clinic opens at nine on weekdays.", ts=time.time() - DAY)
        reply = await brain.handle_query("When does the clinic open?")
        assert reply.outcome == "answered" and "fallback: transcript" in reply.path
        assert reply.text.startswith("I'm not sure, but yesterday at") and "the clinic opens at nine" in reply.text
        assert json.loads(reply.explain)["decision"]["heard_lines"]
        nothing = await brain.handle_query("What is my blood group?")
        assert nothing.outcome == "abstained" and nothing.text == "I don't remember that."
    finally:
        await h.app.stop()


async def test_answers_are_cut_after_three_cited_sentences(tmp_path):
    four = "Rahul is vegan [#1]. He was born in October [#1]. His sister is Kavya [#1]. He likes cricket [#1]."
    h, brain, llm = await _kin(tmp_path, FakeLLM(script=four), answer_floor=0.1, strong_score=0.1)
    try:
        await brain.memory.remember("Rahul is vegan, his birthday is in October and his sister is Kavya.", "fact", 0.9)
        reply = await brain.handle_query("What do I know about Rahul?")
        assert reply.text == "Rahul is vegan. He was born in October. His sister is Kavya."
    finally:
        await h.app.stop()


async def test_a_sentence_that_only_says_where_it_was_heard_is_not_spoken(tmp_path):
    said = "Your gym locker code is 4471 [#1]. This information was heard on Monday, September 28 [#1]."
    h, brain, llm = await _kin(tmp_path, FakeLLM(script=said), answer_floor=0.1, strong_score=0.1)
    try:
        await brain.memory.remember("My gym locker code is 4471.", "fact", 0.9)
        reply = await brain.handle_query("What's my gym locker code?")
        assert reply.text == "Your gym locker code is 4471." and reply.blocked_sentences == 0
    finally:
        await h.app.stop()


@pytest.mark.parametrize("fact, said, spoken", [
    ("Rahul's birthday is on the 22nd of October.", "Sam's birthday is on the 22nd of October [#1].", False),  # someone else's
    ("Someone said: my birthday is on the 22nd of October.", "Sam's birthday is on the 22nd of October [#1].", True),  # unknown speaker
    ("Sam's birthday is on the 22nd of October.", "Sam's birthday is on the 22nd of October [#1].", True),
])
async def test_an_answer_about_one_person_cannot_cite_another(tmp_path, fact, said, spoken):
    h, brain, llm = await _kin(tmp_path, FakeLLM(script=said), answer_floor=0.0, strong_score=0.0)
    try:
        await brain.memory.add_person("Sam")
        await brain.memory.add_person("Rahul")
        await brain.memory.remember(fact, "fact", 0.9)
        reply = await brain.handle_query("When is Sam's birthday?")
        assert ("22nd" in reply.text) is spoken and reply.blocked_sentences == (0 if spoken else 1)
    finally:
        await h.app.stop()


@pytest.mark.parametrize("said, spoken", [
    ("Your yoga class is on Monday at 09:00 [#1].", False),  # 09:00 is in no memory (it was the time now)
    ("Your classes start on the 12th of January [#1].", True),
    ("Your classes start at 5 pm [#1].", True),  # the event is at 17:00
])
async def test_numbers_in_an_answer_must_come_from_the_cited_memory(tmp_path, said, spoken):
    h, brain, llm = await _kin(tmp_path, FakeLLM(script=said), answer_floor=0.0, strong_score=0.0)
    try:
        event = datetime(2027, 1, 12, 17, 0).timestamp()
        heard = datetime(2026, 9, 29, 20, 3).timestamp()  # heard in September: month 9 must not ground "09:00"
        await brain.memory.remember("Classes start on the 12th of January.", "plan", 0.9, event_ts=event, ts=heard)
        reply = await brain.handle_query("When do my classes start?")
        assert (reply.outcome == "answered") is spoken and reply.blocked_sentences == (0 if spoken else 1)
    finally:
        await h.app.stop()


async def test_conflicting_facts_reach_the_model_and_both_can_be_cited(tmp_path):
    answer = "On Saturday you said 4 pm, later 5 pm [#1][#2]."
    h, brain, llm = await _kin(tmp_path, FakeLLM(script=answer), answer_floor=0.1, strong_score=0.1, conflict_similarity=0.5)
    try:
        mem = brain.memory
        t = time.time()
        await mem.remember("Priya and you meet at the library cafe at 4 pm.", "plan", 0.8, ts=t - 2 * DAY, subject="Priya",
                           people=["Priya"], event_ts=t + DAY)
        await mem.remember("Priya and you meet at the library cafe at 5 pm.", "plan", 0.8, ts=t - DAY, subject="Priya",
                           people=["Priya"], event_ts=t + DAY + 3600)
        reply = await brain.handle_query("When am I meeting Priya at the library cafe?")
        assert reply.text == "On Saturday you said 4 pm, later 5 pm." and reply.citations == "1,2"
        sent = [m for _, m in llm.calls if "Question: When am I meeting Priya" in m][0]
        assert "Note: [#1] and [#2] disagree" in sent
        assert json.loads(reply.explain)["conflicts"] == [[1, 2]]
    finally:
        await h.app.stop()


async def test_the_models_rewording_is_fused_with_the_question(tmp_path):
    from kin.brain.llm import ToolCall

    llm = FakeLLM(turns=[ToolCall("recall", {"question": "Rahul diet vegan"}), "Rahul is vegan [#1]."])
    h, brain, _ = await _kin(tmp_path, llm, answer_floor=0.1, strong_score=2.0)
    try:
        await brain.memory.remember("Rahul is vegan now, so no paneer.", "fact", 0.9)
        reply = await brain.handle_query("What can't I cook for him?")
        assert reply.path == "tool: recall" and reply.citations == "1"
        why = json.loads(reply.explain)
        assert why["plan"]["alt"] == "Rahul diet vegan" and "dense:alt" in why["used"][0]["ranks"]
    finally:
        await h.app.stop()


def test_query_plan_describe_is_json():
    p = understand("What did Priya say yesterday?", NOW, NameIndex([Person(1, "Priya")])).with_alt("Priya yesterday")
    d = json.loads(json.dumps(p.describe()))
    assert d["people"] == ["Priya"] and d["time"]["label"] == "yesterday" and d["alt"] == "Priya yesterday"
    assert isinstance(p, QueryPlan) and p.with_alt("what did priya say yesterday?").alt is None
