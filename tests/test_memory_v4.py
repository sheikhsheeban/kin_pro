"""Two-tier memory (schema v4): utterances -> episodes -> extracted facts that are updated, not duplicated.

Fast tests use the fakes (bag-of-words embedder, cue-word router, scripted extractor). The real
Gemma 4 E2B extraction of a split conversation is checked in `test_real_model_merges_fragments`.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from datetime import datetime

import numpy as np
import pytest

from kin.brain.extract import (
    ExtractionError,
    ExtractedFact,
    absolutise,
    build_prompt,
    finish,
    parse_decision,
    parse_facts,
    resolve_speaker,
)
from kin.core.events import EpisodeExtracted, FactSuperseded
from kin.memory.people import NameIndex
from kin.memory.store import SCHEMA_VERSION, MemoryStore, Person, Utterance, WrongKeyError
from tests.conftest import MODELS, FakeEmbedder, FakeLLM, build_fake_app, make_fake_brain_factory

TUE = datetime(2026, 9, 29, 8, 45)  # a Tuesday


def _utt(i: int, text: str, at: datetime = TUE, speaker: str | None = None) -> Utterance:
    return Utterance(i, at.timestamp() + i, text, speaker, 1, "moonshine-small-streaming", True)


# --- schema migration v3 -> v4 ------------------------------------------------------------------------


def _make_v3_db(path, key: str, dim: int = 256) -> list[np.ndarray]:
    """A database exactly as Kin v3 left it: memories + vectors, tasks, and lines waiting in `pending`."""
    import sqlcipher3
    import sqlite_vec

    db = sqlcipher3.connect(str(path))
    db.execute(f"PRAGMA key = \"x'{key}'\"")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.executescript(f"""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('schema_version', '3'), ('embedding_dim', '{dim}');
        CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, kind TEXT NOT NULL,
            speaker TEXT, ts REAL NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL DEFAULT 'passive');
        CREATE INDEX memories_ts ON memories(ts);
        CREATE VIRTUAL TABLE memory_vec USING vec0(embedding float[{dim}] distance_metric=cosine);
        CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('task', 'reminder')), created_ts REAL NOT NULL, due_ts REAL,
            done_ts REAL, fired_ts REAL);
        CREATE TABLE pending (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, ts REAL NOT NULL,
            label TEXT NOT NULL, score REAL NOT NULL, margin REAL NOT NULL);
    """)
    emb = FakeEmbedder(dim)
    texts = ["I'm meeting Priya at 4 pm at the library cafe.", "Rahul is allergic to peanuts.", "What if we paint the hall blue?"]
    vecs = []
    for i, t in enumerate(texts):
        v = emb.embed_document(t)
        vecs.append(v)
        db.execute("INSERT INTO memories(id, text, kind, ts, confidence, source) VALUES (?, ?, 'commitment', ?, 0.9, 'passive')",
                   (i + 5, t, 1_700_000_000 + i))  # ids need not start at 1
        db.execute("INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)", (i + 5, v.astype(np.float32).tobytes()))
    db.execute("INSERT INTO tasks(text, kind, created_ts, due_ts) VALUES ('Email the professor.', 'reminder', 1, 2)")
    db.execute("INSERT INTO pending(text, ts, label, score, margin) VALUES ('The door code is 4512.', 1700000100, 'chatter', 0.5, 0.01)")
    db.execute("INSERT INTO pending(text, ts, label, score, margin) VALUES ('Neha moved to Pune.', 1700000110, 'fact_about_person', 0.6, 0.02)")
    db.commit()
    db.close()
    return vecs


def test_migration_v3_to_v4_keeps_all_data(tmp_path):
    key = secrets.token_hex(32)
    db = tmp_path / "kin.db"
    vecs = _make_v3_db(db, key)
    with pytest.raises(WrongKeyError):
        MemoryStore(db, "0" * 64, 256)
    s = MemoryStore(db, key, 256)
    try:
        assert s._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION) == "6"
        tables = {r[0] for r in s._db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"facts", "utterances", "episodes", "people", "tasks"} <= tables
        assert not {"memories", "memory_vec", "pending"} & tables
        facts = sorted(s.recent(10), key=lambda f: f.id)
        assert [f.id for f in facts] == [5, 6, 7], "same ids: old citations keep working"
        assert all(f.source == "legacy" and f.status == "active" and f.kind == "commitment" for f in facts)
        assert facts[0].heard_ts == 1_700_000_000 and facts[1].people == ("Rahul",)
        hits = s.search(vecs[1], 1, "Rahul peanuts")
        assert hits[0].fact.id == 6 and hits[0].similarity > 0.99, "vectors moved with their facts"
        assert s.open_tasks()[0].text == "Email the professor."
        lines = s.utterances()
        assert sorted(u.text for u in lines) == ["Neha moved to Pune.", "The door code is 4512."]
        ready = s.ready_episodes(time.time())
        assert len(ready) == 1 and ready[0].lines == 2, "lines that waited for the LLM are extracted after the upgrade"
        assert s.search_utterances("door code")[0].text == "The door code is 4512."
    finally:
        s.close()
    s = MemoryStore(db, key, 256)  # opening again is a no-op
    assert s.count() == 3
    s.close()


def test_new_database_is_v4_and_has_fts5(tmp_path):
    s = MemoryStore(tmp_path / "kin.db", secrets.token_hex(32), 8)
    assert s.fts5, "this SQLCipher build ships FTS5"
    assert s._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION) == "6"
    s.close()


def test_keyword_search_finds_what_the_vector_missed_and_fallback_works(tmp_path):
    s = MemoryStore(tmp_path / "kin.db", secrets.token_hex(32), 4)
    try:
        far = np.array([1, 0, 0, 0], np.float32)
        s.add_fact("Neha's extension number is 2314.", "fact", np.array([0, 1, 0, 0], np.float32), confidence=0.8)
        for i in range(30):  # vector neighbours of the query that crowd out the right fact
            s.add_fact(f"Chatter number {i}.", "note", far + np.float32(i) * 0.001, confidence=0.8)
        hits = s.search(far, 5, "What is Neha's extension?")
        assert any("2314" in h.fact.text for h in hits)
        s.fts5 = False  # without FTS5: trigram scan in Python
        assert s._fts_search("fact_fts", "facts", "Neha extension", 3)[0] == 1
    finally:
        s.close()


# --- extraction: schema, time resolution, speakers ----------------------------------------------------


def test_extraction_reply_is_validated():
    ok = parse_facts('```json\n[{"text": "Rahul is vegan.", "kind": "fact", "subject": "Rahul", "people": ["Rahul"], '
                     '"event_time": null, "lines": [1, 9]}]\n```', 2)
    assert ok[0].subject == "Rahul" and ok[0].lines == [1], "out-of-range line numbers are dropped"
    assert parse_facts("[]", 3) == []
    for bad in ("Sure! Here are the facts.", '[{"kind": "fact"}]', '[{"text": "x", "event_time": "next Friday"}]',
                '[{"text": "x", "people": "Rahul"}]', "[1, 2]"):
        with pytest.raises(ExtractionError):
            parse_facts(bad, 3)


def test_relative_times_resolve_from_when_the_line_was_heard():
    lines = [_utt(1, "Let's meet tomorrow at 4."), _utt(2, "Where?"), _utt(3, "Library cafe.")]
    f = ExtractedFact("Someone said: let's meet tomorrow at 4 at the library cafe.", "plan", None, [], None, [1, 2, 3])
    finish([f], lines)
    assert datetime.fromtimestamp(f.event_ts) == datetime(2026, 9, 30, 16, 0), "'at 4' means 4 pm"
    assert "Wednesday 30 September 2026" in f.text and "tomorrow" not in f.text
    # the model's own wrong guess is overridden by the line's words
    g = ExtractedFact("Dentist on Friday at 11:30.", "plan", "owner", [], datetime(2026, 9, 25, 11, 30).timestamp(), [1])
    finish([g], [_utt(1, "I have a dentist appointment on Friday at 11:30.", speaker="owner")])
    assert datetime.fromtimestamp(g.event_ts) == datetime(2026, 10, 2, 11, 30)
    assert absolutise("Mentoring every Thursday at 3.", TUE)[0] == "Mentoring every Thursday at 3."
    assert absolutise("Demo moved to next Tuesday at 10.", TUE)[0] == "Demo moved to Tuesday 6 October 2026 at 10."
    assert absolutise("Badminton on Wednesday nights.", TUE)[0] == "Badminton on Wednesday nights."
    h = ExtractedFact("Mentoring every Thursday at 3.", "plan", None, [], None, [1])
    finish([h], [_utt(1, "Let's do our mentoring sessions every Thursday at 3.")])
    assert h.event_ts is None, "a recurring plan has no single event time"


def test_first_person_becomes_the_speaker_only_when_the_speaker_is_known():
    owner = ExtractedFact("My gym locker code is 4471.", "note", "owner", [], None, [1])
    resolve_speaker(owner, [_utt(1, "My gym locker code is 4471.", speaker="owner")])
    assert owner.text == "The user's gym locker code is 4471." and owner.subject == "owner"
    unknown = ExtractedFact("I'm vegan now.", "fact", "owner", [], None, [1])
    resolve_speaker(unknown, [_utt(1, "I'm vegan now, so no paneer for me.")])
    assert unknown.text == "Someone said: I'm vegan now." and unknown.subject is None, "never guessed to be the owner"
    guessed = ExtractedFact("The user is vegan and does not want paneer.", "fact", None, [], None, [1])  # real E2B output
    resolve_speaker(guessed, [_utt(1, "I'm vegan now, so no paneer for me.")])
    assert guessed.text == "Someone is vegan and does not want paneer."
    addressed = ExtractedFact("Sam will return the user's umbrella on Saturday.", "plan", "Sam", ["Sam"], None, [1])
    resolve_speaker(addressed, [_utt(1, "I'll return your umbrella on Saturday.")])
    assert addressed.text == "Sam will return the user's umbrella on Saturday.", "'your' = the owner was addressed"
    other = ExtractedFact("I'm allergic to peanuts.", "fact", None, [], None, [1])
    resolve_speaker(other, [_utt(1, "I'm allergic to peanuts.", speaker="other")])
    assert other.text.startswith("Someone said:")
    third = ExtractedFact("Rahul is vegan.", "fact", "Rahul", ["Rahul"], None, [1])
    resolve_speaker(third, [_utt(1, "Rahul is vegan.")])
    assert third.text == "Rahul is vegan." and third.subject == "Rahul"


def test_prompt_carries_times_speakers_calendar_and_known_people():
    prompt = build_prompt([_utt(1, "Let's meet tomorrow at 4.", speaker="other"), _utt(2, "Library cafe.")],
                          datetime(2026, 9, 29, 23, 0), ["Priya", "Rahul"])
    assert "[1] Tue 08:45 other: Let's meet tomorrow at 4." in prompt
    assert "[2] Tue 08:45 unknown: Library cafe." in prompt
    assert "tomorrow Wed 30 Sep 2026" in prompt and "Known people: Priya, Rahul" in prompt


def test_dedup_decisions_never_act_on_an_invented_id():
    assert parse_decision("UPDATE 12", {12, 13}) == ("UPDATE", 12)
    assert parse_decision("DUPLICATE <13>", {12, 13}) == ("DUPLICATE", 13)
    assert parse_decision("CONTRADICTS [#12]", {12}) == ("CONTRADICTS", 12)
    assert parse_decision("UPDATE 99", {12, 13}) == ("ADD", None)
    assert parse_decision("I think these are the same", {12}) == ("ADD", None)


# --- names ---------------------------------------------------------------------------------------


def test_misheard_names_are_normalised_but_similar_names_stay_apart():
    idx = NameIndex([Person(1, "Priya"), Person(2, "Priyanka"), Person(3, "Rahul", ("Rahool",))], 80)
    text, fixes = idx.normalise_text("Pria and Raul met Priyanca at Rahool's flat on Monday.")
    assert text == "Priya and Rahul met Priyanka at Rahul's flat on Monday."
    assert fixes == {"Pria": "Priya", "Raul": "Rahul", "Priyanca": "Priyanka", "Rahool": "Rahul"}
    assert idx.canonical("Priyanka") == "Priyanka" and idx.canonical("Priya") == "Priya"
    assert idx.canonical("Pune") is None and idx.canonical("The") is None


# --- the pipeline through the Brain (fakes) ------------------------------------------------------------


async def _brain(tmp_path, llm: FakeLLM | None = None):
    store: dict = {}
    llm = llm or FakeLLM()
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, None, store))
    await h.app.start()
    await h.app.wait_brain()
    return h, h.app.brain, llm, store


async def _hear(brain, text: str, at: float, speaker: str | None = None) -> None:
    await brain._ingest(text, ts=at, speaker=speaker)


async def test_episodes_group_lines_and_filler_is_dropped(tmp_path):
    h, brain, llm, _ = await _brain(tmp_path)
    try:
        t = time.time() - 5000
        await _hear(brain, "Let's meet tomorrow at 4.", t)
        await _hear(brain, "Okay.", t + 5)  # filler: not even an utterance
        await _hear(brain, "Where?", t + 10)
        await _hear(brain, "Library cafe.", t + 15)
        await _hear(brain, "The match was boring.", t + 200)  # 185 s of quiet: a new episode
        await _hear(brain, "I left my keys at Sam's place.", t + 200 + 700)  # > 10 min after the episode began
        lines = await brain.memory.utterances()
        assert sorted(u.text for u in lines) == sorted(
            ["Let's meet tomorrow at 4.", "Where?", "Library cafe.", "The match was boring.", "I left my keys at Sam's place."])
        by_ep: dict[int, list[str]] = {}
        for u in sorted(lines, key=lambda u: u.ts):
            by_ep.setdefault(u.episode_id, []).append(u.text)
        assert list(by_ep.values()) == [["Let's meet tomorrow at 4.", "Where?", "Library cafe."], ["The match was boring."],
                                        ["I left my keys at Sam's place."]]
        counts = await brain.memory.episode_counts()
        assert counts == {"ready": 1, "skipped": 1, "open": 1}, "an all-chatter episode is skipped without the LLM"
        await brain.flush()
        assert llm.extractions == 2  # the plan and the keys; never the match
    finally:
        await h.app.stop()


async def test_fragments_merge_into_one_fact_with_all_evidence(tmp_path):
    llm = FakeLLM()
    llm.extract_replies = [json.dumps([{
        "text": "Someone said: let's meet tomorrow at 4 at the library cafe.", "kind": "plan", "subject": None,
        "people": [], "event_time": None, "lines": [1, 2, 3]}])]
    h, brain, _, _ = await _brain(tmp_path, llm)
    try:
        t = TUE.timestamp()
        for i, text in enumerate(["Let's meet tomorrow at 4.", "Where?", "Library cafe."]):
            await _hear(brain, text, t + i)
        await brain.flush()
        facts = await brain.memory.recent()
        assert len(facts) == 1
        f = facts[0]
        assert "library cafe" in f.text and "Wednesday 30 September 2026" in f.text
        assert datetime.fromtimestamp(f.event_ts) == datetime(2026, 9, 30, 16, 0)
        lines = await brain.memory.utterances(list(f.evidence))
        assert [u.text for u in lines] == ["Let's meet tomorrow at 4.", "Where?", "Library cafe."]
        assert f.source == "extracted" and f.episode_id is not None
    finally:
        await h.app.stop()


async def test_update_chain_supersedes_and_keeps_history(tmp_path):
    llm = FakeLLM()
    h, brain, _, _ = await _brain(tmp_path, llm)
    mem = brain.memory
    try:
        t = time.time() - 9000
        reply = lambda text: json.dumps([{"text": text, "kind": "plan", "subject": "Priya", "people": ["Priya"],  # noqa: E731
                                          "event_time": None, "lines": [1]}])
        llm.extract_replies = [reply("Priya and the user meet at the library cafe at 16:00 on Wednesday 30 September 2026.")]
        await _hear(brain, "Meeting Priya at the library cafe at 4 on Wednesday.", t)
        await brain.flush()
        v1 = (await mem.recent())[0]

        llm.extract_replies = [reply("Priya and the user meet at the library cafe at 17:00 on Wednesday 30 September 2026.")]
        llm.dedup_answer = f"UPDATE {v1.id}"
        await _hear(brain, "Priya says make it 5 instead of 4.", t + 1000)
        await brain.flush()
        v2 = (await mem.recent())[0]

        llm.extract_replies = [reply("Priya and the user meet at the library cafe at 18:00 on Wednesday 30 September 2026.")]
        llm.dedup_answer = f"CONTRADICTS {v2.id}"
        await _hear(brain, "Priya says 5 doesn't work, 6 then.", t + 2000)
        await brain.flush()
        v3 = (await mem.recent())[0]

        assert await mem.count() == 1 and await mem.count(None) == 3, "superseded facts are kept, not deleted"
        chain = await mem.history(v1.id)
        assert [f.id for f in chain] == [v1.id, v2.id, v3.id]
        assert [f.status for f in chain] == ["superseded", "superseded", "active"]
        assert chain[0].superseded_by == v2.id and chain[1].superseded_by == v3.id
        assert [e.new_id for e in h.rec.of(FactSuperseded)] == [v2.id, v3.id]
        hits = await mem.retrieve("When am I meeting Priya?", 5)
        assert [x.fact.id for x in hits] == [v3.id], "only the latest version is offered to answers"

        llm.extract_replies = [reply("Priya and the user meet at the library cafe at 18:00 on Wednesday 30 September 2026.")]
        llm.dedup_answer = f"DUPLICATE {v3.id}"
        await _hear(brain, "So it's 6 at the cafe with Priya.", t + 3000)
        await brain.flush()
        assert await mem.count(None) == 3
        assert len((await mem.get(v3.id)).evidence) == 2, "a duplicate adds its evidence to the existing fact"
        assert h.rec.of(EpisodeExtracted)[-1].duplicates == 1

        await mem.delete([v3.id])  # the user forgets the latest version: the previous one is active again
        assert [f.id for f in await mem.recent()] == [v2.id]
    finally:
        await h.app.stop()


async def test_extraction_retries_once_then_keeps_the_episode_for_later(tmp_path):
    llm = FakeLLM()
    h, brain, _, _ = await _brain(tmp_path, llm)
    try:
        llm.extract_replies = ["Here you go!", '[{"text": "The door code is 4512.", "kind": "note", "subject": null, '
                               '"people": [], "event_time": null, "lines": [1]}]']
        await _hear(brain, "The door code is 4512.", time.time() - 500)
        await brain.flush()
        assert llm.extractions == 2 and (await brain.memory.recent())[0].text == "The door code is 4512."

        llm.extract_replies = ["no", "still no"]
        await _hear(brain, "The bike lock code is 7788.", time.time() - 100)
        await brain.flush()
        failed = h.rec.of(EpisodeExtracted)[-1]
        assert failed.outcome == "failed" and "after one retry" in failed.error
        assert await brain.memory.episode_counts() == {"done": 1, "failed": 1}
        assert brain.pending_count == 1, "kept for a later attempt, not dropped"
        assert await brain.extract_ready(wait_idle=False) == 0, "not retried before retry_failed_after_s"
    finally:
        await h.app.stop()


async def test_misheard_name_in_a_new_fact_is_normalised_and_recorded(tmp_path):
    llm = FakeLLM()
    h, brain, _, _ = await _brain(tmp_path, llm)
    try:
        await brain.memory.add_person("Priya")
        llm.extract_replies = [json.dumps([{"text": "Pria is moving to Bangalore in December.", "kind": "fact",
                                            "subject": "Pria", "people": ["Pria"], "event_time": None, "lines": [1]}])]
        await _hear(brain, "Pria is moving to Bangalore in December.", time.time() - 300)
        await brain.flush()
        f = (await brain.memory.recent())[0]
        assert f.text == "Priya is moving to Bangalore in December." and f.subject == "Priya" and f.people == ("Priya",)
        people = await brain.memory.people()
        assert [(p.name, p.aliases) for p in people] == [("Priya", ("Pria",))]
        hits = await brain.memory.retrieve("Where is Pria moving?", 1)  # a misheard question is normalised too
        assert hits and hits[0].fact.id == f.id
    finally:
        await h.app.stop()


async def test_retention_purges_old_raw_lines_but_keeps_facts(tmp_path):
    h, brain, _, _ = await _brain(tmp_path)
    mem = brain.memory
    try:
        now = time.time()
        await _hear(brain, "The wifi password is tiger42.", now - 20 * 86400)
        await brain.flush()
        await _hear(brain, "Rahul is vegan now.", now - 3600)
        await brain.flush()
        assert await mem.run(lambda s: s.utterance_count()) == 2 and await mem.count() == 2
        assert await mem.purge_utterances(14, now=now) == 1
        assert [u.text for u in await mem.utterances()] == ["Rahul is vegan now."]
        assert await mem.count() == 2, "facts outlive their raw lines"
        old = next(f for f in await mem.recent() if "tiger42" in f.text)
        assert await mem.utterances(list(old.evidence)) == [], "its evidence was purged"
        assert await mem.search_utterances("tiger42") == []
        assert (await mem.episode_counts()) == {"done": 1}
    finally:
        await h.app.stop()


async def test_no_plaintext_in_the_database_file(tmp_path):
    h, brain, _, store = await _brain(tmp_path)
    try:
        await brain.memory.add_person("Zorawar", ["Zorawer"])
        await _hear(brain, "The wifi password is tiger42.", time.time() - 300)
        await _hear(brain, "Quentin said the safe code is 9137.", time.time() - 290)
        await brain.flush()
        await _hear(brain, "Kin heard this line about the marmalade but nobody extracted it yet.", time.time())
        await brain.drain()
        assert await brain.memory.count() >= 1
    finally:
        await h.app.stop()
    raw = store["db"].read_bytes()
    assert not raw.startswith(b"SQLite format 3")
    for secret in (b"tiger42", b"9137", b"Quentin", b"Zorawar", b"Zorawer", b"marmalade", b"password"):
        assert secret not in raw, secret
    assert not list(tmp_path.glob("*-journal")) and not list(tmp_path.glob("*-wal"))


# --- the real model ------------------------------------------------------------------------------------


@pytest.mark.models
@pytest.mark.skipif(not (MODELS / "llm" / "gemma-4-E2B-it.litertlm").exists(), reason="models not downloaded")
async def test_real_model_merges_fragments_and_resolves_time():
    from kin.brain.extract import Extractor
    from kin.brain.llm import create_llm
    from kin.config import load_config

    cfg = load_config("laptop")
    llm = await asyncio.to_thread(create_llm, cfg)
    try:
        ex = Extractor(lambda: llm)
        lines = [_utt(1, "Let's meet tomorrow at 4."), _utt(2, "Where?"), _utt(3, "Library cafe."),
                 _utt(4, "I'm vegan now, so no paneer for me.")]
        t0 = time.monotonic()
        res = await asyncio.to_thread(ex.extract, lines, datetime(2026, 9, 29, 23, 0), ["Priya", "Rahul"])
        print(f"\nextraction {time.monotonic() - t0:.1f} s, {res.llm_calls} call(s): {res.raw[-1]!r}")
        for f in res.facts:
            print(f"  {f.text!r} event={f.event_ts and datetime.fromtimestamp(f.event_ts)} lines={f.lines} notes={f.notes}")
        plan = [f for f in res.facts if "library" in f.text.lower()]
        assert plan, "the place from the third line is merged into the plan"
        assert plan[0].event_ts and datetime.fromtimestamp(plan[0].event_ts) == datetime(2026, 9, 30, 16, 0)
        assert "tomorrow" not in plan[0].text.lower()
        vegan = [f for f in res.facts if "vegan" in f.text.lower()]
        assert vegan and vegan[0].subject != "owner" and not vegan[0].text.lower().startswith("the user")
    finally:
        await asyncio.to_thread(llm.close)


# --- review fixes: day-aware retrieval, dated weekdays, gate only for unsure episodes ---------------------


def test_day_expressions_become_windows():
    from kin.memory.timescope import day_window

    now = datetime(2026, 10, 5, 9, 0)  # a Monday
    day = lambda w: datetime.fromtimestamp(w[0]).strftime("%a %d")  # noqa: E731
    assert day(day_window("What did I plan yesterday?", now)) == "Sun 04"
    assert day(day_window("What did Mum tell me last Monday?", now)) == "Mon 28"
    assert day(day_window("Who did I meet at the cafe on Wednesday?", now)) == "Wed 30", "past tense looks back"
    assert day(day_window("When is my meeting on Friday?", now)) == "Fri 09", "otherwise ahead"
    assert day_window("When is my dentist appointment?", now) is None
    assert absolutise("The quarterly review is on Monday the 12th at 2 pm.", now)[0] == "The quarterly review is on Monday the 12th at 2 pm."


def test_window_facts_join_the_candidates(tmp_path):
    s = MemoryStore(tmp_path / "kin.db", secrets.token_hex(32), 4)
    try:
        q = np.array([1, 0, 0, 0], np.float32)
        monday = datetime(2026, 9, 28, 19, 0).timestamp()
        target = s.add_fact("Anil's wedding is on 14 November in Pune.", "fact", np.array([0, 1, 0, 0], np.float32),
                            confidence=0.8, heard_ts=monday)
        for i in range(10):
            s.add_fact(f"Close but other day {i}.", "note", q + np.float32(i) * 0.01, confidence=0.8, heard_ts=monday + 86400 * 2)
        assert target.id not in [h.fact.id for h in s.search(q, 5)]
        window = (datetime(2026, 9, 28).timestamp(), datetime(2026, 9, 29).timestamp())
        hits = s.search(q, 5, window=window)
        assert target.id in [h.fact.id for h in hits] and hits[0].similarity > 0.9, "the best cosine match still leads"
    finally:
        s.close()


async def test_gate_is_only_asked_for_episodes_that_look_like_chatter(tmp_path):
    llm = FakeLLM()
    llm.keep_answer = "no"
    h, brain, _, _ = await _brain(tmp_path, llm)
    try:
        await _hear(brain, "Rahul is vegan now.", time.time() - 900)  # the router says fact: likely worth keeping
        await brain.flush()
        await _hear(brain, "This dal is way too spicy.", time.time() - 300)  # the stub router: confident chatter
        await brain.flush()
        outcomes = [(e.outcome, e.llm_calls) for e in h.rec.of(EpisodeExtracted)]
        assert outcomes[0][0] == "done", "never gated: the gate wrongly said no to such lines in the benchmark"
        assert outcomes[1] == ("skipped", 0), "confident chatter is skipped without any LLM call"
        assert [f.text for f in await brain.memory.recent()] == ["Rahul is vegan now."]
    finally:
        await h.app.stop()


def test_model_keeps_pm_when_the_line_only_says_at_8():
    sat = datetime(2026, 10, 3, 12, 0)
    f = ExtractedFact("Farewell dinner on Saturday 10 October 2026 at 20:00.", "plan", None, [],
                      datetime(2026, 10, 10, 20, 0).timestamp(), [1])
    finish([f], [_utt(1, "Let's do a farewell dinner before I move, next Saturday at 8.", at=sat)])
    assert datetime.fromtimestamp(f.event_ts) == datetime(2026, 10, 10, 20, 0)
