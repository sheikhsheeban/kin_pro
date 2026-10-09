"""Prompt 4, product readiness: corrections (voice and UI), the change log, the daily review, consent
(only about me, forget a person), encrypted backup and restore, maintenance and health."""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kin.brain.commands import parse_safe_command
from kin.brain.corrections import corrected_event, corrected_text
from kin.memory.backup import BackupError, read_backup, restore_backup, write_backup
from kin.memory.store import MemoryStore
from kin.ui.server import create_app
from tests.conftest import FakeEmbedder, FakeLLM, build_fake_app, make_fake_brain_factory

LOCAL = {"origin": "http://127.0.0.1:8000"}
emb = FakeEmbedder(16)


def _store(tmp_path, dim: int = 16) -> MemoryStore:
    return MemoryStore(tmp_path / "kin.db", secrets.token_hex(32), dim)


def _add(s: MemoryStore, text: str, **kw):
    return s.add_fact(text, "fact", emb.embed_document(text), confidence=0.8, **kw)


async def _kin(tmp_path, llm: FakeLLM | None = None, **memory):
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm or FakeLLM(), None, {}))
    for k, v in memory.items():
        setattr(h.app.cfg.memory, k, v)
    h.app.cfg.memory.retrieval.answer_floor = 0.1
    await h.app.start()
    await h.app.wait_brain()
    return h, h.app.brain


# --- store: every change is logged, nothing is deleted by a correction ----------------------------------


def test_retract_restore_pin_merge_are_logged_and_never_delete(tmp_path):
    s = _store(tmp_path)
    try:
        a = _add(s, "Rahul is vegan.", subject="Rahul")
        b = _add(s, "Rahul does not eat meat.", subject="Rahul")
        assert s.retract(a.id, "voice") and s.get(a.id).status == "retracted" and not s.retract(a.id)
        assert s.restore(a.id) and s.get(a.id).status == "active"
        assert s.set_pinned(a.id, True) and s.get(a.id).pinned
        merged = s.merge(a.id, b.id, "Rahul is vegan and eats no meat.", emb.embed_document("Rahul vegan no meat"))
        assert merged.text == "Rahul is vegan and eats no meat." and s.get(b.id).status == "superseded"
        assert s.count(None) == 2, "merging keeps both rows"
        actions = [(c.fact_id, c.action, c.actor) for c in s.fact_changes(a.id)]
        assert actions == [(a.id, "add", "extracted"), (a.id, "retract", "voice"), (a.id, "restore", "ui"),
                           (a.id, "pin", "ui"), (a.id, "merge", "ui"), (b.id, "merge", "ui")]
        assert s._fts_search("fact_fts", "facts", "meat", 5) == [a.id, b.id] or a.id in s._fts_search("fact_fts", "facts", "eats", 5)
    finally:
        s.close()


def test_forgetting_a_fact_also_forgets_its_history(tmp_path):
    s = _store(tmp_path)
    try:
        a = _add(s, "The door code is 4512.")
        b = _add(s, "The door code is 7788.")
        s.supersede(a.id, b.id)
        s.delete([a.id, b.id])
        assert s._db.execute("SELECT count(*) FROM fact_history").fetchone()[0] == 0, "old texts must not linger"
        raw = (tmp_path / "kin.db").read_bytes()
        assert b"4512" not in raw and b"7788" not in raw
    finally:
        s.close()


def test_fact_retention_keeps_pinned_facts_and_maintenance_checks_integrity(tmp_path):
    s = _store(tmp_path)
    try:
        old = _add(s, "Old plan.", heard_ts=time.time() - 400 * 86400)
        kept = _add(s, "Old but pinned.", heard_ts=time.time() - 400 * 86400)
        s.set_pinned(kept.id, True)
        new = _add(s, "New plan.")
        assert s.purge_facts(time.time() - 365 * 86400) == [old.id]
        assert {f.id for f in s.recent()} == {kept.id, new.id}
        report = s.optimize()
        assert report["integrity"] == "ok" and report["size_after"] > 0
        h = s.health()
        assert h["facts"] == {"active": 2} and h["pinned"] == 1 and h["changes"] >= 3
    finally:
        s.close()


# --- corrections ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("old, said, new", [
    ("Priya and you meet at the library cafe at 4 pm.", "no, it's 5 pm", "Priya and you meet at the library cafe at 5 pm."),
    ("Farewell dinner on Saturday at 8.", "actually make it 8:30", "Farewell dinner on Saturday at 8:30."),
    ("Rahul is vegan.", "Rahul isn't vegan any more", "Rahul isn't vegan any more."),
    ("The rent goes up to 18,000 rupees.", "no, 18,500", "The rent goes up to 18,500 rupees."),
    ("Rahul is vegan.", "no", None),
])
def test_corrected_text(old, said, new):
    assert corrected_text(old, said) == new


def test_corrected_event_keeps_the_day_and_takes_the_new_time():
    four = datetime(2026, 9, 30, 16, 0).timestamp()
    assert datetime.fromtimestamp(corrected_event(four, "Meet at 5:30 pm.")) == datetime(2026, 9, 30, 17, 30)
    assert corrected_event(four, "Meet at the library.") == four and corrected_event(None, "at 5 pm") is None


async def test_voice_correction_acts_on_what_kin_just_cited(tmp_path):
    h, brain = await _kin(tmp_path)
    mem = brain.memory
    try:
        event = datetime(2026, 9, 30, 16, 0).timestamp()
        f = await mem.remember("Priya and you meet at the library cafe at 4 pm.", "plan", 0.9, subject="Priya",
                               people=["Priya"], event_ts=event)
        answer = await brain.handle_query("When am I meeting Priya?")
        assert answer.citations == str(f.id)
        fix = await brain.handle_query("No, it's 5 pm")
        assert fix.path == "tool: correct_fact → correction" and "5 pm" in fix.text
        new = (await mem.recent())[0]
        assert new.text == "Priya and you meet at the library cafe at 5 pm." and new.status == "active"
        assert datetime.fromtimestamp(new.event_ts).hour == 17, "the event time moved with the text"
        old = await mem.get(f.id)
        assert old.status == "superseded" and old.superseded_by == new.id, "kept, not deleted"
        assert [c.action for c in await mem.fact_changes(f.id)][-1] == "correct"

        again = await brain.handle_query("When am I meeting Priya?")
        assert "5 pm" in again.text
        wrong = await brain.handle_query("That's wrong")
        assert "won't use that" in wrong.text and (await mem.get(new.id)).status == "retracted"
        assert await mem.count() == 0 and await mem.count(None) == 2, "retracting needs no confirmation and deletes nothing"
    finally:
        await h.app.stop()


async def test_a_correction_with_nothing_cited_asks_which(tmp_path):
    h, brain = await _kin(tmp_path)
    try:
        reply = await brain.handle_query("That's wrong")
        assert reply.text.startswith("Which memory do you mean?")
    finally:
        await h.app.stop()


async def test_pinned_fact_is_always_offered_for_its_person(tmp_path):
    h, brain = await _kin(tmp_path)
    mem = brain.memory
    try:
        await mem.add_person("Rahul")
        pin = await mem.remember("Rahul's sister Kavya is a doctor.", "fact", 0.9, subject="Rahul", people=["Rahul"])
        for i in range(8):
            await mem.remember(f"Rahul plays cricket on day {i} of the week.", "fact", 0.8, subject="Rahul", people=["Rahul"])
        hits = await mem.retrieve("When does Rahul play cricket?", 3)
        assert pin.id not in [x.fact.id for x in hits]
        await mem.pin(pin.id)
        hits = await mem.retrieve("When does Rahul play cricket?", 3)
        assert pin.id in [x.fact.id for x in hits] and hits[-1].detail.get("pinned")
    finally:
        await h.app.stop()


async def test_daily_review_lists_new_facts_and_changes(tmp_path):
    h, brain = await _kin(tmp_path)
    mem = brain.memory
    try:
        a = await mem.remember("The door code is 4512.", "fact", 0.9)
        await mem.remember("Rahul is vegan.", "fact", 0.9)
        await mem.retract(a.id)
        r = await mem.review()
        assert len(r["added"]) == 2 and r["still_active"] == 1 and [c["action"] for c in r["changes"]] == ["retract"]
        spoken = await brain.handle_query("What did you remember today?")
        assert spoken.path == "tool: review_memory" and spoken.text.startswith("I have 1 new memory today and 1 change.")
        assert "Rahul is vegan" in spoken.text
    finally:
        await h.app.stop()


# --- consent ------------------------------------------------------------------------------------------------


def test_forget_person_is_a_guarded_command():
    assert parse_safe_command("forget everything about Rahul").text == "Rahul"
    assert parse_safe_command("please delete all memories about my Priya").text == "Priya"
    assert parse_safe_command("don't forget everything about Rahul") is None
    assert parse_safe_command("I always forget everything about Rahul") is None


async def test_forget_everything_about_a_person_needs_confirm_and_takes_lines_too(tmp_path):
    h, brain = await _kin(tmp_path)
    mem = brain.memory
    try:
        await mem.add_person("Rahul", ["Raul"])
        await mem.remember("Rahul is vegan.", "fact", 0.9, subject="Rahul")
        await mem.remember("Raul's birthday is on the 22nd.", "fact", 0.9)
        keep = await mem.remember("The door code is 4512.", "fact", 0.9)
        await brain._ingest("Rahul said he moved to Pune.", ts=time.time() - 60)
        ask = await brain.handle_query("Forget everything about Rahul")
        assert ask.text == "That's 2 memories and 1 heard line about Rahul. Say confirm to delete them."
        assert await mem.count() == 3, "nothing deleted before confirm"
        done = await brain.handle_query("confirm")
        assert done.text == "Done. I forgot 2 memories and 1 heard line."
        assert [f.id for f in await mem.recent()] == [keep.id]
        assert not await mem.utterances() and not await mem.people()
    finally:
        await h.app.stop()


async def test_only_about_me_skips_facts_about_other_people(tmp_path):
    llm = FakeLLM()
    h, brain = await _kin(tmp_path, llm)
    mem = brain.memory
    try:
        await mem.set_only_about_me(True)
        llm.extract_replies = [json.dumps([
            {"text": "Rahul is vegan.", "kind": "fact", "subject": "Rahul", "people": ["Rahul"], "event_time": None, "lines": [1]},
            {"text": "The user has a dentist appointment on Friday.", "kind": "plan", "subject": "owner", "people": [],
             "event_time": None, "lines": [1]},
        ])]
        await brain._ingest("Rahul is vegan and I have the dentist on Friday.", ts=time.time() - 500)
        await brain.flush()
        texts = [f.text for f in await mem.recent()]
        assert len(texts) == 1 and "dentist" in texts[0] and "Rahul" not in texts[0]
        assert (await mem.health())["only_about_me"], "the setting is kept in the encrypted database"
    finally:
        await h.app.stop()


# --- backup and restore ---------------------------------------------------------------------------------------


def test_backup_round_trip_is_encrypted_and_needs_the_passphrase(tmp_path):
    s = _store(tmp_path)
    _add(s, "The wifi password is tiger42.")
    out = write_backup(s, tmp_path, "correct horse battery")
    s.close()
    raw = out.read_bytes()
    assert raw.startswith(b"KINBAK1\n") and b"tiger42" not in raw
    header = json.loads(raw.split(b"\n")[1])
    assert header["kdf"] == "scrypt" and len(header["wrapped_key"]) == 64
    with pytest.raises(BackupError, match="wrong passphrase"):
        read_backup(out, "not the passphrase")
    target = tmp_path / "restored"
    info = restore_backup(out, "correct horse battery", target / "kin.db", target / "kin.key")
    key = (target / "kin.key").read_text()
    s2 = MemoryStore(target / "kin.db", key, 16)
    assert s2.recent()[0].text == "The wifi password is tiger42." and info["embedding_dim"] == 16
    s2.close()
    with pytest.raises(BackupError, match="exists"):
        restore_backup(out, "correct horse battery", target / "kin.db", target / "kin.key")
    restore_backup(out, "correct horse battery", target / "kin.db", target / "kin.key", force=True)
    assert (target / "kin.db.before-restore").exists()


def test_backup_refuses_short_passphrases_urls_and_tampering(tmp_path):
    s = _store(tmp_path)
    _add(s, "x is y.")
    try:
        with pytest.raises(BackupError, match="at least"):
            write_backup(s, tmp_path, "short")
        with pytest.raises(BackupError, match="local folder"):
            write_backup(s, "https://example.com/upload", "long enough phrase")
        out = write_backup(s, tmp_path, "long enough phrase")
    finally:
        s.close()
    data = bytearray(out.read_bytes())
    data[-10] ^= 0xFF
    out.write_bytes(bytes(data))
    with pytest.raises(BackupError, match="damaged"):
        read_backup(out, "long enough phrase")


# --- maintenance -------------------------------------------------------------------------------------------


async def test_maintenance_schedule_and_report(tmp_path):
    h, brain = await _kin(tmp_path)
    try:
        m = brain.cfg.memory.maintenance
        brain.last_maintenance = None
        assert brain._maintenance_due(datetime(2026, 10, 8, m.hour, 5), plugged=False)
        assert not brain._maintenance_due(datetime(2026, 10, 8, m.hour - 1, 0), plugged=False)
        assert brain._maintenance_due(datetime(2026, 10, 8, 12, 0), plugged=True), "on mains power: earlier"
        await brain._ingest("An old line about the plumber.", ts=time.time() - 30 * 86400)
        report = await brain.maintain()
        assert report["integrity"] == "ok" and report["purged_lines"] == 1
        assert not brain._maintenance_due(datetime.now(), plugged=True), "not twice within min_interval_h"
        assert (await brain.memory.health())["last_maintenance"]["integrity"] == "ok"
    finally:
        await h.app.stop()


# --- the UI API -----------------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, store={}))
    with TestClient(create_app(h.app)) as c:
        c.portal.call(h.app.wait_brain)
        mem = h.app.brain.memory
        c.portal.call(mem.remember, "I'm meeting Priya at 4 pm at the library cafe.", "commitment", 0.9)
        c.portal.call(mem.remember, "Priya and I meet at the library cafe at 4.", "commitment", 0.9)
        yield c


def test_ui_edit_retract_pin_merge_review_health_and_backup(client, tmp_path):
    a, b = sorted(m["id"] for m in client.get("/api/memories").json()["items"])
    assert client.post(f"/api/memories/{a}/retract", headers={"origin": "https://evil.example"}).status_code == 403
    new = client.post(f"/api/memories/{a}/correct", json={"text": "I'm meeting Priya at 5 pm at the library cafe."},
                      headers=LOCAL).json()
    assert new["text"].endswith("5 pm at the library cafe.") and new["source"] == "explicit"
    assert client.post(f"/api/memories/{new['id']}/pin", json={"pinned": True}, headers=LOCAL).json() == {"pinned": True}
    merged = client.post("/api/memories/merge", json={"keep": new["id"], "other": b}, headers=LOCAL).json()
    assert merged["id"] == new["id"] and merged["pinned"]
    assert client.post(f"/api/memories/{new['id']}/retract", headers=LOCAL).json() == {"retracted": True}
    assert client.post(f"/api/memories/{new['id']}/restore", headers=LOCAL).json() == {"restored": True}
    hist = client.get(f"/api/memories/{new['id']}/history").json()
    assert [c["action"] for c in hist["changes"]][-5:] == ["pin", "merge", "merge", "retract", "restore"]
    review = client.get("/api/review").json()
    assert len(review["added"]) == 3 and {c["action"] for c in review["changes"]} >= {"correct", "merge", "retract"}
    assert client.post("/api/settings", json={"only_about_me": True}, headers=LOCAL).json() == {"only_about_me": True}
    health = client.get("/api/health").json()
    assert health["facts"]["active"] == 1 and health["pinned"] == 1 and health["only_about_me"]
    assert client.post("/api/maintenance", headers=LOCAL).json()["integrity"] == "ok"
    out = client.post("/api/backup", json={"folder": str(tmp_path), "passphrase": "a good long backup phrase", "confirm": "a good long backup phrase"}, headers=LOCAL).json()
    assert Path(out["file"]).exists() and out["bytes"] > 0
    assert client.post("/api/backup", json={"folder": str(tmp_path), "passphrase": "short", "confirm": "short"}, headers=LOCAL).status_code == 400
