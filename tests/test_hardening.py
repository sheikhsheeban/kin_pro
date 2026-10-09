"""Prompt 6 hardening: safe schema upgrades, backup passphrases and the restore drill, crash safety."""

from __future__ import annotations

import secrets

import pytest

import kin.memory.store as store_mod
from kin.memory.backup import read_backup
from kin.memory.store import SCHEMA_VERSION, MemoryStore, MigrationError
from tests.test_memory_v4 import _make_v3_db


def _raw(path, key: str):
    import sqlcipher3

    db = sqlcipher3.connect(str(path))
    db.execute(f"PRAGMA key = \"x'{key}'\"")
    return db


def _version(path, key: str) -> int:
    db = _raw(path, key)
    try:
        return int(db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])
    finally:
        db.close()


def _tables(path, key: str) -> set[str]:
    db = _raw(path, key)
    try:
        return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        db.close()


def _make_v5_db(path, key: str, monkeypatch) -> None:
    """A v5 database: the v3 one upgraded by a Kin whose newest schema was v5."""
    _make_v3_db(path, key)
    monkeypatch.setattr(store_mod, "SCHEMA_VERSION", 5)
    MemoryStore(path, key, 256).close()
    monkeypatch.setattr(store_mod, "SCHEMA_VERSION", SCHEMA_VERSION)
    for b in (path.parent / "backups").glob("*.kinbak"):
        b.unlink()
    assert _version(path, key) == 5


def _fail_at_v6(monkeypatch) -> None:
    real = MemoryStore._to_v6

    def boom(self, stored_dim):
        real(self, stored_dim)  # the step runs, then the disk "fails" before the commit
        raise OSError("disk full (injected)")

    monkeypatch.setattr(MemoryStore, "_to_v6", boom)


# --- migrations: backup first, one transaction, put back on failure ----------------------------------


@pytest.mark.parametrize("start", [3, 5])
def test_upgrade_to_v6_backs_up_first_and_keeps_data(tmp_path, monkeypatch, start):
    key, db = secrets.token_hex(32), tmp_path / "kin.db"
    _make_v3_db(db, key) if start == 3 else _make_v5_db(db, key, monkeypatch)
    s = MemoryStore(db, key, 256)
    try:
        assert sorted(f.id for f in s.recent(10)) == [5, 6, 7]
        assert s._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "6"
    finally:
        s.close()
    backups = list((tmp_path / "backups").glob(f"pre-upgrade-v{start}-to-v6-*.kinbak"))
    assert len(backups) == 1, "one verified backup before the upgrade"
    _header, data, k = read_backup(backups[0], key)  # same format as `kin backup`, unlocked by the key
    assert k == key
    old = tmp_path / "old.db"
    old.write_bytes(data)
    assert _version(old, key) == start, "the backup is the database as it was before the upgrade"
    MemoryStore(db, key, 256).close()
    assert len(list((tmp_path / "backups").glob("*.kinbak"))) == 1, "no backup when nothing is upgraded"


@pytest.mark.parametrize("start", [3, 5])
def test_failed_upgrade_puts_the_database_back_and_stops(tmp_path, monkeypatch, start):
    key, db = secrets.token_hex(32), tmp_path / "kin.db"
    _make_v3_db(db, key) if start == 3 else _make_v5_db(db, key, monkeypatch)
    before = _tables(db, key)
    _fail_at_v6(monkeypatch)
    with pytest.raises(MigrationError) as err:
        MemoryStore(db, key, 256)
    msg = str(err.value)
    assert f"from v{start} to v6 failed" in msg and "disk full (injected)" in msg and "Nothing was lost" in msg
    assert _version(db, key) == start and _tables(db, key) == before, "the database is exactly as it was"
    assert "fact_history" not in _tables(db, key)
    monkeypatch.undo()  # the next start (fixed Kin) upgrades normally, data intact
    s = MemoryStore(db, key, 256)
    try:
        assert sorted(f.id for f in s.recent(10)) == [5, 6, 7]
        assert s.open_tasks()[0].text == "Email the professor."
    finally:
        s.close()


def test_upgrade_does_not_start_without_a_backup(tmp_path, monkeypatch):
    key, db = secrets.token_hex(32), tmp_path / "kin.db"
    _make_v3_db(db, key)

    def no_space(*_a, **_k):
        raise OSError("no space left on device (injected)")

    monkeypatch.setattr("kin.memory.backup.write_backup", no_space)
    with pytest.raises(MigrationError, match="could not back up .* nothing was changed"):
        MemoryStore(db, key, 256)
    assert _version(db, key) == 3


def test_new_database_needs_no_backup(tmp_path):
    MemoryStore(tmp_path / "kin.db", secrets.token_hex(32), 8).close()
    assert not (tmp_path / "backups").exists()


# --- backup passphrase and the restore drill ---------------------------------------------------------


def test_backup_passphrase_strength():
    from kin.memory.backup import passphrase_problem

    for weak in ("short", "abcdefghijk", "123456789012345", "aaaaaaaaaaaaaaaaaaaa", "ababababababababab",
                 "password12345678", "MyPassword1!"):
        assert passphrase_problem(weak), weak
    for strong in ("correct horse battery", "lantern river 42 pebbles", "Tr0ub4dor&3xq", secrets.token_hex(32)):
        assert passphrase_problem(strong) is None, strong


def test_restore_drill_proves_a_backup_without_touching_live_memory(tmp_path):
    import hashlib

    from kin.memory.backup import BackupError, restore_backup, restore_drill, write_backup
    from tests.test_product import _add, _store

    s = _store(tmp_path)
    for t in ("The wifi password is tiger42.", "Rahul is allergic to peanuts.", "Neha moved to Pune."):
        _add(s, t)
    with pytest.raises(BackupError, match="at least"):
        write_backup(s, tmp_path, "too weak")
    out = write_backup(s, tmp_path, "lantern river 42 pebbles")
    s.close()
    live = tmp_path / "kin.db"
    before = hashlib.sha256(live.read_bytes()).hexdigest()

    info = restore_drill(out, "lantern river 42 pebbles")
    assert info["ok"] and info["facts"] == 3 and info["integrity"] == "ok" and info["schema_version"] == 6
    assert hashlib.sha256(live.read_bytes()).hexdigest() == before, "the drill never touches the live database"
    with pytest.raises(BackupError, match="wrong passphrase"):
        restore_drill(out, "lantern river 43 pebbles")
    damaged = tmp_path / "damaged.kinbak"
    raw = bytearray(out.read_bytes())
    raw[-100] ^= 0xFF
    damaged.write_bytes(bytes(raw))
    with pytest.raises(BackupError, match="damaged"):
        restore_drill(damaged, "lantern river 42 pebbles")

    # the full drill: restore onto a new device and use it
    new = tmp_path / "new-device"
    restore_backup(out, "lantern river 42 pebbles", new / "kin.db", new / "kin.key")
    s2 = MemoryStore(new / "kin.db", (new / "kin.key").read_text(), 16)
    try:
        assert sorted(f.text for f in s2.recent()) == sorted(
            ["The wifi password is tiger42.", "Rahul is allergic to peanuts.", "Neha moved to Pune."])
        assert s2._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        s2.close()


def test_ui_backup_needs_the_passphrase_twice_and_checks_backups(tmp_path):
    from fastapi.testclient import TestClient

    from kin.ui.server import create_app
    from tests.conftest import build_fake_app, make_fake_brain_factory
    from tests.test_product import LOCAL

    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, store={}))
    with TestClient(create_app(h.app)) as c:
        c.portal.call(h.app.wait_brain)
        c.portal.call(h.app.brain.memory.remember, "Rahul is allergic to peanuts.", "fact", 0.9)
        good = "lantern river 42 pebbles"
        assert c.post("/api/backup/strength", json={"passphrase": "short"}, headers=LOCAL).json()["problem"]
        assert c.post("/api/backup/strength", json={"passphrase": good}, headers=LOCAL).json()["problem"] is None
        r = c.post("/api/backup", json={"folder": str(tmp_path), "passphrase": good, "confirm": good + "x"}, headers=LOCAL)
        assert r.status_code == 400 and "differ" in r.json()["detail"]
        r = c.post("/api/backup", json={"folder": str(tmp_path), "passphrase": "weak one", "confirm": "weak one"}, headers=LOCAL)
        assert r.status_code == 400 and "at least" in r.json()["detail"]
        out = c.post("/api/backup", json={"folder": str(tmp_path), "passphrase": good, "confirm": good}, headers=LOCAL).json()
        drill = c.post("/api/backup/check", json={"file": out["file"], "passphrase": good}, headers=LOCAL).json()
        assert drill["ok"] and drill["facts"] == 1
        bad = c.post("/api/backup/check", json={"file": out["file"], "passphrase": good + "!"}, headers=LOCAL)
        assert bad.status_code == 400 and "wrong passphrase" in bad.json()["detail"]
        assert c.post("/api/backup/check", json={"file": "https://x.example/b.kinbak", "passphrase": good},
                      headers=LOCAL).status_code == 400


# --- demo data -------------------------------------------------------------------------------------------


def test_demo_week_seeds_resets_and_never_wipes_real_memories(tmp_path):
    from kin.memory import demo_week
    from tests.conftest import FakeEmbedder
    from tests.test_product import _add, _store

    emb = FakeEmbedder(16)
    demo = MemoryStore(tmp_path / "demo.db", secrets.token_hex(32), 16)
    try:
        counts = demo_week.seed(demo, emb.embed_document)
        assert counts["facts"] == len(demo_week.WEEK) and demo.count() == len(demo_week.WEEK) - 1, "one fact was updated"
        assert sum(f.pinned for f in demo.recent(100)) == 3
        assert not demo.ready_episodes(10**10), "seeded lines are not extracted again"
        demo.add_fact("Something added during the demo.", "fact", emb.embed_document("x"), confidence=0.9)
        demo_week.seed(demo, emb.embed_document)  # Reset
        assert demo.count() == len(demo_week.WEEK) - 1
        assert "Something added during the demo." not in [f.text for f in demo.recent(100)]
    finally:
        demo.close()
    real = _store(tmp_path / "real")
    _add(real, "My real secret is 1234.")
    try:
        with pytest.raises(PermissionError, match="not a demo database"):
            demo_week.seed(real, emb.embed_document)
        assert [f.text for f in real.recent()] == ["My real secret is 1234."]
    finally:
        real.close()


def test_demo_reset_endpoint_only_in_demo_mode(tmp_path):
    from fastapi.testclient import TestClient

    from kin.ui.server import create_app
    from tests.conftest import build_fake_app, make_fake_brain_factory
    from tests.test_product import LOCAL

    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, store={}))
    with TestClient(create_app(h.app)) as c:
        c.portal.call(h.app.wait_brain)
        c.portal.call(h.app.brain.memory.remember, "A real memory.", "fact", 0.9)
        assert c.get("/api/demo").json()["demo"] is False
        assert c.post("/api/demo/reset", headers=LOCAL).status_code == 403
        h.app.cfg.demo = True  # even in demo mode, a database that holds real data is refused
        assert c.post("/api/demo/reset", headers=LOCAL).status_code == 403
        assert [f.text for f in c.portal.call(h.app.brain.memory.recent)] == ["A real memory."]


# --- crash safety: the process is killed mid-way; on restart nothing is corrupted and work resumes ---------


def _kill_at(mode: str, *args) -> None:
    """Run tests.crash_child until it reports it is at the crash point, then kill it (TerminateProcess/SIGKILL)."""
    import os
    import subprocess
    import sys

    from tests.conftest import ROOT

    proc = subprocess.Popen([sys.executable, "-m", "tests.crash_child", mode, *map(str, args)], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        for line in proc.stdout:
            if line.strip() == "AT":
                break
        else:
            raise AssertionError(f"child never reached the crash point: {proc.stderr.read()[-3000:]}")
        proc.kill()
    finally:
        proc.wait(timeout=30)
    assert proc.returncode != 0, "the child was killed, not finished"


def _integrity(path, key: str) -> str:
    db = _raw(path, key)
    try:
        return db.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        db.close()


def test_kill_during_a_write_loses_only_the_unfinished_write(tmp_path):
    key, db = secrets.token_hex(32), tmp_path / "kin.db"
    _kill_at("write", db, key)
    assert _integrity(db, key) == "ok"
    s = MemoryStore(db, key, 16)
    try:
        assert sorted(f.text for f in s.recent(10)) == [f"Committed fact number {i}." for i in range(3)]
        assert s._db.execute("SELECT count(*) FROM fact_vec").fetchone()[0] == 3, "no orphan vector"
        assert s._db.execute("SELECT count(*) FROM fact_history").fetchone()[0] == 3
        assert s._db.execute("SELECT count(*) FROM fact_fts").fetchone()[0] == 3, "no orphan keyword row"
        s.add_fact("Written after the crash.", "fact", __import__("numpy").ones(16, "float32"), confidence=0.9)
        assert s.count() == 4
    finally:
        s.close()


@pytest.mark.parametrize("where", ["copy", "file"])
def test_kill_during_a_backup_leaves_memory_intact_and_cleans_up(tmp_path, where):
    from kin.memory.backup import restore_drill, write_backup
    from tests.test_product import _add

    key, db = secrets.token_hex(32), tmp_path / "kin.db"
    s = MemoryStore(db, key, 16)
    _add(s, "Rahul is allergic to peanuts.")
    s.close()
    out = tmp_path / "usb"
    out.mkdir()
    _kill_at(f"backup-{where}", db, key, out)
    assert _integrity(db, key) == "ok"
    assert not list(out.glob("*.kinbak")), "no half-written backup ever looks finished"
    assert bool(list(out.glob("*.part"))) == (where == "file")
    s = MemoryStore(db, key, 16)
    try:
        assert [f.text for f in s.recent()] == ["Rahul is allergic to peanuts."]
        done = write_backup(s, out, "lantern river 42 pebbles")  # the next backup works and tidies up
    finally:
        s.close()
    assert not list(out.glob("*.part")) and [p.name for p in out.glob("*.kinbak")] == [done.name]
    assert restore_drill(done, "lantern river 42 pebbles")["facts"] == 1


async def test_kill_during_fact_extraction_resumes_without_duplicates(tmp_path):
    import json

    from tests.conftest import FakeLLM, build_fake_app, make_fake_brain_factory

    key, db = secrets.token_hex(32), tmp_path / "kin.db"
    _kill_at("extract", db, key)
    assert _integrity(db, key) == "ok"
    s = MemoryStore(db, key, 256)
    try:
        assert [f.text for f in s.recent()] == ["Rahul is allergic to peanuts."], "killed after the first fact"
        assert [e.status for e in s.episodes()] == ["ready"], "the episode was not marked done"
    finally:
        s.close()

    llm = FakeLLM()  # restart: the episode is extracted again, from the start (a model rarely words it the same)
    llm.extract_replies = [json.dumps([
        {"text": "Rahul has a peanut allergy.", "kind": "fact", "subject": "Rahul", "people": ["Rahul"], "event_time": None, "lines": [1]},
        {"text": "The door code is 4512.", "kind": "fact", "subject": None, "people": [], "event_time": None, "lines": [2]},
    ])]
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm=llm, key=key))
    await h.app.start()
    await h.app.wait_brain()
    try:
        brain = h.app.brain
        await brain.extract_ready(wait_idle=False)
        facts = await brain.memory.recent()
        assert sorted(f.text for f in facts) == ["Rahul has a peanut allergy.", "The door code is 4512."], "no duplicate"
        assert await brain.memory.episode_counts() == {"done": 1}
        assert (await brain.maintain())["integrity"] == "ok"
    finally:
        await h.app.stop()


def test_kill_during_a_schema_upgrade_upgrades_cleanly_on_restart(tmp_path):
    key, db = secrets.token_hex(32), tmp_path / "kin.db"
    _make_v3_db(db, key)
    _kill_at("migrate", db, key)
    assert _integrity(db, key) == "ok"
    assert _version(db, key) == 3, "the uncommitted upgrade left the database at v3"
    s = MemoryStore(db, key, 256)
    try:
        assert sorted(f.id for f in s.recent(10)) == [5, 6, 7]
        assert s._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "6"
    finally:
        s.close()
