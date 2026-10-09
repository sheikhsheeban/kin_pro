"""Encrypted memory store: SQLCipher 4 with sqlite-vec and FTS5, in one file.

Two tiers (schema v4; v5 adds a vector per raw line for the transcript fallback; v6 adds pinned facts and
fact_history, the log of every change to a fact: added, corrected, retracted, pinned, merged, replaced):
  utterances  every overheard line that was not filler: the raw evidence. Grouped into
              `episodes` (conversations). Purged after a retention period (default 14 days).
  facts       self-contained statements extracted from episodes (or said explicitly with
              "remember that ..."). A changed fact is never deleted: the old one is marked
              `superseded` and points at its replacement, so the history stays visible.
  people      names Kin knows, with the misheard spellings seen for each (aliases).

- Every page is encrypted with a raw 256-bit key (no passphrase KDF on open), including the
  vec0 vectors and the FTS5 indexes, which live in the same database.
- Recall gathers candidates here (`candidates`: nearest vectors, BM25 keyword matches, facts in a time
  window, facts about named people); ranking happens in retrieval.py.
- A new embedding size (`memory.dim`) drops the vectors and the service re-embeds the text once.
- `secure_delete` is on, so forgotten rows are overwritten in the file, not just unlinked.
- A fact and its vector and full-text entry are written and deleted together.

Not thread-safe by design: create and use it on one thread (the memory worker).
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCHEMA_VERSION = 6
_HEX_KEY = re.compile(r"^[0-9a-fA-F]{64}$")

FACT_STATUSES = ("active", "superseded", "retracted")
SPEAKERS = ("owner", "other")
WORTH_CHATTER, WORTH_UNSURE, WORTH_LIKELY = 0, 1, 2
# open: still collecting lines; ready: closed, waiting for extraction; skipped: the pre-filter found
# nothing worth extracting; done: extracted; failed: extraction failed (retried later)
EPISODE_STATUSES = ("open", "ready", "skipped", "done", "failed")


class MigrationError(RuntimeError):
    """A schema upgrade failed; the database was put back from its pre-upgrade backup."""


class WrongKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    kind: str
    subject: str | None  # a person's name, 'owner', or None (unknown)
    people: tuple[str, ...]
    event_ts: float | None  # when the thing happens (a plan's date-time), if any
    heard_ts: float  # when Kin heard it
    status: str
    superseded_by: int | None
    confidence: float
    source: str  # extracted | explicit | legacy
    evidence: tuple[int, ...]  # utterance ids
    episode_id: int | None = None
    pinned: bool = False  # always offered when a question names this fact's person

    @property
    def ts(self) -> float:
        return self.heard_ts

    @property
    def about_other(self) -> bool:
        """About another person (not the owner, not unknown): shown marked, and skipped with "only about me"."""
        return self.subject not in (None, "", "owner")

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["people"], d["evidence"], d["ts"] = list(self.people), list(self.evidence), self.heard_ts
        d["about_other"] = self.about_other
        return d


_FACT_COLS = ("id, text, kind, subject, people, event_ts, heard_ts, status, superseded_by, confidence, source, evidence, "
              "episode_id, pinned")
HISTORY_ACTIONS = ("add", "correct", "retract", "restore", "pin", "unpin", "merge", "supersede")


def _fact(row) -> Fact:
    r = list(row)
    r[4] = tuple(json.loads(r[4] or "[]"))
    r[11] = tuple(json.loads(r[11] or "[]"))
    r[13] = bool(r[13])
    return Fact(*r)


@dataclass(frozen=True)
class Change:
    """One row of fact_history."""

    id: int
    fact_id: int
    ts: float
    action: str  # see HISTORY_ACTIONS
    actor: str  # extract | explicit | voice | ui | maintenance
    old_text: str | None
    new_text: str | None
    other_id: int | None  # the replacing / merged / new fact

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Utterance:
    id: int
    ts: float
    text: str
    speaker: str | None  # owner | other | None (unknown: never guessed)
    episode_id: int
    stt_model: str | None
    worth: int  # pre-filter level: 0 confident chatter, 1 unsure, 2 likely worth keeping (see WORTH_*)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_UTT_COLS = "id, ts, text, speaker, episode_id, stt_model, worth"


@dataclass(frozen=True)
class Episode:
    id: int
    start_ts: float
    end_ts: float
    status: str
    attempts: int
    worth: int  # the highest level of its lines
    lines: int
    next_try_ts: float | None
    error: str | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_EP_COLS = "id, start_ts, end_ts, status, attempts, worth, lines, next_try_ts, error"


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    source: str = "extracted"  # extracted | user

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "aliases": list(self.aliases), "source": self.source}


@dataclass(frozen=True)
class Task:
    id: int
    text: str
    kind: str  # task | reminder
    created_ts: float
    due_ts: float | None
    done_ts: float | None
    fired_ts: float | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_TASK_COLS = "id, text, kind, created_ts, due_ts, done_ts, fired_ts"


@dataclass(frozen=True)
class Hit:
    fact: Fact
    similarity: float  # cosine to the question (EmbeddingGemma)
    score: float | None = None  # the confidence the gate thresholds (hybrid retrieval); None: plain vector search
    detail: dict = field(default_factory=dict, compare=False, hash=False)  # per-list ranks and scores, for the UI


def load_or_create_key(key_file: Path) -> str:
    """64 hex chars. KIN_DB_KEY wins; otherwise a key file is created once with owner-only permissions."""
    env = os.environ.get("KIN_DB_KEY")
    if env:
        if not _HEX_KEY.match(env):
            raise ValueError("KIN_DB_KEY must be 64 hex characters (a 256-bit key)")
        return env.lower()
    key_file = Path(key_file)
    if key_file.exists():
        key = key_file.read_text(encoding="ascii").strip()
        if not _HEX_KEY.match(key):
            raise ValueError(f"{key_file} does not contain a 256-bit hex key")
        return key.lower()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(key)
    log.info("created new memory encryption key at %s", key_file)
    return key


_FTS_WORD = re.compile(r"[A-Za-z0-9]{2,}")
_FTS_STOP = frozenset(
    "the and for are was were what when where who how does did will with about you your have has had this that "
    "from they them their there here which tell know my me is am do it of to in on at an be so we our us can any "
    "whats who's what's me remember told said say an".split()
)


def fts_query(text: str) -> str | None:
    """A safe FTS5 query: content words OR-ed and quoted (no FTS syntax from user text reaches SQLite)."""
    words = [w for w in _FTS_WORD.findall(text.lower()) if w not in _FTS_STOP]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words)) or None


class MemoryStore:
    def __init__(self, path: Path | str, key_hex: str, dim: int, *, reembed: bool = False) -> None:
        """`reembed`: if the database holds vectors of another size, drop them and let the caller
        re-embed (`missing_vectors` / `set_vector`) instead of refusing to open."""
        import sqlcipher3
        import sqlite_vec

        if not _HEX_KEY.match(key_hex):
            raise ValueError("key must be 64 hex characters")
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.dim = dim
        self._db = sqlcipher3.connect(self.path)
        self._db.execute(f"PRAGMA key = \"x'{key_hex}'\"")
        self._key_hex = key_hex
        try:
            self._db.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlcipher3.DatabaseError as exc:
            self._db.close()
            raise WrongKeyError(f"cannot open {path}: wrong key or not a Kin database") from exc
        # cipher_memory_security stays OFF: it mlocks every allocation, which exceeds the
        # working-set / RLIMIT_MEMLOCK quota on Windows and on a Pi and kills the process.
        self._db.execute("PRAGMA secure_delete = ON")
        self._db.execute("PRAGMA temp_store = MEMORY")  # FTS merges and sorts never spill to a plaintext temp file
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        self.fts5 = self._has_fts5()
        self._reembed = reembed
        self.rebuilt_from_dim: int | None = None  # set when vectors were dropped for a new size
        self._migrate()

    def _has_fts5(self) -> bool:
        opts = {r[0] for r in self._db.execute("PRAGMA compile_options")}
        if "ENABLE_FTS5" not in opts:
            log.warning("this SQLCipher build has no FTS5: keyword search falls back to a scan in Python")
            return False
        return True

    # --- schema ------------------------------------------------------------------

    def _migrate(self) -> None:
        db = self._db
        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        fresh = row is None
        if row is None:  # a new database is created at v1 and migrated like an old one
            with db:
                db.execute(
                    """CREATE TABLE memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        speaker TEXT,
                        ts REAL NOT NULL,
                        confidence REAL NOT NULL,
                        source TEXT NOT NULL DEFAULT 'passive'
                    )"""
                )
                db.execute("CREATE INDEX memories_ts ON memories(ts)")
                db.execute(f"CREATE VIRTUAL TABLE memory_vec USING vec0(embedding float[{self.dim}] distance_metric=cosine)")
                db.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    [("schema_version", "1"), ("embedding_dim", str(self.dim))],
                )
            row = ("1",)
        version = int(row[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"database schema v{version} is newer than this Kin (v{SCHEMA_VERSION})")
        stored_dim = int(db.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()[0])
        if stored_dim != self.dim and not self._reembed:
            raise RuntimeError(f"database holds {stored_dim}-d embeddings but config asks for {self.dim}-d")
        if stored_dim != self.dim and version < 4:
            raise RuntimeError("upgrade to schema v4 at the old embedding size first")  # v1-v3 migrations copy vectors
        if version < SCHEMA_VERSION:
            self._upgrade(version, stored_dim, fresh)
        if stored_dim != self.dim:
            self._rebuild_vectors(stored_dim)

    def _upgrade(self, version: int, stored_dim: int, fresh: bool) -> None:
        """Run every step from `version` to SCHEMA_VERSION in one transaction. An existing database is first
        backed up (same format as `kin backup`, passphrase = the key) and the backup is verified; on any
        failure the transaction is rolled back, the file is put back from the backup and Kin stops."""
        from kin.memory.backup import pre_upgrade_backup, put_back

        backup = None
        if not fresh and self.path != ":memory:":
            try:
                backup = pre_upgrade_backup(self, version, SCHEMA_VERSION)
            except Exception as exc:
                self._db.close()
                raise MigrationError(f"could not back up the memory database before upgrading it from v{version} to "
                                     f"v{SCHEMA_VERSION} ({exc}); nothing was changed") from exc
            log.info("memory database v%d -> v%d: backup %s", version, SCHEMA_VERSION, backup)
        db = self._db
        db.commit()
        level, db.isolation_level = db.isolation_level, None  # one explicit transaction for all the steps
        try:
            db.execute("BEGIN IMMEDIATE")
            for n in range(version + 1, SCHEMA_VERSION + 1):
                getattr(self, f"_to_v{n}")(stored_dim)
                db.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(n),))
            db.execute("COMMIT")
        except BaseException as exc:
            if db.in_transaction:
                db.execute("ROLLBACK")
            db.close()
            if backup is None:
                raise
            try:
                put_back(backup, self.path, self._key_hex)
            except Exception as again:
                raise MigrationError(f"upgrading the memory database from v{version} to v{SCHEMA_VERSION} failed "
                                     f"({exc}), and putting it back failed too ({again}). Restore it with: "
                                     f"kin restore \"{backup}\" --force --key-passphrase") from exc
            raise MigrationError(f"upgrading the memory database from v{version} to v{SCHEMA_VERSION} failed ({exc}). "
                                 f"Nothing was lost: the database is back as it was, and a copy is in {backup}. "
                                 "Kin stopped; please report this error.") from exc
        finally:
            if db.isolation_level is None:
                try:
                    db.isolation_level = level
                except Exception:
                    pass

    def _to_v2(self, stored_dim: int) -> None:
        """v2: tasks and reminders (Phase 3)."""
        db = self._db
        db.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('task', 'reminder')),
                created_ts REAL NOT NULL,
                due_ts REAL,
                done_ts REAL,
                fired_ts REAL
            )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS tasks_due ON tasks(due_ts) WHERE done_ts IS NULL AND fired_ts IS NULL")

    def _to_v3(self, stored_dim: int) -> None:
        """v3: overheard lines the semantic router was unsure about, waiting for the LLM."""
        db = self._db
        db.execute(
            """CREATE TABLE IF NOT EXISTS pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                ts REAL NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                margin REAL NOT NULL
            )"""
        )

    def _to_v5(self, stored_dim: int) -> None:
        """v5: vectors for raw lines, so the transcript fallback searches like facts do."""
        db = self._db
        db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS utterance_vec USING vec0(embedding float[{stored_dim}] distance_metric=cosine)")

    def _to_v6(self, stored_dim: int) -> None:
        """v6: pinned facts and the change log behind corrections and the daily review."""
        db = self._db
        db.execute("ALTER TABLE facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        db.execute(
            """CREATE TABLE fact_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                old_text TEXT,
                new_text TEXT,
                other_id INTEGER
            )"""
        )
        db.execute("CREATE INDEX fact_history_ts ON fact_history(ts)")
        db.execute("CREATE INDEX fact_history_fact ON fact_history(fact_id)")

    def _rebuild_vectors(self, old_dim: int) -> None:
        """A new embedding size: drop every vector (facts and lines) and record the new size. The text
        stays; `missing_vectors` lists what to re-embed. Searching falls back to keywords meanwhile."""
        with self._db:
            for table in ("fact_vec", "utterance_vec"):
                self._db.execute(f"DROP TABLE IF EXISTS {table}")
                self._db.execute(f"CREATE VIRTUAL TABLE {table} USING vec0(embedding float[{self.dim}] distance_metric=cosine)")
            self._db.execute("UPDATE meta SET value = ? WHERE key = 'embedding_dim'", (str(self.dim),))
        self.rebuilt_from_dim = old_dim
        log.info("embedding size changed %d -> %d: vectors dropped, to be re-embedded", old_dim, self.dim)

    def missing_vectors(self, table: str, limit: int = 100_000) -> list[tuple[int, str]]:
        """(id, text) of facts ('facts') or lines ('utterances') that have no vector yet."""
        vec = {"facts": "fact_vec", "utterances": "utterance_vec"}[table]
        return [(int(i), t) for i, t in self._db.execute(
            f"SELECT id, text FROM {table} WHERE id NOT IN (SELECT rowid FROM {vec}) ORDER BY id LIMIT ?", (limit,))]

    def set_vector(self, table: str, rowid: int, vector: np.ndarray) -> None:
        vec = {"facts": "fact_vec", "utterances": "utterance_vec"}[table]
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"embedding must have shape ({self.dim},), got {v.shape}")
        with self._db:
            self._db.execute(f"DELETE FROM {vec} WHERE rowid = ?", (rowid,))
            self._db.execute(f"INSERT INTO {vec}(rowid, embedding) VALUES (?, ?)", (rowid, v.tobytes()))

    def vectors(self, ids: list[int]) -> dict[int, np.ndarray]:
        if not ids:
            return {}
        rows = self._db.execute(f"SELECT rowid, embedding FROM fact_vec WHERE rowid IN ({','.join('?' * len(ids))})", ids)
        return {int(r): np.frombuffer(e, dtype=np.float32) for r, e in rows}

    def _to_v4(self, stored_dim: int) -> None:
        """v4: utterances + episodes + facts (+ people). Memories become legacy facts with the same ids
        and vectors; lines that were waiting in `pending` become utterances of one episode to extract."""
        db = self._db
        db.execute(
            """CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'ready', 'skipped', 'done', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                worth INTEGER NOT NULL DEFAULT 0,
                lines INTEGER NOT NULL DEFAULT 0,
                next_try_ts REAL,
                error TEXT
            )"""
        )
        db.execute("CREATE INDEX episodes_status ON episodes(status, start_ts)")
        db.execute(
            """CREATE TABLE utterances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                text TEXT NOT NULL,
                speaker TEXT CHECK (speaker IN ('owner', 'other')),
                episode_id INTEGER NOT NULL REFERENCES episodes(id),
                stt_model TEXT,
                worth INTEGER NOT NULL DEFAULT 1
            )"""
        )
        db.execute("CREATE INDEX utterances_ts ON utterances(ts)")
        db.execute("CREATE INDEX utterances_episode ON utterances(episode_id)")
        db.execute(
            """CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                kind TEXT NOT NULL,
                subject TEXT,
                people TEXT NOT NULL DEFAULT '[]',
                event_ts REAL,
                heard_ts REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'retracted')),
                superseded_by INTEGER REFERENCES facts(id),
                confidence REAL NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('extracted', 'explicit', 'legacy')),
                evidence TEXT NOT NULL DEFAULT '[]',
                episode_id INTEGER
            )"""
        )
        db.execute("CREATE INDEX facts_heard ON facts(heard_ts)")
        db.execute("CREATE INDEX facts_status ON facts(status)")
        db.execute(f"CREATE VIRTUAL TABLE fact_vec USING vec0(embedding float[{self.dim}] distance_metric=cosine)")
        db.execute(
            """CREATE TABLE people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                aliases TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'extracted',
                created_ts REAL NOT NULL
            )"""
        )
        if self.fts5:
            db.execute("CREATE VIRTUAL TABLE fact_fts USING fts5(text, tokenize='porter unicode61')")
            db.execute("CREATE VIRTUAL TABLE utterance_fts USING fts5(text, tokenize='porter unicode61')")

        # memories -> legacy facts, same ids (old citations and the UI keep working), same vectors
        names_of = _legacy_names
        rows = db.execute("SELECT id, text, kind, ts, confidence FROM memories ORDER BY id").fetchall()
        for mid, text, kind, ts, conf in rows:
            people = names_of(text)
            db.execute(
                "INSERT INTO facts(id, text, kind, subject, people, event_ts, heard_ts, status, confidence, source, evidence) "
                "VALUES (?, ?, ?, NULL, ?, NULL, ?, 'active', ?, 'legacy', '[]')",
                (mid, text, kind, json.dumps(people), ts, conf),
            )
            self._fts_add("fact_fts", mid, text)
        db.execute("INSERT INTO fact_vec(rowid, embedding) SELECT rowid, embedding FROM memory_vec")
        # unsure lines still waiting for the LLM -> one episode, ready for extraction
        pend = db.execute("SELECT text, ts FROM pending ORDER BY ts, id").fetchall()
        if pend:
            cur = db.execute(
                "INSERT INTO episodes(start_ts, end_ts, status, worth, lines) VALUES (?, ?, 'ready', 1, ?)",
                (pend[0][1], pend[-1][1], len(pend)),
            )
            eid = int(cur.lastrowid)
            for text, ts in pend:
                uid = db.execute(
                    "INSERT INTO utterances(ts, text, speaker, episode_id, stt_model, worth) VALUES (?, ?, NULL, ?, NULL, 1)",
                    (ts, text, eid),
                ).lastrowid
                self._fts_add("utterance_fts", int(uid), text)
        db.execute("DROP TABLE memory_vec")
        db.execute("DROP TABLE memories")
        db.execute("DROP TABLE pending")
        log.info("memory store migrated to schema v4 (%d legacy facts, %d pending lines)", len(rows), len(pend))

    # --- full-text helpers -------------------------------------------------------------

    def _fts_add(self, table: str, rowid: int, text: str) -> None:
        if self.fts5:
            self._db.execute(f"INSERT INTO {table}(rowid, text) VALUES (?, ?)", (rowid, text))

    def _fts_delete(self, table: str, rowids: list[int]) -> None:
        if self.fts5 and rowids:
            self._db.execute(f"DELETE FROM {table} WHERE rowid IN ({','.join('?' * len(rowids))})", rowids)

    def _fts_search(self, table: str, base: str, text: str, limit: int) -> list[int]:
        q = fts_query(text)
        if q is None:
            return []
        if self.fts5:
            rows = self._db.execute(f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?", (q, limit))
            return [r[0] for r in rows]
        # Fallback without FTS5: trigram overlap computed in Python over the (decrypted) rows.
        words = [w.strip('"') for w in q.split(" OR ")]
        grams = {w[i : i + 3] for w in words for i in range(max(1, len(w) - 2))}
        scored = []
        for rid, body in self._db.execute(f"SELECT id, text FROM {base}"):
            low = body.lower()
            score = sum(g in low for g in grams)
            if score:
                scored.append((score, rid))
        return [rid for _, rid in sorted(scored, reverse=True)[:limit]]

    # --- tasks & reminders ---------------------------------------------------------

    def add_task(self, text: str, kind: str, due_ts: float | None = None) -> Task:
        now = time.time()
        with self._db:
            cur = self._db.execute(
                "INSERT INTO tasks(text, kind, created_ts, due_ts) VALUES (?, ?, ?, ?)", (text, kind, now, due_ts)
            )
        return Task(int(cur.lastrowid), text, kind, now, due_ts, None, None)

    def open_tasks(self) -> list[Task]:
        rows = self._db.execute(
            f"SELECT {_TASK_COLS} FROM tasks WHERE done_ts IS NULL AND fired_ts IS NULL "
            "ORDER BY due_ts IS NULL, due_ts, created_ts"
        )
        return [Task(*r) for r in rows]

    def due_reminders(self, now: float) -> list[Task]:
        rows = self._db.execute(
            f"SELECT {_TASK_COLS} FROM tasks WHERE kind = 'reminder' AND done_ts IS NULL AND fired_ts IS NULL "
            "AND due_ts <= ? ORDER BY due_ts",
            (now,),
        )
        return [Task(*r) for r in rows]

    def next_due(self) -> float | None:
        row = self._db.execute(
            "SELECT min(due_ts) FROM tasks WHERE kind = 'reminder' AND done_ts IS NULL AND fired_ts IS NULL"
        ).fetchone()
        return row[0]

    def mark_fired(self, task_id: int, ts: float) -> None:
        with self._db:
            self._db.execute("UPDATE tasks SET fired_ts = ? WHERE id = ?", (ts, task_id))

    def complete_task(self, task_id: int) -> bool:
        with self._db:
            cur = self._db.execute("UPDATE tasks SET done_ts = ? WHERE id = ? AND done_ts IS NULL", (time.time(), task_id))
        return cur.rowcount > 0

    def delete_task(self, task_id: int) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount > 0

    def get_task(self, task_id: int) -> Task | None:
        r = self._db.execute(f"SELECT {_TASK_COLS} FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return Task(*r) if r else None

    # --- utterances & episodes -------------------------------------------------------------

    def add_utterance(
        self, text: str, ts: float, *, speaker: str | None = None, stt_model: str | None = None, worth: int = WORTH_LIKELY,
        gap_s: float = 90.0, max_s: float = 600.0, vector: np.ndarray | None = None,
    ) -> tuple[Utterance, list[Episode]]:
        """Store one heard line in the current episode (a new one after `gap_s` of silence or `max_s` in all).

        Returns the utterance and any episode this line closed.
        """
        if speaker not in (None, *SPEAKERS):
            raise ValueError(f"speaker must be one of {SPEAKERS} or None")
        with self._db:
            closed = self._close_open(ts, gap_s, max_s)
            row = self._db.execute(f"SELECT {_EP_COLS} FROM episodes WHERE status = 'open' ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                eid = int(self._db.execute(
                    "INSERT INTO episodes(start_ts, end_ts, status, worth, lines) VALUES (?, ?, 'open', 0, 0)", (ts, ts)
                ).lastrowid)
            else:
                eid = row[0]
            self._db.execute(
                "UPDATE episodes SET end_ts = max(end_ts, ?), lines = lines + 1, worth = max(worth, ?) WHERE id = ?",
                (ts, int(worth), eid),
            )
            uid = int(self._db.execute(
                "INSERT INTO utterances(ts, text, speaker, episode_id, stt_model, worth) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, text, speaker, eid, stt_model, int(worth)),
            ).lastrowid)
            self._fts_add("utterance_fts", uid, text)
            if vector is not None and np.asarray(vector).shape == (self.dim,):
                self._db.execute("INSERT INTO utterance_vec(rowid, embedding) VALUES (?, ?)",
                                 (uid, np.asarray(vector, dtype=np.float32).tobytes()))
        return Utterance(uid, ts, text, speaker, eid, stt_model, worth), closed

    def _close_open(self, now: float, gap_s: float, max_s: float, force: bool = False) -> list[Episode]:
        closed = []
        for r in self._db.execute(f"SELECT {_EP_COLS} FROM episodes WHERE status = 'open'").fetchall():
            ep = Episode(*r)
            if force or now - ep.end_ts > gap_s or now - ep.start_ts > max_s:
                status = "ready" if ep.worth else "skipped"
                self._db.execute("UPDATE episodes SET status = ? WHERE id = ?", (status, ep.id))
                closed.append(Episode(*r[:3], status, *r[4:]))
        return closed

    def close_idle_episodes(self, now: float, gap_s: float, max_s: float, *, force: bool = False) -> list[Episode]:
        with self._db:
            return self._close_open(now, gap_s, max_s, force)

    def ready_episodes(self, now: float, limit: int = 50) -> list[Episode]:
        rows = self._db.execute(
            f"SELECT {_EP_COLS} FROM episodes WHERE status = 'ready' "
            "OR (status = 'failed' AND (next_try_ts IS NULL OR next_try_ts <= ?)) ORDER BY start_ts, id LIMIT ?",
            (now, limit),
        )
        return [Episode(*r) for r in rows]

    def get_episode(self, eid: int) -> Episode | None:
        r = self._db.execute(f"SELECT {_EP_COLS} FROM episodes WHERE id = ?", (eid,)).fetchone()
        return Episode(*r) if r else None

    def episodes(self, limit: int = 100) -> list[Episode]:
        return [Episode(*r) for r in self._db.execute(f"SELECT {_EP_COLS} FROM episodes ORDER BY id DESC LIMIT ?", (limit,))]

    def episode_counts(self) -> dict[str, int]:
        return dict(self._db.execute("SELECT status, count(*) FROM episodes GROUP BY status").fetchall())

    def mark_episode(self, eid: int, status: str, *, error: str | None = None, next_try_ts: float | None = None) -> None:
        if status not in EPISODE_STATUSES:
            raise ValueError(status)
        with self._db:
            self._db.execute(
                "UPDATE episodes SET status = ?, error = ?, next_try_ts = ?, attempts = attempts + ? WHERE id = ?",
                (status, error, next_try_ts, int(status in ("done", "failed")), eid),
            )

    def episode_fact_ids(self, eid: int) -> list[int]:
        """Facts extraction already stored from episode `eid`."""
        return [int(r[0]) for r in self._db.execute(
            "SELECT id FROM facts WHERE episode_id = ? AND source = 'extracted' ORDER BY id", (eid,))]

    def episode_lines(self, eid: int) -> list[Utterance]:
        rows = self._db.execute(f"SELECT {_UTT_COLS} FROM utterances WHERE episode_id = ? ORDER BY ts, id", (eid,))
        return [_utt(r) for r in rows]

    def utterances(self, ids: list[int] | None = None, limit: int = 200) -> list[Utterance]:
        if ids is not None:
            if not ids:
                return []
            rows = self._db.execute(f"SELECT {_UTT_COLS} FROM utterances WHERE id IN ({','.join('?' * len(ids))}) ORDER BY ts", ids)
        else:
            rows = self._db.execute(f"SELECT {_UTT_COLS} FROM utterances ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
        return [_utt(r) for r in rows]

    def search_utterances(self, text: str, limit: int = 20) -> list[Utterance]:
        ids = self._fts_search("utterance_fts", "utterances", text, limit)
        by_id = {u.id: u for u in self.utterances(ids)}
        return [by_id[i] for i in ids if i in by_id]

    def utterance_count(self) -> int:
        return int(self._db.execute("SELECT count(*) FROM utterances").fetchone()[0])

    def purge_utterances(self, before_ts: float) -> int:
        """Retention: delete raw lines older than `before_ts`, and episodes left without lines.

        Facts stay; their evidence ids then point at purged lines (shown as such in the UI).
        """
        with self._db:
            ids = [r[0] for r in self._db.execute("SELECT id FROM utterances WHERE ts < ?", (before_ts,))]
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                self._fts_delete("utterance_fts", chunk)
                self._db.execute(f"DELETE FROM utterance_vec WHERE rowid IN ({','.join('?' * len(chunk))})", chunk)
                self._db.execute(f"DELETE FROM utterances WHERE id IN ({','.join('?' * len(chunk))})", chunk)
            self._db.execute(
                "DELETE FROM episodes WHERE status != 'open' AND NOT EXISTS (SELECT 1 FROM utterances u WHERE u.episode_id = episodes.id)"
            )
        return len(ids)

    # --- facts: writes -------------------------------------------------------------------

    def add_fact(
        self,
        text: str,
        kind: str,
        embedding: np.ndarray,
        *,
        confidence: float,
        source: str = "extracted",
        subject: str | None = None,
        people: list[str] | tuple[str, ...] = (),
        event_ts: float | None = None,
        heard_ts: float | None = None,
        evidence: list[int] | tuple[int, ...] = (),
        episode_id: int | None = None,
        actor: str | None = None,
    ) -> Fact:
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"embedding must have shape ({self.dim},), got {vec.shape}")
        heard_ts = time.time() if heard_ts is None else heard_ts
        people = tuple(dict.fromkeys(people))
        evidence = tuple(dict.fromkeys(int(e) for e in evidence))
        with self._db:
            fid = int(self._db.execute(
                "INSERT INTO facts(text, kind, subject, people, event_ts, heard_ts, status, confidence, source, evidence, episode_id) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)",
                (text, kind, subject, json.dumps(list(people)), event_ts, heard_ts, confidence, source,
                 json.dumps(list(evidence)), episode_id),
            ).lastrowid)
            self._db.execute("INSERT INTO fact_vec(rowid, embedding) VALUES (?, ?)", (fid, vec.tobytes()))
            self._fts_add("fact_fts", fid, text)
            self._log(fid, "add", actor or source, None, text)
        return Fact(fid, text, kind, subject, people, event_ts, heard_ts, "active", None, confidence, source, evidence, episode_id)

    def supersede(self, old_id: int, new_id: int, status: str = "superseded", *, actor: str = "extract",
                  action: str = "supersede") -> bool:
        """Mark `old_id` replaced by `new_id`. Nothing is deleted: the chain stays visible."""
        if status not in ("superseded", "retracted"):
            raise ValueError(status)
        old, new = self.get(old_id), self.get(new_id)
        with self._db:
            cur = self._db.execute(
                "UPDATE facts SET status = ?, superseded_by = ? WHERE id = ? AND status = 'active' AND id != ?",
                (status, new_id, old_id, new_id),
            )
            if cur.rowcount:
                self._log(old_id, action, actor, old.text if old else None, new.text if new else None, new_id)
        return cur.rowcount > 0

    # --- corrections (Prompt 4): nothing here deletes; every change is logged in fact_history ---------------

    def _log(self, fact_id: int, action: str, actor: str, old_text: str | None, new_text: str | None,
             other_id: int | None = None, ts: float | None = None) -> None:
        self._db.execute(
            "INSERT INTO fact_history(fact_id, ts, action, actor, old_text, new_text, other_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fact_id, time.time() if ts is None else ts, action, actor, old_text, new_text, other_id),
        )

    def retract(self, fid: int, actor: str = "ui") -> bool:
        """"That's wrong": the fact stops being used for answers. Not deleted (the user can restore it)."""
        f = self.get(fid)
        if f is None or f.status != "active":
            return False
        with self._db:
            self._db.execute("UPDATE facts SET status = 'retracted', superseded_by = NULL WHERE id = ?", (fid,))
            self._log(fid, "retract", actor, f.text, None)
        return True

    def restore(self, fid: int, actor: str = "ui") -> bool:
        f = self.get(fid)
        if f is None or f.status != "retracted":
            return False
        with self._db:
            self._db.execute("UPDATE facts SET status = 'active', superseded_by = NULL WHERE id = ?", (fid,))
            self._log(fid, "restore", actor, None, f.text)
        return True

    def set_pinned(self, fid: int, pinned: bool, actor: str = "ui") -> bool:
        f = self.get(fid)
        if f is None or f.pinned == pinned:
            return False
        with self._db:
            self._db.execute("UPDATE facts SET pinned = ? WHERE id = ?", (int(pinned), fid))
            self._log(fid, "pin" if pinned else "unpin", actor, None, None)
        return True

    def merge(self, keep_id: int, other_id: int, text: str | None, embedding: np.ndarray | None, actor: str = "ui") -> Fact | None:
        """Two facts that say the same thing: `keep_id` takes the other's evidence and people (and `text`, if
        given); the other is marked superseded by it. Both texts stay in the history."""
        keep, other = self.get(keep_id), self.get(other_id)
        if keep is None or other is None or keep_id == other_id or other.status != "active" or keep.status != "active":
            return None
        new_text = (text or keep.text).strip()
        people = list(dict.fromkeys([*keep.people, *other.people]))
        evidence = list(dict.fromkeys([*keep.evidence, *other.evidence]))
        with self._db:
            self._db.execute("UPDATE facts SET text = ?, people = ?, evidence = ?, pinned = ? WHERE id = ?",
                             (new_text, json.dumps(people), json.dumps(evidence), int(keep.pinned or other.pinned), keep_id))
            if new_text != keep.text:
                self._fts_delete("fact_fts", [keep_id])
                self._fts_add("fact_fts", keep_id, new_text)
                if embedding is not None:
                    self._db.execute("DELETE FROM fact_vec WHERE rowid = ?", (keep_id,))
                    self._db.execute("INSERT INTO fact_vec(rowid, embedding) VALUES (?, ?)",
                                     (keep_id, np.asarray(embedding, dtype=np.float32).tobytes()))
            self._db.execute("UPDATE facts SET status = 'superseded', superseded_by = ? WHERE id = ?", (keep_id, other_id))
            self._log(keep_id, "merge", actor, keep.text, new_text, other_id)
            self._log(other_id, "merge", actor, other.text, new_text, keep_id)
        return self.get(keep_id)

    def changes(self, since: float, until: float | None = None, limit: int = 500) -> list[Change]:
        rows = self._db.execute(
            "SELECT id, fact_id, ts, action, actor, old_text, new_text, other_id FROM fact_history "
            "WHERE ts >= ? AND ts < ? ORDER BY ts, id LIMIT ?", (since, until or 1e18, limit))
        return [Change(*r) for r in rows]

    def fact_changes(self, fid: int) -> list[Change]:
        rows = self._db.execute(
            "SELECT id, fact_id, ts, action, actor, old_text, new_text, other_id FROM fact_history "
            "WHERE fact_id = ? OR other_id = ? ORDER BY ts, id", (fid, fid))
        return [Change(*r) for r in rows]

    def ids_about(self, names: list[str]) -> list[int]:
        """Every fact (any status) whose subject, people or text names one of these spellings."""
        out = []
        pats = [re.compile(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", re.I) for n in names if n.strip()]
        for fid, text, subject, people in self._db.execute("SELECT id, text, subject, people FROM facts"):
            hay = " ".join([text, subject or "", *json.loads(people or "[]")])
            if any(p.search(hay) for p in pats):
                out.append(int(fid))
        return out

    def utterance_ids_about(self, names: list[str]) -> list[int]:
        pats = [re.compile(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", re.I) for n in names if n.strip()]
        return [int(uid) for uid, text in self._db.execute("SELECT id, text FROM utterances") if any(p.search(text) for p in pats)]

    def delete_utterances(self, ids: list[int]) -> int:
        with self._db:
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                marks = ",".join("?" * len(chunk))
                self._fts_delete("utterance_fts", chunk)
                self._db.execute(f"DELETE FROM utterance_vec WHERE rowid IN ({marks})", chunk)
                self._db.execute(f"DELETE FROM utterances WHERE id IN ({marks})", chunk)
        return len(ids)

    def purge_facts(self, before_ts: float) -> list[int]:
        """Fact retention (off by default): delete facts heard before `before_ts`, except pinned ones."""
        ids = [r[0] for r in self._db.execute("SELECT id FROM facts WHERE heard_ts < ? AND pinned = 0", (before_ts,))]
        return self.delete(ids)

    # --- maintenance ------------------------------------------------------------------------------------

    def optimize(self) -> dict:
        """FTS merge, VACUUM (secure_delete is on, so freed pages were already overwritten), integrity check."""
        t0 = time.perf_counter()
        before = self.file_size()
        with self._db:
            if self.fts5:
                for table in ("fact_fts", "utterance_fts"):
                    self._db.execute(f"INSERT INTO {table}({table}) VALUES ('optimize')")
        self._db.execute("VACUUM")
        check = [r[0] for r in self._db.execute("PRAGMA integrity_check")]
        return {"integrity": "ok" if check == ["ok"] else "; ".join(check[:5]), "size_before": before,
                "size_after": self.file_size(), "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def file_size(self) -> int:
        return os.path.getsize(self.path) if self.path != ":memory:" and os.path.exists(self.path) else 0

    def health(self) -> dict:
        q = lambda sql, *a: self._db.execute(sql, a).fetchone()[0]  # noqa: E731
        return {
            "facts": dict(self._db.execute("SELECT status, count(*) FROM facts GROUP BY status").fetchall()),
            "pinned": q("SELECT count(*) FROM facts WHERE pinned = 1"),
            "about_others": q("SELECT count(*) FROM facts WHERE status = 'active' AND subject IS NOT NULL AND subject NOT IN ('', 'owner')"),
            "utterances": q("SELECT count(*) FROM utterances"),
            "episodes": dict(self._db.execute("SELECT status, count(*) FROM episodes GROUP BY status").fetchall()),
            "people": q("SELECT count(*) FROM people"),
            "changes": q("SELECT count(*) FROM fact_history"),
            "db_bytes": self.file_size(),
            "last_fact_ts": q("SELECT max(heard_ts) FROM facts"),
            "last_extraction_ts": q("SELECT max(ts) FROM fact_history WHERE action = 'add' AND actor = 'extracted'"),
        }

    def get_meta(self, key: str) -> str | None:
        r = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                             (key, value))

    def backup_to(self, path: Path | str) -> None:
        """A consistent copy of the whole encrypted database (same key), via SQLite's online backup."""
        import sqlcipher3

        key = self._key_hex
        dst = sqlcipher3.connect(str(path))
        try:
            dst.execute(f"PRAGMA key = \"x'{key}'\"")
            self._db.backup(dst)
        finally:
            dst.close()

    def add_evidence(self, fact_id: int, utterance_ids: list[int]) -> None:
        f = self.get(fact_id)
        if f is None:
            return
        merged = list(dict.fromkeys([*f.evidence, *utterance_ids]))
        with self._db:
            self._db.execute("UPDATE facts SET evidence = ? WHERE id = ?", (json.dumps(merged), fact_id))

    def delete(self, ids: list[int]) -> list[int]:
        """The user's forget: really deletes. A fact that had replaced an older one hands the role back."""
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with self._db:
            found = [r[0] for r in self._db.execute(f"SELECT id FROM facts WHERE id IN ({marks})", ids)]
            if found:
                fm = ",".join("?" * len(found))
                self._db.execute(
                    f"UPDATE facts SET status = 'active', superseded_by = NULL WHERE superseded_by IN ({fm}) AND id NOT IN ({fm})",
                    found + found,
                )
                self._db.execute(f"DELETE FROM fact_vec WHERE rowid IN ({fm})", found)
                self._fts_delete("fact_fts", found)
                self._db.execute(f"DELETE FROM facts WHERE id IN ({fm})", found)
                # the change log holds old texts: a forgotten fact takes its history with it
                self._db.execute(f"DELETE FROM fact_history WHERE fact_id IN ({fm}) OR other_id IN ({fm})", found + found)
        return found

    def delete_since(self, since_ts: float) -> list[int]:
        """Forget everything heard since `since_ts`: facts, raw lines, and their episodes."""
        deleted = self.delete(self.ids_since(since_ts))
        with self._db:
            uids = [r[0] for r in self._db.execute("SELECT id FROM utterances WHERE ts >= ?", (since_ts,))]
            self._fts_delete("utterance_fts", uids)
            self._db.execute("DELETE FROM utterance_vec WHERE rowid IN (SELECT id FROM utterances WHERE ts >= ?)", (since_ts,))
            self._db.execute("DELETE FROM utterances WHERE ts >= ?", (since_ts,))
            self._db.execute("DELETE FROM episodes WHERE NOT EXISTS (SELECT 1 FROM utterances u WHERE u.episode_id = episodes.id)")
        return deleted

    def delete_utterances_since(self, since_ts: float) -> int:
        with self._db:
            uids = [r[0] for r in self._db.execute("SELECT id FROM utterances WHERE ts >= ?", (since_ts,))]
            self._fts_delete("utterance_fts", uids)
            self._db.execute("DELETE FROM utterance_vec WHERE rowid IN (SELECT id FROM utterances WHERE ts >= ?)", (since_ts,))
            self._db.execute("DELETE FROM utterances WHERE ts >= ?", (since_ts,))
            self._db.execute("DELETE FROM episodes WHERE NOT EXISTS (SELECT 1 FROM utterances u WHERE u.episode_id = episodes.id)")
        return len(uids)

    def ids_since(self, since_ts: float) -> list[int]:
        return [r[0] for r in self._db.execute("SELECT id FROM facts WHERE heard_ts >= ? ORDER BY id", (since_ts,))]

    # --- facts: reads ---------------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int, text: str | None = None, *, status: str = "active",
        window: tuple[float, float] | None = None,
    ) -> list[Hit]:
        """Best `k` facts. Candidates: the cosine neighbours, keyword (FTS5) matches, and facts heard or
        happening inside `window` (a day named in the question). The similarity reported is always the cosine."""
        if self.count(status=None) == 0:
            return []
        qv = np.asarray(query, dtype=np.float32).tobytes()
        knn = max(k * 4, 20)
        sims: dict[int, float] = {
            rid: 1.0 - float(dist)
            for rid, dist in self._db.execute("SELECT rowid, distance FROM fact_vec WHERE embedding MATCH ? AND k = ?", (qv, knn))
        }
        keyword = self._fts_search("fact_fts", "facts", text, k * 2) if text else []
        in_window: list[int] = []
        if window is not None:
            in_window = [r[0] for r in self._db.execute(
                "SELECT id FROM facts WHERE status = 'active' AND ((heard_ts >= ? AND heard_ts < ?) OR (event_ts >= ? AND event_ts < ?)) "
                "LIMIT 200",
                (window[0], window[1], window[0], window[1]),
            )]
        extra = [i for i in dict.fromkeys(keyword + in_window) if i not in sims]
        for i in range(0, len(extra), 200):
            chunk = extra[i : i + 200]
            marks = ",".join("?" * len(chunk))
            for rid, dist in self._db.execute(
                f"SELECT rowid, vec_distance_cosine(embedding, ?) FROM fact_vec WHERE rowid IN ({marks})", [qv, *chunk]
            ):
                sims[rid] = 1.0 - float(dist)
        if not sims:
            return []
        marks = ",".join("?" * len(sims))
        sql = f"SELECT {_FACT_COLS} FROM facts WHERE id IN ({marks})"
        args: list = list(sims)
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        facts = {f.id: f for f in (_fact(r) for r in self._db.execute(sql, args))}
        ranked = sorted(facts, key=lambda i: sims[i], reverse=True)
        # Hybrid: the closest facts from the named day keep up to three places, the best keyword matches
        # (a name or a number) up to two, even if their cosine is low; the rest go by cosine. One place
        # always goes by cosine. The list is ordered by cosine, so the gate's floor sees the best vector match.
        dated = sorted((i for i in in_window if i in facts), key=lambda i: sims[i], reverse=True)[:3]
        must = list(dict.fromkeys(dated + [i for i in keyword if i in facts][:2]))[: max(0, k - 1)]
        chosen = must + [i for i in ranked if i not in must][: k - len(must)]
        return sorted((Hit(facts[i], round(sims[i], 4)) for i in chosen), key=lambda h: h.similarity, reverse=True)

    def candidates(
        self, queries: dict[str, tuple[np.ndarray, str]], n: int, *, history: bool = False,
        window: tuple[float, float] | None = None, people: tuple[str, ...] = (),
    ) -> tuple[dict[int, Fact], dict[str, dict[int, float]], dict[str, list[int]]]:
        """The candidate pool for hybrid retrieval, and the raw lists to fuse.

        `queries` maps a name ("q", "alt") to (query vector, query text). The pool is, per query, the `n`
        nearest vectors and the `n` best BM25 keyword matches, plus every fact heard or happening in
        `window` and every fact about one of `people`. Superseded facts join only with `history`.
        Returns the pool, {name: {id: cosine}} over the whole pool, and {name: [ids by BM25]}.
        """
        statuses = ("active", "superseded") if history else ("active",)
        if self._db.execute(f"SELECT count(*) FROM facts WHERE status IN ({','.join('?' * len(statuses))})", statuses).fetchone()[0] == 0:
            return {}, {name: {} for name in queries}, {name: [] for name in queries}
        # A bounded pool whatever the memory size (50,000 facts: Prompt 4 scale check): twice the nearest
        # neighbours (superseded ones are dropped below), and at most 4n of the facts in the window or about a
        # named person, most recent first.
        knn = 2 * n
        cap = 4 * n
        pool: dict[int, None] = {}
        nearest: dict[str, dict[int, float]] = {}
        lexical: dict[str, list[int]] = {}
        for name, (vec, text) in queries.items():
            qv = np.asarray(vec, dtype=np.float32).tobytes()
            nearest[name] = {int(r): 1.0 - float(d) for r, d in self._db.execute(
                "SELECT rowid, distance FROM fact_vec WHERE embedding MATCH ? AND k = ?", (qv, min(knn, 4096)))}
            lexical[name] = self._fts_search("fact_fts", "facts", text, n * 2) if text else []
            pool.update(dict.fromkeys(nearest[name]))
            pool.update(dict.fromkeys(lexical[name]))
        if window is not None:
            pool.update(dict.fromkeys(r[0] for r in self._db.execute(
                "SELECT id FROM facts WHERE ((heard_ts >= ? AND heard_ts < ?) OR (event_ts >= ? AND event_ts < ?)) "
                "AND status IN ({}) ORDER BY heard_ts DESC LIMIT ?".format(",".join("?" * len(statuses))),
                (window[0], window[1], window[0], window[1], *statuses, cap))))
        for p in people:
            like = "%" + p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            pool.update(dict.fromkeys(r[0] for r in self._db.execute(
                "SELECT id FROM facts WHERE (people LIKE ? ESCAPE '\\' OR subject = ? COLLATE NOCASE OR text LIKE ? ESCAPE '\\') "
                "AND status IN ({}) ORDER BY heard_ts DESC LIMIT ?".format(",".join("?" * len(statuses))),
                (like, p, like, *statuses, cap))))
        facts: dict[int, Fact] = {}
        ids = list(pool)
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for r in self._db.execute(f"SELECT {_FACT_COLS} FROM facts WHERE id IN ({marks}) AND status IN ({','.join('?' * len(statuses))})",
                                      [*chunk, *statuses]):
                f = _fact(r)
                facts[f.id] = f
        cos: dict[str, dict[int, float]] = {}
        for name, (vec, _) in queries.items():
            sims = {i: s for i, s in nearest[name].items() if i in facts}
            missing = [i for i in facts if i not in sims]
            qv = np.asarray(vec, dtype=np.float32).tobytes()
            for i in range(0, len(missing), 500):
                chunk = missing[i : i + 500]
                for rid, dist in self._db.execute(
                    f"SELECT rowid, vec_distance_cosine(embedding, ?) FROM fact_vec WHERE rowid IN ({','.join('?' * len(chunk))})", [qv, *chunk]
                ):
                    sims[int(rid)] = 1.0 - float(dist)
            cos[name] = sims
        return facts, cos, {name: [i for i in ids_ if i in facts] for name, ids_ in lexical.items()}

    def utterance_candidates(
        self, queries: dict[str, tuple[np.ndarray, str]], n: int, window: tuple[float, float] | None = None,
    ) -> tuple[dict[int, Utterance], dict[str, dict[int, float]], dict[str, list[int]]]:
        """Like `candidates`, over the raw transcript (lines that have a vector or a keyword match)."""
        pool: dict[int, None] = {}
        nearest: dict[str, dict[int, float]] = {}
        lexical: dict[str, list[int]] = {}
        for name, (vec, text) in queries.items():
            qv = np.asarray(vec, dtype=np.float32).tobytes()
            nearest[name] = {int(r): 1.0 - float(d) for r, d in self._db.execute(
                "SELECT rowid, distance FROM utterance_vec WHERE embedding MATCH ? AND k = ?", (qv, n))}
            lexical[name] = self._fts_search("utterance_fts", "utterances", text, n) if text else []
            pool.update(dict.fromkeys(nearest[name]))
            pool.update(dict.fromkeys(lexical[name]))
        lines = {u.id: u for u in self.utterances(list(pool))}
        if window is not None:
            lines = {i: u for i, u in lines.items() if window[0] <= u.ts < window[1]}
        cos: dict[str, dict[int, float]] = {}
        for name, (vec, _) in queries.items():
            sims = {i: s for i, s in nearest[name].items() if i in lines}
            missing = [i for i in lines if i not in sims]
            if missing:
                qv = np.asarray(vec, dtype=np.float32).tobytes()
                for rid, dist in self._db.execute(
                    f"SELECT rowid, vec_distance_cosine(embedding, ?) FROM utterance_vec WHERE rowid IN ({','.join('?' * len(missing))})",
                    [qv, *missing],
                ):
                    sims[int(rid)] = 1.0 - float(dist)
            cos[name] = sims
        return lines, cos, {name: [i for i in ids if i in lines] for name, ids in lexical.items()}

    def get(self, fid: int) -> Fact | None:
        r = self._db.execute(f"SELECT {_FACT_COLS} FROM facts WHERE id = ?", (fid,)).fetchone()
        return _fact(r) if r else None

    def history(self, fid: int) -> list[Fact]:
        """The whole chain this fact belongs to, oldest first (what it replaced and what replaced it)."""
        chain: list[Fact] = []
        f = self.get(fid)
        if f is None:
            return []
        seen = {f.id}
        back = f
        while True:  # walk to the oldest version
            r = self._db.execute(f"SELECT {_FACT_COLS} FROM facts WHERE superseded_by = ? ORDER BY id LIMIT 1", (back.id,)).fetchone()
            if r is None or r[0] in seen:
                break
            back = _fact(r)
            seen.add(back.id)
        chain.append(back)
        while chain[-1].superseded_by is not None and chain[-1].superseded_by not in {c.id for c in chain}:
            nxt = self.get(chain[-1].superseded_by)
            if nxt is None:
                break
            chain.append(nxt)
        return chain

    def recent(self, limit: int = 100, offset: int = 0, contains: str | None = None, *, status: str | None = "active") -> list[Fact]:
        sql = f"SELECT {_FACT_COLS} FROM facts"
        where, args = [], []
        if status is not None:
            where.append("status = ?")
            args.append(status)
        if contains:
            where.append("text LIKE ? ESCAPE '\\'")
            args.append("%" + contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY heard_ts DESC, id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [_fact(r) for r in self._db.execute(sql, args)]

    def exists_recent(self, text: str, since_ts: float) -> bool:
        return self._db.execute(
            "SELECT 1 FROM facts WHERE text = ? AND heard_ts >= ? AND status = 'active' LIMIT 1", (text, since_ts)
        ).fetchone() is not None

    def count(self, status: str | None = "active") -> int:
        if status is None:
            return int(self._db.execute("SELECT count(*) FROM facts").fetchone()[0])
        return int(self._db.execute("SELECT count(*) FROM facts WHERE status = ?", (status,)).fetchone()[0])

    # --- people --------------------------------------------------------------------------

    def people(self) -> list[Person]:
        rows = self._db.execute("SELECT id, name, aliases, source FROM people ORDER BY name COLLATE NOCASE")
        return [Person(i, n, tuple(json.loads(a)), s) for i, n, a, s in rows]

    def get_person(self, name: str) -> Person | None:
        r = self._db.execute("SELECT id, name, aliases, source FROM people WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        return Person(r[0], r[1], tuple(json.loads(r[2])), r[3]) if r else None

    def add_person(self, name: str, aliases: list[str] | tuple[str, ...] = (), source: str = "extracted") -> Person:
        name = name.strip()
        existing = self.get_person(name)
        if existing is not None:
            if aliases:
                return self.set_aliases(existing.id, [*existing.aliases, *aliases]) or existing
            return existing
        clean = _clean_aliases(name, aliases)
        with self._db:
            pid = int(self._db.execute(
                "INSERT INTO people(name, aliases, source, created_ts) VALUES (?, ?, ?, ?)",
                (name, json.dumps(clean), source, time.time()),
            ).lastrowid)
        return Person(pid, name, tuple(clean), source)

    def set_aliases(self, pid: int, aliases: list[str]) -> Person | None:
        r = self._db.execute("SELECT name, source FROM people WHERE id = ?", (pid,)).fetchone()
        if r is None:
            return None
        clean = _clean_aliases(r[0], aliases)
        with self._db:
            self._db.execute("UPDATE people SET aliases = ? WHERE id = ?", (json.dumps(clean), pid))
        return Person(pid, r[0], tuple(clean), r[1])

    def rename_person(self, pid: int, name: str) -> Person | None:
        """The user corrects a name: the old spelling becomes an alias."""
        r = self._db.execute("SELECT name, aliases FROM people WHERE id = ?", (pid,)).fetchone()
        if r is None:
            return None
        aliases = _clean_aliases(name, [*json.loads(r[1]), r[0]])
        with self._db:
            self._db.execute("UPDATE people SET name = ?, aliases = ?, source = 'user' WHERE id = ?", (name.strip(), json.dumps(aliases), pid))
        return Person(pid, name.strip(), tuple(aliases), "user")

    def delete_person(self, pid: int) -> bool:
        with self._db:
            return self._db.execute("DELETE FROM people WHERE id = ?", (pid,)).rowcount > 0

    def close(self) -> None:
        self._db.close()


def _utt(r) -> Utterance:
    return Utterance(r[0], r[1], r[2], r[3], r[4], r[5], int(r[6]))


def _clean_aliases(name: str, aliases) -> list[str]:
    out: list[str] = []
    for a in aliases:
        a = str(a).strip()
        if a and a.lower() != name.strip().lower() and a.lower() not in {x.lower() for x in out}:
            out.append(a)
    return out


def _legacy_names(text: str) -> list[str]:
    from kin.brain.names import extract_names

    return sorted(extract_names(text))
