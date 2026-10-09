"""A believable week of memories for live demos, in a separate demo database (never the real one).

`kin run --demo` keeps everything under `demo/` (its own database, key, gateway audit and caches); the
UI then shows a Reset button that wipes the demo database and loads this week again. `seed` refuses to
touch a database that it did not create, so real personal data is never wiped by a demo.

Times are relative to `now`: "2 days ago at 09:10" stays two days ago whenever the demo runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

DEMO_MARK = "demo_database"


@dataclass(frozen=True)
class DemoFact:
    days_ago: int
    at: str  # "HH:MM" when it was heard
    heard: str  # the raw line, as overheard
    text: str  # the fact, as extraction would store it
    kind: str = "fact"
    subject: str | None = None
    people: tuple[str, ...] = ()
    event: tuple[int, str] | None = None  # (days from now, "HH:MM"); negative = in the past
    pinned: bool = False
    replaces: int | None = None  # index of an earlier fact this one updates


WEEK: list[DemoFact] = [
    DemoFact(6, "08:40", "The wifi password at home is now tiger42, I changed it last night.",
             "The home wifi password is tiger42.", pinned=True),
    DemoFact(6, "09:15", "Rahul's allergic to peanuts, so no satay at the party.", "Rahul is allergic to peanuts.",
             "fact_about_person", "Rahul", ("Rahul",), pinned=True),
    DemoFact(6, "19:30", "Mum's birthday is on the 14th of November, don't forget this year.",
             "Mum's birthday is on 14 November.", "fact_about_person", "Mum", ("Mum",)),
    DemoFact(5, "10:05", "Priya said the data science course starts in January.",
             "Priya said the data science course starts in January.", "fact_about_person", "Priya", ("Priya",)),
    DemoFact(5, "12:20", "Dentist on Thursday at 11, the one on Park Street.",
             "The user has a dentist appointment on Park Street.", "plan", None, (), (2, "11:00")),
    DemoFact(5, "18:45", "Neha moved to Pune last month for the new job.", "Neha moved to Pune for a new job.",
             "fact_about_person", "Neha", ("Neha",)),
    DemoFact(4, "08:55", "The car's due for its service at 15,000 kilometres.",
             "The car is due for its service at 15,000 km.", "fact"),
    DemoFact(4, "13:10", "Arjun's flat is 4B, third floor, the lift is broken.", "Arjun's flat is 4B on the third floor.",
             "fact_about_person", "Arjun", ("Arjun",)),
    DemoFact(4, "20:00", "Let's do dinner with Rahul on Saturday at 7 at Barbecue Nation.",
             "The user has dinner with Rahul at Barbecue Nation on Saturday at 7 pm.", "plan", None, ("Rahul",), (3, "19:00")),
    DemoFact(3, "09:30", "The office parking permit number is 7731.", "The office parking permit number is 7731.",
             pinned=True),
    DemoFact(3, "11:45", "Neha's extension at work is 2314.", "Neha's work extension is 2314.", "fact_about_person",
             "Neha", ("Neha",)),
    DemoFact(3, "17:20", "Actually, make the Rahul dinner 8 instead of 7, he's stuck at work.",
             "The user has dinner with Rahul at Barbecue Nation on Saturday at 8 pm.", "plan", None, ("Rahul",),
             (3, "20:00"), replaces=8),
    DemoFact(2, "08:10", "I'm meeting Priya at 4 pm at the library cafe for coffee.",
             "The user is meeting Priya at the library cafe for coffee at 4 pm.", "plan", None, ("Priya",), (0, "16:00")),
    DemoFact(2, "10:40", "The plumber, Suresh, comes Monday morning around 9.",
             "Suresh the plumber is coming on Monday at 9 am.", "plan", "Suresh", ("Suresh",), (5, "09:00")),
    DemoFact(2, "21:15", "I've started running again, 5k three times a week.",
             "The user runs 5 km three times a week.", "fact"),
    DemoFact(1, "09:05", "Book club is reading The Midnight Library this month.",
             "The book club is reading The Midnight Library this month.", "fact"),
    DemoFact(1, "12:30", "Rahul's gone vegetarian, he told me at lunch.", "Rahul is vegetarian now.",
             "fact_about_person", "Rahul", ("Rahul",)),
    DemoFact(1, "16:50", "The landlord wants the rent by the 5th from now on.", "The rent is due by the 5th of each month.",
             "fact"),
    DemoFact(0, "08:20", "Mum's flight lands at 6:40 tomorrow evening, terminal 2.",
             "Mum's flight lands at terminal 2 tomorrow at 6:40 pm.", "plan", "Mum", ("Mum",), (1, "18:40")),
    DemoFact(0, "09:45", "Priya's new number ends in 4471.", "Priya's new phone number ends in 4471.",
             "fact_about_person", "Priya", ("Priya",)),
]

CHATTER = [  # heard and kept as raw lines only, like real small talk
    (6, "08:42", "Okay, where did I leave the keys?"),
    (4, "13:12", "Haha, yeah, that lift never works."),
    (2, "10:42", "Can you pass the salt?"),
    (0, "09:47", "Right, let's get going."),
]

TASKS = [("Buy oat milk", "task", None), ("Call the bank about the card", "reminder", (1, "10:00"))]


def _at(now: datetime, days: int, hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    return (now + timedelta(days=days)).replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


def is_demo(store) -> bool:
    return store._db.execute("SELECT value FROM meta WHERE key = ?", (DEMO_MARK,)).fetchone() is not None


def wipe(store) -> None:
    """Delete everything in a demo database. Refuses any database that holds data but is not a demo one."""
    db = store._db
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")}
    used = any(db.execute(f"SELECT 1 FROM {t} LIMIT 1").fetchone() for t in ("facts", "utterances", "tasks") if t in tables)
    if used and not is_demo(store):
        raise PermissionError("this is not a demo database: refusing to wipe it")
    with db:
        for t in ("fact_history", "fact_vec", "fact_fts", "facts", "utterance_vec", "utterance_fts", "utterances",
                  "episodes", "people", "tasks"):
            if t in tables:
                db.execute(f"DELETE FROM {t}")
        db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, '1')", (DEMO_MARK,))


def seed(store, embed_document, now: datetime | None = None) -> dict:
    """Wipe the demo database and load the week. Runs on the store's thread. Returns counts."""
    now = now or datetime.now()
    wipe(store)
    facts = []
    for f in WEEK:
        heard = _at(now, -f.days_ago, f.at)
        u, _ = store.add_utterance(f.heard, heard, vector=embed_document(f.heard))
        event = _at(now, f.event[0], f.event[1]) if f.event else None
        fact = store.add_fact(f.text, f.kind, embed_document(f.text), confidence=0.9, subject=f.subject, people=f.people,
                              event_ts=event, heard_ts=heard, evidence=[u.id], episode_id=u.episode_id, actor="demo")
        if f.replaces is not None:
            store.supersede(facts[f.replaces].id, fact.id, actor="demo")
        if f.pinned:
            store.set_pinned(fact.id, True, actor="demo")
        for p in f.people:
            store.add_person(p, source="demo")
        facts.append(fact)
    for days, at, text in CHATTER:
        store.add_utterance(text, _at(now, -days, at), vector=embed_document(text), worth=0)
    store.close_idle_episodes(time.time() + 3600, 0, 0, force=True)
    for ep in store.episodes(1000):
        store.mark_episode(ep.id, "done")  # already "extracted": the facts above
    for text, kind, due in TASKS:
        store.add_task(text, kind, _at(now, *due) if due else None)
    return {"facts": len(facts), "lines": len(WEEK) + len(CHATTER), "tasks": len(TASKS)}


def seed_demo(cfg) -> str:
    """Create or reset the demo database of a demo config (`cfg.demo`, data under demo/). Loads the embedder."""
    import os

    from kin.memory.embed import GemmaEmbedder, find_embedding_model
    from kin.memory.store import MemoryStore, load_or_create_key

    if not cfg.demo:
        raise PermissionError("not a demo config: the demo week only goes into the demo database")
    m = cfg.memory
    db = cfg.data_path(m.db_path)
    os.environ.pop("KIN_DB_KEY", None)  # the demo database has its own key file
    store = MemoryStore(db, load_or_create_key(cfg.data_path(m.key_file)), m.dim, reembed=True)
    try:
        emb = GemmaEmbedder(find_embedding_model(cfg.model_path(m.embedding_model)), m.embedding_variant, m.dim)
        counts = seed(store, emb.embed_document)
    finally:
        store.close()
    return f"demo database {db}: {counts['facts']} facts, {counts['lines']} heard lines, {counts['tasks']} tasks"
