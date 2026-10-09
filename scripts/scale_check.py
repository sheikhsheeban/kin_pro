"""Scale check: recall latency and RAM with 50,000 facts (Pi 5 profile by default).

Synthetic facts get random unit vectors (exact vec0 search costs the same whatever the vectors mean) and
realistic texts with names, places and numbers, so BM25, the people lookup and the time window do real
work. Questions go through the real hybrid recall: EmbeddingGemma query, fusion, the MiniLM reranker.

  uv run python scripts/scale_check.py                 # 50,000 facts, pi5 profile
  uv run python scripts/scale_check.py --facts 10000 --profile laptop
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import secrets
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES = ["Priya", "Rahul", "Kavya", "Sam", "Neha", "Arjun", "Ananya", "Vikram", "Meera", "Rohan", "Isha", "Karan",
         "Divya", "Aditya", "Sneha", "Farah", "Imran", "Lakshmi", "Nikhil", "Pooja"] + [f"Person{i}" for i in range(180)]
PLACES = ["the library cafe", "Dharampeth", "Sitabuldi", "Pune", "Lonavala", "the office", "the gym", "Barbeque Nation",
          "Edinburgh", "Bangalore", "the airport", "AIIMS Delhi", "the station", "DMart", "the clinic"]
TEMPLATES = [
    "{a} and the user meet at {p} at {h} pm on day {d}.",
    "{a}'s birthday is on the {d}th of month {m}.",
    "{a} is allergic to item {n}.",
    "The {p} wifi password is word{n}.",
    "{a} paid {n} rupees for the groceries at {p}.",
    "{a} said the meeting moved to {p} at {h}.",
    "Someone said: I lent {a} my umbrella number {n}.",
    "{a} works at {p} as role {n}.",
    "The code for locker {n} is {c}.",
    "{a} is moving to {p} in month {m}.",
]
QUESTIONS = ["When am I meeting Priya?", "What is Rahul allergic to?", "What's the wifi password at the gym?",
             "How much did Neha pay for the groceries?", "Where does Arjun work?", "What did Kavya say yesterday?",
             "When is Sam's birthday?", "What's the code for locker 4471?", "Who is moving to Pune?",
             "What do I know about Meera?", "What's my Amazon password?", "Where is Vikram's new flat?"]


def populate(store, n: int, dim: int, seed: int = 7) -> float:
    """Bulk insert in one transaction per 5,000 (a running Kin adds facts one by one; this is setup only)."""
    import numpy as np

    rng = random.Random(seed)
    vecs = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    now = time.time()
    t0 = time.perf_counter()
    db = store._db
    for start in range(0, n, 5000):
        with db:
            for i in range(start, min(n, start + 5000)):
                a = rng.choice(NAMES)
                text = rng.choice(TEMPLATES).format(a=a, p=rng.choice(PLACES), h=rng.randint(1, 11), d=rng.randint(1, 28),
                                                    m=rng.randint(1, 12), n=rng.randint(100, 99999), c=rng.randint(1000, 9999))
                heard = now - rng.uniform(0, 3 * 365) * 86400
                event = heard + rng.uniform(0, 30) * 86400 if rng.random() < 0.4 else None
                status = "superseded" if rng.random() < 0.1 else "active"
                fid = db.execute(
                    "INSERT INTO facts(text, kind, subject, people, event_ts, heard_ts, status, confidence, source, evidence) "
                    "VALUES (?, 'fact', ?, ?, ?, ?, ?, 0.8, 'extracted', '[]')",
                    (text, a, json.dumps([a]), event, heard, status)).lastrowid
                db.execute("INSERT INTO fact_vec(rowid, embedding) VALUES (?, ?)", (fid, vecs[i].tobytes()))
                store._fts_add("fact_fts", fid, text)
        print(f"  {min(n, start + 5000)} facts", flush=True)
    for name in NAMES[:20]:
        store.add_person(name)
    return time.perf_counter() - t0


async def main(args) -> dict:
    import psutil

    from kin.config import load_config
    from kin.core.bus import EventBus
    from kin.memory.embed import GemmaEmbedder, find_embedding_model
    from kin.memory.rerank import CrossEncoder, find_reranker
    from kin.memory.service import MemoryService
    from kin.memory.store import MemoryStore

    cfg = load_config(args.profile)
    m = cfg.memory
    tmp = Path(tempfile.mkdtemp(prefix="kin-scale-"))
    key = secrets.token_hex(32)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e6
    store = MemoryStore(tmp / "kin.db", key, m.dim)
    print(f"populating {args.facts} facts ({m.dim}-d) ...")
    build_s = populate(store, args.facts, m.dim)
    store.close()
    db_mb = (tmp / "kin.db").stat().st_size / 1e6
    model = find_reranker(cfg.model_path(m.retrieval.rerank_model))
    svc = MemoryService(EventBus(), lambda: GemmaEmbedder(find_embedding_model(cfg.model_path(m.embedding_model)),
                                                          m.embedding_variant, m.dim),
                        lambda d: MemoryStore(tmp / "kin.db", key, d), retrieval=m.retrieval,
                        reranker_factory=(lambda: CrossEncoder(model, threads=cfg.runtime.threads)) if (model and m.retrieval.rerank) else None)
    await svc.start()
    rss_loaded = proc.memory_info().rss / 1e6
    lat, conf = [], []
    for rnd in range(args.rounds):
        for q in QUESTIONS:
            r = await svc.recall(q, m.top_k, lines=True)
            lat.append(r.ms)
            conf.append(r.confidence)
    rss_peak = proc.memory_info().rss / 1e6
    await svc.close()
    lat.sort()
    out = {
        "profile": args.profile, "facts": args.facts, "dim": m.dim, "reranker": svc.reranker is not None,
        "populate_s": round(build_s, 1), "db_mb": round(db_mb, 1),
        "recall_ms_median": round(statistics.median(lat), 1), "recall_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
        "recall_ms_max": round(lat[-1], 1),
        "rss_mb_start": round(rss0), "rss_mb_models_loaded": round(rss_loaded), "rss_mb_after_queries": round(rss_peak),
    }
    print(json.dumps(out, indent=1))
    for p in tmp.glob("*"):
        p.unlink()
    os.rmdir(tmp)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts", type=int, default=50_000)
    ap.add_argument("--profile", default="pi5", choices=["laptop", "pi5"])
    ap.add_argument("--rounds", type=int, default=3)
    asyncio.run(main(ap.parse_args()))
