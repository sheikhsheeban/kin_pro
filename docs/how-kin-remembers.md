# How Kin Remembers (memory v2)

Everything below runs on the device. The database is one SQLCipher file (`data/kin.db`, 256-bit key,
`secure_delete` on). Raw audio is never stored.

## 1. What is kept

| Table | What | How long |
|---|---|---|
| `utterances` | every overheard line that is not filler, with time, speaker (owner/other/unknown, never guessed), STT model, and an EmbeddingGemma vector | `memory.utterance_retention_days` (14) |
| `episodes` | lines grouped into conversations (new one after 90 s of silence or 10 min) | with their lines |
| `facts` | self-contained sentences with absolute dates: subject, people, event time, heard time, status (`active`, `superseded`, `retracted`), the lines they rest on, `pinned` | `memory.fact_retention_days` (forever by default; pinned facts are never purged) |
| `fact_history` | every change to a fact: added, replaced, corrected, retracted, restored, pinned, merged; who did it (extraction, voice, UI) and the old and new text | with the fact (forgetting a fact deletes its history too) |
| `people` | names Kin knows, with the misheard spellings it corrected (aliases) | until deleted |

## 2. Writing: from lines to facts

1. A line arrives; filler ("okay") is dropped, the rest is stored as an utterance.
2. When an episode ends and no question is running, Gemma 4 E2B extracts facts as strict JSON, validated
   (one retry; a failed episode is retried an hour later). Code then fixes what a 2B model gets wrong:
   relative times are resolved against when the line was heard, and "I" becomes the speaker only when the
   speaker is known.
3. Misheard names are normalised against the people table (rapidfuzz), keeping the heard spelling as an alias.
4. A new fact close to an active one is ADDed, UPDATEs or CONTRADICTS it (the old one is superseded and
   linked), or is a DUPLICATE (the old one gains the evidence). Nothing is deleted.
5. **Only about me** (Memory tab setting, kept in the database): facts whose subject is another person are
   not stored. Facts about other people are marked as such in the Memory tab either way.

## 3. Reading: answering a question

1. **Understanding, no LLM:** the time phrase ("yesterday", "on Friday", "later in October") becomes a
   window; known people are found (aliases, near-miss spellings); "when is …" prefers upcoming events;
   "originally" searches replaced facts too.
2. **Hybrid retrieval:** EmbeddingGemma and BM25 rank the facts (and the model's rewording, at half weight);
   Reciprocal Rank Fusion (k = 60) combines them; facts in the time window come first; a MiniLM
   cross-encoder reorders the top 10. Pinned facts about a person the question names are always offered.
3. **Abstention:** under the calibrated confidence floor (0.375 of cosine + 0.3 × word coverage) Kin says it
   doesn't remember, or quotes a close raw line ("I'm not sure, but on Saturday I heard …").
4. **Answering:** facts best first with subject, times and status; up to three sentences, each citing its
   ids; both versions when two active facts disagree. A strong match is answered in one LLM call.
5. **The gate** speaks a sentence only if it cites offered facts (or honestly abstains), does not attribute
   one person's fact to another, and holds no number the cited facts lack.
6. **Why this answer?** Every reply in the UI shows what the question was understood as, the facts offered
   with all their scores, what was cited, and what was left out and why.

### What limits recall quality (Prompt 5 diagnosis)

`scripts/diagnose_recall.py` attributes every wrong answer and measures the answer step on its own:

- **Given the right facts, the answer step is good.** With exactly the gold supporting facts, Gemma 4 E2B answers
  93.8% of answerable questions (E4B 100%, Qwen3.5-4B 98.8%; both 2.7-5x slower to the first sentence).
- **Most wrong answers are an abstention with the right fact offered**, and most of those facts say "Someone":
  without speaker identification, "I'm vegan now" (said by Rahul) can only be stored as "Someone is vegan", which
  cannot answer "Who is vegan?". Speaker identification (voice enrolment) is the single biggest fix left.
- Extraction finds 81% of the gold facts, 65% of those spread over several short lines.
- Small talk that E2B copies into a "fact" ("Let's get lunch.") is dropped when the text has no number, time or name
  and the semantic router also calls it chatter.
- Multi-fact questions ("what do I know about Rahul", "what did I plan yesterday") get the matching facts as a
  deterministic list (pinned first, newest first), summarised by the model or read back if it cites nothing.
- A question whose names were normalised ("Raul" -> "Rahul") is searched both ways.

## 4. Correcting Kin

| Said or clicked | What happens |
|---|---|
| "That's wrong" / "I never said that" | the fact cited in the last answer is **retracted**: no longer used, not deleted, restorable |
| "No, it's 5 pm" / "Rahul isn't vegan any more" | a corrected version is stored and the old one **superseded** (both in the history); a new clock time also moves the event time |
| Memory tab: edit, retract, restore, pin, merge | the same operations; merge keeps one fact with both facts' evidence and people |
| "What did you remember today?" / the review card | the facts added today and every change, so mistakes are fixed while fresh |
| "Forget everything about Rahul" | asks "That's N memories and M heard lines about Rahul. Say confirm"; deletes them, their history and the person |
| "Forget the last 10 minutes" / "forget that" / ✕ in the Memory tab | deletes for good, after confirmation |

Corrections act on the facts cited in the last answer (the one closest to what the user names, if several).
With no recent answer, Kin asks which memory is meant rather than guessing.

## 5. Keeping it safe and healthy

- **Backup:** losing the key loses every memory. `kin backup --to <folder>` (or System tab → Encrypted backup)
  writes one file: the encrypted database plus its key wrapped by a passphrase (scrypt, HMAC-checked). Local
  folders only; nothing is uploaded. `kin restore <file>` puts it back (Kin stopped; an existing database is
  kept as `*.before-restore`).
- **Upgrades:** before a schema upgrade, a backup in the same format goes to `data/backups/` (its passphrase is the
  database key, so it is exactly as protected as the database), is read back and checked, and only then does the
  upgrade run, in one transaction. On any failure the transaction is rolled back, the file is put back from the
  backup and Kin stops with the reason (`MigrationError`).
- **Passphrases** for new backups must have 16+ characters, or 12+ mixing three character classes, and must not be
  a common password or a sequence. *Test a backup…* (System tab) is a restore drill that never touches the live file.
- **Crash safety:** SQLite rolls back a write cut off half way; backups are written to `.part` and renamed only when
  complete; an episode whose extraction was killed after some facts were stored is marked not done, and its partial
  facts are dropped before it is extracted again (`tests/test_hardening.py` kills real processes at each point).
- **Maintenance:** once a night after 03:00, or as soon as the device is on mains power, never more often than
  every 20 hours and never during a question: retention purges, FTS optimize, VACUUM, `integrity_check`.
  The System tab shows the health report: counts, database size, last extraction, extraction backlog, last
  maintenance.

## 6. How big can it get

`scripts/scale_check.py`, 50,000 facts, Pi 5 profile, on the development laptop: recall 619 ms median
(661 ms p95), 68 MB database, 467 MB process RSS with the embedder and reranker loaded. Most of the time
is the exact vector scan (about 240 ms per query vector through SQLCipher). At about 40 facts a day that
is over three years of memory; no approximate index is needed yet. If it ever is, an in-RAM exact matrix
(50 MB at 50,000 × 256 floats) would cut the scan to a few milliseconds before any approximation.
