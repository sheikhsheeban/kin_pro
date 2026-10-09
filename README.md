# Kin

A private, always-listening personal AI companion. All reasoning runs on the device.
Architecture: [docs/kin-architecture.md](docs/kin-architecture.md) · Build plan: [docs/kin-build-plan.md](docs/kin-build-plan.md)

**Status: Phase 3 + routing refactor.** Kin remembers what it hears and answers out loud, fully offline.
It can also look up a fact online (weather, news, search). Lookups go only through a separate gateway process
that accepts nothing but `{"kind", "params"}`. Kin also keeps tasks and speaks reminders when they are due.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run kin download-models              # ~3.3 GB into ./models (Gemma 4 E2B, EmbeddingGemma, Moonshine, Piper, VAD, wake word)
uv sync --extra fallback                # optional: llama.cpp backend (prebuilt CPU wheel, no compiler needed)
uv run kin download-models --fallback   # optional: Qwen3.5-2B GGUF for the fallback backend
cp .env.example .env                    # optional: Brave / NewsAPI keys (keyless providers are used otherwise)
```

Set your home city in `kin/config/laptop.yaml` (`user.home_city`). Weather questions without a place use it.

Memories, tasks and reminders share one encrypted database (SQLCipher, 256-bit key). The key is `data/kin.key`
by default, or set `KIN_DB_KEY` (64 hex characters). If you lose the key, you lose the data. The brain and the gateway
authenticate each other with a separate shared secret, `data/gateway.key`.

## Run

```bash
uv run kin run --profile laptop --open    # UI at http://127.0.0.1:8000 (also starts the gateway process)
uv run kin run --profile pi5              # Pi limits: 4 threads, 2K context, 600-token prompts, lazy query STT
uv run kin gateway                        # run the gateway on its own (on the Pi: its own service)
```

**Scripted demo:** the umbrella scenario from the architecture doc, with 13 self-checks.

```bash
uv run python scripts/demo.py                       # real Open-Meteo forecast (needs internet)
uv run python scripts/demo.py --weather fake --voice # canned forecast; your lines are spoken into the mic pipeline
uv run python scripts/demo.py --speak                # also play Kin's replies through the speakers
```

**Demo data, never your own:** `uv run kin run --demo` keeps everything (database, key, gateway audit, caches) under
`demo/` and fills it with a made-up week (`scripts/demo_seed.py` does the same on its own). The header then shows
*DEMO DATA* and a **Reset demo** button that wipes the demo database and reloads the week. Reset refuses any database
that was not created as a demo one, so a live demo never touches real memories.

**Live demo in the UI**

1. Say *"I'm meeting Priya at 4 pm at the library café for coffee."* The line is marked *saved as memory #N*.
2. Hold **Space** (or say "Hey Jarvis") and ask *"Do I need an umbrella for the coffee with Priya I mentioned this morning?"*
   The LED goes **blue** during the lookup. Kin says *"I checked the weather online…"* and decides.
   The **Network** tab shows the exact JSON sent: `{"kind":"weather","params":{"city":"London","date":"…"}}`.
3. Turn **Offline mode** on and ask again. Kin says it can't check online and still uses what it remembers.
   *"When am I meeting Priya?"* still works.
4. *"Remind me to email the professor at 5."* Kin confirms aloud, and at 5 PM it says *"Reminder: Email the professor."*

Ask in your own words: Gemma 4 E2B picks the tool (recall, remember, add a task or reminder, read the list,
mark done, cancel a reminder, look something up) in the same call that answers. Each reply in the UI shows the path it
took, e.g. *tool: recall* or *tool: lookup (weather)* or *answered directly*.

Deleting is different: *forget the last N minutes*, *forget that* and *cancel the reminder to …* are recognised by a strict
pattern only (never "don't forget…" or "I always forget…"), and nothing is deleted until you say **confirm** within
10 seconds (press Space or say the wake word first) or press **Confirm** on the card in the UI.
Talking over Kin interrupts it.

## If "Hey Jarvis" doesn't wake Kin

Hold-to-talk always works; the wake word depends on how the stock model scores your voice on your mic.

```bash
uv run python scripts/wake_probe.py            # say "Hey Jarvis" a few times: prints the best score per second
uv run python scripts/wake_probe.py --play 6   # no voice needed: Piper says it through the speakers
uv run kin run --wake-threshold 0.3            # wake on lower scores (or set wake.threshold in the profile)
```

A working setup scores about 0.00 in silence and 0.8-1.0 for "Hey Jarvis". If your best scores sit between 0.2 and
0.5, lower the threshold a little above your silence level. The System tab shows the live wake score, the threshold,
and the highest score of the last thing you said.

## What gets remembered

Memory has two tiers, both in the one encrypted SQLCipher file (schema v4):

- **Utterances** are the raw evidence. Every finished overheard line is stored, except pure filler ("okay", "yeah sure").
  Each line carries its time, the STT model, and a speaker (`owner`, `other`, or unknown). Lines are grouped into
  **episodes**: a new one starts after 90 s of silence or after 10 minutes. Raw lines are purged after 14 days
  (`memory.utterance_retention_days`, checked daily).
- **Facts** are self-contained sentences with absolute dates. Gemma 4 E2B extracts them from each finished episode in
  the background, through the same priority queue as questions, so it only runs when no question is waiting (on the Pi,
  after a minute of quiet). Each fact has a subject, the people it mentions, when the thing happens, and the ids of the
  lines it came from.

Per episode:

1. **Pre-filter (free).** The semantic router labels each line. An episode in which every line is confident chatter,
   with no number and no name, is skipped.
2. **Gate (~1 s).** One yes/no question: is anything here worth remembering? This stops E2B turning small talk into "facts".
3. **Extraction.** E2B returns a JSON array, which is validated against a schema. An invalid reply gets one retry; if that
   also fails, the episode is kept and tried again an hour later. Code then fixes what a 2B model gets wrong:
   - "tomorrow at 4" in the cited line is resolved against the time that line was heard.
   - First person becomes "the user" only when the owner provably spoke. Otherwise the fact says "Someone said ...".
4. **Names.** Near-miss spellings of known people are corrected with rapidfuzz, and the heard spelling is kept as an
   alias. "Pria" becomes "Priya"; "Priyanka" stays separate. You can edit people in the Memory tab.
5. **Update, don't duplicate.** If an active fact about the same subject or people is close (cosine ≥ 0.75, or the same
   subject and event time), E2B answers ADD, UPDATE <id>, DUPLICATE <id> or CONTRADICTS <id>.
   - UPDATE and CONTRADICTS mark the old fact superseded and link it to the new one.
   - DUPLICATE adds the new evidence to the old fact.
   - Nothing is deleted: the Memory tab shows each fact's history and the raw lines behind it.

"Remember that ..." is still stored immediately. Answers only see active facts.

**Speaker identification.** Moonshine 0.x offers diarization: anonymous speaker indices, re-clustered over a 120 s
window. It has no voice enrolment and no way to match a voice to the owner. So Kin leaves the speaker unknown and does
not guess. The plumbing (`SttLine.speaker` → `TranscriptEvent.speaker` → `utterances.speaker`) is there for a real
identifier. `scripts/eval_recall.py --speakers oracle` measures what one would add.

## Correcting Kin, reviewing, backing up

The whole memory design is in [docs/how-kin-remembers.md](docs/how-kin-remembers.md).

- **"That's wrong"** retracts the fact Kin just cited (it is no longer used, nothing is deleted). **"No, it's 5 pm"**
  or **"Rahul isn't vegan any more"** stores the corrected version and keeps the old one as history.
- The Memory tab can edit, retract, restore, pin (always offered when that person is asked about) and merge facts.
  Every change is logged and shown under *history & evidence*.
- **What I remembered today**: a card at the top of the Memory tab, or ask *"what did you remember today?"*
- **Consent**: *only remember facts about me* (Memory tab); facts about other people are marked; *"forget everything
  about Rahul"* deletes his facts and raw lines after you say confirm. Fact retention: `memory.fact_retention_days`.
- **Backup**: `uv run kin backup --to E:/` writes the encrypted database plus its key wrapped by your passphrase
  (at least 16 characters, or 12 mixing cases, digits and symbols; typed twice). `uv run kin restore
  E:/kin-backup-….kinbak` puts it back. Local folders only; nothing is uploaded. The System tab has
  *Encrypted backup…* and *Test a backup…* (a restore drill: opens a copy in a temporary folder and checks it, without
  touching your memory).
- **Upgrades**: before a schema upgrade Kin writes a verified backup to `data/backups/pre-upgrade-vN-to-vM-….kinbak`
  (unlocked by the database key: `uv run kin restore <file> --force --key-passphrase`) and upgrades in one
  transaction. If anything fails, the database is put back as it was and Kin stops with the reason.
- **Crash safety**: a write, a backup or a fact extraction killed half way leaves no damage: unfinished writes are
  rolled back, a half-written backup is never named `.kinbak` (left-overs are cleaned up by the next backup), and an
  interrupted extraction runs again without duplicating facts.
- **Maintenance** runs nightly (or when on mains power): retention, FTS optimize, VACUUM, integrity check. The
  System tab shows memory health and has *Run maintenance now* and the backup buttons.

## How Kin answers "what did I ...?"

Recall is hybrid and needs no extra LLM call to understand the question (`kin/memory/query.py`, `retrieval.py`):

1. **Understanding.** The words set a time window ("yesterday", "last week", "on Friday", "tomorrow", "on 14 November",
   "3 days ago", "later in October"), relative to now in local time. Past tense looks back (what was heard or happened
   then); otherwise Kin looks ahead (when things happen). Known people are matched against the people table, including
   aliases and near-miss spellings. "When is ...?" prefers upcoming events. "What was it originally?" also searches
   replaced facts.
2. **Retrieval.** EmbeddingGemma (query prompt) and FTS5 BM25 (names, numbers, codes) each rank the facts. If the
   model called `recall(question)`, its rewording is ranked too, at half weight. The lists are fused with Reciprocal
   Rank Fusion (k = 60). Facts inside the time window rank first, and others only fill the remaining places.
   A small cross-encoder (MiniLM, ONNX int8, `memory.retrieval.rerank`) reorders the top 10.
3. **Abstention.** The confidence is cosine + 0.3 × the share of the question's words found in the fact, taken over
   the facts offered. Under `answer_floor` Kin says it doesn't remember. If a raw line it heard is close
   (`utterance_floor`), it quotes it instead: *"I'm not sure, but on Saturday at 10:05 I heard: “…”"*. Both thresholds
   were calibrated on the recall benchmark (see the comments in `kin/config/laptop.yaml`).
4. **Answering.** Facts are listed best first with subject, event time, heard time and status. The answer is at most
   three sentences, each citing its ids. Two active facts that disagree are both given, with when each was heard.
   When the question spells a name differently from memory ("Rahul" / "Raul"), the prompt says they are the same
   person. Before TTS the gate also drops filler ("This information was heard on Monday") and blocks a sentence about
   one person that cites only memories about someone else, or that contains a number none of its memories hold.
   If the confidence reaches `strong_score` and the question is about the user's life (not a command, not the
   weather), the facts go into the first message of the pre-warmed session, so one Gemma call answers.
5. **Why this answer?** Each reply in the UI opens a panel with what the question was understood as, every fact offered
   with its cosine, list ranks, RRF and rerank scores and confidence, which facts were cited, which were backfill
   (outside the time asked), which were left out as superseded, and any raw line quoted.

`uv run kin download-models --reranker` fetches the cross-encoder (~23 MB). Changing `memory.dim` re-embeds every fact
and line once at start. This SQLCipher build includes FTS5; without it, a trigram scan in Python takes over.

Laptop transcribes passive speech with Moonshine Small. The Pi keeps Tiny (`stt.passive.arch`); see
`scripts/stt_cpu_probe.py` for the CPU cost of each.

## Test

```bash
uv run pytest                    # everything (model, mic and internet tests skip when unavailable)
uv run pytest -m "not models"    # fast suite with fake engines (~20 s)
uv run pytest -m models -s       # real models: prints answers, latency, RAM, live weather
uv run pytest -m slow -s         # the WHY phrases through the real Gemma 4 E2B pipeline (~3 min)
uv run pytest -m browser         # the UI in headless Chromium (Playwright; `uv run playwright install chromium`,
                                 # or an installed Edge/Chrome is used); also run by CI (.github/workflows/ci.yml)
```

### Recall benchmark

`tests/eval/recall_benchmark/` holds a scripted week of overheard speech: 160 lines over 7 days, plus an
ASR-noisy copy with about 10% WER and misheard names. It also holds 99 questions with expected answers and
the lines that support them. The script feeds the lines straight into the memory pipeline at fixed
timestamps, with no audio, then asks each question through the real query path.

```bash
uv run python scripts/eval_recall.py                     # clean + noisy, all questions (~25 min each)
uv run python scripts/eval_recall.py --subset smoke      # 20 questions (~5 min); also runs in pytest -m models
uv run python scripts/eval_recall.py --judge             # also grade answers with local Gemma
uv run python scripts/eval_recall.py --report tests/eval/results/*.json   # table + 10 worst failures
uv run pytest tests/eval -m slow -s                      # the full benchmark as a test
uv run python scripts/eval_recall.py --facts-from tests/eval/results/recall-clean-….json tests/eval/results/recall-noisy-….json
                                                         # answer stage only: reuse a run's extracted facts (~12 min each)
uv run python scripts/eval_recall.py --set answer_floor=0 rerank=false   # memory.retrieval overrides
uv run python scripts/eval_retrieval.py --before --dim 256 768 --rerank --sweep
                                                         # retrieval only, no LLM (~1 min per setting)
uv run python scripts/latency_recall.py --profile pi5    # recall-query latency with the real models
uv run python scripts/eval_recall.py --subset all        # dev (99, tuned on) + held-out test (152) with 95% CIs
uv run python scripts/diagnose_recall.py attribution tests/eval/results/recall-clean-all-….json   # why answers fail
uv run python scripts/diagnose_recall.py extraction tests/eval/results/recall-clean-all-….json    # vs gold facts
uv run python scripts/diagnose_recall.py oracle --model litert:llm/gemma-4-E4B-it.litertlm        # answer-step ceiling
uv run python scripts/eval_spoken.py                     # 40 spoken scenarios: Piper -> VAD -> Moonshine -> Kin (audio in RAM)
```

Each run writes a JSON file to `tests/eval/results/` with the git hash and a hash of `kin/`. The file has
every write decision, the top-5 retrieval for each question, the answer Kin spoke, and why it failed:
dropped at write, not retrieved, wrong abstention, or wrong answer.

## The online/offline boundary (architecture §5, laptop edition)

```text
 brain process (kin run)                          gateway process (kin gateway)              internet
 ─────────────────────────                        ─────────────────────────────              ────────
 socket guard: no outbound sockets  ──named pipe / Unix socket, HMAC-authenticated──▶  schema: kind ∈ {weather, search, news},
 Gemma calls lookup(kind,query,when)      {"op":"lookup","request":                    strict params, length caps, no extra keys
 brain copies only allowed fields          {"kind":"weather","params":{…}}}            names scrubbed (salted hashes from the brain)
 LED blue · "I checked … online"   ◀── compact facts + audit entry ──                  offline switch · rate limit · audit log
 gate: [web] only if data came back                                                    host allowlist ──HTTPS GET──▶ Open-Meteo,
                                                                                                                     Wikipedia, Google News
```

| Layer | Laptop (now) | Pi 5 (Phase 4) |
|---|---|---|
| Brain cannot reach the network | In-process socket guard: audit hook on connect and DNS, plus patched asyncio | systemd `PrivateNetwork=yes` |
| Only the gateway's hosts | Allowlist in the gateway's HTTP client | nftables allowlist for the gateway user |
| Narrow interface | Pydantic schema, 512-byte cap, HMAC-authenticated pipe | same |
| No personal names leave | Brain sends salted name hashes, and the gateway scrubs matching words | same |
| Visible to the user | Blue LED, spoken "I checked … online", Network tab with exact JSON and upstream URLs | + packet capture |

## Measured on the development laptop

| | Result |
|---|---|
| Umbrella demo (routing call + lookup + answer call) | first audio 2.8 s, total 6.8 s |
| Recall query, before hybrid recall (routing call + answer call) | first sentence 5.75 s (Pi 5 profile 4.57 s), 1.75 LLM calls |
| Recall query, hybrid recall (`scripts/latency_recall.py`, 16 questions) | first sentence 3.68 s (Pi 5 profile 3.26 s), 1.12 LLM calls; 88% answered in one call |
| Hybrid retrieval (embed + BM25 + RRF + MiniLM rerank of 10) | 104 ms (Pi 5 profile 111 ms; 93 ms without the reranker) |
| Recall benchmark accuracy, clean / noisy (99 questions each) | 52.5% / 45.5% (Prompt 1 baseline) → 66.7% / 55.6%; false answers 20 / 24 → 5 / 14 |
| Back-to-back questions (wait for the session warm-up) | recall ~11 s, lookup ~9 s to first audio |
| Gemma 4 E2B | ~12 tok/s decode, 1.37 GB |
| Pi 5 profile, everything loaded + one query (fresh process) | 2.3 GB RSS, target 3.2 GB |
| Live Open-Meteo lookup (2 HTTPS calls) | ~1.4 s, ~1.8 KB received |

## Layout

```text
kin/
  core/      events · bus · workers · metrics · privacy (audio write guard) · netguard (socket guard) · state (LED)
  hal/       Hardware interface · SimulatedHardware · PiGpioHardware (Phase 4 stub)
  audio/     capture · ring_buffer · vad · wake · stt · pipeline
  brain/     llm (sessions + tools) · orchestrator · extract (episode -> facts) · commands (safety guard) · tools_schema
             semantic_router (pre-filter) · llm_queue
             gate · prompt · lookup · names · text · prompts/
  memory/    embed · store (SQLCipher + sqlite-vec + FTS5: utterances, episodes, facts, people, tasks) · people · service
  tools/     lookup_client · gateway_process · tasks · reminders · timeparse
  gateway/   server (separate process) · schemas · scrub · http (allowlist) · audit · transport · providers/{weather,search,news,fake}
  io/        tts (Piper)
  ui/        server (FastAPI, localhost only) · static/index.html (Device panel · Conversation · Memory · Tasks · Network · System)
  config/    laptop.yaml · pi5.yaml · router_examples.yaml
scripts/     demo.py (umbrella scenario) · download_models.py
```

## Privacy and honesty guarantees

| Rule | How it is enforced | Test |
|---|---|---|
| Raw audio never on disk | RAM ring buffer, zeroed once transcribed. An audit hook blocks writing audio files. Moonshine's WAV-dump option is refused. | `test_privacy.py` |
| Only the gateway touches the network | The brain process's socket guard blocks TCP, UDP, DNS, urllib and asyncio, including `ConnectEx` on Windows | `test_brain_process_cannot_open_an_outbound_socket`, `test_kin_app_installs_the_guard` |
| Narrow lookups only | Enum `kind`, strict params, 512-byte cap, extra keys refused, values validated by the gateway | `test_anything_else_is_refused_and_audited` (11 cases) |
| No transcript or names leave | Tool calls map to allowed fields only, and names are scrubbed by salted hash | `test_network_monitor_payload_is_only_city_and_date`, `test_person_names_are_scrubbed_from_free_text` |
| Only authorised clients | HMAC challenge on the pipe. A wrong key cannot use or crash the gateway. | `test_wrong_key_is_rejected_and_gateway_survives` |
| Online facts are cited honestly | `[web]` is allowed only if the lookup returned data, and "I checked … online" is always spoken | `test_offline_lookup_is_honest_and_recall_still_works` |
| Memories and tasks encrypted at rest | SQLCipher 4, raw 256-bit key, `secure_delete` | `test_memory_store_is_encrypted_on_disk`, `test_tasks_survive_restart_encrypted` |
| Recall answers cite memory ids | Every sentence is checked before TTS. Uncited sentences and invented ids are blocked. A direct answer that skips recall while memory clearly matches is blocked and recall runs. A sentence about one person that cites only memories about someone else is blocked, and so is a number found in none of the cited memories. Under the calibrated confidence floor Kin says it doesn't remember. | `test_uncited_answer_is_blocked`, `test_invented_citation_is_blocked`, `test_direct_answer_about_the_user_is_blocked_and_recall_runs_instead`, `test_an_answer_about_one_person_cannot_cite_another`, `test_numbers_in_an_answer_must_come_from_the_cited_memory` |
| Nothing deleted without confirmation | Only anchored, non-negated patterns; spoken or clicked "confirm" within 10 s; exactly the announced memories | `test_safety_guard.py` |
| Kin never memorises itself | The mic is ducked whenever Kin is audible | `test_kin_never_memorises_its_own_voice` |
| Mute closes the stream; the LED tells the truth | The stream is aborted and closed. The LED is derived from the real stream state. | `test_mute_*`, `test_led_*` |
| UI cannot be abused cross-site | Binds to 127.0.0.1. The Origin header is checked on the WebSocket and on every state change. | `test_cross_site_*`, `test_tasks_api_complete_delete_and_csrf` |
