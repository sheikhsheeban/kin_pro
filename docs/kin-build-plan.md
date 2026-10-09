# Kin — Build Plan for Claude Code (Laptop Demo First)

This plan splits the Kin build into 4 phases. Each phase ends with something you can demo, and each has a ready-to-paste prompt for Claude Code.

**Demo target for now:** laptop only. Everything that would be physical on the Raspberry Pi (mute switch, listening LED, push-to-talk button, speaker state, Wi-Fi switch) is a control in a **Simulated Device Panel** in the web UI. The code talks to these through a hardware interface, so on the Pi you swap the simulated version for a GPIO version without touching anything else.

---

## How to use this with Claude Code

1. Create an empty repo folder, put `CLAUDE.md` (provided separately) in the root, and put this file at `docs/BUILD_PLAN.md`.
2. Also copy `kin-architecture.md` and the diagram into `docs/` so Claude Code can read the architecture.
3. For each phase:
   - Start Claude Code in **plan mode** (Shift+Tab twice) and paste the phase prompt.
   - Review the plan it proposes, then let it build.
   - Run the phase's acceptance checks yourself.
   - Commit with a tag: `git tag phase-1`, and so on.
4. Do not start a phase until the previous phase's checks pass. Small, working steps beat one big unfinished build.

---

## Laptop stand-ins for Pi hardware

| On the Raspberry Pi | On the laptop demo | How it stays honest |
|---|---|---|
| Physical mute switch cutting mic power | **Mute toggle** in the Device Panel | Toggling closes the audio input stream entirely (not just ignoring frames). The UI shows "mic stream: closed". |
| Listening LED wired to mic power | **LED indicator** in the Device Panel | Driven only by the real state of the audio stream, never by app logic. If the stream is open, the LED is on. |
| LED colour states | Same indicator, 5 colours | Off = muted · green = listening · white pulse = attending · amber = thinking · blue = online lookup |
| Push-to-talk button | **Hold-to-talk button** in the UI (and spacebar) | Same event as the wake word on the bus |
| I²S speaker | Laptop speaker via Piper | Same TTS module |
| Wi-Fi on/off | **Offline mode toggle** | Gateway refuses all calls; shows that reasoning still works |
| nftables firewall + `PrivateNetwork=yes` | Separate gateway process + socket guard in the brain process | Brain process blocks any outbound socket at start-up; Network Monitor lists every gateway call |
| 4 GB RAM limit | **"Pi 5 mode"** profile | Uses the Pi models, 4 threads, 2K context, and shows live RAM against a 4 GB bar |

## UI layout (one local web page, `http://localhost:8000`)

- **Left — Simulated Device Panel:** mute toggle, LED, hold-to-talk, offline toggle, Pi 5 mode toggle, live mic level meter.
- **Centre — Conversation:** live transcript stream, Kin's replies, "spoken" indicator, badges on answers ("from memory #42", "checked weather online").
- **Right — tabs:**
  - Memory: list, search, delete, "forget last 10 minutes"
  - Network Monitor: every lookup with timestamp, kind, exact payload sent, response size
  - System: RAM per component, tokens/s, time to first audio, model names
- Built with FastAPI + WebSocket + a single HTML/JS page (no heavy frontend framework needed).

---

## Phase 1 — Foundation and "Ears"

**Goal:** the laptop listens continuously, transcribes live, and the simulated hardware controls really work.

**Scope**
- Repo skeleton matching the software layout in the architecture doc.
- `hal/` hardware interface with `SimulatedHardware` (UI-driven) and an empty `PiGpioHardware` stub.
- Event bus (asyncio) with typed events.
- Audio: capture → 30 s RAM ring buffer → Silero VAD → openWakeWord → Moonshine v2 streaming STT.
- Web UI with the Device Panel, live transcript and mic level meter.
- Config files: `config/laptop.yaml`, `config/pi5.yaml`.

**Acceptance checks**
- Speaking produces a live transcript in the UI within about 1 s.
- Mute toggle closes the audio stream within 200 ms; LED goes off; transcript stops.
- LED colour follows real pipeline state.
- A test proves no audio file is ever written to disk (scan the project and temp folders after a run).
- Wake word ("Hey Kin" or a stock word for now) and hold-to-talk both fire the same `ActivationEvent`.

**Prompt for Claude Code**
```text
Read CLAUDE.md, docs/kin-architecture.md and docs/BUILD_PLAN.md. Build Phase 1 only.

Create the Python project (uv, Python 3.11+) with the module layout from the architecture doc.
1. hal/: an abstract Hardware interface (mute state, LED state, button events, offline flag)
   with SimulatedHardware driven by the web UI and a stub PiGpioHardware that raises NotImplementedError.
2. core/bus.py: asyncio event bus with typed dataclass events.
3. audio/: microphone capture (sounddevice, 16 kHz mono) into a 30-second RAM-only ring buffer,
   Silero VAD gating, openWakeWord detection, Moonshine v2 streaming STT (Tiny for passive, Small for queries).
   Muting must close the input stream, not just drop frames. Never write audio to disk.
4. ui/: FastAPI app with a WebSocket and a single HTML/JS page containing the Simulated Device Panel
   (mute toggle, LED with 5 colour states, hold-to-talk button + spacebar, offline toggle, Pi 5 mode toggle,
   mic level meter) and a live transcript pane. LED state must be derived from the real audio stream state.
5. config/laptop.yaml and config/pi5.yaml; a single `kin run --profile laptop` entry point.
6. Tests: mute closes stream within 200 ms; no .wav/.pcm/.raw files created during a run;
   activation event fires from both wake word and button.
Stop when Phase 1 acceptance checks pass and summarise how to run it.
```

---

## Phase 2 — "Brain and Memory" (core use case end-to-end)

**Goal:** Kin remembers what it hears and answers recall questions out loud, fully offline.

**Scope**
- LLM backend interface with two implementations: LiteRT-LM (Gemma 4 E2B, the Pi target) and llama-cpp-python (fallback). Try LiteRT-LM on the laptop first so there are no surprises on the Pi.
- Intent router (start with rules + Gemma 3 270M prompt; fine-tune later). *Superseded by the routing refactor: regex safety guard + Gemma 4 E2B tool calls + EmbeddingGemma semantic tagger; Gemma 3 270M removed (see architecture §2).*
- Passive capture: tag utterances (`commitment`, `fact_about_person`, `idea`, `chatter`), drop chatter, embed with EmbeddingGemma 300M, store in SQLCipher + sqlite-vec.
- Active recall: retrieve top-k, prompt Gemma, require a memory citation, confidence gate, speak with Piper.
- Memory tab in UI: browse, search, delete, "forget last 10 minutes".
- System tab: RAM per component, tokens/s, time to first audio.

**Acceptance checks**
- Say "I'm meeting Priya at 4 pm at the library café", then later ask "When am I meeting Priya?" Kin answers correctly, aloud, with "from memory #N" in the UI.
- Ask about something never said. Kin says it doesn't remember rather than inventing.
- Works with the laptop's network disconnected.
- In Pi 5 mode, total RAM stays under about 3.2 GB.

**Prompt for Claude Code**
```text
Read CLAUDE.md and docs/. Phase 1 is done (tag phase-1). Build Phase 2 only.

1. brain/llm.py: an LLMBackend interface with LiteRTLMBackend (Gemma 4 E2B int4) and LlamaCppBackend
   (GGUF fallback). Selected by config. Support tool/function calling and a hard context cap (2048 in pi5 profile).
   Add a script that downloads the models into ./models and prints sizes.
2. brain/router.py: classify each utterance into commitment, fact_about_person, idea, chatter, query
   (rules first, then Gemma 3 270M prompt behind the same interface).
   [Superseded: brain/commands.py (safety guard), brain/tools_schema.py, brain/semantic_router.py.]
3. memory/: EmbeddingGemma 300M embeddings (truncate to 256 dims), SQLCipher database with sqlite-vec,
   schema with id, text, kind, speaker, timestamp, confidence. Passive pipeline stores non-chatter items.
4. brain/orchestrator.py: on activation, transcribe with Moonshine Small, retrieve top-k, build a short prompt,
   call the LLM, require the answer to cite memory ids, pass through brain/gate.py (similarity floor,
   missing-citation check, 20 s timeout), then speak with Piper TTS streaming sentence by sentence.
   Play a short "Let me check" acknowledgement while the model works.
5. UI: Memory tab (list, search, delete, forget last 10 minutes) and System tab (RAM per component,
   tokens/s, time to first audio). Reply bubbles show "from memory #N".
6. Tests: recall happy path, unknown-fact abstention, forget command, works with networking disabled.
Stop when Phase 2 acceptance checks pass.
```

---

## Phase 3 — Online boundary and productivity

**Goal:** show the strict online/offline split in action, and add the second domain (tasks and reminders).

**Scope**
- Lookup Gateway as a **separate process**, talking to the brain over a local Unix socket (named pipe on Windows).
- Gateway accepts only `{"kind": "weather|search|news", "params": {...}}`, schema-validated, length-capped, with names from the memory store scrubbed.
- Providers: weather (Open-Meteo, no key needed), search and news (pick APIs with free tiers; keys in `.env`).
- Brain process installs a socket guard at start-up so any outbound connection raises an error.
- LLM `lookup()` tool call wired to the gateway; LED turns blue during calls.
- Network Monitor tab: every call, exact payload, response size, duration.
- Offline toggle: gateway refuses calls; Kin says it can't check online but still reasons locally.
- Local tools: tasks and reminders with spoken confirmation and due-time alerts.

**Acceptance checks**
- The umbrella demo works: meeting details from memory + weather from API + decision made locally.
- Network Monitor shows the payload contains only city and date, no transcript text.
- A test proves the brain process cannot open an outbound socket.
- With offline toggle on, recall still works and lookups fail gracefully with an honest message.
- "Remind me to email the professor at 5" creates a reminder that fires and is spoken.

**Prompt for Claude Code**
```text
Read CLAUDE.md and docs/. Phases 1–2 are done. Build Phase 3 only.

1. gateway/: a separate process (own entry point, own uvicorn/asyncio server on a Unix socket or
   Windows named pipe). Pydantic schemas allowing only kind in {weather, search, news} with strict params,
   max lengths, and scrubbing of any person names found in the memory store. Providers: Open-Meteo weather,
   plus search and news providers with API keys from .env. Every call is appended to an audit log.
2. Brain process: at start-up install a socket guard (patch socket.socket.connect / create_connection)
   so any non-local connection raises. Add a test that proves it.
3. Expose lookup(kind, params) as an LLM tool. Results come back as data and are cited in the reply
   ("I checked the weather online"). Set LED to blue during calls.
4. UI: Network Monitor tab listing timestamp, kind, exact JSON sent, response size, duration.
   Offline toggle makes the gateway refuse calls; Kin must say so honestly and keep reasoning locally.
5. tools/: tasks and reminders stored locally, created by voice, with a scheduler that speaks reminders when due.
6. A scripted demo runner (scripts/demo.py) that plays the umbrella scenario end to end.
Stop when Phase 3 acceptance checks pass.
```

---

## Phase 4 — Hardening, Pi-readiness and demo polish

**Goal:** a reliable, rehearsed laptop demo, and code that is ready to move to the Raspberry Pi 5 with a config change.

**Scope**
- Honest fallback: capability map for out-of-scope requests, retrieval floor tuning, timeout messages.
- Speed: KV cache for the fixed system prompt, shorter retrieval snippets, lazy-load Moonshine Small and the embedder in Pi 5 mode.
- Pi 5 mode realism: 4 threads, 2K context, RAM bar against 4 GB, warning if over budget.
- `PiGpioHardware` implemented (gpiozero) for switch, LED and button, but disabled on the laptop.
- Deploy files for later: `kin-brain.service` (`PrivateNetwork=yes`), `kin-gateway.service`, `nftables.conf`, install script for Raspberry Pi OS Lite 64-bit.
- Nightly memory consolidation job.
- Crash recovery: each module restarts on failure; UI shows component health.
- README with setup, demo script and talking points.

**Acceptance checks**
- 10 back-to-back demo runs without a crash.
- Time to first audio under 2 s on the laptop profile; Pi 5 mode reports its estimated figure.
- Out-of-scope request ("do my tax return") gets an honest, fixed reply.
- `kin run --profile pi5` starts with the right models and limits.
- Full test suite passes.

**Prompt for Claude Code**
```text
Read CLAUDE.md and docs/. Phases 1–3 are done. Build Phase 4 only.

1. brain/gate.py: capability map for out-of-scope requests with fixed honest replies; tune the similarity
   floor using a small labelled set in tests/fixtures.
2. Performance: cache the KV state of the fixed system prompt, trim retrieved snippets, lazy-load
   Moonshine Small and the embedder in the pi5 profile. Log time to first audio per query.
3. Pi 5 mode: 4 threads, 2048 context, RAM usage bar against 4 GB with a warning over 3.5 GB.
4. hal/pi_gpio.py: implement PiGpioHardware with gpiozero (switch on GPIO17, LED on GPIO27, button on GPIO22),
   selected only when profile=pi5 and running on a Pi.
5. deploy/: systemd units (brain with PrivateNetwork=yes, gateway as its own user), nftables allowlist,
   and an install script for Raspberry Pi OS Lite 64-bit. Do not run them on the laptop.
6. memory/consolidate.py: nightly summarise and merge duplicates.
7. Supervisor that restarts failed modules; component health shown in the UI.
8. README.md with setup, demo script and judge talking points. Make sure the full test suite passes.
Stop when Phase 4 acceptance checks pass.
```

---

## After Phase 4: moving to the Raspberry Pi 5

1. Flash Raspberry Pi OS Lite 64-bit, run the install script.
2. `kin run --profile pi5` with `SimulatedHardware` first (UI still works from another device on the network).
3. Measure real RAM and latency; update the architecture doc's estimates.
4. Wire the switch, LED and button; switch to `PiGpioHardware`.
5. Enable the systemd units and nftables rules; repeat the umbrella demo with a packet capture running.
