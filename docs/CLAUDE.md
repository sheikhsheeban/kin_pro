# CLAUDE.md — Kin

Kin is a private, always-listening personal AI companion. All reasoning runs on the device. The only network access allowed is factual lookups (weather, search, news) through a separate gateway process.

Read `docs/kin-architecture.md` for the architecture and `docs/BUILD_PLAN.md` for the phase you are working on. Build only the phase you are asked to build.

## Non-negotiable rules

- **No cloud reasoning.** Never call a hosted LLM or any cloud AI API. The LLM, STT, TTS, embeddings and routing all run locally.
- **Raw audio never leaves RAM.** Do not write audio to disk, logs, temp files or test fixtures. Audio lives only in the ring buffer.
- **Only the gateway touches the network.** The brain process must not open outbound sockets. Lookups go through `lookup(kind, params)` with kind in {weather, search, news}. Never send transcripts, prompts or memory text to the gateway.
- **Mute means the mic stream is closed**, not ignored. The LED state comes from the real stream state.
- **Hardware goes through `hal/`.** Never call GPIO or UI controls directly from other modules.
- **Honest answers.** If the model is unsure or retrieval is weak, say so. Recall answers must cite memory ids.

## Stack

- Python 3.11+, managed with `uv`. asyncio throughout.
- LLM: Gemma 4 E2B via LiteRT-LM (primary), llama-cpp-python GGUF (fallback), behind `brain/llm.py`.
- STT: Moonshine v2 streaming (Tiny passive, Small queries). VAD: Silero. Wake word: openWakeWord.
- Embeddings: EmbeddingGemma 300M. Store: SQLCipher + sqlite-vec.
- Routing, three layers, no extra model:
  - Safety guard (`brain/commands.py`): anchored regexes for destructive commands only (forget, cancel reminder); never negated or habitual forms; nothing is deleted without a spoken or clicked "confirm".
  - Active queries: one Gemma 4 E2B call with native function calling (`brain/tools_schema.py`), at most two tool calls; every LLM use goes through one priority queue (questions first).
  - Passive speech: every non-filler line is an encrypted utterance, grouped into episodes; E2B extracts facts per finished episode in the background (`brain/extract.py`), and updates/supersedes instead of duplicating. The EmbeddingGemma k-NN router (`brain/semantic_router.py`) is only a pre-filter for all-chatter episodes. Speakers are never guessed.
- TTS: Piper (medium voice).
- UI: FastAPI + WebSocket + one HTML/JS page.

## Profiles

- `laptop` — default for the demo; SimulatedHardware; may use larger context.
- `pi5` — Raspberry Pi 5 limits: 4 threads, 2048 context, lazy-loaded models, RAM target under 3.2 GB.

## Working conventions

- Keep modules small and behind interfaces so laptop and Pi builds differ only in config.
- Add tests for every privacy rule (no audio on disk, brain socket guard, mute closes stream).
- Run `uv run pytest` before saying a phase is done.
- Summarise how to run and test at the end of each phase.
