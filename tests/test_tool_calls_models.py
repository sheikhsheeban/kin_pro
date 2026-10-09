"""The WHY phrases through the real pipeline: Gemma 4 E2B with tools, EmbeddingGemma, Piper, the gateway process.

Slow (one model load, ~20 real queries). Skips when the models are not downloaded.
Run: uv run pytest -m slow -s
"""

from __future__ import annotations

import os
import time

import pytest

from kin.app import KinApp
from kin.audio import build_engines
from kin.brain import build_brain
from kin.config import load_config
from kin.core.bus import EventBus
from kin.core.events import ConfirmationRequested, MemoriesDeleted, TaskChanged
from tests.conftest import MODELS, FakeInputStream, Recorder

pytestmark = [
    pytest.mark.models,
    pytest.mark.slow,
    pytest.mark.skipif(not (MODELS / "llm" / "gemma-4-E2B-it.litertlm").exists(), reason="models not downloaded"),
]

MEMORIES = [
    "I'm meeting Priya at 4 pm at the library cafe for coffee.",
    "Rahul only eats vegetarian food, no eggs either.",
    "I left my jacket at Sam's place after dinner.",
    "Priya said the online course on data science starts in January.",
    "Mum shared the news that cousin Anil got a new job in Pune.",
    "Rahul said yesterday that he is moving to Bangalore next month.",
]
RECALL = [
    "can you remind me what Rahul eats",
    "where did I leave my jacket",
    "what did Priya say about the online course",
    "what news did Mum share",
    "hey what did Rahul say yesterday",
]
NOT_DESTRUCTIVE = [
    "don't forget the last 5 minutes of the meeting",
    "I always forget what we said in the last hour",
    "don't forget it",
    "I've finished reading the book Rahul lent me",
]
UMBRELLA = "Do I need an umbrella for the coffee with Priya I mentioned this morning?"


@pytest.fixture
async def real_kin(tmp_path):
    os.environ["KIN_DB_KEY"] = "ab" * 32
    cfg = load_config("laptop")
    cfg.data_dir = tmp_path
    cfg.tts.sink = "null"
    cfg.tts.ack_delay_s = 30
    cfg.gateway.providers = "fake"
    cfg.router.pending_tagger = False  # keep the LLM for the questions under test
    bus = EventBus()
    rec = Recorder(bus)
    app = KinApp(cfg, bus=bus, engines_factory=build_engines, stream_factory=lambda **kw: FakeInputStream(**kw),
                 device_namer=lambda _d: "none", brain_factory=build_brain)
    await app.start()
    await app.wait_gateway()
    await app.wait_brain()
    for i, text in enumerate(MEMORIES):
        await app.brain.memory.remember(text, "commitment", 0.9, ts=time.time() - 7200 + i)  # two hours ago
    await app.brain.sync_names()
    await app.brain.tasks.add("Email the professor")
    yield app, rec
    await app.stop()


async def test_why_phrases_choose_the_right_tool_and_never_delete(real_kin):
    app, rec = real_kin
    brain = app.brain
    problems: list[str] = []
    report: list[str] = []

    async def ask(text):
        r = await brain.handle_query(text)
        report.append(f"  {r.path or '-':38} {r.llm_calls} call(s) {r.time_to_first_audio_ms or 0:>6} ms | {text!r} -> {r.text!r}")
        return r

    for text in RECALL:
        r = await ask(text)
        if "recall" not in r.path:
            problems.append(f"{text!r}: expected recall, got {r.path!r}")
        if r.llm_calls > 2:
            problems.append(f"{text!r}: {r.llm_calls} LLM calls")

    for text in NOT_DESTRUCTIVE:
        await ask(text)
    if rec.of(MemoriesDeleted) or rec.of(ConfirmationRequested):
        problems.append("a non-destructive phrase deleted or asked to delete something")
    if [e for e in rec.of(TaskChanged) if e.action in ("completed", "deleted")]:
        problems.append("a task was completed or deleted")
    if await brain.memory.count() < len(MEMORIES):
        problems.append("memories disappeared")

    chat = await ask("How are you?")
    if chat.llm_calls != 1 or chat.path != "answered directly":
        problems.append(f"chit-chat: {chat.path!r} with {chat.llm_calls} calls")

    umbrella = await ask(UMBRELLA)
    if umbrella.path != "tool: lookup (weather)" or umbrella.llm_calls > 2:
        problems.append(f"umbrella: {umbrella.path!r} with {umbrella.llm_calls} calls")
    if not umbrella.text.startswith("I checked the weather online."):
        problems.append(f"umbrella answer: {umbrella.text!r}")

    # the destructive one: recognised by the guard, deleted only after "confirm"
    await brain.memory.remember("The plumber comes on Tuesday at nine.", "commitment", 0.9)
    count = await brain.memory.count()
    recent = len(await brain.memory.ids_since(30))
    forget = await ask("forget the last half an hour")
    if forget.path != "safety guard: forget_span" or await brain.memory.count() != count:
        problems.append(f"forget: {forget.path!r} / {forget.text!r}")
    if f"That's {recent} memor" not in forget.text:
        problems.append(f"forget announced the wrong count: {forget.text!r} (expected {recent})")
    done = await ask("confirm")
    if await brain.memory.count() != count - recent or await brain.memory.ids_since(30):
        problems.append(f"confirm did not delete exactly the last half hour: {done.text!r}")
    if await brain.memory.count() < len(MEMORIES):
        problems.append("confirm deleted memories older than half an hour")

    print("\n" + "\n".join(report))
    assert not problems, "\n".join(problems)
