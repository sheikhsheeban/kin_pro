"""The safety guard: only anchored, non-negated destructive commands, and never without confirmation.

Every phrase that the old keyword router got wrong is here, plus the positive cases.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from kin.brain.commands import SafeCommand, clean_query, confirmation_answer, parse_duration, parse_safe_command
from kin.brain.llm import ToolCall
from kin.core.events import (
    ActivationEvent,
    AttendingEnded,
    ConfirmationRequested,
    ConfirmationResolved,
    MemoriesDeleted,
    ReplyCompleted,
    TaskChanged,
    TranscriptEvent,
)
from tests.conftest import FakeLLM, FakeLookupClient, build_fake_app, make_fake_brain_factory

# (phrase, what the old keyword router did)
WHY_NOT_DESTRUCTIVE = [
    ("can you remind me what Rahul eats", "remind command"),
    ("don't forget the last 5 minutes of the meeting", "deleted 5 minutes"),
    ("I always forget what we said in the last hour", "deleted 60 minutes"),
    ("don't forget it", "deleted the last memory"),
    ("where did I leave my jacket", "weather lookup"),
    ("what did Priya say about the online course", "web search"),
    ("what news did Mum share", "news lookup"),
    ("hey what did Rahul say yesterday", "chit-chat"),
    ("I've finished reading the book Rahul lent me", "completed a task"),
]


@pytest.mark.parametrize(("text", "old"), WHY_NOT_DESTRUCTIVE)
def test_why_phrases_are_not_destructive_commands(text, old):
    assert parse_safe_command(text) is None, f"was: {old}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("forget the last ten minutes", SafeCommand("forget_span", minutes=10)),
        ("Forget the last ten minutes.", SafeCommand("forget_span", minutes=10)),
        ("forget the last half an hour", SafeCommand("forget_span", minutes=30)),  # was missed
        ("forget the last hour", SafeCommand("forget_span", minutes=60)),
        ("please forget what I said in the last 5 minutes", SafeCommand("forget_span", minutes=5)),
        ("forget everything from the past 2 hours", SafeCommand("forget_span", minutes=120)),
        ("Hey Kin, forget the last eleven minutes", SafeCommand("forget_span", minutes=11)),
        ("can you forget the last 90 mins", SafeCommand("forget_span", minutes=90)),
        ("forget that", SafeCommand("forget_last")),
        ("Forget what I just said.", SafeCommand("forget_last")),
        ("okay Jarvis, forget it", SafeCommand("forget_last")),
        ("cancel the reminder about the oven", SafeCommand("cancel_reminder", text="the oven")),
        ("Delete my reminder to call mom", SafeCommand("cancel_reminder", text="call mom")),
    ],
)
def test_destructive_commands_are_recognised(text, expected):
    assert parse_safe_command(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "forget the last 5 minutes of the meeting",  # not anchored to the end: a remark, not a command
        "I keep forgetting the last hour of the lecture",
        "never forget that",
        "I won't forget that",
        "did you forget that?",
        "why do I forget everything",
        "we should forget the last ten minutes, honestly",
        "remind me to cancel the reminder app subscription",
    ],
)
def test_negated_habitual_or_embedded_forms_are_refused(text):
    assert parse_safe_command(text) is None


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("ten minutes", 10), ("half an hour", 30), ("half hour", 30), ("an hour", 60), ("hour", 60),
        ("eleven minutes", 11), ("90 mins", 90), ("1.5 hours", 90), ("an hour and a half", 90),
        ("one and a half hours", 90), ("twenty five minutes", 25), ("twenty-five minutes", 25),
        ("a couple of hours", 120), ("thirty seconds", 0.5), ("2 hrs", 120),
    ],
)
def test_parse_duration(text, minutes):
    assert parse_duration(text) == pytest.approx(minutes)


@pytest.mark.parametrize("text", ["the meeting", "", "minutes ago", "lots of minutes"])
def test_parse_duration_rejects_non_durations(text):
    assert parse_duration(text) is None


def test_clean_query_and_confirmation_words():
    assert clean_query("hey what did Rahul say yesterday") == "what did Rahul say yesterday"
    assert clean_query("Hey Jarvis, when is it?") == "when is it?"
    assert clean_query("okay kin what's up") == "what's up"
    for yes in ["confirm", "Confirm.", "yes", "Yes please", "Hey Kin, confirm", "go ahead", "do it"]:
        assert confirmation_answer(yes) is True, yes
    for no in ["no", "cancel", "Keep them", "never mind"]:
        assert confirmation_answer(no) is False, no
    for other in ["what time is it", "yes I went to the shop", "confirm my booking"]:
        assert confirmation_answer(other) is None, other


# --- through the brain: nothing is ever deleted without confirmation ------------------------------------

PRIYA = "I'm meeting Priya at 4 pm at the library cafe for coffee."
RAHUL = "Rahul is allergic to peanuts."


@pytest.fixture
async def kin(tmp_path):
    created = []

    async def make(llm=None):
        store: dict = {}
        h = build_fake_app(
            brain_factory=make_fake_brain_factory(tmp_path, llm or FakeLLM(), None, store, lookup_client=FakeLookupClient(), shortcut=False)
        )
        await h.app.start()
        await h.app.wait_brain()
        h.store = store
        created.append(h)
        mem = h.app.brain.memory
        await mem.remember(RAHUL, "fact_about_person", 0.9, ts=time.time() - 120)
        await mem.remember(PRIYA, "commitment", 0.9, ts=time.time() - 60)
        return h

    yield make
    for h in created:
        await h.app.stop()


async def ask(h, text, timeout=5.0) -> ReplyCompleted:
    waiter = asyncio.create_task(h.bus.wait_for(ReplyCompleted, timeout=timeout))
    await asyncio.sleep(0)
    h.bus.publish(ActivationEvent(source="button"))
    h.bus.publish(TranscriptEvent(line_id=f"q{time.monotonic()}", text=text, final=True, mode="query"))
    h.bus.publish(AttendingEnded(reason="button_released"))
    return await waiter


@pytest.mark.parametrize(("text", "old"), WHY_NOT_DESTRUCTIVE)
async def test_why_phrases_delete_and_complete_nothing(kin, text, old):
    # The model's choice is scripted to the *worst* plausible tool for each phrase: completion for
    # "I've finished...", otherwise whatever the fake routes to. The guard and the tools must hold.
    turns = [ToolCall("complete_task", {"text": "reading the book Rahul lent me"})] if "finished" in text else None
    h = await kin(llm=FakeLLM(turns=turns))
    await h.store["tasks"].add("Email the professor")
    reply = await ask(h, text)
    assert not h.rec.of(MemoriesDeleted), f"{text!r} deleted memories (was: {old})"
    assert not h.rec.of(ConfirmationRequested), f"{text!r} asked to delete something"
    assert await h.app.brain.memory.count() == 2
    assert not [e for e in h.rec.of(TaskChanged) if e.action in ("completed", "deleted")]
    assert not reply.path.startswith("safety guard")


@pytest.mark.parametrize(
    ("text", "spoken", "remaining"),
    [
        ("forget the last ten minutes", "That's 2 memories from the last 10 minutes. Say confirm to delete.", 0),
        ("forget the last half an hour", "That's 2 memories from the last 30 minutes. Say confirm to delete.", 0),
        ("forget that", f"That's your last memory: {PRIYA.rstrip('.')}. Say confirm to delete.", 1),
    ],
)
async def test_forget_asks_first_and_deletes_only_on_confirm(kin, text, spoken, remaining):
    h = await kin()
    reply = await ask(h, text)
    assert reply.text == spoken and reply.llm_calls == 0
    assert not h.rec.of(MemoriesDeleted) and await h.app.brain.memory.count() == 2
    req = h.rec.of(ConfirmationRequested)[-1]
    assert req.summary == spoken and req.expires_ts > time.time()

    done = await ask(h, "confirm")
    assert done.text.startswith("Done") and done.path == "confirmation"
    assert await h.app.brain.memory.count() == remaining
    assert h.rec.of(ConfirmationResolved)[-1].outcome == "confirmed"


async def test_no_cancels_and_another_question_cancels(kin):
    h = await kin()
    await ask(h, "forget the last ten minutes")
    reply = await ask(h, "no")
    assert reply.text == "Okay, I kept 2 memories from the last 10 minutes."
    assert await h.app.brain.memory.count() == 2

    await ask(h, "forget that")
    other = await ask(h, "When am I meeting Priya?")  # not an answer: the request is dropped, the question answered
    assert "Priya" in other.text and other.path == "tool: recall"
    assert h.rec.of(ConfirmationResolved)[-1].outcome == "cancelled"
    late = await ask(h, "confirm")  # nothing is pending any more
    assert not late.text.startswith("Done") and await h.app.brain.memory.count() == 2


async def test_confirmation_expires_and_deletes_nothing(kin, monkeypatch):
    import kin.brain.orchestrator as orch

    monkeypatch.setattr(orch, "CONFIRM_WINDOW_S", 0.1)
    h = await kin()
    await ask(h, "forget the last ten minutes")
    await asyncio.sleep(0.3)
    assert h.rec.of(ConfirmationResolved)[-1].outcome == "expired"
    late = await ask(h, "yes")
    assert not late.text.startswith("Done") and await h.app.brain.memory.count() == 2
    assert not h.rec.of(MemoriesDeleted)


async def test_ui_confirm_button_deletes_and_speaks(kin):
    h = await kin()
    await ask(h, "forget the last ten minutes")
    pending = h.app.brain.pending_confirmation
    assert pending and pending["action"] == "forget_span"
    assert not await h.app.brain.resolve_confirmation("wrong-id", True)
    assert await h.app.brain.memory.count() == 2
    assert await h.app.brain.resolve_confirmation(pending["confirm_id"], True)
    assert await h.app.brain.memory.count() == 0 and h.app.brain.pending_confirmation is None


async def test_cancel_reminder_needs_confirmation_by_guard_and_by_tool(kin):
    h = await kin(llm=FakeLLM(turns=[ToolCall("cancel_reminder", {"text": "oven"})]))
    tasks = h.store["tasks"]
    await tasks.add("Check the oven", time.time() + 600)
    # through the model's tool call
    reply = await ask(h, "stop the oven thing")
    assert reply.text == "Cancel the reminder to check the oven? Say confirm." and reply.path == "tool: cancel_reminder"
    assert len(await tasks.open()) == 1
    await ask(h, "no")
    # through the guard
    reply = await ask(h, "cancel the reminder about the oven")
    assert reply.path == "safety guard: cancel_reminder" and len(await tasks.open()) == 1
    await ask(h, "confirm")
    assert await tasks.open() == []


async def test_forget_confirmation_also_drops_raw_lines_not_yet_extracted(kin):
    h = await kin()
    mem = h.app.brain.memory
    await mem.hear("The door code is 4512.", time.time(), speaker=None, stt_model=None, worth=True, gap_s=90, max_s=600)
    await ask(h, "forget the last ten minutes")
    assert await mem.run(lambda s: s.utterance_count()) == 1, "nothing is deleted before confirming"
    await ask(h, "confirm")
    assert await mem.run(lambda s: s.utterance_count()) == 0
    assert await mem.episode_counts() == {}
