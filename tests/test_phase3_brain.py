"""Phase 3 brain behaviour with fakes: lookup tool flow, offline honesty, LED blue, tasks and reminders."""

from __future__ import annotations

import asyncio
import datetime
import json
import time

import pytest

from kin.brain.llm import ToolCall
from kin.core.events import (
    ActivationEvent,
    Announcement,
    AttendingEnded,
    BrainState,
    BrainStateChanged,
    LedChanged,
    LedColor,
    LookupPerformed,
    ReplyCompleted,
    TaskChanged,
    TranscriptEvent,
)
from tests.conftest import FakeLLM, FakeLookupClient, FakeTTS, build_fake_app, make_fake_brain_factory

PRIYA = "I'm meeting Priya at 4 pm at the library cafe for coffee."
UMBRELLA = "Do I need an umbrella for the coffee with Priya I mentioned this morning?"
TODAY = datetime.date.today().isoformat()


@pytest.fixture
async def kin(tmp_path):
    created = []

    async def make(llm=None, tts=None, client="fake"):
        store: dict = {}
        lookup_client = FakeLookupClient() if client == "fake" else client
        h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, tts, store, lookup_client=lookup_client, shortcut=False))
        await h.app.start()
        await h.app.wait_brain()
        h.store, h.client = store, lookup_client
        created.append(h)
        return h

    yield make
    for h in created:
        await h.app.stop()


async def remember(h, text):
    await h.app.brain.memory.remember(text, "commitment", 0.9)


async def ask(h, text, timeout=5.0) -> ReplyCompleted:
    waiter = asyncio.create_task(h.bus.wait_for(ReplyCompleted, timeout=timeout))
    await asyncio.sleep(0)
    h.bus.publish(ActivationEvent(source="button"))
    h.bus.publish(TranscriptEvent(line_id=f"q{time.monotonic()}", text=text, final=True, mode="query"))
    h.bus.publish(AttendingEnded(reason="button_released"))
    return await waiter


def umbrella_llm(answer: str = "Yes, take an umbrella: rain chance is 80% at 4 pm [web] when you meet Priya [#1].") -> FakeLLM:
    return FakeLLM(turns=[ToolCall("lookup", {"kind": "weather", "query": "London", "when": "today"}), answer])


# --- the umbrella demo ----------------------------------------------------------------------------------


async def test_umbrella_demo_memory_plus_weather_decided_locally(kin):
    llm = umbrella_llm()
    h = await kin(llm=llm)
    await remember(h, PRIYA)
    reply = await ask(h, UMBRELLA)

    assert reply.outcome == "answered"
    assert reply.text.startswith("I checked the weather online.")  # honest, fixed sentence
    assert "umbrella" in reply.text and "80%" in reply.text
    assert reply.citations == "1"  # the Priya memory
    assert llm.tools_offered[0] == ["recall", "remember", "add_task", "list_tasks", "complete_task", "cancel_reminder",
                                   "correct_fact", "retract_fact", "review_memory", "lookup"]
    assert reply.path == "tool: lookup (weather)" and reply.llm_calls == 2  # one routing call + one answer
    # the model saw only the question, then the looked-up facts together with the memory, and decided
    messages = llm.queries
    assert messages == [UMBRELLA]
    result = next(m for _, m in llm.calls if m.startswith("TOOL RESULT: Forecast for London"))
    assert "16:00 80%" in result and "[#1]" in result and "library cafe" in result


async def test_network_monitor_payload_is_only_city_and_date(kin):
    h = await kin(llm=umbrella_llm())
    await remember(h, PRIYA)
    await ask(h, UMBRELLA)
    (kind, params), = h.client.requests
    assert kind == "weather" and params == {"city": "London", "date": TODAY}
    ev = h.rec.of(LookupPerformed)[-1]
    assert json.loads(ev.request) == {"kind": "weather", "params": {"city": "London", "date": TODAY}}
    for word in ("Priya", "umbrella", "coffee", "library", "meeting"):
        assert word.lower() not in ev.request.lower()


async def test_led_turns_blue_during_the_lookup(kin):
    h = await kin(llm=umbrella_llm())
    await remember(h, PRIYA)
    await ask(h, UMBRELLA)
    await asyncio.sleep(0.05)
    states = [e.state for e in h.rec.of(BrainStateChanged)]
    assert BrainState.LOOKUP in states
    assert LedColor.BLUE in [e.color for e in h.rec.of(LedChanged)]
    assert h.hw.led is LedColor.GREEN


async def test_offline_lookup_is_honest_and_recall_still_works(kin):
    llm = FakeLLM(turns=[
        ToolCall("lookup", {"kind": "weather", "query": "London"}),
        "It will probably rain [web]. You are meeting Priya at 4 pm at the library cafe [#1].",
    ])
    h = await kin(llm=llm)
    h.client.offline = True
    await remember(h, PRIYA)
    reply = await ask(h, UMBRELLA)
    assert reply.text.startswith("I can't check the weather online right now, because I'm in offline mode.")
    assert "probably rain" not in reply.text, "a [web] claim without web data must be blocked"
    # no second model pass after a failed lookup: the memory is quoted, cited, word for word
    assert reply.text.endswith(f"Here's what I remember: {PRIYA.rstrip('.')}.") and reply.citations == "1"
    assert reply.llm_calls == 1 and reply.outcome == "answered"
    assert h.rec.of(LookupPerformed)[-1].reason == "offline"
    assert not any(message.startswith("TOOL RESULT") for _, message in llm.calls)

    recall = await ask(h, "When am I meeting Priya?")
    assert recall.outcome == "answered" and "Priya" in recall.text


async def test_gateway_down_is_honest(kin):
    from kin.tools.lookup_client import LookupClient

    dead = LookupClient(r"\\.\pipe\kin-nobody-home" if __import__("sys").platform == "win32" else "/nonexistent/kin.sock",
                        "AF_PIPE" if __import__("sys").platform == "win32" else "AF_UNIX", b"k" * 32, timeout_s=1)
    h = await kin(llm=umbrella_llm(), client=dead)
    await remember(h, PRIYA)
    reply = await ask(h, UMBRELLA)
    assert "my lookup service isn't running" in reply.text


async def test_direct_answer_about_the_user_is_blocked_and_recall_runs_instead(kin):
    # Memory clearly matches, but the model answers without calling recall (and gets it wrong).
    llm = FakeLLM(turns=["You are meeting Priya at 5 pm."])
    h = await kin(llm=llm)
    await remember(h, PRIYA)
    reply = await ask(h, "When am I meeting Priya?")
    assert "5 pm" not in reply.text and "4 pm" in reply.text and reply.outcome == "answered"
    assert reply.path == "fallback: recall" and reply.llm_calls == 2 and reply.blocked_sentences == 1
    assert not h.client.requests


async def test_direct_small_talk_is_spoken_without_citations(kin):
    llm = FakeLLM()
    h = await kin(llm=llm)
    await remember(h, PRIYA)
    reply = await ask(h, "How are you?")
    assert reply.text == "Hello! Nice to talk to you." and reply.outcome == "chit_chat"
    assert reply.path == "answered directly" and reply.llm_calls == 1


async def test_tool_call_extra_fields_are_dropped_and_bad_values_refused_by_gateway(kin):
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    llm = FakeLLM(turns=[
        ToolCall("lookup", {"kind": "weather", "query": "London", "when": "tomorrow", "who": "Priya"}),  # extra field
        ToolCall("lookup", {"kind": "weather", "query": "London", "when": "tomorrow"}),  # after one correction
        "Fine [web].",
    ])
    h = await kin(llm=llm)
    reply = await ask(h, "Will it rain tomorrow?")
    assert h.client.requests == [("weather", {"city": "London", "date": tomorrow})]  # "who" never leaves the brain
    assert reply.path == "retry → tool: lookup (weather)" and reply.llm_calls == 3

    from pydantic import ValidationError
    with pytest.raises(ValidationError):  # the brain passes values through; the gateway schema rejects them
        __import__("kin.gateway.schemas", fromlist=["x"]).LOOKUP_ADAPTER.validate_python(
            {"kind": "weather", "params": {"city": "London<script>", "date": tomorrow}})


async def test_person_names_from_memory_are_sent_to_gateway_as_hashes_only(kin):
    h = await kin()
    await remember(h, PRIYA)
    await asyncio.sleep(0.1)
    await h.app.brain.drain()
    assert len(h.client.names) == 1 and "Priya" not in h.client.names[0] and len(h.client.names[0]) == 24


# --- tasks and reminders ----------------------------------------------------------------------------------


async def test_remind_me_at_5_creates_a_reminder_with_spoken_confirmation(kin):
    h = await kin()
    reply = await ask(h, "Remind me to email the professor at 5")
    assert reply.outcome == "command"
    assert reply.text.startswith("Okay, I'll remind you to email the professor at 5 PM")
    task = h.rec.of(TaskChanged)[-1]
    due = datetime.datetime.fromtimestamp(task.due_ts)
    assert task.action == "added" and task.kind == "reminder" and task.text == "Email the professor"
    assert due.hour == 17 and due.minute == 0


async def test_due_reminder_fires_and_is_spoken(kin):
    tts = FakeTTS()
    h = await kin(tts=tts)
    reply = await ask(h, "Remind me to email the professor in 1 second")
    assert "in 1 seconds" in reply.text or "in 1 second" in reply.text
    for _ in range(100):
        if h.rec.of(Announcement):
            break
        await asyncio.sleep(0.05)
    fired = h.rec.of(Announcement)[0]
    assert fired.text == "Reminder: Email the professor."
    await asyncio.sleep(0.2)
    assert "Reminder: Email the professor." in tts.said
    assert [t.action for t in h.rec.of(TaskChanged)] == ["added", "fired"]
    assert await h.store["tasks"].open() == []  # fired once, never again
    await asyncio.sleep(0.3)
    assert len(h.rec.of(Announcement)) == 1


async def test_reminder_waits_for_a_reply_in_progress(kin):
    tts = FakeTTS()
    h = await kin(llm=FakeLLM(delay_s=0.05), tts=tts)
    await remember(h, PRIYA)
    await h.store["tasks"].add("Stretch", time.time() + 0.1)
    await ask(h, "When am I meeting Priya?")
    for _ in range(100):
        if h.rec.of(Announcement):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)
    spoken = [t for t in tts.said if t != "Let me check."]
    assert spoken.index("Reminder: Stretch.") > 0, "the reminder must not cut into the answer"


async def test_task_list_complete_and_cancel(kin):
    h = await kin()
    assert (await ask(h, "Add buy oat milk to my shopping list")).text == "Added to your list: Buy oat milk."
    await ask(h, "Remind me to call mom tomorrow at 9")
    listing = await ask(h, "What's on my list?")
    assert "2 items" in listing.text and "Buy oat milk" in listing.text and "call mom" in listing.text.lower()
    assert "ticked off Buy oat milk" in (await ask(h, "Mark oat milk as done")).text
    assert (await ask(h, "Cancel the reminder to call mom")).text == "Cancel the reminder to call mom? Say confirm."
    assert len(await h.store["tasks"].open()) == 1, "nothing is cancelled before confirming"
    assert "cancelled the reminder to call mom" in (await ask(h, "confirm")).text
    assert (await ask(h, "What's on my list?")).text.startswith("Your list is empty")


async def test_reminder_without_time_becomes_a_task_honestly(kin):
    h = await kin()
    reply = await ask(h, "Remind me to renew my passport")
    assert reply.text == "Added to your list: Renew my passport."  # honest: no reminder is claimed
    assert (await h.store["tasks"].open())[0].kind == "task"


async def test_missed_reminders_are_spoken_late_or_expired(kin):
    tts = FakeTTS()
    h = await kin(tts=tts)
    tasks = h.store["tasks"]
    await h.app.brain.reminders.stop()
    await tasks.add("Old thing", time.time() - 2 * 86400)
    await tasks.add("Recent thing", time.time() - 600)
    h.app.brain.reminders._first = True
    spoken = await h.app.brain.reminders.tick()
    assert spoken == 1
    actions = {(e.text, e.action) for e in h.rec.of(TaskChanged)}
    assert ("Old thing", "expired") in actions and ("Recent thing", "fired") in actions
    assert h.rec.of(Announcement)[-1].text == "Earlier reminder: Recent thing."


async def test_tasks_survive_restart_encrypted(kin, tmp_path):
    h = await kin()
    await ask(h, "Add renew passport to my list")
    raw = (tmp_path / "kin.db").read_bytes()
    assert b"passport" not in raw


# --- passive: unsure lines wait for the LLM, and questions go first ------------------------------


async def test_episodes_wait_for_extraction_and_questions_go_first(tmp_path):
    from kin.core.events import EpisodeExtracted, MemoryStored, PassiveTagged
    from tests.conftest import StubRouter

    llm = FakeLLM(delay_s=0.02)
    store: dict = {}
    router = StubRouter(unsure={"The wifi password is tiger42."})
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, None, store, FakeLookupClient(), router=router, shortcut=False))
    h.app.cfg.extract.episode_gap_s = 0.2  # the conversation is over after 0.2 s of quiet
    await h.app.start()
    await h.app.wait_brain()
    try:
        brain = h.app.brain
        await brain.memory.remember(PRIYA, "commitment", 0.9)
        # A question starts, and while it runs Kin overhears a line; the conversation ends 0.2 s later.
        query = asyncio.create_task(brain.handle_query("When am I meeting Priya?"))
        await asyncio.sleep(0)
        h.bus.publish(TranscriptEvent(line_id="p1", text="The wifi password is tiger42.", final=True, mode="passive"))
        await query
        for _ in range(250):  # the episode loop closes it after the gap; extraction runs once idle
            if h.rec.of(EpisodeExtracted):
                break
            await asyncio.sleep(0.02)
        tags = [e for e in h.rec.of(PassiveTagged) if "tiger42" in e.text]
        assert [t.decision for t in tags] == ["queued"]
        messages = [m for _, m in llm.calls]
        asked = next(i for i, m in enumerate(messages) if m == "When am I meeting Priya?")
        extracted = next(i for i, m in enumerate(messages) if "tiger42" in m)
        assert asked < extracted, "the question must reach the LLM before background extraction"
        note = [e for e in h.rec.of(MemoryStored) if "tiger42" in e.text]
        assert note and note[0].kind == "note" and note[0].source == "extracted" and brain.pending_count == 0
        assert [e.outcome for e in h.rec.of(EpisodeExtracted)] == ["done"]
        raw = (tmp_path / "kin.db").read_bytes()
        assert b"tiger42" not in raw  # raw lines and facts are encrypted like everything else
    finally:
        await h.app.stop()
