"""Tool calling with a mocked LLM: schema validation, retry once, fallback to recall, the two-call
limit, lookup arguments capped and scrubbed, the warm session, and the LLM priority queue."""

from __future__ import annotations

import asyncio
import time

import pytest

from kin.brain.llm import ToolCall
from kin.brain.llm_queue import ACTIVE, BACKGROUND, LlmQueue
from kin.brain.tools_schema import TOOLS, InvalidCall, scrub_lookup_query, system_prompt, validate
from kin.core.events import ActivationEvent, AttendingEnded, ReplyCompleted, TranscriptEvent
from tests.conftest import FakeLLM, FakeLookupClient, build_fake_app, make_fake_brain_factory

PRIYA = "I'm meeting Priya at 4 pm at the library cafe for coffee."

# --- schema validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (ToolCall("recall", {"question": "When am I meeting Priya?"}), {"question": "When am I meeting Priya?"}),
        (ToolCall("add_task", {"text": "email the professor", "due": "at 5"}), {"text": "email the professor", "due": "at 5"}),
        (ToolCall("add_task", {"text": "buy milk", "due": None}), {"text": "buy milk", "due": None}),
        (ToolCall("add_task", {"text": "buy milk"}), {"text": "buy milk", "due": None}),
        (ToolCall("list_tasks", {}), {}),
        (ToolCall("lookup", {"kind": "news", "query": "Mars"}), {"kind": "news", "query": "Mars", "when": None}),
    ],
)
def test_valid_calls(call, expected):
    assert validate(call).args == expected


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (ToolCall("send_email", {"to": "x"}), "unknown tool"),
        (ToolCall("recall", {}), "needs 'question'"),
        (ToolCall("recall", {"question": "  "}), "non-empty"),
        (ToolCall("recall", {"question": "x", "secret": "y"}), "unexpected argument"),
        (ToolCall("lookup", {"kind": "email", "query": "x"}), "weather or news or search"),
        (ToolCall("lookup", {"kind": "news", "query": 42}), "a string"),
        (ToolCall("list_tasks", {"raw": "{bad"}), "JSON object"),
    ],
)
def test_invalid_calls(call, reason):
    with pytest.raises(InvalidCall, match=reason):
        validate(call)


def test_prompt_and_tools_are_fixed():
    # Byte-identical between calls, so the prefix can be prefilled ahead of time.
    assert system_prompt() == system_prompt()
    assert [t.name for t in TOOLS] == ["recall", "remember", "add_task", "list_tasks", "complete_task", "cancel_reminder",
                                         "correct_fact", "retract_fact", "review_memory", "lookup"]
    assert "Now:" not in system_prompt()  # nothing that changes per query lives in the prefix


def test_scrub_lookup_query():
    names = {"Priya", "Priya Shah", "Rahul"}
    assert scrub_lookup_query("what Priya Shah said about the Mars mission", names) == "what said about the Mars mission"
    assert scrub_lookup_query("Rahul's favourite band", names) == "favourite band"
    long = scrub_lookup_query("history of " + "the very long topic " * 10, names)
    assert len(long) <= 80 and not long.endswith(" ")


# --- through the brain ---------------------------------------------------------------------------


@pytest.fixture
async def kin(tmp_path):
    created = []

    async def make(llm, client=None):
        store: dict = {}
        client = client or FakeLookupClient()
        h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, None, store, lookup_client=client, shortcut=False))
        await h.app.start()
        await h.app.wait_brain()
        h.store, h.client, h.llm = store, client, llm
        created.append(h)
        await h.app.brain.memory.remember(PRIYA, "commitment", 0.9)
        await asyncio.sleep(0.05)
        await h.app.brain.drain()
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


async def test_chit_chat_is_one_call_and_recall_is_two(kin):
    h = await kin(FakeLLM())
    chat = await ask(h, "How are you?")
    assert chat.llm_calls == 1 and chat.path == "answered directly"
    recall = await ask(h, "When am I meeting Priya?")
    assert recall.llm_calls == 2 and recall.path == "tool: recall" and recall.outcome == "answered"
    assert recall.citations == "1"


async def test_invalid_call_is_retried_once_with_a_correction(kin):
    llm = FakeLLM(turns=[ToolCall("recal", {"question": "When am I meeting Priya?"}),
                         ToolCall("recall", {"question": "When am I meeting Priya?"})])
    h = await kin(llm)
    reply = await ask(h, "When am I meeting Priya?")
    assert reply.path == "retry → tool: recall" and reply.outcome == "answered" and reply.llm_calls == 3
    correction = [m for _, m in llm.calls if m.startswith("That tool call was invalid")]
    assert correction and "unknown tool 'recal'" in correction[0]


async def test_invalid_twice_falls_back_to_recall(kin):
    llm = FakeLLM(turns=[ToolCall("lookup", {"kind": "email", "query": "x"}), ToolCall("lookup", {"query": "x"})])
    h = await kin(llm)
    reply = await ask(h, "When am I meeting Priya?")
    assert reply.path == "retry → fallback: recall"
    assert reply.outcome == "answered" and "4 pm" in reply.text and reply.citations == "1"
    assert not h.client.requests, "an invalid lookup never reaches the gateway"
    # the fallback is a plain recall prompt in a tool-less session
    assert [] in llm.tools_offered


async def test_at_most_two_tool_calls(kin):
    llm = FakeLLM(turns=[
        ToolCall("recall", {"question": "When am I meeting Priya?"}),
        ToolCall("lookup", {"kind": "weather", "query": "", "when": None}),
        ToolCall("lookup", {"kind": "news", "query": "anything"}),  # a third call is ignored
    ])
    h = await kin(llm)
    reply = await ask(h, "When am I meeting Priya and will it rain?")
    assert reply.path == "tool: recall → tool: lookup (weather)"
    assert [k for k, _ in h.client.requests] == ["weather"]
    assert reply.llm_calls == 3


async def test_lookup_query_is_capped_and_scrubbed_before_the_gateway(kin):
    long = "what Priya thinks about " + "the history of the local football club and its stadium " * 3
    llm = FakeLLM(turns=[ToolCall("lookup", {"kind": "search", "query": long}), "It was founded in 1890 [web]."])
    h = await kin(llm)
    await h.app.brain.sync_names()
    reply = await ask(h, "Look that up for me")
    (kind, params), = h.client.requests
    assert kind == "search" and len(params["query"]) <= 80 and len(params["query"].split()) <= 12
    assert "Priya" not in params["query"]
    assert reply.text.startswith("I looked that up online.") and reply.path == "tool: lookup (search)"


async def test_weather_city_is_never_a_person_or_a_phrase(kin):
    llm = FakeLLM(turns=[ToolCall("lookup", {"kind": "weather", "query": "coffee with Priya"}), "Take one [web]."])
    h = await kin(llm)
    await ask(h, "Do I need an umbrella for the coffee with Priya?")
    (kind, params), = h.client.requests
    assert params["city"] == h.app.cfg.user.home_city


async def test_add_task_due_is_parsed_after_the_call(kin):
    llm = FakeLLM(turns=[ToolCall("add_task", {"text": "email the professor", "due": "in 10 minutes"})])
    h = await kin(llm)
    reply = await ask(h, "Remind me to email the professor in ten minutes")
    assert reply.text == "Okay, I'll remind you to email the professor in 10 minutes." and reply.llm_calls == 1
    (task,) = await h.store["tasks"].open()
    assert task.kind == "reminder" and 590 < task.due_ts - time.time() < 610


async def test_remember_tool_stores_explicitly(kin):
    h = await kin(FakeLLM())
    reply = await ask(h, "Remember that Rahul is vegan")
    assert reply.text.startswith("Okay, I'll remember that. That's memory") and reply.path == "tool: remember"
    assert (await h.app.brain.memory.recent())[0].text == "Rahul is vegan."


async def test_complete_task_needs_a_real_task(kin):
    llm = FakeLLM(turns=[ToolCall("complete_task", {"text": "reading the book"})])
    h = await kin(llm)
    await h.store["tasks"].add("Email the professor")
    reply = await ask(h, "I've finished reading the book")
    assert reply.text == "That isn't on your list, so I didn't mark anything as done."
    assert len(await h.store["tasks"].open()) == 1


async def test_queries_use_the_warm_session(kin):
    llm = FakeLLM()
    h = await kin(llm)
    brain = h.app.brain
    for _ in range(50):
        if brain._spare is not None:
            break
        await asyncio.sleep(0.01)
    assert brain._spare is not None, "a session with the prefix already prefilled is waiting"
    before = llm.sessions
    await ask(h, "How are you?")
    await asyncio.sleep(0.05)
    assert llm.sessions == before + 1, "the query used the spare; only the next spare was created"
    systems = {system for system, msg in llm.calls if msg != "Answer only yes or no."}
    assert brain.system in systems
    tool_sessions = [t for t in llm.tools_offered if t]
    assert all(t == tool_sessions[0] for t in tool_sessions)


# --- the priority queue ------------------------------------------------------------------------------


async def test_queue_runs_active_before_background():
    q = LlmQueue()
    order: list[str] = []
    release = asyncio.Event()

    async def background_holder():
        async with q.slot(BACKGROUND):
            order.append("bg-1")
            await release.wait()

    async def job(name, priority):
        async with q.slot(priority):
            order.append(name)

    holder = asyncio.create_task(background_holder())
    await asyncio.sleep(0)
    bg2 = asyncio.create_task(job("bg-2", BACKGROUND))
    await asyncio.sleep(0)
    assert q.idle.is_set()
    active = asyncio.create_task(job("active", ACTIVE))
    await asyncio.sleep(0)
    assert not q.idle.is_set() and q.active_busy
    release.set()
    await asyncio.gather(holder, bg2, active)
    assert order == ["bg-1", "active", "bg-2"]
    assert q.idle.is_set() and not q.active_busy


async def test_queue_cancelled_waiter_does_not_block():
    q = LlmQueue()
    gate = asyncio.Event()

    async def hold():
        async with q.slot(ACTIVE):
            await gate.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(q.slot(BACKGROUND).__aenter__())
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    gate.set()
    await holder
    async with q.slot(ACTIVE):
        pass
