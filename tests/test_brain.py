"""Phase 2: memory capture, grounded recall, abstention, forget, gate, timeout, barge-in, offline."""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from kin.brain.gate import ABSTAIN_REPLY, BLOCKED_REPLY, TIMEOUT_REPLY
from kin.core.events import (
    ActivationEvent,
    AttendingEnded,
    BrainState,
    BrainStateChanged,
    LedColor,
    MemoriesDeleted,
    MemoryStored,
    ReplyCompleted,
    ReplySentence,
    TranscriptEvent,
)
from tests.conftest import FakeLLM, FakeTTS, build_fake_app, make_fake_brain_factory

PRIYA = "I'm meeting Priya at 4 pm at the library cafe."


@pytest.fixture
async def brain_app(tmp_path):
    """Factory for KinApps with fake mic/STT and a brain built from fakes over the real encrypted store."""
    created = []

    async def make(llm: FakeLLM | None = None, tts: FakeTTS | None = None, cfg: dict | None = None):
        store: dict = {}
        h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm, tts, store))
        for section, values in (cfg or {}).items():
            for k, v in values.items():
                setattr(getattr(h.app.cfg, section), k, v)
        await h.app.start()
        await h.app.wait_brain()
        assert h.app.brain_status == "ready"
        h.store = store
        created.append(h)
        return h

    yield make
    for h in created:
        if h.app.bus._loop is not None and not h.app.bus._closed:
            await h.app.stop()


async def say_passive(h, text: str) -> None:
    """Kin overhears a line; the conversation then ends and its facts are extracted (as after 90 s of quiet)."""
    h.bus.publish(TranscriptEvent(line_id=f"p-{time.monotonic()}", text=text, final=True, mode="passive"))
    await asyncio.sleep(0.05)
    await h.app.brain.flush()


async def ask(h, text: str, timeout: float = 5.0) -> ReplyCompleted:
    waiter = asyncio.create_task(h.bus.wait_for(ReplyCompleted, timeout=timeout))
    await asyncio.sleep(0)
    h.bus.publish(ActivationEvent(source="button"))
    h.bus.publish(TranscriptEvent(line_id=f"q-{time.monotonic()}", text=text, final=True, mode="query"))
    h.bus.publish(AttendingEnded(reason="button_released"))
    return await waiter


async def test_recall_happy_path_cites_memory(brain_app):
    h = await brain_app()
    await say_passive(h, PRIYA)
    stored = h.rec.of(MemoryStored)
    assert len(stored) == 1 and stored[0].kind == "plan" and stored[0].source == "extracted"

    reply = await ask(h, "When am I meeting Priya?")
    assert reply.outcome == "answered"
    assert reply.citations == str(stored[0].id)
    assert "Priya" in reply.text and "4 pm" in reply.text
    assert "[#" not in reply.text, "citation markers must not be spoken"
    sentences = h.rec.of(ReplySentence)
    assert sentences and sentences[0].citations == str(stored[0].id)
    assert reply.time_to_first_audio_ms is not None and reply.backend == "fake-llm"


async def test_unknown_fact_abstains_and_shows_closest_match(brain_app):
    llm = FakeLLM()
    h = await brain_app(llm=llm)
    await say_passive(h, PRIYA)
    reply = await ask(h, "What is my sister's birthday?")
    assert reply.outcome == "abstained"
    assert reply.text == ABSTAIN_REPLY
    # nobody identified the speaker, so the first-person line is not attributed to the owner
    assert reply.closest_text == f"Someone said: {PRIYA}" and reply.closest_similarity < 0.5
    # one routing call; recall then finds nothing above the floor and abstains without a second call
    assert llm.queries == ["What is my sister's birthday?"]
    assert reply.llm_calls == 1 and reply.path == "tool: recall"


async def test_empty_memory_abstains(brain_app):
    h = await brain_app()
    reply = await ask(h, "Where did I park the car?")
    assert reply.outcome == "abstained" and reply.closest_id is None


async def test_uncited_answer_is_blocked(brain_app):
    h = await brain_app(llm=FakeLLM(script="You are meeting Priya at 4 pm."))
    await say_passive(h, PRIYA)
    reply = await ask(h, "When am I meeting Priya?")
    assert reply.outcome == "abstained" and reply.text == BLOCKED_REPLY
    assert reply.blocked_sentences == 1


async def test_invented_citation_is_blocked(brain_app):
    h = await brain_app(llm=FakeLLM(script="You are meeting Priya at noon [#999]."))
    await say_passive(h, PRIYA)
    reply = await ask(h, "When am I meeting Priya?")
    assert reply.outcome == "abstained" and "noon" not in reply.text


async def test_model_abstention_is_spoken(brain_app):
    h = await brain_app(llm=FakeLLM(script="I don't remember that."))
    await say_passive(h, PRIYA)
    reply = await ask(h, "When am I meeting Priya?")
    assert reply.outcome == "abstained" and reply.text == "I don't remember that."


async def test_only_worthwhile_utterances_are_stored_once(brain_app):
    h = await brain_app()
    for text in ["Okay, sounds good.", "The weather is so nice outside.", "Rahul is allergic to peanuts.", PRIYA, PRIYA]:
        await say_passive(h, text)
    kinds = sorted(e.kind for e in h.rec.of(MemoryStored))
    assert kinds == ["fact", "plan"]
    assert await h.app.brain.memory.count() == 2


async def test_forget_last_ten_minutes(brain_app):
    h = await brain_app()
    mem = h.app.brain.memory
    old = await mem.remember("Rahul is allergic to peanuts.", "fact_about_person", 0.9, ts=time.time() - 3600)
    await say_passive(h, PRIYA)
    reply = await ask(h, "Forget the last ten minutes.")
    assert reply.text == "That's 1 memory from the last 10 minutes. Say confirm to delete."
    assert reply.llm_calls == 0 and reply.path == "safety guard: forget_span"
    assert await mem.count() == 2 and not h.rec.of(MemoriesDeleted), "nothing is deleted before confirming"
    reply = await ask(h, "Confirm.")
    assert reply.outcome == "command" and reply.text == "Done. I forgot 1 memory from the last 10 minutes."
    deleted = h.rec.of(MemoriesDeleted)
    assert deleted and deleted[-1].count == 1 and deleted[-1].reason == "forget_command"
    remaining = await mem.recent()
    assert [m.id for m in remaining] == [old.id]
    # the forgotten memory is gone from vector search too
    assert all(PRIYA not in hit.fact.text for hit in await mem.retrieve("meeting Priya", 5))
    assert (await ask(h, "When am I meeting Priya?")).outcome == "abstained"


async def test_remember_command_and_forget_that(brain_app):
    h = await brain_app()
    reply = await ask(h, "Remember that Rahul is vegan")
    assert reply.outcome == "command" and "remember" in reply.text.lower()
    assert (await h.app.brain.memory.recent())[0].text == "Rahul is vegan."
    reply = await ask(h, "Forget that")
    assert reply.text == "That's your last memory: Rahul is vegan. Say confirm to delete."
    assert await h.app.brain.memory.count() == 1
    reply = await ask(h, "Yes")
    assert reply.text == "Done, I forgot that." and await h.app.brain.memory.count() == 0


async def test_wake_word_prefix_is_stripped_from_query(brain_app):
    llm = FakeLLM()
    h = await brain_app(llm=llm)
    await say_passive(h, PRIYA)
    reply = await ask(h, "Hey Jarvis, when am I meeting Priya?")
    assert reply.query == "when am I meeting Priya?" and reply.outcome == "answered"


async def test_generation_timeout_stops_and_says_so(brain_app):
    llm = FakeLLM(script="You are meeting Priya at the library cafe at four in the afternoon today [#1].", delay_s=0.2)
    h = await brain_app(llm=llm, cfg={"llm": {"timeout_s": 0.5}})
    await say_passive(h, PRIYA)
    reply = await ask(h, "When am I meeting Priya?")
    assert reply.outcome == "timeout" and reply.text == TIMEOUT_REPLY
    await asyncio.sleep(0.4)
    assert llm.cancelled, "the LLM must actually be cancelled, not left running"


async def test_new_activation_interrupts_reply(brain_app):
    llm = FakeLLM(script=" ".join(["word"] * 40) + " [#1].", delay_s=0.05)
    h = await brain_app(llm=llm)
    await say_passive(h, PRIYA)
    waiter = asyncio.create_task(ask(h, "When am I meeting Priya?"))
    await asyncio.sleep(0.3)
    h.bus.publish(ActivationEvent(source="button"))
    reply = await waiter
    assert reply.outcome == "interrupted"
    await asyncio.sleep(0.2)
    assert llm.cancelled and h.app.brain.state is BrainState.IDLE


async def test_acknowledgement_plays_while_model_is_slow(brain_app):
    tts = FakeTTS()
    llm = FakeLLM(delay_s=0.1)
    h = await brain_app(llm=llm, tts=tts, cfg={"tts": {"ack_delay_s": 0.05}})
    await say_passive(h, PRIYA)
    await ask(h, "When am I meeting Priya?")
    assert tts.said[0] == "Let me check."  # pre-synthesised at start
    played = h.store["sink"].played
    assert len(played) >= 2, "ack and answer both played"


async def test_no_acknowledgement_when_answer_is_fast(brain_app):
    tts = FakeTTS()
    h = await brain_app(tts=tts, cfg={"tts": {"ack_delay_s": 5.0}})
    await say_passive(h, PRIYA)
    reply = await ask(h, "When am I meeting Priya?")
    assert len(h.store["sink"].played) == len(reply.text.split(". "))


async def test_led_amber_while_thinking_and_mic_ducked_while_speaking(brain_app):
    h = await brain_app(llm=FakeLLM(delay_s=0.05), tts=FakeTTS())
    await say_passive(h, PRIYA)
    colors: list[LedColor] = []
    h.bus.subscribe(BrainStateChanged, lambda e: colors.append(h.hw.led))
    await ask(h, "When am I meeting Priya?")
    await asyncio.sleep(0.1)
    states = [e.state for e in h.rec.of(BrainStateChanged)]
    assert states[:3] == [BrainState.THINKING, BrainState.SPEAKING, BrainState.IDLE]
    assert LedColor.AMBER in [e.color for e in h.rec.events if hasattr(e, "color")]
    assert h.hw.led is LedColor.GREEN
    assert h.app.pipeline._ducked is False


async def test_kin_never_memorises_its_own_voice(brain_app):
    h = await brain_app()
    h.app.brain._set_state(BrainState.SPEAKING)
    await say_passive(h, "You told me you are meeting Priya at 4 pm.")
    h.app.brain._set_state(BrainState.IDLE)
    assert not h.rec.of(MemoryStored)


async def test_works_with_networking_disabled(brain_app, monkeypatch):
    def refuse(*_a, **_k):
        raise OSError("network disabled for this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    h = await brain_app()
    await say_passive(h, PRIYA)
    reply = await ask(h, "When am I meeting Priya?")
    assert reply.outcome == "answered"


async def test_memory_store_is_encrypted_on_disk(brain_app):
    from kin.memory.store import MemoryStore, WrongKeyError

    h = await brain_app()
    await say_passive(h, PRIYA)
    db = h.store["db"]
    await h.app.stop()
    raw = db.read_bytes()
    assert b"Priya" not in raw and b"library" not in raw and not raw.startswith(b"SQLite format 3")
    with pytest.raises(WrongKeyError):
        MemoryStore(db, "0" * 64, 256)
    s = MemoryStore(db, h.store["key"], 256)
    assert s.count() == 1
    s.close()
