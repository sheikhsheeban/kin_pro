"""Fixes from trying Kin live: the wake phrase found in the transcript, facts read right after a question,
no "I don't remember ..." after an answer from memory."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from kin.audio.pipeline import Engines, PerceptionPipeline
from kin.audio.ring_buffer import RingBuffer
from kin.audio.stt import SttLine
from kin.audio.wake import spoken_wake
from kin.config import load_config
from kin.core.events import ActivationEvent, AttendingEnded, TranscriptEvent
from tests.conftest import FakeLLM, FakeVAD, build_fake_app, make_fake_brain_factory, silence, speech
from tests.test_wake import RecordingWake


class ScriptedSTT:
    """Says the next scripted line when stopped (the end of an utterance)."""

    def __init__(self, name: str, lines: list[str]) -> None:
        self.name, self.lines, self._cb = name, lines, None

    def start(self, on_line) -> None:
        self._cb = on_line

    def feed(self, samples) -> None:
        pass

    def stop(self) -> None:
        if self._cb and self.lines:
            self._cb(SttLine(id=1, text=self.lines.pop(0), final=True))
        self._cb = None

    def close(self) -> None:
        pass


class Run:
    def __init__(self, passive: list[str], query: list[str]) -> None:
        self.cfg = load_config("laptop")
        self.cfg.vad.min_silence_ms = 96
        self.wake = RecordingWake(score=0.2)  # this voice never reaches the acoustic threshold (0.5)
        self.events: list = []
        self._lock = threading.Lock()
        q = ScriptedSTT("small", query)
        self.p = PerceptionPipeline(self.cfg, Engines(FakeVAD(), self.wake, ScriptedSTT("tiny", passive), lambda: q),
                                    RingBuffer(30, 16000), self._emit)
        self.p.start()
        self.p.stream_opened()

    def _emit(self, e) -> None:
        with self._lock:
            self.events.append(e)

    def say(self) -> None:
        frames = speech(6) + silence(6)
        target = self.wake.frames + len(frames)
        for f in frames:
            self.p.submit_frame(f)
        deadline = time.monotonic() + 5
        while self.wake.frames < target and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.05)

    def of(self, cls) -> list:
        with self._lock:
            return [e for e in self.events if isinstance(e, cls)]


@pytest.fixture
def run():
    made: list[Run] = []

    def make(passive, query=()):
        made.append(Run(list(passive), list(query)))
        return made[-1]

    yield make
    for r in made:
        r.p.stop()


def test_spoken_wake_phrase_matching():
    for heard, rest in [("Hey Jarvis", ""), ("A Jarvis.", ""), ("Hey Jarvis!", ""), ("hello jarvis", ""),
                        ("Hey, Jarvis, who is my manager?", "who is my manager?"), ("Jarvis what time is it", "what time is it"),
                        ("Hey Arviz, when is the dentist?", "when is the dentist?")]:
        assert spoken_wake(heard, "Hey Jarvis") == rest, heard
    for other in ("I told Jarvis about it", "Hey Travis", "Hey my manager is very good in coding", "Service was slow", ""):
        assert spoken_wake(other, "Hey Jarvis") is None, other


def test_wake_phrase_with_the_question_in_one_line_is_asked(run):
    r = run(["Hey Jarvis, who is my manager?"])
    r.say()
    assert [a.source for a in r.of(ActivationEvent)] == ["wake_phrase"]
    lines = [(t.mode, t.text) for t in r.of(TranscriptEvent) if t.final]
    assert lines == [("query", "who is my manager?")], "asked, and not stored as overheard speech"
    assert [e.reason for e in r.of(AttendingEnded)] == ["query_complete"]


def test_wake_phrase_alone_listens_for_the_question(run):
    r = run(["A Jarvis."], ["Who is my manager?"])
    r.say()
    assert [a.source for a in r.of(ActivationEvent)] == ["wake_phrase"]
    assert not r.of(AttendingEnded), "attending: waiting for the question"
    r.say()
    assert [(t.mode, t.text) for t in r.of(TranscriptEvent) if t.final] == [("query", "Who is my manager?")]
    assert [e.reason for e in r.of(AttendingEnded)] == ["query_complete"]


def test_ordinary_lines_do_not_wake(run):
    r = run(["I told Jarvis about it.", "Hey my manager is very good in coding."])
    r.say()
    r.say()
    assert not r.of(ActivationEvent)
    assert [t.mode for t in r.of(TranscriptEvent) if t.final] == ["passive", "passive"]


async def test_a_question_ends_the_conversation_and_its_facts_are_read_after_the_answer(tmp_path):
    llm = FakeLLM()
    llm.extract_replies = [json.dumps([
        {"text": "The user has a meeting at 10 pm right after dinner.", "kind": "plan", "subject": None, "people": [],
         "event_time": None, "lines": [1]},
    ])]
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, llm=llm))
    await h.app.start()
    await h.app.wait_brain()
    try:
        brain = h.app.brain
        await brain._ingest("I have a meeting at 10 pm right after the dinner.", ts=time.time() - 5)
        assert (await brain.memory.episode_counts()) == {"open": 1}
        h.app.bus.publish(ActivationEvent(source="button"))
        h.app.bus.publish(TranscriptEvent(line_id="2-1", text="What do I have tonight?", final=True, mode="query"))
        h.app.bus.publish(AttendingEnded(reason="query_complete"))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not await brain.memory.recent():
            await asyncio.sleep(0.05)
        facts = [f.text for f in await brain.memory.recent()]  # "Someone has ...": the speaker is never guessed
        assert len(facts) == 1 and facts[0].endswith("has a meeting at 10 pm right after dinner."), \
            "read right after the question, not after 90 s of silence"
    finally:
        await h.app.stop()


async def test_no_i_dont_remember_after_an_answer_from_memory(tmp_path):
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path))
    await h.app.start()
    await h.app.wait_brain()
    try:
        brain = h.app.brain
        a = await brain.memory.remember("My manager name is Ajay.", "fact_about_person", 0.9)
        b = await brain.memory.remember("Parashant is my second level manager.", "fact_about_person", 0.9)
        brain.llm.script = (f"Your manager is Ajay [#{a.id}]. Parashant is your second level manager [#{b.id}]. "
                            "I don't remember who your manager is on Friday.")
        reply = await brain.handle_query("Who is my manager?")
        assert reply.outcome == "answered"
        assert reply.text == "Your manager is Ajay. Parashant is your second level manager."
    finally:
        await h.app.stop()
