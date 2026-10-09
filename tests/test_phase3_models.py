"""Phase 3 acceptance with the real models and a real gateway *process*.

The gateway runs with deterministic fake providers (rain peaking at 80% at 16:00) so the
umbrella decision is checkable; one extra test uses real Open-Meteo and skips without internet.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os

import pytest

from kin.app import KinApp
from kin.audio import build_engines
from kin.brain import build_brain
from kin.config import load_config
from kin.core.bus import EventBus
from kin.core.events import Announcement, LedChanged, LedColor, LookupPerformed
from tests.conftest import MODELS, FakeInputStream, Recorder

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not (MODELS / "llm" / "gemma-4-E2B-it.litertlm").exists(), reason="models not downloaded"),
]

PRIYA = "I'm meeting Priya at 4 pm at the library cafe for coffee."
UMBRELLA = "Do I need an umbrella for the coffee with Priya I mentioned this morning?"


async def _start(tmp_path, providers: str):
    os.environ["KIN_DB_KEY"] = "ab" * 32
    cfg = load_config("laptop")
    cfg.data_dir = tmp_path
    cfg.tts.sink = "null"
    cfg.tts.ack_delay_s = 30
    cfg.gateway.providers = providers
    bus = EventBus()
    rec = Recorder(bus)
    app = KinApp(cfg, bus=bus, engines_factory=build_engines, stream_factory=lambda **kw: FakeInputStream(**kw),
                 device_namer=lambda _d: "none", brain_factory=build_brain)
    await app.start()
    await app.wait_gateway()
    await app.wait_brain()
    assert app.gateway_info["running"] and app.brain_status == "ready", (app.gateway_info, app.brain_status)
    return app, rec


@pytest.fixture
async def kin_fake_weather(tmp_path):
    app, rec = await _start(tmp_path, "fake")
    yield app, rec
    await app.stop()


async def test_umbrella_demo_end_to_end(kin_fake_weather):
    app, rec = kin_fake_weather
    assert app.gateway.proc.pid != os.getpid(), "the gateway must be its own process"
    await app.brain.memory.remember(PRIYA, "commitment", 0.9)

    reply = await app.brain.handle_query(UMBRELLA)
    print(f"\numbrella: {reply.text!r} ({reply.total_ms} ms, first audio {reply.time_to_first_audio_ms} ms)")
    assert reply.outcome == "answered"
    assert reply.text.startswith("I checked the weather online.")
    assert "umbrella" in reply.text.lower() and ("80%" in reply.text or "rain" in reply.text.lower())
    assert "not need" not in reply.text.lower() and "don't need" not in reply.text.lower()

    look = rec.of(LookupPerformed)[-1]
    assert look.status == "ok" and look.kind == "weather"
    home = app.cfg.user.home_city  # whatever the user configured (e.g. Nagpur)
    assert json.loads(look.request) == {"kind": "weather", "params": {"city": home, "date": datetime.date.today().isoformat()}}
    for word in ("priya", "coffee", "library", "umbrella", "meeting"):
        assert word not in look.request.lower()
    assert LedColor.BLUE in [e.color for e in rec.of(LedChanged)]

    entries = await app.network_log()
    assert entries and entries[-1]["request"] == look.request  # the gateway's own audit agrees


async def test_offline_is_honest_and_still_reasons_locally(kin_fake_weather):
    app, rec = kin_fake_weather
    await app.brain.memory.remember(PRIYA, "commitment", 0.9)
    app.hardware.simulate_offline(True)
    await asyncio.sleep(0.3)

    umbrella = await app.brain.handle_query(UMBRELLA)
    print(f"\noffline umbrella: {umbrella.text!r}")
    assert umbrella.text.startswith("I can't check the weather online right now, because I'm in offline mode.")
    assert rec.of(LookupPerformed)[-1].reason == "offline"
    assert "%" not in umbrella.text  # no invented forecast

    recall = await app.brain.handle_query("When am I meeting Priya?")
    print(f"offline recall: {recall.text!r}")
    assert recall.outcome == "answered" and "4" in recall.text and "Priya" in recall.text


async def test_spoken_reminder_with_real_tts(kin_fake_weather):
    app, rec = kin_fake_weather
    reply = await app.brain.handle_query("Remind me to email the professor in 2 seconds")
    assert reply.text == "Okay, I'll remind you to email the professor in 2 seconds."
    for _ in range(100):
        if rec.of(Announcement):
            break
        await asyncio.sleep(0.1)
    assert rec.of(Announcement)[0].text == "Reminder: Email the professor."
    assert app.brain.speaker.sink.played, "the reminder was synthesised by Piper and played"


async def test_live_open_meteo_through_gateway(tmp_path):
    """Real internet, real Open-Meteo, through the gateway process. Skips when offline."""
    app, rec = await _start(tmp_path, "real")
    try:
        await app.brain.memory.remember(PRIYA, "commitment", 0.9)
        reply = await app.brain.handle_query(UMBRELLA)
        look = rec.of(LookupPerformed)[-1] if rec.of(LookupPerformed) else None
        if look is None or look.status != "ok":
            pytest.skip(f"no internet for the gateway ({look.reason if look else 'no lookup'})")
        print(f"\nlive: {reply.text!r}\n  upstream: {look.upstream}\n  {look.response_bytes} B in {look.duration_ms} ms")
        assert reply.text.startswith("I checked the weather online.")
        assert {u.split()[1].split("/")[0] for u in look.upstream.splitlines()} <= {"geocoding-api.open-meteo.com", "api.open-meteo.com"}
        assert "Priya" not in look.upstream and "coffee" not in look.upstream
    finally:
        await app.stop()
