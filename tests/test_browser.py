"""Browser end-to-end tests of the UI with Playwright (headless Chromium), against a real uvicorn server running
the fake app (fake mic stream and models, real encrypted store).

  uv run pytest tests/test_browser.py          # needs Chromium: `uv run playwright install chromium`
                                               # (falls back to an installed Edge or Chrome)
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page, expect, sync_playwright  # noqa: E402

pytestmark = pytest.mark.browser

GOOD = "lantern river 42 pebbles"


class Server:
    """The fake Kin app behind uvicorn, in its own thread and event loop."""

    def __init__(self, tmp: Path) -> None:
        import uvicorn

        from kin.ui.server import create_app
        from tests.conftest import FakeLLM, build_fake_app, make_fake_brain_factory

        self.h = build_fake_app(brain_factory=make_fake_brain_factory(tmp, llm=FakeLLM()))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(create_app(self.h.app), host="127.0.0.1", port=self.port,
                                                    log_level="warning", ws_ping_interval=None))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=lambda: self.loop.run_until_complete(self.server.serve()), daemon=True)
        self.thread.start()
        deadline = time.time() + 30
        while not self.server.started:
            assert time.time() < deadline, "the UI server did not start"
            time.sleep(0.05)
        self.call(self.h.app.wait_brain())

    def call(self, coro, timeout: float = 20):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def publish(self, event) -> None:
        self.loop.call_soon_threadsafe(self.h.app.bus.publish, event)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=30)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = None
        for kw in ({}, {"channel": "chromium"}, {"channel": "msedge"}, {"channel": "chrome"}):
            try:
                b = p.chromium.launch(headless=True, **kw)
                break
            except Exception:
                continue
        if b is None:
            pytest.skip("no Chromium: run `uv run playwright install chromium`")
        yield b
        b.close()


@pytest.fixture
def kin(tmp_path):
    srv = Server(tmp_path)
    yield srv
    srv.stop()


@pytest.fixture
def page(browser, kin) -> Page:
    ctx = browser.new_context()
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(kin.url)
    expect(pg.locator("#conn")).not_to_have_text("connecting…", timeout=10_000)
    yield pg
    ctx.close()
    assert not errors, f"JavaScript errors on the page: {errors}"


def _memory(kin, text: str, kind: str = "fact"):
    return kin.call(kin.h.app.brain.memory.remember(text, kind, 0.9))


def _row(page: Page, fid: int):
    return page.locator(f'[data-mem="{fid}"]')


# --- device panel ------------------------------------------------------------------------------------------


def test_mute_closes_the_stream_and_the_led_follows(page, kin):
    expect(page.locator("#led")).to_have_class("led green")
    expect(page.locator("#streamText")).to_contain_text("open")
    page.locator("#mute + span").click()
    expect(page.locator("#led")).to_have_class("led off", timeout=5000)
    expect(page.locator("#streamText")).to_contain_text("closed")
    assert kin.h.app.snapshot()["stream_open"] is False, "the real stream is closed, not ignored"
    page.locator("#mute + span").click()
    expect(page.locator("#led")).to_have_class("led green", timeout=5000)
    assert kin.h.app.snapshot()["stream_open"] is True


def test_hold_to_talk_attends_while_held(page, kin):
    from kin.core.events import ActivationEvent

    btn = page.locator("#ptt")
    btn.hover()
    page.mouse.down()
    expect(btn).to_have_text("Listening… release to finish")
    expect(page.locator("#led")).to_have_class("led white_pulse", timeout=5000)
    page.mouse.up()
    expect(btn).to_have_text("Hold to talk")
    expect(page.locator("#led")).to_have_class("led green", timeout=10_000)
    assert [e.source for e in kin.h.rec.of(ActivationEvent)] == ["button"]


def test_offline_toggle_and_network_tab(page, kin):
    from kin.core.events import LookupPerformed

    page.locator('[data-tab="network"]').click()
    expect(page.locator("#netStatus")).to_contain_text("Offline mode")
    expect(page.locator("#netCount")).to_contain_text("the brain process cannot open a socket")
    page.locator("#offline + span").click()
    deadline = time.time() + 5
    while not kin.h.app.hardware.offline and time.time() < deadline:
        time.sleep(0.05)
    assert kin.h.app.hardware.offline, "the offline switch reached the device"
    kin.publish(LookupPerformed(request_id="r1", kind="weather", request=json.dumps({"kind": "weather", "params": {"city": "Nagpur"}}),
                                upstream="GET https://api.open-meteo.com/v1/forecast?city=Nagpur", status="refused", reason="offline"))
    expect(page.locator("#netList")).to_contain_text("Nagpur")
    expect(page.locator("#netList")).to_contain_text("offline")
    page.locator("#offline + span").click()
    deadline = time.time() + 5
    while kin.h.app.hardware.offline and time.time() < deadline:
        time.sleep(0.05)
    assert not kin.h.app.hardware.offline


# --- memory -------------------------------------------------------------------------------------------------


def test_memory_edit_retract_restore_pin_and_merge(page, kin):
    a = _memory(kin, "Rahul is allergic to peanuts.")
    b = _memory(kin, "Rahul can't eat peanuts.")
    expect(_row(page, a.id)).to_be_visible()

    page.once("dialog", lambda d: d.accept("Rahul is allergic to peanuts and cashews."))
    _row(page, a.id).locator("[data-edit]").click()
    expect(page.locator("#memList")).to_contain_text("Rahul is allergic to peanuts and cashews.")
    new = next(f for f in kin.call(kin.h.app.brain.memory.recent()) if "cashews" in f.text)
    assert kin.call(kin.h.app.brain.memory.get(a.id)).status == "superseded", "the old version is kept"

    _row(page, new.id).locator("[data-pin]").click()
    expect(_row(page, new.id)).to_have_class("mem pinned")
    _row(page, new.id).locator("[data-pin]").click()
    expect(_row(page, new.id)).to_have_class("mem")

    _row(page, b.id).locator("[data-retract]").click()
    expect(_row(page, b.id)).to_have_count(0)  # out of the active list
    page.locator("#memHistory").check()
    expect(_row(page, b.id)).to_have_class("mem retracted")
    _row(page, b.id).locator("[data-restore]").click()
    expect(_row(page, b.id)).to_have_class("mem")
    page.locator("#memHistory").uncheck()

    answers = iter([str(b.id), "Rahul is allergic to peanuts and cashews; no satay."])
    page.on("dialog", lambda d: d.accept(next(answers)))
    _row(page, new.id).locator("[data-merge]").click()
    expect(page.locator("#memList")).to_contain_text("no satay")
    active = [f.text for f in kin.call(kin.h.app.brain.memory.recent())]
    assert active == ["Rahul is allergic to peanuts and cashews; no satay."]


def test_delete_needs_confirmation(page, kin):
    f = _memory(kin, "The door code is 4512.")
    expect(_row(page, f.id)).to_be_visible()
    page.once("dialog", lambda d: d.dismiss())
    _row(page, f.id).locator("[data-del]").click()
    time.sleep(0.5)
    assert kin.call(kin.h.app.brain.memory.get(f.id)) is not None, "cancel keeps the memory"
    page.once("dialog", lambda d: (d.message.startswith(f"Delete memory #{f.id} for good") or None, d.accept()))
    _row(page, f.id).locator("[data-del]").click()
    expect(_row(page, f.id)).to_have_count(0)
    assert kin.call(kin.h.app.brain.memory.get(f.id)) is None


def test_daily_review_card_lists_today_and_jumps_to_the_memory(page, kin):
    f = _memory(kin, "Neha moved to Pune.")
    _memory(kin, "The plumber comes on Monday.")
    card = page.locator("#reviewCard")
    expect(card).to_contain_text("2 new")
    expect(card).to_contain_text("Neha moved to Pune.")
    card.locator(f'[data-goto="{f.id}"]').click()
    expect(_row(page, f.id)).to_have_class("mem flash")


# --- backup -------------------------------------------------------------------------------------------------


def test_backup_dialog_checks_the_passphrase_and_the_restore_drill(page, kin, tmp_path):
    _memory(kin, "The wifi password is tiger42.")
    page.locator('[data-tab="system"]').click()
    page.locator("#backupBtn").click()
    dlg = page.locator("#backupDlg")
    expect(dlg).to_be_visible()
    page.locator("#bkFolder").fill(str(tmp_path))
    page.locator("#bkPass").fill("password1")
    expect(page.locator("#bkMsg")).to_contain_text("at least")
    expect(page.locator("#bkSave")).to_be_disabled()
    page.locator("#bkPass").fill(GOOD)
    page.locator("#bkPass2").fill(GOOD + "x")
    expect(page.locator("#bkMsg")).to_contain_text("differ")
    expect(page.locator("#bkSave")).to_be_disabled()
    page.locator("#bkPass2").fill(GOOD)
    expect(page.locator("#bkSave")).to_be_enabled()
    page.locator("#bkSave").click()
    expect(page.locator("#bkMsg")).to_contain_text("Saved")
    files = list(tmp_path.glob("kin-backup-*.kinbak"))
    assert len(files) == 1 and b"tiger42" not in files[0].read_bytes()
    page.keyboard.press("Escape")

    page.locator("#checkBtn").click()
    page.locator("#ckFile").fill(str(files[0]))
    page.locator("#ckPass").fill(GOOD + "!")
    page.locator("#ckRun").click()
    expect(page.locator("#ckMsg")).to_contain_text("wrong passphrase")
    page.locator("#ckPass").fill(GOOD)
    page.locator("#ckRun").click()
    expect(page.locator("#ckMsg")).to_contain_text("Restorable")
    expect(page.locator("#ckMsg")).to_contain_text("1 facts")
