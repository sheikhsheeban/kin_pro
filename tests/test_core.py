"""Ring buffer, event bus, config, LED derivation and HAL."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
from pydantic import ValidationError

from kin.audio.ring_buffer import RingBuffer
from kin.config import load_config
from kin.core.bus import EventBus
from kin.core.events import (
    ButtonChanged,
    Event,
    LedChanged,
    LedColor,
    MuteChanged,
    OfflineChanged,
    PipelineState,
    SpeechStarted,
)
from kin.core.state import derive_led
from kin.hal import SimulatedHardware, create_hardware

# --- ring buffer ------------------------------------------------------------


def test_ring_buffer_wraps_and_reads_by_absolute_index():
    rb = RingBuffer(seconds=1, sample_rate=10)  # capacity 10
    rb.write(np.arange(7, dtype=np.float32))
    end = rb.write(np.arange(7, 13, dtype=np.float32))
    assert end == 13 and rb.start == 3
    np.testing.assert_array_equal(rb.read(3), np.arange(3, 13))
    np.testing.assert_array_equal(rb.read(0, 5), np.arange(3, 5))  # clamped to what is held
    assert len(rb.read(13)) == 0


def test_ring_buffer_oversized_write_keeps_newest():
    rb = RingBuffer(seconds=1, sample_rate=10)
    rb.write(np.arange(25, dtype=np.float32))
    assert rb.end == 25
    np.testing.assert_array_equal(rb.read(0), np.arange(15, 25))


def test_ring_buffer_zero_and_clear():
    rb = RingBuffer(seconds=1, sample_rate=10)
    rb.write(np.ones(10, dtype=np.float32))
    rb.zero(2, 5)
    np.testing.assert_array_equal(rb.read(0), [1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    rb.clear()
    assert not rb.read(0).any()


def test_ring_buffer_is_thirty_seconds_and_has_no_persistence_api():
    cfg = load_config("laptop")
    rb = RingBuffer(cfg.audio.ring_buffer_seconds, cfg.audio.sample_rate)
    assert rb.capacity == 30 * 16000
    assert not [m for m in dir(rb) if any(w in m.lower() for w in ("save", "dump", "export", "write_to", "pickle"))]
    with pytest.raises(Exception):
        __import__("pickle").dumps(rb)  # holds a lock, so it cannot be serialised by accident


# --- bus ----------------------------------------------------------------------


async def test_bus_typed_delivery_order_and_isolation():
    bus = EventBus()
    got_mute: list[bool] = []
    got_all: list[str] = []

    def boom(_e):
        raise RuntimeError("bad subscriber")

    bus.subscribe(MuteChanged, lambda e: got_mute.append(e.muted))
    bus.subscribe(Event, lambda e: got_all.append(e.type))
    bus.subscribe(Event, boom)
    await bus.start()
    for v in (True, False, True):
        bus.publish(MuteChanged(muted=v))
    bus.publish(SpeechStarted())
    await asyncio.sleep(0.05)
    assert got_mute == [True, False, True]
    assert got_all == ["MuteChanged"] * 3 + ["SpeechStarted"]
    await bus.stop()


async def test_bus_publish_threadsafe_and_wait_for():
    bus = EventBus()
    await bus.start()
    waiter = asyncio.create_task(bus.wait_for(ButtonChanged, lambda e: e.pressed, timeout=2))
    await asyncio.sleep(0)
    threading.Thread(target=lambda: bus.publish_threadsafe(ButtonChanged(pressed=True))).start()
    assert (await waiter).pressed is True
    await bus.stop()


def test_event_serialises_enums():
    d = LedChanged(color=LedColor.WHITE_PULSE).to_dict()
    assert d["type"] == "LedChanged" and d["color"] == "white_pulse" and isinstance(d["ts"], float)


# --- config ---------------------------------------------------------------------


def test_profiles_load_with_expected_limits():
    laptop, pi5 = load_config("laptop"), load_config("pi5")
    assert laptop.profile == "laptop" and laptop.hardware == "simulated"
    assert pi5.runtime.threads == 4 and pi5.llm.context_tokens == 2048
    assert laptop.llm.context_tokens == 4096
    assert pi5.memory.prompt_token_budget < laptop.memory.prompt_token_budget
    assert pi5.memory.dim == laptop.memory.dim == 256
    assert laptop.llm.backend == "litert" and laptop.llm.model.endswith(".litertlm")
    assert pi5.stt.lazy_load_query is True and laptop.stt.lazy_load_query is False
    assert pi5.runtime.ram_target_mb == 3200
    assert laptop.stt.passive.arch == "small-streaming" and laptop.stt.query.arch == "small-streaming"
    assert laptop.ui.host == "127.0.0.1"
    assert laptop.models_dir.is_absolute()


def test_config_rejects_unknown_profile_and_keys():
    with pytest.raises(ValueError):
        load_config("toaster")
    with pytest.raises(ValidationError):
        load_config("laptop", overrides={"cloud_llm": "gpt"})


# --- LED -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stream_open", "state", "color"),
    [
        (False, PipelineState.LISTENING, LedColor.OFF),
        (False, PipelineState.ATTENDING, LedColor.OFF),
        (False, PipelineState.LOOKUP, LedColor.OFF),
        (True, PipelineState.LISTENING, LedColor.GREEN),
        (True, PipelineState.ATTENDING, LedColor.WHITE_PULSE),
        (True, PipelineState.THINKING, LedColor.AMBER),
        (True, PipelineState.LOOKUP, LedColor.BLUE),
    ],
)
def test_led_is_off_iff_stream_closed(stream_open, state, color):
    assert derive_led(stream_open, state) is color


# --- HAL -------------------------------------------------------------------------


async def test_simulated_hardware_publishes_input_changes_once():
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(Event, events.append)
    await bus.start()
    hw = SimulatedHardware(bus)
    hw.simulate_mute(True)
    hw.simulate_mute(True)  # no change, no event
    hw.simulate_button(True)
    hw.simulate_offline(True)
    hw.set_led(LedColor.GREEN)
    await asyncio.sleep(0.05)
    assert [type(e) for e in events] == [MuteChanged, ButtonChanged, OfflineChanged, LedChanged]
    assert hw.muted and hw.button_pressed and hw.offline and hw.led is LedColor.GREEN
    await bus.stop()


def test_pi_gpio_is_a_stub_for_now():
    with pytest.raises(NotImplementedError):
        create_hardware("pi_gpio", EventBus())
    with pytest.raises(ValueError):
        create_hardware("quantum", EventBus())
