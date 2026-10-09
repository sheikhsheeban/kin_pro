"""Laptop stand-in for the Pi's switch, button and LED, driven by the web UI."""

from __future__ import annotations

from kin.core.bus import EventBus
from kin.core.events import ButtonChanged, LedChanged, LedColor, MuteChanged, OfflineChanged
from kin.hal.base import Hardware


class SimulatedHardware(Hardware):
    name = "simulated"

    def __init__(self, bus: EventBus, *, muted: bool = False, offline: bool = False) -> None:
        super().__init__(bus)
        self._muted = muted
        self._offline = offline
        self._button = False
        self._led = LedColor.OFF

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def offline(self) -> bool:
        return self._offline

    @property
    def button_pressed(self) -> bool:
        return self._button

    @property
    def led(self) -> LedColor:
        return self._led

    def set_led(self, color: LedColor) -> None:
        if color != self._led:
            self._led = color
            self._bus.publish(LedChanged(color=color))

    # --- inputs, called by the UI (the "physical" controls) ---------------

    def simulate_mute(self, muted: bool) -> None:
        if muted != self._muted:
            self._muted = muted
            self._bus.publish(MuteChanged(muted=muted))

    def simulate_button(self, pressed: bool) -> None:
        if pressed != self._button:
            self._button = pressed
            self._bus.publish(ButtonChanged(pressed=pressed))

    def simulate_offline(self, offline: bool) -> None:
        if offline != self._offline:
            self._offline = offline
            self._bus.publish(OfflineChanged(offline=offline))
