"""Raspberry Pi GPIO hardware (switch on GPIO17, LED on GPIO27, button on GPIO22).

Implemented in Phase 4 with gpiozero. Kept as a stub so the interface is fixed now.
"""

from __future__ import annotations

from kin.core.bus import EventBus
from kin.core.events import LedColor
from kin.hal.base import Hardware


class PiGpioHardware(Hardware):
    name = "pi_gpio"

    def __init__(self, bus: EventBus) -> None:
        raise NotImplementedError("PiGpioHardware is implemented in Phase 4; use hardware: simulated")

    @property
    def muted(self) -> bool:
        raise NotImplementedError

    @property
    def offline(self) -> bool:
        raise NotImplementedError

    @property
    def button_pressed(self) -> bool:
        raise NotImplementedError

    @property
    def led(self) -> LedColor:
        raise NotImplementedError

    def set_led(self, color: LedColor) -> None:
        raise NotImplementedError
