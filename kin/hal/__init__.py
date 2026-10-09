"""Hardware abstraction layer. Other modules must only use `Hardware`."""

from __future__ import annotations

from kin.core.bus import EventBus
from kin.hal.base import Hardware
from kin.hal.simulated import SimulatedHardware


def create_hardware(kind: str, bus: EventBus) -> Hardware:
    if kind == "simulated":
        return SimulatedHardware(bus)
    if kind == "pi_gpio":
        from kin.hal.pi_gpio import PiGpioHardware

        return PiGpioHardware(bus)
    raise ValueError(f"unknown hardware kind {kind!r}")


__all__ = ["Hardware", "SimulatedHardware", "create_hardware"]
