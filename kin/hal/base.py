"""Hardware interface. Every physical control goes through here.

Inputs (mute switch, push-to-talk button, offline switch) are published on
the bus as events. Outputs (the status LED colour) are set via `set_led`.
On the Pi the mute switch also cuts mic power in hardware; software only
reads it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kin.core.bus import EventBus
from kin.core.events import LedColor


class Hardware(ABC):
    name: str = "abstract"

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    @property
    @abstractmethod
    def muted(self) -> bool: ...

    @property
    @abstractmethod
    def offline(self) -> bool: ...

    @property
    @abstractmethod
    def button_pressed(self) -> bool: ...

    @property
    @abstractmethod
    def led(self) -> LedColor: ...

    @abstractmethod
    def set_led(self, color: LedColor) -> None: ...

    async def start(self) -> None:  # noqa: B027 - optional hook
        pass

    async def stop(self) -> None:  # noqa: B027 - optional hook
        pass
