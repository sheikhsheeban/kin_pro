"""LED derivation. The LED is a pure function of the real stream state and the pipeline state."""

from __future__ import annotations

from kin.core.events import LedColor, PipelineState

_COLORS = {
    PipelineState.IDLE: LedColor.GREEN,  # stream open but pipeline not yet running
    PipelineState.LISTENING: LedColor.GREEN,
    PipelineState.ATTENDING: LedColor.WHITE_PULSE,
    PipelineState.THINKING: LedColor.AMBER,
    PipelineState.SPEAKING: LedColor.WHITE_PULSE,
    PipelineState.LOOKUP: LedColor.BLUE,
}


def derive_led(stream_open: bool, state: PipelineState) -> LedColor:
    """Off if and only if the mic stream is closed. Otherwise the colour shows what Kin is doing."""
    if not stream_open:
        return LedColor.OFF
    return _COLORS[state]
