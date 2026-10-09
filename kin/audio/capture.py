"""Microphone capture (16 kHz mono float32) via sounddevice.

Mute closes the input stream itself; it does not just drop frames. While
muted there is no open stream, so there is nothing to read from. `is_open`
reports the real stream state and is what the LED is derived from.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from kin.config import AudioConfig

log = logging.getLogger(__name__)


class InputStreamLike(Protocol):
    @property
    def active(self) -> bool: ...
    @property
    def closed(self) -> bool: ...
    def start(self) -> None: ...
    def abort(self) -> None: ...
    def close(self) -> None: ...


StreamFactory = Callable[..., InputStreamLike]
FrameSink = Callable[[np.ndarray], None]


def sounddevice_stream_factory(*, samplerate: int, blocksize: int, device: Any, callback: Callable) -> InputStreamLike:
    import sounddevice as sd

    return sd.InputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        device=device,
        channels=1,
        dtype="float32",
        callback=callback,
    )


def describe_device(device: Any) -> str:
    try:
        import sounddevice as sd

        return str(sd.query_devices(device, "input")["name"])
    except Exception:
        return "unknown input device"


class MicCapture:
    def __init__(
        self,
        cfg: AudioConfig,
        on_frame: FrameSink,
        stream_factory: StreamFactory = sounddevice_stream_factory,
        device_namer: Callable[[Any], str] = describe_device,
    ) -> None:
        self._cfg = cfg
        self._on_frame = on_frame
        self._factory = stream_factory
        self._namer = device_namer
        self._stream: InputStreamLike | None = None
        self._accepting = False
        self.device_name: str | None = None
        self.overflows = 0

    @property
    def is_open(self) -> bool:
        s = self._stream
        return s is not None and not s.closed and s.active

    def open(self) -> None:
        if self.is_open:
            return
        self.close()
        stream = self._factory(
            samplerate=self._cfg.sample_rate,
            blocksize=self._cfg.frame_samples,
            device=self._cfg.device,
            callback=self._callback,
        )
        self._accepting = True
        try:
            stream.start()
        except Exception:
            self._accepting = False
            stream.close()
            raise
        self._stream = stream
        self.device_name = self._namer(self._cfg.device)
        log.info("mic stream open: %s @ %d Hz", self.device_name, self._cfg.sample_rate)

    def close(self) -> None:
        """Stop delivering frames immediately and release the device."""
        self._accepting = False
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort()  # discard pending buffers; returns as soon as the callback stops
        finally:
            stream.close()
        log.info("mic stream closed")

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if not self._accepting:
            return
        if status and getattr(status, "input_overflow", False):
            self.overflows += 1
        self._on_frame(indata[:, 0].copy())
