"""Streaming speech-to-text with Moonshine v2 (moonshine-voice).

Tiny streams the passive transcript; Small transcribes queries. Each engine
runs one stream at a time: `start` -> `feed`* -> `stop` per utterance.

Only text leaves this module. Moonshine's `TranscriptLine` also carries the
line's audio; we never read or keep it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# The native library can dump its input to a WAV file. That would break the
# "raw audio never on disk" rule, so any such option is refused outright.
FORBIDDEN_MOONSHINE_OPTIONS = frozenset({"save_input_wav_path"})


@dataclass(frozen=True)
class SttLine:
    id: int
    text: str
    final: bool
    latency_ms: int | None = None
    # 'owner' / 'other' only from real speaker identification. Moonshine 0.x offers diarization
    # (anonymous, re-clustered speaker indices) but no enrolment or voice matching, so Kin leaves
    # this None rather than guess who spoke.
    speaker: str | None = None


LineCallback = Callable[[SttLine], None]


class SttEngine(Protocol):
    name: str

    def start(self, on_line: LineCallback) -> None: ...
    def feed(self, samples: np.ndarray) -> None: ...
    def stop(self) -> None:
        """Flush remaining audio; final lines are delivered before returning."""
        ...

    def close(self) -> None: ...


def find_moonshine_model(models_dir: Path, arch: str, language: str = "en") -> Path:
    """Locate a downloaded model folder such as .../tiny-streaming-en/<version>/."""
    candidates = sorted(p.parent for p in Path(models_dir).glob(f"**/{arch}-{language}/*/encoder.ort"))
    if not candidates:
        raise FileNotFoundError(
            f"Moonshine model {arch}-{language} not found under {models_dir}. Run: uv run kin download-models"
        )
    return candidates[-1]


def check_options(options: dict[str, str] | None) -> dict[str, str]:
    options = dict(options or {})
    bad = {k for k in options if k in FORBIDDEN_MOONSHINE_OPTIONS or "wav" in k.lower()}
    if bad:
        raise ValueError(f"Moonshine options {sorted(bad)} would write audio to disk and are not allowed")
    return options


class MoonshineSTT:
    def __init__(
        self,
        model_dir: Path,
        arch: str,
        update_interval_s: float = 0.3,
        options: dict[str, str] | None = None,
    ) -> None:
        from moonshine_voice import Transcriber, string_to_model_arch

        self.name = f"moonshine-{arch}"
        self._transcriber = Transcriber(
            str(model_dir), string_to_model_arch(arch), update_interval_s, options=check_options(options) or None
        )
        self._stream = self._transcriber.create_stream(update_interval_s)
        self._stream.add_listener(self._on_event)
        self._on_line: LineCallback | None = None
        self._running = False

    def start(self, on_line: LineCallback) -> None:
        if self._running:
            self.stop()
        self._on_line = on_line
        self._stream.start()
        self._running = True

    def feed(self, samples: np.ndarray) -> None:
        if self._running and len(samples):
            self._stream.add_audio(np.asarray(samples, dtype=np.float32).tolist(), 16000)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._stream.stop()
        finally:
            self._on_line = None

    def close(self) -> None:
        self.stop()
        self._stream.close()
        self._transcriber.close()

    def _on_event(self, event: object) -> None:
        from moonshine_voice.transcriber import LineCompleted, LineTextChanged

        cb = self._on_line
        if cb is None or not isinstance(event, (LineTextChanged, LineCompleted)):
            return
        line = event.line
        text = (line.text or "").strip()
        if not text:
            return
        latency = getattr(line, "last_transcription_latency_ms", None)
        cb(SttLine(id=int(line.line_id), text=text, final=isinstance(event, LineCompleted), latency_ms=latency))
