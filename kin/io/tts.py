"""Text-to-speech with Piper, streamed sentence by sentence.

Synthesis and playback run on separate worker threads, so sentence N+1 is
synthesised while sentence N plays. Generated speech is played straight from
memory and never written to disk.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from kin.core.workers import ModelWorker

log = logging.getLogger(__name__)


class TextToSpeech(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Mono float32 audio for `text`."""
        ...


class AudioSink(Protocol):
    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Blocking until played or stopped."""
        ...

    def stop(self) -> None:
        """Called from another thread: abort playback immediately."""
        ...


class PiperTTS:
    def __init__(self, voice_path: Path) -> None:
        from piper import PiperVoice

        if not Path(voice_path).exists():
            raise FileNotFoundError(f"Piper voice not found at {voice_path}. Run: uv run kin download-models")
        self._voice = PiperVoice.load(str(voice_path))
        self.sample_rate = int(self._voice.config.sample_rate)
        self.name = f"piper:{Path(voice_path).stem}"

    def synthesize(self, text: str) -> np.ndarray:
        parts = [c.audio_float_array for c in self._voice.synthesize(text)]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


class SoundDeviceSink:
    def __init__(self, device: str | int | None = None) -> None:
        self._device = device
        self._stream = None
        self._stopped = False

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        self._stopped = False
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", device=self._device)
        self._stream = stream
        stream.start()
        try:
            block = sample_rate // 10
            for i in range(0, len(audio), block):
                if self._stopped:
                    break
                stream.write(audio[i : i + block].reshape(-1, 1))
        finally:
            self._stream = None
            stream.abort() if self._stopped else stream.stop()
            stream.close()

    def stop(self) -> None:
        self._stopped = True


class NullSink:
    """Headless sink: 'plays' instantly and records what was said (tests, no speaker)."""

    def __init__(self) -> None:
        self.played: list[float] = []

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        self.played.append(len(audio) / sample_rate)

    def stop(self) -> None:
        pass


Kind = Literal["ack", "answer"]


class Turn:
    """One spoken reply: an ordered queue of sentences, cancellable as a whole."""

    def __init__(self, speaker: Speaker, on_audio: Callable[[], None] | None = None) -> None:
        self._sp = speaker
        self._on_audio = on_audio
        self._queue: asyncio.Queue[tuple[Kind, np.ndarray] | None] = asyncio.Queue()
        self._player = asyncio.create_task(self._play_loop())
        self.cancelled = False
        self.first_audio_at: float | None = None  # monotonic time the first answer audio started
        self.spoken_seconds = 0.0

    async def say(self, text: str, kind: Kind = "answer") -> None:
        if self.cancelled or not text.strip():
            return
        audio = await self._sp.synthesize(text)
        if not self.cancelled:
            self._queue.put_nowait((kind, audio))

    def play_cached(self, audio: np.ndarray, kind: Kind = "ack") -> None:
        if not self.cancelled:
            self._queue.put_nowait((kind, audio))

    async def finish(self) -> None:
        self._queue.put_nowait(None)
        await self._player

    def cancel(self) -> None:
        self.cancelled = True
        self._sp.sink.stop()
        self._queue.put_nowait(None)

    @property
    def answer_started(self) -> bool:
        return self.first_audio_at is not None

    async def _play_loop(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None or self.cancelled:
                return
            kind, audio = item
            if self._on_audio is not None:
                self._on_audio()  # every clip: keeps the mic ducked whenever Kin is audible
            if kind == "answer" and self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
            await self._sp.play(audio)
            self.spoken_seconds += len(audio) / self._sp.sample_rate


class Speaker:
    def __init__(self, tts: TextToSpeech | None, sink: AudioSink) -> None:
        self.tts = tts
        self.sink = sink
        self.sample_rate = tts.sample_rate if tts else 22050
        self._synth = ModelWorker("tts")
        self._out = ModelWorker("audio-out")
        self._cache: dict[str, np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return self.tts is not None

    async def synthesize(self, text: str) -> np.ndarray:
        if self.tts is None:
            return np.zeros(int(self.sample_rate * 0.05 * max(1, len(text.split()))), np.float32)
        return await self._synth.run(self.tts.synthesize, text)

    async def precache(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = await self.synthesize(text)
        return self._cache[text]

    def cached(self, text: str) -> np.ndarray | None:
        return self._cache.get(text)

    async def play(self, audio: np.ndarray) -> None:
        await self._out.run(self.sink.play, audio, self.sample_rate)

    def turn(self, on_audio: Callable[[], None] | None = None) -> Turn:
        return Turn(self, on_audio)

    def close(self) -> None:
        self.sink.stop()
        self._synth.shutdown()
        self._out.shutdown()
