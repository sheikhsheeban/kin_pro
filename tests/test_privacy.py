"""Privacy rules: raw audio never touches disk; only text and scalars leave the audio layer."""

from __future__ import annotations

import os
import tempfile
import typing
import wave
from enum import Enum
from pathlib import Path

import pytest

import kin.core.events as ev
from kin.audio.stt import check_options
from kin.core.privacy import AudioWriteBlocked, install_audio_write_guard
from tests.conftest import ROOT, silence, speech, wake_frames

AUDIO_MAGIC = (b"RIFF", b"fLaC", b"OggS", b"ID3", b"FORM", b".snd", b"\x1aE\xdf\xa3")
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache"}


def watched_dirs() -> list[Path]:
    dirs = [ROOT, Path(tempfile.gettempdir())]
    try:
        from moonshine_voice.utils import get_cache_dir  # type: ignore[attr-defined]

        dirs.append(Path(get_cache_dir()))
    except Exception:
        pass
    return [d for d in dirs if d.exists()]


def snapshot_files() -> dict[Path, float]:
    out: dict[Path, float] = {}
    for base in watched_dirs():
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                p = Path(dirpath, name)
                try:
                    out[p] = p.stat().st_mtime
                except OSError:
                    pass
    return out


def audio_files_written(before: dict[Path, float]) -> list[Path]:
    """New or modified files that look like audio, by extension or by header bytes."""
    from kin.core.privacy import AUDIO_EXTENSIONS

    bad = []
    for p, mtime in snapshot_files().items():
        if before.get(p) == mtime:
            continue
        if p.suffix.lower() in AUDIO_EXTENSIONS | {".npy", ".npz"}:
            bad.append(p)
            continue
        try:
            with open(p, "rb") as f:
                head = f.read(4)
        except OSError:
            continue
        if head.startswith(AUDIO_MAGIC):
            bad.append(p)
    return bad


def test_audio_write_guard_blocks_audio_files(tmp_path):
    install_audio_write_guard()
    for name in ("clip.wav", "clip.PCM", "x.raw", "x.flac", "x.mp3"):
        with pytest.raises(AudioWriteBlocked):
            open(tmp_path / name, "wb")
    with pytest.raises(AudioWriteBlocked):
        wave.open(str(tmp_path / "w.wav"), "wb")
    with pytest.raises(AudioWriteBlocked):
        os.open(tmp_path / "fd.wav", os.O_WRONLY | os.O_CREAT)
    (tmp_path / "notes.txt").write_text("text is fine")
    (tmp_path / "x.tmp").write_bytes(b"x")
    with pytest.raises(AudioWriteBlocked):
        os.rename(tmp_path / "x.tmp", tmp_path / "x.wav")
    assert not list(tmp_path.glob("*.wav"))


def test_moonshine_audio_dump_option_is_refused():
    with pytest.raises(ValueError):
        check_options({"save_input_wav_path": "debug"})
    with pytest.raises(ValueError):
        check_options({"some_wav_option": "x"})
    assert check_options({"identify_speakers": "false"}) == {"identify_speakers": "false"}


def _allowed(tp) -> bool:
    origin = typing.get_origin(tp)
    if origin is typing.Literal:
        return True
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        return all(_allowed(a) for a in typing.get_args(tp))
    return tp in (str, int, float, bool, type(None)) or (isinstance(tp, type) and issubclass(tp, Enum))


def test_events_carry_no_audio():
    """Every bus event field is a scalar, string or enum. No arrays, bytes or lists."""
    event_classes = [c for c in vars(ev).values() if isinstance(c, type) and issubclass(c, ev.Event)]
    assert len(event_classes) > 10
    for cls in event_classes:
        for name, tp in typing.get_type_hints(cls).items():
            assert _allowed(tp), f"{cls.__name__}.{name}: {tp} could carry audio"


async def test_no_audio_files_written_during_a_run(harness):
    before = snapshot_files()
    harness.stream.push(silence(10) + speech(30) + silence(5) + wake_frames(4) + speech(10) + silence(10))
    harness.hw.simulate_button(True)
    harness.stream.push(speech(10))
    harness.hw.simulate_button(False)
    harness.hw.simulate_mute(True)
    harness.hw.simulate_mute(False)
    await harness.settle()
    assert harness.rec.events, "pipeline did not run"
    assert audio_files_written(before) == []


def test_no_audio_files_in_project():
    """Nothing audio-like is committed or left around anywhere in the repo (models excluded)."""
    from kin.core.privacy import AUDIO_EXTENSIONS

    found = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and not (set(p.relative_to(ROOT).parts) & (SKIP_DIRS | {"models"}))
    ]
    assert found == []
