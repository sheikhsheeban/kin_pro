"""Runtime privacy guards.

Rule: raw audio never leaves RAM. Besides simply never writing audio in our
own code, we install a Python audit hook that refuses to open any file with an
audio extension for writing, from any library running in this process.
Native code that bypasses Python's `open` is handled where it is loaded (see
`kin.audio.stt.FORBIDDEN_MOONSHINE_OPTIONS`).
"""

from __future__ import annotations

import os
import sys

AUDIO_EXTENSIONS = frozenset(
    {
        ".wav", ".wave", ".pcm", ".raw", ".flac", ".mp3", ".ogg", ".opus", ".oga",
        ".m4a", ".aac", ".aif", ".aiff", ".au", ".snd", ".webm", ".amr", ".caf",
    }
)

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

_installed = False


class AudioWriteBlocked(PermissionError):
    pass


def is_audio_path(path: object) -> bool:
    if isinstance(path, int):  # file descriptor
        return False
    try:
        name = os.fsdecode(path)  # type: ignore[arg-type]
    except TypeError:
        return False
    return os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS


def _is_write(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _audit(event: str, args: tuple) -> None:
    if event == "open" and len(args) >= 3:
        path, mode, flags = args[0], args[1], args[2]
        if _is_write(mode, flags) and is_audio_path(path):
            raise AudioWriteBlocked(f"Kin privacy guard: refusing to write audio file {path!r}")
    elif event == "os.rename" and len(args) >= 2 and is_audio_path(args[1]):
        raise AudioWriteBlocked(f"Kin privacy guard: refusing to create audio file {args[1]!r}")


def install_audio_write_guard() -> None:
    """Install once per process. Audit hooks cannot be removed, by design."""
    global _installed
    if _installed:
        return
    sys.addaudithook(_audit)
    _installed = True


def guard_installed() -> bool:
    return _installed
