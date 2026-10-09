"""Local transport between the brain and the gateway.

`multiprocessing.connection` gives one API over a Windows named pipe and a Unix
socket, with HMAC challenge-response authentication: a process without the
shared key cannot talk to the gateway. Messages are JSON bytes only; we never
use `send()`/`recv()`, which would unpickle untrusted data.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

MAX_MESSAGE_BYTES = 64 * 1024  # replies can carry an audit page; requests are capped far lower


def gateway_address(data_dir: Path) -> tuple[str, str]:
    """(address, family). Unique per data directory, so two Kin installs never collide."""
    tag = hashlib.sha256(str(Path(data_dir).resolve()).encode()).hexdigest()[:12]
    if sys.platform == "win32":
        return rf"\\.\pipe\kin-gateway-{tag}", "AF_PIPE"
    run = Path(data_dir) / "run"
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    return str(run / "gateway.sock"), "AF_UNIX"


def load_or_create_secret(path: Path) -> bytes:
    """32-byte key shared by brain and gateway (env KIN_GATEWAY_KEY overrides, hex)."""
    env = os.environ.get("KIN_GATEWAY_KEY")
    if env:
        return bytes.fromhex(env)
    path = Path(path)
    if path.exists():
        return bytes.fromhex(path.read_text(encoding="ascii").strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # created concurrently by the other process
        return bytes.fromhex(path.read_text(encoding="ascii").strip())
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(key)
    return bytes.fromhex(key)


def encode(msg: dict[str, Any]) -> bytes:
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decode(data: bytes) -> dict[str, Any]:
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("message must be a JSON object")
    return obj
