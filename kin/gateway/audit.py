"""Append-only audit log of every lookup the gateway handled (accepted or refused).

Each line records exactly what was asked (the canonical request JSON), what
went out to the internet (method, host, path, query), how much came back and
how long it took. Nothing else: the gateway never sees transcripts or memories.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

KEEP_LINES = 5000


class AuditLog:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self._recent: deque[dict[str, Any]] = deque(maxlen=500)
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                lines = self.path.read_text(encoding="utf-8").splitlines()[-KEEP_LINES:]
                self.path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
                for ln in lines[-self._recent.maxlen :]:
                    try:
                        self._recent.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue

    def append(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._recent.append(entry)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def tail(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent)[-limit:]
