"""Brain-side client for the Lookup Gateway. The only way the brain reaches outside facts.

Sends `lookup(kind, params)` over the authenticated local pipe/socket. Never sends
transcripts, prompts or memory text; the gateway would reject unknown fields anyway.
Blocking and thread-safe; the brain calls it from a worker thread.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from kin.gateway.transport import MAX_MESSAGE_BYTES, decode, encode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LookupResult:
    ok: bool
    status: str  # ok | refused | error
    reason: str | None = None
    data: dict[str, Any] | None = None
    detail: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)


class GatewayUnavailable(ConnectionError):
    pass


class LookupClient:
    def __init__(self, address: str, family: str, authkey: bytes, timeout_s: float = 10.0) -> None:
        self.address, self.family = address, family
        self._authkey = authkey
        self.timeout_s = timeout_s
        self._conn = None
        self._lock = threading.Lock()

    # --- API ---------------------------------------------------------------------------

    def lookup(self, kind: str, params: dict[str, Any]) -> LookupResult:
        rid = uuid.uuid4().hex[:8]
        try:
            r = self._call({"op": "lookup", "id": rid, "request": {"kind": kind, "params": params}})
        except GatewayUnavailable as e:
            return LookupResult(False, "error", "gateway_unavailable", detail=str(e),
                                audit={"id": rid, "kind": kind, "status": "error", "reason": "gateway_unavailable"})
        return LookupResult(bool(r.get("ok")), r.get("status", "error"), r.get("reason"), r.get("data"),
                            r.get("detail"), r.get("audit") or {})

    def ping(self) -> dict[str, Any]:
        return self._call({"op": "ping"})

    def set_offline(self, offline: bool) -> dict[str, Any]:
        return self._call({"op": "set_offline", "value": offline})

    def set_names(self, salt: bytes, hashes: list[str]) -> dict[str, Any]:
        return self._call({"op": "set_names", "salt": salt.hex(), "hashes": hashes})

    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._call({"op": "audit", "limit": limit}).get("entries", [])

    def close(self) -> None:
        with self._lock:
            self._drop()

    # --- transport -----------------------------------------------------------------------

    def _call(self, msg: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for attempt in (1, 2):  # one reconnect, e.g. after the gateway restarted
                try:
                    conn = self._connect()
                    conn.send_bytes(encode(msg))
                    if not conn.poll(self.timeout_s):
                        self._drop()
                        raise GatewayUnavailable(f"gateway did not answer within {self.timeout_s:.0f}s")
                    return decode(conn.recv_bytes(MAX_MESSAGE_BYTES))
                except GatewayUnavailable:
                    raise
                except (OSError, EOFError) as e:
                    self._drop()
                    if attempt == 2:
                        raise GatewayUnavailable(f"gateway unreachable: {e}") from e
        raise AssertionError("unreachable")

    def _connect(self):
        if self._conn is None:
            from multiprocessing.connection import Client

            from multiprocessing import AuthenticationError

            try:
                self._conn = Client(self.address, family=self.family, authkey=self._authkey)
            except AuthenticationError as e:
                raise GatewayUnavailable("gateway rejected our key") from e
            except (OSError, EOFError) as e:
                raise GatewayUnavailable(f"gateway not running at {self.address}") from e
        return self._conn

    def _drop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
