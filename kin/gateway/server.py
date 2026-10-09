"""The Lookup Gateway: a separate process and the only Kin code allowed on the network.

Protocol (JSON over an authenticated local pipe/socket, one request -> one reply):
  {"op": "ping"}
  {"op": "lookup", "id": "...", "request": {"kind": "weather|search|news", "params": {...}}}
  {"op": "set_offline", "value": true|false}           simulated Wi-Fi switch
  {"op": "set_names", "salt": "<hex>", "hashes": [...]} hashed person names to scrub
  {"op": "audit", "limit": 100}                         recent audit entries

Every lookup, accepted or refused, is written to the audit log.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from multiprocessing import AuthenticationError
from typing import Any

from pydantic import ValidationError

from kin.gateway import schemas
from kin.gateway.audit import AuditLog
from kin.gateway.http import Exchange, HostNotAllowed, HttpClient, UpstreamError
from kin.gateway.providers import Provider
from kin.gateway.scrub import NameScrubber
from kin.gateway.transport import MAX_MESSAGE_BYTES, decode, encode

log = logging.getLogger("kin.gateway")

MAX_LOOKUPS_PER_MINUTE = 30


class GatewayService:
    """Transport-independent core: validates, scrubs, rate-limits, calls providers, audits."""

    def __init__(self, providers: dict[str, Provider], audit: AuditLog, *, timeout_s: float = 6.0) -> None:
        self.providers = providers
        self.audit = audit
        self.scrubber = NameScrubber()
        self.offline = False
        hosts = set().union(*(p.hosts for p in providers.values())) if providers else set()
        self.http = HttpClient(hosts, timeout_s)
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return self.http.allowed_hosts

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        op = msg.get("op")
        if op == "ping":
            return {"ok": True, "offline": self.offline, "providers": {k: p.name for k, p in self.providers.items()},
                    "hosts": sorted(self.allowed_hosts), "names": self.scrubber.size}
        if op == "lookup":
            return self._lookup(msg)
        if op == "set_offline":
            self.offline = bool(msg.get("value"))
            log.info("offline mode %s", "on" if self.offline else "off")
            return {"ok": True, "offline": self.offline}
        if op == "set_names":
            hashes = [h for h in msg.get("hashes", []) if isinstance(h, str) and len(h) == 24]
            self.scrubber.update(bytes.fromhex(str(msg.get("salt", ""))), hashes)
            return {"ok": True, "names": self.scrubber.size}
        if op == "audit":
            return {"ok": True, "entries": self.audit.tail(max(1, min(int(msg.get("limit", 100)), 500)))}
        return {"ok": False, "status": "refused", "reason": "unknown_op"}

    # --- lookups ------------------------------------------------------------------------

    def _lookup(self, msg: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        rid = str(msg.get("id") or uuid.uuid4().hex[:8])[:32]
        raw = msg.get("request")
        entry: dict[str, Any] = {"ts": time.time(), "id": rid, "kind": None, "request": None, "upstream": [],
                                 "status": "refused", "reason": None, "provider": None, "response_bytes": 0,
                                 "reply_bytes": 0, "scrubbed": 0, "duration_ms": 0}
        try:
            if len(encode(raw if isinstance(raw, dict) else {})) > schemas.MAX_REQUEST_BYTES:
                return self._finish(entry, t0, reason="too_large")
            req = schemas.LOOKUP_ADAPTER.validate_python(raw)
            entry["kind"] = req.kind
            req, scrubbed, emptied = self._scrub(req)
            entry["request"], entry["scrubbed"] = schemas.canonical(req), scrubbed
            if emptied:
                return self._finish(entry, t0, reason="personal_name")
            if self.offline:
                return self._finish(entry, t0, reason="offline")
            if not self._allow_rate(req.kind):
                return self._finish(entry, t0, reason="rate_limited")
            provider = self.providers.get(req.kind)
            if provider is None:
                return self._finish(entry, t0, reason="no_provider")
            entry["provider"] = provider.name
            ex = Exchange()
            try:
                data = provider.fetch(req.params, self.http, ex)
            finally:
                entry["upstream"], entry["response_bytes"] = ex.requests, ex.bytes_received
            reply = {"ok": True, "status": "ok", "data": data}
            entry["reply_bytes"] = len(encode(data))
            return self._finish(entry, t0, status="ok", reply=reply)
        except ValidationError as e:
            entry["kind"] = raw.get("kind") if isinstance(raw, dict) and isinstance(raw.get("kind"), str) else None
            first = e.errors()[0]
            return self._finish(entry, t0, reason="invalid", detail=f"{'.'.join(map(str, first['loc']))}: {first['msg']}")
        except HostNotAllowed as e:
            return self._finish(entry, t0, reason="host_not_allowed", detail=str(e))
        except UpstreamError as e:
            return self._finish(entry, t0, status="error", reason="upstream_error", detail=str(e))
        except Exception as e:  # noqa: BLE001 - never crash the gateway on one bad call
            log.exception("lookup failed")
            return self._finish(entry, t0, status="error", reason="internal_error", detail=type(e).__name__)

    def _scrub(self, req):
        """Drop person names from free-text params. A city that is a person's name is refused."""
        p = req.params
        if isinstance(p, schemas.WeatherParams):
            city, n = self.scrubber.scrub(p.city)
            return req, n, n > 0
        field = "query" if isinstance(p, schemas.SearchParams) else "topic"
        cleaned, n = self.scrubber.scrub(getattr(p, field))
        if not n:
            return req, 0, False
        if not cleaned:
            return req.model_copy(update={"params": p.model_copy(update={field: "?"})}), n, True
        return req.model_copy(update={"params": p.model_copy(update={field: cleaned})}), n, False

    def _allow_rate(self, kind: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._calls[kind]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= MAX_LOOKUPS_PER_MINUTE:
                return False
            q.append(now)
            return True

    def _finish(self, entry, t0, *, status="refused", reason=None, detail=None, reply=None) -> dict[str, Any]:
        entry.update(status=status, reason=reason, duration_ms=int((time.monotonic() - t0) * 1000))
        if detail:
            entry["detail"] = detail[:200]
        self.audit.append(entry)
        out = reply or {"ok": False, "status": status, "reason": reason, "detail": entry.get("detail")}
        out["audit"] = entry
        return out


# --- process / transport ---------------------------------------------------------------------


class GatewayServer:
    def __init__(self, service: GatewayService, address: str, family: str, authkey: bytes) -> None:
        from multiprocessing.connection import Listener

        self.service = service
        self.listener = Listener(address=address, family=family, authkey=authkey)
        self.address = address

    def serve_forever(self) -> None:
        log.info("gateway listening on %s; allowed hosts: %s", self.address, ", ".join(sorted(self.service.allowed_hosts)))
        while True:
            try:
                conn = self.listener.accept()
            except (OSError, EOFError, AuthenticationError):
                # A client without the key, or one that went away. Never let it take the gateway down.
                log.warning("rejected a connection that failed authentication")
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="gateway-conn").start()

    def _serve(self, conn) -> None:
        with conn:
            while True:
                try:
                    data = conn.recv_bytes(MAX_MESSAGE_BYTES)
                except (EOFError, OSError):
                    return
                try:
                    reply = self.service.handle(decode(data))
                except (ValueError, UnicodeDecodeError):
                    reply = {"ok": False, "status": "refused", "reason": "bad_message"}
                conn.send_bytes(encode(reply))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from kin.config import PROFILES, load_config
    from kin.gateway.providers import default_providers
    from kin.gateway.transport import gateway_address, load_or_create_secret

    parser = argparse.ArgumentParser(prog="kin gateway", description="Kin Lookup Gateway (separate process)")
    parser.add_argument("--profile", choices=PROFILES, default="laptop")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--providers", choices=["real", "fake"], default=None,
                        help="fake: canned answers for demos without internet (default: from the profile)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.profile)
    if args.data_dir:
        cfg.data_dir = __import__("pathlib").Path(args.data_dir)
    try:
        from dotenv import load_dotenv

        load_dotenv(cfg.data_path(".env")) or load_dotenv()
    except ImportError:
        pass
    if (args.providers or cfg.gateway.providers) == "fake":
        from kin.gateway.providers.fake import fake_providers

        providers = fake_providers()
    else:
        providers = default_providers()
    service = GatewayService(providers, AuditLog(cfg.data_path(cfg.gateway.audit_log)), timeout_s=cfg.gateway.upstream_timeout_s)
    address, family = gateway_address(cfg.data_dir)
    server = GatewayServer(service, address, family, load_or_create_secret(cfg.data_path(cfg.gateway.key_file)))
    if family == "AF_UNIX":
        os.chmod(address, 0o600)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
