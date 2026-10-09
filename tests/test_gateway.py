"""Phase 3: the Lookup Gateway (schemas, scrubbing, offline, audit, allowlist, process + auth) and the socket guard."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from kin.gateway.audit import AuditLog
from kin.gateway.http import Exchange, HostNotAllowed, HttpClient
from kin.gateway.providers.fake import fake_providers
from kin.gateway.scrub import NameScrubber, name_hash
from kin.gateway.server import GatewayService
from kin.gateway.transport import gateway_address, load_or_create_secret
from kin.tools.lookup_client import LookupClient
from tests.conftest import ROOT

TODAY = datetime.date.today().isoformat()


@pytest.fixture
def service(tmp_path):
    return GatewayService(fake_providers(), AuditLog(tmp_path / "audit.jsonl"))


def lookup(service, request):
    return service.handle({"op": "lookup", "id": "t1", "request": request})


# --- schema: only {kind, params} with strict params ----------------------------------------------


def test_valid_weather_lookup_returns_facts_and_exact_audit(service):
    r = lookup(service, {"kind": "weather", "params": {"city": "London", "date": TODAY}})
    assert r["ok"] and r["data"]["hourly"] and r["data"]["city"] == "London"
    a = r["audit"]
    assert json.loads(a["request"]) == {"kind": "weather", "params": {"city": "London", "date": TODAY}}
    assert a["status"] == "ok" and a["response_bytes"] > 0 and a["duration_ms"] >= 0


@pytest.mark.parametrize(
    "request_",
    [
        {"kind": "weather", "params": {"city": "London", "date": TODAY, "transcript": "I'm meeting Priya"}},
        {"kind": "weather", "params": {"city": "London"}},
        {"kind": "email", "params": {"to": "x@example.com"}},
        {"kind": "search", "params": {"query": "x" * 121}},
        {"kind": "search", "params": {"query": "one two three four five six seven eight nine ten eleven twelve thirteen"}},
        {"kind": "search", "params": {"query": "<script>alert(1)</script>"}},
        {"kind": "news", "params": {"topic": "a\nb"}},
        {"kind": "weather", "params": {"city": "London", "date": "2001-01-01"}},
        {"kind": "weather", "params": {"city": "London", "date": TODAY}, "prompt": "You are Kin..."},
        "weather London",
        None,
    ],
)
def test_anything_else_is_refused_and_audited(service, request_):
    r = lookup(service, request_)
    assert not r["ok"] and r["status"] == "refused" and r["reason"] in ("invalid", "too_large")
    assert service.audit.tail(1)[0]["status"] == "refused"


def test_request_size_cap(service):
    r = lookup(service, {"kind": "search", "params": {"query": "hi"}, "pad": "x" * 600})
    assert r["reason"] == "too_large"


# --- names are scrubbed without the gateway ever seeing them ------------------------------------------


def test_person_names_are_scrubbed_from_free_text(service):
    salt = os.urandom(16)
    service.handle({"op": "set_names", "salt": salt.hex(), "hashes": [name_hash(salt, "Priya"), name_hash(salt, "Rahul Mehta")]})
    r = lookup(service, {"kind": "search", "params": {"query": "coffee with Priya and Rahul Mehta at the library"}})
    sent = json.loads(r["audit"]["request"])["params"]["query"]
    assert "Priya" not in sent and "Rahul" not in sent and "Mehta" not in sent
    assert sent == "coffee with and at the library" and r["audit"]["scrubbed"] == 3


def test_city_that_is_a_person_name_is_refused(service):
    salt = os.urandom(16)
    service.handle({"op": "set_names", "salt": salt.hex(), "hashes": [name_hash(salt, "Priya")]})
    r = lookup(service, {"kind": "weather", "params": {"city": "Priya", "date": TODAY}})
    assert r["reason"] == "personal_name"


def test_scrubber_holds_only_hashes():
    s = NameScrubber()
    salt = b"salt"
    s.update(salt, [name_hash(salt, "Priya")])
    assert "Priya" not in repr(s.__dict__) and "priya" not in repr(s.__dict__)
    assert s.scrub("ask Priya about it") == ("ask about it", 1)


# --- offline, rate limit, audit ---------------------------------------------------------------------


def test_offline_mode_refuses_every_lookup(service):
    service.handle({"op": "set_offline", "value": True})
    for kind, params in [("weather", {"city": "London", "date": TODAY}), ("news", {"topic": "pi"}), ("search", {"query": "pi"})]:
        r = lookup(service, {"kind": kind, "params": params})
        assert r["reason"] == "offline" and r["audit"]["upstream"] == []
    service.handle({"op": "set_offline", "value": False})
    assert lookup(service, {"kind": "news", "params": {"topic": "pi"}})["ok"]


def test_rate_limit(service, monkeypatch):
    monkeypatch.setattr("kin.gateway.server.MAX_LOOKUPS_PER_MINUTE", 3)
    results = [lookup(service, {"kind": "news", "params": {"topic": "pi"}}) for _ in range(5)]
    assert [r["ok"] for r in results] == [True, True, True, False, False]
    assert results[-1]["reason"] == "rate_limited"


def test_audit_log_persists_and_contains_no_transcript(service, tmp_path):
    lookup(service, {"kind": "weather", "params": {"city": "London", "date": TODAY}})
    lookup(service, {"kind": "weather", "params": {"city": "London", "date": TODAY, "note": "I'm meeting Priya at 4"}})
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(x) for x in raw.splitlines()]
    assert len(lines) == 2 and lines[0]["status"] == "ok" and lines[1]["status"] == "refused"
    assert "Priya" not in raw and "meeting" not in raw  # refused payloads are not echoed into the log
    assert AuditLog(tmp_path / "audit.jsonl").tail(10) == lines


# --- the HTTP door ------------------------------------------------------------------------------------


def test_http_client_refuses_hosts_off_the_allowlist_and_plain_http():
    http = HttpClient({"api.open-meteo.com"})
    with pytest.raises(HostNotAllowed):
        http.get("https://evil.example/collect", {"q": "x"}, Exchange())
    with pytest.raises(HostNotAllowed):
        http.get("http://api.open-meteo.com/v1/forecast", None, Exchange())


def test_allowlist_is_exactly_the_provider_hosts():
    from kin.gateway.providers.news import GoogleNewsRss
    from kin.gateway.providers.search import WikipediaSearch
    from kin.gateway.providers.weather import OpenMeteo

    svc = GatewayService({"weather": OpenMeteo(), "search": WikipediaSearch(), "news": GoogleNewsRss()}, AuditLog(None))
    assert svc.allowed_hosts == {"geocoding-api.open-meteo.com", "api.open-meteo.com", "en.wikipedia.org", "news.google.com"}


# --- the real separate process over the authenticated pipe/socket -------------------------------------------


@pytest.fixture
def gateway_proc(tmp_path):
    key = load_or_create_secret(tmp_path / "data" / "gateway.key")
    proc = subprocess.Popen([sys.executable, "-m", "kin.gateway", "--data-dir", str(tmp_path), "--providers", "fake"], cwd=ROOT)
    addr, fam = gateway_address(tmp_path)
    client = LookupClient(addr, fam, key, timeout_s=5)
    for _ in range(100):
        try:
            client.ping()
            break
        except Exception:
            time.sleep(0.1)
    yield proc, client, addr, fam
    client.close()
    proc.terminate()
    proc.wait(5)


def test_gateway_runs_as_a_separate_process(gateway_proc):
    proc, client, _a, _f = gateway_proc
    assert proc.pid != os.getpid()
    info = client.ping()
    assert info["ok"] and info["providers"]["weather"] == "fake-weather"
    r = client.lookup("weather", {"city": "London", "date": TODAY})
    assert r.ok and r.data["max_rain_chance_pct"] == 80
    assert json.loads(r.audit["request"]) == {"kind": "weather", "params": {"city": "London", "date": TODAY}}
    client.set_offline(True)
    assert client.lookup("weather", {"city": "London", "date": TODAY}).reason == "offline"
    assert len(client.audit(10)) == 2


def test_wrong_key_is_rejected_and_gateway_survives(gateway_proc):
    _p, client, addr, fam = gateway_proc
    intruder = LookupClient(addr, fam, b"\x00" * 32, timeout_s=3)
    assert intruder.lookup("weather", {"city": "London", "date": TODAY}).reason == "gateway_unavailable"
    assert client.ping()["ok"]


def test_client_reports_gateway_down_honestly(tmp_path):
    addr, fam = gateway_address(tmp_path)
    r = LookupClient(addr, fam, b"k" * 32, timeout_s=1).lookup("news", {"topic": "pi"})
    assert not r.ok and r.reason == "gateway_unavailable"


# --- the brain process cannot open an outbound socket ------------------------------------------------------------

GUARD_SCRIPT = textwrap.dedent(
    """
    import asyncio, json, socket, threading, urllib.request
    from kin.core.netguard import NetworkBlocked, install_network_guard

    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen()
    port = srv.getsockname()[1]
    threading.Thread(target=srv.accept, daemon=True).start()
    install_network_guard()

    def blocked(fn):
        try:
            fn()
            return False
        except NetworkBlocked:
            return True
        except Exception as e:
            return isinstance(getattr(e, "reason", None), NetworkBlocked) or "network guard" in str(e)

    async def raw():
        s = socket.socket(); s.setblocking(False)
        await asyncio.get_running_loop().sock_connect(s, ("1.1.1.1", 80))

    out = {
        "tcp": blocked(lambda: socket.create_connection(("1.1.1.1", 80), timeout=2)),
        "connect": blocked(lambda: socket.socket().connect(("8.8.8.8", 53))),
        "connect_ex": blocked(lambda: socket.socket().connect_ex(("8.8.8.8", 53))),
        "udp": blocked(lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53))),
        "dns": blocked(lambda: socket.getaddrinfo("example.com", 443)),
        "urllib": blocked(lambda: urllib.request.urlopen("https://example.com", timeout=3)),
        "asyncio_ip": blocked(lambda: asyncio.run(asyncio.open_connection("1.1.1.1", 80))),
        "asyncio_host": blocked(lambda: asyncio.run(asyncio.open_connection("example.com", 80))),
        "asyncio_sock": blocked(lambda: asyncio.run(raw())),
        "ipv6": blocked(lambda: socket.socket(socket.AF_INET6).connect(("2606:4700:4700::1111", 80))),
        "loopback_ok": not blocked(lambda: socket.create_connection(("127.0.0.1", port), timeout=2).close()),
    }
    print("RESULT " + json.dumps(out))
    """
)


def test_brain_process_cannot_open_an_outbound_socket():
    out = subprocess.run([sys.executable, "-c", GUARD_SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=60)
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT "))
    result = json.loads(line[7:])
    assert all(result.values()), result


def test_kin_app_installs_the_guard(tmp_path):
    """`kin run` (KinApp.start) installs the guard in its own process."""
    script = textwrap.dedent(
        """
        import asyncio, socket
        from tests.conftest import build_fake_app
        from kin.core.netguard import NetworkBlocked, guard_installed

        async def main():
            h = build_fake_app()
            await h.app.start()
            try:
                socket.create_connection(("1.1.1.1", 443), timeout=2)
                print("RESULT open")
            except NetworkBlocked:
                print("RESULT blocked", guard_installed(), h.app.snapshot()["network_guard"])
            finally:
                await h.app.stop()
        asyncio.run(main())
        """
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert "RESULT blocked True True" in out.stdout, out.stdout + out.stderr[-1500:]
