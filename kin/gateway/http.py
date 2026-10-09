"""The gateway's only door to the internet.

- HTTPS GET only, to an explicit host allowlist (mirrors the nftables rule on the Pi).
- Response size is capped and every request is recorded for the audit log.
- No cookies, no redirects to other hosts, a fixed User-Agent.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

USER_AGENT = "Kin/0.1 (private on-device assistant; factual lookups only)"
MAX_RESPONSE_BYTES = 512 * 1024


class HostNotAllowed(PermissionError):
    pass


class UpstreamError(RuntimeError):
    pass


@dataclass
class Exchange:
    """What one provider call sent and received, for the audit log."""

    requests: list[str] = field(default_factory=list)  # "GET host/path?query" exactly as sent
    bytes_received: int = 0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise UpstreamError(f"redirect to {urllib.parse.urlsplit(newurl).hostname} refused")


class HttpClient:
    def __init__(self, allowed_hosts: set[str], timeout_s: float = 6.0) -> None:
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        self.timeout_s = timeout_s
        self._opener = urllib.request.build_opener(_NoRedirect())

    def get(
        self, url: str, params: dict | None, ex: Exchange, *, headers: dict | None = None, redact: set[str] = frozenset()
    ) -> bytes:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https" or host not in self.allowed_hosts:
            raise HostNotAllowed(f"gateway refuses {parts.scheme}://{host}: not on the allowlist")
        query = urllib.parse.urlencode(params or {})
        full = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))
        shown = {k: ("<redacted>" if k in redact else v) for k, v in (params or {}).items()}
        ex.requests.append(f"GET {host}{parts.path}?{urllib.parse.urlencode(shown)}")
        req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})})
        t0 = time.monotonic()
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                body = resp.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            raise UpstreamError(f"{host} answered HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise UpstreamError(f"{host} unreachable ({getattr(e, 'reason', e)})") from e
        finally:
            ex.requests[-1] += f"  [{int((time.monotonic() - t0) * 1000)} ms]"
        if len(body) > MAX_RESPONSE_BYTES:
            raise UpstreamError(f"{host} response exceeded {MAX_RESPONSE_BYTES // 1024} KB")
        ex.bytes_received += len(body)
        return body
