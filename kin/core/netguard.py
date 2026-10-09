"""Socket guard for the brain process: no outbound network, ever.

On the Pi the real enforcement is the OS (systemd PrivateNetwork=yes, nftables).
On a laptop this guard is the stand-in. Once installed it refuses, for the rest
of the process lifetime:

  - socket.connect / connect_ex / sendto to any non-loopback address (audit hook,
    so it covers every library that uses the socket module)
  - DNS resolution of any non-local host name (audit hook on getaddrinfo etc.)
  - asyncio connections, which on Windows bypass socket.connect via ConnectEx
    (patched event-loop methods)

Loopback, Unix sockets and the gateway's named pipe stay allowed. Audit hooks
cannot be removed, by design. The gateway is a separate process and never installs this.
"""

from __future__ import annotations

import ipaddress
import socket
import sys

_installed = False
_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", ""})


class NetworkBlocked(PermissionError):
    pass


def _is_local_host(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if not isinstance(host, str):
        return False
    h = host.strip("[]").split("%", 1)[0].lower()
    if h in _LOCAL_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def is_local_address(address: object) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path or abstract socket
        return True
    if isinstance(address, tuple) and address:
        return _is_local_host(address[0])
    return False


def _refuse(what: str, target: object) -> None:
    raise NetworkBlocked(f"Kin network guard: the brain process may not {what} {target!r}; lookups go through the gateway")


def _audit(event: str, args: tuple) -> None:
    if event in ("socket.connect", "socket.sendto"):
        sock, address = args[0], args[-1]
        if getattr(sock, "family", None) == getattr(socket, "AF_UNIX", object()):
            return
        if not is_local_address(address):
            _refuse("connect to", address)
    elif event == "socket.getaddrinfo":
        if not _is_local_host(args[0]):
            _refuse("resolve", args[0])
    elif event in ("socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr"):
        if not _is_local_host(args[0]):
            _refuse("resolve", args[0])


def _patch_asyncio() -> None:
    import asyncio
    from asyncio import base_events, proactor_events, selector_events

    def guard_sock_connect(original):
        async def sock_connect(self, sock, address):
            if getattr(sock, "family", None) != getattr(socket, "AF_UNIX", object()) and not is_local_address(address):
                _refuse("connect to", address)
            return await original(self, sock, address)

        sock_connect.__kin_guard__ = True
        return sock_connect

    for cls in (base_events.BaseEventLoop, proactor_events.BaseProactorEventLoop, selector_events.BaseSelectorEventLoop):
        current = cls.__dict__.get("sock_connect")
        if current is not None and not getattr(current, "__kin_guard__", False):
            cls.sock_connect = guard_sock_connect(current)

    original_create = base_events.BaseEventLoop.create_connection
    if not getattr(original_create, "__kin_guard__", False):

        async def create_connection(self, protocol_factory, host=None, port=None, *args, **kwargs):
            if kwargs.get("sock") is None and not _is_local_host(host):
                _refuse("connect to", (host, port))
            return await original_create(self, protocol_factory, host, port, *args, **kwargs)

        create_connection.__kin_guard__ = True
        base_events.BaseEventLoop.create_connection = create_connection
    assert asyncio  # imported for side effects only


def install_network_guard() -> None:
    global _installed
    if _installed:
        return
    sys.addaudithook(_audit)
    _patch_asyncio()
    _installed = True


def guard_installed() -> bool:
    return _installed
