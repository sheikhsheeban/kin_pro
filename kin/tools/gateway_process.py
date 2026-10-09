"""Start and stop the Lookup Gateway as its own OS process (`python -m kin.gateway`).

On the laptop `kin run` spawns it; on the Pi it becomes its own systemd unit
running as its own user (Phase 4). Either way the brain only ever talks to it
through `LookupClient` over the authenticated local pipe/socket.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from kin.config import KinConfig
from kin.gateway.transport import gateway_address, load_or_create_secret
from kin.tools.lookup_client import GatewayUnavailable, LookupClient

log = logging.getLogger(__name__)


def make_client(cfg: KinConfig) -> LookupClient:
    address, family = gateway_address(cfg.data_dir)
    key = load_or_create_secret(cfg.data_path(cfg.gateway.key_file))
    return LookupClient(address, family, key, cfg.gateway.client_timeout_s)


class GatewayProcess:
    def __init__(self, cfg: KinConfig) -> None:
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.client = make_client(cfg)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, wait_s: float = 15.0) -> dict:
        """Spawn (unless one is already answering) and wait until it responds to ping."""
        try:
            return self.client.ping()  # already running (e.g. its own service on the Pi)
        except GatewayUnavailable:
            pass
        address, family = gateway_address(self.cfg.data_dir)
        if family == "AF_UNIX":
            Path(address).unlink(missing_ok=True)  # stale socket from a crashed run
        cmd = [sys.executable, "-m", "kin.gateway", "--profile", self.cfg.profile,
               "--data-dir", str(self.cfg.data_dir), "--providers", self.cfg.gateway.providers]
        log.info("starting gateway process: %s", " ".join(cmd[1:]))
        self.proc = subprocess.Popen(cmd)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"gateway process exited with code {self.proc.returncode}")
            try:
                return self.client.ping()
            except GatewayUnavailable:
                time.sleep(0.1)
        raise RuntimeError("gateway did not become ready in time")

    def stop(self) -> None:
        self.client.close()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
