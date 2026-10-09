"""`kin` command line.

  kin run --profile laptop       start Kin and the local UI
  kin download-models            fetch the models into ./models (~3.3 GB)
  kin gateway --profile laptop   run the Lookup Gateway on its own (kin run starts it for you)
  kin devices                    list audio input devices
  kin backup --to D:/backups     encrypted backup of every memory and its key (asks for a passphrase)
  kin restore FILE [--force]     put a backup back (Kin must be stopped)
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
import webbrowser

from kin.config import PROFILES, load_config


def _run(args: argparse.Namespace) -> int:
    import uvicorn

    from kin.app import KinApp
    from kin.ui.server import create_app

    overrides: dict = {}
    if args.device is not None:
        overrides["audio"] = {**load_config(args.profile).audio.model_dump(), "device": _device_arg(args.device)}
    cfg = load_config(args.profile, overrides)
    if args.demo:
        _demo_config(cfg)
    if args.port:
        cfg.ui.port = args.port
    if args.wake_threshold is not None:
        if not 0.0 < args.wake_threshold <= 1.0:
            print("--wake-threshold must be between 0 and 1", file=sys.stderr)
            return 2
        cfg.wake.threshold = args.wake_threshold
    kin = KinApp(cfg)
    app = create_app(kin)
    url = f"http://{cfg.ui.host}:{cfg.ui.port}"
    print(f"Kin [{cfg.profile}] starting. Open {url}")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(app, host=cfg.ui.host, port=cfg.ui.port, log_level="warning", ws_ping_interval=20)
    return 0


def _demo_config(cfg) -> None:
    """Demo mode: every file (database, key, gateway audit, caches) lives under demo/, never next to real data."""
    from kin.config import project_root

    cfg.data_dir = project_root() / "demo"
    cfg.demo = True
    os.environ.pop("KIN_DB_KEY", None)  # the demo database has its own key file
    if not cfg.data_path(cfg.memory.db_path).exists():
        from kin.memory.demo_week import seed_demo

        print(seed_demo(cfg))


def _device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _devices(_args: argparse.Namespace) -> int:
    import sounddevice as sd

    print(sd.query_devices())
    return 0


def _download(args: argparse.Namespace) -> int:
    from kin.models import download_models

    download_models(load_config(args.profile), fallback=args.fallback, reranker=args.reranker)
    return 0


def _gateway(args: argparse.Namespace) -> int:
    from kin.gateway.server import main as gateway_main

    extra = ["--profile", args.profile] + (["--providers", args.providers] if args.providers else [])
    return gateway_main(extra)


def _passphrase(confirm: bool) -> str:
    p = getpass.getpass("Backup passphrase: ")
    if confirm and getpass.getpass("Repeat it: ") != p:
        raise SystemExit("the passphrases differ")
    return p


def _backup(args: argparse.Namespace) -> int:
    """Back up while Kin is stopped (the UI's Backup button does the same while it runs)."""
    from kin.memory.backup import BackupError, write_backup
    from kin.memory.store import MemoryStore, load_or_create_key

    cfg = load_config(args.profile)
    db = cfg.data_path(cfg.memory.db_path)
    if not db.exists():
        print(f"no memory database at {db}")
        return 1
    store = MemoryStore(db, load_or_create_key(cfg.data_path(cfg.memory.key_file)), cfg.memory.dim, reembed=True)
    try:
        out = write_backup(store, args.to, args.passphrase or _passphrase(confirm=True))
    except BackupError as exc:
        print(f"backup failed: {exc}")
        return 1
    finally:
        store.close()
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB). Keep the passphrase: without it the backup cannot be opened.")
    return 0


def _restore(args: argparse.Namespace) -> int:
    from kin.memory.backup import BackupError, restore_backup

    cfg = load_config(args.profile)
    passphrase = args.passphrase
    if args.key_passphrase:  # a pre-upgrade backup (data/backups/pre-upgrade-*) is unlocked by the database key
        from kin.memory.store import load_or_create_key

        key_file = cfg.data_path(cfg.memory.key_file)
        if not key_file.exists() and not os.environ.get("KIN_DB_KEY"):
            print(f"no database key at {key_file}")
            return 1
        passphrase = load_or_create_key(key_file)
    try:
        info = restore_backup(args.file, passphrase or _passphrase(confirm=False), cfg.data_path(cfg.memory.db_path),
                              cfg.data_path(cfg.memory.key_file), force=args.force)
    except BackupError as exc:
        print(f"restore failed: {exc}")
        return 1
    print(f"restored the backup from {info['created']} to {info['restored']}; key written to {info['key_file']}"
          " (unset KIN_DB_KEY if you use it).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kin", description="Kin - private on-device companion")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start Kin and the local web UI")
    run.add_argument("--profile", choices=PROFILES, default="laptop")
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--device", default=None, help="input device index or name (see `kin devices`)")
    run.add_argument("--open", action="store_true", help="open the UI in a browser")
    run.add_argument("--wake-threshold", type=float, default=None,
                     help="wake word score needed to activate (profile default 0.5); measure yours with scripts/wake_probe.py")
    run.add_argument("--demo", action="store_true",
                     help="use a separate demo database under demo/ with a made-up week (UI: Reset demo)")
    run.set_defaults(func=_run)

    dl = sub.add_parser("download-models", help="download models into ./models")
    dl.add_argument("--profile", choices=PROFILES, default="laptop")
    dl.add_argument("--fallback", action="store_true", help="also fetch the GGUF fallback LLM (Qwen3.5-2B)")
    dl.add_argument("--reranker", action="store_true", help="also fetch the optional recall reranker (MiniLM cross-encoder, ~23 MB)")
    dl.set_defaults(func=_download)

    gw = sub.add_parser("gateway", help="run the Lookup Gateway process (the only networked part of Kin)")
    gw.add_argument("--profile", choices=PROFILES, default="laptop")
    gw.add_argument("--providers", choices=["real", "fake"], default=None)
    gw.set_defaults(func=_gateway)

    bk = sub.add_parser("backup", help="write an encrypted backup (database + passphrase-wrapped key) to a local folder")
    bk.add_argument("--to", required=True, help="a local folder, e.g. a USB drive")
    bk.add_argument("--profile", choices=PROFILES, default="laptop")
    bk.add_argument("--passphrase", default=None, help=argparse.SUPPRESS)  # tests only; prompt otherwise
    bk.set_defaults(func=_backup)

    rs = sub.add_parser("restore", help="restore an encrypted backup (stop Kin first)")
    rs.add_argument("file")
    rs.add_argument("--profile", choices=PROFILES, default="laptop")
    rs.add_argument("--force", action="store_true", help="replace an existing database (kept as *.before-restore)")
    rs.add_argument("--key-passphrase", action="store_true",
                    help="unlock with this device's database key (for the automatic pre-upgrade backups in data/backups)")
    rs.add_argument("--passphrase", default=None, help=argparse.SUPPRESS)
    rs.set_defaults(func=_restore)

    dev = sub.add_parser("devices", help="list audio devices")
    dev.set_defaults(func=_devices)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
