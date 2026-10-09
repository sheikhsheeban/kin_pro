"""Encrypted backup and restore of the memory database. Local folders only: nothing is ever uploaded.

Losing the key means losing every memory, so a backup carries the key too, wrapped by a passphrase:

  kin-backup-<stamp>.kinbak
    line 1  KINBAK1
    line 2  JSON header: scrypt parameters and salt, the wrapped key, the database's SHA-256, an HMAC tag
    rest    the database file itself (already SQLCipher-encrypted with the key)

Key wrapping uses only the standard library: scrypt(passphrase, salt, n=2^15, r=8, p=1) gives 64 bytes;
the first 32 are XORed with the 32-byte database key (a fresh salt makes this pad single-use), the
last 32 key an HMAC-SHA256 over the header fields, the wrapped key and the database hash. A wrong
passphrase or a changed file fails the tag check before anything is written.

The database copy is made with SQLite's online backup API, so it is consistent while Kin runs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

MAGIC = b"KINBAK1\n"
SCRYPT = {"n": 2**15, "r": 8, "p": 1}
MIN_PASSPHRASE = 8


_COMMON = {"password", "passphrase", "qwerty", "letmein", "iloveyou", "welcome", "admin", "kinbackup", "123456789"}


class BackupError(RuntimeError):
    pass


def passphrase_problem(passphrase: str) -> str | None:
    """Why a new backup passphrase is too weak, or None. Losing the backup is bad; a guessable passphrase is
    worse (whoever finds the file gets every memory). At least 16 characters (a few words), or 12 that mix
    three of: lower case, upper case, digits, symbols; not one repeated character or a common password."""
    p = passphrase or ""
    classes = sum(any(f(c) for c in p) for f in (str.islower, str.isupper, str.isdigit, lambda c: not c.isalnum()))
    if len(p) < 12 or (len(p) < 16 and classes < 3):
        return "the passphrase must have at least 16 characters (a few words), or at least 12 mixing upper and lower case, digits and symbols"
    if len(set(p.lower())) < 6:
        return "the passphrase repeats too few different characters"
    rest = "".join(c for c in p.lower() if c.isalnum())
    for w in _COMMON:
        rest = rest.replace(w, "")
    if len(_drop_runs(rest)) < 6:
        return "the passphrase is too close to a common password or a sequence like 12345678"
    return None


def _drop_runs(s: str) -> str:
    """`s` without runs of 3+ consecutive or repeated characters (abc, 4321, zzz)."""
    out, i = [], 0
    while i < len(s):
        j = i + 1
        if j < len(s) and abs(ord(s[j]) - ord(s[i])) <= 1:
            step = ord(s[j]) - ord(s[i])
            while j < len(s) and ord(s[j]) - ord(s[j - 1]) == step:
                j += 1
        if j - i >= 3:
            i = j
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _kdf(passphrase: str, salt: bytes, params: dict) -> tuple[bytes, bytes]:
    if len(passphrase) < MIN_PASSPHRASE:
        raise BackupError(f"the passphrase must have at least {MIN_PASSPHRASE} characters")
    k = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=params["n"], r=params["r"], p=params["p"],
                       maxmem=128 * 1024 * 1024, dklen=64)
    return k[:32], k[32:]


def _tag(mac_key: bytes, header: dict) -> str:
    body = json.dumps({k: header[k] for k in sorted(header) if k != "tag"}, sort_keys=True).encode()
    return hmac.new(mac_key, body, hashlib.sha256).hexdigest()


def _local_folder(folder: Path | str) -> Path:
    s = str(folder)
    if "://" in s:
        raise BackupError("backups go to a local folder only")
    p = Path(s).expanduser()
    if not p.is_dir():
        raise BackupError(f"{p} is not a folder")
    return p


def wrap(db_bytes_sha: str, key_hex: str, passphrase: str) -> dict:
    salt = secrets.token_bytes(16)
    enc, mac = _kdf(passphrase, salt, SCRYPT)
    wrapped = bytes(a ^ b for a, b in zip(bytes.fromhex(key_hex), enc))
    header = {"version": 1, "kdf": "scrypt", **SCRYPT, "salt": salt.hex(), "wrapped_key": wrapped.hex(),
              "db_sha256": db_bytes_sha, "created": datetime.now().isoformat(timespec="seconds")}
    header["tag"] = _tag(mac, header)
    return header


def unwrap(header: dict, passphrase: str) -> str:
    if header.get("kdf") != "scrypt" or header.get("version") != 1:
        raise BackupError("not a Kin backup this version can read")
    enc, mac = _kdf(passphrase, bytes.fromhex(header["salt"]), header)
    if not hmac.compare_digest(_tag(mac, header), header.get("tag", "")):
        raise BackupError("wrong passphrase, or the backup file was changed")
    return bytes(a ^ b for a, b in zip(bytes.fromhex(header["wrapped_key"]), enc)).hex()


def _check_copy(path: Path, key_hex: str) -> None:
    """The copy opens with the key and passes SQLite's integrity check."""
    import sqlcipher3

    db = sqlcipher3.connect(str(path))
    try:
        db.execute(f"PRAGMA key = \"x'{key_hex}'\"")
        result = db.execute("PRAGMA integrity_check").fetchone()[0]
    except sqlcipher3.DatabaseError as exc:
        raise BackupError(f"the database copy does not open: {exc}") from exc
    finally:
        db.close()
    if result != "ok":
        raise BackupError(f"the database copy failed its integrity check: {result}")


def clean_interrupted(folder: Path) -> int:
    """Remove what a backup killed half way leaves behind: `.part` files in the folder and the temporary
    database copy (both encrypted, but useless and in the way). Never touches a finished `.kinbak`."""
    n = 0
    for part in Path(folder).glob("*-*.part"):
        if part.name.startswith(("kin-backup-", "pre-upgrade-")):
            part.unlink(missing_ok=True)
            n += 1
    for tmp in Path(tempfile.gettempdir()).glob("kin-backup-*"):
        if tmp.is_dir() and time.time() - tmp.stat().st_mtime > 60:  # not another backup running right now
            shutil.rmtree(tmp, ignore_errors=True)
            n += 1
    return n


def write_backup(store, folder: Path | str, passphrase: str, *, name: str = "kin-backup") -> Path:
    """Run on the memory worker (the store's thread). Returns the backup file."""
    out_dir = _local_folder(folder)
    problem = passphrase_problem(passphrase)  # only new backups: old ones still open with their 8+ characters
    if problem:
        raise BackupError(problem)
    clean_interrupted(out_dir)
    with tempfile.TemporaryDirectory(prefix="kin-backup-") as tmp:
        copy = Path(tmp) / "kin.db"
        store.backup_to(copy)
        _check_copy(copy, store._key_hex)
        data = copy.read_bytes()
    header = wrap(hashlib.sha256(data).hexdigest(), store._key_hex, passphrase)
    target = out_dir / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.kinbak"
    part = target.with_suffix(".part")  # a crash leaves at most a .part file, never a half-written backup
    with open(part, "wb") as f:
        f.write(MAGIC + json.dumps(header).encode() + b"\n" + data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, target)
    return target


def pre_upgrade_backup(store, from_version: int, to_version: int) -> Path:
    """Before a schema upgrade: a backup in `<data>/backups/`, same format as `kin backup`, whose passphrase is
    the database key itself (it sits next to the key file, so it is exactly as protected as the database).
    Read back and checked before the upgrade may start."""
    folder = Path(store.path).parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    path = write_backup(store, folder, store._key_hex, name=f"pre-upgrade-v{from_version}-to-v{to_version}")
    read_backup(path, store._key_hex)  # tag and database hash
    return path


def put_back(backup: Path, db_path: Path | str, key_hex: str) -> None:
    """Replace the database file with the one inside a pre-upgrade backup (passphrase = key)."""
    _header, data, key = read_backup(backup, key_hex)
    if key != key_hex:
        raise BackupError("the backup belongs to another database key")
    db_path = Path(db_path)
    part = db_path.with_name(db_path.name + ".restoring")
    part.write_bytes(data)
    for extra in ("-journal", "-wal", "-shm"):  # a rolled-back upgrade must not be replayed onto the old file
        db_path.with_name(db_path.name + extra).unlink(missing_ok=True)
    os.replace(part, db_path)


def read_backup(path: Path | str, passphrase: str) -> tuple[dict, bytes, str]:
    raw = Path(path).read_bytes()
    if not raw.startswith(MAGIC):
        raise BackupError(f"{path} is not a Kin backup")
    head, _, data = raw[len(MAGIC):].partition(b"\n")
    header = json.loads(head)
    key = unwrap(header, passphrase)
    if hashlib.sha256(data).hexdigest() != header["db_sha256"]:
        raise BackupError("the database inside the backup is damaged")
    return header, data, key


def restore_drill(path: Path | str, passphrase: str) -> dict:
    """Prove a backup can be restored without touching the live database: unwrap the key, check the hash,
    write the database to a temporary folder, open it with the key and run SQLite's integrity check."""
    header, data, key = read_backup(path, passphrase)
    with tempfile.TemporaryDirectory(prefix="kin-drill-") as tmp:
        copy = Path(tmp) / "kin.db"
        copy.write_bytes(data)  # still SQLCipher-encrypted
        _check_copy(copy, key)
        import sqlcipher3

        db = sqlcipher3.connect(str(copy))
        try:
            db.execute(f"PRAGMA key = \"x'{key}'\"")
            meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            facts = db.execute("SELECT count(*) FROM facts").fetchone()[0] if "facts" in tables else None
            lines = db.execute("SELECT count(*) FROM utterances").fetchone()[0] if "utterances" in tables else None
        finally:
            db.close()
    return {"ok": True, "created": header.get("created"), "schema_version": int(meta.get("schema_version", 0)),
            "facts": facts, "lines": lines, "integrity": "ok", "bytes": len(data)}


def restore_backup(path: Path | str, passphrase: str, db_path: Path, key_file: Path, *, force: bool = False) -> dict:
    """Write the database and its key back. Kin must not be running. An existing database is kept as *.before-restore."""
    header, data, key = read_backup(path, passphrase)
    db_path, key_file = Path(db_path), Path(key_file)
    if db_path.exists() and not force:
        raise BackupError(f"{db_path} exists; use --force to replace it (it is kept as {db_path.name}.before-restore)")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        shutil.copy2(db_path, db_path.with_name(db_path.name + ".before-restore"))
        if key_file.exists():
            shutil.copy2(key_file, key_file.with_name(key_file.name + ".before-restore"))
    part = db_path.with_name(db_path.name + ".restoring")
    part.write_bytes(data)
    from kin.memory.store import MemoryStore

    dim = None
    try:  # prove the key opens it before replacing anything
        import sqlcipher3

        db = sqlcipher3.connect(str(part))
        db.execute(f"PRAGMA key = \"x'{key}'\"")
        dim = int(db.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()[0])
        db.close()
        MemoryStore(part, key, dim).close()
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise BackupError(f"the restored database does not open: {exc}") from exc
    os.replace(part, db_path)
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(key)
    return {"restored": str(db_path), "key_file": str(key_file), "created": header.get("created"), "embedding_dim": dim}
