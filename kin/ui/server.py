"""Local web UI: FastAPI + one WebSocket + one HTML page.

Bound to localhost only. The WebSocket also checks the Origin header, so a
web page from elsewhere cannot connect to it and read transcripts or toggle
the (simulated) mute switch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from kin.app import KinApp
from kin.config import PROFILES
from kin.core.events import AudioLevel, Event, WakeLevel
from kin.hal import SimulatedHardware

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
OUTBOX_LIMIT = 500


def origin_allowed(origin: str | None) -> bool:
    if not origin:  # non-browser clients send no Origin
        return True
    return urlparse(origin).hostname in LOCAL_HOSTS


def create_app(kin: KinApp, manage_lifecycle: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if manage_lifecycle:
            await kin.start()
        try:
            yield
        finally:
            if manage_lifecycle:
                await kin.stop()

    app = FastAPI(title="Kin", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(kin.snapshot())

    @app.get("/api/system")
    async def system() -> JSONResponse:
        return JSONResponse(kin.system_stats())

    def memory_service():
        if kin.brain is None:
            raise HTTPException(503, "memory is still loading")
        return kin.brain.memory

    def same_origin(request: Request) -> None:
        # State-changing calls must come from the Kin page itself (CSRF protection).
        if not origin_allowed(request.headers.get("origin")):
            raise HTTPException(403, "cross-origin request refused")

    @app.get("/api/memories")
    async def memories(q: str | None = None, mode: str = "semantic", limit: int = 100, history: bool = False) -> JSONResponse:
        """Facts. Active ones by default; history=1 also lists superseded facts (nothing is deleted on update)."""
        svc = memory_service()
        limit = max(1, min(limit, 500))
        if q and mode == "semantic":
            hits = await svc.retrieve(q, min(limit, 20))
            items = [h.fact.to_dict() | {"similarity": h.similarity} for h in hits]
        else:
            items = [m.to_dict() for m in await svc.recent(limit, contains=q or None, status=None if history else "active")]
        return JSONResponse({
            "items": items,
            "total": await svc.count(),
            "superseded": await svc.count("superseded"),
            "episodes": await svc.episode_counts(),
        })

    async def _fact_or_404(memory_id: int):
        f = await memory_service().get(memory_id)
        if f is None:
            raise HTTPException(404, "no such memory")
        return f

    @app.post("/api/memories/{memory_id}/correct")
    async def correct_memory(memory_id: int, request: Request) -> JSONResponse:
        """Edit: the corrected text becomes a new version; the old one is kept as superseded (logged)."""
        same_origin(request)
        text = str((await _json_body(request)).get("text") or "").strip()
        if not 3 <= len(text) <= 400:
            raise HTTPException(400, "text must be 3-400 characters")
        await _fact_or_404(memory_id)
        new = await memory_service().correct(memory_id, text, actor="ui")
        if new is None:
            raise HTTPException(409, "only an active memory can be corrected")
        return JSONResponse(new.to_dict())

    @app.post("/api/memories/{memory_id}/retract")
    async def retract_memory(memory_id: int, request: Request) -> JSONResponse:
        same_origin(request)
        await _fact_or_404(memory_id)
        return JSONResponse({"retracted": await memory_service().retract(memory_id, actor="ui")})

    @app.post("/api/memories/{memory_id}/restore")
    async def restore_memory(memory_id: int, request: Request) -> JSONResponse:
        same_origin(request)
        await _fact_or_404(memory_id)
        return JSONResponse({"restored": await memory_service().restore(memory_id, actor="ui")})

    @app.post("/api/memories/{memory_id}/pin")
    async def pin_memory(memory_id: int, request: Request) -> JSONResponse:
        same_origin(request)
        pinned = bool((await _json_body(request)).get("pinned", True))
        await _fact_or_404(memory_id)
        await memory_service().pin(memory_id, pinned, actor="ui")
        return JSONResponse({"pinned": pinned})

    @app.post("/api/memories/merge")
    async def merge_memories(request: Request) -> JSONResponse:
        same_origin(request)
        body = await _json_body(request)
        try:
            keep, other = int(body["keep"]), int(body["other"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, "keep and other must be memory ids") from exc
        text = str(body.get("text") or "").strip() or None
        fact = await memory_service().merge(keep, other, text, actor="ui")
        if fact is None:
            raise HTTPException(409, "both memories must exist, differ and be active")
        return JSONResponse(fact.to_dict())

    @app.get("/api/review")
    async def review(day: str | None = None) -> JSONResponse:
        """The daily "what I remembered" card: facts added that day and every change to one."""
        from datetime import datetime as _dt

        try:
            when = _dt.fromisoformat(day) if day else None
        except ValueError as exc:
            raise HTTPException(400, "day must be YYYY-MM-DD") from exc
        return JSONResponse(await memory_service().review(when))

    @app.get("/api/demo")
    async def demo() -> JSONResponse:
        return JSONResponse({"demo": kin.cfg.demo, "data_dir": str(kin.cfg.data_dir) if kin.cfg.demo else None})

    @app.post("/api/demo/reset")
    async def demo_reset(request: Request) -> JSONResponse:
        """Only in demo mode (`kin run --demo`, its own database under demo/): wipe it and load the demo week."""
        same_origin(request)
        if not kin.cfg.demo:
            raise HTTPException(403, "not in demo mode: real memories are never reset")
        try:
            return JSONResponse(await memory_service().demo_reset())
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(await memory_service().health())

    @app.post("/api/maintenance")
    async def maintenance(request: Request) -> JSONResponse:
        same_origin(request)
        if kin.brain is None:
            raise HTTPException(503, "memory is still loading")
        return JSONResponse(await kin.brain.maintain())

    @app.get("/api/settings")
    async def settings() -> JSONResponse:
        return JSONResponse({"only_about_me": memory_service().only_about_me})

    @app.post("/api/settings")
    async def set_settings(request: Request) -> JSONResponse:
        same_origin(request)
        body = await _json_body(request)
        if "only_about_me" in body:
            await memory_service().set_only_about_me(bool(body["only_about_me"]))
        return JSONResponse({"only_about_me": memory_service().only_about_me})

    @app.post("/api/backup")
    async def backup(request: Request) -> JSONResponse:
        """Encrypted backup to a local folder (database + passphrase-wrapped key). Never uploaded."""
        from kin.memory.backup import BackupError

        same_origin(request)
        body = await _json_body(request)
        passphrase = str(body.get("passphrase") or "")
        if str(body.get("confirm") or "") != passphrase:
            raise HTTPException(400, "the two passphrases differ")
        try:
            out = await memory_service().backup(str(body.get("folder") or ""), passphrase)
        except BackupError as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse({"file": str(out), "bytes": out.stat().st_size})

    @app.post("/api/backup/check")
    async def backup_check(request: Request) -> JSONResponse:
        """Restore drill: open a backup in a temporary folder and check it, without touching the live memory."""
        from kin.memory.backup import BackupError, restore_drill

        same_origin(request)
        body = await _json_body(request)
        path = str(body.get("file") or "")
        if "://" in path or not Path(path).is_file():
            raise HTTPException(400, "not a local backup file")
        try:
            info = await asyncio.to_thread(restore_drill, path, str(body.get("passphrase") or ""))
        except (BackupError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse(info)

    @app.post("/api/backup/strength")
    async def backup_strength(request: Request) -> JSONResponse:
        from kin.memory.backup import passphrase_problem

        same_origin(request)
        body = await _json_body(request)
        return JSONResponse({"problem": passphrase_problem(str(body.get("passphrase") or ""))})

    @app.get("/api/memories/{memory_id}/history")
    async def memory_history(memory_id: int) -> JSONResponse:
        """The fact's version chain (oldest first) and the raw lines it came from (until retention purges them)."""
        svc = memory_service()
        chain = await svc.history(memory_id)
        if not chain:
            raise HTTPException(404, "no such memory")
        ids = sorted({u for f in chain for u in f.evidence})
        lines = {u.id: u.to_dict() for u in await svc.utterances(ids)}
        changes = {c.id: c for f in chain for c in await svc.fact_changes(f.id)}
        return JSONResponse({
            "chain": [f.to_dict() for f in chain],
            "evidence": [lines.get(i, {"id": i, "purged": True}) for i in ids],
            "changes": [c.to_dict() for c in sorted(changes.values(), key=lambda c: (c.ts, c.id))],
        })

    @app.get("/api/utterances")
    async def utterances(q: str | None = None, limit: int = 100) -> JSONResponse:
        svc = memory_service()
        limit = max(1, min(limit, 500))
        items = await (svc.search_utterances(q, limit) if q else svc.utterances(None, limit))
        return JSONResponse({"items": [u.to_dict() for u in items], "retention_days": kin.cfg.memory.utterance_retention_days})

    @app.get("/api/people")
    async def people() -> JSONResponse:
        return JSONResponse({"items": [p.to_dict() for p in await memory_service().people()]})

    @app.post("/api/people")
    async def add_person(request: Request) -> JSONResponse:
        same_origin(request)
        body = await _json_body(request)
        name = str(body.get("name") or "").strip()
        if not name or len(name) > 60:
            raise HTTPException(400, "name must be 1-60 characters")
        aliases = _aliases(body.get("aliases"))
        return JSONResponse((await memory_service().add_person(name, aliases, "user")).to_dict())

    @app.patch("/api/people/{person_id}")
    async def edit_person(person_id: int, request: Request) -> JSONResponse:
        """Rename (the old spelling becomes an alias) and/or replace the alias list."""
        same_origin(request)
        body = await _json_body(request)
        svc = memory_service()
        person = None
        if body.get("name"):
            name = str(body["name"]).strip()
            if not name or len(name) > 60:
                raise HTTPException(400, "name must be 1-60 characters")
            person = await svc.rename_person(person_id, name)
        if "aliases" in body:
            person = await svc.set_aliases(person_id, _aliases(body["aliases"]))
        if person is None:
            raise HTTPException(404, "no such person")
        return JSONResponse(person.to_dict())

    @app.delete("/api/people/{person_id}")
    async def delete_person(person_id: int, request: Request) -> JSONResponse:
        same_origin(request)
        if not await memory_service().delete_person(person_id):
            raise HTTPException(404, "no such person")
        return JSONResponse({"deleted": person_id})

    @app.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: int, request: Request) -> JSONResponse:
        same_origin(request)
        deleted = await memory_service().delete([memory_id], "user_delete")
        if not deleted:
            raise HTTPException(404, "no such memory")
        return JSONResponse({"deleted": deleted})

    @app.get("/api/network")
    async def network(limit: int = 100) -> JSONResponse:
        return JSONResponse({
            "gateway": {**kin.gateway_info, "offline": kin.hardware.offline},
            "network_guard": kin.snapshot()["network_guard"],
            "entries": await kin.network_log(max(1, min(limit, 500))),
        })

    def task_service():
        if kin.brain is None or kin.brain.tasks is None:
            raise HTTPException(503, "tasks are still loading")
        return kin.brain.tasks

    @app.get("/api/tasks")
    async def tasks() -> JSONResponse:
        return JSONResponse({"items": [t.to_dict() for t in await task_service().open()]})

    @app.post("/api/tasks/{task_id}/done")
    async def task_done(task_id: int, request: Request) -> JSONResponse:
        same_origin(request)
        if await task_service().complete(task_id) is None:
            raise HTTPException(404, "no such open task")
        return JSONResponse({"completed": task_id})

    @app.delete("/api/tasks/{task_id}")
    async def task_delete(task_id: int, request: Request) -> JSONResponse:
        same_origin(request)
        if not await task_service().delete(task_id):
            raise HTTPException(404, "no such task")
        return JSONResponse({"deleted": task_id})

    @app.post("/api/memories/forget")
    async def forget(request: Request, minutes: float = 10) -> JSONResponse:
        same_origin(request)
        if not 0 < minutes <= 24 * 60:
            raise HTTPException(400, "minutes must be between 0 and 1440")
        deleted = await memory_service().forget_last(minutes, "forget_button")
        return JSONResponse({"deleted": deleted})

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not origin_allowed(socket.headers.get("origin")):
            await socket.close(code=1008)
            return
        await socket.accept()
        outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=OUTBOX_LIMIT)

        def forward(event: Event) -> None:
            try:
                outbox.put_nowait(event.to_dict())
            except asyncio.QueueFull:
                if not isinstance(event, (AudioLevel, WakeLevel)):  # meters: dropping one is harmless
                    log.warning("UI outbox full; dropping %s", event.type)

        sub = kin.bus.subscribe(Event, forward, name="ui.ws")
        await socket.send_json(kin.snapshot())

        async def sender() -> None:
            while True:
                await socket.send_json(await outbox.get())

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = json.loads(await socket.receive_text())
                reply = await handle_command(kin, msg)
                if reply is not None:
                    await outbox.put(reply)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            kin.bus.unsubscribe(sub)
            send_task.cancel()

    return app


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "expected a JSON object") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "expected a JSON object")
    return body


def _aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list) or len(value) > 30:
        raise HTTPException(400, "aliases must be a list of up to 30 names")
    return [str(a).strip()[:60] for a in value if str(a).strip()]


async def handle_command(kin: KinApp, msg: dict[str, Any]) -> dict[str, Any] | None:
    cmd = msg.get("cmd")
    hw = kin.hardware
    if cmd == "snapshot":
        return kin.snapshot()
    if cmd == "profile":
        profile = msg.get("value")
        if profile not in PROFILES:
            return {"type": "Error", "message": f"unknown profile {profile!r}"}
        asyncio.create_task(kin.switch_profile(profile))
        return None
    if cmd in {"mute", "button", "offline"}:
        if not isinstance(hw, SimulatedHardware):
            return {"type": "Error", "message": "physical controls are read-only on real hardware"}
        value = bool(msg.get("value"))
        if cmd == "mute":
            hw.simulate_mute(value)
        elif cmd == "button":
            hw.simulate_button(value)
        else:
            hw.simulate_offline(value)
        return None
    if cmd == "confirm":  # Confirm / Cancel on a destructive command's card
        value = msg.get("value") or {}
        if kin.brain is None or not isinstance(value, dict) or not isinstance(value.get("id"), str):
            return {"type": "Error", "message": "nothing to confirm"}
        ok = await kin.brain.resolve_confirmation(value["id"], bool(value.get("accept")))
        return None if ok else {"type": "Error", "message": "that confirmation has expired"}
    return {"type": "Error", "message": f"unknown command {cmd!r}"}
