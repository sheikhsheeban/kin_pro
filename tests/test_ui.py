"""Web UI: snapshot, commands over the WebSocket, origin checks."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from kin.ui.server import create_app, origin_allowed
from tests.conftest import build_fake_app


def test_origin_check():
    assert origin_allowed(None)
    assert origin_allowed("http://localhost:8000")
    assert origin_allowed("http://127.0.0.1:8000")
    assert not origin_allowed("https://evil.example")
    assert not origin_allowed("http://localhost.evil.example")


@pytest.fixture
def harness():
    # The app must live on the TestClient's event loop, so it is started by the app lifespan.
    return build_fake_app()


@pytest.fixture
def client(harness):
    with TestClient(create_app(harness.app)) as c:
        yield c


def test_index_and_state(client, harness):
    assert "Simulated device" in client.get("/").text
    state = client.get("/api/state").json()
    assert state["profile"] == "laptop" and state["stream_open"] is True and state["led"] == "green"


def test_foreign_origin_websocket_is_rejected(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()


def _drain_until(ws, type_, pred=lambda m: True, limit=200):
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == type_ and pred(m):
            return m
    raise AssertionError(f"no {type_}")


def test_websocket_mute_and_button_commands(client, harness):
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8000"}) as ws:
        snap = ws.receive_json()
        assert snap["type"] == "Snapshot" and snap["muted"] is False

        ws.send_json({"cmd": "button", "value": True})
        act = _drain_until(ws, "ActivationEvent")
        assert act["source"] == "button"
        ws.send_json({"cmd": "button", "value": False})
        _drain_until(ws, "AttendingEnded")

        ws.send_json({"cmd": "mute", "value": True})
        _drain_until(ws, "StreamStateChanged", lambda m: m["open"] is False)
        _drain_until(ws, "LedChanged", lambda m: m["color"] == "off")
        assert harness.stream.closed

        ws.send_json({"cmd": "nonsense"})
        assert _drain_until(ws, "Error")["message"].startswith("unknown command")


# --- Phase 2: memory and system API ------------------------------------------------------


@pytest.fixture
def brain_client(tmp_path):
    from tests.conftest import make_fake_brain_factory

    store: dict = {}
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, store=store))
    with TestClient(create_app(h.app)) as c:
        c.portal.call(h.app.wait_brain)
        mem = h.app.brain.memory
        c.portal.call(mem.remember, "I'm meeting Priya at 4 pm at the library cafe.", "commitment", 0.9)
        c.portal.call(mem.remember, "Rahul is allergic to peanuts.", "fact_about_person", 0.85)
        yield c


LOCAL = {"origin": "http://127.0.0.1:8000"}


def test_memory_list_and_search(brain_client):
    data = brain_client.get("/api/memories").json()
    assert data["total"] == 2 and data["items"][0]["text"].startswith("Rahul")
    hits = brain_client.get("/api/memories", params={"q": "meeting Priya"}).json()["items"]
    assert hits[0]["text"].startswith("I'm meeting Priya") and hits[0]["similarity"] > 0.5
    text = brain_client.get("/api/memories", params={"q": "peanut", "mode": "text"}).json()["items"]
    assert [m["text"] for m in text] == ["Rahul is allergic to peanuts."]


def test_memory_delete_and_forget(brain_client):
    first = brain_client.get("/api/memories").json()["items"][0]
    assert brain_client.delete(f"/api/memories/{first['id']}", headers=LOCAL).json()["deleted"] == [first["id"]]
    assert brain_client.delete(f"/api/memories/{first['id']}", headers=LOCAL).status_code == 404
    assert len(brain_client.post("/api/memories/forget?minutes=10", headers=LOCAL).json()["deleted"]) == 1
    assert brain_client.get("/api/memories").json()["total"] == 0
    assert brain_client.post("/api/memories/forget?minutes=0", headers=LOCAL).status_code == 400


def test_cross_site_requests_cannot_change_memory(brain_client):
    evil = {"origin": "https://evil.example"}
    mid = brain_client.get("/api/memories").json()["items"][0]["id"]
    assert brain_client.delete(f"/api/memories/{mid}", headers=evil).status_code == 403
    assert brain_client.post("/api/memories/forget", headers=evil).status_code == 403
    assert brain_client.get("/api/memories").json()["total"] == 2


def test_fact_history_evidence_and_people_api(brain_client):
    old = brain_client.get("/api/memories").json()["items"][-1]
    hist = brain_client.get(f"/api/memories/{old['id']}/history").json()
    assert [f["id"] for f in hist["chain"]] == [old["id"]] and hist["evidence"] == []
    assert brain_client.get("/api/memories/9999/history").status_code == 404

    p = brain_client.post("/api/people", json={"name": "Pria"}, headers=LOCAL).json()
    renamed = brain_client.patch(f"/api/people/{p['id']}", json={"name": "Priya"}, headers=LOCAL).json()
    assert renamed["name"] == "Priya" and renamed["aliases"] == ["Pria"], "the old spelling becomes an alias"
    edited = brain_client.patch(f"/api/people/{p['id']}", json={"aliases": "Pria, Preeya"}, headers=LOCAL).json()
    assert edited["aliases"] == ["Pria", "Preeya"]
    assert brain_client.get("/api/people").json()["items"][0]["name"] == "Priya"
    assert brain_client.post("/api/people", json={"name": ""}, headers=LOCAL).status_code == 400
    evil = {"origin": "https://evil.example"}
    assert brain_client.post("/api/people", json={"name": "Mallory"}, headers=evil).status_code == 403
    assert brain_client.patch(f"/api/people/{p['id']}", json={"name": "X"}, headers=evil).status_code == 403
    assert brain_client.delete(f"/api/people/{p['id']}", headers=evil).status_code == 403
    assert brain_client.delete(f"/api/people/{p['id']}", headers=LOCAL).json() == {"deleted": p["id"]}
    assert brain_client.get("/api/people").json()["items"] == []
    data = brain_client.get("/api/memories", params={"history": 1}).json()
    assert data["superseded"] == 0 and "episodes" in data
    assert brain_client.get("/api/utterances").json()["retention_days"] == 14


def test_system_stats(brain_client):
    s = brain_client.get("/api/system").json()
    assert s["rss_mb"] > 0 and s["brain"] == "ready"
    assert "embedder + memory" in s["components"] and "speech (VAD, wake word, STT)" in s["components"]


# --- Phase 3: network monitor and tasks API ----------------------------------------------------------


def test_network_and_tasks_api(brain_client):
    net = brain_client.get("/api/network").json()
    assert net["gateway"]["running"] is False and net["entries"] == []
    assert brain_client.get("/api/tasks").json()["items"] == []


def test_tasks_api_complete_delete_and_csrf(tmp_path):
    from tests.conftest import make_fake_brain_factory

    store: dict = {}
    h = build_fake_app(brain_factory=make_fake_brain_factory(tmp_path, store=store))
    with TestClient(create_app(h.app)) as c:
        c.portal.call(h.app.wait_brain)
        tasks = h.app.brain.tasks
        a = c.portal.call(tasks.add, "Buy oat milk")
        b = c.portal.call(tasks.add, "Email the professor", 2_000_000_000.0)
        items = c.get("/api/tasks").json()["items"]
        assert [t["text"] for t in items] == ["Email the professor", "Buy oat milk"]
        evil = {"origin": "https://evil.example"}
        assert c.post(f"/api/tasks/{a.id}/done", headers=evil).status_code == 403
        assert c.delete(f"/api/tasks/{b.id}", headers=evil).status_code == 403
        assert c.post(f"/api/tasks/{a.id}/done", headers=LOCAL).status_code == 200
        assert c.delete(f"/api/tasks/{b.id}", headers=LOCAL).status_code == 200
        assert c.get("/api/tasks").json()["items"] == []


# --- routing refactor: router stats and the confirm card ------------------------------------------------


def test_system_stats_show_the_passive_router(brain_client):
    r = brain_client.get("/api/system").json()["router"]
    assert r["mode"] == "semantic" and r["pending"] == 0 and r["recent"] == []
    assert 0 < r["accept_score"] < 1 and 0 < r["accept_margin"] < 1


async def test_confirm_command_goes_to_the_brain():
    from kin.ui.server import handle_command

    calls = []

    class Brain:
        async def resolve_confirmation(self, confirm_id, accept):
            calls.append((confirm_id, accept))
            return confirm_id == "abc"

    class Kin:
        brain = Brain()
        hardware = None

    assert await handle_command(Kin(), {"cmd": "confirm", "value": {"id": "abc", "accept": True}}) is None
    assert (await handle_command(Kin(), {"cmd": "confirm", "value": {"id": "old", "accept": True}}))["type"] == "Error"
    assert (await handle_command(Kin(), {"cmd": "confirm", "value": "abc"}))["type"] == "Error"  # malformed
    assert calls == [("abc", True), ("old", True)]
