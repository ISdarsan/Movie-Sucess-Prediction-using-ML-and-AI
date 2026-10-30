"""Idempotent run creation and key-based recovery. All mocked: no Ollama, no network.
Run: .venv/Scripts/python -m pytest backend/tests/test_agent_idempotency.py -q
"""
import hashlib
import sqlite3
import threading
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.agent import config, ollama_client, orchestrator, store
from backend.app.main import app

BODY = {"duration": 120, "title_year": 2010, "genres": ["Drama", "Romance"], "country": "USA", "content_rating": "PG-13", "budget": 30_000_000}
H = "Idempotency-Key"


@pytest.fixture(autouse=True)
def guards(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "runs.sqlite")
    monkeypatch.setattr(config, "RESEARCH_ENABLED", False)
    def refuse(*a, **k):
        raise AssertionError("a mocked test tried to use the network")
    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setattr(ollama_client, "status", lambda timeout=3.0: {"available": False, "model": "x", "reason": "ollama_unreachable"})


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key() -> str:
    return uuid.uuid4().hex


def wait_done(client, run_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/agent/runs/{run_id}").json()
        if j["status"] in ("completed", "failed"):
            return j
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def rows():
    c = store._conn()                       # the app's own opener: creates/migrates the table, so "no runs yet" reads as []
    try:
        return c.execute("SELECT id, idem, fp FROM runs").fetchall()
    finally:
        c.close()


# ---------------------------------------------------------------- duplicate prevention
def test_same_key_and_body_returns_the_same_run_and_creates_only_one(client):
    k = key()
    a = client.post("/api/agent/runs", json=BODY, headers={H: k})
    b = client.post("/api/agent/runs", json=BODY, headers={H: k})
    assert a.status_code == b.status_code == 202
    assert a.json()["run_id"] == b.json()["run_id"] and "Idempotent-Replayed" not in a.headers and b.headers["Idempotent-Replayed"] == "true"
    wait_done(client, a.json()["run_id"])
    assert len(rows()) == 1


def test_replay_after_completion_reports_the_current_status_not_queued(client):
    k = key()
    rid = client.post("/api/agent/runs", json=BODY, headers={H: k}).json()["run_id"]
    wait_done(client, rid)
    again = client.post("/api/agent/runs", json=BODY, headers={H: k}).json()
    assert again == {"run_id": rid, "status": "completed", "poll_url": f"/api/agent/runs/{rid}"}


def test_body_key_order_and_default_fields_do_not_defeat_the_replay(client):
    k = key()
    a = client.post("/api/agent/runs", json=BODY, headers={H: k}).json()["run_id"]
    reordered = dict(reversed(list(BODY.items())))
    assert client.post("/api/agent/runs", json=reordered, headers={H: k}).json()["run_id"] == a


def test_without_a_key_behaviour_is_unchanged_and_each_post_is_a_new_run(client):
    a = client.post("/api/agent/runs", json=BODY).json()["run_id"]
    b = client.post("/api/agent/runs", json=BODY).json()["run_id"]
    assert a != b and len(a) == 32 and len(b) == 32


def test_different_keys_make_different_runs(client):
    assert client.post("/api/agent/runs", json=BODY, headers={H: key()}).json()["run_id"] != client.post("/api/agent/runs", json=BODY, headers={H: key()}).json()["run_id"]


def test_same_key_with_different_inputs_is_a_conflict_and_creates_nothing(client):
    k = key()
    first = client.post("/api/agent/runs", json=BODY, headers={H: k}).json()["run_id"]
    r = client.post("/api/agent/runs", json={**BODY, "duration": 90}, headers={H: k})
    assert r.status_code == 409 and r.json()["error"] == "idempotency_conflict"
    wait_done(client, first)
    assert len(rows()) == 1
    assert client.post("/api/agent/runs", json={**BODY, "research_title": "Inception"}, headers={H: k}).status_code == 409     # the research title is part of the request


@pytest.mark.parametrize("bad", ["short", "x" * 65, "has space in it 123456", "semi;colon;1234567890", "dotted.key.1234567890ab", "a/b/c/d/e/f/g/h/i/j"])
def test_malformed_keys_are_rejected(bad, client):
    r = client.post("/api/agent/runs", json=BODY, headers={H: bad})
    assert r.status_code == 422 and r.json()["error"] == "invalid_idempotency_key" and rows() == []


def test_invalid_body_with_a_key_still_creates_nothing(client):
    assert client.post("/api/agent/runs", json={**BODY, "gross": 5}, headers={H: key()}).status_code == 422
    assert rows() == []


def test_a_replay_is_not_blocked_by_the_busy_limit(client, monkeypatch):
    k = key()
    first = client.post("/api/agent/runs", json=BODY, headers={H: k}).json()["run_id"]
    wait_done(client, first)
    monkeypatch.setattr(store, "active_count", lambda: 999)
    assert client.post("/api/agent/runs", json=BODY, headers={H: k}).json()["run_id"] == first          # creates nothing, so it is not "busy"
    assert client.post("/api/agent/runs", json=BODY, headers={H: key()}).status_code == 429             # a genuinely new run still is


def test_simultaneous_creation_with_one_key_yields_exactly_one_run():
    k = hashlib.sha256(key().encode()).hexdigest()
    ids, created = [], []
    barrier = threading.Barrier(8)
    def go(i):
        barrier.wait()
        v, c = store.create_idempotent(f"{i:032x}", {"n": i}, k, "fp")
        ids.append(v["id"]); created.append(c)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sum(created) == 1 and len(set(ids)) == 1 and len(rows()) == 1


def test_orchestrator_submit_does_not_queue_a_second_job_for_a_replay(monkeypatch):
    queued = []
    monkeypatch.setattr(orchestrator._pool, "submit", lambda fn, *a, **k: queued.append(a))
    k = hashlib.sha256(b"k").hexdigest()
    r1 = orchestrator.submit({"a": 1}, k, "fp")
    r2 = orchestrator.submit({"a": 1}, k, "fp")
    assert r1[1] is True and r2 == (r1[0], False) and len(queued) == 1


# ---------------------------------------------------------------- privacy of what is stored
def test_only_hashes_are_stored_never_the_key(client):
    k = key()
    client.post("/api/agent/runs", json=BODY, headers={H: k})
    raw = config.DB_PATH.read_bytes()
    assert k.encode() not in raw and hashlib.sha256(k.encode()).hexdigest().encode() in raw
    (rid, idem, fp), = rows()
    assert len(idem) == 64 and len(fp) == 64


# ---------------------------------------------------------------- recovery by key
def test_lost_response_is_recoverable_by_key_without_resending_the_inputs(client):
    k = key()
    rid = client.post("/api/agent/runs", json=BODY, headers={H: k}).json()["run_id"]       # imagine this response never arrived
    r = client.get(f"/api/agent/runs/by-key/{k}")
    assert r.status_code == 200 and r.json()["run_id"] == rid and r.json()["poll_url"] == f"/api/agent/runs/{rid}"
    assert set(r.json()) == {"run_id", "status", "poll_url"}                                 # no inputs, no result, no research title
    wait_done(client, rid)
    assert client.get(f"/api/agent/runs/by-key/{k}").json()["status"] == "completed"


def test_unknown_key_means_the_submission_never_arrived(client):
    r = client.get(f"/api/agent/runs/by-key/{key()}")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


@pytest.mark.parametrize("bad", ["short", "x" * 65, "bad key with spaces 1234", "%2e%2e%2f%2e%2e%2fetc"])
def test_malformed_lookup_keys_are_404_not_errors(bad, client):
    assert client.get(f"/api/agent/runs/by-key/{bad}").status_code == 404


def test_one_runs_key_does_not_reveal_another_and_runs_cannot_be_listed(client):
    a, b = key(), key()
    ra = client.post("/api/agent/runs", json=BODY, headers={H: a}).json()["run_id"]
    rb = client.post("/api/agent/runs", json={**BODY, "duration": 100}, headers={H: b}).json()["run_id"]
    assert client.get(f"/api/agent/runs/by-key/{a}").json()["run_id"] == ra != rb
    assert client.get("/api/agent/runs").status_code in (404, 405) and client.get("/api/agent/runs/").status_code in (404, 405)
    wait_done(client, ra); wait_done(client, rb)


def test_the_run_id_is_not_derivable_from_the_key(client):
    k = key()
    rid = client.post("/api/agent/runs", json=BODY, headers={H: k}).json()["run_id"]
    assert rid != k and rid != hashlib.sha256(k.encode()).hexdigest()[:32] and rid != hashlib.md5(k.encode()).hexdigest()


# ---------------------------------------------------------------- ids and migration
def test_new_ids_are_128_bit_and_old_12_char_ids_still_read(client):
    rid = client.post("/api/agent/runs", json=BODY).json()["run_id"]
    assert len(rid) == 32 and int(rid, 16) >= 0
    done = wait_done(client, rid)
    assert done["status"] == "completed"
    store.create("abcdef012345", {"legacy": True})
    assert client.get("/api/agent/runs/abcdef012345").status_code == 200
    for bad in ("abcdef01234", "abcdef0123456", "ABCDEF012345", "g" * 12, "a" * 31, "a" * 33):
        assert client.get(f"/api/agent/runs/{bad}").status_code == 404


def test_a_database_created_before_this_change_is_migrated_in_place():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.DB_PATH) as c:
        c.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, created REAL, status TEXT, request TEXT, result TEXT)")
        c.execute("INSERT INTO runs VALUES ('abcdef012345', 1.0, 'completed', '{}', '{\"x\": 1}')")
    assert store.get("abcdef012345")["status"] == "completed"
    v, created = store.create_idempotent("f" * 32, {"a": 1}, "h" * 64, "fp")
    assert created and store.find_by_idem("h" * 64)["id"] == "f" * 32 and store.get("abcdef012345")["result"] == {"x": 1}
    store.create("e" * 32, {"b": 2})                      # plain creation still works on the migrated table


def test_cors_allows_the_idempotency_header_and_exposes_nothing_new(client):
    r = client.options("/api/agent/runs", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "idempotency-key,content-type"})
    assert r.status_code == 200 and "idempotency-key" in r.headers["access-control-allow-headers"].lower() or r.headers["access-control-allow-headers"] == "*"


def test_existing_predict_endpoint_ignores_the_header_and_is_unchanged(client):
    a = client.post("/api/predict", json=BODY).json()
    b = client.post("/api/predict", json=BODY, headers={H: key()}).json()
    assert a == b and set(a) == {"label", "probabilities", "inputs", "imputed", "warnings", "model", "caveats"}
