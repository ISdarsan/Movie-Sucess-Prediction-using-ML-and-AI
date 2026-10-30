"""/api/agent/* routes. Additive: no existing route is changed. Disabled entirely with FRAME_AGENT_ENABLED=0."""
from __future__ import annotations

import hashlib
import json
import re

from fastapi import APIRouter, Header, HTTPException, Response

from ..app.model import service
from . import config, ollama_client, orchestrator, store
from .schemas import AgentRunRequest

router = APIRouter(prefix="/api/agent", tags=["agent"])
_ID = re.compile(r"^(?:[0-9a-f]{12}|[0-9a-f]{32})$")      # 12 = runs created before ids became 128-bit; both stay readable
_KEY = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_sha = lambda s: hashlib.sha256(s.encode()).hexdigest()   # noqa: E731


def _err(status: int, error: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": error, "message": message})


@router.get("/status")
def agent_status():
    return {"enabled": True, "llm": ollama_client.status(), "max_tool_calls": config.MAX_TOOL_CALLS,
            "research": {"enabled": config.RESEARCH_ENABLED, "provider": "wikipedia", "max_requests": config.RESEARCH_MAX_REQUESTS},
            "note": "If the local language model is unavailable, runs still complete with a deterministic analysis."}


def _accepted(run_id: str, status: str) -> dict:
    return {"run_id": run_id, "status": status, "poll_url": f"/api/agent/runs/{run_id}"}


@router.post("/runs", status_code=202)
def create_run(req: AgentRunRequest, response: Response, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    """Start an analysis. Accepts the POST /api/predict body (pre-release inputs only; any other extra field is rejected with 422)
    plus an optional research_title that is used only for contextual research.

    Optional `Idempotency-Key` header (16-64 chars of A-Z a-z 0-9 _ -): repeating a request with the same key and body returns the run it
    created instead of making another, so a client that lost the response can safely retry. The same key with a different body is a 409.
    Without the header nothing changes. Only SHA-256 hashes of the key and of the request are stored."""
    if not service.ready:
        raise _err(503, "model_unavailable", "The prediction model is not available right now.")
    payload = req.model_dump()
    idem = fp = None
    if idempotency_key is not None:
        if not _KEY.match(idempotency_key):
            raise _err(422, "invalid_idempotency_key", "Idempotency-Key must be 16 to 64 characters: letters, numbers, underscore or hyphen.")
        idem, fp = _sha(idempotency_key), _sha(json.dumps(payload, sort_keys=True, default=str))
        known = store.find_by_idem(idem)
        if known:                                     # a replay never counts against the busy limit: it creates nothing
            return _replay(known, fp, response)
    if store.active_count() >= orchestrator.MAX_ACTIVE:
        raise _err(429, "busy", "Too many analyses are in progress. Try again shortly.")
    run_id, created = orchestrator.submit(payload, idem, fp)
    if not created:                                   # lost a race with a simultaneous request carrying the same key
        return _replay(store.find_by_idem(idem), fp, response)
    return _accepted(run_id, "queued")


def _replay(known: dict, fp: str, response: Response) -> dict:
    if known["fp"] != fp:
        raise _err(409, "idempotency_conflict", "That Idempotency-Key was already used with different inputs.")
    response.headers["Idempotent-Replayed"] = "true"
    return _accepted(known["id"], known["status"])


@router.get("/runs/by-key/{key}")
def run_by_key(key: str):
    """Recover the run created under an Idempotency-Key whose response was lost (e.g. the page was refreshed mid-request).
    Returns only the run id and status, never inputs or results. Knowing the key is the only credential; runs cannot be listed."""
    known = store.find_by_idem(_sha(key)) if _KEY.match(key) else None
    if known is None:
        raise _err(404, "not_found", "No run was created for that key.")
    return _accepted(known["id"], known["status"])


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    if not _ID.match(run_id):
        raise _err(404, "not_found", "No such run.")
    row = store.get(run_id)
    if row is None:
        raise _err(404, "not_found", "No such run.")
    return {"run_id": row["id"], "status": row["status"], **(row["result"] or {}), "created": row["created"]}
