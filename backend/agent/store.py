"""Tiny SQLite run store (standard library). One short-lived connection per call, so it is safe across worker threads."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from typing import Any

from . import config


def _conn() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH, timeout=10)
    c.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL, status TEXT, request TEXT, result TEXT)")
    # additive migration for idempotent creation: only HASHES are stored (of the client's key and of the validated request), never the key itself
    cols = {r[1] for r in c.execute("PRAGMA table_info(runs)")}
    for col in ("idem", "fp"):
        if col not in cols:
            c.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS runs_idem ON runs(idem) WHERE idem IS NOT NULL")
    c.commit()
    return c


def create(run_id: str, request: dict) -> None:
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO runs (id, created, status, request, result) VALUES (?,?,?,?,?)", (run_id, time.time(), "queued", json.dumps(request), None))


def find_by_idem(idem: str) -> dict[str, Any] | None:
    """Minimal view of the run created under this key hash: no inputs, no result."""
    with closing(_conn()) as c:
        row = c.execute("SELECT id, status, fp FROM runs WHERE idem=?", (idem,)).fetchone()
    return {"id": row[0], "status": row[1], "fp": row[2]} if row else None


def create_idempotent(run_id: str, request: dict, idem: str, fp: str) -> tuple[dict[str, Any], bool]:
    """Create the run unless one already exists for this key hash. The check and the insert share one write transaction, so two
    simultaneous requests with the same key can never both create a run. Returns (run view, created)."""
    with closing(_conn()) as c:
        c.isolation_level = None
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute("SELECT id, status, fp FROM runs WHERE idem=?", (idem,)).fetchone()
            if row:
                c.execute("COMMIT")
                return {"id": row[0], "status": row[1], "fp": row[2]}, False
            c.execute("INSERT INTO runs (id, created, status, request, result, idem, fp) VALUES (?,?,?,?,?,?,?)",
                      (run_id, time.time(), "queued", json.dumps(request), None, idem, fp))
            c.execute("COMMIT")
            return {"id": run_id, "status": "queued", "fp": fp}, True
        except Exception:
            c.execute("ROLLBACK")
            raise


def save(run_id: str, status: str, result: dict | None = None) -> None:
    with closing(_conn()) as c, c:
        c.execute("UPDATE runs SET status=?, result=COALESCE(?, result) WHERE id=?", (status, json.dumps(result) if result is not None else None, run_id))


def get(run_id: str) -> dict[str, Any] | None:
    with closing(_conn()) as c:
        row = c.execute("SELECT id, created, status, request, result FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    return {"id": row[0], "created": row[1], "status": row[2], "request": json.loads(row[3]), "result": json.loads(row[4]) if row[4] else None}


def active_count() -> int:
    with closing(_conn()) as c:
        return c.execute("SELECT COUNT(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0]


def fail_stale() -> int:
    """Runs left queued/running by a previous process can never finish; mark them failed."""
    with closing(_conn()) as c, c:
        return c.execute("UPDATE runs SET status='failed' WHERE status IN ('queued','running')").rowcount
