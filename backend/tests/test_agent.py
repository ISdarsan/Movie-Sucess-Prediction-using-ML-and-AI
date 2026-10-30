"""Agent layer tests. The local LLM is faked, so these run without Ollama. Run: .venv/Scripts/python -m pytest backend/tests -q"""
import time

import pytest
from fastapi.testclient import TestClient

import copy

import httpx

from backend.agent import config, facts as factsmod, ollama_client, orchestrator, report, store
from backend.agent.tools import ComparablesArgs, ToolError, ToolRegistry, PUBLIC_COLUMNS
from backend.app.main import app

BODY = {"duration": 120, "title_year": 2010, "genres": ["Drama", "Romance"], "country": "USA", "content_rating": "PG-13", "budget": 30_000_000}


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "runs.sqlite")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait_done(client, run_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/agent/runs/{run_id}").json()
        if j["status"] in ("completed", "failed"):
            return j
        time.sleep(0.2)
    raise AssertionError("run did not finish")


# ---------------------------------------------------------------- tools
def test_tool_args_are_coerced_from_model_style_strings():
    t = ToolRegistry()
    out = t.call("get_cohort", {"genres": "['Drama']", "year": "2005"})   # exactly what llama3.1:8b returned in the probe
    assert out["final"]["n"] > 0


def test_unknown_tool_and_invalid_args_and_budget():
    t = ToolRegistry(max_calls=1)
    with pytest.raises(ToolError) as e:
        t.call("delete_everything", {})
    assert e.value.kind == "unknown_tool"
    with pytest.raises(ToolError) as e:
        t.call("get_cohort", {"genres": ["Drama"], "gross": 5})            # extra field rejected
    assert e.value.kind == "invalid_arguments"
    t.call("get_cohort", {"genres": ["Drama"]})
    with pytest.raises(ToolError) as e:
        t.call("get_cohort", {"genres": ["Drama"]})
    assert e.value.kind == "tool_budget_exceeded"


def test_comparables_are_earlier_and_expose_only_public_columns():
    rows = ToolRegistry().call("get_comparables", {"genres": ["Drama"], "year": 1995, "duration": 110})
    assert 0 < len(rows) <= 8
    for r in rows:
        assert set(r) == set(PUBLIC_COLUMNS)
        assert r["year"] <= 1995
    banned = {"gross", "imdb_score", "num_voted_users", "num_critic_for_reviews", "movie_facebook_likes", "plot_keywords"}
    assert not banned & set().union(*[set(r) for r in rows])


def test_comparables_limit_is_capped():
    with pytest.raises(Exception):
        ComparablesArgs(genres=["Drama"], limit=50)


