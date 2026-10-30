"""Phase 3 agent tests: strict validation, claim verification, fallback, and the API contract.
EVERYTHING HERE IS MOCKED (no Ollama needed). The one real-model smoke test lives outside the suite.
Run: .venv/Scripts/python -m pytest backend/tests/test_agent_report.py -q
"""
import copy
import json
import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.agent import config, ollama_client, report, store
from backend.app.main import app

BODY = {"duration": 120, "title_year": 2010, "genres": ["Drama", "Romance"], "country": "USA", "content_rating": "PG-13", "budget": 30_000_000}
PRED = {"label": "Hit", "probabilities": {"Flop": 0.0213, "Average": 0.3412, "Hit": 0.6375},
        "inputs": {"duration": 120.0, "title_year": 2010, "genres": ["Drama", "Romance"], "country": "USA", "content_rating": "PG-13"}}
COMPS = [{"title": "Alpha", "year": 2001, "Classify": "Hit"}, {"title": "Beta", "year": 1999, "Classify": "Average"}, {"title": "Gamma", "year": 2005, "Classify": "Hit"}]


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "runs.sqlite")


@pytest.fixture(autouse=True)
def never_call_real_ollama(monkeypatch):
    """Everything in this file is mocked: fail loudly if any test would reach the real model. Tests that need httpx behaviour patch httpx.post."""
    def refuse(*a, **k):
        raise AssertionError("a mocked test reached the real Ollama client")
    monkeypatch.setattr(httpx, "post", refuse)       # the client's own failure-mapping tests install their own fake httpx.post
    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setattr(ollama_client, "status", lambda timeout=3.0: {"available": False, "model": "x", "reason": "ollama_unreachable"})
    monkeypatch.setattr(config, "RESEARCH_ENABLED", False)     # research has its own mocked tests; these files must never touch the network


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


def make_evidence(changes=False):
    seg = ([{"from": 60, "to": 200, "label": "Hit"}] if not changes
           else [{"from": 60, "to": 110, "label": "Average"}, {"from": 111, "to": 200, "label": "Hit"}])
    return {"cohort": {"final": {"n": 125, "counts": {"Flop": 1, "Average": 36, "Hit": 88}, "mean_score": 6.31}, "thin": False, "empty": False},
            "sensitivity": {"label": "Runtime (min)", "top_class_by_range": seg}, "comparables": COMPS}


@pytest.fixture
def ctx():
    f = report.build_facts(PRED, make_evidence())
    ids = {x.ref: x.id for x in reversed(f)}          # first fact of each ref
    return f, report.template_analysis(PRED, f), ids


def good(ids):
    return {"summary": "The model ranks Hit highest at 64%.",
            "drivers": [{"claim": "The saved Random Forest ranks Hit highest at 64%, then Average at 34% and Flop at 2%.", "evidence_ref": "prediction", "fact_ids": [ids["prediction"]]},
                        {"claim": "125 films in the dataset match: 88 Hit, 36 Average, 1 Flop.", "evidence_ref": "cohort", "fact_ids": [ids["cohort"]]}],
            }


def chk(raw, ctx):
    return report.check(raw, ctx[0], PRED, ctx[1])


def with_driver(ctx, claim, ref="sensitivity", fid=None):
    raw = good(ctx[2])
    raw["drivers"].append({"claim": claim, "evidence_ref": ref, "fact_ids": [fid or ctx[2][ref]]})
    return raw


# ---------------------------------------------------------------- facts, template, happy path
def test_good_output_passes_untouched(ctx):
    o = chk(good(ctx[2]), ctx)
    assert o["status"] == "ok" and o["guardrails"]["passed"] and o["guardrails"]["replaced"] == []
    assert o["analysis"]["summary"] == "The model ranks Hit highest at 64%."


def test_llm_supplied_risks_are_ignored_not_used(ctx):
    raw = good(ctx[2])
    raw["risks"] = ["The cohort is empty and these films share the same genre."]            # invented by a "model"
    o = chk(raw, ctx)
    assert o["status"] == "ok" and "risks" not in o["analysis"] and o["guardrails"]["llm_risks_ignored"] is True
    assert "empty" not in json.dumps(o["analysis"])


def test_template_passes_its_own_checker(ctx):
    o = report.check(ctx[1], ctx[0], PRED, ctx[1])
    assert o["status"] == "ok" and o["guardrails"]["passed"], o["guardrails"]


def test_facts_contain_exact_scores_counts_and_comparables(ctx):
    text = " ".join(x.text for x in ctx[0])
    for needle in ("64%", "34%", "2%", "125", "88 Hit", "Alpha (2001)", "does not change the predicted class"):
        assert needle in text


def test_analysis_schema_matches_pydantic_model():
    assert set(report.ANALYSIS_SCHEMA["properties"]) == set(report.AnalysisModel.model_fields)
    assert set(report.ANALYSIS_SCHEMA["properties"]["drivers"]["items"]["properties"]) == set(report.DriverModel.model_fields)
    assert set(report.ANALYSIS_SCHEMA["required"]) == set(report.AnalysisModel.model_fields)


# ---------------------------------------------------------------- strict schema validation
D1 = {"claim": "c", "evidence_ref": "prediction", "fact_ids": ["F1"]}


@pytest.mark.parametrize("raw", [
    {}, [], None, "text", 42,
    {"drivers": [D1], "risks": []},                                                              # missing summary
    {"summary": "", "drivers": [D1], "risks": []},
    {"summary": "   ", "drivers": [D1], "risks": []},
    {"summary": "s", "drivers": "none", "risks": []},                                            # drivers as a string
    {"summary": "s", "drivers": [], "risks": []},                                                # empty drivers
    {"summary": "s", "drivers": ["a claim"], "risks": []},                                       # driver is a string
    {"summary": "s", "drivers": [{**D1, "claim": 5}], "risks": []},                              # wrong field type
    {"summary": 7, "drivers": [D1], "risks": []},
    {"summary": "s", "drivers": [{"claim": "c", "evidence_ref": "prediction"}], "risks": []},    # missing fact_ids
    {"summary": "s", "drivers": [{**D1, "fact_ids": []}], "risks": []},
    {"summary": "s", "drivers": [{**D1, "fact_ids": "F1"}], "risks": []},
    {"summary": "s", "drivers": [{**D1, "evidence_ref": "gross"}], "risks": []},                 # invalid evidence_ref
    {"summary": "s", "drivers": [{**D1, "x": 1}], "risks": []},                                  # extra key
    {"summary": "s", "extra": 1, "drivers": [D1], "risks": []},
    {"summary": "s", "drivers": [D1, D1, D1, D1], "risks": []},                                  # too many drivers
])
def test_malformed_shapes_fall_back_to_template_and_never_raise(raw, ctx):
    o = chk(raw, ctx)
    assert o["status"] == "fallback" and o["reason"] == "invalid_schema" and o["analysis"] == ctx[1]
    assert o["guardrails"]["schema_valid"] is False and o["guardrails"]["passed"] is False


# ---------------------------------------------------------------- claim verification
def test_unsupported_direction_claim_is_replaced_with_safe_wording(ctx):
    o = chk(with_driver(ctx, "A longer runtime is associated with hits."), ctx)
    assert o["status"] == "partial" and "driver 3" in o["guardrails"]["replaced"]
    sens = [d for d in o["analysis"]["drivers"] if d["evidence_ref"] == "sensitivity"]
    assert sens and "associated" not in sens[0]["claim"] and "does not change the predicted class" in sens[0]["claim"]
    assert any("direction/causal" in c for c in o["guardrails"]["unsupported_claims"])


@pytest.mark.parametrize("claim", ["Higher runtime increases the Hit score.", "Because of its runtime the film does better.",
                                   "Runtime influences the outcome.", "Hit is more likely for longer films."])
def test_causal_and_direction_vocabulary_rejected(claim, ctx):
    assert "driver 3" in chk(with_driver(ctx, claim), ctx)["guardrails"]["replaced"]


def test_sensitivity_claim_contradicting_computed_flag_is_rejected(ctx):
    assert "driver 3" in chk(with_driver(ctx, "Changing the runtime changes the top class."), ctx)["guardrails"]["replaced"]
    ok = chk(with_driver(ctx, "Sweeping runtime does not change the top class, which stays Hit."), ctx)
    assert ok["guardrails"]["replaced"] == [] and ok["status"] == "ok"


def test_sensitivity_claim_that_denies_a_real_change_is_rejected():
    f = report.build_facts(PRED, make_evidence(changes=True))
    tmpl = report.template_analysis(PRED, f)
    sid = next(x.id for x in f if x.ref == "sensitivity")
    pid = next(x.id for x in f if x.ref == "prediction")
    raw = {"summary": "Hit ranks highest at 64%.", "risks": [],
           "drivers": [{"claim": "Hit ranks highest at 64%.", "evidence_ref": "prediction", "fact_ids": [pid]},
                       {"claim": "Runtime does not change the top class.", "evidence_ref": "sensitivity", "fact_ids": [sid]}]}
    assert "driver 2" in report.check(raw, f, PRED, tmpl)["guardrails"]["replaced"]


def test_invalid_evidence_references(ctx):
    assert "driver 3" in chk(with_driver(ctx, "Runtime sweep result.", fid="F99"), ctx)["guardrails"]["replaced"]          # unknown id
    mism = chk(with_driver(ctx, "125 films match.", ref="cohort", fid=ctx[2]["prediction"]), ctx)                           # id from another ref
    assert "driver 3" in mism["guardrails"]["replaced"] and any("do not belong" in c for c in mism["guardrails"]["unsupported_claims"])


def test_invented_numbers_rejected_in_summary_driver_and_risk(ctx):
    raw = good(ctx[2])
    raw["summary"] = "Hit is 90% certain."
    raw["drivers"][1]["claim"] = "999 films in the dataset match."
    o = chk(raw, ctx)
    assert o["status"] == "partial" and set(o["guardrails"]["replaced"]) == {"summary", "driver 2"}
    assert o["guardrails"]["numbers_verified"] is False
    blob = str(o["analysis"])
    assert "90%" not in blob and "999" not in blob


@pytest.mark.parametrize("text,accepted", [("Hit scores 64%.", True), ("Hit scores 0.64.", True),
                                           ("Hit scores 65%.", False), ("Hit scores 0.65.", False),
                                           ("Hit scores 63.8%.", False), ("Hit scores 0.6375.", False)])   # finer precision than the facts show is not in the evidence
def test_percentage_conversions(text, accepted, ctx):
    raw = good(ctx[2])
    raw["drivers"][0]["claim"] = text
    assert ("driver 1" not in chk(raw, ctx)["guardrails"]["replaced"]) is accepted


@pytest.mark.parametrize("claim", ["125 films match: 88 Average, 36 Hit, 1 Flop.", "Average scores 64%.", "The model gives Flop 64%."])
def test_class_number_mismatch_is_rejected(claim, ctx):
    raw = good(ctx[2])
    raw["drivers"][1] = {"claim": claim, "evidence_ref": "cohort", "fact_ids": [ctx[2]["cohort"]]} if "films" in claim else \
                        {"claim": claim, "evidence_ref": "prediction", "fact_ids": [ctx[2]["prediction"]]}
    assert chk(raw, ctx)["guardrails"]["replaced"], claim


@pytest.mark.parametrize("claim", ["This film will be profitable.", "It should do well at the box office.",
                                   "Expect strong revenue and a guaranteed win.", "A good return on investment."])
def test_financial_claims_rejected(claim, ctx):
    raw = good(ctx[2])
    raw["summary"] = claim
    o = chk(raw, ctx)
    assert "summary" in o["guardrails"]["replaced"] and o["guardrails"]["forbidden_terms"]
    assert claim not in o["analysis"]["summary"]


def test_negated_financial_wording_is_allowed(ctx):
    raw = good(ctx[2])
    raw["summary"] = "Hit ranks highest at 64%; this is not a profit forecast."
    assert "summary" not in chk(raw, ctx)["guardrails"]["replaced"]


def test_all_drivers_failing_falls_back_whole(ctx):
    raw = good(ctx[2])
    for d in raw["drivers"]:
        d["claim"] = "This is guaranteed to be profitable."
    o = chk(raw, ctx)
    assert o["status"] == "fallback" and o["reason"] == "failed_verification" and o["analysis"] == ctx[1]


def test_generic_output_without_any_figures_is_not_accepted(ctx):
    raw = {"summary": "Movie classification result", "risks": [],
           "drivers": [{"claim": "The movie is a Hit", "evidence_ref": "prediction", "fact_ids": [ctx[2]["prediction"]]}]}
    o = chk(raw, ctx)
    assert o["status"] == "fallback" and o["reason"] == "not_specific"


def test_comparable_film_title_counts_as_specific(ctx):
    fid = next(x.id for x in ctx[0] if x.text.startswith("Alpha"))
    raw = {"summary": "Hit ranks highest at 64%.", "risks": [],
           "drivers": [{"claim": "Alpha was classed Hit.", "evidence_ref": "comparables", "fact_ids": [fid]}]}
    assert chk(raw, ctx)["status"] in ("ok", "partial")


# ---------------------------------------------------------------- Ollama client failure mapping (httpx mocked)
class FakeResp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("x", request=None, response=self)

    def json(self):
        return self._b


@pytest.mark.parametrize("resp,kind", [
    (FakeResp({"message": {"content": "not json {"}, "done_reason": "stop"}), "invalid_json"),
    (FakeResp({"message": {"content": "[1,2]"}, "done_reason": "stop"}), "invalid_json"),
    (FakeResp({"message": {"content": '{"a": 1'}, "done_reason": "length"}), "truncated"),
    (FakeResp({}), "invalid_json"),
    (FakeResp({}, status=500), "http_error"),
    (httpx.ReadTimeout("t"), "timeout"),
    (httpx.ConnectError("c"), "unreachable"),
])
def test_chat_json_maps_every_failure_to_an_error_kind(resp, kind, monkeypatch):
    def post(*a, **k):
        if isinstance(resp, Exception):
            raise resp
        return resp
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ollama_client.OllamaError) as e:
        ollama_client.chat_json("s", "u", {})
    assert e.value.kind == kind


GOOD = {"message": {"content": '{"summary": "ok"}'}, "done_reason": "stop", "eval_count": 3}


def test_a_transient_5xx_is_retried_once_and_can_succeed(monkeypatch):
    seen = []

    def post(*a, **k):
        seen.append(1)
        return FakeResp({}, status=500) if len(seen) == 1 else FakeResp(GOOD)
    monkeypatch.setattr(httpx, "post", post)
    parsed, meta = ollama_client.chat_json("s", "u", {})
    assert parsed == {"summary": "ok"} and len(seen) == 2 and meta["model"] == config.OLLAMA_MODEL


def test_a_persistent_5xx_is_tried_exactly_twice_then_reported(monkeypatch):
    seen = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (seen.append(1), FakeResp({}, status=500))[1])
    with pytest.raises(ollama_client.OllamaError) as e:
        ollama_client.chat_json("s", "u", {})
    assert e.value.kind == "http_error" and len(seen) == 2


@pytest.mark.parametrize("resp", [FakeResp({}, status=404), FakeResp({}, status=400), httpx.ReadTimeout("t"), httpx.ConnectError("c")])
def test_timeouts_connection_failures_and_client_errors_are_never_retried(resp, monkeypatch):
    seen = []

    def post(*a, **k):
        seen.append(1)
        if isinstance(resp, Exception):
            raise resp
        return resp
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ollama_client.OllamaError):
        ollama_client.chat_json("s", "u", {})
    assert len(seen) == 1


# ---------------------------------------------------------------- orchestrator: the prediction survives anything
def llm_up(monkeypatch):
    monkeypatch.setattr(ollama_client, "status", lambda timeout=3.0: {"available": True, "model": "fake"})


def run_to_end(client):
    return wait_done(client, client.post("/api/agent/runs", json=BODY).json()["run_id"])


def llm_returning(raw):
    return lambda prediction, facts, timeout=None: (copy.deepcopy(raw), {"model": "fake", "seconds": 0.1})


def llm_good(prediction, facts, timeout=None):
    return report.template_analysis(prediction, facts), {"model": "fake", "seconds": 0.1}


BAD_RAWS = [{}, [], None, "x", {"summary": "s", "drivers": "none", "risks": []}, {"summary": "", "drivers": [], "risks": []},
            {"summary": "Profit guaranteed 99%", "drivers": [{"claim": "profit", "evidence_ref": "prediction", "fact_ids": ["F1"]}], "risks": []}]


@pytest.mark.parametrize("raw", BAD_RAWS)
def test_run_with_bad_llm_output_keeps_prediction_and_completes(raw, client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "llm_analysis", llm_returning(raw))
    done = run_to_end(client)
    assert done["status"] == "completed"
    assert done["prediction"] == client.post("/api/predict", json=BODY).json()
    assert done["analysis"]["generated_by"]["fallback"] is True and done["analysis"]["summary"] and done["analysis"]["drivers"]


def test_worker_survives_a_bad_response_and_runs_the_next_job(client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "llm_analysis", llm_returning({"summary": "s", "drivers": "none", "risks": []}))
    assert run_to_end(client)["status"] == "completed"
    monkeypatch.setattr(report, "llm_analysis", llm_good)
    second = run_to_end(client)
    assert second["status"] == "completed" and second["analysis"]["generated_by"]["fallback"] is False


def test_checker_crash_is_contained(client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "llm_analysis", llm_good)
    monkeypatch.setattr(report, "check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    done = run_to_end(client)
    assert done["status"] == "completed" and done["analysis"]["generated_by"]["reason"] == "verification_error"
    assert done["prediction"] == client.post("/api/predict", json=BODY).json()
    assert "boom" not in json.dumps(done)                       # internal error text is not exposed


def test_fact_builder_crash_is_contained(client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "build_facts", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    done = run_to_end(client)
    assert done["status"] == "completed" and done["analysis"]["generated_by"]["reason"] == "analysis_error"
    assert done["prediction"]["label"] in ("Flop", "Average", "Hit") and done["analysis"]["drivers"] == []


@pytest.mark.parametrize("kind", ["timeout", "invalid_json", "truncated", "unreachable", "http_error"])
def test_transport_failures_fall_back_with_reason(kind, client, monkeypatch):
    llm_up(monkeypatch)

    def boom(*a, **k):
        raise ollama_client.OllamaError(kind)
    monkeypatch.setattr(report, "llm_analysis", boom)
    done = run_to_end(client)
    assert done["analysis"]["generated_by"] == {"model": None, "fallback": True, "reason": kind}
    assert done["prediction"] == client.post("/api/predict", json=BODY).json()


def test_partial_replacement_is_reported_not_hidden(client, monkeypatch):
    llm_up(monkeypatch)

    def partial(prediction, facts, timeout=None):
        raw = report.template_analysis(prediction, facts)
        raw["summary"] = "This film will be profitable."
        return raw, {"model": "fake", "seconds": 0.1}
    monkeypatch.setattr(report, "llm_analysis", partial)
    a = (done := run_to_end(client))["analysis"]
    assert a["generated_by"]["fallback"] is False and a["generated_by"]["partial"] is True
    assert "profitable" not in a["summary"] and done["guardrails"]["replaced"] == ["summary"] and done["guardrails"]["passed"] is False


def test_successful_run_keeps_prediction_separate_and_verbatim(client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "llm_analysis", llm_good)
    done = run_to_end(client)
    assert done["prediction"] == client.post("/api/predict", json=BODY).json()
    assert "label" not in done["analysis"] and "probabilities" not in done["analysis"]
    assert done["guardrails"]["passed"] and done["guardrails"]["schema_valid"]
    assert done["evidence"]["research"]["available"] is False and done["analysis"]["limitations"] and done["disclaimers"]


def test_run_falls_back_when_ollama_is_down(client, monkeypatch):
    monkeypatch.setattr(ollama_client, "status", lambda timeout=3.0: {"available": False, "model": "x", "reason": "ollama_unreachable"})
    done = run_to_end(client)
    assert done["status"] == "completed" and done["analysis"]["generated_by"] == {"model": None, "fallback": True, "reason": "ollama_unreachable"}


def test_run_output_has_no_secret_like_content(client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "llm_analysis", llm_good)
    blob = json.dumps(run_to_end(client))
    assert not re.search(r"(?i)api[_-]?key|secret|token|password|bearer", blob)
    assert not re.search(r"[A-Za-z]:\\\\", blob)


# ---------------------------------------------------------------- API contract
def test_post_rejects_post_release_fields(client):
    assert client.post("/api/agent/runs", json={**BODY, "gross": 1_000_000}).status_code == 422
    assert client.post("/api/agent/runs", json={**BODY, "imdb_score": 8}).status_code == 422


def test_unknown_or_malformed_run_id_is_json_404(client):
    for rid in ("deadbeef0000", "x", "ABCDEF123456"):
        r = client.get(f"/api/agent/runs/{rid}")
        assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert client.get("/api/agent/runs/../etc/passwd").status_code == 404


def test_status_endpoint_never_raises(client, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:59999")
    assert client.get("/api/agent/status").json()["llm"]["available"] is False


def test_stale_runs_are_failed_on_startup():
    store.create("aaaaaaaaaaaa", {})
    assert store.fail_stale() == 1 and store.get("aaaaaaaaaaaa")["status"] == "failed"


def test_existing_predict_contract_unchanged(client):
    assert set(client.post("/api/predict", json=BODY).json()) == {"label", "probabilities", "inputs", "imputed", "warnings", "model", "caveats"}
