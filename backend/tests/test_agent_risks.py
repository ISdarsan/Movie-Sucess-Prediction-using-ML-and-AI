"""Deterministic risk-rule tests. All mocked or pure; no Ollama needed.
Run: .venv/Scripts/python -m pytest backend/tests/test_agent_risks.py -q
Thresholds are read from backend/agent/config.py, never hard-coded here, so the boundary tests follow the documented constants.
"""
import copy
import itertools
import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from backend.agent import config, facts as factsmod, ollama_client, report, risks
from backend.agent.tools import ToolRegistry
from backend.app import insight
from backend.app.main import app

T, M = config.COHORT_THIN_N, config.MIN_COMPARABLES
RARE_BODY = {"duration": 90, "title_year": 2010, "genres": ["Western", "Horror", "Musical"], "country": "Japan", "content_rating": "G"}   # 0 matching films
COMMON_BODY = {"duration": 120, "title_year": 2010, "genres": ["Drama", "Romance"], "country": "USA", "content_rating": "PG-13", "budget": 30_000_000}
PRED = {"label": "Hit", "probabilities": {"Flop": 0.02, "Average": 0.34, "Hit": 0.64}, "inputs": {"duration": 120.0, "title_year": 2010},
        "caveats": ["Caveat one.", "Caveat two."]}


def cohort(n):
    return {"final": {"n": n, "counts": {"Flop": 0, "Average": 0, "Hit": n}, "mean_score": 6.0}}


def comps(k):
    return [{"title": f"Film {i}", "year": 2000 + i, "Classify": "Hit"} for i in range(k)]


def sens(*labels):
    return {"label": "Runtime (min)", "top_class_by_range": [{"from": 60 + 10 * i, "to": 69 + 10 * i, "label": lab} for i, lab in enumerate(labels)]}


def ids(evidence):
    return [i["id"] for i in risks.generate(PRED, evidence)]


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "runs.sqlite")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- cohort rules
def test_empty_versus_non_empty_cohort():
    assert ids({"cohort": cohort(0)}) == ["cohort_empty"]
    assert ids({"cohort": cohort(500)}) == []


@pytest.mark.parametrize("n,expected", [(1, ["cohort_thin"]), (T - 1, ["cohort_thin"]), (T, []), (T + 1, [])])
def test_cohort_threshold_boundaries(n, expected):
    assert ids({"cohort": cohort(n)}) == expected


def test_cohort_item_is_structured_and_traceable():
    (item,) = risks.generate(PRED, {"cohort": cohort(T - 1)})
    assert item["id"] == "cohort_thin" and item["category"] == "cohort" and item["severity"] == "caution"
    assert item["evidence"] == {"cohort.final.n": T - 1} and str(T) in item["rule"] and str(T - 1) in item["text"] and str(T) in item["text"]


def test_thin_rule_agrees_with_the_existing_cohort_endpoint_flag():
    if T != 30:
        pytest.skip("threshold overridden by environment")
    for g in (["Drama"], ["Western"], ["Western", "Horror"], ["Musical", "War"], ["Documentary", "Sci-Fi"], ["Film-Noir"], ["Sport", "Romance", "Comedy"]):
        c = insight.cohort(g, None, None, None, None)
        assert risks.cohort_is_thin(c) is bool(c["thin"]), (g, c["final"]["n"])
    for g, ct, rt, yr in ((["Western"], "USA", "PG", 2000), (["Horror"], "Japan", "R", 2005), (["Drama"], "France", "R", 1995)):
        c = insight.cohort(g, ct, rt, yr, 100)
        assert risks.cohort_is_thin(c) is bool(c["thin"]), (g, c["final"]["n"])


# ---------------------------------------------------------------- comparables rules
@pytest.mark.parametrize("k,expected", [(0, ["comparables_none"]), (1, ["comparables_few"]), (M - 1, ["comparables_few"]), (M, []), (M + 1, []), (8, [])])
def test_comparables_threshold_boundaries(k, expected):
    assert ids({"comparables": comps(k)}) == expected


def test_comparables_failure_is_not_reported_as_none_found():
    assert ids({"comparables": None}) == []                 # tool failed / evidence missing: no guess
    assert ids({}) == []


def test_only_valid_comparables_are_counted():
    bad = comps(M)
    bad[0] = {"title": "", "year": 1999, "Classify": "Hit"}
    assert ids({"comparables": bad}) == ["comparables_few"]
    assert ids({"comparables": comps(M) + [None, "x", {"title": "No year"}]}) == []     # junk entries do not change a sufficient count


# ---------------------------------------------------------------- sensitivity rules
def test_sensitivity_no_change_is_scoped_to_the_tested_range():
    (item,) = risks.generate(PRED, {"sensitivity": sens("Hit")})
    assert item["id"] == "sensitivity_no_change" and item["severity"] == "info"
    assert "tested range" in item["text"] and "every other input unchanged" in item["text"] and "unimportant" in item["text"]
    assert not re.search(r"\b(irrelevant|never|always)\b", item["text"])
    assert item["evidence"] == {"sensitivity.segments": 1, "sensitivity.top_class": "Hit"}


def test_sensitivity_that_changes_the_top_class():
    (item,) = risks.generate(PRED, {"sensitivity": sens("Average", "Hit")})
    assert item["id"] == "sensitivity_changes" and item["evidence"] == {"sensitivity.segments": 2}
    assert "not why a film is rated" in item["text"]


# ---------------------------------------------------------------- missing / malformed evidence
MALFORMED = [
    None, [], "x", 5, {},
    {"cohort": None}, {"cohort": []}, {"cohort": {}}, {"cohort": {"final": None}}, {"cohort": {"final": {}}}, {"cohort": {"final": {"n": "12"}}},
    {"cohort": {"final": {"n": True}}}, {"cohort": {"final": {"n": -1}}}, {"cohort": {"final": {"n": 2.5}}}, {"cohort": {"final": {"n": None}}},
    {"comparables": "x"}, {"comparables": {"a": 1}}, {"comparables": [None]}, {"comparables": [{"title": "x"}]}, {"comparables": [{"title": "x", "year": "y", "Classify": "Hit"}]},
    {"comparables": [{"title": "x", "year": 2000, "Classify": "Blockbuster"}]},
    {"sensitivity": None}, {"sensitivity": {}}, {"sensitivity": {"label": "Runtime", "top_class_by_range": []}}, {"sensitivity": {"top_class_by_range": [{"from": 1, "to": 2, "label": "Hit"}]}},
    {"sensitivity": {"label": "Runtime", "top_class_by_range": [{"from": "a", "to": 2, "label": "Hit"}]}},
    {"sensitivity": {"label": "Runtime", "top_class_by_range": [{"from": 1, "to": 2, "label": "Smash"}]}},
    {"sensitivity": {"label": "Runtime", "top_class_by_range": [{"from": 1, "to": 2}]}},
    {"sensitivity": {"label": "Runtime", "top_class_by_range": "Hit"}},
    {"sensitivity": {"label": "", "top_class_by_range": [{"from": 1, "to": 2, "label": "Hit"}]}},
]


@pytest.mark.parametrize("evidence", MALFORMED)
def test_missing_or_malformed_evidence_yields_no_item_and_never_raises(evidence):
    assert risks.generate(PRED, evidence) == []


def test_partial_evidence_only_fires_the_rules_it_supports():
    assert ids({"cohort": cohort(0), "comparables": None, "sensitivity": {"label": "x", "top_class_by_range": "junk"}}) == ["cohort_empty"]


# ---------------------------------------------------------------- determinism and claim hygiene
def test_identical_inputs_give_identical_output_and_inputs_are_not_mutated():
    ev = {"cohort": cohort(3), "comparables": comps(2), "sensitivity": sens("Average", "Hit")}
    snapshot = copy.deepcopy(ev)
    a, b = risks.generate(PRED, ev), risks.generate(copy.deepcopy(PRED), copy.deepcopy(ev))
    assert json.dumps(a) == json.dumps(b) and ev == snapshot
    assert [i["id"] for i in a] == ["cohort_thin", "comparables_few", "sensitivity_changes"]        # fixed order


GRID = list(itertools.product([None, cohort(0), cohort(T - 1), cohort(T), cohort(400)],
                              [None, comps(0), comps(M - 1), comps(M), comps(8)],
                              [None, sens("Hit"), sens("Average", "Hit")]))


def expected_ids(c, k, s):
    out = []
    if c and c["final"]["n"] == 0:
        out.append("cohort_empty")
    elif c and c["final"]["n"] < T:
        out.append("cohort_thin")
    if k is not None and len(k) == 0:
        out.append("comparables_none")
    elif k is not None and len(k) < M:
        out.append("comparables_few")
    if s:
        out.append("sensitivity_no_change" if len(s["top_class_by_range"]) == 1 else "sensitivity_changes")
    return out


@pytest.mark.parametrize("c,k,s", GRID)
def test_items_appear_exactly_when_their_rule_and_evidence_say_so(c, k, s):
    ev = {key: v for key, v in (("cohort", c), ("comparables", k), ("sensitivity", s)) if v is not None}
    assert ids(ev) == expected_ids(c, k, s)


def test_no_unsupported_genre_era_sample_or_causal_wording_ever():
    banned = re.compile(r"\b(genre|era|decade|period|same|small sample|limited evidence|representative|generali[sz]e to other)\b", re.I)
    for c, k, s in GRID:
        ev = {key: v for key, v in (("cohort", c), ("comparables", k), ("sensitivity", s)) if v is not None}
        for item in risks.generate(PRED, ev):
            t = item["text"]
            assert not banned.search(t.replace("too few to generalise from", "")), t
            assert not report._CAUSAL.search(t), t
            assert not any(re.search(rf"\b{w}\b", t.lower()) for w in report.FORBIDDEN), t
            allowed = factsmod.numbers_in(json.dumps(ev)) | {str(T), str(M)} | ({str(len(k))} if k is not None else set())   # a list's length is a counted fact
            assert set(re.findall(r"\d+(?:\.\d+)?", t)) <= allowed | {n for a in allowed for n in factsmod.pct_forms(a)}, t


def test_cohort_fact_says_small_sample_only_when_the_rule_does():
    for n, flag in ((T - 1, True), (T, False), (T + 1, False)):
        text = " ".join(f.text for f in factsmod.build_facts({"label": "Hit", "probabilities": PRED["probabilities"], "inputs": {}}, {"cohort": cohort(n)}) if f.ref == "cohort")
        assert ("small sample" in text) is flag


def test_schema_sent_to_the_llm_has_no_risks_field_and_prompt_forbids_them():
    assert "risks" not in report.ANALYSIS_SCHEMA["properties"] and "risks" not in report.AnalysisModel.model_fields
    assert "Do not write risks" in report.SYSTEM


# ---------------------------------------------------------------- end to end (LLM mocked)
def wait_done(client, run_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/agent/runs/{run_id}").json()
        if j["status"] in ("completed", "failed"):
            return j
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def run_to_end(client, body=COMMON_BODY):
    return wait_done(client, client.post("/api/agent/runs", json=body).json()["run_id"])


def llm_up(monkeypatch):
    """LLM 'available' but answered by a local stub: these tests must never reach the real model."""
    monkeypatch.setattr(ollama_client, "status", lambda timeout=3.0: {"available": True, "model": "fake"})
    monkeypatch.setattr(report, "llm_analysis", lambda p, f, timeout=None: (report.template_analysis(p, f), {"model": "fake", "seconds": 0.0}))


@pytest.fixture(autouse=True)
def never_call_real_ollama(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a mocked test reached the real Ollama client")
    monkeypatch.setattr(ollama_client, "chat_json", refuse)
    monkeypatch.setattr(ollama_client, "status", lambda timeout=3.0: {"available": False, "model": "x", "reason": "ollama_unreachable"})
    monkeypatch.setattr(config, "RESEARCH_ENABLED", False)     # research has its own mocked tests; these files must never touch the network


@pytest.mark.parametrize("raw", [{}, "x", None, {"summary": "s", "drivers": "none"}, {"summary": "This film is guaranteed profit.", "drivers": []}])
def test_malformed_llm_response_still_gets_deterministic_risks_on_a_valid_fallback(raw, client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "llm_analysis", lambda p, f, timeout=None: (copy.deepcopy(raw), {"model": "fake", "seconds": 0.1}))
    done = run_to_end(client, RARE_BODY)
    assert done["status"] == "completed" and done["analysis"]["generated_by"]["fallback"] is True
    assert done["analysis"]["summary"] and done["analysis"]["drivers"]
    assert done["analysis"]["risks"] == risks.generate(done["prediction"], done["evidence"])
    assert "cohort_empty" in [i["id"] for i in done["analysis"]["risks"]]
    assert done["prediction"] == client.post("/api/predict", json=RARE_BODY).json()


def test_risks_are_attached_on_the_analysis_error_path(client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(report, "build_facts", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    done = run_to_end(client, RARE_BODY)
    assert done["analysis"]["generated_by"]["reason"] == "analysis_error"
    assert [i["id"] for i in done["analysis"]["risks"]][0] == "cohort_empty"


def test_a_crash_in_risk_generation_is_contained_and_leaves_no_invented_risks(client, monkeypatch):
    llm_up(monkeypatch)
    monkeypatch.setattr(risks, "generate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    done = run_to_end(client, RARE_BODY)
    assert done["status"] == "completed" and done["analysis"]["risks"] == []
    assert any(t["step"] == "risks" and t["status"] == "failed" for t in done["trace"]) and "boom" not in json.dumps(done)
    assert done["prediction"] == client.post("/api/predict", json=RARE_BODY).json()


def test_failed_comparables_tool_yields_no_comparables_risk(client, monkeypatch):
    llm_up(monkeypatch)
    tool = ToolRegistry.TOOLS["get_comparables"]
    monkeypatch.setattr(tool, "fn", lambda a: (_ for _ in ()).throw(RuntimeError("down")))
    done = run_to_end(client, COMMON_BODY)
    assert done["evidence"]["comparables"] is None
    assert not [i for i in done["analysis"]["risks"] if i["category"] == "comparables"]


def test_model_caveats_stay_intact_and_separate_from_risks(client, monkeypatch):
    llm_up(monkeypatch)
    done = run_to_end(client, COMMON_BODY)
    plain = client.post("/api/predict", json=COMMON_BODY).json()
    assert done["disclaimers"] == plain["caveats"] and done["prediction"]["caveats"] == plain["caveats"]
    assert done["analysis"]["limitations"][:len(plain["caveats"])] == plain["caveats"]
    assert any("IMDb-rating band" in c for c in plain["caveats"]) and any("calibrated" in c for c in plain["caveats"])
    for item in done["analysis"]["risks"]:
        assert item["text"] not in plain["caveats"] and {"id", "category", "severity", "text", "rule", "evidence"} <= set(item)


def test_repeating_a_run_gives_identical_risks(client, monkeypatch):
    llm_up(monkeypatch)
    a, b = run_to_end(client, RARE_BODY), run_to_end(client, RARE_BODY)
    assert json.dumps(a["analysis"]["risks"]) == json.dumps(b["analysis"]["risks"])
