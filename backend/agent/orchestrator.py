"""Coordinator: a small linear pipeline, not a free-running agent loop.

    predict (existing route, unchanged) -> analyse (validated tools) -> research (bounded Wikipedia context, display-only)
    -> report (LLM from evidence, verified, template fallback)

The code decides which tools run and in what order (measured: llama3.1:8b over-calls tools). The LLM sees only the finished
evidence. Every step is timed and recorded in `trace`; a failure in one step degrades the result instead of crashing the run.
"""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import config, ollama_client, report, research, risks, store
from .tools import ToolError, ToolRegistry

log = logging.getLogger("frame.agent")
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent")  # one local LLM: runs queue instead of competing for it
MAX_ACTIVE = 5

LLM_NOTE = ("The AI analysis is written by a local language model from the evidence above. It is explanation, not a prediction: "
            "the class scores come only from the saved Random Forest.")


def submit(request: dict, idem: str | None = None, fp: str | None = None) -> tuple[str, bool]:
    """Queue a run. With an idempotency key hash, a second submission of the same key returns the first run instead of creating another.
    Returns (run_id, created). The id is 128 random bits: it is the only handle needed to read a run, so it must not be guessable."""
    run_id = uuid.uuid4().hex
    if idem:
        view, created = store.create_idempotent(run_id, request, idem, fp or "")
        if not created:
            return view["id"], False
    else:
        store.create(run_id, request)
    _pool.submit(execute, run_id, request)
    return run_id, True


def _step(trace: list, name: str, fn, *, degrade: bool = True):
    t = time.perf_counter()
    try:
        out = fn()
        trace.append({"step": name, "status": "ok", "seconds": round(time.perf_counter() - t, 2)})
        return out
    except Exception as exc:  # noqa: BLE001
        kind = getattr(exc, "kind", type(exc).__name__)
        trace.append({"step": name, "status": "failed", "seconds": round(time.perf_counter() - t, 2), "detail": str(kind)[:80]})
        if not degrade:
            raise
        return None


def execute(run_id: str, request: dict) -> None:
    started = time.perf_counter()
    trace: list[dict] = []
    store.save(run_id, "running")
    try:
        result = _run(run_id, request, trace, started)
        store.save(run_id, "completed", result)
    except Exception as exc:  # noqa: BLE001
        log.exception("agent run failed")
        store.save(run_id, "failed", {"run_id": run_id, "status": "failed", "trace": trace,
                                      "error": {"error": getattr(exc, "kind", "run_failed"), "message": "The run could not produce a prediction."}})


def _run(run_id: str, request: dict, trace: list, started: float) -> dict:
    from fastapi import HTTPException

    from ..app.main import predict as predict_route  # lazy: main imports this package's router
    from ..app.schemas import PredictRequest

    tools = ToolRegistry()
    # research_title belongs to the research step only: removed here so it can never reach the prediction request or the model.
    research_title = request.get("research_title")
    pred_request = {k: v for k, v in request.items() if k != "research_title"}

    # 1. Prediction agent: the existing endpoint's own function, so the payload is identical to POST /api/predict.
    def do_predict():
        try:
            return predict_route(PredictRequest(**pred_request))
        except HTTPException as exc:
            raise ToolError("model_unavailable" if exc.status_code == 503 else "prediction_failed") from exc

    prediction = _step(trace, "prediction", do_predict, degrade=False)

    # 2. Movie Analyst agent: descriptive evidence from the dataset's pre-release fields.
    genres, values = pred_request["genres"], {**pred_request, "genres": "|".join(pred_request["genres"])}
    trait = {"genres": genres, "country": pred_request.get("country"), "content_rating": pred_request.get("content_rating"),
             "year": pred_request.get("title_year"), "duration": pred_request.get("duration"),
             "title": research_title}      # read-only: only lets the cohort/comparables leave the queried film itself out; never reaches the model
    evidence = {
        "cohort": _step(trace, "analyst.cohort", lambda: tools.call("get_cohort", trait)),
        "sensitivity": _step(trace, "analyst.sensitivity", lambda: tools.call("get_sensitivity", {"inputs": values, "feature": "duration"})),
        "comparables": _step(trace, "analyst.comparables", lambda: tools.call("get_comparables", trait)),     # None when the tool failed: risks must not read that as "none found"
    }

    # 3. Research agent: fixed, code-chosen Wikipedia lookups as of a pre-release cutoff. Not prediction/report input; movie chat gets
    #    bounded excerpts separately as untrusted reference data.
    evidence["research"] = _step(trace, "research", lambda: tools.call("research_web", {"title": research_title, "genres": genres, "year": pred_request["title_year"]})) \
        or research.disabled_block()
    if evidence["research"].get("status") not in ("ok", None) and trace and trace[-1]["step"] == "research":
        trace[-1]["detail"] = str(evidence["research"].get("status"))

    # 4. Report agent.
    analysis, generated_by, guard = _write_analysis(trace, prediction, evidence, started)
    # Risks are deterministic and attached after generation on every path (LLM, partial, fallback, analysis_error); the LLM never writes them.
    risk_items = _step(trace, "risks", lambda: risks.generate(prediction, evidence)) or []

    return {
        "run_id": run_id, "status": "completed", "inputs": prediction["inputs"],
        "prediction": prediction,                                   # verbatim POST /api/predict payload: the only source of class scores
        "evidence": evidence,
        "analysis": {**analysis, "risks": risk_items, "limitations": prediction["caveats"] + [LLM_NOTE], "generated_by": generated_by},
        "guardrails": guard,
        "disclaimers": prediction["caveats"],
        "trace": trace, "tool_calls": len(tools.calls),
        "timings": {"total_seconds": round(time.perf_counter() - started, 1)},
    }


def _write_analysis(trace: list, prediction: dict, evidence: dict, started: float):
    """Never raises: any failure leaves the deterministic prediction untouched and yields the template (or a minimal notice)."""
    try:
        return _write_analysis_inner(trace, prediction, evidence, started)
    except Exception:  # noqa: BLE001 - a defect here must not cost the user a successful prediction
        log.exception("analysis step failed")
        trace.append({"step": "report", "status": "failed", "detail": "analysis_error"})
        return ({"summary": "The written analysis could not be produced. The model prediction above is unaffected.", "drivers": []},
                {"model": None, "fallback": True, "reason": "analysis_error"}, report._guard(True))


def _write_analysis_inner(trace: list, prediction: dict, evidence: dict, started: float):
    facts = report.build_facts(prediction, evidence)
    template = report.template_analysis(prediction, facts)
    no_guard = report._guard(True)
    remaining = config.RUN_DEADLINE_S - (time.perf_counter() - started)
    st = ollama_client.status()
    if not st["available"]:
        trace.append({"step": "report.llm", "status": "skipped", "detail": st["reason"]})
        return template, {"model": None, "fallback": True, "reason": st["reason"]}, no_guard
    if remaining < 20:
        trace.append({"step": "report.llm", "status": "skipped", "detail": "run_deadline"})
        return template, {"model": None, "fallback": True, "reason": "run_deadline"}, no_guard
    try:
        t = time.perf_counter()
        raw, meta = report.llm_analysis(prediction, facts, timeout=min(config.LLM_TIMEOUT_S, remaining))
        trace.append({"step": "report.llm", "status": "ok", "seconds": round(time.perf_counter() - t, 2)})
    except ollama_client.OllamaError as exc:
        trace.append({"step": "report.llm", "status": "failed", "detail": exc.kind})
        return template, {"model": None, "fallback": True, "reason": exc.kind}, no_guard
    try:
        out = report.check(raw, facts, prediction, template)
    except Exception:  # noqa: BLE001 - checker bug or unexpected data: fall back, keep the run alive
        log.exception("verification crashed")
        trace.append({"step": "report.verify", "status": "failed", "detail": "verification_error"})
        return template, {"model": None, "fallback": True, "reason": "verification_error"}, no_guard
    trace.append({"step": "report.verify", "status": out["status"]})
    if out["status"] == "fallback":
        return out["analysis"], {"model": None, "fallback": True, "reason": out["reason"]}, out["guardrails"]
    return out["analysis"], {"model": meta["model"], "fallback": False, "partial": out["status"] == "partial", "seconds": meta["seconds"]}, out["guardrails"]
