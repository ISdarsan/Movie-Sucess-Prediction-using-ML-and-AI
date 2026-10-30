"""Analysis writing and checking.

Pipeline: facts (code) -> LLM explains them as JSON -> strict schema validation -> claim checks -> keep / replace / fall back.

What the checks DO guarantee: the JSON has the full expected shape; every cited fact id and evidence_ref exists and agree with each
other; every number in a driver appears in the facts it cites (percent readings of fractions allowed); numbers written next to a class
name belong to that class in the cited facts; no profit, box-office or certainty wording; no causal or direction vocabulary; a
sensitivity claim does not contradict the code-computed "changes the top class" flag.

What they do NOT guarantee: they are pattern checks, not a proof of meaning. A sentence can still be misleading while using only
allowed numbers and words, and film titles or other nouns are not fact-checked. Anything that fails is replaced by safe template
wording, or the whole analysis falls back to the template.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..app.config import CLASSES
from . import ollama_client
from .facts import NUM, Fact, build_facts, numbers_in, pct_forms  # noqa: F401  (build_facts re-exported for the orchestrator)

EVIDENCE_REFS = ["prediction", "cohort", "sensitivity", "comparables"]
Ref = Literal["prediction", "cohort", "sensitivity", "comparables"]


class DriverModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    claim: str = Field(min_length=1, max_length=320)
    evidence_ref: Ref
    fact_ids: list[str] = Field(min_length=1, max_length=3)


class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    summary: str = Field(min_length=1, max_length=400)
    drivers: list[DriverModel] = Field(min_length=1, max_length=3)


# Hand-written (not model_json_schema()) so Ollama's grammar builder never meets $ref/$defs. A test keeps it in sync with AnalysisModel.
ANALYSIS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "drivers": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"claim": {"type": "string"}, "evidence_ref": {"type": "string", "enum": EVIDENCE_REFS},
                           "fact_ids": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}}},
            "required": ["claim", "evidence_ref", "fact_ids"]}},
    },
    "required": ["summary", "drivers"],
}

SYSTEM = (
    "You explain a movie-classification result by rewording numbered FACTS. Use only the facts; never add knowledge of your own. "
    "Hit, Average and Flop are IMDb-score bands (Flop below 3, Average 3 to 6, Hit 6 or above), not profit, box office or revenue, "
    "and the scores are uncalibrated, so never call them chances or confidence. "
    "Each driver: a claim that restates one to three facts and quotes their numbers exactly as written; evidence_ref is the tag of those "
    "facts (all cited facts must share it); fact_ids lists the ids you used. "
    "Do not say what causes, drives, increases, improves or is associated with anything, and do not infer how an input affects the "
    "result: only repeat what the sensitivity fact says. Do not compute new numbers. "
    "summary: one sentence naming the top class and its score. At most 3 drivers. Do not write risks, cautions or limitations: they are added separately by the application."
)

FORBIDDEN = ["profit", "profitable", "profitability", "box office", "revenue", "earnings", "guarantee", "guaranteed", "return on investment", "roi"]
_NEGATION = re.compile(r"\b(not|no|nor|never|neither|isn't|aren't|doesn't|does not|cannot|can't|without|rather than)\b", re.I)
_CAUSAL = re.compile(r"\b(associated with|correlat\w*|caus\w*|leads? to|led to|because|due to|increas\w*|decreas\w*|boost\w*|improv\w*|rais\w*|"
                     r"lower\w*|higher|drives?|driven|results? in|tends? to|more likely|less likely|impact\w*|influenc\w*|effect\w*)\b", re.I)
_CHANGE = re.compile(r"\b(chang\w*|shift\w*|switch\w*|flip\w*|differ\w*|vary|varies)\b", re.I)
_CLS = "|".join(CLASSES)
_NUM_THEN_CLASS = re.compile(rf"(\d+(?:\.\d+)?)%?\s+({_CLS})\b")
_CLASS_THEN_NUM = re.compile(rf"\b({_CLS})\b[^\d.]{{0,20}}?(\d+(?:\.\d+)?)")


# ---------------------------------------------------------------- claim checks
def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _issues(text: str, facts: list[Fact], extra_numbers: set[str], sensitivity_flag: bool | None = None) -> list[tuple[str, str]]:
    """Problems in one piece of LLM text, checked against the facts it cites (or all facts for the summary and risks)."""
    out: list[tuple[str, str]] = []
    allowed = set(extra_numbers)
    for f in facts:
        allowed |= f.numbers
    for n in NUM.findall(text):
        if n not in allowed:
            out.append(("number", f"number not in the cited facts: {n}"))
    # a number written beside a class name must belong to that class in the cited facts
    pairs = [(m.group(2), m.group(1)) for m in _NUM_THEN_CLASS.finditer(text)]                       # "88 Hit"
    pairs += [(m.group(1), m.group(2)) for m in _CLASS_THEN_NUM.finditer(_NUM_THEN_CLASS.sub(" ", text))]  # "Hit at 65%"
    for cls, num in pairs:
        own = set().union(*[f.by_class.get(cls, set()) for f in facts])
        others = set().union(*[v for f in facts for k, v in f.by_class.items() if k != cls])
        if num not in own and num in others:
            out.append(("class_number", f"{num} is not a {cls} figure in the cited facts"))
    for s in _sentences(text):
        low = s.lower()
        for term in FORBIDDEN:
            if re.search(rf"\b{re.escape(term)}\b", low) and not _NEGATION.search(low):
                out.append(("forbidden", term))
        m = _CAUSAL.search(s)
        if m:
            out.append(("causal", f"direction/causal wording: '{m.group(0)}'"))
        if sensitivity_flag is not None and _CHANGE.search(s):
            negated = bool(_NEGATION.search(s))
            if (sensitivity_flag is False and not negated) or (sensitivity_flag is True and negated):
                out.append(("sensitivity", "claim contradicts the computed sensitivity result"))
    return out


def _pc(x: float) -> str:
    return f"{x * 100:.0f}%"


def template_analysis(prediction: dict, facts: list[Fact]) -> dict:
    """Deterministic analysis: each driver is one fact, verbatim. Used when the LLM is unavailable, fails, or a driver must be replaced."""
    probs, top = prediction["probabilities"], prediction["label"]
    by_ref = {}
    for f in facts:
        by_ref.setdefault(f.ref, f)
    order = [r for r in ("prediction", "cohort", "sensitivity", "comparables") if r in by_ref][:3]
    drivers = [{"claim": by_ref[r].text, "evidence_ref": r, "fact_ids": [by_ref[r].id]} for r in order]
    return {"summary": f"For these pre-release traits the model ranks {top} highest at {_pc(probs[top])}. This is an IMDb-score band, not a profit or box-office forecast.",
            "drivers": drivers}


def _guard(schema_valid=True, issues=None, replaced=None) -> dict:
    issues = issues or []
    return {"schema_valid": schema_valid, "numbers_verified": not any(k in ("number", "class_number") for k, _ in issues),
            "unsupported_claims": [d for k, d in issues if k not in ("forbidden",)], "forbidden_terms": sorted({d for k, d in issues if k == "forbidden"}),
            "replaced": replaced or [], "passed": schema_valid and not issues}


def check(raw: object, facts: list[Fact], prediction: dict, template: dict) -> dict:
    """Validate and verify the LLM output. Returns {analysis, status: ok|partial|fallback, reason, guardrails}. Never raises on bad LLM data."""
    # Risks are generated deterministically by risks.py. A model that sends a "risks" key anyway is ignored, never used.
    ignored_risks = isinstance(raw, dict) and "risks" in raw
    if ignored_risks:
        raw = {k: v for k, v in raw.items() if k != "risks"}
    try:
        parsed = AnalysisModel.model_validate(raw)
    except ValidationError as exc:
        msgs = [f"schema: {'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}"[:120] for e in exc.errors()[:5]]
        g = _guard(False, [("schema", m) for m in msgs])
        return {"analysis": template, "status": "fallback", "reason": "invalid_schema", "guardrails": g}

    by_id = {f.id: f for f in facts}
    inputs_numbers = numbers_in(json.dumps(prediction.get("inputs", {})))
    issues: list[tuple[str, str]] = []
    replaced: list[str] = []

    summary = parsed.summary
    s_issues = _issues(summary, facts, inputs_numbers)
    if s_issues:
        issues += [(k, f"summary: {d}") for k, d in s_issues]
        summary = template["summary"]; replaced.append("summary")

    tmpl_by_ref = {d["evidence_ref"]: d for d in template["drivers"]}
    drivers, kept_llm, specific = [], 0, False
    used_refs: set[str] = set()
    for i, d in enumerate(parsed.drivers, 1):
        cited = [by_id.get(x) for x in d.fact_ids]
        problems: list[tuple[str, str]] = []
        if any(c is None for c in cited):
            problems.append(("bad_reference", f"unknown fact id in {d.fact_ids}"))
        elif any(c.ref != d.evidence_ref for c in cited):
            problems.append(("bad_reference", f"fact ids {d.fact_ids} do not belong to evidence_ref '{d.evidence_ref}'"))
        else:
            flag = next((c.changes_top_class for c in cited if c.changes_top_class is not None), None)
            problems += _issues(d.claim, cited, inputs_numbers, flag if d.evidence_ref == "sensitivity" else None)
        if problems:
            issues += [(k, f"driver {i}: {x}") for k, x in problems]
            replaced.append(f"driver {i}")
            safe = tmpl_by_ref.get(d.evidence_ref)
            if safe and d.evidence_ref not in used_refs:
                drivers.append(safe); used_refs.add(d.evidence_ref)
            continue
        kept_llm += 1
        specific = specific or bool(NUM.search(d.claim)) or any(" (" in c.text and c.text.split(" (")[0] in d.claim for c in cited)
        drivers.append({"claim": d.claim, "evidence_ref": d.evidence_ref, "fact_ids": d.fact_ids}); used_refs.add(d.evidence_ref)

    if kept_llm == 0:
        return {"analysis": template, "status": "fallback", "reason": "failed_verification", "guardrails": _guard(True, issues, replaced)}
    if not specific:
        issues.append(("not_specific", "no driver quotes a number or comparable film from its facts"))
        return {"analysis": template, "status": "fallback", "reason": "not_specific", "guardrails": _guard(True, issues, replaced)}

    analysis = {"summary": summary, "drivers": drivers}
    return {"analysis": analysis, "status": "partial" if replaced else "ok", "reason": None, "guardrails": {**_guard(True, issues, replaced), "llm_risks_ignored": ignored_risks}}


def llm_analysis(prediction: dict, facts: list[Fact], timeout: float | None = None) -> tuple[object, dict]:
    """Ask the local model to explain the facts. Returns (raw parsed JSON, meta). Raises OllamaError on transport/JSON failures."""
    lines = [f"{f.id} [{f.ref}]: {f.text}" for f in facts]
    user = f"Inputs: {json.dumps(prediction['inputs'])}\nFACTS:\n" + "\n".join(lines) + "\nWrite the JSON now."
    return ollama_client.chat_json(SYSTEM, user, ANALYSIS_SCHEMA, num_predict=600, timeout=timeout)
