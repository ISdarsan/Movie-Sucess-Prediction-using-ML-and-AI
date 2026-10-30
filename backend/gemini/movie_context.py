"""Movie context for the movie-aware assistant: ONE explicitly selected, COMPLETED agent run, read-only, through an allow-list.

Why this module exists and what it guarantees
  * Context is resolved server-side from the stored run. Nothing about the film is accepted from the browser except the run id.
  * The run is read with a read-only SQLite connection of its own (never store._conn(), which creates tables and migrates columns), so
    answering a question can never write to, migrate or lock the run database.
  * Only fields named below are ever copied out. The stored run also holds the trace, timings, tool counts, guardrail internals, model file
    path, the run id and the request; none of those reach Gemini. The pipeline never stores post-release columns, and the allow-list
    would drop them anyway (comparables keep title/year/genres/country/rating/class only).
  * Two prompt channels, because some strings cannot be trusted:
      facts  -> the SYSTEM instruction. Only values this code computed or fixed text it wrote: class, scores, counts, risk items, caveats.
      data   -> a delimited block in the USER turn. Everything typed by a person or fetched from the web: title, free-text inputs, comparable
                titles, historical Wikipedia excerpts and the earlier written analysis. The block is fenced with a random per-request boundary and
                labelled as data to read, never instructions to follow.
  * Nothing is invented: absent evidence is reported as absent, and the prompt says so.
Single-user, local assumption: there are no user accounts. A run id is the capability to read that run (128 random bits for runs created
since the idempotency change; older 12-character ids are still accepted). Runs cannot be listed, and this module never searches."""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
from dataclasses import dataclass

from ..agent import config as agent_config

RUN_ID_RE = re.compile(r"^(?:[0-9a-f]{12}|[0-9a-f]{32})$")
MAX_DATA_CHARS = 9000                 # the untrusted data block
MAX_EXCERPT_CHARS = 400
MAX_SECTION_CHARS = 700
SECTION_KINDS = {"cast", "production", "writing"}
CLASS_BANDS = {"Flop": "IMDb score below 3", "Average": "IMDb score from 3 to below 6", "Hit": "IMDb score 6 or above"}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁠-⁩﻿]")
_WS = re.compile(r"\s+")
_WIKI = re.compile(r"^https://en\.wikipedia\.org/w/")


class ContextError(Exception):
    """Why a movie context could not be built. `status`/`error`/`message` become the HTTP answer; the message is safe to show."""
    CATALOG = {
        "run_not_found": (404, "run_not_found", "No analysis was found for that run. Check the run id or run the analysis again."),
        "run_not_ready": (409, "run_not_ready", "That analysis has not finished yet. Wait for it to complete, then ask again."),
        "run_failed": (409, "run_failed", "That analysis failed, so there is nothing to discuss. Run it again."),
        "run_incomplete": (409, "run_incomplete", "That analysis finished without a prediction, so there is nothing to discuss."),
        "context_unavailable": (503, "context_unavailable", "The saved analyses could not be read right now. Please try again."),
    }

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind
        self.status, self.error, self.message = self.CATALOG[kind]


@dataclass(frozen=True)
class MovieContext:
    facts: dict                      # trusted: goes into the system instruction
    data: dict                       # untrusted: goes into the delimited data block
    summary: dict                    # tiny, safe description for the UI (no internals)
    diagnostics: dict | None = None  # counts and headings only, for server logs (never text, never secrets)

    def reached(self, prompt_text: str) -> dict[str, bool]:
        """For each Wikipedia section kind in this context: did its actual text reach the final prompt string?"""
        out: dict[str, bool] = {}
        for source in _lst(_rec(self.data.get("web_research")).get("sources")):
            for section in _lst(_rec(source).get("sections")):
                text = _rec(section).get("text")
                if isinstance(text, str) and text:
                    out[section["kind"]] = json.dumps(text[:60], ensure_ascii=False)[1:-1] in prompt_text
        return out

    def facts_text(self) -> str:
        return json.dumps(self.facts, ensure_ascii=False, indent=1)

    def data_block(self, boundary: str) -> str:
        """The untrusted data, fenced by `boundary` (random per request, so text inside cannot forge the closing fence)."""
        return f"<<<{boundary}\n{json.dumps(self.data, ensure_ascii=False, indent=1)}\n{boundary}>>>"


def new_boundary() -> str:
    return "DATA-" + secrets.token_hex(8)


# ---------------------------------------------------------------- reading
def _read_run(run_id: str) -> tuple[str, dict, dict | None]:
    """(status, request, result) for exactly this id, from a read-only connection. Never creates or migrates anything."""
    path = agent_config.DB_PATH
    if not path.is_file():
        raise ContextError("run_not_found")
    try:
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)      # read-only: cannot create, migrate or write
        try:
            row = con.execute("SELECT status, request, result FROM runs WHERE id=?", (run_id,)).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        raise ContextError("context_unavailable") from None
    if row is None:
        raise ContextError("run_not_found")
    status, request, result = row
    try:
        return str(status), json.loads(request) if request else {}, json.loads(result) if result else None
    except ValueError:
        raise ContextError("context_unavailable") from None


# ---------------------------------------------------------------- small safe readers
def _txt(v, n: int = 120) -> str | None:
    if not isinstance(v, str):
        return None
    s = _WS.sub(" ", _CTRL.sub("", v)).strip()
    return s[:n] or None


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


def _rec(v) -> dict:
    return v if isinstance(v, dict) else {}


def _lst(v) -> list:
    return v if isinstance(v, list) else []


def _scores(v) -> dict | None:
    p = _rec(v)
    out = {c: _num(p.get(c)) for c in CLASS_BANDS}
    return out if all(x is not None for x in out.values()) else None


# ---------------------------------------------------------------- builder
def build(run_id: str) -> MovieContext:
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise ContextError("run_not_found")
    status, request, result = _read_run(run_id)
    if status in ("queued", "running"):
        raise ContextError("run_not_ready")
    if status == "failed":
        raise ContextError("run_failed")
    if status != "completed" or not isinstance(result, dict):
        raise ContextError("run_incomplete")
    pred = _rec(result.get("prediction"))
    label, scores = pred.get("label"), _scores(pred.get("probabilities"))
    if label not in CLASS_BANDS or scores is None:
        raise ContextError("run_incomplete")

    evidence, analysis = _rec(result.get("evidence")), _rec(result.get("analysis"))
    research = _rec(evidence.get("research"))
    title = _txt(_rec(research.get("query")).get("title")) or _txt(request.get("research_title"))

    # ---- trusted facts: numbers, enums, and text this project wrote
    facts: dict = {
        "random_forest_prediction": {
            "predicted_class": label,
            "class_scores": {c: round(float(scores[c]), 4) for c in CLASS_BANDS},
            "what_the_classes_mean": CLASS_BANDS,
            "model_stage": "pre-release (uses only information known before release)",
            "scores_are_calibrated": False,
        },
        "model_limitations": [t for t in (_txt(x, 400) for x in _lst(pred.get("caveats"))) if t][:6],
    }
    cohort = _rec(evidence.get("cohort"))
    final = _rec(cohort.get("final"))
    counts = _rec(final.get("counts"))
    if _num(final.get("n")) is not None and all(_num(counts.get(c)) is not None for c in CLASS_BANDS):
        facts["historical_dataset_cohort"] = {
            "films_matching_these_traits": final["n"], "class_counts": {c: counts[c] for c in CLASS_BANDS}, "mean_imdb_score": _num(final.get("mean_score")),
            "is_a_small_sample": bool(cohort.get("thin")), "matches_nothing": bool(cohort.get("empty")),
            "note": "Descriptive counts of OTHER films in the dataset (1916-2016). Not a model output and not a forecast.",
        }
    else:
        facts["historical_dataset_cohort"] = "unavailable for this analysis"
    sens = _rec(evidence.get("sensitivity"))
    segs = [s for s in _lst(sens.get("top_class_by_range")) if _rec(s).get("label") in CLASS_BANDS and _num(_rec(s).get("from")) is not None and _num(_rec(s).get("to")) is not None]
    facts["sensitivity_sweep"] = ({"input_varied": _txt(sens.get("label"), 40), "top_class_by_range": [{"from": s["from"], "to": s["to"], "top_class": s["label"]} for s in segs[:6]],
                                   "note": "How the model responds when only this input changes. It does not show what causes a film's rating."} if segs else "unavailable for this analysis")
    risks = []
    for r in _lst(analysis.get("risks"))[:8]:
        r = _rec(r)
        if all(isinstance(r.get(k), str) for k in ("id", "category", "severity", "text")):
            risks.append({"id": _txt(r["id"], 40), "category": _txt(r["category"], 30), "severity": _txt(r["severity"], 20), "text": _txt(r["text"], 420),
                          "rule": _txt(r.get("rule"), 120), "evidence_values": {str(k)[:40]: v for k, v in list(_rec(r.get("evidence")).items())[:6] if _num(v) is not None or isinstance(v, str) and len(v) < 30}})
    facts["evidence_based_risks"] = risks or "no evidence-specific risks were triggered (this does not mean there are none)"

    # ---- untrusted data: typed by a person, fetched from the web, or written by a language model
    inputs = _rec(pred.get("inputs"))
    profile = {k: ((_txt(v, 60) or "").replace("|", " / ") if isinstance(v, str) else _num(v)) for k, v in inputs.items() if k in ("title_year", "duration", "genres", "country", "content_rating", "budget", "language", "color", "aspect_ratio")}
    imputed = [{"field": _txt(i.get("field"), 30), "filled_with": _txt(i.get("value"), 30) if isinstance(i.get("value"), str) else _num(i.get("value")), "rule": _txt(i.get("rule"), 60)}
               for i in map(_rec, _lst(pred.get("imputed"))[:8]) if i.get("field")]
    comps = []
    for c in map(_rec, _lst(evidence.get("comparables"))[:8]):
        if _txt(c.get("title")) and _num(c.get("year")) is not None and c.get("Classify") in CLASS_BANDS:
            comps.append({"title": _txt(c["title"], 100), "year": int(c["year"]), "genres": (_txt(c.get("genres"), 80) or "").replace("|", " / "), "country": _txt(c.get("country"), 40), "content_rating": _txt(c.get("content_rating"), 12),
                          "dataset_class": c["Classify"]})
    sources = []
    for s in map(_rec, _lst(research.get("sources"))[:4]):
        url = s.get("url")
        if isinstance(url, str) and _WIKI.match(url) and _txt(s.get("title")):
            source_title = _txt(s["title"], 100)
            source_kind = _txt(s.get("kind"), 12)
            revision_id = s.get("revision_id")
            source = {"kind": source_kind, "title": source_title, "url": url[:300],
                      "revision_id": revision_id if isinstance(revision_id, int) and not isinstance(revision_id, bool) else None,
                      "page_version_dated": _txt(s.get("revision_timestamp"), 25), "cutoff": _txt(s.get("cutoff") or research.get("cutoff"), 25),
                      "retrieved_at": _txt(s.get("retrieved_at"), 25), "excerpt": _txt(s.get("excerpt"), MAX_EXCERPT_CHARS)}
            sections = []
            for section in map(_rec, _lst(s.get("sections"))[:4] if source_kind == "film" else []):
                section_kind = section.get("kind")
                heading, raw_text = _txt(section.get("heading"), 100), section.get("text")
                text = _txt(raw_text, MAX_SECTION_CHARS)
                if section_kind in SECTION_KINDS | {"introduction"} and heading and text:
                    was_truncated = isinstance(raw_text, str) and len(_WS.sub(" ", _CTRL.sub("", raw_text)).strip()) > len(text)
                    sections.append({"kind": section_kind, "heading": heading, "text": text, "truncated": was_truncated,
                                     "source_title": source_title, "url": url[:300], "revision_id": source["revision_id"],
                                     "page_version_dated": source["page_version_dated"], "cutoff": source["cutoff"]})
            source["sections"] = sections
            source["unavailable_sections"] = [
                {"kind": section.get("kind"), "reason": _txt(section.get("reason"), 40) or "unavailable"}
                for section in map(_rec, _lst(s.get("unavailable_sections"))[:3])
                if section.get("kind") in SECTION_KINDS
            ]
            sources.append(source)
    status_r = _txt(research.get("status"), 20)
    data: dict = {
        "film_title_given_by_the_user": title or "none was given (the film is described only by its pre-release traits)",
        "prediction_inputs_not_independently_verified": profile,
        "pipeline_imputations_not_independently_verified": imputed,
        "input_warnings": [t for t in (_txt(_rec(w).get("message"), 200) for w in _lst(pred.get("warnings"))[:4]) if t],
        "comparable_earlier_films_from_the_dataset": comps if isinstance(evidence.get("comparables"), list) else "unavailable for this analysis",     # null = the tool failed, [] = none found
        "web_research": {
            "status": status_r or "unknown", "available": bool(sources),
            "what_it_is": "Wikipedia pages as they stood BEFORE the film's planned release year began. Background context only: it is not an input to the model and not evidence that the prediction is right or wrong.",
            "cutoff": _txt(research.get("cutoff"), 25), "sources": sources,
            "not_retrieved": [{"subject": _txt(_rec(k).get("subject"), 60), "reason": _txt(_rec(k).get("reason"), 40)} for k in _lst(research.get("skipped"))[:5]],
        },
    }
    gen = _rec(analysis.get("generated_by"))
    drivers = [t for t in (_txt(_rec(d).get("claim"), 320) for d in _lst(analysis.get("drivers"))[:3]) if t]
    if _txt(analysis.get("summary"), 400):
        data["earlier_written_analysis"] = {"written_by": "a template" if gen.get("fallback") is not False else "a local language model", "summary": _txt(analysis.get("summary"), 400), "points": drivers,
                                            "caution": "Secondary summary of the numbers above; if it disagrees with them, the numbers win."}

    # keep the untrusted block bounded: shrink the least important parts first
    def size() -> int:
        boundary = "DATA-" + ("0" * 16)
        encoded = json.dumps(data, ensure_ascii=False, indent=1)
        return len(f"<<<{boundary}\n{encoded}\n{boundary}>>>")

    def shrink_sections(limit: int) -> None:
        for source in sources:
            for section in source["sections"]:
                if len(section["text"]) > limit:
                    section["text"] = section["text"][:limit].rstrip()
                    section["truncated"] = True

    steps = (
        lambda: shrink_sections(500),
        lambda: [s.update(excerpt=(s["excerpt"] or "")[:160] or None) for s in sources],
        lambda: data.pop("earlier_written_analysis", None),
        lambda: data.update(comparable_earlier_films_from_the_dataset=comps[:4]) if comps else None,
        lambda: [s.update(excerpt=None) for s in sources],
        lambda: shrink_sections(240),
        lambda: data.update(comparable_earlier_films_from_the_dataset=[]),
        lambda: data["web_research"].update(not_retrieved=[]),
        lambda: shrink_sections(120),
        lambda: data["web_research"].update(sources=[s for s in sources if s["kind"] == "film"]),
    )
    shrunk = False
    for step in steps:
        if size() <= MAX_DATA_CHARS:
            break
        step()
        shrunk = True

    summary = {"title": title, "predicted_class": label, "class_scores": facts["random_forest_prediction"]["class_scores"], "year": profile.get("title_year"),
               "genres": profile.get("genres") if isinstance(profile.get("genres"), str) else None, "country": profile.get("country"), "content_rating": profile.get("content_rating"),
               "has_written_analysis": "earlier_written_analysis" in data, "research_sources": len(sources), "comparable_films": len(comps), "risk_items": len(risks)}
    kept = [s for s in _lst(data["web_research"].get("sources")) if isinstance(s, dict)]
    diagnostics = {"run": run_id[:6] + "..", "sources": len(kept), "sections": sum(len(s["sections"]) for s in kept),
                   "section_headings": [f"{x['kind']}:{x['heading']}" for s in kept for x in s["sections"]],
                   "unavailable_sections": [u["kind"] for s in kept for u in s["unavailable_sections"]],
                   "truncated_sections": sum(1 for s in kept for x in s["sections"] if x["truncated"]), "context_shrunk": shrunk}
    return MovieContext(facts=facts, data=data, summary=summary, diagnostics=diagnostics)
