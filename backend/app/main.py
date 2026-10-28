"""FRAME / Film Intelligence API.

Run from the project root:  .venv/Scripts/python -m uvicorn backend.app.main:app --reload --port 8000
"""
from __future__ import annotations

import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import data, insight, overview
from .config import ARTIFACTS, CLASSES, CORS_ORIGINS
from .model import ModelUnavailable, service
from .schemas import PredictRequest, SensitivityRequest

log = logging.getLogger("frame.api")

FIELD_LABELS = {
    "duration": "Duration", "title_year": "Release year", "genres": "Genres", "country": "Country",
    "content_rating": "Content rating", "num_voted_users": "Votes", "budget": "Budget", "gross": "Gross",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    data.movies()  # warm the dataset cache
    service.load()  # model is loaded once; failures surface through /api/health
    yield


app = FastAPI(title="FRAME / Film Intelligence API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    fields = []
    for e in exc.errors():
        loc = [str(p) for p in e["loc"] if p != "body"]
        fields.append({"field": loc[0] if loc else "request", "message": e["msg"].removeprefix("Value error, ")})
    return JSONResponse(status_code=422, content={"error": "validation_error", "message": "Some inputs are invalid.", "fields": fields})


@app.exception_handler(StarletteHTTPException)
async def http_handler(_: Request, exc: StarletteHTTPException):
    body = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error", "message": "Something went wrong on the server."})


def _artifact(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        raise HTTPException(503, detail={"error": "artifact_missing",
                                         "message": f"{name} has not been generated. Run backend/scripts/build_artifacts.py."})
    return _load_json(str(path))


@lru_cache(maxsize=4)
def _load_json(path: str) -> dict:
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    df = data.movies()
    ok = service.ready
    return JSONResponse(status_code=200 if ok else 503, content={
        "status": "ok" if ok else "degraded", "model": service.describe(),
        "dataset": {"rows": len(df)},
        "artifacts": {n: (ARTIFACTS / n).exists() for n in ("model_metrics.json", "explain.json")},
    })


@lru_cache(maxsize=1)
def _wall_payload() -> dict:
    """Every film as a compact frame: ordered by IMDb score so the class structure is visible at a glance."""
    df = data.movies().sort_values(["imdb_score", "id"]).reset_index(drop=True)
    code = {"Flop": 0, "Average": 1, "Hit": 2}
    return {
        "n": int(len(df)),
        "ids": df["id"].astype(int).tolist(),
        "titles": df["title"].tolist(),
        "years": [None if v != v else int(v) for v in df["year"]],
        "scores": df["imdb_score"].round(1).tolist(),
        "classes": [code[c] for c in df["Classify"]],
        "class_codes": ["Flop", "Average", "Hit"],
    }


@app.get("/api/wall")
def wall():
    return _wall_payload()


@app.get("/api/filters")
def filters():
    return data.filter_options()


# ------------------------------------------------------------------ overview
@app.get("/api/overview")
def get_overview(year_min: Optional[int] = None, year_max: Optional[int] = None,
                 genre: Optional[str] = None, country: Optional[str] = None):
    return overview.build_overview(year_min, year_max, genre or None, country or None)


# ------------------------------------------------------------------ model
@app.get("/api/model-metrics")
def model_metrics():
    return _artifact("model_metrics.json")


@app.get("/api/explain")
def explain():
    return _artifact("explain.json")


@app.get("/api/predict/schema")
def predict_schema():
    """What the saved pre-release pipeline accepts, so the form is generated from the real model rather than hard-coded."""
    if not service.ready:
        raise HTTPException(503, detail={"error": "model_unavailable", "message": "The prediction model is not available."})
    sc = service.schema
    return {
        "stage": "pre-release",
        "columns": service.columns,
        "numeric": service.numeric,
        "numeric_ranges": sc["numeric"],
        "categorical": service.categorical,
        "genres": service.genres,
        "defaults_when_blank": {"numeric_median": service.medians, "categorical": sc["defaults_when_blank"]["categorical"]},
        "classes": CLASSES,
        "excluded": [{"column": e["column"], "timing": e["timing"], "reason": e["reason"]} for e in sc["excluded"]],
    }


def _unseen(values: dict) -> list[dict]:
    """Values the training data never contained. The fitted pipeline handles them (pooled 'infrequent' bucket / ignored genre
    flag), so they are reported as warnings instead of being rejected."""
    out = []
    for col in ("country", "language", "content_rating", "color"):
        v = values.get(col)
        if v is not None and v != "" and v not in service.categorical.get(col, []):
            out.append({"field": col, "message": f"'{v}' was not seen in the training data, so it is treated as an uncommon {col.replace('_', ' ')}."})
    for g in str(values.get("genres", "")).split("|"):
        if g and g not in service.genres:
            out.append({"field": "genres", "message": f"The genre '{g}' was not seen in the training data and is ignored."})
    return out


@app.get("/api/cohort")
def get_cohort(genres: str = Query("", max_length=200), country: Optional[str] = Query(None, max_length=60),
               content_rating: Optional[str] = Query(None, max_length=20), year: Optional[int] = Query(None, ge=1800, le=2100),
               duration: Optional[float] = Query(None, ge=1, le=600)):
    """How the real dataset narrows as characteristics are chosen, with the actual class mix at every step."""
    g = sorted({x.strip() for x in genres.split(",") if x.strip()})[:8]
    return insight.cohort(g, country or None, content_rating or None, year, duration)


@app.post("/api/sensitivity")
def sensitivity(req: SensitivityRequest):
    """Re-score the saved Random Forest while one input sweeps its observed range; all other inputs stay as submitted."""
    if not service.ready:
        raise HTTPException(503, detail={"error": "model_unavailable", "message": "The prediction model is not available right now."})
    values = req.inputs.model_dump()
    values["genres"] = "|".join(values["genres"])
    try:
        return insight.sensitivity(values, req.feature)
    except Exception:  # noqa: BLE001
        log.exception("sensitivity failed")
        raise HTTPException(500, detail={"error": "sensitivity_failed", "message": "The model could not be swept for this input."})


def _caveats() -> list[str]:
    """Limitations stated from the model's own evaluation file, never from memory."""
    out = ["Class scores are the model's estimated class probabilities. They have not been calibrated, so treat them as a ranking of the three classes, not as real-world chances.",
           "The label is an IMDb-rating band (Flop below 3, Average 3 to 6, Hit 6 or above), not box office or profit. The model sees only information available before release."]
    try:
        m = _artifact("model_metrics.json")
        h, ev = m["held_out"], m["evaluation"]
        base = max(ev["test_class_counts"].values()) / ev["test_size"]
        out.append(f"On the held-out test set this model scored {h['accuracy']:.1%} accuracy; always answering the most common class scores {base:.1%}. "
                   f"Its balanced accuracy is {h['balanced_accuracy']:.1%} (chance is 33.3%), so pre-release information explains only a modest part of how a film is rated.")
        fl = h["per_class"]["Flop"]
        out.append(f"Flop is rare ({ev['train_class_counts']['Flop']} training films). On the test set the model found {h['confusion_matrix'][0][0]} of {fl['support']} Flops, too few to be reliable either way.")
    except HTTPException:
        pass
    return out


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not service.ready:
        raise HTTPException(503, detail={"error": "model_unavailable", "message": "The prediction model is not available right now."})
    values = req.model_dump()
    values["genres"] = "|".join(values["genres"])

    warnings = _unseen(values)
    if req.title_year < 1916 or req.title_year > 2016:
        warnings.append({"field": "title_year", "message": "The training data covers 1916–2016; the model has never seen films outside that range."})

    try:
        result = service.predict(values)
    except ModelUnavailable:
        raise HTTPException(503, detail={"error": "model_unavailable", "message": "The prediction model is not available right now."})
    except Exception:  # noqa: BLE001
        log.exception("prediction failed")
        raise HTTPException(500, detail={"error": "prediction_failed", "message": "The model could not score this input."})

    imputed = []
    for c in service.columns:
        if c == "genres" or values.get(c) not in (None, ""):
            continue
        if c in service.numeric:
            imputed.append({"field": c, "value": service.medians[c], "rule": "training median"})
        else:
            imputed.append({"field": c, "value": "Unknown", "rule": "its own 'Unknown' category"})
    try:
        name = _artifact("model_metrics.json")["selected_model"]["name"]
    except HTTPException:
        name = "Pre-release model"
    return {
        **result,
        "inputs": {k: v for k, v in values.items() if v is not None},
        "imputed": imputed,
        "warnings": warnings,
        "model": {"name": name, "source": "pre_release_movie_classifier.joblib", "stage": "pre-release"},
        "caveats": _caveats(),
    }


# ------------------------------------------------------------------ explorer
LIST_COLS = ["id", "title", "year", "genres", "country", "content_rating", "duration", "imdb_score", "Classify", "budget", "gross", "num_voted_users", "director_name"]
SORTABLE = {"title", "year", "imdb_score", "duration", "budget", "gross", "num_voted_users"}


def _filtered(year_min, year_max, genre, country, content_rating, classify, q):
    return data.apply_filters(data.movies(), year_min=year_min, year_max=year_max, genre=genre or None, country=country or None,
                              content_rating=content_rating or None, classify=classify or None, q=q or None)


@app.get("/api/movies")
def list_movies(page: int = Query(1, ge=1), page_size: int = Query(25, ge=5, le=100),
                sort: str = "imdb_score", order: Literal["asc", "desc"] = "desc",
                year_min: Optional[int] = None, year_max: Optional[int] = None, genre: Optional[str] = None,
                country: Optional[str] = None, content_rating: Optional[str] = None,
                classify: Optional[Literal["Flop", "Average", "Hit"]] = None, q: Optional[str] = Query(None, max_length=80)):
    if sort not in SORTABLE:
        raise HTTPException(400, detail={"error": "bad_sort", "message": f"Cannot sort by '{sort}'."})
    df = _filtered(year_min, year_max, genre, country, content_rating, classify, q)
    total = len(df)
    key = df["title"].str.lower() if sort == "title" else df[sort]
    df = df.assign(_k=key).sort_values(["_k", "id"], ascending=[order == "asc", True], na_position="last")
    start = (page - 1) * page_size
    rows = [{c: data.clean(v) for c, v in zip(LIST_COLS, r)} for r in df.iloc[start:start + page_size][LIST_COLS].itertuples(index=False)]
    return {"total": total, "page": page, "page_size": page_size, "pages": max(1, -(-total // page_size)), "rows": rows}


@app.get("/api/movies/export.csv")
def export_movies(year_min: Optional[int] = None, year_max: Optional[int] = None, genre: Optional[str] = None,
                  country: Optional[str] = None, content_rating: Optional[str] = None,
                  classify: Optional[Literal["Flop", "Average", "Hit"]] = None, q: Optional[str] = Query(None, max_length=80)):
    df = _filtered(year_min, year_max, genre, country, content_rating, classify, q).sort_values("imdb_score", ascending=False)
    cols = ["title", "year", "genres", "country", "language", "content_rating", "duration", "budget", "gross", "num_voted_users",
            "num_critic_for_reviews", "num_user_for_reviews", "director_name", "actor_1_name", "imdb_score", "Classify"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in df[cols].itertuples(index=False):
        w.writerow(["" if (v is None or (isinstance(v, float) and v != v)) else v for v in r])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="frame-movies.csv"'})


@app.get("/api/movies/{movie_id}")
def movie_detail(movie_id: int):
    df = data.movies()
    hit = df[df["id"] == movie_id]
    if hit.empty:
        raise HTTPException(404, detail={"error": "not_found", "message": "No such movie."})
    r = hit.iloc[0]
    fields = ["id", "title", "year", "genres", "country", "language", "content_rating", "color", "duration", "aspect_ratio", "budget", "gross",
              "imdb_score", "Classify", "num_voted_users", "num_critic_for_reviews", "num_user_for_reviews", "director_name",
              "actor_1_name", "actor_2_name", "actor_3_name", "movie_facebook_likes", "cast_total_facebook_likes", "director_facebook_likes",
              "plot_keywords", "movie_imdb_link"]
    out = {f: data.clean(r[f]) for f in fields}
    out["genre_list"] = list(r["genre_list"])
    out["keywords"] = (r["plot_keywords"].split("|") if isinstance(r["plot_keywords"], str) else [])
    out["budget_currency_note"] = None if r["country"] == "USA" else "Budget is recorded in the production country's local currency."
    return out
