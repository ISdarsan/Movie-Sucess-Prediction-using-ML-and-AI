"""Two honest, data-driven helpers that react to the characteristics a user picks.

* cohort(): how the REAL dataset narrows as characteristics are chosen, with the actual class mix at every step.
  Purely descriptive statistics from the CSV — no model involved.
* sensitivity(): the REAL saved Random Forest re-scored while ONE input sweeps across its observed range and every other
  input stays as submitted. A response curve of the model, not a statement about cause and effect.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .config import CLASSES
from .data import movies
from .model import service

# feature -> (label, log-scale on the x axis?)
SWEEPABLE: dict[str, tuple[str, bool]] = {
    "budget": ("Budget (USD)", True),
    "duration": ("Runtime (min)", False),
    "title_year": ("Release year", False),
    "aspect_ratio": ("Aspect ratio", False),
}


def _stage(label: str, df: pd.DataFrame) -> dict:
    vc = df["Classify"].value_counts()
    return {
        "label": label,
        "n": int(len(df)),
        "counts": {c: int(vc.get(c, 0)) for c in CLASSES},
        "mean_score": None if df.empty else round(float(df["imdb_score"].mean()), 2),
    }


def _norm_title(t) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(t))).strip().casefold() if isinstance(t, str) else ""


def self_match(df: pd.DataFrame, title: str | None, year: int | None) -> pd.Series:
    """Rows that ARE the queried film: same normalised title AND same release year. Needs both, so a remake or a same-named film from
    another year is never removed, and nothing is removed when the title or year is unknown. (The dataset keeps one row per title+year.)"""
    none = pd.Series(False, index=df.index)
    key = _norm_title(title)
    if not key or year is None:
        return none
    return (df["title"].map(_norm_title) == key) & (df["year"] == year)


def cohort(genres: list[str], country: str | None, content_rating: str | None, year: int | None, duration: float | None,
           title: str | None = None) -> dict:
    """`title` (optional): when it identifies a film in the dataset (title + year), that film is left out of every count and statistic, so a
    film already in the dataset is never part of its own evidence. Without a title the behaviour is unchanged."""
    df = movies()
    me = self_match(df, title, year)
    if me.any():
        df = df[~me]
    cur = df
    stages = [_stage("All films", cur)]
    if genres:
        cur = cur[cur["genre_list"].map(lambda g: all(x in g for x in genres))]
        stages.append(_stage(" + ".join(genres), cur))
    if country:
        cur = cur[cur["country"] == country]
        stages.append(_stage(country, cur))
    if content_rating:
        cur = cur[cur["content_rating"] == content_rating]
        stages.append(_stage(f"Rated {content_rating}", cur))
    if year is not None:
        cur = cur[(cur["year"] >= year - 10) & (cur["year"] <= year + 10)]
        stages.append(_stage(f"{year - 10}–{year + 10}", cur))
    if duration is not None:
        cur = cur[(cur["duration"] >= duration - 20) & (cur["duration"] <= duration + 20)]
        stages.append(_stage(f"{int(duration - 20)}–{int(duration + 20)} min", cur))
    final = stages[-1]
    out = {
        "stages": stages,
        "final": final,
        "thin": 0 < final["n"] < 30,
        "empty": final["n"] == 0,
        "note": "Descriptive counts from the dataset's films (1916–2016). Era = ±10 years, runtime = ±20 minutes. Not a model output.",
    }
    if title is not None:
        out["excluded_self"] = bool(me.any())      # only present when a title was supplied
    return out


def sensitivity(values: dict, feature: str, points: int = 26) -> dict:
    """values: validated model inputs (genres already joined). Returns the model's class probabilities along a sweep of `feature`."""
    label, log = SWEEPABLE[feature]
    col = movies()[feature if feature != "title_year" else "year"].dropna()
    col = col[col > 0] if log else col
    grid = np.quantile(col.to_numpy(), np.linspace(0.03, 0.97, points))
    grid = np.unique(np.round(grid) if feature in ("title_year", "duration") else np.round(grid, 3))
    frames = []
    for g in grid:
        row = dict(values)
        row[feature] = float(g)
        frames.append(service.frame(row))
    batch = pd.concat(frames, ignore_index=True)
    proba = service.pipeline.predict_proba(batch)
    order = [list(service.pipeline.named_steps["clf"].classes_).index(c) for c in CLASSES]
    cur_val = values.get(feature)
    imputed = cur_val is None
    shown = float(service.medians[feature]) if imputed else float(cur_val)
    cur_p = service.pipeline.predict_proba(service.frame({**values, feature: shown}))[0]
    return {
        "feature": feature, "label": label, "log": log,
        "values": [float(g) for g in grid],
        "probabilities": {c: [round(float(p[order[i]]), 4) for p in proba] for i, c in enumerate(CLASSES)},
        "current": {"value": shown, "imputed": imputed, "probabilities": {c: round(float(cur_p[order[i]]), 4) for i, c in enumerate(CLASSES)}},
        "note": "Each point is a real run of the saved Random Forest with only this input changed and every other input as submitted. "
                "It shows how the model responds, not what causes a film's rating.",
    }
