"""Dataset access. Mirrors the notebook: target from imdb_score, exact duplicates removed."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..ml.features import load_clean
from .config import CLASSES, DATA_PATH


@lru_cache(maxsize=1)
def movies() -> pd.DataFrame:
    """The same cleaned dataset the pre-release model was trained and tested on (see backend/ml/features.py: load_clean)."""
    df, _ = load_clean(str(DATA_PATH))
    df["id"] = np.arange(len(df))
    df["year"] = df["title_year"]
    df["genre_list"] = df["genres"].fillna("").str.split("|").map(lambda g: [x for x in g if x])
    return df.reset_index(drop=True)


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    vc = df["Classify"].value_counts()
    return {c: int(vc.get(c, 0)) for c in CLASSES}


def apply_filters(df: pd.DataFrame, *, year_min=None, year_max=None, genre=None, country=None,
                  content_rating=None, classify=None, q=None) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if year_min is not None:
        m &= df["year"] >= year_min
    if year_max is not None:
        m &= df["year"] <= year_max
    if genre:
        m &= df["genre_list"].map(lambda g: genre in g)
    if country:
        m &= df["country"] == country
    if content_rating:
        m &= df["content_rating"] == content_rating
    if classify:
        m &= df["Classify"] == classify
    if q:
        needle = q.strip().lower()
        hay = (df["title"].fillna("") + " " + df["director_name"].fillna("") + " " + df["actor_1_name"].fillna("")
               + " " + df["actor_2_name"].fillna("") + " " + df["actor_3_name"].fillna("")).str.lower()
        m &= hay.str.contains(needle, regex=False)
    return df[m]


def filter_options() -> dict:
    df = movies()
    genres = df["genre_list"].explode().value_counts()
    countries = df["country"].value_counts()
    ratings = df["content_rating"].value_counts()
    return {
        "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
        "genres": [{"value": k, "count": int(v)} for k, v in genres.items()],
        "countries": [{"value": k, "count": int(v)} for k, v in countries.items()],
        "content_ratings": [{"value": k, "count": int(v)} for k, v in ratings.items()],
    }


def clean(v):
    """JSON-safe scalar: NaN/NaT -> None, numpy -> python."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return list(v)
    if pd.isna(v):
        return None
    return v
