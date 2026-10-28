"""Aggregations for the Overview page. Everything is computed from the dataset at request time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES
from .data import apply_filters, class_counts, movies


def _median(s: pd.Series):
    s = s.dropna()
    return None if s.empty else float(s.median())


def build_overview(year_min=None, year_max=None, genre=None, country=None) -> dict:
    full = movies()
    df = apply_filters(full, year_min=year_min, year_max=year_max, genre=genre, country=country)
    n = len(df)
    active = {k: v for k, v in dict(year_min=year_min, year_max=year_max, genre=genre, country=country).items() if v is not None and v != ""}
    # An active year filter at the dataset bounds is not a real filter.
    if year_min is not None and year_min <= full["year"].min():
        active.pop("year_min", None)
    if year_max is not None and year_max >= full["year"].max():
        active.pop("year_max", None)

    if n == 0:
        return {"empty": True, "filters": active, "n_movies": 0, "total_movies": len(full)}

    counts = class_counts(df)
    usa = df[df["country"] == "USA"]  # budgets for non-US films are in local currency
    fin = usa.dropna(subset=["budget", "gross"])
    fin = fin[(fin["budget"] > 10_000) & (fin["gross"] > 10_000)]  # drop obvious unit errors (e.g. budget of 218)

    # --- charts ---
    by_year = df.dropna(subset=["year"]).groupby("year")
    trend = [{"year": int(y), "count": int(len(g)), **{c: int((g["Classify"] == c).sum()) for c in CLASSES},
              "mean_score": round(float(g["imdb_score"].mean()), 3)} for y, g in by_year]

    genres = df["genre_list"].explode().dropna()
    genre_rows = []
    for g, cnt in genres.value_counts().head(14).items():
        sub = df[df["genre_list"].map(lambda l: g in l)]
        genre_rows.append({"genre": g, "count": int(cnt), "mean_score": round(float(sub["imdb_score"].mean()), 3),
                           **{c: int((sub["Classify"] == c).sum()) for c in CLASSES}})

    bins = np.arange(1.0, 10.01, 0.5)
    hist, _ = np.histogram(df["imdb_score"], bins=bins)
    score_hist = [{"lo": float(bins[i]), "hi": float(bins[i + 1]), "count": int(hist[i]),
                   "class": "Flop" if bins[i] < 3 else "Average" if bins[i] < 6 else "Hit"} for i in range(len(hist))]

    countries = []
    for c, cnt in df["country"].dropna().value_counts().head(10).items():
        sub = df[df["country"] == c]
        countries.append({"country": c, "count": int(cnt), "mean_score": round(float(sub["imdb_score"].mean()), 3)})

    scatter_src = fin
    if len(scatter_src) > 1400:
        scatter_src = scatter_src.sample(1400, random_state=7)
    scatter = [{"id": int(r.id), "title": r.title, "year": None if pd.isna(r.year) else int(r.year), "budget": float(r.budget),
                "gross": float(r.gross), "class": r.Classify, "score": float(r.imdb_score)} for r in scatter_src.itertuples()]

    # Spearman correlation of numeric features with IMDb score (descriptive only)
    cand = {"num_voted_users": "Votes", "num_critic_for_reviews": "Critic reviews", "num_user_for_reviews": "User reviews",
            "duration": "Duration (min)", "gross": "Gross", "budget": "Budget", "movie_facebook_likes": "Movie Facebook likes",
            "director_facebook_likes": "Director Facebook likes", "cast_total_facebook_likes": "Cast Facebook likes", "year": "Release year"}
    corr = []
    for col, label in cand.items():
        src = usa if col in ("gross", "budget") else df
        pair = src[[col, "imdb_score"]].dropna()
        if len(pair) >= 30:
            corr.append({"feature": label, "rho": round(float(pair[col].corr(pair["imdb_score"], method="spearman")), 3), "n": int(len(pair)),
                         "usa_only": col in ("gross", "budget")})
    corr.sort(key=lambda d: -abs(d["rho"]))

    # --- KPIs ---
    kpis = {
        "movies": n,
        "mean_score": round(float(df["imdb_score"].mean()), 2),
        "median_score": round(float(df["imdb_score"].median()), 2),
        "class_counts": counts,
        "class_share": {c: round(counts[c] / n, 4) for c in CLASSES},
        "median_budget_usd": _median(usa["budget"].where(usa["budget"] > 10_000)),
        "median_gross_usd": _median(usa["gross"].where(usa["gross"] > 10_000)),
        "budget_known": int(usa["budget"].notna().sum()), "gross_known": int(usa["gross"].notna().sum()),
        "usa_movies": int(len(usa)),
    }

    # --- insights: every sentence is derived from the numbers above ---
    insights = []
    top_class = max(counts, key=counts.get)
    insights.append(f"{counts[top_class] / n:.0%} of the {n:,} films in this view are labelled {top_class}; "
                    f"Flop ({counts['Flop']:,} films, {counts['Flop'] / n:.1%}) is the rarest class.")
    if len(genre_rows) >= 2:
        eligible = [g for g in genre_rows if g["count"] >= 30]
        if eligible:
            hi = max(eligible, key=lambda g: g["mean_score"])
            lo = min(eligible, key=lambda g: g["mean_score"])
            insights.append(f"Among genres with at least 30 films, {hi['genre']} has the highest mean IMDb score ({hi['mean_score']:.2f}) "
                            f"and {lo['genre']} the lowest ({lo['mean_score']:.2f}).")
    if corr:
        top = corr[0]
        direction = "positively" if top["rho"] > 0 else "negatively"
        insights.append(f"{top['feature']} has the strongest rank correlation with IMDb score in this view "
                        f"(Spearman ρ = {top['rho']:+.2f}, n = {top['n']:,}), moving {direction} with it. "
                        "This describes an association in the data, not a cause.")
    if len(fin) >= 30:
        r = float(np.corrcoef(np.log10(fin["budget"]), np.log10(fin["gross"]))[0, 1])
        insights.append(f"For {len(fin):,} US films with both figures, log budget and log gross correlate at r = {r:.2f}; "
                        "a larger budget is associated with, but does not guarantee, a larger gross.")
    if len(trend) >= 5:
        peak = max(trend, key=lambda t: t["count"])
        insights.append(f"The dataset has the most releases in {peak['year']} ({peak['count']:,} films); "
                        "recent years are thinner partly because the file was compiled in 2016.")

    return {
        "empty": False, "filters": active, "n_movies": n, "total_movies": len(full),
        "kpis": kpis, "class_distribution": [{"class": c, "count": counts[c], "share": round(counts[c] / n, 4)} for c in CLASSES],
        "year_trend": trend, "genres": genre_rows, "score_histogram": score_hist, "countries": countries,
        "budget_gross": {"points": scatter, "n_plotted": len(scatter), "n_eligible": int(len(fin)),
                         "note": "USA productions only: budgets of other countries are recorded in local currency."},
        "correlations": corr, "insights": insights,
        "source": {"dataset": "IMDb 5000 Movie Dataset (Kaggle, ODbL 1.0)", "rows_after_dedup": len(full),
                   "year_span": [int(full["year"].min()), int(full["year"].max())]},
    }
