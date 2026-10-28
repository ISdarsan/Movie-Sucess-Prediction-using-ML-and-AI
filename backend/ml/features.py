"""Pre-release feature definition for the Movie Success classifier.

Everything the model may see is decided here, in one place, so the notebook, the training script, the API and the
tests cannot drift apart. The question asked of every column is:

    "Could this value realistically have been known when the prediction is made, i.e. BEFORE the film is released?"

Columns whose timing cannot be established are excluded from the strict model rather than assumed to be safe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
CLASSES = ["Flop", "Average", "Hit"]            # label order used everywhere in reports
TARGET_BINS = [0, 3, 6, 10]                     # unchanged from the original project: [0,3) Flop, [3,6) Average, [6,10] Hit

# ------------------------------------------------------------------ the strict pre-release inputs
NUMERIC = ["budget", "duration", "title_year", "aspect_ratio"]
CATEGORICAL = ["country", "language", "content_rating", "color"]
GENRES = "genres"                                # multi-label: 'Action|Adventure|Sci-Fi'
PEOPLE = ["director_name", "actor_1_name", "actor_2_name", "actor_3_name"]   # only used to derive experience counts
BASE_INPUTS = NUMERIC + CATEGORICAL + [GENRES]
PEOPLE_INPUTS = PEOPLE                            # appended when the 'experience' feature set is used


@dataclass(frozen=True)
class Audit:
    column: str
    timing: str        # before_release | maybe_before_release | after_release | identifier | target
    used: str          # strict | strict (derived) | no
    reason: str


AUDIT: list[Audit] = [
    Audit("budget", "before_release", "strict", "Production budget is fixed before release. NOTE: non-US budgets are in local currency (not converted); 9.9% missing."),
    Audit("duration", "before_release", "strict", "Final runtime is known at the end of production, before release."),
    Audit("title_year", "before_release", "strict", "Planned release year. Also lets the model see era effects, which is a limitation for future films."),
    Audit("aspect_ratio", "before_release", "strict", "A production decision made before release."),
    Audit("genres", "before_release", "strict", "Known from the project description. Encoded as multi-hot genre flags, not as one category per combination."),
    Audit("country", "before_release", "strict", "Production country."),
    Audit("language", "before_release", "strict", "Spoken language."),
    Audit("content_rating", "maybe_before_release", "strict", "MPAA-style rating is assigned after the final cut but before release; 6.1% missing (kept as 'Unknown')."),
    Audit("color", "before_release", "strict", "Colour or black-and-white is a production decision."),
    Audit("director_name", "before_release", "strict (derived)", "Used ONLY to count the director's earlier films in the training data; the name itself is not a feature (2,398 directors would let the model memorise people)."),
    Audit("actor_1_name", "before_release", "strict (derived)", "Used only to count the lead actor's earlier films in the training data."),
    Audit("actor_2_name", "before_release", "strict (derived)", "Used only for the supporting-cast experience count."),
    Audit("actor_3_name", "before_release", "strict (derived)", "Used only for the supporting-cast experience count."),
    Audit("gross", "after_release", "no", "Box-office revenue exists only after release."),
    Audit("num_voted_users", "after_release", "no", "IMDb votes accumulate after release (and track the target: Spearman 0.39 with IMDb score)."),
    Audit("num_user_for_reviews", "after_release", "no", "User reviews are written after release."),
    Audit("num_critic_for_reviews", "after_release", "no", "Critic reviews are written after release."),
    Audit("movie_facebook_likes", "after_release", "no", "Social engagement with the film; collected in 2016, long after release."),
    Audit("director_facebook_likes", "maybe_before_release", "no", "Popularity snapshot collected once (2016); its timing relative to each film cannot be established, and it can reflect that film's success."),
    Audit("actor_1_facebook_likes", "maybe_before_release", "no", "Same as above: a 2016 snapshot, not a pre-release value."),
    Audit("actor_2_facebook_likes", "maybe_before_release", "no", "Same as above."),
    Audit("actor_3_facebook_likes", "maybe_before_release", "no", "Same as above."),
    Audit("cast_total_facebook_likes", "maybe_before_release", "no", "Same as above; aggregate of the cast's 2016 popularity."),
    Audit("facenumber_in_poster", "maybe_before_release", "no", "Poster artwork is a marketing asset whose timing is not recorded; weak signal, so excluded rather than assumed safe."),
    Audit("plot_keywords", "after_release", "no", "Keywords are user-generated on IMDb after release."),
    Audit("movie_title", "identifier", "no", "Identifier. Used only to find near-duplicate rows."),
    Audit("movie_imdb_link", "identifier", "no", "Identifier."),
    Audit("imdb_score", "target", "no", "Used ONLY to create the Hit/Average/Flop label. Never an input."),
]
EXCLUDED = [a.column for a in AUDIT if a.used == "no"]
assert not set(EXCLUDED) & set(BASE_INPUTS + PEOPLE_INPUTS), "an excluded column leaked into the input list"


# ------------------------------------------------------------------ data preparation (documented, deterministic)
def load_clean(path: str) -> tuple[pd.DataFrame, dict]:
    """Load the CSV, build the label, drop duplicates. Returns the frame and a log of what was removed."""
    raw = pd.read_csv(path)
    log = {"raw_rows": int(len(raw))}
    df = raw.copy()
    df["Classify"] = pd.cut(df["imdb_score"], bins=TARGET_BINS, labels=CLASSES, right=False, include_lowest=True).astype(str)
    df = df.drop_duplicates().copy()
    log["after_exact_duplicate_rows"] = int(len(df))
    # The same film scraped twice (identical title+year, vote counts differing by a few) survives exact-row dedup and would
    # sit on both sides of a random split. Keep the first occurrence of each title+year.
    # (a missing year is written as the literal 'NA' so two unrelated films that both lack a year are NOT collapsed into one)
    key = df["movie_title"].str.replace(" ", " ").str.strip().str.lower() + "|" + df["title_year"].astype("string").fillna("NA")
    df = df.loc[~key.duplicated(keep="first")].copy()
    log["after_title_year_duplicates"] = int(len(df))
    df["title"] = df["movie_title"].str.replace(" ", " ").str.strip()
    return df.reset_index(drop=True), log


# ------------------------------------------------------------------ transformers (fitted on the training set only)
def split_genres(s: str) -> list[str]:
    """Tokenizer for CountVectorizer. Module-level so the fitted pipeline can be pickled."""
    return [g for g in str(s).split("|") if g]


def _log_budget(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, 0, None))        # NaN stays NaN; imputation happens afterwards


def _names_out(self, input_features=None):      # noqa: ARG001
    return np.asarray(self.names_, dtype=object)


class PriorFilmCounts(BaseEstimator, TransformerMixin):
    """Experience features: how many films did these people make EARLIER than this film, according to the TRAINING data?

    Fit stores (person -> sorted years) from training rows only. For any row, only films with a strictly earlier year are
    counted, so a film never counts itself or its future. People unseen in training get 0. No target information is used.
    Input columns (in order): director, actor_1, actor_2, actor_3, year.
    """

    names_ = ["director_prior_films", "lead_actor_prior_films", "supporting_cast_prior_films"]

    def fit(self, X, y=None):
        X = pd.DataFrame(X).iloc[:, :5]
        X.columns = ["d", "a1", "a2", "a3", "year"]
        self.median_year_ = float(pd.to_numeric(X["year"], errors="coerce").median())
        self.index_: dict[str, dict[str, np.ndarray]] = {}
        years = pd.to_numeric(X["year"], errors="coerce").fillna(self.median_year_)
        for col in ["d", "a1", "a2", "a3"]:
            g = pd.DataFrame({"n": X[col], "y": years}).dropna()
            self.index_[col] = {n: np.sort(v.to_numpy()) for n, v in g.groupby("n")["y"]}
        self.n_features_in_ = 5
        return self

    def _count(self, col: str, names: pd.Series, years: np.ndarray) -> np.ndarray:
        idx = self.index_[col]
        out = np.zeros(len(names))
        for i, (n, y) in enumerate(zip(names, years)):
            arr = idx.get(n)
            if arr is not None:
                out[i] = np.searchsorted(arr, y, side="left")      # films strictly before year y
        return out

    def transform(self, X):
        X = pd.DataFrame(X).iloc[:, :5].copy()
        X.columns = ["d", "a1", "a2", "a3", "year"]
        years = pd.to_numeric(X["year"], errors="coerce").fillna(self.median_year_).to_numpy()
        d = self._count("d", X["d"], years)
        a1 = self._count("a1", X["a1"], years)
        sup = self._count("a2", X["a2"], years) + self._count("a3", X["a3"], years)
        return np.column_stack([d, a1, sup])

    get_feature_names_out = _names_out


def build_preprocessor(use_people: bool) -> ColumnTransformer:
    """All statistics (medians, scalers, category lists, genre vocabulary, experience index) are learned from training rows only."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    budget = Pipeline([
        ("log", FunctionTransformer(_log_budget, feature_names_out="one-to-one")),   # heavy-tailed, spans 4 orders of magnitude
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # rare levels are pooled; unseen levels at inference fall into the pooled bucket instead of raising
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15, sparse_output=False)),
    ])
    parts = [
        ("budget", budget, ["budget"]),
        ("numeric", numeric, ["duration", "title_year", "aspect_ratio"]),
        ("categorical", categorical, CATEGORICAL),
        ("genres", CountVectorizer(tokenizer=split_genres, lowercase=False, binary=True, token_pattern=None), GENRES),
    ]
    if use_people:
        parts.append(("experience", Pipeline([("count", PriorFilmCounts()), ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]),
                      PEOPLE + ["title_year"]))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)


def input_columns(use_people: bool) -> list[str]:
    return BASE_INPUTS + (PEOPLE_INPUTS if use_people else [])


def make_pipeline(estimator, use_people: bool):
    return Pipeline([("prep", build_preprocessor(use_people)), ("clf", estimator)])
