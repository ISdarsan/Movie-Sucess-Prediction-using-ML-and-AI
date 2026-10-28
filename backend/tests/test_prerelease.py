"""ML-correctness tests for the pre-release study: leakage, cleaning, splitting, train-only fitting, reproducibility, selection integrity.
Run from the project root: .venv/Scripts/python -m pytest backend/tests/test_prerelease.py -q"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from backend.ml import features as F

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
CSV = str(ROOT / "dataset" / "movie_metadata.csv")


@pytest.fixture(scope="module")
def data():
    df, log = F.load_clean(CSV)
    train, test = train_test_split(df, test_size=0.2, stratify=df["Classify"], random_state=F.SEED)
    return df, log, train, test


@pytest.fixture(scope="module")
def pipe():
    return joblib.load(ROOT / "pre_release_movie_classifier.joblib")


@pytest.fixture(scope="module")
def results():
    return json.loads((ART / "prerelease_results_v1.json").read_text(encoding="utf8"))


# ---------------------------------------------------------------- leakage
def test_audit_covers_every_dataset_column():
    cols = set(pd.read_csv(CSV, nrows=1).columns)
    assert cols <= {a.column for a in F.AUDIT}, cols - {a.column for a in F.AUDIT}


def test_no_post_release_or_target_column_reaches_the_model(pipe):
    inputs = set(pipe.named_steps["prep"].feature_names_in_)
    assert inputs.isdisjoint(F.EXCLUDED)
    assert inputs.isdisjoint({"imdb_score", "Classify"})
    for banned in ("gross", "num_voted_users", "num_user_for_reviews", "num_critic_for_reviews", "movie_facebook_likes"):
        assert banned not in inputs
        assert not any(banned in n for n in pipe.named_steps["prep"].get_feature_names_out())


def test_saved_schema_matches_the_pipeline(pipe):
    schema = json.loads((ART / "prerelease_schema.json").read_text(encoding="utf8"))
    assert list(pipe.named_steps["prep"].feature_names_in_) == schema["inputs"]
    assert sorted(pipe.classes_) == sorted(F.CLASSES)


# ---------------------------------------------------------------- target + cleaning + split
def test_target_definition_is_unchanged():
    s = pd.Series([0.0, 2.9, 3.0, 5.9, 6.0, 9.9])
    got = pd.cut(s, bins=F.TARGET_BINS, labels=F.CLASSES, right=False, include_lowest=True).astype(str).tolist()
    assert got == ["Flop", "Flop", "Average", "Average", "Hit", "Hit"]


def test_cleaning_is_documented_and_does_not_delete_films_with_missing_year(data):
    df, log, *_ = data
    assert (log["raw_rows"], log["after_exact_duplicate_rows"], log["after_title_year_duplicates"]) == (5043, 4998, 4919)
    known = df.dropna(subset=["title_year"])
    assert not known.duplicated(["title", "title_year"]).any()          # no film appears twice with the same title+year
    assert df["title_year"].isna().sum() > 50                           # films without a year are kept, not collapsed together


def test_split_is_stratified_and_disjoint(data):
    df, _, train, test = data
    assert set(train.index).isdisjoint(test.index) and len(train) + len(test) == len(df)
    for c in F.CLASSES:
        assert abs((train.Classify == c).mean() - (test.Classify == c).mean()) < 0.01
    key = lambda d: set((d.title.str.lower() + "|" + d.title_year.astype("string").fillna("NA")).tolist())   # same key as the cleaning step
    assert key(train).isdisjoint(key(test)), "the same film (title+year) sits on both sides of the split"


# ---------------------------------------------------------------- preprocessing is fitted on training rows only
def test_preprocessing_statistics_come_from_the_training_set(data):
    _, _, train, test = data
    prep = F.build_preprocessor(use_people=False).fit(train[F.BASE_INPUTS])
    imputer = prep.named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == pytest.approx(train["duration"].median())
    full_median = pd.concat([train, test])["duration"].median()
    if full_median != train["duration"].median():
        assert imputer.statistics_[0] != pytest.approx(full_median)
    seen = set(prep.named_transformers_["genres"].vocabulary_)
    assert seen == {g for s in train["genres"] for g in s.split("|") if g}   # vocabulary learned from training rows only


def test_prior_film_counts_use_only_earlier_years_and_training_rows():
    tr = pd.DataFrame({"d": ["Ann", "Ann", "Bob"], "a1": ["X", "X", "Y"], "a2": ["", "", ""], "a3": ["", "", ""], "year": [2000, 2005, 2010]})
    t = F.PriorFilmCounts().fit(tr)
    out = t.transform(pd.DataFrame({"d": ["Ann", "Ann", "Ann", "Zed"], "a1": ["X", "X", "X", "Q"], "a2": [""] * 4, "a3": [""] * 4, "year": [2000, 2005, 2006, 2006]}))
    assert out[:, 0].tolist() == [0, 1, 2, 0]      # strictly earlier only; a film never counts itself; unseen person = 0


def test_oversampling_exists_only_at_fit_time():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = pd.Series(["Hit"] * 50 + ["Flop"] * 10)
    p = ImbPipeline([("over", RandomOverSampler(random_state=0)), ("clf", RandomForestClassifier(20, random_state=0))]).fit(X, y)
    assert len(p.predict(X.iloc[:7])) == 7          # inference never resamples
    cv_code = (ROOT / "backend" / "ml" / "run_prerelease.py").read_text(encoding="utf8")
    assert 'ImbPipeline([("prep"' in cv_code and "RandomOverSampler" in cv_code and "fit_resample" not in cv_code   # sampler lives inside the CV pipeline


# ---------------------------------------------------------------- reproducibility + selection integrity
def test_saved_model_reproduces_the_reported_test_metrics(data, pipe):
    _, _, _, test = data
    pred = pipe.predict(test[list(pipe.named_steps["prep"].feature_names_in_)])
    rep = json.loads((ART / "model_metrics.json").read_text(encoding="utf8"))["held_out"]
    assert round(accuracy_score(test.Classify, pred), 4) == rep["accuracy"]
    assert round(balanced_accuracy_score(test.Classify, pred), 4) == rep["balanced_accuracy"]
    assert round(f1_score(test.Classify, pred, average="macro"), 4) == rep["macro_f1"]


def test_model_survives_a_save_load_roundtrip(tmp_path, data, pipe):
    _, _, _, test = data
    X = test[list(pipe.named_steps["prep"].feature_names_in_)].head(50)
    joblib.dump(pipe, tmp_path / "m.joblib")
    again = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(again.predict_proba(X), pipe.predict_proba(X))


def test_recorded_selection_follows_the_frozen_rule_and_ignores_the_test_set(results):
    models = results["models"]
    pool = {n: r for n, r in models.items() if not n.startswith("Dummy")}

    def key(n):
        r = pool[n]
        c = r["cv"] if r["tuned"] else r.get("cv_fresh", r["cv"])
        return c["macro_f1"][0], c["balanced_accuracy"][0]
    best = max(pool, key=lambda n: key(n)[0])
    near = [n for n in pool if key(best)[0] - key(n)[0] <= 0.005]
    assert results["selected"] in near                      # selected by CV macro F1 (ties broken by balanced accuracy)
    assert max(near, key=lambda n: key(n)[1]) == results["selected"]
    # the test set was NOT what decided it: some other candidate scored a higher test macro F1
    assert any(r["test"]["macro_f1"] > models[results["selected"]]["test"]["macro_f1"] for n, r in models.items() if n != results["selected"])


def test_selected_model_beats_the_dummy_baseline_where_it_should(results):
    sel = results["models"][results["selected"]]
    dummy = results["models"]["Dummy (most frequent)"]
    assert sel["cv"]["macro_f1"][0] > dummy["cv"]["macro_f1"][0] + 0.15
    assert sel["cv"]["balanced_accuracy"][0] > dummy["cv"]["balanced_accuracy"][0] + 0.1
