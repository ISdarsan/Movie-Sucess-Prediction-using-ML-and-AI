"""Integration tests for the CANDIDATE artifact produced by run_comprehensive.py. They run BEFORE promotion: the website's live
model is only replaced after these pass.  Skipped automatically when no candidate exists (i.e. after promotion).

    .venv/Scripts/python -m pytest backend/tests/test_candidate.py -q
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from backend.app import insight
from backend.app import main as app_main
from backend.app.model import ModelService
from backend.ml import features as F

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
CAND, CSCHEMA, CRES = ROOT / "pre_release_movie_classifier.candidate.joblib", ART / "prerelease_schema.candidate.json", ART / "comprehensive_results.json"
pytestmark = pytest.mark.skipif(not (CAND.exists() and CSCHEMA.exists() and CRES.exists()), reason="no candidate artifact (already promoted or study not run)")

VALID = {"duration": 120, "title_year": 2012, "genres": ["Drama", "Romance"], "country": "USA", "content_rating": "PG-13", "budget": 40_000_000, "language": "English", "color": "Color"}


class CandidateService(ModelService):
    def load(self, *a, **k):                    # always the candidate, never the live files
        super().load(CAND, CSCHEMA)


@pytest.fixture(scope="module")
def svc():
    s = CandidateService(); s.load()
    assert s.ready, f"candidate failed to load: {s.error}"
    return s


@pytest.fixture(scope="module")
def pipe():
    return joblib.load(CAND)


@pytest.fixture(scope="module")
def results():
    return json.loads(CRES.read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def split():
    df, _ = F.load_clean(str(ROOT / "dataset" / "movie_metadata.csv"))
    return train_test_split(df, test_size=0.2, stratify=df["Classify"], random_state=F.SEED)


def test_candidate_schema_matches_pipeline_and_nine_inputs(pipe):
    schema = json.loads(CSCHEMA.read_text(encoding="utf8"))
    assert list(pipe.named_steps["prep"].feature_names_in_) == schema["inputs"] == F.BASE_INPUTS
    assert sorted(map(str, pipe.classes_)) == sorted(F.CLASSES)


def test_candidate_has_no_leaky_input(pipe):
    names = set(pipe.named_steps["prep"].feature_names_in_)
    assert names.isdisjoint(F.EXCLUDED) and not {"imdb_score", "Classify"} & names


def test_candidate_scores_valid_inputs(svc):
    r = svc.predict({**VALID, "genres": "Drama|Romance"})
    assert r["label"] in F.CLASSES and abs(sum(r["probabilities"].values()) - 1) < 1e-3 and all(0 <= p <= 1 for p in r["probabilities"].values())


def test_service_path_equals_direct_pipeline_on_real_films(svc, pipe, split):
    _, test = split
    for i in range(0, 200, 10):
        row = test.iloc[[i]]
        values = {c: (None if pd.isna(row[c].iloc[0]) else row[c].iloc[0]) for c in F.BASE_INPUTS}
        got = svc.predict(values)
        want = pipe.predict_proba(row[F.BASE_INPUTS])[0]
        for c, p in zip(map(str, pipe.classes_), want):
            assert abs(got["probabilities"][c] - p) < 1e-3
        assert got["label"] == str(pipe.predict(row[F.BASE_INPUTS])[0])


def test_unseen_categories_and_blank_optionals_do_not_break_it(svc):
    r = svc.predict({"duration": 100, "title_year": 2001, "genres": "Western|War", "country": "Atlantis", "content_rating": "ZZ"})
    assert r["label"] in F.CLASSES


def test_candidate_reproduces_the_recorded_test_metrics(pipe, split, results):
    _, test = split
    pred = pipe.predict(test[F.BASE_INPUTS])
    rec = next(f for f in results["finalists"] if f["name"] == results["selection"]["chosen"])["test"]
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    assert round(accuracy_score(test.Classify, pred), 4) == rec["accuracy"]
    assert round(balanced_accuracy_score(test.Classify, pred), 4) == rec["balanced_accuracy"]
    assert round(f1_score(test.Classify, pred, average="macro", labels=F.CLASSES), 4) == rec["macro_f1"]


def test_candidate_roundtrips_through_joblib(tmp_path, pipe, split):
    _, test = split
    X = test[F.BASE_INPUTS].head(40)
    joblib.dump(pipe, tmp_path / "m.joblib")
    np.testing.assert_allclose(joblib.load(tmp_path / "m.joblib").predict_proba(X), pipe.predict_proba(X))


def test_whole_api_works_with_the_candidate(monkeypatch):
    s = CandidateService()
    monkeypatch.setattr(app_main, "service", s)
    monkeypatch.setattr(insight, "service", s)
    with TestClient(app_main.app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        r = c.post("/api/predict", json=VALID)
        assert r.status_code == 200 and r.json()["label"] in F.CLASSES
        assert c.post("/api/predict", json={**VALID, "gross": 5}).status_code == 422          # post-release field still refused
        sw = c.post("/api/sensitivity", json={"inputs": VALID, "feature": "budget"})
        assert sw.status_code == 200
        cur = sw.json()["current"]["probabilities"]
        assert all(abs(cur[k] - r.json()["probabilities"][k]) < 1e-3 for k in cur)           # what-if 'current' point == /api/predict
