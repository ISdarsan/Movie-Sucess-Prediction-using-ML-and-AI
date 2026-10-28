"""API integration tests (pre-release model). Run from the project root: .venv/Scripts/python -m pytest backend/tests -q"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.config import ARTIFACTS, MODEL_PATH
from backend.app.main import app

VALID = {"duration": 120, "title_year": 2012, "genres": ["Drama", "Romance"], "country": "USA", "content_rating": "PG-13",
         "budget": 40_000_000, "language": "English", "color": "Color"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["model"]["stage"] == "pre-release" and j["dataset"]["rows"] == 4919
    assert j["model"]["n_input_features"] == 9


def test_prediction_matches_direct_pipeline(client):
    r = client.post("/api/predict", json=VALID)
    assert r.status_code == 200, r.text
    j = r.json()
    pipe = joblib.load(MODEL_PATH)
    row = {c: np.nan for c in pipe.named_steps["prep"].feature_names_in_}
    row.update({k: v for k, v in VALID.items() if k != "genres"}); row["genres"] = "Drama|Romance"
    df = pd.DataFrame([row])
    assert j["label"] == str(pipe.predict(df)[0])
    for c, p in zip(pipe.classes_, pipe.predict_proba(df)[0]):
        assert abs(j["probabilities"][str(c)] - p) < 1e-4
    assert abs(sum(j["probabilities"].values()) - 1) < 1e-3
    assert j["model"]["stage"] == "pre-release"
    assert any(i["field"] == "aspect_ratio" for i in j["imputed"])  # blanks are disclosed


@pytest.mark.parametrize("patch,field", [({"duration": -5}, "duration"), ({"title_year": 1500}, "title_year"), ({"genres": []}, "genres"), ({"budget": -1}, "budget")])
def test_invalid_values_rejected(client, patch, field):
    r = client.post("/api/predict", json={**VALID, **patch})
    assert r.status_code == 422 and r.json()["fields"][0]["field"] == field


def test_missing_required(client):
    r = client.post("/api/predict", json={k: v for k, v in VALID.items() if k != "duration"})
    assert r.status_code == 422 and r.json()["fields"][0]["field"] == "duration"


@pytest.mark.parametrize("leaky", ["num_voted_users", "gross", "num_user_for_reviews", "num_critic_for_reviews", "movie_facebook_likes", "imdb_score", "cast_total_facebook_likes"])
def test_post_release_fields_are_refused(client, leaky):
    """The strict pre-release contract: a post-release column in the request is an error, never silently used or ignored."""
    r = client.post("/api/predict", json={**VALID, leaky: 123})
    assert r.status_code == 422 and r.json()["fields"][0]["field"] == leaky


def test_unseen_category_is_scored_with_a_warning(client):
    r = client.post("/api/predict", json={**VALID, "country": "Atlantis"})
    assert r.status_code == 200 and any(w["field"] == "country" for w in r.json()["warnings"])


def test_minimal_input_works(client):
    body = {k: VALID[k] for k in ("duration", "title_year", "genres", "country", "content_rating")}
    j = client.post("/api/predict", json=body).json()
    assert {i["field"] for i in j["imputed"]} == {"budget", "language", "color", "aspect_ratio"}


def test_schema_lists_exclusions(client):
    s = client.get("/api/predict/schema").json()
    assert set(s["columns"]) == {"budget", "duration", "title_year", "aspect_ratio", "country", "language", "content_rating", "color", "genres"}
    assert {"gross", "num_voted_users", "imdb_score"} <= {e["column"] for e in s["excluded"]}


def test_overview_and_filters(client):
    j = client.get("/api/overview").json()
    assert j["n_movies"] == 4919 and j["kpis"]["class_counts"] == {"Flop": 41, "Average": 1383, "Hit": 3495}
    f = client.get("/api/overview", params={"genre": "Drama", "year_min": 2000}).json()
    assert 0 < f["n_movies"] < 4919 and f["filters"]
    assert client.get("/api/overview", params={"country": "Atlantis"}).json()["empty"] is True


def test_metrics_file_matches_saved_model(client):
    h = client.get("/api/model-metrics").json()
    chosen = json.loads((ARTIFACTS / "comprehensive_results.json").read_text(encoding="utf8"))["selection"]["chosen"]
    assert h["stage"] == "pre-release" and h["selected_model"]["name"] == chosen
    assert h["held_out"]["selected"] and sum(sum(r) for r in h["held_out"]["confusion_matrix"]) == h["evaluation"]["test_size"]


def test_explain_shapes(client):
    e = client.get("/api/explain").json()
    m = e["method"]
    assert m["sample_size"] == 200 and m["values_shape"] == [200, m["n_transformed_features"], 3]
    assert not any(k in b["feature"] for b in e["shap"]["Hit"]["bar"] for k in ("votes", "gross", "likes"))


def test_explorer(client):
    j = client.get("/api/movies", params={"q": "avatar", "page_size": 5}).json()
    assert j["total"] >= 1 and j["rows"][0]["title"].lower().startswith("avatar")
    assert client.get("/api/movies", params={"sort": "evil"}).status_code == 400
    assert client.get("/api/movies/0").json()["title"]
    assert client.get("/api/movies/999999").status_code == 404
    assert client.get("/api/movies/export.csv", params={"genre": "Western"}).text.startswith("title,")


def test_wall_is_every_film_sorted_by_score(client):
    w = client.get("/api/wall").json()
    assert w["n"] == 4919 == len(w["titles"]) == len(w["classes"])
    assert w["scores"] == sorted(w["scores"])
    assert [w["classes"].count(i) for i in (0, 1, 2)] == [41, 1383, 3495]


def test_cohort_narrows_and_matches_dataset(client):
    full = client.get("/api/cohort").json()
    assert full["final"]["n"] == 4919 and full["final"]["counts"] == {"Flop": 41, "Average": 1383, "Hit": 3495}
    j = client.get("/api/cohort", params={"genres": "Drama,Romance", "country": "USA", "content_rating": "PG-13", "year": 2005, "duration": 110}).json()
    ns = [s["n"] for s in j["stages"]]
    assert ns == sorted(ns, reverse=True) and len(ns) == 6
    assert sum(j["final"]["counts"].values()) == j["final"]["n"]
    assert client.get("/api/cohort", params={"country": "Atlantis"}).json()["empty"] is True


def test_sensitivity_is_the_real_model(client):
    r = client.post("/api/sensitivity", json={"inputs": VALID, "feature": "budget"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["values"]) >= 10 and j["values"] == sorted(j["values"])
    for i in range(len(j["values"])):
        assert abs(sum(j["probabilities"][c][i] for c in ("Flop", "Average", "Hit")) - 1) < 1e-3
    p = client.post("/api/predict", json=VALID).json()["probabilities"]      # the 'current' point must equal /api/predict
    assert all(abs(j["current"]["probabilities"][c] - p[c]) < 1e-3 for c in p)
    assert client.post("/api/sensitivity", json={"inputs": VALID, "feature": "gross"}).status_code == 422      # post-release: not sweepable
