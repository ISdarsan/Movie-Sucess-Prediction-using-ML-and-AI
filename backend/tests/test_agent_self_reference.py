"""A film that is already in the dataset must never be part of its own cohort or comparables. Real dataset, no network, no LLM."""
import time

import pytest
from fastapi.testclient import TestClient

from backend.agent import config, ollama_client, orchestrator, research
from backend.agent.tools import ComparablesArgs, CohortArgs, cohort_tool, comparables_tool
from backend.app import data, insight
from backend.app.main import app

INCEPTION = {"genres": ["Action", "Adventure", "Sci-Fi", "Thriller"], "country": "USA", "content_rating": "PG-13", "year": 2010, "duration": 148.0}
INCEPTION_REQUEST = {"duration": 148.0, "title_year": 2010, "genres": INCEPTION["genres"], "country": "USA", "content_rating": "PG-13",
                     "budget": 160000000.0, "color": "Color", "aspect_ratio": 2.35}


def titles(rows):
    return [(r["title"], int(r["year"])) for r in rows]


# 1. Inception
def test_inception_is_not_its_own_comparable_or_part_of_its_cohort():
    before_c, before_r = cohort_tool(CohortArgs(**INCEPTION)), comparables_tool(ComparablesArgs(**INCEPTION))
    assert ("Inception", 2010) in titles(before_r) and before_c["final"]["n"] == 7 and "excluded_self" not in before_c    # unchanged without a title
    c, r = cohort_tool(CohortArgs(**INCEPTION, title="Inception")), comparables_tool(ComparablesArgs(**INCEPTION, title="Inception"))
    assert ("Inception", 2010) not in titles(r)
    assert titles(r)[:7] == [t for t in titles(before_r) if t != ("Inception", 2010)]          # the others keep their order; the next-best film fills the gap
    assert len(r) == 8
    assert c["excluded_self"] is True
    assert [s["n"] for s in c["stages"]] == [4918, 63, 50, 35, 27, 6]                          # every stage is one smaller: Inception was in all of them
    assert c["final"]["counts"] == {"Flop": 0, "Average": 2, "Hit": 4} and c["final"]["mean_score"] != before_c["final"]["mean_score"]
    assert c["thin"] is True and not c["empty"]


def test_title_matching_is_normalised_but_still_needs_the_year():
    for t in ("inception", "  INCEPTION  ", "INCEPTION"):
        assert cohort_tool(CohortArgs(**INCEPTION, title=t))["excluded_self"] is True
    # 4. ambiguous: same title in another year, a different title, or no usable year: nothing is removed
    for kwargs in ({"title": "Inception", "year": 2011}, {"title": "Inception", "year": 2009}, {"title": "Inception 2"}, {"title": "Incep"}, {"title": ""}, {"title": "   "}):
        c = cohort_tool(CohortArgs(**{**INCEPTION, **kwargs}))
        assert c["excluded_self"] is False and c["stages"][0]["n"] == 4919, kwargs
    assert cohort_tool(CohortArgs(**{**INCEPTION, "year": None, "title": "Inception"}))["stages"][0]["n"] == 4919
    later = comparables_tool(ComparablesArgs(**{**INCEPTION, "year": 2011, "title": "Inception"}))
    assert ("Inception", 2010) in titles(later)                                                 # for a 2011 query it is a valid EARLIER film


# 2. other films that are in the dataset
@pytest.mark.parametrize("name,year", [("Armageddon", 1998), ("The Core", 2003), ("Avatar", 2009)])
def test_other_films_in_the_dataset_are_excluded_too(name, year):
    df = data.movies()
    row = df[(df["title"] == name) & (df["year"] == year)].iloc[0]
    traits = {"genres": list(row["genre_list"]), "country": row["country"], "content_rating": row["content_rating"], "year": int(year), "duration": float(row["duration"])}
    base, own = cohort_tool(CohortArgs(**traits)), cohort_tool(CohortArgs(**traits, title=name))
    assert base["final"]["n"] - own["final"]["n"] == 1
    assert own["final"]["counts"][row["Classify"]] == base["final"]["counts"][row["Classify"]] - 1
    assert (name, year) in titles(comparables_tool(ComparablesArgs(**traits)))
    assert (name, year) not in titles(comparables_tool(ComparablesArgs(**traits, title=name)))


# 3. a film that is not in the dataset
def test_a_film_not_in_the_dataset_behaves_exactly_as_before():
    base_c, base_r = cohort_tool(CohortArgs(**INCEPTION)), comparables_tool(ComparablesArgs(**INCEPTION))
    c = cohort_tool(CohortArgs(**INCEPTION, title="A Film That Does Not Exist"))
    r = comparables_tool(ComparablesArgs(**INCEPTION, title="A Film That Does Not Exist"))
    assert r == base_r and c.pop("excluded_self") is False and c == base_c


# 5. statistics are computed after the exclusion
def test_cohort_statistics_equal_an_independent_recount_without_the_film():
    df = data.movies()
    rest = df[~((df["title"] == "Inception") & (df["year"] == 2010))]
    m = rest[rest["genre_list"].map(lambda g: all(x in g for x in INCEPTION["genres"])) & (rest["country"] == "USA") & (rest["content_rating"] == "PG-13")
             & rest["year"].between(2000, 2020) & rest["duration"].between(128, 168)]
    c = cohort_tool(CohortArgs(**INCEPTION, title="Inception"))["final"]
    assert c["n"] == len(m)
    assert c["counts"] == {k: int((m["Classify"] == k).sum()) for k in ("Flop", "Average", "Hit")}
    assert c["mean_score"] == round(float(m["imdb_score"].mean()), 2)


def test_a_cohort_emptied_by_the_exclusion_is_reported_as_empty_not_hidden():
    df = data.movies()
    only = df[df["country"] == "Aruba"]                          # a country with a single film in the dataset
    assert len(only) == 1
    r = only.iloc[0]
    traits = {"genres": list(r["genre_list"]), "country": "Aruba", "year": int(r["year"])}
    base, own = cohort_tool(CohortArgs(**traits)), cohort_tool(CohortArgs(**traits, title=r["title"]))
    assert base["final"]["n"] == 1 and own["final"]["n"] == 0 and own["empty"] is True and own["thin"] is False


def test_the_public_cohort_function_is_unchanged_without_a_title():
    assert insight.cohort(INCEPTION["genres"], "USA", "PG-13", 2010, 148.0) == insight.cohort(INCEPTION["genres"], "USA", "PG-13", 2010, 148.0, None)
    assert "excluded_self" not in insight.cohort(["Drama"], None, None, None, None)


def test_the_dataset_itself_still_contains_the_film():
    df = data.movies()
    assert len(df) == 4919 and ((df["title"] == "Inception") & (df["year"] == 2010)).sum() == 1


# 6. the prediction is exactly the same
@pytest.fixture(scope="module")
def loaded_model():
    with TestClient(app):                                        # starts the app lifespan, which loads the saved Random Forest
        yield


def _run(monkeypatch, tmp_path, request):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "runs.sqlite")
    monkeypatch.setattr(research, "run", lambda *a, **k: research.disabled_block("test"))

    def down(*a, **k):
        raise ollama_client.OllamaError("unreachable")
    monkeypatch.setattr(ollama_client, "chat_json", down)
    return orchestrator._run("t" * 12, request, [], time.perf_counter())


def test_the_run_prediction_is_identical_with_or_without_a_title_and_matches_the_pre_fix_scores(loaded_model, monkeypatch, tmp_path):
    with_title = _run(monkeypatch, tmp_path, {**INCEPTION_REQUEST, "research_title": "Inception"})
    without = _run(monkeypatch, tmp_path, dict(INCEPTION_REQUEST))
    assert with_title["prediction"] == without["prediction"]
    assert with_title["prediction"]["label"] == "Hit"
    assert with_title["prediction"]["probabilities"] == {"Flop": 0.0209, "Average": 0.3061, "Hit": 0.673}      # recorded before the fix
    assert "research_title" not in with_title["inputs"] and "title" not in with_title["prediction"]["inputs"]
    ev = with_title["evidence"]
    assert ev["cohort"]["excluded_self"] is True and ev["cohort"]["final"]["n"] == 6 and ("Inception", 2010) not in titles(ev["comparables"])
    assert "excluded_self" not in without["evidence"]["cohort"] and ("Inception", 2010) in titles(without["evidence"]["comparables"])
    facts = " ".join(f.text for f in orchestrator.report.build_facts(with_title["prediction"], ev))      # the report's facts use the post-exclusion numbers
    assert "6 films in the dataset" in facts and "Inception (2010)" not in facts
