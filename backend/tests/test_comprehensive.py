"""Integrity tests for the comprehensive model comparison (backend/ml/run_comprehensive.py).
Run from the project root: .venv/Scripts/python -m pytest backend/tests/test_comprehensive.py -q"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from backend.ml import features as F
from backend.ml import run_comprehensive as RC

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
REQUIRED = {"dummy", "logreg", "lda", "dt", "rf", "et", "hgb", "gb", "ada", "lgbm", "xgb", "cat", "svm", "knn"}


@pytest.fixture(scope="module")
def R():
    return json.loads((ART / "comprehensive_results.json").read_text(encoding="utf8"))


def test_every_requested_algorithm_was_actually_executed(R):
    done = {s["algorithm"] for s in R["screening"] if s["ok"]}
    assert REQUIRED <= done, REQUIRED - done
    assert {a["key"] for a in R["algorithms"] if a["evaluated"]} == done
    assert all(s["ok"] for s in R["screening"])                               # nothing silently failed
    assert [s.get("algorithm") for s in R["skipped"]] == ["Linear Regression"]  # the only exclusion, and it is explained
    assert all("cv" in s and np.isfinite(s["cv"]["macro_f1"][0]) for s in R["screening"])


def test_setup_is_the_same_validated_problem(R):
    assert all(v for k, v in R["verification"].items() if k != "n_inputs") and R["verification"]["n_inputs"] == 9
    assert R["feature_set"]["inputs"] == F.BASE_INPUTS
    df, _ = F.load_clean(str(ROOT / "dataset" / "movie_metadata.csv"))
    tr, te = train_test_split(df, test_size=0.2, stratify=df["Classify"], random_state=F.SEED)
    assert R["data"]["classes_train"] == tr.Classify.value_counts().reindex(F.CLASSES).astype(int).to_dict()
    assert R["data"]["classes_test"] == te.Classify.value_counts().reindex(F.CLASSES).astype(int).to_dict()


def test_fold_class_counts_are_audited_and_flop_is_flagged_as_unreliable(R):
    fa = R["fold_audit"]
    assert fa["flops_per_validation_fold"][1] <= 8 and "cannot be reliable" in fa["conclusion"]


def test_selection_is_reproducible_from_the_recorded_fold_scores(R):
    fins = {f["name"]: f for f in R["finalists"]}
    folds = {n: np.array(f["cv_final"]["macro_f1_folds"]) for n, f in fins.items()}
    assert len({len(v) for v in folds.values()}) == 1 and len(next(iter(folds.values()))) == 15       # identical 5x3 folds for every finalist
    best = max(folds, key=lambda n: folds[n].mean())
    equiv = [n for n in folds if n == best or RC.corrected_t(folds[best], folds[n])[2] < RC.T_STAT_LIMIT]
    chosen = sorted(equiv, key=lambda n: (RC.ALGOS[fins[n]["algorithm"]][2], fins[n]["cv_final"]["macro_f1"][1], -fins[n]["cv_final"]["balanced_accuracy"][0]))[0]
    assert (best, chosen) == (R["selection"]["best_by_mean"], R["selection"]["chosen"])
    assert sorted(equiv) == sorted(R["selection"]["indistinguishable_from_best"])


def test_selection_code_never_reads_test_data():
    src = (ROOT / "backend" / "ml" / "run_comprehensive.py").read_text(encoding="utf8")
    start, end = src.index("the pre-declared rule (no test data touched so far)"), src.index("report: test metrics for finalists")
    assert 0 < start < end
    block = src[start:end]
    assert not re.search(r"\b(Xte|yte|test_metrics|temporal|fit_and_test)\b", block), "the selection block touches test/temporal data"


def test_only_finalists_and_baselines_have_test_metrics(R):
    assert all("test" not in s for s in R["screening"])
    assert all("test" in f for f in R["finalists"]) and all("test" in b for b in R["baselines"])
    assert R["selection"]["chosen"] in {f["name"] for f in R["finalists"]} and R["selection"]["deployed_before"] in {f["name"] for f in R["finalists"]}


def test_resampling_is_a_step_inside_the_cv_pipeline_never_applied_before_splitting():
    for strat in ("oversample", "smote"):
        p = RC.build("rf", strat, 20)
        assert [n for n, _ in p.steps] == ["prep", "sampler", "clf"]
    src = (ROOT / "backend" / "ml" / "run_comprehensive.py").read_text(encoding="utf8")
    assert "fit_resample" not in src


def test_deployed_config_reproduces_its_previous_study_numbers(R):
    prev = json.loads((ART / "prerelease_results_v1.json").read_text(encoding="utf8"))
    old = prev["models"][prev["selected"]]["test"]
    new = next(f for f in R["finalists"] if f["deployed_before"])["test"]
    assert (old["accuracy"], old["balanced_accuracy"], old["macro_f1"]) == (new["accuracy"], new["balanced_accuracy"], new["macro_f1"])


def test_temporal_windows_train_strictly_before_they_test_and_report_baselines(R):
    for wd in [R["temporal"]["main"]] + R["temporal"]["rolling"]:
        assert wd["majority_baseline"]["accuracy"] > 0 and sum(wd["classes_test"].values()) == wd["test_size"]
        assert R["selection"]["chosen"] in wd["models"]
    assert sum(R["temporal"]["main"]["classes_test"].values()) == R["temporal"]["main"]["test_size"]


def test_weighted_classifier_computes_weights_from_the_fit_labels_only():
    from backend.ml.estimators import WeightedClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3)); y = np.array(["Hit"] * 100 + ["Flop"] * 20)
    m = WeightedClassifier(GradientBoostingClassifier(n_estimators=10, random_state=0), balance=True).fit(X, y)
    assert list(m.classes_) == ["Flop", "Hit"] and m.predict_proba(X).shape == (120, 2)
