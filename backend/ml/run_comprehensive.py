"""Comprehensive pre-release model comparison.

    .venv/Scripts/python -m backend.ml.run_comprehensive            # full study
    .venv/Scripts/python -m backend.ml.run_comprehensive --quick    # smoke test, writes nothing permanent

SAME problem as backend/ml/run_prerelease.py (same cleaned data, 9 pre-release inputs, target, stratified 80/20 split with seed 42,
preprocessing fitted inside every training fold). What is new is breadth and a stricter, pre-declared selection rule.

PROTOCOL (fixed before any test number is computed)
  Stage 1  Screening.   Every algorithm x imbalance-strategy configuration, 5-fold x 3-repeat stratified CV on the TRAINING set (seed 42).
  Stage 2  Tuning.      The best configuration of each of the 5 strongest ALGORITHMS is tuned (randomised search, 5-fold, macro F1).
  Stage 3  Finalists.   The top-8 untuned configurations, the 5 tuned ones and the currently deployed LightGBM are re-scored on FRESH
                        folds (seed 44), identical for all, so they are compared on exactly the same splits and winner's-curse optimism
                        from stages 1-2 is not carried into the comparison.
  RULE     best = highest mean macro F1 on the stage-3 folds.  A finalist is "not distinguishable from best" if the Nadeau-Bengio
           corrected paired t-statistic of (best - finalist) is below 2.0.  Among the indistinguishable set choose the SIMPLEST model
           (fixed complexity ranking below); ties -> lower fold-to-fold std of macro F1 (more stable); then higher balanced accuracy.
           The test set, the temporal evaluation and the regression experiment play NO part in selection.
  Report   After the choice is frozen: held-out test metrics for every finalist and baseline, temporal evaluation, confusion matrices.

COMPLEXITY RANK (lower = simpler): 0 dummy · 1 logistic regression, LDA · 2 decision tree, k-NN · 3 SVM · 4 random forest, extra trees,
AdaBoost · 5 gradient boosting (sklearn GB, HistGB, LightGBM, XGBoost, CatBoost).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import features as F
from .estimators import WeightedClassifier

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
warnings.filterwarnings("ignore")
SEED_SCREEN, SEED_SEARCH, SEED_FINAL = 42, 42, 44
T_STAT_LIMIT = 2.0

# key -> (label, library, complexity rank, wrapped-in-WeightedClassifier)
ALGOS = {
    "dummy": ("Dummy baseline", "scikit-learn", 0, False),
    "logreg": ("Logistic Regression", "scikit-learn", 1, False),
    "lda": ("Linear Discriminant Analysis", "scikit-learn", 1, False),
    "dt": ("Decision Tree", "scikit-learn", 2, False),
    "knn": ("k-Nearest Neighbours", "scikit-learn", 2, False),
    "svm": ("Support Vector Machine (RBF)", "scikit-learn", 3, False),
    "rf": ("Random Forest", "scikit-learn", 4, False),
    "et": ("Extra Trees", "scikit-learn", 4, False),
    "ada": ("AdaBoost", "scikit-learn", 4, True),
    "gb": ("Gradient Boosting", "scikit-learn", 5, True),
    "hgb": ("HistGradientBoosting", "scikit-learn", 5, False),
    "lgbm": ("LightGBM", "lightgbm", 5, False),
    "xgb": ("XGBoost", "xgboost", 5, True),
    "cat": ("CatBoost", "catboost", 5, True),
}
STRATEGIES = {
    "dummy": ["majority", "stratified", "uniform"],
    "logreg": ["none", "weights", "oversample", "smote"], "lda": ["none", "weights"], "dt": ["none", "weights", "oversample"],
    "knn": ["none", "oversample"], "svm": ["none", "weights", "oversample"],
    "rf": ["none", "weights", "oversample", "smote"], "et": ["none", "weights", "oversample"],
    "ada": ["none", "weights"], "gb": ["none", "weights", "oversample"],
    "hgb": ["none", "weights", "oversample", "smote"], "lgbm": ["none", "weights", "oversample", "smote"],
    "xgb": ["none", "weights", "oversample"], "cat": ["none", "weights"],
}
STRAT_LABEL = {"none": "unweighted", "weights": "class weights", "oversample": "oversampling (in fold)", "smote": "SMOTE (in fold)",
               "majority": "majority class", "stratified": "random, class proportions", "uniform": "random, uniform"}


def base_estimator(algo: str, strat: str, n_est: int):
    w = strat == "weights"
    cw = "balanced" if w else None
    if algo == "dummy":
        return DummyClassifier(strategy={"majority": "most_frequent", "stratified": "stratified", "uniform": "uniform"}[strat], random_state=F.SEED)
    if algo == "logreg":
        return LogisticRegression(max_iter=4000, class_weight=cw, random_state=F.SEED)
    if algo == "lda":
        return LinearDiscriminantAnalysis(priors=[1 / 3] * 3 if w else None)          # uniform priors = the LDA analogue of class weighting
    if algo == "dt":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight=cw, random_state=F.SEED)
    if algo == "knn":
        return KNeighborsClassifier(n_neighbors=25, weights="distance")
    if algo == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight=cw, random_state=F.SEED)
    if algo == "rf":
        return RandomForestClassifier(n_estimators=n_est, min_samples_leaf=2, class_weight=cw, random_state=F.SEED, n_jobs=1)
    if algo == "et":
        return ExtraTreesClassifier(n_estimators=n_est, min_samples_leaf=2, class_weight=cw, random_state=F.SEED, n_jobs=1)
    if algo == "ada":
        return WeightedClassifier(AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=max(30, n_est // 2), learning_rate=0.5, random_state=F.SEED), balance=w)
    if algo == "gb":
        return WeightedClassifier(GradientBoostingClassifier(n_estimators=max(30, n_est // 2), learning_rate=0.06, max_depth=3, subsample=0.8, random_state=F.SEED), balance=w)
    if algo == "hgb":
        return HistGradientBoostingClassifier(class_weight=cw, max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20, random_state=F.SEED)
    if algo == "lgbm":
        return LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=20, class_weight=cw, random_state=F.SEED, verbosity=-1, n_jobs=1)
    if algo == "xgb":
        return WeightedClassifier(XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, eval_metric="mlogloss", tree_method="hist", random_state=F.SEED),
                                  balance=w, encode_labels=True)
    if algo == "cat":
        return WeightedClassifier(__import__("catboost").CatBoostClassifier(iterations=300, depth=6, learning_rate=0.06, loss_function="MultiClass", verbose=0, thread_count=1, allow_writing_files=False,
                                                                            random_seed=F.SEED, auto_class_weights="Balanced" if w else None), balance=False)
    raise KeyError(algo)


def build(algo: str, strat: str, n_est: int):
    est = base_estimator(algo, strat, n_est)
    prep = F.build_preprocessor(False)
    if strat == "oversample":
        return ImbPipeline([("prep", prep), ("sampler", RandomOverSampler(random_state=F.SEED)), ("clf", est)])
    if strat == "smote":
        return ImbPipeline([("prep", prep), ("sampler", SMOTE(random_state=F.SEED, k_neighbors=5)), ("clf", est)])
    return Pipeline([("prep", prep), ("clf", est)])


def cname(algo: str, strat: str) -> str:
    return f"{ALGOS[algo][0]} · {STRAT_LABEL[strat]}"


SPACES = {
    "logreg": {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "lda": {"solver": ["lsqr"], "shrinkage": [None, 0.1, 0.3, 0.5, 0.8, "auto"]},
    "dt": {"max_depth": [3, 4, 6, 8, 12], "min_samples_leaf": [5, 10, 20, 40]},
    "knn": {"n_neighbors": [5, 15, 25, 50, 100], "weights": ["uniform", "distance"], "p": [1, 2]},
    "svm": {"C": [0.1, 0.3, 1, 3, 10], "gamma": ["scale", 0.003, 0.01, 0.03]},
    "rf": {"n_estimators": [300, 500], "max_depth": [None, 8, 12, 20], "min_samples_leaf": [1, 2, 3, 5, 10], "max_features": ["sqrt", 0.3, 0.5]},
    "et": {"n_estimators": [300, 500], "max_depth": [None, 8, 12, 20], "min_samples_leaf": [1, 2, 3, 5, 10], "max_features": ["sqrt", 0.3, 0.5]},
    "ada": {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.1, 0.3, 0.5, 1.0], "estimator__max_depth": [1, 2, 3]},
    "gb": {"n_estimators": [100, 200, 300], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4], "subsample": [0.6, 0.8, 1.0], "min_samples_leaf": [5, 20, 50]},
    "hgb": {"learning_rate": [0.02, 0.04, 0.06, 0.1], "max_leaf_nodes": [6, 10, 15, 31], "min_samples_leaf": [10, 20, 40, 80], "l2_regularization": [0.0, 1.0, 10.0], "max_iter": [100, 200, 300]},
    "lgbm": {"n_estimators": [150, 250, 400], "learning_rate": [0.02, 0.04, 0.08], "num_leaves": [7, 15, 31], "min_child_samples": [10, 20, 40], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0]},
    "xgb": {"n_estimators": [150, 250, 400], "max_depth": [2, 3, 4, 6], "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 10], "reg_lambda": [1, 5, 20]},
    "cat": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.06, 0.1], "l2_leaf_reg": [1, 3, 10], "iterations": [200, 300, 500]},
}


def prefixed(algo: str) -> dict:
    p = "clf__base__" if ALGOS[algo][3] else "clf__"
    return {p + k: v for k, v in SPACES[algo].items()}


def scorer(est, X, y):
    pred = est.predict(X)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=F.CLASSES, zero_division=0)
    out = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": float(np.mean(f1))}
    for i, c in enumerate(F.CLASSES):
        out[f"{c.lower()}_precision"], out[f"{c.lower()}_recall"], out[f"{c.lower()}_f1"] = float(pr[i]), float(rc[i]), float(f1[i])
    return out


METRICS = ["accuracy", "balanced_accuracy", "macro_f1"] + [f"{c.lower()}_{m}" for c in F.CLASSES for m in ("precision", "recall", "f1")]


def cv_eval(pipe, X, y, cv) -> dict:
    r = cross_validate(pipe, X, y, cv=cv, scoring=scorer, n_jobs=-1, error_score="raise")
    out = {m: [round(float(np.mean(r[f"test_{m}"])), 5), round(float(np.std(r[f"test_{m}"])), 5)] for m in METRICS}
    out["fit_seconds"] = round(float(np.mean(r["fit_time"])), 3)
    out["score_seconds"] = round(float(np.mean(r["score_time"])), 3)
    out["folds"] = int(len(r["test_macro_f1"]))
    out["macro_f1_folds"] = [round(float(v), 5) for v in r["test_macro_f1"]]
    return out


def test_metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=F.CLASSES, zero_division=0, output_dict=True)
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4), "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(np.mean([rep[c]["f1-score"] for c in F.CLASSES])), 4),
            "per_class": {c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])} for c in F.CLASSES},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=F.CLASSES).tolist()}


def corrected_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Nadeau-Bengio corrected resampled t for paired fold scores (5-fold: test/train ratio 0.25). Returns mean diff, SE, t."""
    d = a - b
    k = len(d)
    se = float(np.sqrt((1 / k + 0.25) * d.var(ddof=1))) if k > 1 and d.var(ddof=1) > 0 else 0.0
    return float(d.mean()), se, float(d.mean() / se) if se > 0 else (0.0 if d.mean() == 0 else float("inf"))


def main(quick: bool | None = None) -> None:
    if quick is None:
        ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); quick = ap.parse_args().quick
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731
    n_est, repeats, n_iter, n_tune = (40, 1, 2, 2) if quick else (300, 3, 12, 5)
    n_top_untuned = 3 if quick else 8

    # ------------------------------------------------------------------ verification of the existing setup
    df, dlog = F.load_clean(str(ROOT / "dataset" / "movie_metadata.csv"))
    train, test = train_test_split(df, test_size=0.2, stratify=df["Classify"], random_state=F.SEED)
    ytr, yte = train["Classify"], test["Classify"]
    cols = F.BASE_INPUTS
    prev = json.loads((ART / "prerelease_results_v1.json").read_text(encoding="utf8")) if (ART / "prerelease_results_v1.json").exists() else json.loads((ART / "prerelease_results.json").read_text(encoding="utf8"))
    schema_prev = json.loads((ART / ("prerelease_schema_v1.json" if (ART / "prerelease_schema_v1.json").exists() else "prerelease_schema.json")).read_text(encoding="utf8"))
    verification = {
        "schema_inputs_equal_feature_set": schema_prev["inputs"] == cols, "excluded_columns_not_inputs": set(F.EXCLUDED).isdisjoint(cols), "target_not_input": not {"imdb_score", "Classify"} & set(cols),
        "split_matches_previous_study": (ytr.value_counts().reindex(F.CLASSES).astype(int).to_dict() == prev["data"]["classes_train"] and yte.value_counts().reindex(F.CLASSES).astype(int).to_dict() == prev["data"]["classes_test"]),
        "stratified": all(abs((ytr == c).mean() - (yte == c).mean()) < 0.01 for c in F.CLASSES), "n_inputs": len(cols),
    }
    assert all(v for k, v in verification.items() if k != "n_inputs"), verification
    log("verification OK", verification)
    Xtr, Xte = train[cols], test[cols]
    dist = {k: v.value_counts().reindex(F.CLASSES).astype(int).to_dict() for k, v in (("all", df["Classify"]), ("train", ytr), ("test", yte))}

    cv1 = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED_SCREEN)
    fl = [(int((ytr.to_numpy()[a] == "Flop").sum()), int((ytr.to_numpy()[b] == "Flop").sum())) for a, b in cv1.split(Xtr, ytr)]
    fold_audit = {"folds": len(fl), "flops_in_training_set": dist["train"]["Flop"], "flops_per_training_fold": [min(a for a, _ in fl), max(a for a, _ in fl)],
                  "flops_per_validation_fold": [min(b for _, b in fl), max(b for _, b in fl)],
                  "conclusion": "Each validation fold holds only a handful of Flops, so fold-level and test-level Flop metrics cannot be reliable; they are reported but not trusted."}
    log("fold audit", fold_audit)

    # ------------------------------------------------------------------ stage 1: screen every configuration
    screening, bad = [], []
    for algo, strats in STRATEGIES.items():
        for strat in strats:
            name = cname(algo, strat)
            try:
                cv = cv_eval(build(algo, strat, n_est), Xtr, ytr, cv1)
                screening.append({"name": name, "algorithm": algo, "algorithm_label": ALGOS[algo][0], "strategy": strat, "strategy_label": STRAT_LABEL[strat], "library": ALGOS[algo][1],
                                  "complexity": ALGOS[algo][2], "ok": True, "cv": cv})
                log(f"S1 {name:58s} macroF1 {cv['macro_f1'][0]:.4f}±{cv['macro_f1'][1]:.3f} bal {cv['balanced_accuracy'][0]:.3f} acc {cv['accuracy'][0]:.3f} flopR {cv['flop_recall'][0]:.2f} fit {cv['fit_seconds']:.1f}s")
            except Exception as e:  # noqa: BLE001 - recorded, never hidden
                bad.append({"name": name, "error": f"{type(e).__name__}: {str(e)[:160]}"})
                log("S1 FAILED", name, type(e).__name__, str(e)[:120])
    ok = [s for s in screening if s["algorithm"] != "dummy"]

    # ------------------------------------------------------------------ stage 2: tune the best config of each of the top-5 algorithms
    best_per_algo = {}
    for s in ok:
        if s["algorithm"] not in best_per_algo or s["cv"]["macro_f1"][0] > best_per_algo[s["algorithm"]]["cv"]["macro_f1"][0]:
            best_per_algo[s["algorithm"]] = s
    top_algos = sorted(best_per_algo, key=lambda a: -best_per_algo[a]["cv"]["macro_f1"][0])[:n_tune]
    log("tuning algorithms:", top_algos)
    tuning, tuned_fitted = [], {}
    for algo in top_algos:
        s = best_per_algo[algo]
        pipe = build(algo, s["strategy"], n_est)
        search = RandomizedSearchCV(pipe, prefixed(algo), n_iter=n_iter, scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=SEED_SEARCH), n_jobs=-1, random_state=F.SEED, refit=True)
        t1 = time.time(); search.fit(Xtr, ytr)
        tname = f"{ALGOS[algo][0]} · {STRAT_LABEL[s['strategy']]} · tuned"
        tuned_fitted[tname] = (algo, s["strategy"], search.best_estimator_)
        tuning.append({"name": tname, "algorithm": algo, "algorithm_label": ALGOS[algo][0], "strategy": s["strategy"], "base_config": s["name"], "n_iter": n_iter, "search_cv_macro_f1": round(float(search.best_score_), 5),
                       "best_params": {k.replace("clf__base__", "").replace("clf__", ""): (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in search.best_params_.items()},
                       "search_space": {k: v for k, v in SPACES[algo].items()}, "seconds": round(time.time() - t1, 1)})
        log(f"S2 {tname:60s} search macroF1 {search.best_score_:.4f} (untuned {s['cv']['macro_f1'][0]:.4f})  {tuning[-1]['best_params']}")

    # ------------------------------------------------------------------ stage 3: finalists on identical fresh folds
    top_untuned = sorted(ok, key=lambda s: -s["cv"]["macro_f1"][0])[:n_top_untuned]
    deployed_name = cname("lgbm", "weights")
    members: dict[str, dict] = {}
    for s in top_untuned:
        members[s["name"]] = {"algo": s["algorithm"], "strat": s["strategy"], "tuned": False, "source": "top untuned", "est": build(s["algorithm"], s["strategy"], n_est)}
    for tname, (algo, strat, est) in tuned_fitted.items():
        members[tname] = {"algo": algo, "strat": strat, "tuned": True, "source": "tuned", "est": est}
    if deployed_name not in members:
        members[deployed_name] = {"algo": "lgbm", "strat": "weights", "tuned": False, "source": "currently deployed", "est": build("lgbm", "weights", n_est)}
    else:
        members[deployed_name]["source"] += " + currently deployed"
    cvF = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED_FINAL)
    for name, m in members.items():
        m["cv_final"] = cv_eval(m["est"], Xtr, ytr, cvF)
        c = m["cv_final"]
        log(f"S3 {name:60s} macroF1 {c['macro_f1'][0]:.4f}±{c['macro_f1'][1]:.3f} bal {c['balanced_accuracy'][0]:.3f} acc {c['accuracy'][0]:.3f}")

    # ------------------------------------------------------------------ the pre-declared rule (no test data touched so far)
    arr = {n: np.array(m["cv_final"]["macro_f1_folds"]) for n, m in members.items()}
    best = max(members, key=lambda n: arr[n].mean())
    paired, equiv = {}, []
    for n in members:
        d, se, t = corrected_t(arr[best], arr[n]) if n != best else (0.0, 0.0, 0.0)
        paired[n] = {"best_minus_this": round(d, 5), "corrected_se": round(se, 5), "t": round(t, 3) if np.isfinite(t) else None}
        if n == best or t < T_STAT_LIMIT:
            equiv.append(n)
    chosen = sorted(equiv, key=lambda n: (ALGOS[members[n]["algo"]][2], members[n]["cv_final"]["macro_f1"][1], -members[n]["cv_final"]["balanced_accuracy"][0]))[0]
    selection = {"rule": __doc__.split("RULE")[1].split("Report")[0].strip(), "best_by_mean": best, "indistinguishable_from_best": equiv, "paired_vs_best": paired, "chosen": chosen,
                 "chosen_complexity_rank": ALGOS[members[chosen]["algo"]][2], "deployed_before": deployed_name,
                 "outcome": "same as deployed" if chosen == deployed_name else "differs from deployed"}
    log("SELECTION (CV only): best by mean =", best, "| indistinguishable set =", equiv, "| CHOSEN =", chosen)

    # ------------------------------------------------------------------ report: test metrics for finalists + baselines
    def fit_and_test(name, est):
        c = clone(est)                                   # identical hyper-parameters, refit on the full training set (also gives honest timings)
        t1 = time.time(); c.fit(Xtr, ytr); fit_s = time.time() - t1
        t1 = time.time(); pred = c.predict(Xte); pred_ms = (time.time() - t1) / len(Xte) * 1000 * 1000
        return c, test_metrics(yte, pred), {"fit_seconds": round(fit_s, 2), "predict_ms_per_1000_films": round(pred_ms, 2)}
    finalists, fitted = [], {}
    for name, m in members.items():
        est_fit, tm, timing = fit_and_test(name, m["est"])
        fitted[name] = est_fit
        finalists.append({"name": name, "algorithm": m["algo"], "algorithm_label": ALGOS[m["algo"]][0], "strategy": m["strat"], "strategy_label": STRAT_LABEL[m["strat"]], "library": ALGOS[m["algo"]][1],
                          "complexity": ALGOS[m["algo"]][2], "tuned": m["tuned"], "source": m["source"], "selected": name == chosen, "deployed_before": name == deployed_name,
                          "cv_final": m["cv_final"], "test": tm, "timing": timing})
        log(f"TEST {name:60s} acc {tm['accuracy']:.3f} bal {tm['balanced_accuracy']:.3f} macroF1 {tm['macro_f1']:.3f} Flop R {tm['per_class']['Flop']['recall']:.2f}")
    baselines = []
    for strat in STRATEGIES["dummy"]:
        name = cname("dummy", strat)
        c, tm, timing = fit_and_test(name, build("dummy", strat, n_est))
        cv = next(s["cv"] for s in screening if s["name"] == name)
        baselines.append({"name": name, "algorithm": "dummy", "algorithm_label": "Dummy baseline", "strategy": strat, "strategy_label": STRAT_LABEL[strat], "complexity": 0, "cv": cv, "test": tm, "timing": timing})
        log(f"TEST baseline {name:44s} acc {tm['accuracy']:.3f} bal {tm['balanced_accuracy']:.3f} macroF1 {tm['macro_f1']:.3f}")

    final = fitted[chosen]
    oof = cross_val_predict(clone(final), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED_FINAL + 1), n_jobs=-1)
    pooled = test_metrics(ytr, oof)

    # ------------------------------------------------------------------ temporal generalisation (reporting only)
    yr = df["title_year"]
    def window(tr_mask, te_mask, label):
        tr_df, te_df = df[tr_mask], df[te_mask]
        maj = tr_df["Classify"].value_counts().idxmax()
        rows = {"label": label, "train_size": int(len(tr_df)), "test_size": int(len(te_df)), "classes_train": tr_df["Classify"].value_counts().reindex(F.CLASSES).fillna(0).astype(int).to_dict(),
                "classes_test": te_df["Classify"].value_counts().reindex(F.CLASSES).fillna(0).astype(int).to_dict(), "majority_class": maj,
                "majority_baseline": test_metrics(te_df["Classify"], np.repeat(maj, len(te_df))), "models": {}}
        for name, m in members.items():
            p = clone(m["est"]).fit(tr_df[cols], tr_df["Classify"])
            rows["models"][name] = test_metrics(te_df["Classify"], p.predict(te_df[cols]))
        return rows
    temporal = {"note": "Preprocessing and models are refit on the earlier period only. Films with a missing year are excluded. Reporting only: not used for selection.",
                "main": window(yr <= 2009, yr >= 2010, "train ≤ 2009 → test ≥ 2010"),
                "rolling": [window(yr <= c, (yr > c) & (yr <= c + 4), f"train ≤ {c} → test {c + 1}–{c + 4}") for c in (2004, 2008, 2012)]}
    log("TEMPORAL main:", {n: (m["accuracy"], m["macro_f1"]) for n, m in temporal["main"]["models"].items() if n == chosen}, "baseline acc", temporal["main"]["majority_baseline"]["accuracy"], "test classes", temporal["main"]["classes_test"])

    # ------------------------------------------------------------------ separate experiment: regression on the IMDb score, converted to classes
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    reg_rows = []
    ytr_score = train["imdb_score"].to_numpy()
    for rname, reg in (("Ridge regression", Ridge(alpha=3.0)), ("Random Forest regressor", RandomForestRegressor(n_estimators=n_est, min_samples_leaf=3, random_state=F.SEED, n_jobs=1)),
                       ("HistGradientBoosting regressor", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, random_state=F.SEED))):
        scores = []
        for a, b in cv1.split(Xtr, ytr):
            p = Pipeline([("prep", F.build_preprocessor(False)), ("reg", clone(reg))]).fit(Xtr.iloc[a], ytr_score[a])
            pred_cls = pd.cut(np.clip(p.predict(Xtr.iloc[b]), 0, 9.99), bins=F.TARGET_BINS, labels=F.CLASSES, right=False, include_lowest=True).astype(str)
            scores.append(scorer(type("P", (), {"predict": staticmethod(lambda X, _p=pred_cls: _p)})(), None, ytr.iloc[b]))
        reg_rows.append({"name": rname, **{m: [round(float(np.mean([s[m] for s in scores])), 5), round(float(np.std([s[m] for s in scores])), 5)] for m in ("accuracy", "balanced_accuracy", "macro_f1", "flop_recall")}})
        log(f"REG {rname:34s} macroF1 {reg_rows[-1]['macro_f1'][0]:.4f} bal {reg_rows[-1]['balanced_accuracy'][0]:.3f}")
    regression = {"definition": "Predict the continuous IMDb score from the same 9 inputs, then convert with the target's own bins (below 3 = Flop, 3 to below 6 = Average, 6 and above = Hit). Same CV folds as stage 1. A separate experiment: not eligible for selection.", "rows": reg_rows}

    out = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "quick": quick, "seed": F.SEED,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, **{k: __import__(k).__version__ for k in ("lightgbm", "xgboost", "catboost", "imblearn")}},
        "protocol": __doc__, "verification": verification, "data": {**dlog, "classes_all": dist["all"], "classes_train": dist["train"], "classes_test": dist["test"], "train_size": int(len(train)), "test_size": int(len(test)),
                                                                    "split": "train_test_split(test_size=0.2, stratify=Classify, random_state=42) after cleaning (unchanged)"},
        "feature_set": {"inputs": cols, "use_people": False}, "audit": [a.__dict__ for a in F.AUDIT], "fold_audit": fold_audit,
        "algorithms": [{"key": k, "label": v[0], "library": v[1], "complexity": v[2], "strategies": [STRAT_LABEL[s] for s in STRATEGIES[k]], "evaluated": any(s["algorithm"] == k for s in screening),
                        "note": ("" if any(s["algorithm"] == k for s in screening) else "not evaluated: every configuration failed")} for k, v in ALGOS.items()],
        "skipped": [{"algorithm": "Linear Regression", "reason": "A regressor, not a classifier. Evaluated only as a separate experiment (regression on the IMDb score, converted to classes) and not eligible for selection."}] + bad,
        "screening": screening, "tuning": tuning, "finalists": finalists, "baselines": baselines, "selection": selection, "selected_pooled_oof_train": pooled,
        "temporal": temporal, "regression_experiment": regression,
        "cv_protocols": {"screening": f"RepeatedStratifiedKFold(5 x {repeats}, seed {SEED_SCREEN}) on the training set", "tuning": f"RandomizedSearchCV({n_iter} draws, StratifiedKFold(5, seed {SEED_SEARCH}), macro F1)",
                         "finalists": f"RepeatedStratifiedKFold(5 x {repeats}, seed {SEED_FINAL}) — fresh folds, identical for every finalist"},
        "deployed_before": {"name": deployed_name, "previous_study_test": prev["models"][prev["selected"]]["test"]},
    }
    if quick:
        log("quick run complete; nothing saved"); return

    # candidate artifacts: the live files are NOT touched here (promotion happens only after the integration tests pass)
    joblib.dump(final, ROOT / "pre_release_movie_classifier.candidate.joblib")
    num = {c: {"min": float(train[c].min()), "max": float(train[c].max()), "p01": float(train[c].quantile(.01)), "p99": float(train[c].quantile(.99)), "median": float(train[c].median()), "missing_pct": round(float(train[c].isna().mean() * 100), 1)} for c in F.NUMERIC}
    schema = {"model_file": "pre_release_movie_classifier.joblib", "classes": F.CLASSES, "inputs": cols, "use_people": False, "numeric": num, "categorical": {c: sorted(train[c].dropna().unique().tolist()) for c in F.CATEGORICAL},
              "genres": sorted({g for s in train[F.GENRES].dropna() for g in s.split("|") if g}), "people_columns": [], "defaults_when_blank": {"numeric_median": {c: num[c]["median"] for c in num}, "categorical": "Unknown (own category)"},
              "excluded": [a.__dict__ for a in F.AUDIT if a.used == "no"]}
    (ART / "prerelease_schema.candidate.json").write_text(json.dumps(schema, indent=2), encoding="utf8")
    (ART / "comprehensive_results.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    log("saved candidate model, candidate schema and comprehensive_results.json")


if __name__ == "__main__":
    sys.exit(main())
