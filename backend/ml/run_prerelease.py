"""FIRST pre-release study (kept for the record; its outputs are the *_v1 files and it can no longer overwrite the live model).
The current model choice comes from backend/ml/run_comprehensive.py.

Run from the project root:

    .venv/Scripts/python -m backend.ml.run_prerelease            # full study (~10-20 min)
    .venv/Scripts/python -m backend.ml.run_prerelease --quick    # smoke test (1-2 min), writes nothing permanent

Protocol (fixed in advance, in this order):
  1. Clean data -> stratified 80/20 split. The TEST SET IS NOT TOUCHED until step 7.
  2. Feature-set ablation and a reference run with the old post-release columns (CV only).
  3. Every candidate algorithm: 5-fold x 3-repeat stratified CV on the TRAINING set (macro F1 is the selection metric).
  4. The two best candidates are tuned (randomised search, 5-fold, macro F1) on the training set only.
  5. Tuned candidates are re-estimated on FRESH folds (different seed) to reduce selection optimism.
  6. SELECTION RULE (frozen before step 7): highest fresh-CV macro F1; if two models are within 0.005, prefer higher
     balanced accuracy, then the simpler model.  No test-set number is used.
  7. All candidates are scored once on the test set (reporting only), the selected one is saved.
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
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
                             make_scorer, recall_score)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import features as F

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
warnings.filterwarnings("ignore")


def flop_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=["Flop"], average="macro", zero_division=0)


SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro", "flop_recall": make_scorer(flop_recall)}


def cv_eval(pipe, X, y, cv, n_jobs=-1) -> dict:
    r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs, error_score="raise")
    out = {k: [float(np.mean(r[f"test_{k}"])), float(np.std(r[f"test_{k}"]))] for k in SCORING}
    out["fit_seconds"] = float(np.mean(r["fit_time"]))
    out["folds"] = int(len(r["test_macro_f1"]))
    out["_macro_f1_folds"] = [float(v) for v in r["test_macro_f1"]]
    return out


def test_metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=F.CLASSES, zero_division=0, output_dict=True)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])} for c in F.CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=F.CLASSES).tolist(),
    }


def candidates(n_est: int) -> dict[str, tuple]:
    """name -> (estimator or None, family, note, sampler?)"""
    rf = lambda **kw: RandomForestClassifier(n_estimators=n_est, random_state=F.SEED, n_jobs=1, **kw)  # noqa: E731
    return {
        "Dummy (most frequent)": (DummyClassifier(strategy="most_frequent"), "baseline", "Always predicts Hit."),
        "Logistic Regression (balanced)": (LogisticRegression(class_weight="balanced", max_iter=3000, random_state=F.SEED), "linear", "class_weight='balanced'"),
        "Decision Tree (balanced)": (DecisionTreeClassifier(class_weight="balanced", max_depth=6, min_samples_leaf=20, random_state=F.SEED), "tree", "depth 6, leaf>=20"),
        "Random Forest (balanced)": (rf(class_weight="balanced", min_samples_leaf=2), "ensemble", "class_weight='balanced'"),
        "Random Forest (unweighted)": (rf(min_samples_leaf=2), "ensemble", "no class weighting"),
        "Random Forest + oversampling": ("ROS_RF", "ensemble", "RandomOverSampler inside each training fold only"),
        "Extra Trees (balanced)": (ExtraTreesClassifier(n_estimators=n_est, class_weight="balanced", min_samples_leaf=2, random_state=F.SEED, n_jobs=1), "ensemble", "class_weight='balanced'"),
        "HistGradientBoosting (balanced)": (HistGradientBoostingClassifier(class_weight="balanced", max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20, random_state=F.SEED), "boosting", "class_weight='balanced'"),
        "HistGradientBoosting (unweighted)": (HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20, random_state=F.SEED), "boosting", "no class weighting"),
        "LightGBM (balanced)": (LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=20, class_weight="balanced", random_state=F.SEED, verbosity=-1, n_jobs=1), "boosting", "class_weight='balanced'"),
    }


def build(name: str, est, use_people: bool, n_est: int):
    if est == "ROS_RF":
        return ImbPipeline([("prep", F.build_preprocessor(use_people)), ("over", RandomOverSampler(random_state=F.SEED)),
                            ("clf", RandomForestClassifier(n_estimators=n_est, min_samples_leaf=2, random_state=F.SEED, n_jobs=1))])
    return F.make_pipeline(clone(est), use_people)


SPACES = {
    "RandomForestClassifier": {"clf__n_estimators": [300, 500], "clf__max_depth": [None, 8, 12, 20], "clf__min_samples_leaf": [1, 2, 3, 5, 10],
                               "clf__max_features": ["sqrt", 0.3, 0.5], "clf__class_weight": ["balanced", "balanced_subsample"]},
    "ExtraTreesClassifier": {"clf__n_estimators": [300, 500], "clf__max_depth": [None, 8, 12, 20], "clf__min_samples_leaf": [1, 2, 3, 5, 10],
                             "clf__max_features": ["sqrt", 0.3, 0.5], "clf__class_weight": ["balanced", "balanced_subsample"]},
    "HistGradientBoostingClassifier": {"clf__learning_rate": [0.02, 0.04, 0.06, 0.1], "clf__max_leaf_nodes": [6, 10, 15, 31], "clf__min_samples_leaf": [10, 20, 40, 80],
                                       "clf__l2_regularization": [0.0, 1.0, 10.0], "clf__max_iter": [100, 200, 300], "clf__class_weight": ["balanced", None]},
    "LGBMClassifier": {"clf__n_estimators": [150, 250, 400], "clf__learning_rate": [0.02, 0.04, 0.08], "clf__num_leaves": [7, 15, 31], "clf__min_child_samples": [10, 20, 40],
                       "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__reg_lambda": [0.0, 1.0, 10.0]},
    "LogisticRegression": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3]},
    "DecisionTreeClassifier": {"clf__max_depth": [3, 4, 6, 8], "clf__min_samples_leaf": [5, 10, 20, 40]},
}


def main(quick: bool | None = None) -> None:
    if quick is None:                       # called from the command line; a notebook passes quick=False explicitly
        ap = argparse.ArgumentParser()
        ap.add_argument("--quick", action="store_true")
        quick = ap.parse_args().quick
    t0 = time.time()
    n_est, repeats, n_iter = (60, 1, 3) if quick else (300, 3, 16)
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731

    # ---------------------------------------------------------- 1. data + split
    df, dlog = F.load_clean(str(ROOT / "dataset" / "movie_metadata.csv"))
    dist_all = df["Classify"].value_counts().reindex(F.CLASSES).astype(int).to_dict()
    train, test = train_test_split(df, test_size=0.2, stratify=df["Classify"], random_state=F.SEED)
    ytr, yte = train["Classify"], test["Classify"]
    dist_tr = ytr.value_counts().reindex(F.CLASSES).astype(int).to_dict()
    dist_te = yte.value_counts().reindex(F.CLASSES).astype(int).to_dict()
    log("data", dlog, "classes", dist_all, "train", dist_tr, "test", dist_te)
    cv5x = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=F.SEED)

    # ---------------------------------------------------------- 2. ablations (training CV only)
    ref_rf = lambda: RandomForestClassifier(n_estimators=n_est, class_weight="balanced", min_samples_leaf=2, random_state=F.SEED, n_jobs=1)  # noqa: E731
    abl = {}
    for label, up in (("base (no people)", False), ("base + experience counts", True)):
        abl[label] = cv_eval(F.make_pipeline(ref_rf(), up), train[F.input_columns(up)], ytr, cv5x)
        log("ablation", label, {k: round(v[0], 4) for k, v in abl[label].items() if isinstance(v, list)})
    gain = np.array(abl["base + experience counts"]["_macro_f1_folds"]) - np.array(abl["base (no people)"]["_macro_f1_folds"])
    bacc_gain = abl["base + experience counts"]["balanced_accuracy"][0] - abl["base (no people)"]["balanced_accuracy"][0]
    use_people = bool(gain.mean() > 0.01 and bacc_gain >= 0)
    log(f"paired macro-F1 gain from experience features: {gain.mean():+.4f} (balanced-acc change {bacc_gain:+.4f}) -> use_people={use_people}")
    cols = F.input_columns(use_people)
    Xtr, Xte = train[cols], test[cols]

    # reference: what the ORIGINAL post-release columns would score on this same clean split (context only, never a candidate)
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    post_num = ["num_critic_for_reviews", "duration", "director_facebook_likes", "actor_3_facebook_likes", "actor_1_facebook_likes", "gross", "num_voted_users",
                "cast_total_facebook_likes", "facenumber_in_poster", "num_user_for_reviews", "budget", "title_year", "actor_2_facebook_likes", "aspect_ratio", "movie_facebook_likes"]
    post_cat = ["color", "genres", "language", "country", "content_rating"]
    post_prep = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), post_num),
                                   ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), post_cat)])
    old_rf = lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=F.SEED, n_jobs=1)  # noqa: E731  (the original tuned model's settings)
    old_pipe = Pipeline([("prep", post_prep), ("clf", old_rf())])
    reference = cv_eval(old_pipe, train[post_num + post_cat], ytr, cv5x)
    reference["test_reporting_only"] = test_metrics(yte, clone(old_pipe).fit(train[post_num + post_cat], ytr).predict(test[post_num + post_cat]))
    log("REFERENCE original post-release columns + original RF settings, refit on the clean split | CV:", {k: round(v[0], 4) for k, v in reference.items() if isinstance(v, list)},
        "| test:", {k: reference["test_reporting_only"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1")})

    # ---------------------------------------------------------- 3. candidate CV
    cand = candidates(n_est)
    results: dict[str, dict] = {}
    for name, (est, family, note) in cand.items():
        pipe = build(name, est, use_people, n_est)
        results[name] = {"family": family, "note": note, "cv": cv_eval(pipe, Xtr, ytr, cv5x), "tuned": False}
        c = results[name]["cv"]
        log(f"CV {name:36s} macroF1 {c['macro_f1'][0]:.4f}±{c['macro_f1'][1]:.3f}  bal {c['balanced_accuracy'][0]:.4f}  acc {c['accuracy'][0]:.4f}  flopR {c['flop_recall'][0]:.3f}  fit {c['fit_seconds']:.1f}s")

    # ---------------------------------------------------------- 4/5. tune the two best, re-estimate on fresh folds
    ranked = sorted((n for n in results if not n.startswith("Dummy")), key=lambda n: -results[n]["cv"]["macro_f1"][0])
    to_tune, seen = [], set()
    for n in ranked:                      # two best DISTINCT estimator types (an unweighted twin of the same type is not a new candidate)
        est = cand[n][0]
        kind = "RandomForestClassifier" if est == "ROS_RF" else type(est).__name__
        if kind in seen or kind not in SPACES:
            continue
        seen.add(kind); to_tune.append((n, kind))
        if len(to_tune) == 2:
            break
    fresh = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=F.SEED + 1)
    tuned_models: dict[str, object] = {}
    for n, kind in to_tune:
        est = cand[n][0]
        base = build(n, est, use_people, n_est)
        search = RandomizedSearchCV(base, {k: v for k, v in SPACES[kind].items() if not (est == "ROS_RF" and k == "clf__class_weight")}, n_iter=n_iter,
                                    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=F.SEED), n_jobs=-1, random_state=F.SEED, refit=True)
        search.fit(Xtr, ytr)
        tname = f"{n} — tuned"
        results[tname] = {"family": cand[n][1], "note": f"RandomizedSearchCV ({n_iter} draws, 5-fold, macro F1)", "tuned": True, "search_cv_macro_f1": float(search.best_score_),
                          "best_params": {k.replace("clf__", ""): (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in search.best_params_.items()},
                          "space": {k.replace("clf__", ""): [None if x is None else x for x in v] for k, v in SPACES[kind].items()},
                          "cv": cv_eval(search.best_estimator_, Xtr, ytr, fresh)}
        tuned_models[tname] = search.best_estimator_
        c = results[tname]["cv"]
        log(f"TUNED {tname:40s} search macroF1 {search.best_score_:.4f} | fresh-fold macroF1 {c['macro_f1'][0]:.4f}±{c['macro_f1'][1]:.3f} bal {c['balanced_accuracy'][0]:.4f} flopR {c['flop_recall'][0]:.3f}  {results[tname]['best_params']}")
        # untuned twin re-estimated on the SAME fresh folds, so tuned vs untuned is compared like for like
        results[n]["cv_fresh"] = cv_eval(build(n, est, use_people, n_est), Xtr, ytr, fresh)

    # ---------------------------------------------------------- 6. frozen selection rule (no test data involved)
    def key(n):
        c = results[n].get("cv_fresh", results[n]["cv"]) if not results[n]["tuned"] else results[n]["cv"]
        return c["macro_f1"][0], c["balanced_accuracy"][0]
    pool = [n for n in results if not n.startswith("Dummy")]
    best = max(pool, key=key)
    near = [n for n in pool if key(best)[0] - key(n)[0] <= 0.005]
    if len(near) > 1:
        best = max(near, key=lambda n: (key(n)[1], -len(results[n]["note"])))
    selected = best
    log("SELECTED (CV only):", selected, "| within 0.005 of best:", near)

    def folds(n):
        r = results[n]
        return np.array((r["cv"] if r["tuned"] else r.get("cv_fresh", r["cv"]))["_macro_f1_folds"])
    paired = {}
    for n in pool:
        if n == selected or len(folds(n)) != len(folds(selected)):
            continue
        d = folds(selected) - folds(n)
        k = len(d)
        se = float(np.sqrt((1 / k + 0.25) * d.var(ddof=1))) if k > 1 else float("nan")   # Nadeau & Bengio (2003) corrected resampled t
        paired[n] = {"mean_diff_macro_f1": float(d.mean()), "corrected_se": se, "t": float(d.mean() / se) if se and se == se else None, "selected_better_in_folds": float((d > 0).mean())}
    log("paired vs selected (positive = selected better):", {n: (round(v["mean_diff_macro_f1"], 4), round(v["t"], 2) if v["t"] is not None else None) for n, v in paired.items() if n in near or abs(v["mean_diff_macro_f1"]) < 0.03})

    # ---------------------------------------------------------- 7. one test evaluation per candidate (reporting) + save selected
    fitted: dict[str, object] = {}
    for n, r in results.items():
        if r["tuned"]:
            fitted[n] = tuned_models[n]
        else:
            est = cand[n][0]
            fitted[n] = build(n, est, use_people, n_est).fit(Xtr, ytr)
        pred = fitted[n].predict(Xte)
        r["test"] = test_metrics(yte, pred)
        r["selected"] = n == selected
        log(f"TEST {n:44s} acc {r['test']['accuracy']:.3f} bal {r['test']['balanced_accuracy']:.3f} macroF1 {r['test']['macro_f1']:.3f} Flop recall {r['test']['per_class']['Flop']['recall']:.2f}")

    final = fitted[selected]
    # pooled out-of-fold predictions on the training set give the Flop class 5x more evidence than the 8-film test set
    oof = cross_val_predict(clone(final), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=F.SEED + 2), n_jobs=-1)
    pooled = test_metrics(ytr, oof)
    log("pooled out-of-fold (train) per-class recall:", {c: pooled["per_class"][c]["recall"] for c in F.CLASSES})

    # ---------------------------------------------------------- extra experiment (reporting only): does it generalise to LATER films?
    # A random split mixes eras. Here the model is refit on films up to 2009 and judged on films from 2010 on, which is closer to how a
    # pre-release model would really be used. It does not influence selection.
    yr = df["title_year"]
    t_tr, t_te = df[yr <= 2009], df[yr >= 2010]
    temporal_pipe = clone(final).fit(t_tr[cols], t_tr["Classify"])
    tp = temporal_pipe.predict(t_te[cols])
    base_class = t_tr["Classify"].value_counts().idxmax()
    temporal = {"train_through": 2009, "test_from": 2010, "train_size": int(len(t_tr)), "test_size": int(len(t_te)),
                "classes_train": t_tr["Classify"].value_counts().reindex(F.CLASSES).astype(int).to_dict(), "classes_test": t_te["Classify"].value_counts().reindex(F.CLASSES).astype(int).to_dict(),
                "model": test_metrics(t_te["Classify"], tp),
                "majority_baseline": test_metrics(t_te["Classify"], np.repeat(base_class, len(t_te))), "note": "Films with a missing year are excluded from this experiment."}
    log("TEMPORAL (train <=2009, test >=2010):", {k: temporal["model"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}, "baseline acc", temporal["majority_baseline"]["accuracy"], "test classes", temporal["classes_test"])

    out = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "protocol": __doc__, "quick": quick, "seed": F.SEED,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "data": {**dlog, "classes_all": dist_all, "train_size": int(len(train)), "test_size": int(len(test)), "classes_train": dist_tr, "classes_test": dist_te,
                 "split": "train_test_split(test_size=0.2, stratify=Classify, random_state=42) after cleaning"},
        "feature_set": {"use_people": use_people, "inputs": cols, "ablation": {k: {m: v[m] for m in ("macro_f1", "balanced_accuracy", "accuracy", "flop_recall")} for k, v in abl.items()},
                        "paired_macro_f1_gain": float(gain.mean()), "rule": "adopt experience counts only if paired CV macro-F1 gain > 0.01 and balanced accuracy does not fall"},
        "audit": [a.__dict__ for a in F.AUDIT],
        "reference_post_release_rf_cv": {k: v for k, v in reference.items()},
        "cv_protocol": f"RepeatedStratifiedKFold(5 folds x {repeats} repeats, seed {F.SEED}) on the training set; tuned models re-estimated on fresh folds (seed {F.SEED + 1})",
        "selection_rule": "highest fresh-fold CV macro F1; within 0.005 prefer higher balanced accuracy. Test set not used.",
        "selected": selected,
        "paired_vs_selected": paired,
        "models": {n: {k: v for k, v in r.items() if k != "cv" or True} for n, r in results.items()},
        "selected_pooled_oof_train": pooled,
        "temporal_validation": temporal,
    }
    for r in list(out["models"].values()) + [out["reference_post_release_rf_cv"]]:
        for blk in ("cv", "cv_fresh"):
            if blk in r and "_macro_f1_folds" in r[blk]:
                r[blk]["macro_f1_folds"] = r[blk].pop("_macro_f1_folds")
    out["reference_post_release_rf_cv"].pop("_macro_f1_folds", None)
    if quick:
        log("quick run complete; nothing saved")
        return

    # ---------------------------------------------------------- save artifacts
    joblib.dump(final, ROOT / "pre_release_movie_classifier.v1_lightgbm.joblib")
    num = {c: {"min": float(train[c].min()), "max": float(train[c].max()), "p01": float(train[c].quantile(.01)), "p99": float(train[c].quantile(.99)), "median": float(train[c].median()), "missing_pct": round(float(train[c].isna().mean() * 100), 1)} for c in F.NUMERIC}
    cat = {c: sorted(train[c].dropna().unique().tolist()) for c in F.CATEGORICAL}
    genre_vocab = sorted({g for s in train[F.GENRES].dropna() for g in s.split("|") if g})
    schema = {
        "model_file": "pre_release_movie_classifier.joblib", "classes": F.CLASSES, "inputs": cols, "use_people": use_people,
        "numeric": num, "categorical": cat, "genres": genre_vocab, "people_columns": F.PEOPLE if use_people else [],
        "defaults_when_blank": {"numeric_median": {c: num[c]["median"] for c in num}, "categorical": "Unknown (own category)"},
        "excluded": [a.__dict__ for a in F.AUDIT if a.used == "no"],
    }
    (ART / "prerelease_schema_v1.json").write_text(json.dumps(schema, indent=2), encoding="utf8")
    (ART / "prerelease_results_v1.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    log("saved pre_release_movie_classifier.v1_lightgbm.joblib, prerelease_schema_v1.json, prerelease_results_v1.json (the live files are only changed by promote_candidate.py)")


if __name__ == "__main__":
    sys.exit(main())
