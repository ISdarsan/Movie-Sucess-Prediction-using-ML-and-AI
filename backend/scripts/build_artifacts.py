"""LEGACY: builds the artifacts of the ORIGINAL post-release Random Forest (final_movie_classifier.joblib).
The website now serves the pre-release model; its artifacts come from backend/ml/run_prerelease.py + export_site_artifacts.py.
Output goes to legacy_post_release_*.json so it can never overwrite the files the site reads.

Original description: build the static evaluation + explainability artifacts served by the API.

Run from the project root:  .venv/Scripts/python backend/scripts/build_artifacts.py

What it does (and does not do)
------------------------------
* The final Random Forest is LOADED from final_movie_classifier.joblib and evaluated on the
  reproduced held-out split. It is never retrained or modified.
* The other comparison models are re-fitted from the exact definitions in PHASE1.ipynb on the same
  training split, so their held-out metrics can be shown next to the Random Forest. Each result is
  cross-checked against the number printed in the notebook and flagged `matches_notebook`.
* Cross-validation and grid-search figures cannot be cheaply reproduced, so they are copied from the
  notebook's saved cell outputs and labelled as "recorded".
* SHAP values are computed once here (TreeExplainer, 200-row sample, random_state=42) so the API never
  computes them per request.
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "backend" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Flop", "Average", "Hit"]

DROP = ["movie_title", "director_name", "actor_1_name", "actor_2_name", "actor_3_name",
        "movie_imdb_link", "plot_keywords", "imdb_score", "Classify"]


def load_split():
    df = pd.read_csv(ROOT / "dataset" / "movie_metadata.csv")
    df["Classify"] = pd.cut(df["imdb_score"], bins=[0, 3, 6, 10], labels=LABELS, right=False, include_lowest=True)
    df = df.drop_duplicates().copy()
    X, y = df.drop(columns=DROP), df["Classify"]
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {
            c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])}
            for c in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def make_preprocessor(X_train, scale: bool, dense: bool):
    num = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    steps = [("imputer", SimpleImputer(strategy="median"))] + ([("scaler", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("numeric", Pipeline(steps), num),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))]), cat),
    ])


# Values printed in PHASE1.ipynb (cell outputs) used only to cross-check the reproduction.
NOTEBOOK_RECORDED = {
    "Dummy baseline": None,
    "Logistic Regression": (0.733, 0.5722, 0.5047),
    "Random Forest (tuned, final)": (0.804, 0.5194, 0.5119),
    "Extra Trees": None,
    "HistGradientBoosting": (0.836, 0.5207, 0.527),
    "HistGradientBoosting (balanced)": (0.815, 0.5304, 0.5214),
    "XGBoost": (0.825, 0.5083, 0.5155),
    "CatBoost": (0.812, 0.4842, 0.4964),
    "LightGBM (balanced)": (0.806, 0.5312, 0.5173),
}


def clean_feature(name: str) -> str:
    kind, _, rest = name.partition("__")
    if kind == "numeric":
        return rest
    col, _, val = rest.partition("_")
    for c in ("content_rating", "genres", "language", "country", "color"):
        if rest.startswith(c + "_"):
            return f"{c} = {rest[len(c) + 1:]}"
    return rest


def original_column(name: str) -> str:
    kind, _, rest = name.partition("__")
    if kind == "numeric":
        return rest
    for c in ("content_rating", "genres", "language", "country", "color"):
        if rest.startswith(c + "_"):
            return c
    return rest


def main() -> None:
    t0 = time.time()
    X_train, X_test, y_train, y_test = load_split()
    print("train/test", len(X_train), len(X_test))

    final = joblib.load(ROOT / "final_movie_classifier.joblib")
    assert isinstance(final, Pipeline) and isinstance(final.named_steps["classifier"], RandomForestClassifier)

    results: list[dict] = []

    def add(name, y_pred, note, family):
        m = metrics(y_test, y_pred)
        rec = NOTEBOOK_RECORDED.get(name)
        match = None
        if rec:
            match = bool(abs(m["accuracy"] - rec[0]) < 0.0015 and abs(m["balanced_accuracy"] - rec[1]) < 0.0015
                         and abs(m["macro_f1"] - rec[2]) < 0.0015)
        results.append({"name": name, "family": family, "note": note, "selected": name.startswith("Random Forest"),
                        "matches_notebook": match, **m})
        print(f"{name:34s} acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} f1={m['macro_f1']:.4f} match={match}")

    # --- final model: loaded, not retrained ---
    add("Random Forest (tuned, final)", final.predict(X_test),
        "Loaded from final_movie_classifier.joblib; not retrained.", "saved model")

    # --- comparison models: re-fitted from the notebook's definitions ---
    pre_sparse = make_preprocessor(X_train, scale=True, dense=False)
    Xtr_p, Xte_p = pre_sparse.fit_transform(X_train), pre_sparse.transform(X_test)
    add("Dummy baseline", DummyClassifier(strategy="most_frequent").fit(Xtr_p, y_train).predict(Xte_p),
        "Always predicts the most frequent class.", "baseline")
    add("Logistic Regression",
        Pipeline([("p", make_preprocessor(X_train, True, True)),
                  ("c", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))]).fit(X_train, y_train).predict(X_test),
        "class_weight='balanced'; scaled numeric inputs.", "linear")
    add("Extra Trees", ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        .fit(Xtr_p, y_train).predict(Xte_p), "300 trees, balanced class weights.", "ensemble")
    dense = make_preprocessor(X_train, scale=False, dense=True)
    dense.set_params()  # median impute only (as in the notebook's HGB experiments)
    hgb = Pipeline([("p", clone(dense)), ("c", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=42))])
    add("HistGradientBoosting", clone(hgb).fit(X_train, y_train).predict(X_test), "No class weighting.", "boosting")
    hgb_b = clone(hgb).fit(X_train, y_train, c__sample_weight=compute_sample_weight("balanced", y_train))
    add("HistGradientBoosting (balanced)", hgb_b.predict(X_test), "Balanced sample weights.", "boosting")

    le = LabelEncoder().fit(y_train)
    ytr_e = le.transform(y_train)
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    add("XGBoost", le.inverse_transform(xgb.fit(Xtr_p, ytr_e).predict(Xte_p).astype(int)), "300 trees, depth 4.", "boosting")
    from catboost import CatBoostClassifier
    cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass", verbose=0,
                             random_seed=42, thread_count=-1, allow_writing_files=False)
    add("CatBoost", le.inverse_transform(cat.fit(Xtr_p, ytr_e).predict(Xte_p).ravel().astype(int)), "500 iterations, depth 6.", "boosting")
    from lightgbm import LGBMClassifier
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, class_weight="balanced", random_state=42, verbosity=-1, n_jobs=-1)
    add("LightGBM (balanced)", le.inverse_transform(lgb.fit(Xtr_p, ytr_e).predict(Xte_p).astype(int)), "Balanced class weights.", "boosting")

    rf_clf = final.named_steps["classifier"]
    prm = rf_clf.get_params()
    sel_params = {k: prm[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",
                                       "max_features", "class_weight", "criterion", "bootstrap", "random_state")}
    counts = lambda s: {c: int((s == c).sum()) for c in LABELS}
    metrics_json = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "evaluation": {
            "split": "Stratified 80/20 train/test split, random_state=42, after exact-duplicate removal (4,998 rows).",
            "train_size": len(X_train), "test_size": len(X_test),
            "train_class_counts": counts(y_train), "test_class_counts": counts(y_test),
            "labels": LABELS,
            "source": "Random Forest evaluated live from the saved pipeline; other models re-fitted from notebook definitions.",
        },
        "selected_model": {
            "name": "Random Forest (tuned)", "params": sel_params,
            "n_features_in": int(final.named_steps["preprocessor"].n_features_in_),
            "n_transformed_features": int(len(final.named_steps["preprocessor"].get_feature_names_out())),
            "tuning": {
                "method": "GridSearchCV, 5-fold stratified, scoring = macro F1, 8 candidates (40 fits) on the training split",
                "grid": {"n_estimators": [200, 400], "max_depth": [None, 20], "min_samples_leaf": [1, 3], "max_features": ["sqrt"]},
                "best_cv_macro_f1": 0.522, "origin": "recorded from PHASE1.ipynb cell 87",
            },
        },
        "held_out": next(r for r in results if r["selected"]),
        "comparison": results,
        "cross_validation": {
            "origin": "recorded from PHASE1.ipynb (cells 61 and 92); not recomputed here",
            "five_fold_train": {
                "protocol": "5-fold stratified CV on the training split (default 300-tree Random Forest, not the tuned model)",
                "rows": [
                    {"name": "Random Forest (300 trees)", "accuracy": 0.7906, "balanced_accuracy": 0.5206, "macro_f1": 0.5208, "macro_f1_std": 0.0416},
                    {"name": "Logistic Regression", "accuracy": 0.7379, "balanced_accuracy": 0.5852, "macro_f1": 0.5141, "macro_f1_std": 0.0220},
                    {"name": "HistGradientBoosting", "accuracy": 0.8157, "balanced_accuracy": 0.5065, "macro_f1": 0.5110, "macro_f1_std": 0.0033},
                ],
            },
            "repeated": {
                "protocol": "5-fold x 3 repeats stratified CV on the training split",
                "rows": [
                    {"name": "LightGBM (balanced)", "macro_f1": 0.5399, "macro_f1_std": 0.0443, "balanced_accuracy": 0.5403, "balanced_accuracy_std": 0.0289},
                    {"name": "HistGradientBoosting (balanced)", "macro_f1": 0.5301, "macro_f1_std": 0.0458, "balanced_accuracy": 0.5292, "balanced_accuracy_std": 0.0313},
                ],
            },
        },
    }
    (OUT / "legacy_post_release_metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf8")

    # ---------------- explainability ----------------
    pre = final.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    Xte_t = pd.DataFrame(pre.transform(X_test), columns=names, index=X_test.index)
    sample = Xte_t.sample(n=min(200, len(Xte_t)), random_state=42)
    sv = shap.TreeExplainer(rf_clf)(sample)  # (n, features, classes)
    vals = np.asarray(sv.values)
    classes = list(rf_clf.classes_)
    print("shap", vals.shape, classes)
    raw_sample = X_test.loc[sample.index]

    def class_block(cls: str, n_top: int = 15) -> dict:
        ci = classes.index(cls)
        v = vals[:, :, ci]
        imp = np.abs(v).mean(axis=0)
        order = np.argsort(imp)[::-1]
        bar = [{"feature": clean_feature(names[i]), "mean_abs_shap": round(float(imp[i]), 5)} for i in order[:20]]
        swarm = []
        for i in order[:n_top]:
            col = sample.iloc[:, i].to_numpy()
            lo, hi = np.percentile(col, 5), np.percentile(col, 95)
            norm = np.clip((col - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(col)
            swarm.append({
                "feature": clean_feature(names[i]),
                "points": [[round(float(v[r, i]), 5), round(float(norm[r]), 3)] for r in range(len(col))],
            })
        by_col: dict[str, float] = {}
        for i, n in enumerate(names):
            by_col[original_column(n)] = by_col.get(original_column(n), 0.0) + float(imp[i])
        grouped = sorted(({"column": k, "mean_abs_shap": round(x, 5)} for k, x in by_col.items()), key=lambda d: -d["mean_abs_shap"])
        return {"bar": bar, "beeswarm": swarm, "grouped_by_column": grouped}

    rf_imp = rf_clf.feature_importances_
    order = np.argsort(rf_imp)[::-1]
    gi: dict[str, float] = {}
    for i, n in enumerate(names):
        gi[original_column(n)] = gi.get(original_column(n), 0.0) + float(rf_imp[i])
    explain = {
        "generated_at": metrics_json["generated_at"],
        "method": {
            "explainer": "shap.TreeExplainer on the fitted Random Forest classifier",
            "sample_size": int(len(sample)), "sample_from": "held-out test split, random_state=42",
            "n_transformed_features": len(names), "classes": classes, "shap_version": shap.__version__,
            "values_shape": list(vals.shape),
        },
        "feature_importance": {
            "kind": "Random Forest impurity-based importance (mean decrease in impurity), fitted on the training split",
            "top": [{"feature": clean_feature(names[i]), "importance": round(float(rf_imp[i]), 5)} for i in order[:20]],
            "grouped_by_column": sorted(({"column": k, "importance": round(v, 5)} for k, v in gi.items()), key=lambda d: -d["importance"]),
        },
        "shap": {c: class_block(c) for c in LABELS},
    }
    (OUT / "legacy_post_release_explain.json").write_text(json.dumps(explain), encoding="utf8")
    print("done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    sys.exit(main())
