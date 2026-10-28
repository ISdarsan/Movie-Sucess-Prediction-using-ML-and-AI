"""Turn comprehensive_results.json + the saved (promoted) pipeline into the two files the website reads:
   backend/artifacts/model_metrics.json   (Performance + Methodology pages)
   backend/artifacts/explain.json         (Explainability page: global importance + SHAP, computed once here)

Run from the project root AFTER the candidate has passed its integration tests and been promoted:
    .venv/Scripts/python -m backend.ml.export_site_artifacts
Works for whichever algorithm was selected (tree models use shap.TreeExplainer; others a model-agnostic permutation explainer).
Previous generations are kept as *_v1.json / legacy_post_release_*.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from . import features as F
from .run_comprehensive import corrected_t

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"


def pretty(name: str) -> str:
    kind, _, rest = name.partition("__")
    if rest.startswith("missingindicator_"):
        return f"{rest.removeprefix('missingindicator_')} missing"
    if kind == "budget":
        return "log budget"
    if kind == "numeric":
        return rest
    if kind == "genres":
        return f"genre = {rest}"
    if kind == "categorical":
        for c in F.CATEGORICAL:
            if rest.startswith(c + "_"):
                return f"{c} = {rest[len(c) + 1:]}"
    return rest


def source_column(name: str) -> str:
    kind, _, rest = name.partition("__")
    if rest.startswith("missingindicator_"):
        return rest.removeprefix("missingindicator_")
    if kind in ("budget", "genres"):
        return kind
    if kind == "categorical":
        for c in F.CATEGORICAL:
            if rest.startswith(c + "_"):
                return c
    return rest


def shap_values(clf, Xt_sample: pd.DataFrame, Xt_background: pd.DataFrame) -> tuple[np.ndarray, str]:
    """(rows, features, classes) SHAP array and a description of the explainer actually used."""
    core = getattr(clf, "base_", clf)
    try:
        sv = np.asarray(shap.TreeExplainer(core)(Xt_sample).values)
        if sv.ndim == 3:
            return sv, f"shap.TreeExplainer on the fitted {type(core).__name__}"
    except Exception:  # noqa: BLE001 - not a tree model (or unsupported): use the model-agnostic explainer
        pass
    bg = shap.sample(Xt_background, 80, random_state=F.SEED)
    ex = shap.Explainer(lambda X: clf.predict_proba(pd.DataFrame(X, columns=Xt_sample.columns)), bg, algorithm="permutation", max_evals=2 * Xt_sample.shape[1] + 1)
    return np.asarray(ex(Xt_sample).values), f"shap permutation explainer on the fitted {type(core).__name__} (model-agnostic)"


def global_importance(clf, Xt_train: pd.DataFrame, y_train) -> tuple[np.ndarray, str]:
    core = getattr(clf, "base_", clf)
    if hasattr(core, "booster_") and hasattr(core.booster_, "feature_importance"):
        g = np.asarray(core.booster_.feature_importance(importance_type="gain"), dtype=float)
        return g / g.sum(), "LightGBM total gain, normalised to sum to 1 (fitted on the training split)"
    if hasattr(core, "feature_importances_"):
        g = np.asarray(core.feature_importances_, dtype=float)
        return g / g.sum(), f"{type(core).__name__} feature_importances_, normalised to sum to 1 (fitted on the training split)"
    if hasattr(core, "coef_"):
        g = np.abs(np.asarray(core.coef_)).mean(axis=0)
        return g / g.sum(), "mean absolute coefficient across classes (inputs are scaled), normalised to sum to 1"
    from sklearn.inspection import permutation_importance
    idx = np.random.default_rng(F.SEED).choice(len(Xt_train), size=min(800, len(Xt_train)), replace=False)
    r = permutation_importance(clf, Xt_train.iloc[idx], np.asarray(y_train)[idx], scoring="f1_macro", n_repeats=5, random_state=F.SEED, n_jobs=-1)
    g = np.clip(r.importances_mean, 0, None)
    return g / g.sum(), "permutation importance: macro-F1 drop when a feature is shuffled (800 training films), normalised"


def main() -> None:
    res = json.loads((ART / "comprehensive_results.json").read_text(encoding="utf8"))
    v1 = json.loads((ART / "prerelease_results_v1.json").read_text(encoding="utf8"))
    schema = json.loads((ART / "prerelease_schema.json").read_text(encoding="utf8"))
    pipe = joblib.load(ROOT / "pre_release_movie_classifier.joblib")
    sel = res["selection"]["chosen"]
    fin = {f["name"]: f for f in res["finalists"]}
    clf = pipe.named_steps["clf"]
    core = getattr(clf, "base_", clf)

    # ------------------------------------------------------------------ metrics for the Performance / Methodology pages
    def row(f: dict, cv: dict | None = None) -> dict:
        cv = cv or f["cv_final"]
        return {"name": f["name"], "family": f["algorithm_label"], "algorithm": f["algorithm_label"], "strategy": f["strategy_label"], "complexity": f["complexity"],
                "note": f["strategy_label"] + (" · tuned" if f.get("tuned") else ""), "selected": f["name"] == sel, "tuned": bool(f.get("tuned")), "deployed_before": bool(f.get("deployed_before")),
                "matches_notebook": None, "cv_macro_f1": cv["macro_f1"][0], "cv_macro_f1_std": cv["macro_f1"][1], "cv_balanced_accuracy": cv["balanced_accuracy"][0], "cv_accuracy": cv["accuracy"][0],
                "cv_flop_recall": cv["flop_recall"][0], "fit_seconds": (f.get("timing") or {}).get("fit_seconds"), "predict_ms_per_1000": (f.get("timing") or {}).get("predict_ms_per_1000_films"), **f["test"]}
    finalists_rows = sorted((row(f) for f in res["finalists"]), key=lambda r: -r["cv_macro_f1"])
    baseline_rows = [row({**b, "tuned": False, "name": b["name"], "algorithm_label": b["algorithm_label"], "strategy_label": b["strategy_label"]}, b["cv"]) for b in res["baselines"]]
    comparison = finalists_rows + baseline_rows
    params = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in core.get_params().items()
              if k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "class_weight", "max_depth", "max_features", "min_samples_leaf", "C", "gamma", "kernel", "n_neighbors", "weights", "p",
                       "solver", "shrinkage", "max_iter", "subsample", "colsample_bytree", "reg_lambda", "l2_regularization", "depth", "iterations", "random_state", "auto_class_weights", "max_leaf_nodes")}
    twin = next((t for t in res["tuning"] if t["name"] == sel), None)
    base_search = next((t for t in res["tuning"] if t["base_config"] == sel), None)
    sel_f = fin[sel]
    folds = {n: np.array(f["cv_final"]["macro_f1_folds"]) for n, f in fin.items()}
    paired = {}
    for n in fin:
        if n == sel:
            continue
        d, se, t = corrected_t(folds[sel], folds[n])
        paired[n] = {"mean_diff_macro_f1": round(d, 5), "corrected_se": round(se, 5), "t": round(t, 3) if np.isfinite(t) else None, "selected_better_in_folds": round(float((folds[sel] > folds[n]).mean()), 3)}
    tmain = res["temporal"]["main"]
    tv = {"train_through": 2009, "test_from": 2010, "train_size": tmain["train_size"], "test_size": tmain["test_size"], "classes_train": tmain["classes_train"], "classes_test": tmain["classes_test"],
          "model": {**tmain["models"][sel], "name": sel}, "majority_baseline": tmain["majority_baseline"], "note": res["temporal"]["note"]}
    best_per_algo: dict[str, dict] = {}
    for s in res["screening"]:
        if s["ok"] and s["algorithm"] != "dummy" and (s["algorithm"] not in best_per_algo or s["cv"]["macro_f1"][0] > best_per_algo[s["algorithm"]]["cv"]["macro_f1"][0]):
            best_per_algo[s["algorithm"]] = s
    metrics = {
        "generated_at": res["generated_at"], "stage": "pre-release",
        "evaluation": {"split": res["data"]["split"], "train_size": res["data"]["train_size"], "test_size": res["data"]["test_size"], "train_class_counts": res["data"]["classes_train"],
                       "test_class_counts": res["data"]["classes_test"], "labels": F.CLASSES, "source": "The saved pre-release pipeline, evaluated once on the untouched 20% test split after model selection was frozen.",
                       "data": res["data"], "cv_protocol": "; ".join(f"{k}: {v}" for k, v in res["cv_protocols"].items()), "selection_rule": res["selection"]["rule"],
                       "n_configurations": len(res["screening"]), "n_algorithms": sum(1 for a in res["algorithms"] if a["evaluated"])},
        "selected_model": {"name": sel, "algorithm": sel_f["algorithm_label"], "strategy": sel_f["strategy_label"], "params": params, "n_features_in": len(schema["inputs"]),
                           "n_transformed_features": int(len(pipe.named_steps["prep"].get_feature_names_out())),
                           "tuning": {"method": (f"Tuned: {twin['n_iter']}-draw randomised search over {sel_f['algorithm_label']}; the tuned settings were re-scored on fresh folds." if twin else
                                                 (f"Default settings. A {base_search['n_iter']}-draw randomised search was also run and re-scored on fresh folds; it was not selected." if base_search else "Default settings; not among the five algorithms tuned.")),
                                      "grid": (twin or base_search or {"search_space": {}})["search_space"], "best_cv_macro_f1": sel_f["cv_final"]["macro_f1"][0], "origin": "computed in backend/ml/run_comprehensive.py"}},
        "held_out": next(c for c in comparison if c["selected"]), "comparison": comparison,
        "screening": [{"name": s["name"], "algorithm": s["algorithm_label"], "strategy": s["strategy_label"], "complexity": s["complexity"], "ok": s["ok"], "macro_f1": s["cv"]["macro_f1"], "balanced_accuracy": s["cv"]["balanced_accuracy"],
                       "accuracy": s["cv"]["accuracy"], "flop_recall": s["cv"]["flop_recall"], "flop_f1": s["cv"]["flop_f1"], "fit_seconds": s["cv"]["fit_seconds"]} for s in res["screening"]],
        "cross_validation": {"origin": "computed in backend/ml/run_comprehensive.py (training set only)",
                             "five_fold_train": {"protocol": f"Best configuration of each algorithm in the screening stage ({res['cv_protocols']['screening']})",
                                                 "rows": [{"name": s["name"], "accuracy": s["cv"]["accuracy"][0], "balanced_accuracy": s["cv"]["balanced_accuracy"][0], "macro_f1": s["cv"]["macro_f1"][0], "macro_f1_std": s["cv"]["macro_f1"][1],
                                                           "flop_recall": s["cv"]["flop_recall"][0]} for s in sorted(best_per_algo.values(), key=lambda s: -s["cv"]["macro_f1"][0])]},
                             "repeated": {"protocol": f"Finalists on identical fresh folds ({res['cv_protocols']['finalists']})",
                                          "rows": [{"name": r["name"], "macro_f1": r["cv_macro_f1"], "macro_f1_std": r["cv_macro_f1_std"], "balanced_accuracy": r["cv_balanced_accuracy"], "balanced_accuracy_std": 0.0} for r in finalists_rows]}},
        "selection": res["selection"], "fold_audit": res["fold_audit"], "algorithms": res["algorithms"], "skipped": res["skipped"], "tuning": res["tuning"], "regression_experiment": res["regression_experiment"],
        "temporal": res["temporal"], "temporal_validation": tv, "feature_set": {**v1["feature_set"], "inputs": schema["inputs"]}, "feature_audit": res["audit"], "paired_vs_selected": paired,
        "pooled_oof_train": res["selected_pooled_oof_train"], "deployed_before": res["deployed_before"],
        "reference_post_release": {"description": "The ORIGINAL feature set (votes, reviews, gross, social-media likes) with the original tuned Random Forest settings, refit on the same clean split. Not a candidate: shown only to quantify what the excluded post-release columns were contributing.",
                                   "cv_accuracy": round(v1["reference_post_release_rf_cv"]["accuracy"][0], 4), "cv_balanced_accuracy": round(v1["reference_post_release_rf_cv"]["balanced_accuracy"][0], 4),
                                   "cv_macro_f1": round(v1["reference_post_release_rf_cv"]["macro_f1"][0], 4), "cv_flop_recall": round(v1["reference_post_release_rf_cv"]["flop_recall"][0], 4), "test": v1["reference_post_release_rf_cv"]["test_reporting_only"]},
    }
    (ART / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")

    # ------------------------------------------------------------------ SHAP + importance for the explainability page
    df, _ = F.load_clean(str(ROOT / "dataset" / "movie_metadata.csv"))
    train, test = train_test_split(df, test_size=0.2, stratify=df["Classify"], random_state=F.SEED)
    prep = pipe.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    Xt_test = pd.DataFrame(np.asarray(prep.transform(test[schema["inputs"]])), columns=names, index=test.index)
    Xt_train = pd.DataFrame(np.asarray(prep.transform(train[schema["inputs"]])), columns=names, index=train.index)
    sample = Xt_test.sample(n=min(200, len(Xt_test)), random_state=F.SEED)
    sv, how = shap_values(clf, sample, Xt_train)
    classes = list(map(str, clf.classes_))
    assert sv.shape == (len(sample), len(names), len(classes)), sv.shape

    def block(cls: str, n_top: int = 15) -> dict:
        v = sv[:, :, classes.index(cls)]
        imp = np.abs(v).mean(0)
        order = np.argsort(imp)[::-1]
        swarm = []
        for i in order[:n_top]:
            col = sample.iloc[:, i].to_numpy()
            lo, hi = np.percentile(col, 5), np.percentile(col, 95)
            norm = np.clip((col - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(col)
            swarm.append({"feature": pretty(names[i]), "points": [[round(float(v[r, i]), 5), round(float(norm[r]), 3)] for r in range(len(col))]})
        grp: dict[str, float] = {}
        for i, n in enumerate(names):
            grp[source_column(n)] = grp.get(source_column(n), 0.0) + float(imp[i])
        return {"bar": [{"feature": pretty(names[i]), "mean_abs_shap": round(float(imp[i]), 5)} for i in order[:20]], "beeswarm": swarm,
                "grouped_by_column": sorted(({"column": k, "mean_abs_shap": round(x, 5)} for k, x in grp.items()), key=lambda d: -d["mean_abs_shap"])}

    gain, kind = global_importance(clf, Xt_train, train["Classify"])
    order = np.argsort(gain)[::-1]
    gi: dict[str, float] = {}
    for i, n in enumerate(names):
        gi[source_column(n)] = gi.get(source_column(n), 0.0) + float(gain[i])
    explain = {"generated_at": res["generated_at"],
               "method": {"explainer": how, "sample_size": int(len(sample)), "sample_from": "held-out test split, random_state=42", "n_transformed_features": len(names), "classes": classes, "shap_version": shap.__version__, "values_shape": list(sv.shape)},
               "feature_importance": {"kind": kind, "top": [{"feature": pretty(names[i]), "importance": round(float(gain[i]), 5)} for i in order[:20]],
                                      "grouped_by_column": sorted(({"column": k, "importance": round(v, 5)} for k, v in gi.items()), key=lambda d: -d["importance"])},
               "shap": {c: block(c) for c in F.CLASSES}}
    (ART / "explain.json").write_text(json.dumps(explain), encoding="utf8")
    print("wrote model_metrics.json and explain.json for", sel, "|", how)


if __name__ == "__main__":
    sys.exit(main())
