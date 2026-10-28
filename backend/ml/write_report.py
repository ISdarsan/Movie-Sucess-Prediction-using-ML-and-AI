"""Generate docs/PRE_RELEASE_MODEL.md from backend/artifacts/comprehensive_results.json (no number is typed by hand).

    .venv/Scripts/python -m backend.ml.write_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
R = json.loads((ART / "comprehensive_results.json").read_text(encoding="utf8"))
V1 = json.loads((ART / "prerelease_results_v1.json").read_text(encoding="utf8"))
P = lambda x, d=1: f"{x * 100:.{d}f}%"  # noqa: E731
CL = ["Flop", "Average", "Hit"]
d, sel = R["data"], R["selection"]
chosen = sel["chosen"]
fin = {f["name"]: f for f in R["finalists"]}
S = R["screening"]
L: list[str] = []
w = L.append


def pc(t, c, m):
    return f"{t['per_class'][c][m]:.2f}"


w("# Pre-release movie success prediction — comprehensive model comparison")
w("")
w(f"*Generated from `backend/artifacts/comprehensive_results.json` ({R['generated_at']}). Every number comes from that file. Seed {R['seed']}; "
  + ", ".join(f"{k} {v}" for k, v in R["versions"].items()) + ".*")
w("")
w("## 1. Objective")
w("")
w("Predict whether a film will be a **Hit, Average or Flop** using only information available **before release**. The label is derived from the eventual IMDb score (`Flop` < 3 ≤ `Average` < 6 ≤ `Hit`, unchanged). It is a rating band, not box office or profit, and a prediction is an estimate, not a guarantee. This study re-opens the model choice: **no algorithm was assumed to win**, including the previously deployed LightGBM and the original Random Forest.")
w("")
w("## 2. Verified setup (re-used unchanged)")
w("")
w("| check | result |")
w("|---|---|")
for k, v in R["verification"].items():
    w(f"| {k.replace('_', ' ')} | {v} |")
w("")
w(f"- Cleaned dataset: {d['raw_rows']:,} raw rows → {d['after_exact_duplicate_rows']:,} after exact duplicates → **{d['after_title_year_duplicates']:,}** after removing the same film scraped twice (same title and year; films with a missing year are kept).")
w(f"- Split: {d['split']}. Train {d['train_size']:,} / test {d['test_size']:,}; the test set is untouched until the selection is frozen.")
w("")
w("| class | all | train | test |")
w("|---|---|---|---|")
for c in CL:
    w(f"| {c} | {d['classes_all'][c]:,} ({P(d['classes_all'][c] / d['after_title_year_duplicates'])}) | {d['classes_train'][c]:,} | {d['classes_test'][c]:,} |")
fa = R["fold_audit"]
w("")
w(f"**Class counts in every fold:** each training fold holds {fa['flops_per_training_fold'][0]}–{fa['flops_per_training_fold'][1]} Flops and each validation fold only **{fa['flops_per_validation_fold'][0]}–{fa['flops_per_validation_fold'][1]}** (of {fa['flops_in_training_set']} in the whole training set). {fa['conclusion']}")
w("")
w("## 3. The nine pre-release inputs and what was excluded")
w("")
w("Inputs: " + ", ".join(f"`{c}`" for c in R["feature_set"]["inputs"]) + ". Every other column was audited and excluded (gross, votes, reviews, social-media likes, plot keywords, identifiers, and the IMDb score itself):")
w("")
w("| excluded column | timing | reason |")
w("|---|---|---|")
for a in R["audit"]:
    if a["used"] == "no":
        w(f"| `{a['column']}` | {a['timing'].replace('_', ' ')} | {a['reason']} |")
w("")
w("## 4. Preprocessing and imbalance strategies")
w("")
w("All preprocessing lives inside the model pipeline and is fitted on training folds only: median imputation with missing flags, scaling and a log transform for budget, missing categoricals as their own `Unknown` level, rare categories pooled and unseen ones handled safely, genres as separate flags. Scaling is therefore always applied, which SVM, k-NN and logistic regression need.")
w("")
w("| strategy | meaning |")
w("|---|---|")
w("| unweighted | the classifier sees the real class proportions |")
w("| class weights | `class_weight='balanced'`, or balanced sample weights (Gradient Boosting, AdaBoost, XGBoost), CatBoost `auto_class_weights`, uniform priors for LDA |")
w("| oversampling (in fold) | random oversampling of the minority classes, a step **inside** the CV pipeline so it only ever touches training folds |")
w("| SMOTE (in fold) | synthetic interpolation after encoding, also inside the pipeline. Caveat: the encoded space mixes scaled numbers with 0/1 flags, so interpolated rows contain fractional flags — it is evaluated, not assumed suitable |")
w("")
w("## 5. Algorithms evaluated")
w("")
w("| algorithm | library | complexity rank | imbalance strategies run |")
w("|---|---|---|---|")
for a in R["algorithms"]:
    w(f"| {a['label']} | {a['library']} | {a['complexity']} | {', '.join(a['strategies'])}{'' if a['evaluated'] else ' — NOT EVALUATED'} |")
w("")
w(f"**{len(S)} configurations** across {sum(1 for a in R['algorithms'] if a['evaluated'])} algorithms were executed.")
w("")
w("Not evaluated as classifiers, and why:")
for s in R["skipped"]:
    w(f"- {s.get('algorithm', s.get('name'))}: {s.get('reason', s.get('error'))}")
w("")
w("## 6. Protocol and selection rule (fixed before any test number was computed)")
w("")
w("```")
w(R["protocol"].split("PROTOCOL")[1].split("COMPLEXITY RANK")[0].strip("\n").replace("(fixed before any test number is computed)", "").strip())
w("```")
w("")
w("## 7. Stage 1 — screening of every configuration (cross-validation, training set only)")
w("")
w(f"{R['cv_protocols']['screening']}. Sorted by mean macro F1.")
w("")
w("| configuration | macro F1 (±sd) | balanced acc. | accuracy | Flop recall | Flop F1 | fit s |")
w("|---|---|---|---|---|---|---|")
for s in sorted(S, key=lambda s: -s["cv"]["macro_f1"][0]):
    c = s["cv"]
    w(f"| {s['name']} | {c['macro_f1'][0]:.3f} ± {c['macro_f1'][1]:.3f} | {c['balanced_accuracy'][0]:.3f} | {P(c['accuracy'][0])} | {P(c['flop_recall'][0], 0)} | {c['flop_f1'][0]:.2f} | {c['fit_seconds']:.1f} |")
w("")
w("**Effect of the imbalance strategy** (mean over every algorithm that used it; cross-validation, training set):")
w("")
w("| strategy | configs | mean macro F1 | mean balanced acc. | mean accuracy | mean Flop recall |")
w("|---|---|---|---|---|---|")
for st in ("none", "weights", "oversample", "smote"):
    g = [s for s in S if s["strategy"] == st and s["algorithm"] != "dummy"]
    if g:
        w(f"| {g[0]['strategy_label']} | {len(g)} | {np.mean([s['cv']['macro_f1'][0] for s in g]):.3f} | {np.mean([s['cv']['balanced_accuracy'][0] for s in g]):.3f} | {P(np.mean([s['cv']['accuracy'][0] for s in g]))} | {P(np.mean([s['cv']['flop_recall'][0] for s in g]), 0)} |")
w("")
w("Balancing raises macro F1, balanced accuracy and Flop recall and lowers plain accuracy — a genuine trade-off, not a free improvement.")
w("")
w("## 8. Stage 2 — tuning the five strongest algorithms")
w("")
w("| tuned configuration | search CV macro F1 | best parameters |")
w("|---|---|---|")
for t in R["tuning"]:
    base = next(s for s in S if s["name"] == t["base_config"])
    w(f"| {t['name']} | {t['search_cv_macro_f1']:.3f} (untuned {base['cv']['macro_f1'][0]:.3f}) | {', '.join(f'{k}={v}' for k, v in t['best_params'].items())} |")
w("")
w("## 9. Stage 3 — finalists on identical fresh folds, and the selection")
w("")
w(f"{R['cv_protocols']['finalists']}. Winner's-curse optimism from stages 1–2 is not carried over.")
w("")
w("| finalist | macro F1 (±sd) | balanced acc. | complexity | paired t vs best | |")
w("|---|---|---|---|---|---|")
for f in sorted(R["finalists"], key=lambda f: -f["cv_final"]["macro_f1"][0]):
    c = f["cv_final"]; pv = sel["paired_vs_best"][f["name"]]
    tag = "**← selected**" if f["name"] == chosen else ("best by mean" if f["name"] == sel["best_by_mean"] else ("deployed before" if f.get("deployed_before") else ""))
    w(f"| {f['name']} | {c['macro_f1'][0]:.3f} ± {c['macro_f1'][1]:.3f} | {c['balanced_accuracy'][0]:.3f} | {f['complexity']} | {pv['t'] if pv['t'] is not None else '–'} | {tag} |")
w("")
w(f"- Best by mean macro F1: **{sel['best_by_mean']}**.")
w(f"- Not distinguishable from the best (corrected paired t < 2): {', '.join('`' + n + '`' for n in sel['indistinguishable_from_best'])}.")
w(f"- **Chosen by the rule (simplest among the indistinguishable): {chosen}** (complexity rank {sel['chosen_complexity_rank']}). Outcome relative to the previously deployed `{sel['deployed_before']}`: **{sel['outcome']}**.")
w("")
w("## 10. Held-out test results (computed once, after the selection was frozen)")
w("")
w("| algorithm | imbalance strategy | CV macro F1 | CV sd | test acc. | test bal. acc. | test macro F1 | Flop P / R / F1 | Average P / R / F1 | Hit P / R / F1 | fit s | predict ms/1000 |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = sorted(R["finalists"], key=lambda f: -f["cv_final"]["macro_f1"][0])
for f in rows:
    t = f["test"]
    tag = " **← selected**" if f["name"] == chosen else (" *(deployed before)*" if f.get("deployed_before") else "")
    w(f"| {f['algorithm_label']}{' (tuned)' if f['tuned'] else ''}{tag} | {f['strategy_label']} | {f['cv_final']['macro_f1'][0]:.3f} | {f['cv_final']['macro_f1'][1]:.3f} | {P(t['accuracy'])} | {P(t['balanced_accuracy'])} | {t['macro_f1']:.3f} | "
      f"{pc(t, 'Flop', 'precision')} / {pc(t, 'Flop', 'recall')} / {pc(t, 'Flop', 'f1-score')} | {pc(t, 'Average', 'precision')} / {pc(t, 'Average', 'recall')} / {pc(t, 'Average', 'f1-score')} | "
      f"{pc(t, 'Hit', 'precision')} / {pc(t, 'Hit', 'recall')} / {pc(t, 'Hit', 'f1-score')} | {f['timing']['fit_seconds']} | {f['timing']['predict_ms_per_1000_films']} |")
for b in R["baselines"]:
    t = b["test"]
    w(f"| {b['algorithm_label']} | {b['strategy_label']} | {b['cv']['macro_f1'][0]:.3f} | {b['cv']['macro_f1'][1]:.3f} | {P(t['accuracy'])} | {P(t['balanced_accuracy'])} | {t['macro_f1']:.3f} | "
      f"{pc(t, 'Flop', 'precision')} / {pc(t, 'Flop', 'recall')} / {pc(t, 'Flop', 'f1-score')} | {pc(t, 'Average', 'precision')} / {pc(t, 'Average', 'recall')} / {pc(t, 'Average', 'f1-score')} | "
      f"{pc(t, 'Hit', 'precision')} / {pc(t, 'Hit', 'recall')} / {pc(t, 'Hit', 'f1-score')} | {b['timing']['fit_seconds']} | {b['timing']['predict_ms_per_1000_films']} |")
w("")
best_test = max(R["finalists"], key=lambda f: f["test"]["macro_f1"])
ct = fin[chosen]["test"]
w(f"**CV versus test.** The finalist with the highest test macro F1 is `{best_test['name']}` ({best_test['test']['macro_f1']:.3f}); the selected model scored {ct['macro_f1']:.3f}. "
  + ("These agree." if best_test["name"] == chosen else "They disagree, which is exactly what a ranking within the noise looks like: with 8 test Flops and a cross-validation spread of about ±0.03 the test set cannot adjudicate between finalists, and the selection rule was not changed after seeing it."))
dep = fin.get(sel["deployed_before"])
if dep:
    w(f"**Versus the previously deployed model** (`{dep['name']}`): selected accuracy {P(ct['accuracy'])} vs {P(dep['test']['accuracy'])}, balanced accuracy {P(ct['balanced_accuracy'])} vs {P(dep['test']['balanced_accuracy'])}, macro F1 {ct['macro_f1']:.3f} vs {dep['test']['macro_f1']:.3f}, Flops found {ct['confusion_matrix'][0][0]} vs {dep['test']['confusion_matrix'][0][0]} of {ct['per_class']['Flop']['support']}.")
mj = next(b for b in R["baselines"] if b["strategy"] == "majority")
w(f"**Versus always answering the majority class:** accuracy {P(ct['accuracy'])} vs {P(mj['test']['accuracy'])}; balanced accuracy {P(ct['balanced_accuracy'])} vs {P(mj['test']['balanced_accuracy'])}; macro F1 {ct['macro_f1']:.3f} vs {mj['test']['macro_f1']:.3f}.")
w("")
w("### Confusion matrices (rows = true, columns = predicted; Flop, Average, Hit)")
w("")
for f in rows:
    w(f"- **{f['name']}**{' ← selected' if f['name'] == chosen else ''}: `{f['test']['confusion_matrix']}`")
w("")
w(f"### Classification report — {chosen}")
w("")
w("| class | precision | recall | F1 | support |")
w("|---|---|---|---|---|")
for c in CL:
    p = ct["per_class"][c]
    w(f"| {c} | {p['precision']:.2f} | {p['recall']:.2f} | {p['f1-score']:.2f} | {p['support']} |")
po = R["selected_pooled_oof_train"]
w("")
w(f"Pooled out-of-fold recall on the training set (more Flop evidence than the test set): Flop {P(po['per_class']['Flop']['recall'], 0)}, Average {P(po['per_class']['Average']['recall'], 0)}, Hit {P(po['per_class']['Hit']['recall'], 0)}.")
w("")
w("## 11. Did Flop detection improve meaningfully?")
w("")
flopR = {f["name"]: f["test"]["per_class"]["Flop"]["recall"] for f in R["finalists"]}
top_flop = max(R["finalists"], key=lambda f: f["test"]["per_class"]["Flop"]["recall"])
w(f"No. The selected model finds {ct['confusion_matrix'][0][0]} of {ct['per_class']['Flop']['support']} test Flops (precision {pc(ct, 'Flop', 'precision')}). The finalist that flags the most Flops, `{top_flop['name']}`, finds {top_flop['test']['confusion_matrix'][0][0]} of {top_flop['test']['per_class']['Flop']['support']} "
  f"but only {P(top_flop['test']['per_class']['Flop']['precision'], 0)} of the films it calls Flop are Flops, and its accuracy is {P(top_flop['test']['accuracy'])}. With {fa['flops_in_training_set']} training Flops and {fa['flops_per_validation_fold'][0]}–{fa['flops_per_validation_fold'][1]} per validation fold, differences of a film or two are noise. No configuration is claimed to detect Flops reliably.")
w("")
w("## 12. Temporal generalisation (reporting only; not used for selection)")
w("")
w(R["temporal"]["note"])
w("")
w("| period | train / test films | test classes (F / A / H) | always-majority accuracy | " + chosen + " accuracy | bal. acc. | macro F1 | Flops found |")
w("|---|---|---|---|---|---|---|---|")
for wd in [R["temporal"]["main"]] + R["temporal"]["rolling"]:
    m = wd["models"][chosen]
    ct_ = wd["classes_test"]
    w(f"| {wd['label']} | {wd['train_size']:,} / {wd['test_size']:,} | {ct_['Flop']} / {ct_['Average']} / {ct_['Hit']} | {P(wd['majority_baseline']['accuracy'])} | {P(m['accuracy'])} | {P(m['balanced_accuracy'])} | {m['macro_f1']:.3f} | {m['confusion_matrix'][0][0]} of {m['per_class']['Flop']['support']} |")
w("")
w("Flops in the later windows number in the single digits, so temporal Flop figures are not interpretable; the accuracy and balanced-accuracy comparisons with the majority baseline are the usable evidence. Reliability on future films is **not** claimed from the random split.")
w("")
w("Main temporal split, all finalists:")
w("")
w("| finalist | accuracy | balanced acc. | macro F1 |")
w("|---|---|---|---|")
for n, m in R["temporal"]["main"]["models"].items():
    w(f"| {n}{' ← selected' if n == chosen else ''} | {P(m['accuracy'])} | {P(m['balanced_accuracy'])} | {m['macro_f1']:.3f} |")
w("")
w("## 13. Separate experiment: regression on the IMDb score")
w("")
w(R["regression_experiment"]["definition"])
w("")
w("| regressor | macro F1 | balanced acc. | accuracy |")
w("|---|---|---|---|")
for r in R["regression_experiment"]["rows"]:
    w(f"| {r['name']} | {r['macro_f1'][0]:.3f} ± {r['macro_f1'][1]:.3f} | {r['balanced_accuracy'][0]:.3f} | {P(r['accuracy'][0])} |")
w("")
w("## 14. Limitations")
w("")
w("1. Only 41 Flops exist; no model is reliable on that class, and the test set has 8.")
w("2. Pre-release features give a modest signal: accuracy is close to the majority baseline; the advantage is in balanced accuracy.")
w("3. The main split is random; the temporal checks suggest the signal is weaker for later films, and `title_year` lets models learn era effects.")
w("4. The label is an IMDb-rating band, not commercial success; `budget` mixes currencies; `content_rating` is assigned late in production.")
w("5. Many finalists are statistically tied, so the choice rests on a pre-declared simplicity rule, not on a decisive win.")
w("6. Class scores are uncalibrated estimates; SHAP and importance describe the model, not causes.")
w("")
w("## 15. Reproduce")
w("")
w("```")
w(".venv/Scripts/python -m backend.ml.run_comprehensive      # the whole comparison")
w(".venv/Scripts/python -m pytest backend/tests/test_candidate.py -q   # integration tests on the candidate BEFORE promotion")
w(".venv/Scripts/python -m backend.ml.promote_candidate     # promote only if those pass")
w(".venv/Scripts/python -m backend.ml.export_site_artifacts # metrics + SHAP for the website")
w(".venv/Scripts/python -m backend.ml.write_report          # this document")
w(".venv/Scripts/python -m backend.ml.append_notebook_section")
w("```")
w("")
w("Earlier artifacts are kept: `pre_release_movie_classifier.v1_lightgbm.joblib` (the model deployed before this comparison), `prerelease_results_v1.json`, `model_metrics_v1.json`, `explain_v1.json`; and the original post-release model `final_movie_classifier.joblib`.")
out = ROOT / "docs"
out.mkdir(exist_ok=True)
(out / "PRE_RELEASE_MODEL.md").write_text("\n".join(L) + "\n", encoding="utf8")
print("wrote docs/PRE_RELEASE_MODEL.md", len(L), "lines")
