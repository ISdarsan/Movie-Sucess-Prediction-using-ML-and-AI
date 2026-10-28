"""Append (or replace) the pre-release study section at the end of PHASE1.ipynb, then execute ONLY those new cells.

    .venv/Scripts/python -m backend.ml.append_notebook_section

Existing cells are left byte-for-byte untouched (the file round-trips exactly). The old cells that call joblib.dump on
final_movie_classifier.joblib are NOT executed.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .notebook_v2 import CELLS_V2, TAG2

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "PHASE1.ipynb"
TAG = "prerelease-section"


def md(src: str):
    return nbformat.v4.new_markdown_cell(src.strip("\n"), metadata={"tags": [TAG]})


def code(src: str):
    return nbformat.v4.new_code_cell(src.strip("\n"), metadata={"tags": [TAG]})


CELLS = [
    md("""
# Pre-release modelling study (strict) — FIRST STUDY, kept for the record

> *The model chosen here (LightGBM) was re-opened by the **comprehensive comparison** in the next section, which evaluates many more algorithms under a stricter, pre-declared rule. This section reads its own preserved `*_v1` files.*

**Added section. Everything above this cell is the original exploration and is left exactly as it was.** Those earlier models use columns that only exist *after* a film is released (votes, reviews, gross, social-media likes), so they classify films retrospectively.

This section answers the capstone's actual question: **can we predict Hit / Average / Flop using only information available *before* release?**

Rules followed, in order:
1. Every column is audited for release timing. Anything whose timing cannot be established is excluded.
2. `imdb_score` is used only to create the label (`Flop < 3 <= Average < 6 <= Hit`, unchanged).
3. Exact duplicates *and* the same film scraped twice (same title + year) are removed before splitting.
4. One stratified 80/20 split. **The test set is not used for any modelling decision.**
5. All preprocessing (medians, scalers, category and genre vocabularies) is fitted on training rows only.
6. Candidates are compared with 5-fold x 3-repeat stratified cross-validation on the training set; macro F1 is the selection metric.
7. The two best candidate types are tuned; the selection rule is fixed *before* any test number is looked at.
8. Test results are reported once, for every candidate, after selection.

The code lives in `backend/ml/` (`features.py`, `run_prerelease.py`) so the notebook, the API and the tests all share one definition.
"""),
    code("""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from backend.ml import features as F
pd.set_option('display.max_colwidth', 150); pd.set_option('display.width', 220)

ART = ROOT / 'backend' / 'artifacts'
# To reproduce this first study: python -m backend.ml.run_prerelease  (writes only the *_v1 files, never the live model)
R = json.loads((ART / 'prerelease_results_v1.json').read_text(encoding='utf8'))
print('Study generated', R['generated_at'], '| seed', R['seed'], '| versions', R['versions'])
"""),
    md("""
## 1. Which columns may the model see?
Question asked of every column: *could this value realistically have been known when the prediction is made, before release?*  Popularity snapshots (Facebook likes) were collected once in 2016 and cannot be tied to each film's release date, so they are excluded rather than assumed safe.
"""),
    code("""
audit = pd.DataFrame(R['audit'])[['column', 'timing', 'used', 'reason']]
display(audit)
print(f"\\nUsed by the strict model: {[c for c in R['feature_set']['inputs']]}")
print('Excluded:', len([a for a in R['audit'] if a['used'] == 'no']), 'columns')
"""),
    md("""
## 2. Clean dataset, target and split
Near-duplicates matter: the same film scraped twice (identical title and year, vote counts differing by a few) survives exact-row de-duplication and can land on both sides of a random split.
"""),
    code("""
d = R['data']
print(f"raw rows {d['raw_rows']:,}  ->  exact duplicate rows removed {d['after_exact_duplicate_rows']:,}  ->  same title+year removed {d['after_title_year_duplicates']:,}")
dist = pd.DataFrame({'all films': d['classes_all'], 'train': d['classes_train'], 'test': d['classes_test']}).loc[F.CLASSES]
dist.loc['total'] = dist.sum()
share = (dist / dist.loc['total']).round(4)
display(dist.join(share, rsuffix=' share'))
print(d['split'])
print("\\nFlop is", f"{d['classes_all']['Flop']} of {d['after_title_year_duplicates']:,} films ({d['classes_all']['Flop']/d['after_title_year_duplicates']:.1%}). The test set holds only {d['classes_test']['Flop']} of them: any Flop figure on it is very noisy.")
"""),
    code("""
# Leakage guard, run live on the saved model: no excluded column and no target column may be an input.
pipe = joblib.load(ROOT / 'pre_release_movie_classifier.v1_lightgbm.joblib')
inputs = list(pipe.named_steps['prep'].feature_names_in_)
assert set(inputs).isdisjoint(F.EXCLUDED) and not {'imdb_score', 'Classify'} & set(inputs)
print('Model inputs :', inputs)
print('Encoded features:', len(pipe.named_steps['prep'].get_feature_names_out()))
print('Leakage guard passed: no post-release, identifier or target column is an input.')
"""),
    md("""
## 3. Feature-set ablation, and what the post-release columns were really buying
Experience features (how many earlier films a director / lead actor made, counted from **training rows only**, strictly earlier years) were tried; they are adopted only if they improve paired CV macro F1 by more than 0.01 without lowering balanced accuracy.

The reference row refits the **original** feature set and the original tuned Random Forest settings on the *same* clean split. It is *not* a candidate; it only quantifies what the excluded columns contributed.
"""),
    code("""
ab = pd.DataFrame({k: {m: v[m][0] if isinstance(v[m], list) else v[m] for m in ('accuracy', 'balanced_accuracy', 'macro_f1', 'flop_recall')} for k, v in R['feature_set']['ablation'].items()}).T
ref = R['reference_post_release_rf_cv']
ab.loc['REFERENCE: original post-release columns (not a candidate)'] = [ref[m][0] for m in ('accuracy', 'balanced_accuracy', 'macro_f1', 'flop_recall')]
display(ab.round(4))
print(f"Paired macro-F1 gain from experience features: {R['feature_set']['paired_macro_f1_gain']:+.4f}  ->  adopted: {R['feature_set']['use_people']}")
rt = ref['test_reporting_only']
print(f"Reference on the TEST split (reporting only): accuracy {rt['accuracy']:.3f}, balanced accuracy {rt['balanced_accuracy']:.3f}, macro F1 {rt['macro_f1']:.3f}")
"""),
    md("""
## 4. Model comparison
Cross-validation (training set only) is what selection uses. The test columns are reported afterwards, for every model, and played no part in the choice.
"""),
    code("""
rows = []
for n, r in R['models'].items():
    c, t = r['cv'], r['test']; pc = t['per_class']
    rows.append({'model': n, 'sel': '<<' if n == R['selected'] else '',
                 'CV macro-F1': f"{c['macro_f1'][0]:.3f} +/- {c['macro_f1'][1]:.3f}", 'CV bal-acc': round(c['balanced_accuracy'][0], 3), 'CV Flop recall': round(c['flop_recall'][0], 3),
                 'test acc': t['accuracy'], 'test bal-acc': t['balanced_accuracy'], 'test macro-F1': t['macro_f1'],
                 'Flop P/R': f"{pc['Flop']['precision']:.2f}/{pc['Flop']['recall']:.2f}", 'Avg P/R': f"{pc['Average']['precision']:.2f}/{pc['Average']['recall']:.2f}", 'Hit P/R': f"{pc['Hit']['precision']:.2f}/{pc['Hit']['recall']:.2f}",
                 '_k': c['macro_f1'][0]})
cmp_df = pd.DataFrame(rows).sort_values('_k', ascending=False).drop(columns='_k').reset_index(drop=True)
display(cmp_df)
"""),
    code("""
names = list(R['models']); cvm = np.array([R['models'][n]['cv']['macro_f1'][0] for n in names]); cvs = np.array([R['models'][n]['cv']['macro_f1'][1] for n in names])
order = np.argsort(cvm)
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ['#2ee6c8' if names[i] == R['selected'] else '#7c5cff' for i in order]
ax.barh([names[i] for i in order], cvm[order], xerr=cvs[order], color=colors, ecolor='#555', capsize=3)
ax.axvline(R['models']['Dummy (most frequent)']['cv']['macro_f1'][0], ls='--', c='gray'); ax.set_xlabel('cross-validated macro F1 (training set, 5 folds x 3 repeats)')
ax.set_title('Candidates are close: differences are within the fold-to-fold spread (error bars = 1 s.d.)'); plt.tight_layout(); plt.show()
"""),
    md("""
## 5. Selection (rule fixed in advance, test set not used)
**Rule:** highest fresh-fold CV macro F1; if models are within 0.005, prefer higher balanced accuracy. Because the finalists are statistically indistinguishable, this is a *tie broken by a pre-declared rule*, not a clear win; that is stated plainly rather than hidden.
"""),
    code("""
print('Selected:', R['selected'])
print('Selection rule:', R['selection_rule'])
pv = pd.DataFrame(R['paired_vs_selected']).T[['mean_diff_macro_f1', 'corrected_se', 't', 'selected_better_in_folds']].round(4)
display(pv.sort_values('mean_diff_macro_f1'))
print('Nadeau-Bengio corrected paired t-statistics: |t| < 2 means no reliable difference. Positive mean_diff = the selected model scored higher.')
best_test = max(R['models'], key=lambda n: R['models'][n]['test']['macro_f1'])
print(f"\\nFor transparency: the highest TEST macro F1 belongs to '{best_test}' ({R['models'][best_test]['test']['macro_f1']}), the selected model scored {R['models'][R['selected']]['test']['macro_f1']}. Picking by test score would be exactly the mistake this protocol avoids.")
"""),
    md("## 6. Final held-out evaluation of the selected model (reproduced live from the saved artifact)"),
    code("""
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
df, _ = F.load_clean(str(ROOT / 'dataset' / 'movie_metadata.csv'))
train, test = train_test_split(df, test_size=0.2, stratify=df['Classify'], random_state=F.SEED)
pred = pipe.predict(test[inputs])
print(classification_report(test['Classify'], pred, labels=F.CLASSES, zero_division=0))
acc, bal, f1m = accuracy_score(test['Classify'], pred), balanced_accuracy_score(test['Classify'], pred), f1_score(test['Classify'], pred, average='macro')
rep = R['models'][R['selected']]['test']
assert (round(acc, 4), round(bal, 4), round(f1m, 4)) == (rep['accuracy'], rep['balanced_accuracy'], rep['macro_f1']), 'saved model does not reproduce the recorded test metrics'
print(f'Reproduced from the saved model: accuracy {acc:.4f}, balanced accuracy {bal:.4f}, macro F1 {f1m:.4f}  (matches the study record)')
base = test['Classify'].value_counts(normalize=True).max()
print(f"Always answering 'Hit' would score accuracy {base:.4f}. Compare the model's balanced accuracy ({bal:.3f}) with 0.333 for chance.")
fig, ax = plt.subplots(figsize=(4.8, 4.2)); ConfusionMatrixDisplay.from_predictions(test['Classify'], pred, labels=F.CLASSES, cmap='Blues', ax=ax, colorbar=False); ax.set_title('Held-out test set'); plt.show()
"""),
    code("""
pooled = R['selected_pooled_oof_train']
print('Pooled out-of-fold predictions on the TRAINING set (5-fold): more Flop evidence than the 8-film test set')
display(pd.DataFrame(pooled['per_class']).T.round(3))
"""),
    md("""
## 7. Does it work on *later* films? (extra experiment, not used for selection)
A random split mixes eras. Here the selected configuration is refit on films up to 2009 and judged on films from 2010 on, which is closer to real pre-release use.
"""),
    code("""
tv = R['temporal_validation']
print(f"train <= {tv['train_through']}: {tv['train_size']:,} films | test >= {tv['test_from']}: {tv['test_size']:,} films | test classes {tv['classes_test']}")
display(pd.DataFrame({'model': {k: tv['model'][k] for k in ('accuracy', 'balanced_accuracy', 'macro_f1')}, 'always-Hit baseline': {k: tv['majority_baseline'][k] for k in ('accuracy', 'balanced_accuracy', 'macro_f1')}}).T)
print('Per-class recall on later films:', {c: tv['model']['per_class'][c]['recall'] for c in F.CLASSES})
"""),
    md("## 8. What the model relies on (association, not cause)"),
    code("""
E = json.loads((ART / 'explain_v1.json').read_text(encoding='utf8'))
gb = pd.DataFrame(E['feature_importance']['grouped_by_column']).head(8)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
axs[0].barh(gb['column'][::-1], gb['importance'][::-1], color='#7c5cff'); axs[0].set_title('Model gain by input column')
sh = pd.DataFrame(E['shap']['Hit']['bar']).head(10)
axs[1].barh(sh['feature'][::-1], sh['mean_abs_shap'][::-1], color='#2ee6c8'); axs[1].set_title(f"Mean |SHAP|, class Hit ({E['method']['sample_size']} test films)")
plt.tight_layout(); plt.show()
print('These show which inputs the model uses, not what makes a film succeed. Genre, runtime, budget and era mostly describe the kind of film.')
"""),
    md("## 9. Conclusions and limitations"),
    code("""
sel = R['models'][R['selected']]; t = sel['test']; dummy = R['models']['Dummy (most frequent)']['test']; refT = R['reference_post_release_rf_cv']['test_reporting_only']
display(Markdown(f'''
**Selected model:** {R['selected']}, chosen by cross-validated macro F1 with a tie-break fixed in advance; the finalists are statistically indistinguishable.

**Held-out test (strict pre-release features):** accuracy {t['accuracy']:.1%} (always-Hit baseline {dummy['accuracy']:.1%}), balanced accuracy {t['balanced_accuracy']:.1%} (chance 33.3%), macro F1 {t['macro_f1']:.3f}.
Flop: {t['confusion_matrix'][0][0]} of {t['per_class']['Flop']['support']} found. Average recall {t['per_class']['Average']['recall']:.0%}, Hit recall {t['per_class']['Hit']['recall']:.0%}.

**What the post-release columns were buying:** the original model reaches {refT['accuracy']:.1%} accuracy on the same clean split but only {refT['macro_f1']:.3f} macro F1, and it finds {refT['confusion_matrix'][0][0]} of {refT['per_class']['Flop']['support']} Flops. Most of the extra accuracy is separating Average from Hit using votes and reviews, information that does not exist before release.

**Limitations.**
1. Only {d['classes_all']['Flop']} Flops exist (the test set has {d['classes_test']['Flop']}); no model here can be trusted on that class, and test-set Flop numbers swing by 12.5 points per film.
2. Pre-release features carry only a modest signal for an IMDb band: accuracy is barely above the majority baseline, and on films from {tv['test_from']} on it is {tv['model']['accuracy']:.1%} against a {tv['majority_baseline']['accuracy']:.1%} baseline.
3. The target is an IMDb-rating band, not box office or profit.
4. `budget` mixes currencies for non-US films, and `title_year` lets the model learn era effects that may not hold for future films.
5. Class scores are uncalibrated model estimates, not real-world probabilities.
'''))
"""),
]


def main() -> None:
    raw = NB.read_bytes()
    text = raw.decode("utf8", errors="surrogateescape")          # keeps the few non-UTF-8 bytes in old cells exactly as they are
    nb = json.loads(text)
    assert (json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf8", errors="surrogateescape") == raw, "notebook does not round-trip; refusing to touch it"
    old = [c for c in nb["cells"] if not ({TAG, TAG2} & set(c.get("metadata", {}).get("tags", [])))]
    n_old = len(old)

    # execute ONLY the new cells, in a throw-away notebook, from the project root
    tmp = nbformat.v4.new_notebook(cells=[copy.deepcopy(c) for c in CELLS + CELLS_V2], metadata={"kernelspec": nb["metadata"]["kernelspec"], "language_info": nb["metadata"].get("language_info", {})})
    NotebookClient(tmp, timeout=900, kernel_name=nb["metadata"]["kernelspec"]["name"], resources={"metadata": {"path": str(ROOT)}}).execute()
    base = max((c.get("execution_count") or 0 for c in old if c["cell_type"] == "code"), default=0)
    new_cells = []
    for c in tmp.cells:
        d = json.loads(nbformat.writes(nbformat.v4.new_notebook(cells=[c])))["cells"][0]
        if d["cell_type"] == "code" and d.get("execution_count") is not None:
            d["execution_count"] = base + d["execution_count"]
            for o in d.get("outputs", []):
                if "execution_count" in o:
                    o["execution_count"] += base
        new_cells.append(d)
    nb["cells"] = old + new_cells
    out = (json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf8", errors="surrogateescape")
    NB.write_bytes(out)
    print(f"kept {n_old} existing cells untouched; appended {len(new_cells)} executed cells")


if __name__ == "__main__":
    sys.exit(main())
