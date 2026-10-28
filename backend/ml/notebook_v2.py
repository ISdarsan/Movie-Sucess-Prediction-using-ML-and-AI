"""Cells of the COMPREHENSIVE model-comparison section appended to PHASE1.ipynb (see append_notebook_section.py).
Every number shown by these cells is read from backend/artifacts/comprehensive_results.json, produced by backend/ml/run_comprehensive.py."""
import nbformat

TAG2 = "comprehensive-section"


def md2(src: str):
    return nbformat.v4.new_markdown_cell(src.strip(), metadata={"tags": [TAG2]})


def code2(src: str):
    return nbformat.v4.new_code_cell(src.strip(), metadata={"tags": [TAG2]})


CELLS_V2 = [
    md2("""
# Comprehensive pre-release model comparison

**This section re-opens the model choice.** The earlier section chose LightGBM from a handful of candidates. Here *every suitable classifier* is evaluated on the same cleaned data, the same nine pre-release inputs, the same target and the same stratified split, under a protocol and selection rule **fixed before any test number was computed**:

1. **Screen** every algorithm x imbalance-strategy configuration with 5-fold x 3-repeat stratified CV on the training set.
2. **Tune** the best configuration of the five strongest algorithms (randomised search, macro F1).
3. **Re-score the finalists on identical fresh folds**, so stage 1-2 winner's-curse optimism is not carried over.
4. **Rule:** best = highest mean macro F1. Anything not significantly worse (Nadeau-Bengio corrected paired t < 2) is *indistinguishable*; among those choose the **simplest** model (fixed complexity ranking), then the more stable, then higher balanced accuracy.
5. Only then: held-out test metrics, temporal evaluation, confusion matrices. The test set, the temporal evaluation and the regression experiment play no part in selection.

Nothing is fabricated: a model that could not run would be listed as not evaluated.
"""),
    code2("""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from backend.ml import features as F
pd.set_option('display.max_colwidth', 120); pd.set_option('display.width', 240); pd.set_option('display.max_rows', 100)
ART = ROOT / 'backend' / 'artifacts'

RUN_STUDY = False   # True re-runs the comparison (a few minutes). It writes a CANDIDATE model and comprehensive_results.json; the live model is only changed by backend/ml/promote_candidate.py
if RUN_STUDY:
    from backend.ml import run_comprehensive
    run_comprehensive.main(quick=False)

R = json.loads((ART / 'comprehensive_results.json').read_text(encoding='utf8'))
print('Generated', R['generated_at'], '| seed', R['seed']); print(R['versions'])
chosen = R['selection']['chosen']; fin = {f['name']: f for f in R['finalists']}
"""),
    md2("## 1. Verified setup, class counts and fold audit"),
    code2("""
display(pd.Series(R['verification'], name='check'))
d = R['data']
dist = pd.DataFrame({'all films': d['classes_all'], 'train': d['classes_train'], 'test': d['classes_test']}).loc[F.CLASSES]
display(dist.join((dist / dist.sum()).round(4), rsuffix=' share'))
fa = R['fold_audit']
print(f"Flops in the training set: {fa['flops_in_training_set']}. Per TRAINING fold: {fa['flops_per_training_fold']}. Per VALIDATION fold: {fa['flops_per_validation_fold']}.")
print(fa['conclusion'])
print('Inputs:', R['feature_set']['inputs'])
"""),
    md2("## 2. Algorithms evaluated, and not evaluated"),
    code2("""
algos = pd.DataFrame(R['algorithms'])[['label', 'library', 'complexity', 'evaluated', 'strategies']]
algos['strategies'] = algos['strategies'].map(', '.join)
display(algos)
print(f"{len(R['screening'])} configurations were executed.")
print('Not evaluated as classifiers:'); [print(' -', s.get('algorithm', s.get('name')), '->', s.get('reason', s.get('error'))) for s in R['skipped']]
"""),
    md2("""
## 3. Stage 1 - screening (cross-validation, training set only)
The imbalance strategies are compared like for like. Oversampling and SMOTE are steps *inside* the CV pipeline, so they only ever touch training folds.
"""),
    code2("""
rows = []
for s in R['screening']:
    c = s['cv']
    rows.append({'configuration': s['name'], 'macro F1': round(c['macro_f1'][0], 4), '+/-': round(c['macro_f1'][1], 4), 'bal. acc': round(c['balanced_accuracy'][0], 4), 'accuracy': round(c['accuracy'][0], 4),
                 'Flop recall': round(c['flop_recall'][0], 3), 'Flop F1': round(c['flop_f1'][0], 3), 'Avg F1': round(c['average_f1'][0], 3), 'Hit F1': round(c['hit_f1'][0], 3), 'fit s': c['fit_seconds']})
scr = pd.DataFrame(rows).sort_values('macro F1', ascending=False).reset_index(drop=True)
display(scr)
"""),
    code2("""
best = {}
for s in R['screening']:
    if s['algorithm'] != 'dummy' and (s['algorithm'] not in best or s['cv']['macro_f1'][0] > best[s['algorithm']]['cv']['macro_f1'][0]): best[s['algorithm']] = s
b = sorted(best.values(), key=lambda s: s['cv']['macro_f1'][0])
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
axs[0].barh([f"{s['algorithm_label']}\\n({s['strategy_label']})" for s in b], [s['cv']['macro_f1'][0] for s in b], xerr=[s['cv']['macro_f1'][1] for s in b], color='#7c5cff', ecolor='#555', capsize=3)
axs[0].axvline(R['baselines'][0]['cv']['macro_f1'][0], ls='--', c='gray'); axs[0].set_xlabel('CV macro F1 (best configuration per algorithm)'); axs[0].set_title('Everything non-trivial sits within the error bars')
strat = {}
for s in R['screening']:
    if s['algorithm'] != 'dummy': strat.setdefault(s['strategy_label'], []).append(s['cv'])
lab = list(strat); w = 0.27
for i, (m, c) in enumerate([('macro_f1', '#2ee6c8'), ('balanced_accuracy', '#ffb23e'), ('accuracy', '#ff6b57')]):
    axs[1].bar(np.arange(len(lab)) + i * w, [np.mean([x[m][0] for x in strat[l]]) for l in lab], w, label=m.replace('_', ' '), color=c)
axs[1].set_xticks(np.arange(len(lab)) + w); axs[1].set_xticklabels(lab, rotation=15); axs[1].set_title('Effect of the imbalance strategy (mean over algorithms)'); axs[1].legend()
plt.tight_layout(); plt.show()
"""),
    md2("## 4. Stage 2 - tuning the five strongest algorithms"),
    code2("""
t = pd.DataFrame([{'tuned': x['name'], 'search CV macro F1': x['search_cv_macro_f1'], 'untuned screening macro F1': next(s['cv']['macro_f1'][0] for s in R['screening'] if s['name'] == x['base_config']), 'best parameters': x['best_params']} for x in R['tuning']])
display(t)
print('Search CV scores are optimistic by construction (the best of many draws); that is why stage 3 re-scores everything on fresh folds.')
"""),
    md2("""
## 5. Stage 3 - finalists on identical fresh folds, and the selection
The rule was written down before this cell could be run.
"""),
    code2("""
sel = R['selection']
print('RULE:', sel['rule'].replace('\\n', ' '), '\\n')
fr = []
for f in sorted(R['finalists'], key=lambda f: -f['cv_final']['macro_f1'][0]):
    c = f['cv_final']; pv = sel['paired_vs_best'][f['name']]
    fr.append({'finalist': f['name'], 'macro F1': round(c['macro_f1'][0], 4), '+/-': round(c['macro_f1'][1], 4), 'bal. acc': round(c['balanced_accuracy'][0], 4), 'complexity': f['complexity'], 'paired t vs best': pv['t'], 'indistinguishable': f['name'] in sel['indistinguishable_from_best'], 'selected': '<<' if f['name'] == chosen else ''})
display(pd.DataFrame(fr))
print('Best by mean :', sel['best_by_mean']); print('CHOSEN       :', chosen, '| complexity rank', sel['chosen_complexity_rank']); print('Previously deployed:', sel['deployed_before'], '->', sel['outcome'])
"""),
    code2("""
names = [f['name'] for f in sorted(R['finalists'], key=lambda f: f['cv_final']['macro_f1'][0])]
m = [fin[n]['cv_final']['macro_f1'][0] for n in names]; e = [fin[n]['cv_final']['macro_f1'][1] for n in names]
fig, ax = plt.subplots(figsize=(10, 5.4)); ax.barh(names, m, xerr=e, color=['#2ee6c8' if n == chosen else '#7c5cff' for n in names], ecolor='#555', capsize=3)
ax.set_xlim(0.40, 0.56); ax.set_xlabel('macro F1 on identical fresh CV folds (error bar = 1 s.d. across folds)'); ax.set_title('The finalists are statistically indistinguishable'); plt.tight_layout(); plt.show()
"""),
    md2("""
## 6. Held-out test results (computed once, after the selection was frozen)
Cross-validation columns come from stage 3; test columns from the untouched 20% test set. Per-class precision / recall / F1 for Flop, Average and Hit are shown for every finalist and the baselines.
"""),
    code2("""
def trow(f, cv, algo, strat, tuned=False, tag=''):
    t = f['test']; pc = t['per_class']
    r = {'algorithm': algo + (' (tuned)' if tuned else '') + tag, 'imbalance': strat, 'CV macro F1': round(cv['macro_f1'][0], 4), 'CV sd': round(cv['macro_f1'][1], 4), 'test acc': t['accuracy'], 'test bal acc': t['balanced_accuracy'], 'test macro F1': t['macro_f1']}
    for c in F.CLASSES: r[f'{c} P/R/F1'] = f"{pc[c]['precision']:.2f}/{pc[c]['recall']:.2f}/{pc[c]['f1-score']:.2f}"
    r['fit s'] = f['timing']['fit_seconds']; r['predict ms/1000'] = f['timing']['predict_ms_per_1000_films']; return r
rows = [trow(f, f['cv_final'], f['algorithm_label'], f['strategy_label'], f['tuned'], ' <<SELECTED' if f['name'] == chosen else (' [deployed before]' if f.get('deployed_before') else '')) for f in sorted(R['finalists'], key=lambda f: -f['cv_final']['macro_f1'][0])]
rows += [trow(b, b['cv'], b['algorithm_label'], b['strategy_label']) for b in R['baselines']]
display(pd.DataFrame(rows))
bt = max(R['finalists'], key=lambda f: f['test']['macro_f1'])
print(f"Highest TEST macro F1: {bt['name']} ({bt['test']['macro_f1']}). Selected: {chosen} ({fin[chosen]['test']['macro_f1']}).")
print('If these differ, that is what a ranking inside the noise looks like (8 test Flops, CV spread ~0.03); the rule is NOT changed after seeing test data.')
"""),
    code2("""
fs = sorted(R['finalists'], key=lambda f: -f['cv_final']['macro_f1'][0])
n = len(fs); cols = 5; rws = int(np.ceil(n / cols))
fig, axs = plt.subplots(rws, cols, figsize=(3.3 * cols, 3.3 * rws)); axs = np.atleast_1d(axs).ravel()
for ax, f in zip(axs, fs):
    cm = np.array(f['test']['confusion_matrix']); ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j, i, cm[i, j], ha='center', va='center', color='k', fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(['F', 'A', 'H']); ax.set_yticks(range(3)); ax.set_yticklabels(['F', 'A', 'H'])
    ax.set_title(f['name'].replace(' · ', '\\n').replace(' (in fold)', ''), fontsize=7.5, color='#0a7' if f['name'] == chosen else 'k')
for ax in axs[n:]: ax.axis('off')
plt.suptitle('Confusion matrices on the held-out test set (rows true, columns predicted; F=Flop, A=Average, H=Hit)', y=1.0); plt.tight_layout(); plt.show()
"""),
    code2("""
# Reproduce the selected model's classification report LIVE from the saved artifact (candidate if present, else the live model)
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
path = ROOT / ('pre_release_movie_classifier.candidate.joblib' if (ROOT / 'pre_release_movie_classifier.candidate.joblib').exists() else 'pre_release_movie_classifier.joblib')
model = joblib.load(path)
df, _ = F.load_clean(str(ROOT / 'dataset' / 'movie_metadata.csv')); train, test = train_test_split(df, test_size=0.2, stratify=df['Classify'], random_state=F.SEED)
pred = model.predict(test[F.BASE_INPUTS])
print('artifact:', path.name); print(classification_report(test['Classify'], pred, labels=F.CLASSES, zero_division=0))
rec = fin[chosen]['test']
assert (round(accuracy_score(test['Classify'], pred), 4), round(balanced_accuracy_score(test['Classify'], pred), 4)) == (rec['accuracy'], rec['balanced_accuracy']), 'artifact does not reproduce the recorded test metrics'
print('Reproduced from the saved artifact: matches the recorded test metrics.')
dep = fin.get(sel['deployed_before']); mj = R['baselines'][0]['test']
print(f"vs previously deployed ({sel['deployed_before']}): acc {rec['accuracy']} vs {dep['test']['accuracy']}, bal {rec['balanced_accuracy']} vs {dep['test']['balanced_accuracy']}, macro F1 {rec['macro_f1']} vs {dep['test']['macro_f1']}")
print(f"vs majority class: acc {rec['accuracy']} vs {mj['accuracy']}, bal {rec['balanced_accuracy']} vs {mj['balanced_accuracy']}, macro F1 {rec['macro_f1']} vs {mj['macro_f1']}")
"""),
    md2("""
## 7. Temporal generalisation (reporting only)
Everything is refit on the earlier period only. Class counts and the majority baseline are shown for each window; Flop counts in later windows are too small to interpret.
"""),
    code2("""
tw = R['temporal']; rows = []
for wd in [tw['main']] + tw['rolling']:
    m = wd['models'][chosen]; dp = wd['models'].get(sel['deployed_before'])
    rows.append({'window': wd['label'], 'train': wd['train_size'], 'test': wd['test_size'], 'test classes F/A/H': '/'.join(str(wd['classes_test'][c]) for c in F.CLASSES), 'majority acc': wd['majority_baseline']['accuracy'],
                 'selected acc': m['accuracy'], 'selected bal acc': m['balanced_accuracy'], 'selected macro F1': m['macro_f1'], 'Flops found': f"{m['confusion_matrix'][0][0]}/{m['per_class']['Flop']['support']}",
                 'deployed-before acc': dp['accuracy'] if dp else None, 'deployed-before macro F1': dp['macro_f1'] if dp else None})
display(pd.DataFrame(rows)); print(tw['note'])
display(pd.DataFrame({n: {k: m[k] for k in ('accuracy', 'balanced_accuracy', 'macro_f1')} for n, m in tw['main']['models'].items()}).T)
"""),
    md2("""
## 8. Separate experiment: regression on the IMDb score
Linear regression is not a classifier, so it is not a candidate. As its own experiment: predict the continuous IMDb score from the same nine inputs, then convert with the target's own bins. Not eligible for selection.
"""),
    code2("""
print(R['regression_experiment']['definition'])
display(pd.DataFrame([{'regressor': r['name'], 'macro F1': r['macro_f1'][0], '+/-': r['macro_f1'][1], 'bal acc': r['balanced_accuracy'][0], 'accuracy': r['accuracy'][0]} for r in R['regression_experiment']['rows']]))
"""),
    md2("## 9. Conclusions and limitations"),
    code2("""
ct = fin[chosen]['test']; top_flop = max(R['finalists'], key=lambda f: f['test']['per_class']['Flop']['recall']); mj = R['baselines'][0]['test']; tm = R['temporal']['main']
display(Markdown(f'''
**Selected: {chosen}** (complexity rank {sel['chosen_complexity_rank']}). {len(sel['indistinguishable_from_best'])} of {len(R['finalists'])} finalists were statistically indistinguishable from the best, so the pre-declared rule chose the simplest of them. This is a defensible trade-off, **not a decisive winner**; the previously deployed model was `{sel['deployed_before']}` ({sel['outcome']}).

**Held-out test:** accuracy {ct['accuracy']:.1%} (majority class {mj['accuracy']:.1%}), balanced accuracy {ct['balanced_accuracy']:.1%} (chance 33.3%), macro F1 {ct['macro_f1']:.3f}. Flops found: {ct['confusion_matrix'][0][0]} of {ct['per_class']['Flop']['support']}.

**Flop detection did not meaningfully improve.** The finalist flagging most Flops, `{top_flop['name']}`, finds {top_flop['test']['confusion_matrix'][0][0]} of {top_flop['test']['per_class']['Flop']['support']} but only {top_flop['test']['per_class']['Flop']['precision']:.0%} of its Flop calls are right. With {R['fold_audit']['flops_in_training_set']} training Flops and {R['fold_audit']['flops_per_validation_fold'][0]}-{R['fold_audit']['flops_per_validation_fold'][1]} per validation fold, no Flop figure is reliable.

**Later films ({tm['label']}):** accuracy {tm['models'][chosen]['accuracy']:.1%} vs {tm['majority_baseline']['accuracy']:.1%} for the majority class; test classes {tm['classes_test']}.

**Limitations:** pre-release information is a modest predictor; the label is an IMDb-rating band, not commercial success; the main split is random; `title_year` lets models learn era effects; `budget` mixes currencies; class scores are uncalibrated.
'''))
"""),
]
