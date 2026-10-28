import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Counter } from '../components/fx'
import { Disclosure, ErrorState, LoadingBlock, Note, PageHeader, Panel, Reveal } from '../components/ui'
import { CLASS_COLOR, CLASS_RANGE, CLASSES, fmtInt, fmtPct } from '../lib/format'
import type { ModelMetrics, PredictSchema } from '../lib/types'
import { useApi } from '../lib/useApi'

const TECH = [
  ['Modelling', 'Python 3.13 · pandas · scikit-learn (Pipeline, ColumnTransformer) · LightGBM · XGBoost · CatBoost · imbalanced-learn · joblib'],
  ['Explainability', 'SHAP TreeExplainer, computed once offline'],
  ['API', 'FastAPI · Pydantic v2 · uvicorn'],
  ['Interface', 'React 19 · TypeScript · Vite · Tailwind CSS 4 · Recharts · Motion'],
]

function Step({ n, title, children }: { n: string; title: string; children: ReactNode }) {
  return (
    <motion.li initial={{ opacity: 0, x: -24 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="relative grid grid-cols-[3rem_1fr] gap-4 pb-8 last:pb-0 sm:grid-cols-[4.5rem_1fr]">
      <span aria-hidden className="absolute bottom-0 left-[1.45rem] top-12 w-px bg-gradient-to-b from-pink/60 via-violet/40 to-transparent sm:left-[2.05rem]" />
      <span className="relative z-10 flex h-12 w-12 items-center justify-center rounded-full border border-pink/50 bg-[#10092e] font-mono text-sm font-bold text-pink shadow-[0_0_24px_-4px_#ff4fa3] sm:h-[4.1rem] sm:w-[4.1rem] sm:text-base">{n}</span>
      <div className="pt-1.5"><h3 className="font-display text-xl font-bold text-fg">{title}</h3><div className="mt-2 space-y-2 text-[0.92rem] leading-relaxed text-fg2">{children}</div></div>
    </motion.li>
  )
}

function FunnelStrip({ m }: { m: ModelMetrics }) {
  const d = m.evaluation.data
  const items: { v: number; l: string; sub: string; c: string }[] = [
    { v: d.raw_rows, l: 'raw rows', sub: 'IMDb 5000 file', c: '#8b5cff' },
    { v: d.after_exact_duplicate_rows, l: 'no exact duplicates', sub: `${d.raw_rows - d.after_exact_duplicate_rows} identical rows removed`, c: '#25e8d3' },
    { v: d.after_title_year_duplicates, l: 'unique films', sub: `${d.after_exact_duplicate_rows - d.after_title_year_duplicates} same-title-and-year copies removed`, c: '#25e8d3' },
    { v: m.evaluation.train_size, l: 'train', sub: 'stratified 80%', c: '#ffb23e' },
    { v: m.evaluation.test_size, l: 'held-out test', sub: 'stratified 20%, untouched', c: '#ff4fa3' },
    { v: m.selected_model.n_features_in, l: 'pre-release inputs', sub: `→ ${m.selected_model.n_transformed_features} encoded features`, c: '#8b5cff' },
  ]
  return (
    <div className="glass glow-edge is-on relative overflow-hidden rounded-md p-6 sm:p-8">
      <p className="eyebrow mb-5">From raw file to model input</p>
      <ol className="grid grid-cols-2 gap-x-6 gap-y-8 md:grid-cols-3 xl:grid-cols-6">
        {items.map((it, i) => (
          <li key={it.l} className="relative">
            <p className="font-display text-[2.4rem] font-extrabold leading-none sm:text-5xl" style={{ color: it.c, textShadow: `0 0 30px ${it.c}66` }}><Counter value={it.v} duration={1.2 + i * 0.1} /></p>
            <p className="mt-2 text-sm font-semibold text-fg">{it.l}</p>
            <p className="text-[0.74rem] leading-snug text-fg3">{it.sub}</p>
          </li>
        ))}
      </ol>
    </div>
  )
}

export default function About() {
  const { data: m, error, retry } = useApi<ModelMetrics>('/api/model-metrics', { cache: true })
  return (
    <>
      <PageHeader eyebrow="Methodology" title="How it was built — and where it breaks" highlight="where it breaks">
        A capstone on predicting whether a film is a Flop, Average or Hit from its eventual IMDb score band, using <b className="font-semibold text-fg">only information available before release</b>. This page documents the pipeline behind every number in FRAME.
      </PageHeader>
      {error && !m && <ErrorState title="Methodology details unavailable" message={error.message} onRetry={retry} />}
      {!m && !error && <LoadingBlock height="h-96" label="Loading methodology" />}
      {m && <Body m={m} />}
    </>
  )
}

function Body({ m }: { m: ModelMetrics }) {
  const ev = m.evaluation, d = ev.data, h = m.held_out
  const dummy = m.comparison.find((c) => c.name.startsWith('Dummy'))
  const algoNames = (m.algorithms ?? []).filter((a) => a.evaluated && a.key !== 'dummy').map((a) => a.label).join(', ')
  const ref = m.reference_post_release, tv = m.temporal_validation
  const excluded = (m.feature_audit ?? []).filter((a) => a.used === 'no')
  const flops = d.classes_all.Flop, total = d.after_title_year_duplicates
  return (
    <div className="space-y-6">
      <Reveal><FunnelStrip m={m} /></Reveal>

      <Reveal>
        <div className="grid gap-4 xl:grid-cols-12">
          <Panel className="xl:col-span-7" title="Goal" marks tone="#ff4fa3">
            <p className="text-[0.92rem] leading-relaxed text-fg2">Predict a film’s category <b className="font-medium text-fg">at the moment its release is being planned</b>, compare several classifiers fairly, pick one with a rule fixed in advance, and make its behaviour — including its failures — inspectable. The target is <b className="font-medium text-fg">derived from the eventual IMDb score</b>; it measures audience rating, not box office or profit.</p>
            <ul className="mt-4 grid gap-2 sm:grid-cols-3">
              {CLASSES.slice().reverse().map((c) => (
                <li key={c} className="rounded-2xl border border-line bg-white/[0.04] p-3.5"><p className="flex items-center gap-2 font-display text-xl text-fg"><span aria-hidden className="h-2.5 w-2.5 rounded-full" style={{ background: CLASS_COLOR[c] }} />{c}</p>
                  <p className="mt-1 font-mono text-[0.72rem] text-fg3">{CLASS_RANGE[c]}</p>
                  <p className="tnum mt-1 text-[0.8rem] text-fg2">{fmtInt(d.classes_all[c])} films · {fmtPct(d.classes_all[c] / total)}</p></li>
              ))}
            </ul>
            <p className="mt-3 font-mono text-[0.72rem] text-fg3">Rule (unchanged from the original project): pd.cut(imdb_score, bins=[0, 3, 6, 10], right=False). IMDb score is used only to make the label — never as an input.</p>
          </Panel>
          <Panel className="xl:col-span-5" title="Dataset" tone="#25e8d3">
            <dl className="space-y-2.5 text-[0.88rem] text-fg2">
              <div><dt className="font-medium text-fg">IMDb 5000 Movie Dataset</dt><dd>Kaggle (Yueming), ODbL 1.0. {fmtInt(d.raw_rows)} rows × 28 columns, 1916–2016, 66 countries.</dd></div>
              <div><dt className="font-medium text-fg">Cleaning</dt><dd>{fmtInt(d.raw_rows - d.after_exact_duplicate_rows)} identical rows removed, then {fmtInt(d.after_exact_duplicate_rows - d.after_title_year_duplicates)} rows that are the same film (same title and year, vote counts differing by a few) — otherwise one copy could sit in training and its twin in the test set. Films with a missing year are kept.</dd></div>
              <div><dt className="font-medium text-fg">Known quirks</dt><dd>Budget is missing for about 10% of films and is in local currency for non-US films (e.g. ₩12.2 billion). The file ends in 2016 and about 75% of films are American.</dd></div>
            </dl>
          </Panel>
        </div>
      </Reveal>

      <Reveal>
        <Panel title="Pipeline">
          <ol>
            <Step n="01" title="Decide what is allowed">
              <p>Every column was asked: <i>could this value realistically be known before release?</i> {m.selected_model.n_features_in} columns pass (genres, runtime, budget, release year, country, language, content rating, colour, aspect ratio). <b className="font-medium text-fg">{excluded.length} are excluded</b>: gross, votes, reviews and social-media likes only exist after release, and the Facebook-likes columns are a single 2016 snapshot whose timing cannot be tied to each film. Where timing is unknown, the column is left out rather than assumed safe.</p>
              <Disclosure title={`Show the ${excluded.length} excluded columns and why`}>
                <ul className="grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
                  {excluded.map((e) => <li key={e.column}><span className="font-mono text-[0.78rem] text-fg">{e.column}</span> <span className="font-mono text-[0.72rem] uppercase tracking-wider text-fg3">{e.timing.replaceAll('_', ' ')}</span><span className="block text-[0.8rem] text-fg3">{e.reason}</span></li>)}
                </ul>
              </Disclosure>
            </Step>
            <Step n="02" title="Split before anything is learned">
              <p>One stratified 80/20 split (<span className="font-mono">random_state=42</span>): {fmtInt(ev.train_size)} training and {fmtInt(ev.test_size)} test films. The test set holds {ev.test_class_counts.Flop} Flops, {ev.test_class_counts.Average} Average and {ev.test_class_counts.Hit} Hit films, and <b className="font-medium text-fg">was not used for any modelling decision</b>.</p>
            </Step>
            <Step n="03" title="Preprocessing — fitted on training rows only">
              <p>Numeric columns: median imputation with a “was missing” flag, then scaling; budget is log-transformed because it spans four orders of magnitude. Categorical columns: missing becomes its own “Unknown” category, rare categories are pooled, and categories never seen in training fall into that pooled bucket instead of failing. Genres become separate yes/no flags, so any mix of genres is valid. Result: <b className="font-medium text-fg">{m.selected_model.n_transformed_features} encoded features</b>. All of this lives inside the saved scikit-learn <span className="font-mono">Pipeline</span>, so the website applies exactly the preprocessing the model was trained with.</p>
              <p>Director and cast “experience” (how many earlier films they made, counted from training rows only) was tried as an extra feature set and <b className="font-medium text-fg">rejected</b>: it changed paired cross-validated macro F1 by {(m.feature_set?.paired_macro_f1_gain ?? 0).toFixed(3)}, below the 0.01 bar set in advance.</p>
              <InputColumns />
            </Step>
            <Step n="04" title="Compare algorithms fairly">
              <p>{ev.n_configurations ?? m.comparison.length} configurations — {ev.n_algorithms} algorithms ({algoNames}) crossed with imbalance strategies (unweighted, class weights, oversampling applied <i>inside each training fold only</i>, and SMOTE for a subset) — were screened with 5-fold × 3-repeat stratified cross-validation on the training set, with preprocessing refitted inside every fold. Macro F1 was the selection metric because it weighs the rare class as much as the common ones. Linear regression is not a classifier; it appears only as a separate experiment (regress the IMDb score, then convert to classes).</p>
              <MetricGuide />
            </Step>
            <Step n="05" title="Select with a rule fixed in advance">
              <p>The five strongest algorithms were tuned, then the top untuned configurations, the tuned ones and the previously deployed model were re-scored on <b className="font-medium text-fg">identical fresh folds</b>. Rule: anything not significantly worse than the best (corrected paired t &lt; 2) is “indistinguishable”; among those the <b className="font-medium text-fg">simplest</b> model is chosen, then the most stable. {m.selection ? <>Outcome: <b className="font-medium text-fg">{m.selected_model.name}</b> — {m.selection.indistinguishable_from_best.length} of {m.comparison.filter((c) => !c.name.startsWith('Dummy')).length} finalists were tied, so this is a defensible trade-off, not a decisive win. Relative to the previously deployed model: {m.selection.outcome}.</> : null} Several finalists score higher on the test set, which is exactly why the test set was not allowed to decide.</p>
              <TuningDetail m={m} />
            </Step>
            <Step n="06" title="Evaluate once, then stress-test">
              <p>Held-out test, evaluated once: accuracy {fmtPct(h.accuracy)} (always answering “Hit” scores {fmtPct(dummy?.accuracy ?? 0)}), balanced accuracy {fmtPct(h.balanced_accuracy)}, macro F1 {h.macro_f1.toFixed(3)}. Two further checks: the original post-release columns refit on the same data{ref ? ` reach ${fmtPct(ref.test.accuracy)} accuracy but only ${ref.test.macro_f1.toFixed(3)} macro F1` : ''}; and on later films{tv ? ` (train ≤ ${tv.train_through}, test ≥ ${tv.test_from}) accuracy is ${fmtPct(tv.model.accuracy)} against ${fmtPct(tv.majority_baseline.accuracy)} for always-“Hit”` : ''}. See <Link className="text-accent underline underline-offset-4" to="/performance">Model performance</Link>.</p>
            </Step>
            <Step n="07" title="Explain, without over-claiming">
              <p>SHAP values over a 200-film test sample, and the model’s own feature importance. They describe what the model uses, not what causes a film to be liked. See <Link className="text-accent underline underline-offset-4" to="/explain">Explainability</Link>.</p>
            </Step>
          </ol>
        </Panel>
      </Reveal>

      <Reveal>
        <div className="grid gap-4 xl:grid-cols-2">
          <Note title="Limitations">
            <ul className="list-disc space-y-1.5 pl-4">
              <li><b className="font-medium text-fg">Class imbalance.</b> Only {flops} of {fmtInt(total)} films are Flops ({ev.train_class_counts.Flop} in training, {ev.test_class_counts.Flop} in the test set). The model found {h.confusion_matrix[0][0]} of {h.per_class.Flop.support}; no candidate is reliable on this class, and one film moves test-set Flop recall by {(100 / h.per_class.Flop.support).toFixed(1)} points.</li>
              <li><b className="font-medium text-fg">Modest signal.</b> Pre-release information barely beats guessing “Hit” on accuracy; its advantage is in balanced accuracy ({fmtPct(h.balanced_accuracy)} vs 33.3% chance).</li>
              <li><b className="font-medium text-fg">Not validated on the future.</b> The main split is random. The temporal experiment{tv ? ` (accuracy ${fmtPct(tv.model.accuracy)} vs a ${fmtPct(tv.majority_baseline.accuracy)} baseline)` : ''} suggests the signal is weaker for later films, and <span className="font-mono">title_year</span> lets the model learn era effects.</li>
              <li><b className="font-medium text-fg">Target definition.</b> Hit/Average/Flop are IMDb-rating bands, not commercial success.</li>
              <li><b className="font-medium text-fg">Data quirks.</b> Budget mixes currencies for non-US films; content rating is assigned late in production, so it is “probably” rather than certainly pre-release.</li>
              <li><b className="font-medium text-fg">Uncalibrated scores.</b> Class scores are model estimates, not verified probabilities.</li>
            </ul>
          </Note>
          <Note tone="info" title="Future improvements">
            <ul className="list-disc space-y-1.5 pl-4">
              <li>Collect more low-rated films, or merge Average/Flop, so the rare class can be learned and evaluated reliably.</li>
              <li>Add genuinely pre-release signals the dataset lacks: script or franchise information, studio, marketing spend, release window, trailer reception.</li>
              <li>Validate with a rolling temporal split on a dataset extending beyond 2016.</li>
              <li>Convert budgets to one currency and adjust for inflation.</li>
              <li>Calibrate the class probabilities and report calibration curves.</li>
              <li>Add per-prediction SHAP explanations served from the API.</li>
            </ul>
          </Note>
        </div>
      </Reveal>

      <Reveal>
        <div className="grid gap-4 xl:grid-cols-2">
          <Panel title="Technologies"><dl className="space-y-3 text-[0.88rem]">{TECH.map(([k, v]) => <div key={k}><dt className="eyebrow mb-1">{k}</dt><dd className="text-fg2">{v}</dd></div>)}</dl></Panel>
          <Panel title="Integrity of the numbers">
            <ul className="space-y-2 text-[0.88rem] leading-relaxed text-fg2">
              <li>Predictions come from the saved <span className="font-mono">pre_release_movie_classifier.joblib</span> pipeline. The API checks at start-up that the pipeline’s input columns match the saved schema, and refuses to serve otherwise.</li>
              <li>Requests containing a post-release column (votes, gross, reviews, likes, IMDb score) are rejected with an error, never silently used or ignored.</li>
              <li>Every figure on the Performance page is produced by <span className="font-mono">backend/ml/run_comprehensive.py</span>, repeated in the notebook’s pre-release section, and re-checked against the saved model by automated tests.</li>
              <li>The original post-release model file is kept untouched for comparison; the site does not use it.</li>
            </ul>
          </Panel>
        </div>
      </Reveal>
    </div>
  )
}

/* ---- disclosures: every figure comes from the live API ---- */
function InputColumns() {
  const sc = useApi<PredictSchema>('/api/predict/schema', { cache: true }).data
  return (
    <Disclosure title={`Show the ${sc?.columns.length ?? 9} input columns the saved model actually receives`}>
      {!sc ? 'Loading…' : (
        <div className="grid gap-4 sm:grid-cols-2">
          <div><p className="eyebrow mb-2">Numeric · median-imputed, scaled</p><ul className="flex flex-wrap gap-1.5">{sc.numeric.map((c) => <li key={c} className="rounded-md bg-white/[0.07] px-2 py-0.5 font-mono text-[0.72rem] text-fg">{c}</li>)}</ul></div>
          <div><p className="eyebrow mb-2">Categorical · own “Unknown” level, one-hot / genre flags</p><ul className="flex flex-wrap gap-1.5">{sc.columns.filter((c) => !sc.numeric.includes(c)).map((c) => <li key={c} className="rounded-md bg-violet/25 px-2 py-0.5 font-mono text-[0.72rem] text-fg">{c}</li>)}</ul></div>
        </div>
      )}
    </Disclosure>
  )
}

function MetricGuide() {
  return (
    <Disclosure title="Why accuracy alone misleads here">
      <ul className="space-y-2">
        <li><b className="font-semibold text-fg">Accuracy</b> — share of films labelled correctly. Always answering “Hit” already scores about 71%, because about 71% of films are Hits.</li>
        <li><b className="font-semibold text-fg">Balanced accuracy</b> — the average of each class’s recall, so a missed Flop costs as much as a missed Hit. Chance level is 33%.</li>
        <li><b className="font-semibold text-fg">Macro F1</b> — the average F1 over the three classes, equally weighted. A class the model never finds drags it down, exactly as it should.</li>
      </ul>
    </Disclosure>
  )
}

function TuningDetail({ m }: { m: ModelMetrics }) {
  const t = m.selected_model.tuning
  return (
    <Disclosure title="Show the hyperparameter search and the saved settings">
      <div className="space-y-3">
        <p>{t.method}</p>
        <div className="grid gap-4 sm:grid-cols-2">
          <div><p className="eyebrow mb-2">Search space</p><dl className="space-y-1 font-mono text-[0.75rem]">{Object.entries(t.grid).map(([k, v]) => <div key={k} className="flex justify-between gap-3"><dt className="text-fg3">{k}</dt><dd className="text-right text-fg">{v.map((x) => (x === null ? 'None' : String(x))).join(', ')}</dd></div>)}</dl></div>
          <div><p className="eyebrow mb-2">Saved model</p><dl className="space-y-1 font-mono text-[0.75rem]">{Object.entries(m.selected_model.params).map(([k, v]) => <div key={k} className="flex justify-between gap-3"><dt className="text-fg3">{k}</dt><dd className="text-cyan">{String(v)}</dd></div>)}</dl></div>
        </div>
      </div>
    </Disclosure>
  )
}
