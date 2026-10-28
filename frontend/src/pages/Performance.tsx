import { AnimatePresence, motion } from 'motion/react'
import { lazy, Suspense, useState } from 'react'
import { Gauge } from '../components/fx'
import { Disclosure, ErrorState, LoadingBlock, Note, PageHeader, Panel, Reveal, Segmented, cx } from '../components/ui'
import { CLASS_COLOR, CLASSES, fmtInt, fmtPct } from '../lib/format'
import type { ModelMetrics, ModelResult, ScreeningRow } from '../lib/types'
import { useApi } from '../lib/useApi'

export default function Performance() {
  const { data, error, loading, retry } = useApi<ModelMetrics>('/api/model-metrics', { cache: true })
  return (
    <>
      <PageHeader eyebrow="Model performance" title="How good is the model, really?" highlight="really">
        Held-out results for the saved pre-release model, set against every other algorithm tried on the same features. Pre-release information is a weak predictor of how a film is rated — that is shown first, not buried.
      </PageHeader>
      {error && !data && <ErrorState title="Metrics unavailable" message={error.message} onRetry={retry} />}
      {loading && !data && <LoadingBlock height="h-[32rem]" label="Loading metrics" />}
      {data && <Body m={data} />}
    </>
  )
}

function Body({ m }: { m: ModelMetrics }) {
  const h = m.held_out
  const ev = m.evaluation
  const dummy = m.comparison.find((c) => c.name.startsWith('Dummy'))
  const base = Math.max(...Object.values(ev.test_class_counts)) / ev.test_size
  const flopsFound = h.confusion_matrix[0][0]
  const ref = m.reference_post_release
  const tv = m.temporal_validation
  const topFlop = [...m.comparison].filter((c) => !c.name.startsWith('Dummy')).sort((a, b) => b.per_class.Flop.recall - a.per_class.Flop.recall)[0]
  const near = Object.entries(m.paired_vs_selected ?? {}).filter(([, v]) => v.t !== null && Math.abs(v.t) < 2).length
  const total = Object.keys(m.paired_vs_selected ?? {}).length
  return (
    <div className="space-y-5">
      <Reveal>
        <Note title="Read this first: a modest signal, and almost no Flop detection">
          <p>On the held-out test set the model scores <b className="text-fg">{fmtPct(h.accuracy)}</b> accuracy; always answering “Hit” scores <b className="text-fg">{fmtPct(dummy?.accuracy ?? base)}</b>. Its edge shows up in <b className="text-fg">balanced accuracy ({fmtPct(h.balanced_accuracy)} vs 33.3% for chance)</b> and macro F1, which weigh every class equally.</p>
          <p>Flop is rare: {ev.train_class_counts.Flop} training films and only <b className="text-fg">{ev.test_class_counts.Flop}</b> in the test set. The model found <b className="text-fg">{flopsFound}</b> of them, {topFlop && topFlop.name !== h.name ? <>The candidate that flags the most Flops, <b className="text-fg">{topFlop.name}</b>, finds {topFlop.confusion_matrix[0][0]} of {topFlop.per_class.Flop.support} — but only {fmtPct(topFlop.per_class.Flop.precision, 0)} of the films it calls Flop really are, and its overall accuracy falls to {fmtPct(topFlop.accuracy, 0)}. </> : null}No candidate finds Flops reliably, and any Flop figure here is very noisy (one film moves it by {(100 / h.per_class.Flop.support).toFixed(1)} points).</p>
        </Note>
      </Reveal>

      <Reveal>
        <div className="glass glow-edge is-on grid grid-cols-2 gap-y-8 rounded-md px-4 py-8 sm:px-8 xl:grid-cols-4">
          <Gauge value={h.accuracy} label="Accuracy" hint={`always answering “Hit” scores ${fmtPct(dummy?.accuracy ?? base)}`} color="#25e8d3" />
          <Gauge value={h.balanced_accuracy} label="Balanced accuracy" hint="mean of per-class recall; chance is 33.3%" color="#8b5cff" />
          <Gauge value={h.macro_f1} label="Macro F1" hint="unweighted mean F1 over the 3 classes" color="#ffb23e" />
          <Gauge value={h.per_class.Flop.recall} label="Flop recall" hint={`${flopsFound} of ${h.per_class.Flop.support} Flops found`} color="#ff4f7b" />
        </div>
        <p className="mt-2 text-[0.75rem] text-fg3">Held-out test set: {fmtInt(ev.test_size)} films, never used for training, tuning or model selection. {ev.split}</p>
      </Reveal>

      <Reveal>
        <div className="grid gap-4 xl:grid-cols-12">
          <Panel className="min-w-0 xl:col-span-6" title="Confusion matrix" subtitle="Rows are the true class, columns the model’s answer. Colour = share of that true class.">
            <ConfusionMatrix r={h} />
          </Panel>
          <Panel className="min-w-0 xl:col-span-6" title="Per-class precision, recall and F1" subtitle="Precision: of films predicted as the class, how many were right. Recall: of films truly in the class, how many were found.">
            <PerClass r={h} />
          </Panel>
        </div>
      </Reveal>

      {(ref || tv) && (
        <Reveal>
          <div className="grid gap-4 xl:grid-cols-2">
            {ref && (
              <Panel className="min-w-0" title="Why the numbers are lower than before" subtitle="The original model used votes, reviews, gross and social-media likes — none of which exist before release." tone="#ffb23e">
                <CompareTable rows={[
                  { label: 'Accuracy', a: ref.test.accuracy, b: h.accuracy },
                  { label: 'Balanced accuracy', a: ref.test.balanced_accuracy, b: h.balanced_accuracy },
                  { label: 'Macro F1', a: ref.test.macro_f1, b: h.macro_f1 },
                  { label: 'Flops found', a: `${ref.test.confusion_matrix[0][0]} of ${ref.test.per_class.Flop.support}`, b: `${flopsFound} of ${h.per_class.Flop.support}` },
                ]} aLabel="Original (post-release columns)" bLabel="This model (pre-release)" />
                <p className="mt-3 text-[0.78rem] leading-snug text-fg3">Same cleaned data, same split. The post-release columns mostly help tell Average from Hit (accuracy), not find Flops. {ref.description}</p>
              </Panel>
            )}
            {tv && (
              <Panel className="min-w-0" title="Does it work on later films?" subtitle={`Refit on films up to ${tv.train_through}, judged on the ${fmtInt(tv.test_size)} films from ${tv.test_from} on. An extra experiment — not used for selection.`} tone="#25e8d3">
                <CompareTable rows={[
                  { label: 'Accuracy', a: tv.majority_baseline.accuracy, b: tv.model.accuracy },
                  { label: 'Balanced accuracy', a: tv.majority_baseline.balanced_accuracy, b: tv.model.balanced_accuracy },
                  { label: 'Macro F1', a: tv.majority_baseline.macro_f1, b: tv.model.macro_f1 },
                  { label: 'Flops found', a: '0', b: `${tv.model.confusion_matrix[0][0]} of ${tv.model.per_class.Flop.support}` },
                ]} aLabel="Always “Hit”" bLabel="This model" />
                <p className="mt-3 text-[0.78rem] leading-snug text-fg3">A random split mixes eras. On genuinely later films the model is only {((tv.model.accuracy - tv.majority_baseline.accuracy) * 100).toFixed(1)} points more accurate than guessing “Hit”, so treat it as a weak, era-dependent signal.</p>
                {m.temporal && (
                  <table className="mt-4 w-full text-[0.78rem]">
                    <caption className="sr-only">Rolling time windows</caption>
                    <thead><tr className="text-right font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-1 text-left font-medium">Window</th><th className="font-medium">Flops in test</th><th className="font-medium">Always “Hit”</th><th className="font-medium text-cyan">Model</th></tr></thead>
                    <tbody className="divide-y divide-line tnum font-mono">
                      {m.temporal.rolling.map((w) => { const r = w.models[m.selected_model.name]; return <tr key={w.label} className="text-right"><td className="py-1.5 text-left font-sans text-fg2">{w.label.replace('train ≤ ', '≤').replace(' → test ', ' → ')}</td><td className="text-fg3">{w.classes_test.Flop}</td><td className="text-fg2">{fmtPct(w.majority_baseline.accuracy, 1)}</td><td className="font-semibold text-fg">{r ? fmtPct(r.accuracy, 1) : '—'}</td></tr> })}
                    </tbody>
                  </table>
                )}
              </Panel>
            )}
          </div>
        </Reveal>
      )}

      <Reveal>
        <Panel className="min-w-0" title="Model comparison" subtitle="Every algorithm on the same features and split. Selection used only the CV column (macro F1 on fresh folds of the training set); the test columns were computed afterwards for reporting.">
          <Comparison m={m} near={near} total={total} />
        </Panel>
      </Reveal>

      {m.selection && (
        <Reveal>
          <Panel className="min-w-0" title="How the model was chosen" subtitle="A protocol and a rule fixed before any test number existed." tone="#8b5cff">
            <Selection m={m} />
          </Panel>
        </Reveal>
      )}

      <Reveal>
        <div className="grid gap-4 xl:grid-cols-12">
          <Panel className="min-w-0 xl:col-span-7" title="Cross-validation" subtitle="Training set only: this is what the model was chosen on. Not the same thing as the held-out test above.">
            <CrossVal m={m} />
          </Panel>
          <Panel className="min-w-0 xl:col-span-5" title="Selected model" subtitle={`${m.selected_model.name} — preprocessing + classifier in one saved pipeline`}>
            <Selected m={m} />
          </Panel>
        </div>
      </Reveal>

      <Reveal>
        <Panel title="How these numbers were produced">
          <ul className="grid gap-3 text-[0.85rem] leading-relaxed text-fg2 lg:grid-cols-3">
            <li><b className="font-medium text-fg">One protocol, fixed in advance.</b> {m.evaluation.cv_protocol}. Selection rule: {m.evaluation.selection_rule}</li>
            <li><b className="font-medium text-fg">Reproducible.</b> Every figure comes from <span className="font-mono text-[0.8em]">backend/ml/run_comprehensive.py</span> (seed 42) and is repeated in the notebook’s pre-release section; the saved model is re-evaluated on the test split by the test-suite.</li>
            <li><b className="font-medium text-fg">Honest limits.</b> {m.evaluation.source} Artifacts generated {new Date(m.generated_at).toLocaleString()}.</li>
          </ul>
        </Panel>
      </Reveal>
    </div>
  )
}

function CompareTable({ rows, aLabel, bLabel }: { rows: { label: string; a: number | string; b: number | string }[]; aLabel: string; bLabel: string }) {
  const f = (v: number | string) => (typeof v === 'number' ? fmtPct(v, 1) : v)
  return (
    <table className="w-full text-sm">
      <thead><tr className="text-right font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-1 text-left font-medium"><span className="sr-only">Metric</span></th><th className="font-medium">{aLabel}</th><th className="font-medium text-cyan">{bLabel}</th></tr></thead>
      <tbody className="divide-y divide-line tnum font-mono text-[0.85rem]">
        {rows.map((r) => <tr key={r.label} className="text-right"><td className="py-2 text-left font-sans text-fg">{r.label}</td><td className="text-fg2">{f(r.a)}</td><td className="font-semibold text-fg">{f(r.b)}</td></tr>)}
      </tbody>
    </table>
  )
}

function ConfusionMatrix({ r }: { r: ModelResult }) {
  const [mode, setMode] = useState<'count' | 'pct'>('count')
  const [sel, setSel] = useState<[number, number]>([0, 0])
  const cm = r.confusion_matrix
  const biggestMiss = Math.max(...cm.flatMap((row, i) => row.filter((_, j) => i !== j)))
  const explain = (() => {
    const [i, j] = sel, v = cm[i][j], total = cm[i].reduce((a, b) => a + b, 0), row = CLASSES[i], col = CLASSES[j]
    const pct = fmtPct(total ? v / total : 0, 0)
    const lead = i === j
      ? `${fmtInt(v)} of ${fmtInt(total)} films that are truly ${row} (${pct}) were correctly labelled ${row}.${v === 0 ? ' The model found none of them.' : ''}`
      : `${fmtInt(v)} of ${fmtInt(total)} films that are truly ${row} (${pct}) were wrongly labelled ${col}.${v === biggestMiss && v > 0 ? ' This is the largest single kind of mistake.' : ''}`
    const small = total < 30 ? ` With only ${total} ${row} films in the test set, each film moves this percentage by ${(100 / total).toFixed(0)} points, so treat it as very noisy.` : ''
    return { lead, small, tone: i === j ? CLASS_COLOR[row] : 'var(--color-fg3)' }
  })()
  return (
    <div>
      <div className="mb-3"><Segmented label="Matrix values" value={mode} onChange={setMode} options={[{ value: 'count', label: 'Counts' }, { value: 'pct', label: '% of true class' }]} /></div>
      <div className="grid grid-cols-[auto_repeat(3,minmax(0,1fr))] gap-1.5 text-center" role="table" aria-label="Confusion matrix">
        <div className="contents" role="row">
          <div role="columnheader"><span className="sr-only">True class</span></div>
          {CLASSES.map((c) => <div key={c} className="pb-1 text-[0.72rem] font-mono uppercase tracking-wider text-fg3" role="columnheader">Pred. {c}</div>)}
        </div>
        {CLASSES.map((rowC, i) => {
          const total = cm[i].reduce((a, b) => a + b, 0)
          return (
            <div key={rowC} className="contents" role="row">
              <div className="flex items-center pr-2 text-left text-[0.72rem] font-mono uppercase tracking-wider text-fg3 [writing-mode:horizontal-tb]" role="rowheader">True {rowC}</div>
              {CLASSES.map((colC, j) => {
                const v = cm[i][j]; const share = total ? v / total : 0
                const diag = i === j
                const col = diag ? CLASS_COLOR[rowC] : 'var(--color-fg3)'
                return (
                  <motion.div key={colC} role="cell" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.05 * (i * 3 + j), duration: 0.25 }}
                    tabIndex={0} aria-label={`True ${rowC}, predicted ${colC}: ${v} films`} onPointerEnter={() => setSel([i, j])} onFocus={() => setSel([i, j])} onClick={() => setSel([i, j])} whileHover={{ scale: 1.035 }} whileTap={{ scale: 0.98 }}
                    className={`flex aspect-[4/3] min-h-16 cursor-pointer flex-col items-center justify-center rounded-2xl border border-line outline-none transition-shadow ${sel[0] === i && sel[1] === j ? 'ring-2 ring-white/80 ring-offset-2 ring-offset-[#06061a]' : ''}`}
                    style={{ background: `color-mix(in srgb, ${col} ${Math.round(8 + share * 62)}%, #0d0b2e)`, borderColor: diag ? `color-mix(in srgb, ${col} 70%, transparent)` : undefined, boxShadow: diag && share > 0.05 ? `0 0 34px -6px ${col}` : undefined }}
                    title={`${v} of ${total} true ${rowC} films predicted ${colC}`}>
                    <span className="tnum font-display text-xl font-extrabold text-fg sm:text-3xl">{mode === 'count' ? fmtInt(v) : fmtPct(share, 0)}</span>
                    <span className="tnum font-mono text-[0.72rem] text-fg2">{mode === 'count' ? fmtPct(share, 0) : fmtInt(v)}</span>
                  </motion.div>
                )
              })}
            </div>
          )
        })}
      </div>
      <div className="mt-4 min-h-[5.5rem] rounded-2xl border border-line bg-black/20 p-4" aria-live="polite">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={`${sel[0]}-${sel[1]}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.16 }}>
            <p className="eyebrow mb-1.5 flex items-center gap-2"><span aria-hidden className="h-2 w-2 rounded-full" style={{ background: explain.tone }} />True {CLASSES[sel[0]]} → predicted {CLASSES[sel[1]]}</p>
            <p className="text-[0.88rem] leading-relaxed text-fg">{explain.lead}<span className="text-fg3">{explain.small}</span></p>
          </motion.div>
        </AnimatePresence>
        <p className="mt-2 text-[0.72rem] text-fg3">Hover, focus or tap any cell to read it. The outlined diagonal holds correct answers.</p>
      </div>
    </div>
  )
}

const PerClassChart = lazy(() => import('../components/ModelCharts').then((m) => ({ default: m.PerClassChart })))

function PerClass({ r }: { r: ModelResult }) {
  return (
    <>
      <Suspense fallback={<div aria-hidden className="skeleton h-60 w-full" />}><PerClassChart r={r} /></Suspense>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[22rem] text-sm">
          <caption className="sr-only">Per-class metrics</caption>
          <thead><tr className="text-left font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-2 pr-3 font-medium">Class</th><th className="font-medium">Precision</th><th className="font-medium">Recall</th><th className="font-medium">F1</th><th className="text-right font-medium">Support</th></tr></thead>
          <tbody className="divide-y divide-line tnum font-mono text-[0.82rem]">
            {CLASSES.map((c) => (
              <tr key={c}><td className="py-2 pr-3 font-sans text-fg"><span aria-hidden className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: CLASS_COLOR[c] }} />{c}</td>
                <td>{fmtPct(r.per_class[c].precision, 0)}</td><td className={c === 'Flop' ? 'text-danger' : ''}>{fmtPct(r.per_class[c].recall, 0)}</td><td>{fmtPct(r.per_class[c]['f1-score'], 0)}</td><td className="text-right text-fg3">{fmtInt(r.per_class[c].support)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

type SortKey = 'cv_macro_f1' | 'accuracy' | 'balanced_accuracy' | 'macro_f1'
const num = (r: ModelResult, k: SortKey) => (r[k] as number | undefined) ?? 0
function Comparison({ m, near, total }: { m: ModelMetrics; near: number; total: number }) {
  const [sort, setSort] = useState<SortKey>('cv_macro_f1')
  const rows = [...m.comparison].sort((a, b) => num(b, sort) - num(a, sort))
  const best = (k: SortKey) => Math.max(...m.comparison.map((r) => num(r, k)))
  const th = (k: SortKey, label: string) => (
    <th scope="col" className="py-2 text-right font-medium" aria-sort={sort === k ? 'descending' : 'none'}>
      <button onClick={() => setSort(k)} className={cx('min-h-9 rounded-xs px-2 uppercase tracking-wider hover:text-fg', sort === k ? 'text-pink' : 'text-fg3')}>{label}{sort === k ? ' ↓' : ''}</button>
    </th>
  )
  return (
    <div className="relative overflow-x-auto">
      <table className="w-full min-w-[58rem] text-sm">
        <caption className="sr-only">Cross-validation and held-out test metrics for every model</caption>
        <thead><tr className="border-b border-line font-mono text-[0.72rem] text-fg3">
          <th scope="col" className="py-2 text-left font-medium uppercase tracking-wider">Model</th>
          {th('cv_macro_f1', 'CV macro F1')}
          {th('accuracy', 'Test acc.')}{th('balanced_accuracy', 'Test bal. acc.')}{th('macro_f1', 'Test macro F1')}
          <th scope="col" className="py-2 text-right font-medium uppercase tracking-wider">Recall F / A / H</th>
        </tr></thead>
        <tbody className="divide-y divide-line">
          {rows.map((r) => (
            <tr key={r.name} className={cx(r.selected && 'bg-pink/[0.09] outline outline-1 -outline-offset-1 outline-pink/60')}>
              <td className="py-3 pr-3"><span className="font-medium text-fg">{r.name}</span>{r.selected && <span className="ml-2 rounded-full bg-[image:var(--grad-brand)] px-2 py-0.5 font-mono text-[0.72rem] font-bold uppercase text-[#12082a]">Selected</span>}
                <span className="block text-[0.72rem] text-fg3">{r.note}</span></td>
              <td className="tnum min-w-28 text-right font-mono text-[0.85rem] text-fg">{fmtPct(r.cv_macro_f1 ?? 0, 1)} <span className="text-fg3">± {((r.cv_macro_f1_std ?? 0) * 100).toFixed(1)}</span>
                <span aria-hidden className="ml-auto mt-1.5 block h-1 w-20 overflow-hidden rounded-full bg-white/10"><span className="block h-full rounded-full" style={{ width: `${(r.cv_macro_f1 ?? 0) * 100}%`, background: r.selected ? 'var(--grad-brand)' : 'rgb(255 255 255 / .45)' }} /></span></td>
              {(['accuracy', 'balanced_accuracy', 'macro_f1'] as SortKey[]).map((k) => (
                <td key={k} className={cx('tnum min-w-24 text-right font-mono text-[0.85rem]', num(r, k) === best(k) ? 'font-semibold text-hit' : 'text-fg2')}>{fmtPct(num(r, k), 1)}{num(r, k) === best(k) && <span className="sr-only"> (best)</span>}
                  <span aria-hidden className="ml-auto mt-1.5 block h-1 w-20 overflow-hidden rounded-full bg-white/10"><span className="block h-full rounded-full" style={{ width: `${num(r, k) * 100}%`, background: r.selected ? 'var(--grad-brand)' : 'rgb(255 255 255 / .45)' }} /></span></td>
              ))}
              <td className="tnum text-right font-mono text-[0.8rem] text-fg3">{CLASSES.map((c) => fmtPct(r.per_class[c].recall, 0)).join(' / ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-[0.78rem] leading-snug text-fg3">Green = best value in each column. <b className="font-medium text-fg2">The finalists are tied:</b> in paired fold-by-fold tests (Nadeau–Bengio corrected) {near} of {total} other candidates differ from the selected model by less than 2 standard errors, so the choice rests on a rule fixed in advance — among models not significantly worse than the best, take the simplest, then the most stable — not on a clear win. Some other models score higher on the test set; that is exactly why the test set was not allowed to decide.</p>
    </div>
  )
}

function CrossVal({ m }: { m: ModelMetrics }) {
  const f = m.cross_validation.five_fold_train
  const r = m.cross_validation.repeated
  return (
    <div className="space-y-5">
      <div>
        <p className="mb-2 text-[0.8rem] text-fg2">{f.protocol}</p>
        <table className="w-full text-sm"><thead><tr className="text-right font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-1 text-left font-medium">Model</th><th className="font-medium">Acc.</th><th className="font-medium">Bal. acc.</th><th className="font-medium">Macro F1 ± sd</th></tr></thead>
          <tbody className="divide-y divide-line tnum font-mono text-[0.82rem]">{f.rows.map((x) => <tr key={x.name} className="text-right"><td className="py-2 text-left font-sans text-fg">{x.name}</td><td>{fmtPct(x.accuracy)}</td><td>{fmtPct(x.balanced_accuracy)}</td><td>{fmtPct(x.macro_f1)} <span className="text-fg3">± {(x.macro_f1_std * 100).toFixed(1)}</span></td></tr>)}</tbody></table>
      </div>
      <div>
        <p className="mb-2 text-[0.8rem] text-fg2">{r.protocol}</p>
        <table className="w-full text-sm"><thead><tr className="text-right font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-1 text-left font-medium">Model</th><th className="font-medium">Bal. acc. ± sd</th><th className="font-medium">Macro F1 ± sd</th></tr></thead>
          <tbody className="divide-y divide-line tnum font-mono text-[0.82rem]">{r.rows.map((x) => <tr key={x.name} className="text-right"><td className="py-2 text-left font-sans text-fg">{x.name}</td><td>{fmtPct(x.balanced_accuracy)} <span className="text-fg3">± {(x.balanced_accuracy_std * 100).toFixed(1)}</span></td><td>{fmtPct(x.macro_f1)} <span className="text-fg3">± {(x.macro_f1_std * 100).toFixed(1)}</span></td></tr>)}</tbody></table>
      </div>
      <p className="text-[0.75rem] leading-snug text-fg3">Fold-to-fold spread (± sd) is large because each validation fold holds only about {Math.round(m.evaluation.train_class_counts.Flop / 5)} Flops. Differences of a point or two between models are within that noise.</p>
    </div>
  )
}

function Selected({ m }: { m: ModelMetrics }) {
  const s = m.selected_model
  return (
    <div className="space-y-4">
      <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-[0.82rem]">
        {Object.entries(s.params).map(([k, v]) => <div key={k} className="contents"><dt className="font-mono text-fg3">{k}</dt><dd className="text-right font-mono text-fg">{String(v)}</dd></div>)}
      </dl>
      <div className="border-t border-line pt-3 text-[0.8rem] leading-relaxed text-fg2">
        <p><b className="font-medium text-fg">Input:</b> {s.n_features_in} pre-release columns → {fmtInt(s.n_transformed_features)} encoded features (genres as separate flags, rare categories pooled, budget on a log scale).</p>
        <p className="mt-2"><b className="font-medium text-fg">Tuning:</b> {s.tuning.method}</p>
        <p className="mt-2"><b className="font-medium text-fg">Cross-validated macro F1:</b> {fmtPct(s.tuning.best_cv_macro_f1)} <span className="text-fg3">(fresh folds, training set)</span></p>
      </div>
    </div>
  )
}


function Selection({ m }: { m: ModelMetrics }) {
  const sel = m.selection!
  const fin = m.comparison.filter((c) => !c.name.startsWith('Dummy'))
  const best = fin.find((c) => c.name === sel.best_by_mean)
  const chosen = m.comparison.find((c) => c.name === sel.chosen)
  const scr = [...(m.screening ?? [])].sort((a, b) => b.macro_f1[0] - a.macro_f1[0])
  const mx = Math.max(...scr.map((r) => r.macro_f1[0]))
  const strat: Record<string, ScreeningRow[]> = {}
  for (const r of scr) if (!r.name.startsWith('Dummy')) (strat[r.strategy] ??= []).push(r)
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1)
  const COMPLEXITY = ['baseline', 'linear', 'single tree / neighbours', 'kernel', 'bagged trees', 'boosted trees']
  return (
    <div className="space-y-5">
      <ol className="grid gap-3 text-[0.85rem] leading-relaxed text-fg2 lg:grid-cols-4">
        <li><b className="font-medium text-fg">1 · Screen.</b> {m.evaluation.n_configurations} configurations ({m.evaluation.n_algorithms} algorithms × imbalance strategies) by repeated stratified CV on the training set.</li>
        <li><b className="font-medium text-fg">2 · Tune.</b> The five strongest algorithms by randomised search ({m.tuning?.length ?? 0} tuned).</li>
        <li><b className="font-medium text-fg">3 · Re-score.</b> {fin.length} finalists on <i>identical fresh folds</i>, so earlier optimism is not carried over.</li>
        <li><b className="font-medium text-fg">4 · Rule.</b> Not significantly worse than the best (corrected paired t &lt; 2) ⇒ pick the <b className="font-medium text-fg">simplest</b>, then the most stable. Test data plays no part.</li>
      </ol>
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-line bg-white/[0.04] p-4"><p className="eyebrow">Best by mean macro F1</p><p className="mt-2 font-display text-lg font-bold leading-snug text-fg">{sel.best_by_mean}</p><p className="mt-1 font-mono text-[0.75rem] text-fg3">{best ? fmtPct(best.cv_macro_f1 ?? 0, 1) : ''} on fresh folds</p></div>
        <div className="rounded-2xl border border-line bg-white/[0.04] p-4"><p className="eyebrow">Statistically tied with it</p><p className="mt-2 font-display text-3xl font-extrabold text-fg">{sel.indistinguishable_from_best.length} <span className="text-lg font-semibold text-fg3">of {fin.length}</span></p><p className="mt-1 text-[0.75rem] text-fg3">every finalist is within noise of the best</p></div>
        <div className="rounded-2xl border border-pink/50 bg-pink/10 p-4"><p className="eyebrow">Chosen by the rule</p><p className="mt-2 font-display text-lg font-bold leading-snug text-fg">{sel.chosen}</p><p className="mt-1 text-[0.75rem] text-fg2">simplest of the tied ({COMPLEXITY[sel.chosen_complexity_rank]}){chosen ? `, CV macro F1 ${fmtPct(chosen.cv_macro_f1 ?? 0, 1)}` : ''}</p></div>
      </div>
      <p className="text-[0.82rem] leading-relaxed text-fg2">This is <b className="font-medium text-fg">a tie broken by a simplicity rule, not a decisive win</b>. Relative to the previously deployed model (<span className="font-mono text-[0.85em]">{sel.deployed_before}</span>) the outcome is: <b className="font-medium text-fg">{sel.outcome}</b>. If the held-out ranking looks different from the cross-validation ranking, that is expected when eight test Flops and a ±0.03 fold spread cannot separate the finalists; the rule was not changed after seeing the test data.</p>
      {m.fold_audit && <p className="rounded-2xl border border-avg/40 bg-avg/10 px-4 py-3 text-[0.8rem] leading-snug text-fg2"><b className="font-medium text-fg">Flop evidence per fold:</b> {m.fold_audit.flops_per_validation_fold[0]}–{m.fold_audit.flops_per_validation_fold[1]} Flops in each validation fold (of {m.fold_audit.flops_in_training_set} in training). {m.fold_audit.conclusion}</p>}

      <Disclosure title={`Show all ${scr.length} configurations screened (cross-validation only)`}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[40rem] text-[0.8rem]">
            <caption className="sr-only">Screening results for every configuration</caption>
            <thead><tr className="border-b border-line text-right font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-2 text-left font-medium">Configuration</th><th className="font-medium">Macro F1</th><th className="font-medium">Bal. acc.</th><th className="font-medium">Accuracy</th><th className="font-medium">Flop recall</th><th className="font-medium">Fit s</th></tr></thead>
            <tbody className="divide-y divide-line tnum font-mono">
              {scr.map((r) => (
                <tr key={r.name} className="text-right">
                  <td className="py-1.5 pr-3 text-left font-sans text-fg">{r.name}<span aria-hidden className="mt-1 block h-1 w-28 overflow-hidden rounded-full bg-white/10"><span className="block h-full rounded-full bg-[image:var(--grad-brand)]" style={{ width: `${(r.macro_f1[0] / mx) * 100}%` }} /></span></td>
                  <td className="text-fg">{r.macro_f1[0].toFixed(3)} <span className="text-fg3">± {r.macro_f1[1].toFixed(3)}</span></td><td className="text-fg2">{r.balanced_accuracy[0].toFixed(3)}</td><td className="text-fg2">{fmtPct(r.accuracy[0], 1)}</td><td className="text-fg3">{fmtPct(r.flop_recall[0], 0)}</td><td className="text-fg3">{r.fit_seconds.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div><p className="eyebrow mb-2">Effect of the imbalance strategy (mean over algorithms)</p>
            <table className="w-full text-[0.78rem]"><thead><tr className="text-right font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-1 text-left font-medium">Strategy</th><th className="font-medium">Macro F1</th><th className="font-medium">Bal. acc.</th><th className="font-medium">Accuracy</th><th className="font-medium">Flop rec.</th></tr></thead>
              <tbody className="divide-y divide-line tnum font-mono">{Object.entries(strat).map(([k, rows]) => <tr key={k} className="text-right"><td className="py-1.5 text-left font-sans text-fg">{k}</td><td>{mean(rows.map((r) => r.macro_f1[0])).toFixed(3)}</td><td>{mean(rows.map((r) => r.balanced_accuracy[0])).toFixed(3)}</td><td>{fmtPct(mean(rows.map((r) => r.accuracy[0])), 1)}</td><td>{fmtPct(mean(rows.map((r) => r.flop_recall[0])), 0)}</td></tr>)}</tbody></table>
            <p className="mt-2 text-[0.72rem] leading-snug text-fg3">Balancing trades plain accuracy for balanced accuracy and Flop recall; it is not a free improvement.</p></div>
          <div><p className="eyebrow mb-2">Not run as classifiers · regression experiment</p>
            <ul className="space-y-1.5 text-[0.78rem] text-fg2">{(m.skipped ?? []).map((s, i) => <li key={i}><b className="font-medium text-fg">{s.algorithm ?? s.name}.</b> {s.reason ?? s.error}</li>)}</ul>
            {m.regression_experiment && <table className="mt-3 w-full text-[0.78rem]"><thead><tr className="text-right font-mono text-[0.72rem] uppercase tracking-wider text-fg3"><th className="py-1 text-left font-medium">Regressor → classes</th><th className="font-medium">Macro F1</th><th className="font-medium">Bal. acc.</th></tr></thead>
              <tbody className="divide-y divide-line tnum font-mono">{m.regression_experiment.rows.map((r) => <tr key={r.name} className="text-right"><td className="py-1.5 text-left font-sans text-fg">{r.name}</td><td>{r.macro_f1[0].toFixed(3)}</td><td>{r.balanced_accuracy[0].toFixed(3)}</td></tr>)}</tbody></table>}
            <p className="mt-2 text-[0.72rem] leading-snug text-fg3">{m.regression_experiment?.definition}</p></div>
        </div>
      </Disclosure>
    </div>
  )
}
