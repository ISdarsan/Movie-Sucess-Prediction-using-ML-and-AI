import { motion } from 'motion/react'
import { lazy, Suspense, useMemo, useState } from 'react'
import { InView } from '../components/fx'
import { ErrorState, LoadingBlock, Note, PageHeader, Panel, Reveal, Segmented } from '../components/ui'
import { CLASS_COLOR, CLASSES, fmtInt, type MovieClass } from '../lib/format'
import type { Explain, ShapBlock } from '../lib/types'
import { useApi } from '../lib/useApi'

export default function ExplainPage() {
  const { data, error, loading, retry } = useApi<Explain>('/api/explain', { cache: true })
  return (
    <>
      <PageHeader eyebrow="Explainable AI" title="What the model looks at" highlight="looks at">
        Two views of the same model: how much each input contributes overall (feature importance), and how each input pushes individual predictions toward or away from a class (SHAP). Both describe the model’s behaviour — not the causes of a film’s success.
      </PageHeader>
      {error && !data && <ErrorState title="Explainability results unavailable" message={error.message} onRetry={retry} />}
      {loading && !data && <LoadingBlock height="h-[32rem]" label="Loading explainability results" />}
      {data && <Body e={data} />}
    </>
  )
}

const pretty = (s: string) => s.replace(/_/g, ' ').replace(' = ', ' = ')

function Body({ e }: { e: Explain }) {
  const [cls, setCls] = useState<MovieClass>('Hit')
  const [grouped, setGrouped] = useState<'feature' | 'column'>('column')
  const block = e.shap[cls]
  const insights = useMemo(() => deriveInsights(block, cls), [block, cls])

  return (
    <div className="space-y-5">
      <Reveal>
        <Panel marks title="How this was computed" bodyClass="pt-2">
          <dl className="grid gap-x-8 gap-y-3 text-[0.85rem] sm:grid-cols-2 xl:grid-cols-4">
            <div><dt className="eyebrow">Explainer</dt><dd className="mt-1 text-fg">TreeExplainer · shap {e.method.shap_version}</dd></div>
            <div><dt className="eyebrow">Sample</dt><dd className="mt-1 text-fg">{fmtInt(e.method.sample_size)} films, drawn from the held-out test split</dd></div>
            <div><dt className="eyebrow">Features</dt><dd className="mt-1 text-fg">{fmtInt(e.method.n_transformed_features)} after one-hot encoding</dd></div>
            <div><dt className="eyebrow">Output</dt><dd className="mt-1 text-fg">{e.method.values_shape.join(' × ')} (rows × features × classes)</dd></div>
          </dl>
          <p className="mt-4 text-[0.8rem] text-fg3">Computed once offline and stored (generated {new Date(e.generated_at).toLocaleDateString()}), so this page loads instantly instead of recomputing SHAP per visit. Because only 200 films were sampled, rankings of closely scored features can change with a different sample.</p>
        </Panel>
      </Reveal>

      <Reveal>
        <Panel title="Model feature importance" subtitle={e.feature_importance.kind}
          actions={<Segmented label="Grouping" value={grouped} onChange={setGrouped} options={[{ value: 'column', label: 'By input column' }, { value: 'feature', label: 'By encoded feature' }]} />}>
          <ImportanceBars
            rows={grouped === 'column' ? e.feature_importance.grouped_by_column.slice(0, 14).map((r) => ({ name: r.column, v: r.importance })) : e.feature_importance.top.map((r) => ({ name: r.feature, v: r.importance }))}
            label="Importance" color="var(--color-fg2)" />
          <p className="mt-2 text-[0.78rem] leading-snug text-fg3">“By input column” adds up every one-hot dummy that came from the same original column (e.g. all <span className="font-mono">country = …</span> columns), so a high-cardinality column isn’t split into many small bars. A model’s own importance score tends to favour numeric inputs and inputs with many distinct values.</p>
        </Panel>
      </Reveal>

      <Reveal>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-display text-2xl text-fg">SHAP for class <span style={{ color: CLASS_COLOR[cls] }}>{cls}</span></h2>
          <Segmented label="Class to explain" value={cls} onChange={setCls} options={[...CLASSES].reverse().map((c) => ({ value: c, label: c, color: CLASS_COLOR[c] }))} />
        </div>
      </Reveal>

      <Reveal>
        <Panel title="Feature lanes" tone={CLASS_COLOR[cls]}
          subtitle={<>One lane per feature. The bar is its average push on the <b className="font-medium text-fg">{cls}</b> score (mean |SHAP|); the dots are {e.method.sample_size} films — right of the centre line pushes <i>toward</i> {cls}, left pushes <i>away</i>. Select a lane to read it.</>}>
          <InView minHeight={420}><Lanes block={block} cls={cls} /></InView>
        </Panel>
      </Reveal>

      <Reveal>
        <div className="grid gap-4 xl:grid-cols-12">
          <Panel className="xl:col-span-7" title={`Reading the ${cls} plot`} subtitle="Generated from the SHAP values on screen. Association only." marks>
            <ul className="space-y-3">
              {insights.map((t, i) => <li key={i} className="flex gap-3 text-[0.9rem] leading-relaxed text-fg2"><span aria-hidden className="mt-2 h-1 w-3 shrink-0 bg-accent" />{t}</li>)}
            </ul>
          </Panel>
          <Panel className="xl:col-span-5" title="How to read SHAP values">
            <dl className="space-y-3 text-[0.85rem] leading-relaxed text-fg2">
              <div><dt className="font-medium text-fg">Positive SHAP value</dt><dd>This input moved the model’s score for the class up, for that film.</dd></div>
              <div><dt className="font-medium text-fg">Negative SHAP value</dt><dd>This input moved the score down for that film.</dd></div>
              <div><dt className="font-medium text-fg">Dot colour</dt><dd>The feature’s value for that film, from low (periwinkle) to high (hot pink). One-hot columns are only 0 or 1.</dd></div>
              <div><dt className="font-medium text-fg">Magnitude</dt><dd>Values are in the units of the model’s class score (its raw output for that class), so they are comparable across features but not across models.</dd></div>
            </dl>
          </Panel>
        </div>
      </Reveal>

      <Reveal>
        <Note title="Importance is not causation">
          <p>These plots show which inputs the model relies on, not what makes a film good. Genre, runtime, budget and release year mostly describe the kind of film and the era it came from, and a feature such as <span className="font-mono">country</span> can stand in for other things (language, budget scale, genre mix) it happens to correlate with. This model sees only pre-release information, so none of these are audience reactions. Do not read a bar as a lever a studio could pull.</p>
          <p><b className="font-medium text-fg">Per-film explanations are not offered in this build.</b> Only the global, precomputed SHAP results above are shown; the app never invents an explanation for an individual prediction.</p>
        </Note>
      </Reveal>
    </div>
  )
}

const ImportanceChart = lazy(() => import('../components/ModelCharts').then((m) => ({ default: m.ImportanceChart })))

function ImportanceBars({ rows, label, color }: { rows: { name: string; v: number }[]; label: string; color: string }) {
  const data = rows.map((r) => ({ ...r, name: pretty(r.name) }))
  return (
    <Suspense fallback={<div aria-hidden style={{ height: Math.max(240, data.length * 25 + 30) }} className="skeleton w-full" />}>
      <ImportanceChart data={data} label={label} color={color} />
    </Suspense>
  )
}

/* ---------- feature lanes: bar + beeswarm in one interactive row ---------- */
const LOW = [120, 132, 190], HIGH = [255, 79, 163]
const ramp = (t: number) => `rgb(${LOW.map((l, i) => Math.round(l + (HIGH[i] - l) * t)).join(',')})`

function corr(points: [number, number][]) {
  const n = points.length
  const mx = points.reduce((s, p) => s + p[1], 0) / n, my = points.reduce((s, p) => s + p[0], 0) / n
  let sxy = 0, sxx = 0, syy = 0
  for (const [y, x] of points) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2 }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0
}

function Lanes({ block, cls }: { block: ShapBlock; cls: MovieClass }) {
  const [sel, setSel] = useState<string | null>(null)
  const rows = block.beeswarm.slice(0, 14)
  const imp = useMemo(() => new Map(block.bar.map((b) => [b.feature, b.mean_abs_shap])), [block])
  const topImp = Math.max(...rows.map((r) => imp.get(r.feature) ?? 0), 1e-6)
  const max = useMemo(() => Math.max(0.01, ...rows.flatMap((r) => r.points.map((p) => Math.abs(p[0])))) * 1.06, [rows])
  const color = CLASS_COLOR[cls]
  const W = 560, H = 36, pad = 8
  const x = (v: number) => pad + ((v + max) / (2 * max)) * (W - 2 * pad)
  return (
    <div>
      <ul role="list" className="space-y-1.5">
        {rows.map((r, idx) => {
          const on = sel === r.feature
          const c = corr(r.points)
          const bins = new Map<number, number>()
          const dots = r.points.slice().sort((a, b) => a[0] - b[0]).map(([v, t]) => {
            const b = Math.round(x(v) / 4.2); const n = bins.get(b) ?? 0; bins.set(b, n + 1)
            const off = Math.ceil(n / 2) * 3.6 * (n % 2 ? 1 : -1)
            return { cx: x(v), cy: H / 2 + Math.max(-H / 2 + 3, Math.min(H / 2 - 3, off)), t }
          })
          const w = ((imp.get(r.feature) ?? 0) / topImp) * 100
          return (
            <li key={r.feature}>
              <button type="button" onClick={() => setSel(on ? null : r.feature)} aria-expanded={on}
                className={`group grid w-full items-center gap-x-4 gap-y-1 rounded-2xl px-3 py-2 text-left transition-colors duration-200 sm:grid-cols-[11rem_9rem_1fr] ${on ? 'bg-white/[0.08]' : 'hover:bg-white/[0.05]'}`}>
                <span className="truncate font-mono text-[0.76rem] text-fg" title={pretty(r.feature)}><span className="mr-2 text-fg3">{String(idx + 1).padStart(2, '0')}</span>{pretty(r.feature)}</span>
                <span className="flex items-center gap-2">
                  <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-white/[0.07]"><motion.span className="block h-full rounded-full" style={{ background: `linear-gradient(90deg, color-mix(in srgb, ${color} 45%, #06061a), ${color})`, boxShadow: `0 0 14px ${color}88` }} initial={{ width: 0 }} animate={{ width: `${w}%` }} transition={{ duration: 0.9, delay: idx * 0.04, ease: [0.16, 1, 0.3, 1] }} /></span>
                  <span className="tnum w-12 text-right font-mono text-[0.72rem] text-fg3">{(imp.get(r.feature) ?? 0).toFixed(3)}</span>
                </span>
                <svg viewBox={`0 0 ${W} ${H}`} className="block w-full sm:col-auto" role="img" aria-label={`${pretty(r.feature)}: SHAP values from ${(-max).toFixed(2)} to ${max.toFixed(2)}`}>
                  <line x1={x(0)} x2={x(0)} y1={0} y2={H} stroke="rgb(255 255 255 / .25)" />
                  {dots.map((d, i) => <circle key={i} cx={d.cx} cy={d.cy} r={2.6} fill={ramp(d.t)} fillOpacity={0.9} />)}
                </svg>
              </button>
              {on && (
                <motion.p initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="mx-3 mb-2 overflow-hidden rounded-xl bg-black/25 px-4 py-3 text-[0.85rem] leading-relaxed text-fg2">
                  <b className="font-semibold text-fg">{pretty(r.feature)}</b> has an average push of {(imp.get(r.feature) ?? 0).toFixed(4)} on the {cls} score.{' '}
                  {Math.abs(c) < 0.25
                    ? 'Its feature value and its push are only weakly related in this sample, so no single direction summarises it.'
                    : `Higher values tend to push ${c > 0 ? 'toward' : 'away from'} ${cls} (correlation between feature value and SHAP value ${c > 0 ? '+' : ''}${c.toFixed(2)}).`}{' '}
                  <span className="text-fg3">This describes the model’s behaviour on 200 test films, not a cause.</span>
                </motion.p>
              )}
            </li>
          )
        })}
      </ul>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3 text-[0.72rem] text-fg3">
        <span className="font-mono">← pushes away from {cls} · SHAP value · pushes toward {cls} →</span>
        <span className="flex items-center gap-2">Feature value <span>low</span><span aria-hidden className="h-2 w-24 rounded-full" style={{ background: `linear-gradient(90deg, ${ramp(0)}, ${ramp(1)})` }} /><span>high</span></span>
      </div>
    </div>
  )
}

/** Plain-English reading of the plot, computed from the actual SHAP points. */
function deriveInsights(block: ShapBlock, cls: MovieClass): string[] {
  const out: string[] = []
  const top = block.bar.slice(0, 3).map((b) => pretty(b.feature))
  out.push(`The three features with the largest average effect on the ${cls} score are ${top[0]}, ${top[1]} and ${top[2]}.`)
  const total = block.grouped_by_column.reduce((s, c) => s + c.mean_abs_shap, 0)
  const g = block.grouped_by_column[0]
  if (g && total > 0) out.push(`Grouping the encoded features by their original column, ${pretty(g.column)} accounts for ${((g.mean_abs_shap / total) * 100).toFixed(0)}% of the total average effect.`)
  for (const r of block.beeswarm.slice(0, 5)) {
    const n = r.points.length
    const mx = r.points.reduce((s, p) => s + p[1], 0) / n, my = r.points.reduce((s, p) => s + p[0], 0) / n
    let sxy = 0, sxx = 0, syy = 0
    for (const [y, xv] of r.points) { sxy += (xv - mx) * (y - my); sxx += (xv - mx) ** 2; syy += (y - my) ** 2 }
    const c = sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0
    if (Math.abs(c) < 0.25) continue
    out.push(`${pretty(r.feature)}: higher values tend to ${c > 0 ? 'push toward' : 'push away from'} ${cls} (correlation between feature value and SHAP value ${c > 0 ? '+' : ''}${c.toFixed(2)} in this sample).`)
  }
  if (cls === 'Flop') out.push('Flop SHAP values are tiny compared with Hit and Average, consistent with the model having almost nothing to learn from for this class (Flops are under 1% of films).')
  return out
}
