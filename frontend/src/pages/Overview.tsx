import { ArrowRight, FilterX } from 'lucide-react'
import { motion, useMotionValue, useReducedMotion, useSpring, useTransform } from 'motion/react'
import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ClassToggle } from '../components/charts'
import { Counter, InView, MagneticLink, Reel, Tilt } from '../components/fx'
import { FrameStrip, OutcomeReel } from '../components/hero'
import { Button, EmptyState, ErrorState, LoadingBlock, Note, Panel, Reveal } from '../components/ui'
import { qs } from '../lib/api'
import { CLASS_COLOR, fmtInt, fmtPct, fmtUSD, type MovieClass } from '../lib/format'
import type { FilterOptions, ModelMetrics, Overview } from '../lib/types'
import { useApi } from '../lib/useApi'

/** The chart chunk (~100 KB) is fetched on first interaction rather than at load, so it costs nothing until someone actually starts reading. */
const prefetchChartsOnInteraction = () => {
  const events = ['scroll', 'pointerdown', 'keydown', 'touchstart'] as const
  const go = () => { events.forEach((e) => window.removeEventListener(e, go)); void import('../components/OverviewCharts') }
  events.forEach((e) => window.addEventListener(e, go, { passive: true, once: true }))
  return () => events.forEach((e) => window.removeEventListener(e, go))
}
const lazyChart = <K extends 'ScoreHistogram' | 'YearTrend' | 'GenreBars' | 'BudgetGross'>(k: K) => lazy(() => import('../components/OverviewCharts').then((m) => ({ default: m[k] })))
const ScoreHistogram = lazyChart('ScoreHistogram')
const YearTrend = lazyChart('YearTrend')
const GenreBars = lazyChart('GenreBars')
const BudgetGross = lazyChart('BudgetGross')
/** Reserves the chart's height so the lazy chunk arriving never moves the page. */
const ChartFallback = ({ h }: { h: number }) => <div aria-hidden style={{ height: h }} className="skeleton w-full" />

const selectCls = 'min-h-11 w-full rounded-full border border-line-strong bg-black/25 px-4 text-sm text-fg transition-colors hover:border-white/40 focus:border-cyan sm:min-h-10'

export default function OverviewPage() {
  const [sp, setSp] = useSearchParams()
  const f = { year_min: sp.get('year_min'), year_max: sp.get('year_max'), genre: sp.get('genre'), country: sp.get('country') }
  const opts = useApi<FilterOptions>('/api/filters', { cache: true })
  const metrics = useApi<ModelMetrics>('/api/model-metrics', { cache: true })
  const ov = useApi<Overview>(`/api/overview${qs(f)}`)
  const data = ov.data
  const ready = Boolean(data)
  useEffect(() => (ready ? prefetchChartsOnInteraction() : undefined), [ready])
  const hasFilters = Object.values(f).some(Boolean)
  const set = (k: string, v: string) => { const n = new URLSearchParams(sp); if (v) n.set(k, v); else n.delete(k); setSp(n, { replace: true }) }
  const years = useMemo(() => (opts.data ? Array.from({ length: opts.data.year_max - opts.data.year_min + 1 }, (_, i) => opts.data!.year_min + i) : []), [opts.data])

  return (
    <>
      <Hero metrics={metrics.data} />

      <section id="data" aria-label="Filters" className="glass mb-8 mt-10 rounded-[1.75rem] p-4 sm:p-5">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-[1fr_1fr_1.4fr_1.4fr_auto] md:items-end">
          <label className="block"><span className="eyebrow mb-1.5 block pl-2">From year</span>
            <select className={selectCls} value={f.year_min ?? ''} onChange={(e) => set('year_min', e.target.value)}><option value="">{opts.data?.year_min ?? 'Earliest'}</option>{years.map((y) => <option key={y}>{y}</option>)}</select></label>
          <label className="block"><span className="eyebrow mb-1.5 block pl-2">To year</span>
            <select className={selectCls} value={f.year_max ?? ''} onChange={(e) => set('year_max', e.target.value)}><option value="">{opts.data?.year_max ?? 'Latest'}</option>{years.map((y) => <option key={y}>{y}</option>)}</select></label>
          <label className="block"><span className="eyebrow mb-1.5 block pl-2">Genre</span>
            <select className={selectCls} value={f.genre ?? ''} onChange={(e) => set('genre', e.target.value)}><option value="">All genres</option>{opts.data?.genres.map((g) => <option key={g.value} value={g.value}>{g.value} ({fmtInt(g.count)})</option>)}</select></label>
          <label className="block"><span className="eyebrow mb-1.5 block pl-2">Country</span>
            <select className={selectCls} value={f.country ?? ''} onChange={(e) => set('country', e.target.value)}><option value="">All countries</option>{opts.data?.countries.map((c) => <option key={c.value} value={c.value}>{c.value} ({fmtInt(c.count)})</option>)}</select></label>
          <Button className="col-span-2 md:col-span-1" disabled={!hasFilters} onClick={() => setSp({}, { replace: true })} icon={<FilterX size={15} aria-hidden />}>Reset</Button>
        </div>
        <p className="mt-3 pl-2 text-[0.78rem] text-fg3" aria-live="polite">
          {hasFilters ? <>Filtered view: <b className="tnum font-semibold text-pink">{fmtInt(data?.n_movies ?? 0)}</b> of {fmtInt(data?.total_movies ?? 4998)} films — every figure below follows this filter.</> : 'Showing the full dataset. Filters apply to everything below.'}
        </p>
      </section>

      {ov.error && !data && <ErrorState title="Couldn’t load the overview" message={ov.error.message} onRetry={ov.retry} />}
      {!data && !ov.error && <div className="space-y-4"><LoadingBlock height="h-40" /><div className="grid gap-4 xl:grid-cols-2"><LoadingBlock height="h-80" /><LoadingBlock height="h-80" /></div></div>}
      {data?.empty && <EmptyState title="No films match these filters" message="Try widening the year range or removing the genre or country filter." action={<Button onClick={() => setSp({}, { replace: true })}>Reset filters</Button>} />}
      {data && !data.empty && <Body d={data} stale={ov.loading} />}
    </>
  )
}

/* ================= HERO ================= */
function Hero({ metrics }: { metrics: ModelMetrics | null }) {
  const reduce = useReducedMotion()
  const mx = useSpring(useMotionValue(0), { stiffness: 60, damping: 20 })
  const my = useSpring(useMotionValue(0), { stiffness: 60, damping: 20 })
  const far = { x: useTransform(mx, (v) => v * -14), y: useTransform(my, (v) => v * -10) }
  const mid = { x: useTransform(mx, (v) => v * 26), y: useTransform(my, (v) => v * 18) }
  const near = { x: useTransform(mx, (v) => v * 48), y: useTransform(my, (v) => v * 34) }
  const h = metrics?.held_out
  const base = metrics ? Math.max(...Object.values(metrics.evaluation.test_class_counts)) / metrics.evaluation.test_size : null
  const chips: { text: ReactNode; cls: string; color: string; delay: string; l: string }[] = [
    { text: <><Counter value={metrics?.evaluation.data.after_title_year_duplicates ?? 0} /> films</>, cls: 'left-[2%] top-[8%]', color: '#25e8d3', delay: '0s', l: 'cleaned dataset' },
    { text: <>{metrics ? (metrics.selected_model.algorithm ?? metrics.selected_model.name) : '…'}</>, cls: 'right-[0%] top-[16%]', color: '#8b5cff', delay: '-2s', l: 'selected model' },
    { text: <><Counter value={metrics?.selected_model.n_features_in ?? 0} /> inputs</>, cls: 'left-[-2%] bottom-[22%]', color: '#ffb23e', delay: '-4s', l: 'all known pre-release' },
    ...(h ? [{ text: <><Counter value={h.accuracy * 100} decimals={1} suffix="%" /> accuracy</>, cls: 'right-[4%] bottom-[10%]', color: '#2ee6c8', delay: '-1s', l: `test · always-Hit scores ${((base ?? 0) * 100).toFixed(1)}%` },
      { text: <>Flops found {h.confusion_matrix[0][0]} of {h.per_class.Flop.support}</>, cls: 'left-[30%] bottom-[-4%]', color: '#ff4f7b', delay: '-3s', l: 'the honest number' }] : []),
  ]
  return (
    <section aria-label="Introduction" className="relative isolate -mx-[var(--gutter)] overflow-hidden px-[var(--gutter)] pb-4 pt-2 md:pb-10"
      onPointerMove={(e) => { if (reduce || e.pointerType !== 'mouse') return; const r = e.currentTarget.getBoundingClientRect(); mx.set((e.clientX - r.left) / r.width - 0.5); my.set((e.clientY - r.top) / r.height - 0.5) }}>
      {/* spotlight beam */}
      <motion.div aria-hidden style={far} className="beam pointer-events-none absolute -top-24 right-[8%] -z-10 h-[140%] w-[46%] [clip-path:polygon(44%_0,56%_0,100%_100%,0_100%)] bg-[linear-gradient(180deg,rgb(255_255_255/.22),rgb(139_92_255/.08)_55%,transparent)]" />
      <div className="grid items-center gap-10 lg:grid-cols-[1.15fr_0.85fr]">
        <div>
          <motion.p initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6 inline-flex items-center gap-2.5 rounded-full border border-line-strong bg-black/25 px-4 py-1.5 font-mono text-[0.72rem] uppercase tracking-[0.16em] text-fg2">
            <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-cyan shadow-[0_0_10px_#25e8d3]" />AI film laboratory · IMDb 5000
          </motion.p>
          <h1 className="font-display text-[2.7rem] font-extrabold leading-[0.95] tracking-tight sm:text-7xl xl:text-[5.6rem]">
            {['Will it be a'].map((w) => <motion.span key={w} className="block" initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}>{w}</motion.span>)}
            <motion.span className="grad-text block" initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.08 }}>Hit, Average</motion.span>
            <motion.span className="block" initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.16 }}>or a <span className="text-flop">Flop?</span></motion.span>
          </h1>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="mt-6 max-w-xl text-[1.02rem] leading-relaxed text-fg2">
            A machine-learning model trained on {metrics ? metrics.evaluation.data.after_title_year_duplicates.toLocaleString() : <span aria-hidden className="opacity-0">0,000</span>} films sorts movies into IMDb-rating bands using only what is known before release. Explore the data, see exactly where the model struggles, then test it on a film of your own.
          </motion.p>
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.45 }} className="mt-8 flex flex-wrap items-center gap-3">
            <MagneticLink to="/predict" className="btn-grad inline-flex min-h-14 items-center gap-3 rounded-full px-8 text-base font-bold">Open the Predictor <ArrowRight size={18} aria-hidden /></MagneticLink>
            <MagneticLink to="/performance" className="inline-flex min-h-14 items-center rounded-full border border-line-strong bg-white/[0.04] px-7 text-base font-semibold text-fg hover:bg-white/10">See where it fails</MagneticLink>
          </motion.div>
          <p className="mt-5 max-w-lg text-[0.78rem] leading-snug text-fg3">An estimate, not a guarantee. The target is an IMDb-score band, not box office. The model sees only genre, runtime, budget, country, rating, language, colour, aspect ratio and year — nothing from after release.</p>
        </div>

        {/* reel + orbiting real numbers */}
        <div className="relative mx-auto aspect-square w-full max-w-[32rem]">
          <motion.div style={mid} className="absolute inset-[13%]"><OutcomeReel /></motion.div>
          <motion.div style={far} aria-hidden className="pointer-events-none absolute inset-[26%] rounded-full bg-[radial-gradient(circle,rgb(255_79_163/.55),transparent_65%)] blur-2xl" />
          <motion.div style={near} className="pointer-events-none absolute inset-0" aria-hidden>
            {chips.map((c) => (
              <div key={c.l} className={`float absolute ${c.cls}`} style={{ animationDelay: c.delay }}>
                <div className="glass rounded-2xl px-3.5 py-2" style={{ boxShadow: `0 0 0 1px ${c.color}55 inset, 0 12px 34px -12px ${c.color}` }}>
                  <p className="font-display text-[1.02rem] font-bold leading-tight" style={{ color: c.color }}>{c.text}</p>
                  <p className="font-mono text-[0.72rem] uppercase tracking-widest text-fg3">{c.l}</p>
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </div>
      <FrameStrip />
    </section>
  )
}

/* ================= BODY ================= */
function Body({ d, stale }: { d: Overview; stale: boolean }) {
  const k = d.kpis
  const [off, setOff] = useState<MovieClass[]>([])
  const toggle = (c: MovieClass) => setOff((o) => (o.includes(c) ? o.filter((x) => x !== c) : o.length >= 2 ? o : [...o, c]))
  return (
    <div className={`space-y-6 transition-opacity duration-200 ${stale ? 'opacity-60' : ''}`} aria-busy={stale}>
      {/* oversized stat band — one surface, not six cards */}
      <Reveal>
        <div className="glass glow-edge is-on grid grid-cols-2 overflow-hidden rounded-[2rem] sm:grid-cols-3 xl:grid-cols-6 [&>*]:border-b [&>*]:border-r [&>*]:border-line [&>*]:p-5 sm:[&>*]:p-6 xl:[&>*]:border-b-0">
          <Stat label="Films" value={<Counter value={k.movies} />} hint={d.n_movies < d.total_movies ? `of ${fmtInt(d.total_movies)}` : 'duplicates removed'} />
          <Stat label="Mean IMDb" value={<Counter value={k.mean_score} decimals={2} />} hint={`median ${k.median_score.toFixed(1)}`} />
          <Stat tone="Hit" label="Hit" value={<Counter value={k.class_counts.Hit} />} hint={`${fmtPct(k.class_share.Hit)} · ≥ 6`} />
          <Stat tone="Average" label="Average" value={<Counter value={k.class_counts.Average} />} hint={`${fmtPct(k.class_share.Average)} · 3–6`} />
          <Stat tone="Flop" label="Flop" value={<Counter value={k.class_counts.Flop} />} hint={`${fmtPct(k.class_share.Flop)} · < 3`} />
          <Stat label="Median US gross" value={k.median_gross_usd == null ? '—' : fmtUSD(k.median_gross_usd)} hint={k.median_gross_usd == null ? 'no US films here' : `budget ${fmtUSD(k.median_budget_usd)}`} />
        </div>
      </Reveal>

      {/* asymmetric: tall spotlight + wide histogram */}
      <Reveal>
        <div className="grid gap-5 xl:grid-cols-12">
          <Panel className="xl:col-span-5" title="Who gets which label" subtitle="Share of films in each IMDb-score class" tone="#2ee6c8"><ClassStack d={d} /></Panel>
          <Panel className="xl:col-span-7" title="Where IMDb scores land" subtitle="Films per 0.5-point bin; colours mark the class boundaries at 3 and 6" actions={<ClassToggle off={off} toggle={toggle} />} tone="#8b5cff">
            <InView minHeight={256}><Suspense fallback={<ChartFallback h={256} />}><ScoreHistogram d={d} off={off} /></Suspense></InView>
          </Panel>
        </div>
      </Reveal>

      <Reveal>
        <Panel title="A century of releases" subtitle="Films per release year, stacked by class. The file was compiled in 2016, so the last year is partial." actions={<ClassToggle off={off} toggle={toggle} />} tone="#ff4fa3">
          <InView minHeight={288}><Suspense fallback={<ChartFallback h={288} />}><YearTrend d={d} off={off} /></Suspense></InView>
        </Panel>
      </Reveal>

      <Reveal>
        <div className="grid gap-5 xl:grid-cols-12">
          <Panel className="xl:col-span-5" title="Genres" subtitle="A film can carry several genres" actions={<ClassToggle off={off} toggle={toggle} />} tone="#ffb23e"><InView minHeight={Math.max(280, d.genres.length * 27 + 30)}><Suspense fallback={<ChartFallback h={Math.max(280, d.genres.length * 27 + 30)} />}><GenreBars d={d} off={off} /></Suspense></InView></Panel>
          <Panel className="xl:col-span-7" title="Budget vs gross" subtitle="US productions only — other countries record budget in local currency. Log axes; dashed line = gross equals budget." actions={<ClassToggle off={off} toggle={toggle} />} tone="#25e8d3">
            <InView minHeight={320}><Suspense fallback={<ChartFallback h={320} />}><BudgetGross d={d} off={off} /></Suspense></InView>
            {d.budget_gross.points.length >= 5 && <p className="mt-2 text-[0.75rem] text-fg3">{fmtInt(d.budget_gross.n_plotted)} of {fmtInt(d.budget_gross.n_eligible)} eligible US films plotted{d.budget_gross.n_plotted < d.budget_gross.n_eligible ? ' (random sample for legibility)' : ''}. Films missing a budget or gross are omitted.</p>}
          </Panel>
        </div>
      </Reveal>

      <Reveal>
        <div className="grid gap-5 xl:grid-cols-12">
          <Panel className="xl:col-span-7" title="What moves with IMDb score" subtitle="Spearman rank correlation. An association, not a cause." tone="#ff4fa3"><Correlations d={d} /></Panel>
          <Panel className="xl:col-span-5" title="Where the films come from" subtitle="Top 10 countries · bar = films, value = mean IMDb score" tone="#8b5cff"><Countries d={d} /></Panel>
        </div>
      </Reveal>

      <Reveal>
        <Panel title="Reading the data" subtitle="Generated from the numbers above, so it changes with your filters" marks tone="#ffb23e">
          <ul className="grid gap-4 lg:grid-cols-2">
            {d.insights.map((t, i) => (
              <li key={i} className="flex gap-3 text-[0.92rem] leading-relaxed text-fg2"><span aria-hidden className="mt-2 h-1 w-4 shrink-0 rounded-full bg-[image:var(--grad-brand)]" />{t}</li>
            ))}
          </ul>
        </Panel>
      </Reveal>

      <Reveal>
        <Note tone="info" title="About missing values">
          <p>Budget is missing for {fmtInt(k.usa_movies - k.budget_known)} and gross for {fmtInt(k.usa_movies - k.gross_known)} of the {fmtInt(k.usa_movies)} US films in this view. Missing values are left out of medians and charts, never counted as zero. Source: {d.source.dataset}.</p>
        </Note>
      </Reveal>

      <Reveal>
        <Tilt className="rounded-[2rem]">
          <Link to="/predict" className="glass glow-edge is-on group relative flex flex-col items-start justify-between gap-5 overflow-hidden rounded-[2rem] p-7 sm:flex-row sm:items-center sm:p-10">
            <Reel className="pointer-events-none absolute -right-16 -top-16 h-64 w-64 opacity-30" />
            <div className="relative"><p className="eyebrow">Your turn</p><p className="mt-2 font-display text-3xl font-extrabold sm:text-4xl">Put a film on the <span className="grad-text">screen</span>.</p></div>
            <span className="btn-grad relative inline-flex min-h-12 items-center gap-2 rounded-full px-7 font-bold transition-transform group-hover:translate-x-1">Run the model <ArrowRight size={18} aria-hidden /></span>
          </Link>
        </Tilt>
      </Reveal>
    </div>
  )
}

function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: string; tone?: MovieClass }) {
  const c = tone ? CLASS_COLOR[tone] : undefined
  return (
    <div className="relative">
      <p className="eyebrow flex items-center gap-2">{tone && <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: c, boxShadow: `0 0 10px ${c}` }} />}{label}</p>
      <p className={`tnum mt-3 font-display text-[2.3rem] font-extrabold leading-none sm:text-[2.7rem] ${tone ? '' : 'text-fg'}`} style={c ? { color: c, textShadow: `0 0 28px ${c}66` } : undefined}>{value}</p>
      {hint && <p className="mt-2 text-[0.76rem] text-fg3">{hint}</p>}
    </div>
  )
}

/* ---- class distribution: stacked "film frames" bar + counts ---- */
function ClassStack({ d }: { d: Overview }) {
  const rows = d.class_distribution.slice().reverse()
  return (
    <div>
      <div className="flex h-16 gap-1 overflow-hidden rounded-2xl" role="img" aria-label={rows.map((r) => `${r.class} ${fmtPct(r.share)}`).join(', ')}>
        {rows.map((r) => (
          <motion.div key={r.class} initial={{ flexGrow: 0 }} whileInView={{ flexGrow: Math.max(r.share, 0.006) }} viewport={{ once: true }} transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
            className="rounded-lg" style={{ flexBasis: 0, background: `linear-gradient(180deg, ${CLASS_COLOR[r.class]}, color-mix(in srgb, ${CLASS_COLOR[r.class]} 55%, #06061a))`, boxShadow: `0 0 28px -4px ${CLASS_COLOR[r.class]}88` }} />
        ))}
      </div>
      <dl className="mt-6 space-y-3">
        {rows.map((r) => (
          <div key={r.class} className="flex items-end justify-between border-b border-line pb-3">
            <dt className="flex items-center gap-2.5 text-[0.95rem] font-semibold" style={{ color: CLASS_COLOR[r.class] }}>{r.class}<span className="font-mono text-[0.72rem] font-normal text-fg3">{r.class === 'Hit' ? 'IMDb ≥ 6' : r.class === 'Average' ? '3 ≤ IMDb < 6' : 'IMDb < 3'}</span></dt>
            <dd className="tnum text-right"><span className="font-display text-2xl font-extrabold text-fg">{fmtInt(r.count)}</span><span className="ml-2 font-mono text-sm text-fg3">{fmtPct(r.share)}</span></dd>
          </div>
        ))}
      </dl>
      {d.kpis.class_counts.Flop < 60 && <p className="mt-4 text-[0.8rem] leading-snug text-fg3">Flops are very rare ({fmtInt(d.kpis.class_counts.Flop)} films). That imbalance is why the classifier cannot learn them — see Model performance.</p>}
    </div>
  )
}

function Countries({ d }: { d: Overview }) {
  const max = Math.max(...d.countries.map((c) => c.count))
  return (
    <ol className="space-y-3">
      {d.countries.map((c, i) => (
        <li key={c.country} className="grid grid-cols-[6rem_1fr_auto] items-center gap-3 text-sm sm:grid-cols-[7.5rem_1fr_auto]">
          <span className="truncate text-fg2">{c.country}</span>
          <span className="h-2.5 overflow-hidden rounded-full bg-white/[0.07]"><motion.span className="block h-full rounded-full bg-[image:var(--grad-cool)]" initial={{ width: 0 }} whileInView={{ width: `${(c.count / max) * 100}%` }} viewport={{ once: true }} transition={{ duration: 0.9, delay: i * 0.05, ease: [0.16, 1, 0.3, 1] }} /></span>
          <span className="tnum w-24 text-right font-mono text-[0.78rem] text-fg">{fmtInt(c.count)} <span className="text-fg3">· {c.mean_score.toFixed(2)}</span></span>
        </li>
      ))}
    </ol>
  )
}

function Correlations({ d }: { d: Overview }) {
  if (!d.correlations.length) return <EmptyState title="Too few films" message="Correlations need at least 30 films with the feature present." />
  return (
    <ul className="space-y-3">
      {d.correlations.map((c, i) => (
        <li key={c.feature} className="grid grid-cols-[8.5rem_1fr_3.2rem] items-center gap-3 text-sm sm:grid-cols-[11rem_1fr_3.5rem]">
          <span className="truncate text-fg2" title={c.usa_only ? 'US films only' : undefined}>{c.feature}{c.usa_only && <sup className="ml-0.5 text-fg3">US</sup>}</span>
          <span className="relative h-2.5 rounded-full bg-white/[0.07]" role="img" aria-label={`rho ${c.rho}`}>
            <span className="absolute inset-y-[-4px] left-1/2 w-px bg-white/25" />
            <motion.span className="absolute top-0 h-full rounded-full" style={{ background: c.rho >= 0 ? 'linear-gradient(90deg,#ff4fa3,#ffb23e)' : 'linear-gradient(270deg,#25e8d3,#8b5cff)', [c.rho >= 0 ? 'left' : 'right']: '50%' }}
              initial={{ width: 0 }} whileInView={{ width: `${Math.abs(c.rho) * 50}%` }} viewport={{ once: true }} transition={{ duration: 0.9, delay: i * 0.05, ease: [0.16, 1, 0.3, 1] }} />
          </span>
          <span className="tnum text-right font-mono text-[0.78rem] text-fg">{c.rho > 0 ? '+' : ''}{c.rho.toFixed(2)}</span>
        </li>
      ))}
      <li className="pt-1 text-[0.75rem] text-fg3">Bars run from −1 to +1 around the centre line: warm = rises with score, cool = falls. Votes and reviews exist only after release.</li>
    </ul>
  )
}
