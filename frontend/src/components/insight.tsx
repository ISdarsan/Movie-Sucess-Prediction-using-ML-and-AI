/* Two signature visualisations for the Predictor, both fed by real data:
   CohortLens — how the actual dataset narrows as you describe a film (descriptive counts, no model).
   WhatIf     — the saved model re-scored while one input sweeps its observed range (a response curve, not a cause). */
import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from 'recharts'
import { ApiError, api, qs } from '../lib/api'
import { CLASS_COLOR, CLASSES, fmtInt, fmtPct, type MovieClass } from '../lib/format'
import { useApi } from '../lib/useApi'
import { AXIS, tickStyle, useChartMotion } from './charts'
import { Reel } from './fx'
import { TipBox } from './ui'

const EASE = [0.16, 1, 0.3, 1] as const
type Counts = Record<MovieClass, number>
interface Stage { label: string; n: number; counts: Counts; mean_score: number | null }
interface Cohort { stages: Stage[]; final: Stage; thin: boolean; empty: boolean; note: string }

function MixBar({ counts, n, height = 'h-3' }: { counts: Counts; n: number; height?: string }) {
  return (
    <div className={`flex w-full gap-px overflow-hidden rounded-full bg-white/[0.06] ${height}`} role="img" aria-label={CLASSES.slice().reverse().map((c) => `${c} ${fmtPct(n ? counts[c] / n : 0, 0)}`).join(', ')}>
      {CLASSES.slice().reverse().map((c) => (
        <motion.span key={c} initial={false} animate={{ flexGrow: n ? Math.max(counts[c] / n, counts[c] ? 0.01 : 0) : 0 }} transition={{ duration: 0.6, ease: EASE }} style={{ flexBasis: 0, background: CLASS_COLOR[c], boxShadow: `0 0 10px ${CLASS_COLOR[c]}66` }} />
      ))}
    </div>
  )
}

export function CohortLens({ genres, country, contentRating, year, duration, compact }: { genres: string[]; country: string; contentRating: string; year: number | null; duration: number | null; compact?: boolean }) {
  // debounce so typing a runtime doesn't fire a request per keystroke
  const key = qs({ genres: genres.join(','), country, content_rating: contentRating, year, duration })
  const [q, setQ] = useState(key)
  useEffect(() => { const t = setTimeout(() => setQ(key), 280); return () => clearTimeout(t) }, [key])
  const { data, error, loading } = useApi<Cohort>(`/api/cohort${q}`)
  const picked = genres.length > 0 || !!country || !!contentRating || year !== null || duration !== null
  const total = data?.stages[0].n ?? 1

  return (
    <div className={compact ? 'mt-6 border-t border-line pt-4' : 'relative px-5 py-6 sm:px-7'} aria-live="polite">
      {!compact && <div aria-hidden className="beam pointer-events-none absolute left-1/2 top-0 h-full w-[140%] -translate-x-1/2 [clip-path:polygon(47%_0,53%_0,100%_100%,0_100%)] bg-[linear-gradient(180deg,rgb(255_255_255/.1),rgb(139_92_255/.03)_70%,transparent)]" />}
      <div className="relative">
        <p className="eyebrow flex items-center gap-2"><Reel spin={false} className="h-4 w-4" />Cohort lens · real films like yours</p>
        {error && <p role="alert" className="mt-3 text-sm text-danger">{error.message}</p>}
        {!data && !error && <div className="skeleton mt-4 h-40" />}
        {data && (
          <div className={loading ? 'opacity-70 transition-opacity' : 'transition-opacity'}>
            <div className="mt-3 flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
              <p className="font-display text-[2.6rem] font-extrabold leading-none text-fg sm:text-[3.2rem]">
                {data.empty ? '0' : fmtInt(data.final.n)}<span className="ml-2 font-sans text-[0.85rem] font-medium tracking-normal text-fg2">{data.final.n === 1 ? 'film' : 'films'} match</span>
              </p>
              {data.final.mean_score != null && <p className="font-mono text-[0.75rem] text-fg2">mean IMDb <b className="text-fg">{data.final.mean_score.toFixed(2)}</b></p>}
            </div>
            <div className="mt-4"><MixBar counts={data.final.counts} n={data.final.n} height="h-4" /></div>
            <dl className="mt-3 grid grid-cols-3 gap-2">
              {CLASSES.slice().reverse().map((c) => (
                <div key={c}>
                  <dt className="flex items-center gap-1.5 text-[0.72rem]" style={{ color: CLASS_COLOR[c] }}><span aria-hidden className="h-2 w-2 rounded-full" style={{ background: CLASS_COLOR[c] }} />{c}</dt>
                  <dd className="tnum font-display text-[1.5rem] font-bold leading-tight text-fg">{data.final.n ? fmtPct(data.final.counts[c] / data.final.n, 0) : '—'}<span className="ml-1.5 font-mono text-[0.72rem] font-normal text-fg3">{fmtInt(data.final.counts[c])}</span></dd>
                </div>
              ))}
            </dl>
            {data.empty && <p className="mt-3 rounded-xl border border-flop/40 bg-flop/10 px-3 py-2 text-[0.8rem] text-fg2">No film in the dataset has this exact combination — loosen one characteristic.</p>}
            {data.thin && <p className="mt-3 rounded-xl border border-avg/40 bg-avg/10 px-3 py-2 text-[0.8rem] text-fg2">Only {data.final.n} films match, so these percentages are very rough.</p>}

            {!compact && (
              <ol className="mt-5 space-y-2.5" aria-label="How the dataset narrows">
                {data.stages.map((s, i) => (
                  <li key={`${i}-${s.label}`} className="grid grid-cols-[minmax(0,8.5rem)_1fr_3.2rem] items-center gap-3 text-[0.78rem]">
                    <span className="truncate text-fg2" title={s.label}>{i === 0 ? 'All films' : <><span className="text-fg3">+ </span>{s.label}</>}</span>
                    <motion.span className="block min-w-3" initial={false} animate={{ width: `${Math.max((s.n / total) * 100, s.n ? 4 : 0)}%` }} transition={{ duration: 0.6, ease: EASE }}><MixBar counts={s.counts} n={s.n} height="h-2" /></motion.span>
                    <span className="tnum text-right font-mono text-fg">{fmtInt(s.n)}</span>
                  </li>
                ))}
              </ol>
            )}
            <p className="mt-5 text-[0.72rem] leading-snug text-fg3">
              {picked ? data.note : 'Pick genres, a country, a rating, a year or a runtime and this lens narrows to the real films that share them. '}
              {!compact && !picked && 'It describes the dataset only — the model has not run yet.'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- what-if */
const FEATURES: { value: string; label: string }[] = [
  { value: 'duration', label: 'Runtime' }, { value: 'title_year', label: 'Release year' }, { value: 'budget', label: 'Budget' }, { value: 'aspect_ratio', label: 'Aspect ratio' },
]
interface Sweep {
  feature: string; label: string; log: boolean; values: number[]; probabilities: Record<MovieClass, number[]>
  current: { value: number; imputed: boolean; probabilities: Record<MovieClass, number> }; note: string
}
const compactNum = (v: number) => (v < 100 && v % 1 !== 0 ? v.toFixed(2) : null) ??  (v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(v >= 1e4 ? 0 : 1)}K` : String(Math.round(v)))
const top = (p: Record<MovieClass, number>) => CLASSES.reduce((a, c) => (p[c] > p[a] ? c : a), 'Flop' as MovieClass)

export function WhatIf({ body }: { body: Record<string, unknown> }) {
  const [feature, setFeature] = useState('duration')
  const [data, setData] = useState<Sweep | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(true)
  const m = useChartMotion()

  useEffect(() => {
    let live = true
    setBusy(true); setErr(null)
    api.post<Sweep>('/api/sensitivity', { inputs: body, feature }).then(
      (d) => { if (live) { setData(d); setBusy(false) } },
      (e: ApiError) => { if (live) { setErr(e.message); setBusy(false) } },
    )
    return () => { live = false }
  }, [body, feature])

  const rows = data ? data.values.map((x, i) => ({ x, Flop: data.probabilities.Flop[i], Average: data.probabilities.Average[i], Hit: data.probabilities.Hit[i] })) : []
  const first = data ? top({ Flop: rows[0].Flop, Average: rows[0].Average, Hit: rows[0].Hit }) : null
  const last = data ? top({ Flop: rows.at(-1)!.Flop, Average: rows.at(-1)!.Average, Hit: rows.at(-1)!.Hit }) : null
  const Tip = ({ active, payload }: TooltipContentProps) => {
    if (!active || !payload?.length || !data) return null
    const p = payload[0].payload as (typeof rows)[number]
    return <TipBox title={`${data.label}: ${compactNum(p.x)}`} rows={[...CLASSES].reverse().map((c) => ({ label: c, value: fmtPct(p[c], 0), color: CLASS_COLOR[c] }))} />
  }

  return (
    <div className="mt-6 rounded-2xl border border-line bg-black/20 p-4">
      <h3 className="font-display text-[0.95rem] font-bold">What if one input changed?</h3>
      <p className="mt-0.5 text-[0.72rem] text-fg3">The saved model, re-run with only this input swept across its range in the dataset. Everything else stays as you submitted it.</p>
      <div role="radiogroup" aria-label="Input to sweep" className="mt-3 flex flex-wrap gap-1.5">
        {FEATURES.map((f) => {
          const on = f.value === feature
          return (
            <button key={f.value} type="button" role="radio" aria-checked={on} onClick={() => setFeature(f.value)}
              className={`min-h-9 rounded-full border px-3.5 text-[0.78rem] font-semibold transition-all duration-200 active:scale-95 ${on ? 'border-transparent bg-cyan text-[#04120f] shadow-[0_6px_20px_-8px_rgb(46_230_200/.9)]' : 'border-line-strong text-fg2 hover:-translate-y-0.5 hover:border-white/50 hover:text-fg'}`}>{f.label}</button>
          )
        })}
      </div>
      {err && <p role="alert" className="mt-3 text-sm text-danger">{err}</p>}
      <div className="mt-3 h-52 w-full" aria-busy={busy}>
        {busy && !data ? <div className="skeleton h-full" /> : data && (
          <div className={busy ? 'h-full opacity-60 transition-opacity' : 'h-full transition-opacity'}>
            <ResponsiveContainer>
              <AreaChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid vertical={false} />
                <XAxis type="number" dataKey="x" scale={data.log ? 'log' : 'linear'} domain={['dataMin', 'dataMax']} {...AXIS} tick={tickStyle} tickFormatter={compactNum} tickCount={5} />
                <YAxis {...AXIS} tick={tickStyle} axisLine={false} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} width={46} />
                <Tooltip content={Tip} cursor={{ stroke: 'rgba(255,255,255,.4)' }} />
                <ReferenceLine x={data.current.value} stroke="#fff" strokeDasharray="4 3" label={{ value: data.current.imputed ? 'median (blank)' : 'yours', position: 'insideTopRight', fill: '#fff', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
                {CLASSES.map((c) => <Area key={c} type="monotone" dataKey={c} stroke={CLASS_COLOR[c]} strokeWidth={2} fill={CLASS_COLOR[c]} fillOpacity={0.1} {...m} />)}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      {data && (
        <p className="mt-2 text-[0.8rem] leading-snug text-fg2" aria-live="polite">
          {first === last
            ? <>Across the observed range of <b className="text-fg">{data.label.toLowerCase()}</b>, the model’s top class stays <b style={{ color: CLASS_COLOR[first!] }}>{first}</b>; only its confidence moves.</>
            : <>As <b className="text-fg">{data.label.toLowerCase()}</b> rises across its observed range, the model’s top class moves from <b style={{ color: CLASS_COLOR[first!] }}>{first}</b> to <b style={{ color: CLASS_COLOR[last!] }}>{last}</b>.</>}
          {' '}<span className="text-fg3">{data.note}</span>
        </p>
      )}
    </div>
  )
}
