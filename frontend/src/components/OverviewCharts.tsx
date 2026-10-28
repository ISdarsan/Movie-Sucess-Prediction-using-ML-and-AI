import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, type TooltipContentProps } from 'recharts'
import { AXIS, tickStyle, useChartMotion } from './charts'
import { ClassGradients, EmptyState, TipBox } from './ui'
import { CLASS_COLOR, CLASSES, fmtInt, fmtUSD, type MovieClass } from '../lib/format'
import type { Overview } from '../lib/types'

/* The Recharts-based Overview charts live here so the Overview route can paint without downloading the chart library. */

export function ScoreHistogram({ d, off }: { d: Overview; off: MovieClass[] }) {
  const m = useChartMotion()
  const rows = d.score_histogram.map((b) => ({ ...b, label: b.lo.toFixed(1) }))
  const Tip = ({ active, payload }: TooltipContentProps) => active && payload?.length
    ? <TipBox title={`IMDb ${payload[0].payload.lo.toFixed(1)} – ${payload[0].payload.hi.toFixed(1)}`} rows={[{ label: payload[0].payload.class, value: fmtInt(payload[0].payload.count), color: CLASS_COLOR[payload[0].payload.class as MovieClass] }]} /> : null
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barCategoryGap={3}>
          <ClassGradients />
          <CartesianGrid vertical={false} />
          <XAxis dataKey="label" {...AXIS} tick={tickStyle} interval={1} />
          <YAxis {...AXIS} tick={tickStyle} axisLine={false} width={44} />
          <Tooltip content={Tip} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
          <Bar dataKey="count" radius={[6, 6, 0, 0]} {...m}>{rows.map((r) => <Cell key={r.lo} fill={`url(#cg-${r.class})`} fillOpacity={off.includes(r.class) ? 0.07 : 1} stroke={CLASS_COLOR[r.class]} strokeOpacity={off.includes(r.class) ? 0.1 : 0.6} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function YearTrend({ d, off }: { d: Overview; off: MovieClass[] }) {
  const m = useChartMotion()
  const Tip = ({ active, payload, label }: TooltipContentProps) => active && payload?.length
    ? <TipBox title={String(label)} rows={[...[...CLASSES].reverse().map((c): { label: string; value: string; color?: string } => ({ label: c, value: fmtInt(payload[0].payload[c]), color: CLASS_COLOR[c] })), { label: 'Total', value: fmtInt(payload[0].payload.count) }]} /> : null
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <AreaChart data={d.year_trend} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
          <ClassGradients />
          <CartesianGrid vertical={false} />
          <XAxis dataKey="year" {...AXIS} tick={tickStyle} minTickGap={28} />
          <YAxis {...AXIS} tick={tickStyle} axisLine={false} width={44} />
          <Tooltip content={Tip} />
          {CLASSES.filter((c) => !off.includes(c)).map((c) => <Area key={c} type="monotone" dataKey={c} stackId="1" stroke={CLASS_COLOR[c]} fill={`url(#cg-${c})`} strokeWidth={1.6} {...m} />)}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export function GenreBars({ d, off }: { d: Overview; off: MovieClass[] }) {
  const m = useChartMotion()
  const Tip = ({ active, payload }: TooltipContentProps) => active && payload?.length ? (
    <TipBox title={payload[0].payload.genre} rows={[
      { label: 'Films', value: fmtInt(payload[0].payload.count) }, { label: 'Mean score', value: payload[0].payload.mean_score.toFixed(2) },
      ...[...CLASSES].reverse().map((c) => ({ label: c, value: fmtInt(payload[0].payload[c]), color: CLASS_COLOR[c] })),
    ]} />) : null
  return (
    <div style={{ height: Math.max(280, d.genres.length * 27 + 30) }} className="w-full">
      <ResponsiveContainer>
        <BarChart data={d.genres} layout="vertical" margin={{ top: 0, right: 12, left: 0, bottom: 0 }} barCategoryGap={5}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" {...AXIS} tick={tickStyle} />
          <YAxis type="category" dataKey="genre" {...AXIS} tick={{ ...tickStyle, fill: 'var(--color-fg2)' }} axisLine={false} width={78} interval={0} />
          <Tooltip content={Tip} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
          {CLASSES.filter((c) => !off.includes(c)).map((c, i, a) => <Bar key={c} dataKey={c} stackId="g" fill={CLASS_COLOR[c]} radius={i === a.length - 1 ? [0, 6, 6, 0] : 0} {...m} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function BudgetGross({ d, off }: { d: Overview; off: MovieClass[] }) {
  const m = useChartMotion()
  const g = d.budget_gross
  if (g.points.length < 5) return <EmptyState title="Not enough US films" message="Fewer than five US films in this view report both budget and gross." />
  const lo = 4e4, hi = 1e9
  const Tip = ({ active, payload }: TooltipContentProps) => {
    if (!active || !payload?.length) return null
    const p = payload[0].payload as typeof g.points[number]
    return <TipBox title={`${p.title}${p.year ? ` (${p.year})` : ''}`} rows={[{ label: 'Budget', value: fmtUSD(p.budget) }, { label: 'Gross', value: fmtUSD(p.gross) }, { label: `IMDb · ${p.class}`, value: p.score.toFixed(1), color: CLASS_COLOR[p.class] }]} />
  }
  return (
    <>
      <div className="h-80 w-full">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 28, left: -4, bottom: 4 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="budget" name="Budget" scale="log" domain={[lo, hi]} allowDataOverflow {...AXIS} tick={tickStyle} tickFormatter={fmtUSD} ticks={[1e5, 1e6, 1e7, 1e8, 1e9]} />
            <YAxis type="number" dataKey="gross" name="Gross" scale="log" domain={[lo, 2e9]} allowDataOverflow {...AXIS} tick={tickStyle} axisLine={false} tickFormatter={fmtUSD} width={56} ticks={[1e5, 1e6, 1e7, 1e8, 1e9]} />
            <ZAxis range={[30, 30]} />
            <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#fff" strokeOpacity={0.4} strokeDasharray="4 4" />
            <Tooltip content={Tip} cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,.3)' }} />
            {CLASSES.filter((c) => !off.includes(c)).map((c) => <Scatter key={c} data={g.points.filter((p) => p.class === c)} fill={CLASS_COLOR[c]} fillOpacity={0.7} {...m} />)}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}
