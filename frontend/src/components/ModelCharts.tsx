import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from 'recharts'
import { AXIS, tickStyle, useChartMotion } from './charts'
import { TipBox } from './ui'
import { CLASSES, fmtInt, fmtPct } from '../lib/format'
import type { ModelResult } from '../lib/types'

/* Recharts-based charts for the Performance and Explain pages, split out so those routes render without the chart library. */

export function PerClassChart({ r }: { r: ModelResult }) {
  const m = useChartMotion()
  const rows = CLASSES.slice().reverse().map((c) => ({ class: c, Precision: r.per_class[c].precision, Recall: r.per_class[c].recall, F1: r.per_class[c]['f1-score'], support: r.per_class[c].support }))
  const Tip = ({ active, payload, label }: TooltipContentProps) => active && payload?.length
    ? <TipBox title={`${label} · ${fmtInt(payload[0].payload.support)} test films`} rows={payload.map((p) => ({ label: String(p.name), value: fmtPct(Number(p.value), 0), color: String(p.color) }))} /> : null
  return (
    <div className="h-60 w-full">
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={3}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="class" {...AXIS} tick={{ ...tickStyle, fill: 'var(--color-fg2)' }} />
            <YAxis {...AXIS} tick={tickStyle} axisLine={false} domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} width={48} />
            <Tooltip content={Tip} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            <Legend iconType="square" wrapperStyle={{ fontSize: 12, color: 'var(--color-fg2)' }} />
            <Bar dataKey="Precision" fill="#8b5cff" radius={[6, 6, 0, 0]} {...m} />
            <Bar dataKey="Recall" fill="#25e8d3" radius={[6, 6, 0, 0]} {...m} />
            <Bar dataKey="F1" fill="#ff4fa3" radius={[6, 6, 0, 0]} {...m} />
          </BarChart>
        </ResponsiveContainer>
      </div>
  )
}

export function ImportanceChart({ data, label, color }: { data: { name: string; v: number }[]; label: string; color: string }) {
  const m = useChartMotion()
  const Tip = ({ active, payload }: TooltipContentProps) => active && payload?.length
    ? <TipBox title={payload[0].payload.name} rows={[{ label, value: Number(payload[0].payload.v).toFixed(4), color }]} /> : null
  return (
    <div style={{ height: Math.max(240, data.length * 25 + 30) }} className="w-full">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 14, left: 0, bottom: 0 }} barCategoryGap={4}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" {...AXIS} tick={tickStyle} tickFormatter={(v) => Number(v).toFixed(2)} />
          <YAxis type="category" dataKey="name" {...AXIS} axisLine={false} interval={0} width={168} tick={{ ...tickStyle, fill: 'var(--color-fg2)' }} tickFormatter={(s: string) => (s.length > 27 ? `${s.slice(0, 26)}…` : s)} />
          <Tooltip content={Tip} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <Bar dataKey="v" radius={[0, 2, 2, 0]} {...m}>{data.map((d) => <Cell key={d.name} fill={color} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
