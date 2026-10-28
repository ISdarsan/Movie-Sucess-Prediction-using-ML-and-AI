import { useReducedMotion } from 'motion/react'
import { CLASS_COLOR, CLASS_RANGE, CLASSES, type MovieClass } from '../lib/format'

export const AXIS = { stroke: 'var(--color-line-strong)', tickLine: false } as const
export const tickStyle = { fill: 'var(--color-fg3)', fontSize: 11, fontFamily: 'var(--font-mono)' }

/** Recharts animation flag that respects prefers-reduced-motion. */
export function useChartMotion() {
  const reduce = useReducedMotion()
  return { isAnimationActive: !reduce, animationDuration: 600, animationEasing: 'ease-out' as const }
}

export function ClassLegend({ ranges }: { ranges?: boolean }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[0.75rem] text-fg2" aria-label="Class legend">
      {CLASSES.slice().reverse().map((c) => (
        <li key={c} className="flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-2.5 rounded-xs" style={{ background: CLASS_COLOR[c] }} />
          {c}{ranges && <span className="font-mono text-[0.72rem] text-fg3">{CLASS_RANGE[c]}</span>}
        </li>
      ))}
    </ul>
  )
}

/** Legend that actually does something: select a class to hide it from every chart that shares this state. */
export function ClassToggle({ off, toggle }: { off: MovieClass[]; toggle: (c: MovieClass) => void }) {
  return (
    <div role="group" aria-label="Show or hide classes in the charts" className="flex flex-wrap items-center gap-1.5">
      {CLASSES.slice().reverse().map((c) => {
        const hidden = off.includes(c)
        return (
          <button key={c} type="button" aria-pressed={!hidden} onClick={() => toggle(c)}
            className="flex min-h-9 items-center gap-1.5 rounded-full border px-3 text-[0.75rem] font-medium transition-all duration-200 active:scale-95"
            style={{ borderColor: hidden ? 'rgb(255 255 255 / .12)' : `color-mix(in srgb, ${CLASS_COLOR[c]} 55%, transparent)`, background: hidden ? 'transparent' : `color-mix(in srgb, ${CLASS_COLOR[c]} 14%, transparent)`, color: hidden ? 'var(--color-fg3)' : 'var(--color-fg)' }}>
            <span aria-hidden className="h-2 w-2 rounded-full transition-opacity" style={{ background: CLASS_COLOR[c], opacity: hidden ? 0.3 : 1 }} />
            <span className={hidden ? 'line-through decoration-white/30' : ''}>{c}</span>
          </button>
        )
      })}
    </div>
  )
}
