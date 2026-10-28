/* Hero visuals built from real data: the outcome ring (class shares of the whole dataset) and the top-rated frame strip. */
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CLASS_COLOR, fmtInt, fmtPct, type MovieClass } from '../lib/format'
import type { MovieList, Overview } from '../lib/types'
import { useApi } from '../lib/useApi'
import { Counter, Reel } from './fx'
import { cx } from './ui'

const R = 168, C = 2 * Math.PI * R, GAP = 7
const BAND: Record<MovieClass, string> = { Flop: 'IMDb below 3', Average: 'IMDb 3 to 6', Hit: 'IMDb 6 and above' }
const NOTE: Record<MovieClass, string> = {
  Hit: 'The majority class — any model has to beat a guess of “Hit”.',
  Average: 'The middle band, and the one the model finds hardest to separate from Hit.',
  Flop: 'So rare that the model has almost nothing to learn from.',
}

/** Donut whose arc lengths are the real class shares. Hover, focus or click an arc to read it in the hub. */
export function OutcomeReel() {
  const { data } = useApi<Overview>('/api/overview', { cache: true })
  const reduce = useReducedMotion()
  const [hover, setHover] = useState<MovieClass | null>(null)
  const [pinned, setPinned] = useState<MovieClass | null>(null)
  const active = hover ?? pinned
  const k = data && !data.empty ? data.kpis : null
  const order: MovieClass[] = ['Hit', 'Average', 'Flop']
  let start = 0
  const arcs = order.map((c) => {
    const share = k ? k.class_share[c] : { Hit: 0.7105, Average: 0.2811, Flop: 0.0083 }[c]
    const len = Math.max(share * C - GAP, 5)
    const a = { c, share, len, start }
    start += share * C
    return a
  })
  return (
    <div className="relative h-full w-full">
      <svg viewBox="0 0 400 400" className="absolute inset-0 h-full w-full" role="group" aria-label="Class shares of the whole dataset. Select a segment to read it.">
        <foreignObject x="68" y="68" width="264" height="264" aria-hidden className="pointer-events-none"><Reel className="h-full w-full opacity-60" /></foreignObject>
        <circle cx="200" cy="200" r={R} fill="none" stroke="rgb(255 255 255 / .07)" strokeWidth="16" />
        {arcs.map((a, i) => {
          const on = active === a.c, dim = active !== null && !on
          const label = `${a.c}: ${k ? fmtInt(k.class_counts[a.c]) : '…'} films, ${fmtPct(a.share)}. ${BAND[a.c]}.`
          return (
            <g key={a.c} role="button" tabIndex={0} aria-pressed={pinned === a.c} aria-label={label} className="cursor-pointer outline-none [&:focus-visible>circle:first-child]:stroke-white"
              onPointerEnter={() => setHover(a.c)} onPointerLeave={() => setHover(null)} onFocus={() => setHover(a.c)} onBlur={() => setHover(null)}
              onClick={() => setPinned((p) => (p === a.c ? null : a.c))} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setPinned((p) => (p === a.c ? null : a.c)) } }}>
              {/* fat invisible hit area so even the tiny Flop arc is easy to reach */}
              <circle cx="200" cy="200" r={R} fill="none" stroke="transparent" strokeWidth="46" strokeDasharray={`${a.len + 16} ${C}`} strokeDashoffset={-(a.start - 8)} transform="rotate(-90 200 200)" />
              <motion.circle cx="200" cy="200" r={R} fill="none" stroke={CLASS_COLOR[a.c]} strokeLinecap="round" transform="rotate(-90 200 200)" strokeDashoffset={-a.start}
                initial={{ strokeDasharray: `0 ${C}`, strokeWidth: 16 }} animate={{ strokeDasharray: `${a.len} ${C - a.len}`, strokeWidth: on ? 26 : 16, opacity: dim ? 0.3 : 1 }}
                transition={{ strokeDasharray: { duration: reduce ? 0 : 1.1, delay: reduce ? 0 : 0.5 + i * 0.18, ease: [0.16, 1, 0.3, 1] }, strokeWidth: { duration: 0.25 }, opacity: { duration: 0.25 } }}
                style={{ filter: `drop-shadow(0 0 ${on ? 14 : 7}px ${CLASS_COLOR[a.c]})` }} />
            </g>
          )
        })}
      </svg>
      {/* hub readout */}
      <div className="pointer-events-none absolute inset-[24%] flex items-center justify-center rounded-full bg-[radial-gradient(circle,rgb(6_6_26/.9),rgb(6_6_26/.55)_70%,transparent)] text-center">
        <AnimatePresence mode="wait">
          {active && k ? (
            <motion.div key={active} initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }} transition={{ duration: 0.18 }} className="px-2">
              <p className="font-mono text-[0.72rem] uppercase tracking-[0.18em]" style={{ color: CLASS_COLOR[active] }}>{active} · {BAND[active]}</p>
              <p className="font-display text-[2.6rem] font-extrabold leading-none text-fg sm:text-[3.2rem]"><Counter value={k.class_counts[active]} duration={0.5} /></p>
              <p className="font-mono text-[0.72rem] text-fg2">{fmtPct(k.class_share[active])} of films</p>
              <p className="mx-auto mt-1.5 hidden max-w-[11.5rem] text-[0.72rem] leading-snug text-fg3 sm:block">{NOTE[active]}</p>
            </motion.div>
          ) : (
            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
              <p className="font-display text-[2.6rem] font-extrabold leading-none text-fg sm:text-[3.2rem]"><Counter value={k?.movies ?? 0} /></p>
              <p className="mt-1 font-mono text-[0.72rem] uppercase tracking-[0.16em] text-fg3">real films</p>
              <p className="mt-2 text-[0.72rem] text-fg3">Hover or tap a segment</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

const hueOf = (g: string | null) => { let h = 0; for (const ch of (g ?? '').split('|')[0]) h = (h * 31 + ch.charCodeAt(0)) % 360; return h }

/** A slow strip of the highest-rated real films; pauses on hover/focus, each frame opens Explore. */
export function FrameStrip() {
  const { data } = useApi<MovieList>('/api/movies?sort=imdb_score&order=desc&page_size=24', { cache: true })
  if (!data) return <div className="skeleton -mx-[var(--gutter)] mt-10 h-36" aria-hidden />
  const frames = data.rows.slice(0, 20)
  const row = (hidden: boolean) => (
    <ul className="flex shrink-0 gap-3 pr-3" aria-hidden={hidden || undefined}>
      {frames.map((m) => {
        const c = CLASS_COLOR[m.Classify], h = hueOf(m.genres)
        return (
          <li key={m.id} className="shrink-0">
            <Link to={`/explore?q=${encodeURIComponent(m.title)}`} tabIndex={hidden ? -1 : 0}
              className="group relative flex h-32 w-52 shrink-0 flex-col justify-between overflow-hidden rounded-xl border border-white/15 p-3 transition-all duration-300 hover:-translate-y-1 hover:border-white/50"
              style={{ background: `radial-gradient(90% 80% at 90% 0%, ${c}55, transparent 60%), linear-gradient(155deg, hsl(${h} 72% 36%), hsl(${(h + 70) % 360} 62% 15%))` }}>
              <span aria-hidden className="perf-edge pointer-events-none absolute inset-x-0 top-0 h-2.5 opacity-60" />
              <span className="mt-2 flex items-start justify-between">
                <span className="font-mono text-[0.72rem] uppercase tracking-widest text-fg2">{m.year ?? '—'}</span>
                <span className="font-display text-3xl font-extrabold leading-none tabular-nums" style={{ color: c, textShadow: `0 0 20px ${c}aa` }}>{m.imdb_score.toFixed(1)}</span>
              </span>
              <span className="line-clamp-2 text-[0.85rem] font-semibold leading-tight text-fg">{m.title}</span>
            </Link>
          </li>
        )
      })}
    </ul>
  )
  return (
    <div className="-mx-[var(--gutter)] mt-10">
      <p className="eyebrow mb-3 px-[var(--gutter)]">Highest rated in the archive · select a frame to open it</p>
      <div className="group/strip overflow-hidden [mask-image:linear-gradient(90deg,transparent,#000_8%,#000_92%,transparent)]">
        <div className={cx('strip-run flex w-max [animation-duration:90s] hover:[animation-play-state:paused] focus-within:[animation-play-state:paused]')}>{row(false)}{row(true)}</div>
      </div>
    </div>
  )
}
