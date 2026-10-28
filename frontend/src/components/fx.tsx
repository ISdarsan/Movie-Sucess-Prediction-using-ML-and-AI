import { animate, motion, useInView, useMotionValue, useReducedMotion, useSpring, useTransform } from 'motion/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link, type LinkProps } from 'react-router-dom'
import { cx } from './ui'

/* ---------- fixed atmosphere: gradient orbs + grain + vignette (transform-only animation) ---------- */
export function Backdrop() {
  return (
    <div className="backdrop" aria-hidden>
      <div className="orb orb-v" /><div className="orb orb-c" /><div className="orb orb-p" />
      <div className="grain" /><div className="vignette" />
    </div>
  )
}

/* ---------- count-up that starts when scrolled into view ---------- */
export function Counter({ value, decimals = 0, suffix = '', prefix = '', duration = 1.6, className }: { value: number; decimals?: number; suffix?: string; prefix?: string; duration?: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: '0px 0px -10% 0px' })
  const reduce = useReducedMotion()
  const [shown, setShown] = useState(reduce ? value : 0)
  useEffect(() => {
    if (reduce) { setShown(value); return }
    if (!inView) return
    const c = animate(0, value, { duration, ease: [0.16, 1, 0.3, 1], onUpdate: setShown })
    return () => c.stop()
  }, [inView, value, duration, reduce])
  return <span ref={ref} className={cx('tnum', className)}>{prefix}{shown.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>
}

/* ---------- mount children only when visible, so chart entrance animations play on scroll ---------- */
export function InView({ children, minHeight = 240, className }: { children: ReactNode; minHeight?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const seen = useInView(ref, { once: true, margin: '0px 0px -8% 0px' })
  return <div ref={ref} className={className} style={{ minHeight }}>{seen ? children : null}</div>
}

/* ---------- pointer-reactive tilt ---------- */
export function Tilt({ children, className, max = 7 }: { children: ReactNode; className?: string; max?: number }) {
  const reduce = useReducedMotion()
  const rx = useSpring(useMotionValue(0), { stiffness: 220, damping: 22 })
  const ry = useSpring(useMotionValue(0), { stiffness: 220, damping: 22 })
  const gx = useMotionValue(50), gy = useMotionValue(50)
  const glare = useTransform([gx, gy], ([x, y]) => `radial-gradient(420px circle at ${x}% ${y}%, rgb(255 255 255 / .13), transparent 55%)`)
  if (reduce) return <div className={className}>{children}</div>
  return (
    <motion.div
      className={cx('relative [transform-style:preserve-3d]', className)}
      style={{ rotateX: rx, rotateY: ry, transformPerspective: 900 }}
      onPointerMove={(e) => {
        if (e.pointerType !== 'mouse') return
        const r = e.currentTarget.getBoundingClientRect()
        const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height
        ry.set((px - 0.5) * max * 2); rx.set(-(py - 0.5) * max * 2); gx.set(px * 100); gy.set(py * 100)
      }}
      onPointerLeave={() => { rx.set(0); ry.set(0) }}
    >
      {children}
      <motion.span aria-hidden className="pointer-events-none absolute inset-0 rounded-[inherit]" style={{ background: glare }} />
    </motion.div>
  )
}

/* ---------- magnetic link-button ---------- */
export function MagneticLink({ children, className, ...rest }: LinkProps & { className?: string }) {
  const reduce = useReducedMotion()
  const x = useSpring(useMotionValue(0), { stiffness: 260, damping: 18 })
  const y = useSpring(useMotionValue(0), { stiffness: 260, damping: 18 })
  return (
    <motion.span style={{ x, y, display: 'inline-block' }} whileTap={{ scale: 0.96 }}
      onPointerMove={(e) => {
        if (reduce || e.pointerType !== 'mouse') return
        const r = e.currentTarget.getBoundingClientRect()
        x.set((e.clientX - (r.left + r.width / 2)) * 0.28); y.set((e.clientY - (r.top + r.height / 2)) * 0.28)
      }}
      onPointerLeave={() => { x.set(0); y.set(0) }}>
      <Link className={className} {...rest}>{children}</Link>
    </motion.span>
  )
}

/* ---------- film reel (SVG) ---------- */
export function Reel({ className, spin = true }: { className?: string; spin?: boolean }) {
  const holes = Array.from({ length: 6 }, (_, i) => i * 60)
  return (
    <svg viewBox="0 0 200 200" className={cx(spin && 'reel-spin', className)} aria-hidden>
      <defs>
        <linearGradient id="reelg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#ffb23e" /><stop offset=".5" stopColor="#ff4fa3" /><stop offset="1" stopColor="#8b5cff" /></linearGradient>
      </defs>
      <circle cx="100" cy="100" r="94" fill="none" stroke="url(#reelg)" strokeWidth="3" />
      <circle cx="100" cy="100" r="84" fill="rgb(255 255 255 / .03)" stroke="rgb(255 255 255 / .18)" />
      {holes.map((a) => <circle key={a} cx="100" cy="46" r="17" fill="#06061a" stroke="rgb(255 255 255 / .28)" strokeWidth="1.5" transform={`rotate(${a} 100 100)`} />)}
      <circle cx="100" cy="100" r="15" fill="none" stroke="url(#reelg)" strokeWidth="3" />
      <circle cx="100" cy="100" r="5" fill="#f6f3ff" />
    </svg>
  )
}

/* ---------- looping film strip ribbon ---------- */
export function FilmStrip({ className, hues = ['#ff4fa3', '#8b5cff', '#25e8d3', '#ffb23e'] }: { className?: string; hues?: string[] }) {
  const frames = Array.from({ length: 20 }, (_, i) => i)
  return (
    <div className={cx('overflow-hidden', className)} aria-hidden>
      <div className="strip-run flex w-[200%]">
        {[0, 1].map((k) => (
          <div key={k} className="flex w-1/2 shrink-0 gap-2 border-y-[10px] border-[rgb(255_255_255/.07)] bg-[rgb(255_255_255/.03)] px-1">
            {frames.map((i) => (
              <div key={i} className="my-0 h-16 w-24 shrink-0 rounded-[3px] opacity-80" style={{ background: `linear-gradient(135deg, ${hues[i % hues.length]}55, ${hues[(i + 2) % hues.length]}22)` }} />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/* ---------- cinematic processing loader (projector beam + reel) ---------- */
export function ProjectorLoader({ steps, label = 'Working' }: { steps: string[]; label?: string }) {
  const [i, setI] = useState(0)
  useEffect(() => { const t = setInterval(() => setI((n) => Math.min(n + 1, steps.length - 1)), 520); return () => clearInterval(t) }, [steps.length])
  return (
    <div role="status" aria-live="polite" className="relative flex min-h-72 flex-col items-center justify-center overflow-hidden rounded-[inherit] px-6 py-10 text-center">
      <div aria-hidden className="beam absolute left-1/2 top-0 h-full w-[130%] -translate-x-1/2 [clip-path:polygon(46%_0,54%_0,100%_100%,0_100%)] bg-[linear-gradient(180deg,rgb(255_255_255/.2),rgb(139_92_255/.05)_70%,transparent)]" />
      <div aria-hidden className="scanline absolute inset-x-6 top-0 h-px bg-gradient-to-r from-transparent via-[#25e8d3] to-transparent shadow-[0_0_18px_#25e8d3]" />
      <Reel className="relative h-20 w-20" />
      <p className="eyebrow relative mt-6">{label}</p>
      <ul className="relative mt-3 space-y-1.5 font-mono text-[0.8rem]">
        {steps.map((s, k) => (
          <li key={s} className={cx('transition-all duration-300', k < i ? 'text-hit' : k === i ? 'text-fg' : 'text-fg3/60')}>
            {k < i ? '✓ ' : k === i ? '▸ ' : '· '}{s}
          </li>
        ))}
      </ul>
      <span className="sr-only">{steps[i]}</span>
    </div>
  )
}

/* ---------- radial gauge ---------- */
export function Gauge({ value, label, hint, color = '#25e8d3', size = 148 }: { value: number; label: string; hint?: string; color?: string; size?: number }) {
  const ref = useRef<SVGSVGElement>(null)
  const inView = useInView(ref, { once: true })
  const r = 52, c = 2 * Math.PI * r
  const gid = `gg-${label.replace(/[^a-z0-9]/gi, '')}`
  return (
    <figure className="flex flex-col items-center text-center">
      <svg ref={ref} width={size} height={size} viewBox="0 0 128 128" role="img" aria-label={`${label}: ${(value * 100).toFixed(1)}%`}>
        <defs><linearGradient id={gid} x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor={color} /><stop offset="1" stopColor="#8b5cff" /></linearGradient></defs>
        <circle cx="64" cy="64" r={r} fill="none" stroke="rgb(255 255 255 / .09)" strokeWidth="10" />
        <motion.circle cx="64" cy="64" r={r} fill="none" stroke={`url(#${gid})`} strokeWidth="10" strokeLinecap="round" transform="rotate(-90 64 64)"
          strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: inView ? c * (1 - value) : c }} transition={{ duration: 1.3, ease: [0.16, 1, 0.3, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${color}99)` }} />
        <text x="64" y="70" textAnchor="middle" className="fill-[#f6f3ff] font-display" style={{ fontSize: 24, fontWeight: 700 }}>{(value * 100).toFixed(1)}%</text>
      </svg>
      <figcaption className="mt-2 text-sm font-medium text-fg">{label}</figcaption>
      {hint && <p className="mt-0.5 max-w-[11rem] text-[0.75rem] leading-snug text-fg3">{hint}</p>}
    </figure>
  )
}
