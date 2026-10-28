import { AlertTriangle, ChevronDown, Clapperboard, RotateCw } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useId, useState, type ButtonHTMLAttributes, type CSSProperties, type ReactNode } from 'react'
import { CLASS_COLOR, type MovieClass } from '../lib/format'

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(' ')
const EASE = [0.16, 1, 0.3, 1] as const

/* ---------- page header: oversized display type, optional gradient phrase ---------- */
export function PageHeader({ eyebrow, title, highlight, children, actions }: { eyebrow: string; title: string; highlight?: string; children?: ReactNode; actions?: ReactNode }) {
  const parts = highlight && title.includes(highlight) ? title.split(highlight) : null
  return (
    <header className="relative mb-8 flex flex-col gap-5 md:mb-10 lg:flex-row lg:items-end lg:justify-between">
      <div className="max-w-4xl">
        <motion.p initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, ease: EASE }} className="eyebrow mb-4 flex items-center gap-3">
          <span aria-hidden className="inline-block h-[3px] w-8 rounded-full bg-[image:var(--grad-brand)]" />{eyebrow}
        </motion.p>
        <h1 className="font-display text-[2.1rem] font-extrabold leading-[1.02] text-fg sm:text-5xl lg:text-[3.6rem]" aria-label={title}>
          {(parts ?? [title]).map((chunk, k) => (
            <span key={k} aria-hidden>
              <Words text={chunk} delay={k === 0 ? 0.05 : 0.05 + (parts?.[0].split(' ').length ?? 0) * 0.05} />
              {parts && k === 0 && <span className="grad-text"><Words text={highlight!} delay={0.05 + parts[0].split(' ').length * 0.05} /></span>}
            </span>
          ))}
        </h1>
        {children && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25, duration: 0.6 }} className="mt-4 max-w-2xl text-[0.98rem] leading-relaxed text-fg2">{children}</motion.div>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}

/* word-by-word rise inside a clipping mask — the headline entrance used on every page */
export function Words({ text, delay = 0 }: { text: string; delay?: number }) {
  return (
    <>
      {text.split(/(\s+)/).map((w, i) => /^\s+$/.test(w) || w === '' ? w : (
        <span key={i} className="inline-block overflow-hidden pb-[0.12em] align-bottom">
          <motion.span className="inline-block" initial={{ y: '110%' }} animate={{ y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: delay + i * 0.025 }}>{w}</motion.span>
        </span>
      ))}
    </>
  )
}

/* ---------- panel: dark glass with a travelling gradient edge on hover ---------- */
export function Panel({ title, subtitle, actions, children, className, bodyClass, id, marks, tone }: {
  title?: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string; id?: string; marks?: boolean; tone?: string
}) {
  return (
    <section id={id} aria-labelledby={id ? `${id}-t` : undefined} style={tone ? ({ '--edge': tone } as CSSProperties) : undefined}
      className={cx('glass glow-edge rounded-[1.5rem]', marks && 'is-on', className)}>
      {(title || actions) && (
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-5 pt-5 sm:px-6 sm:pt-6">
          <div className="min-w-0">
            {title && <h2 id={id ? `${id}-t` : undefined} className="font-display text-[1.15rem] font-bold text-fg">{title}</h2>}
            {subtitle && <p className="mt-1 text-[0.8rem] leading-snug text-fg3">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className={cx('p-5 sm:p-6', bodyClass)}>{children}</div>
    </section>
  )
}

/* ---------- KPI: oversized numeral, tinted by class ---------- */
export function Kpi({ label, value, hint, accent, tone }: { label: string; value: ReactNode; hint?: ReactNode; accent?: boolean; tone?: MovieClass }) {
  const c = tone ? CLASS_COLOR[tone] : undefined
  return (
    <div className={cx('glass relative overflow-hidden rounded-[1.25rem] p-4 sm:p-5', accent && 'is-on glow-edge')} style={c ? ({ boxShadow: `0 0 0 1px color-mix(in srgb, ${c} 28%, transparent) inset, 0 20px 50px -26px ${c}` } as CSSProperties) : undefined}>
      {c && <span aria-hidden className="absolute -right-8 -top-8 h-24 w-24 rounded-full opacity-40 blur-2xl" style={{ background: c }} />}
      <p className="eyebrow relative flex items-center gap-2">{tone && <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: c, boxShadow: `0 0 10px ${c}` }} />}{label}</p>
      <p className={cx('tnum relative mt-3 font-display text-[2.1rem] font-extrabold leading-none sm:text-[2.5rem]', !tone && 'text-fg')} style={c ? { color: c } : undefined}>{value}</p>
      {hint && <p className="relative mt-2 text-[0.78rem] leading-snug text-fg3">{hint}</p>}
    </div>
  )
}

/* ---------- class badge ---------- */
export function ClassBadge({ value, size = 'sm' }: { value: MovieClass; size?: 'sm' | 'lg' }) {
  const c = CLASS_COLOR[value]
  return (
    <span className={cx('inline-flex items-center gap-1.5 rounded-full border font-mono font-medium uppercase tracking-wider', size === 'sm' ? 'px-2 py-0.5 text-[0.72rem]' : 'px-3.5 py-1.5 text-sm')}
      style={{ color: c, borderColor: `color-mix(in srgb, ${c} 50%, transparent)`, background: `color-mix(in srgb, ${c} 14%, transparent)` }}>
      <span aria-hidden className="h-1.5 w-1.5 rounded-full" style={{ background: c, boxShadow: `0 0 8px ${c}` }} />{value}
    </span>
  )
}

/* ---------- buttons ---------- */
type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'ghost' | 'quiet'; loading?: boolean; icon?: ReactNode }
export function Button({ variant = 'ghost', loading, icon, children, className, disabled, ...rest }: BtnProps) {
  const base = 'inline-flex min-h-11 items-center justify-center gap-2 rounded-full px-5 text-sm font-semibold transition-[background-color,border-color,color,transform,box-shadow] duration-200 active:scale-[0.97] disabled:opacity-50 sm:min-h-10'
  const v = { primary: 'btn-grad', ghost: 'border border-line-strong bg-white/[0.04] text-fg hover:border-white/40 hover:bg-white/[0.09]', quiet: 'text-fg2 hover:bg-white/[0.08] hover:text-fg' }[variant]
  return (
    <button className={cx(base, v, className)} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
      {loading ? <span aria-hidden className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" /> : icon}
      {children}
    </button>
  )
}

/* ---------- segmented control with a sliding pill ---------- */
export function Segmented<T extends string>({ label, value, onChange, options }: { label: string; value: T; onChange: (v: T) => void; options: { value: T; label: string; color?: string }[] }) {
  const gid = useId()
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap gap-0.5 rounded-full border border-line bg-black/25 p-1">
      {options.map((o) => {
        const on = o.value === value
        return (
          <button key={o.value} role="radio" aria-checked={on} onClick={() => onChange(o.value)}
            className={cx('relative flex min-h-9 items-center gap-1.5 rounded-full px-3.5 text-[0.8rem] font-semibold transition-colors duration-200', on ? 'text-[#12082a]' : 'text-fg2 hover:text-fg')}>
            {on && <motion.span layoutId={`seg-${gid}`} className="absolute inset-0 rounded-full" style={{ background: o.color ?? 'var(--grad-brand)' }} transition={{ type: 'spring', stiffness: 420, damping: 34 }} />}
            <span className="relative flex items-center gap-1.5">{o.color && !on && <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: o.color }} />}{o.label}</span>
          </button>
        )
      })}
    </div>
  )
}

/* ---------- states ---------- */
export function Skeleton({ className }: { className?: string }) { return <div aria-hidden className={cx('skeleton', className)} /> }
export function LoadingBlock({ height = 'h-64', label = 'Loading' }: { height?: string; label?: string }) {
  return <div role="status" aria-live="polite" className={cx('skeleton w-full', height)}><span className="sr-only">{label}…</span></div>
}

export function ErrorState({ title = 'Something went wrong', message, onRetry }: { title?: string; message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-3 rounded-[1.25rem] border border-flop/40 bg-flop/10 p-5">
      <div className="flex items-center gap-2 text-danger"><AlertTriangle size={18} aria-hidden /><h3 className="font-display text-base font-bold">{title}</h3></div>
      <p className="max-w-prose text-sm text-fg2">{message}</p>
      {onRetry && <Button variant="ghost" icon={<RotateCw size={15} aria-hidden />} onClick={onRetry}>Try again</Button>}
    </div>
  )
}

export function EmptyState({ title, message, action }: { title: string; message: ReactNode; action?: ReactNode }) {
  return (
    <div className="glass flex flex-col items-center justify-center gap-3 rounded-[1.5rem] border-dashed px-6 py-14 text-center">
      <span className="flex h-14 w-14 items-center justify-center rounded-full bg-white/[0.06]"><Clapperboard size={24} className="text-accent" aria-hidden /></span>
      <h3 className="font-display text-lg font-bold">{title}</h3>
      <p className="max-w-md text-sm text-fg3">{message}</p>
      {action}
    </div>
  )
}

/* ---------- note (limitations etc.) ---------- */
export function Note({ tone = 'warn', title, children }: { tone?: 'warn' | 'info'; title: string; children: ReactNode }) {
  const c = tone === 'warn' ? 'var(--color-flop)' : 'var(--color-cyan)'
  return (
    <aside className="relative overflow-hidden rounded-[1.25rem] p-5 sm:p-6" style={{ background: `linear-gradient(120deg, color-mix(in srgb, ${c} 16%, transparent), rgb(255 255 255 / .03) 60%)`, border: `1px solid color-mix(in srgb, ${c} 38%, transparent)` }}>
      <span aria-hidden className="absolute inset-y-0 left-0 w-1" style={{ background: c, boxShadow: `0 0 18px ${c}` }} />
      <h2 className="font-display text-[1.02rem] font-bold text-fg">{title}</h2>
      <div className="mt-1.5 space-y-2 text-[0.88rem] leading-relaxed text-fg2">{children}</div>
    </aside>
  )
}

/* ---------- scroll-triggered reveal ---------- */
export const stagger = { hidden: {}, show: {} }
export function Reveal({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div initial={{ opacity: 0, y: 26 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.08, margin: '0px 0px -6% 0px' }} transition={{ duration: 0.7, ease: EASE, delay }} className={className}>
      {children}
    </motion.div>
  )
}

/* ---------- chart tooltip shell ---------- */
export function TipBox({ title, rows }: { title?: ReactNode; rows: { label: string; value: ReactNode; color?: string }[] }) {
  return (
    <div className="pointer-events-none min-w-40 rounded-2xl border border-white/20 bg-[#0e0b2e]/95 px-3.5 py-2.5 text-xs shadow-2xl shadow-black/60 backdrop-blur">
      {title && <p className="mb-1.5 font-display text-[0.85rem] font-bold text-fg">{title}</p>}
      <ul className="space-y-1">
        {rows.map((r) => (
          <li key={r.label} className="flex items-center justify-between gap-4 text-fg2">
            <span className="flex items-center gap-1.5">{r.color && <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: r.color, boxShadow: `0 0 8px ${r.color}` }} />}{r.label}</span>
            <span className="tnum font-mono text-fg">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ---------- shared bar-fill gradient (for chart <defs>) ---------- */
export function ClassGradients() {
  return (
    <defs>
      {(['Hit', 'Average', 'Flop'] as MovieClass[]).map((c) => (
        <linearGradient key={c} id={`cg-${c}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={CLASS_COLOR[c]} stopOpacity={0.95} /><stop offset="1" stopColor={CLASS_COLOR[c]} stopOpacity={0.35} />
        </linearGradient>
      ))}
    </defs>
  )
}

/* ---------- expandable detail (keyboard + screen-reader friendly) ---------- */
export function Disclosure({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const id = useId()
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="mt-3 overflow-hidden rounded-2xl border border-line bg-black/20">
      <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)}
        className="flex min-h-11 w-full items-center justify-between gap-3 px-4 text-left text-[0.85rem] font-semibold text-fg transition-colors hover:bg-white/[0.05]">
        {title}
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.25 }} className="text-fg3"><ChevronDown size={16} aria-hidden /></motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div id={id} role="region" aria-label={title} initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.28, ease: EASE }} className="overflow-hidden">
            <div className="border-t border-line px-4 py-3.5 text-[0.84rem] leading-relaxed text-fg2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
