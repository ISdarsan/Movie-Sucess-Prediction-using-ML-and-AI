import { BookOpen, Bot, Crosshair, Film, LayoutGrid, Lightbulb, Menu, MessagesSquare, ScanLine, X } from 'lucide-react'
import { AnimatePresence, motion, useScroll, useSpring } from 'motion/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import type { Health } from '../lib/types'
import { useApi } from '../lib/useApi'
import { Backdrop } from './fx'
import { cx } from './ui'

type NavItem = { to: string; label: string; short: string; icon: typeof Menu; end?: boolean }
export const NAV: NavItem[] = [
  { to: '/', label: 'Overview', short: 'Overview', icon: LayoutGrid, end: true },
  { to: '/predict', label: 'Predictor', short: 'Predictor', icon: Crosshair },
  { to: '/performance', label: 'Model performance', short: 'Performance', icon: ScanLine },
  { to: '/explain', label: 'Explainability', short: 'Explain', icon: Lightbulb },
  { to: '/explore', label: 'Explore movies', short: 'Explore', icon: Film },
  { to: '/researcher', label: 'AI Movie Researcher', short: 'Researcher', icon: Bot },
  { to: '/assistant', label: 'AI Assistant', short: 'Assistant', icon: MessagesSquare },
  { to: '/about', label: 'Methodology', short: 'Method', icon: BookOpen },
]

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden className="shrink-0">
        <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#ffb23e" /><stop offset=".5" stopColor="#ff4fa3" /><stop offset="1" stopColor="#8b5cff" /></linearGradient></defs>
        <path d="M5 12V5h7M20 5h7v7M27 20v7h-7M12 27H5v-7" fill="none" stroke="url(#lg)" strokeWidth="2.8" strokeLinecap="round" />
        <rect x="12" y="12" width="8" height="8" rx="2.5" fill="#f6f3ff" />
      </svg>
      <span className="font-display text-[1.2rem] font-extrabold tracking-[0.16em] text-fg">FRAME</span>
    </div>
  )
}

function ModelStatus({ compact }: { compact?: boolean }) {
  const { data, error, loading } = useApi<Health>('/api/health')
  const ok = data?.status === 'ok'
  const text = loading ? 'Checking model…' : error ? 'API unreachable' : ok ? 'Pre-release model online' : 'Model unavailable'
  const color = loading ? 'var(--color-fg3)' : ok ? 'var(--color-hit)' : 'var(--color-flop)'
  return (
    <div className="flex items-center gap-2 text-[0.75rem] text-fg2" role="status" title={text}>
      <span aria-hidden className="relative flex h-2.5 w-2.5 shrink-0">
        {ok && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: color }} />}
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      </span>
      <span className={compact ? 'sr-only' : ''}>{text}</span>
    </div>
  )
}

export function AppShell() {
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  const menuBtn = useRef<HTMLButtonElement>(null)
  const closeBtn = useRef<HTMLButtonElement>(null)
  const { scrollYProgress } = useScroll()
  const bar = useSpring(scrollYProgress, { stiffness: 140, damping: 28 })

  useEffect(() => { setOpen(false); window.scrollTo({ top: 0 }) }, [loc.pathname])
  // pointer-following highlight for every .glass surface (CSS reads --mx/--my); skipped for touch + reduced motion
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let raf = 0
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== 'mouse') return
      const el = (e.target as HTMLElement | null)?.closest<HTMLElement>('.glass')
      if (!el) return
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => { const r = el.getBoundingClientRect(); el.style.setProperty('--mx', `${e.clientX - r.left}px`); el.style.setProperty('--my', `${e.clientY - r.top}px`) })
    }
    document.addEventListener('pointermove', onMove, { passive: true })
    return () => { document.removeEventListener('pointermove', onMove); cancelAnimationFrame(raf) }
  }, [])
  useEffect(() => {
    if (!open) return
    closeBtn.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setOpen(false); menuBtn.current?.focus() } }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = '' }
  }, [open])

  const current = NAV.find((n) => (n.end ? loc.pathname === n.to : loc.pathname.startsWith(n.to)))

  return (
    <div className="min-h-dvh">
      <Backdrop />
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[100] focus:rounded-full focus:bg-white focus:px-4 focus:py-2 focus:text-black">Skip to content</a>
      <motion.div aria-hidden className="fixed inset-x-0 top-0 z-[60] h-[3px] origin-left bg-[image:var(--grad-brand)]" style={{ scaleX: bar }} />

      {/* floating pill navigation */}
      <header className="sticky top-0 z-40 px-3 pt-3 sm:px-5 sm:pt-4">
        <div className="glass mx-auto flex h-14 max-w-[1440px] items-center justify-between gap-3 rounded-full pl-5 pr-2.5">
          <NavLink to="/" aria-label="FRAME home"><Logo /></NavLink>

          <nav aria-label="Primary" className="hidden items-center gap-0.5 md:flex">
            {NAV.map(({ to, label, short, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end} title={label}
                className={({ isActive }) => cx('relative flex min-h-10 items-center gap-2 rounded-full px-3.5 text-[0.85rem] font-medium transition-colors duration-200 xl:px-4', isActive ? 'text-[#12082a]' : 'text-fg2 hover:text-fg')}>
                {({ isActive }) => (
                  <>
                    {isActive && <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-full bg-[image:var(--grad-brand)] shadow-[0_6px_22px_-6px_rgb(255_79_163/.8)]" transition={{ type: 'spring', stiffness: 440, damping: 36 }} />}
                    <Icon size={16} aria-hidden className="relative shrink-0" />
                    <span className="relative hidden xl:inline">{short}</span>
                    <span className="sr-only xl:hidden">{label}</span>
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <div className="hidden min-[1440px]:block"><ModelStatus /></div>
            <div className="hidden md:block min-[1440px]:hidden"><ModelStatus compact /></div>
            <span className="font-mono text-[0.72rem] uppercase tracking-wider text-fg3 md:hidden">{current?.short}</span>
            <button ref={menuBtn} onClick={() => setOpen(true)} aria-label="Open navigation" aria-expanded={open} aria-controls="mobile-nav"
              className="flex h-11 w-11 items-center justify-center rounded-full text-fg2 hover:bg-white/10 hover:text-fg md:hidden"><Menu size={22} aria-hidden /></button>
          </div>
        </div>
      </header>

      <AnimatePresence>
        {open && (
          <div className="fixed inset-0 z-50 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation" id="mobile-nav">
            <motion.div className="absolute inset-0 bg-black/75 backdrop-blur-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setOpen(false)} />
            <motion.div className="glass absolute inset-y-3 right-3 flex w-[min(20rem,88vw)] flex-col rounded-[2rem] px-4"
              initial={{ x: '110%' }} animate={{ x: 0 }} exit={{ x: '110%' }} transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}>
              <div className="flex h-16 items-center justify-between">
                <Logo />
                <button ref={closeBtn} onClick={() => { setOpen(false); menuBtn.current?.focus() }} aria-label="Close navigation" className="flex h-11 w-11 items-center justify-center rounded-full text-fg2 hover:bg-white/10"><X size={22} aria-hidden /></button>
              </div>
              <nav aria-label="Primary" className="mt-2 flex flex-1 flex-col gap-1.5">
                {NAV.map(({ to, label, icon: Icon, end }, i) => (
                  <motion.div key={to} initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.06 * i + 0.1, duration: 0.35 }}>
                    <NavLink to={to} end={end} onClick={() => setOpen(false)}
                      className={({ isActive }) => cx('flex min-h-12 items-center gap-3 rounded-2xl px-4 text-[1rem] font-semibold', isActive ? 'bg-[image:var(--grad-brand)] text-[#12082a]' : 'text-fg2 hover:bg-white/10 hover:text-fg')}>
                      <Icon size={19} aria-hidden />{label}
                    </NavLink>
                  </motion.div>
                ))}
              </nav>
              <div className="pb-6 pt-4"><ModelStatus /></div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <main id="main" tabIndex={-1} className="flex min-h-[100dvh] flex-col outline-none">
        <div className="mx-auto w-full max-w-[1440px] flex-1 px-[var(--gutter)] pb-24 pt-8 md:pt-12">
          <PageFade key={loc.pathname}><Outlet /></PageFade>
        </div>
        <footer className="mx-auto mt-auto w-full max-w-[1440px] px-[var(--gutter)] pb-10 text-center text-[0.75rem] text-fg3">
          FRAME · IMDb-rating categories from the IMDb 5000 dataset · estimates from a pre-release model, not a guarantee of commercial success
        </footer>
      </main>
    </div>
  )
}

function PageFade({ children }: { children: ReactNode }) {
  return <motion.div initial={{ opacity: 0, y: 14, filter: 'blur(6px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }} transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}>{children}</motion.div>
}
