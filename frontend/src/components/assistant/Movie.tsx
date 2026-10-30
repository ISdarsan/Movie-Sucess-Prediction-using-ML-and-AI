import { Film, Loader2, MessagesSquare, X } from 'lucide-react'
import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { RUN_ID_RE, type Analysis } from '../../lib/analyses'
import { mapError, MOVIE_STARTER_PROMPTS, movieName, type MovieRef } from '../../lib/chat'
import { ClassBadge, cx } from '../ui'
import type { MovieClass } from '../../lib/format'

/* ---------------------------------------------------------------- which film is being discussed */
export function MovieChip({ movie, onChange, disabled }: { movie: MovieRef; onChange: () => void; disabled: boolean }) {
  return (
    <div data-testid="movie-chip" className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line bg-cyan/[0.07] px-4 py-2.5 sm:px-5">
      <Film size={16} aria-hidden className="shrink-0 text-cyan" />
      <p className="min-w-0 flex-1 basis-60 text-[0.82rem] leading-snug text-fg2">
        <span className="text-fg3">Discussing </span><b className="font-semibold text-fg" data-testid="movie-chip-title">{movieName(movie)}</b>{' '}
        <span className="whitespace-nowrap"><ClassBadge value={movie.predictedClass as MovieClass} /></span>
        <span className="block text-[0.74rem] text-fg3">A stored analysis. The assistant explains it; it cannot change the prediction.</span>
      </p>
      <button type="button" onClick={onChange} disabled={disabled} data-testid="change-movie"
        className="inline-flex min-h-10 items-center rounded-full border border-line-strong px-3.5 text-[0.8rem] font-semibold text-fg2 hover:border-white/50 hover:text-fg disabled:opacity-50">Change movie</button>
    </div>
  )
}

/* ---------------------------------------------------------------- empty state of a movie conversation */
export function MovieEmptyChat({ movie, onPick, disabled }: { movie: MovieRef; onPick: (p: string) => void; disabled: boolean }) {
  const prompts = [`Why did the model predict ${movie.predictedClass}?`, ...MOVIE_STARTER_PROMPTS.slice(1)]
  return (
    <div className="flex h-full flex-col items-center justify-center gap-5 px-4 py-8 text-center" data-testid="empty-movie-chat">
      <span aria-hidden className="flex h-14 w-14 items-center justify-center rounded-full bg-cyan/15 text-cyan ring-1 ring-cyan/40"><Film size={24} /></span>
      <div>
        <h2 className="font-display text-xl font-bold text-fg">Ask about this analysis</h2>
        <p className="mx-auto mt-1.5 max-w-md text-[0.88rem] text-fg3">{movieName(movie)} · predicted {movie.predictedClass}</p>
      </div>
      <ul className="flex max-w-2xl flex-wrap justify-center gap-2" aria-label="Suggested questions">
        {prompts.map((p) => (
          <li key={p}>
            <button type="button" disabled={disabled} onClick={() => onPick(p)} data-testid="starter"
              className="min-h-11 rounded-full border border-line-strong bg-white/[0.04] px-4 py-2 text-left text-[0.84rem] font-medium text-fg2 transition-all duration-200 hover:-translate-y-0.5 hover:border-white/50 hover:bg-white/[0.09] hover:text-fg active:scale-95 disabled:opacity-50 disabled:hover:translate-y-0">{p}</button>
          </li>
        ))}
      </ul>
      <ul className="max-w-md list-disc space-y-1 pl-5 text-left text-[0.76rem] leading-snug text-fg3">
        <li>The answers explain the stored result. Gemini cannot change the predicted class or its scores.</li>
        <li>It keeps three things apart: the model’s prediction, historical dataset evidence about other films, and web research background.</li>
        <li>Hit, Average and Flop are IMDb-score classes, not financial success, and the scores are uncalibrated.</li>
      </ul>
    </div>
  )
}

/* ---------------------------------------------------------------- an offer, never an automatic switch */
export function OfferBanner({ movie, existing, onStart, onOpen, onDismiss }: { movie: MovieRef; existing: boolean; onStart: () => void; onOpen: () => void; onDismiss: () => void }) {
  return (
    <section aria-label="Offer: discuss an analysis" data-testid="offer" className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-3 rounded-[1.25rem] border border-cyan/40 bg-cyan/[0.08] px-5 py-4">
      <MessagesSquare size={20} aria-hidden className="shrink-0 text-cyan" />
      <div className="min-w-0 flex-1 basis-64">
        <p className="text-[0.92rem] font-semibold text-fg">Talk about <span data-testid="offer-title">{movieName(movie)}</span>?</p>
        <p className="text-[0.8rem] text-fg2">{existing ? 'You already have a conversation about this analysis.' : 'Start a movie-aware conversation about this analysis. Your current conversation is not changed.'}</p>
      </div>
      <div className="flex gap-2">
        <button type="button" onClick={existing ? onOpen : onStart} data-testid="offer-accept" className="btn-grad inline-flex min-h-11 items-center rounded-full px-5 text-sm font-semibold sm:min-h-10">{existing ? 'Open it' : 'Start conversation'}</button>
        <button type="button" onClick={onDismiss} data-testid="offer-dismiss" className="inline-flex min-h-11 items-center rounded-full border border-line-strong px-4 text-sm font-semibold text-fg2 hover:text-fg sm:min-h-10">Not now</button>
      </div>
    </section>
  )
}

/* ---------------------------------------------------------------- picker: analyses from this tab, or a pasted run id */
export function MoviePicker({ analyses, lookup, onPick, onClose }: { analyses: Analysis[]; lookup: (runId: string) => Promise<MovieRef>; onPick: (m: MovieRef) => void; onClose: () => void }) {
  const tid = useId()
  const box = useRef<HTMLDivElement>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pasted, setPasted] = useState('')

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    const focusables = () => Array.from(box.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled])') ?? [])
    focusables()[0]?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose() }
      if (e.key === 'Tab') {
        const f = focusables(); if (!f.length) return
        if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus() }
        else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus() }
      }
    }
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('keydown', onKey); opener?.focus?.() }
  }, [onClose])

  async function choose(runId: string) {
    if (busy) return
    setBusy(runId); setError(null)
    try { onPick(await lookup(runId)) }                         // the server confirms the analysis exists and is complete before a conversation starts
    catch (e) { setError(mapError(e, 0).message) }
    finally { setBusy(null) }
  }
  const pastedId = pasted.trim().toLowerCase()
  const validPasted = RUN_ID_RE.test(pastedId)
  const submit = (e: FormEvent) => { e.preventDefault(); if (validPasted) void choose(pastedId) }

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/75 backdrop-blur-sm" onClick={onClose} aria-hidden />
      <div ref={box} role="dialog" aria-modal="true" aria-labelledby={tid} data-testid="movie-picker" className="glass relative flex max-h-[85dvh] w-full max-w-lg flex-col rounded-[1.5rem] p-5 shadow-2xl sm:p-6">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 id={tid} className="font-display text-lg font-bold text-fg">Discuss an analysis</h2>
            <p className="mt-1 text-[0.82rem] text-fg3">Pick a completed analysis. The conversation will be about that film only; to discuss another, start another conversation.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-fg2 hover:bg-white/10"><X size={18} aria-hidden /></button>
        </div>

        <div className="mt-4 min-h-0 flex-1 overflow-y-auto">
          {analyses.length ? (
            <ul className="space-y-2" aria-label="Analyses from this session" data-testid="picker-list">
              {analyses.map((a) => (
                <li key={a.runId}>
                  <button type="button" onClick={() => void choose(a.runId)} disabled={!!busy} data-testid="picker-item"
                    className={cx('flex min-h-12 w-full items-center gap-3 rounded-2xl border border-line bg-white/[0.03] px-4 py-2.5 text-left hover:border-white/40 disabled:opacity-60')}>
                    <Film size={16} aria-hidden className="shrink-0 text-cyan" />
                    <span className="min-w-0 flex-1"><span className="block truncate text-[0.88rem] font-semibold text-fg">{movieName(a)}</span>
                      <span className="block truncate text-[0.74rem] text-fg3">{[a.title ? [a.genres, a.year].filter(Boolean).join(' · ') : null, new Date(a.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })].filter(Boolean).join(' · ')}</span></span>
                    {busy === a.runId ? <Loader2 size={16} aria-hidden className="animate-spin motion-reduce:animate-none" /> : <ClassBadge value={a.predictedClass as MovieClass} />}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="rounded-2xl border border-dashed border-line px-4 py-6 text-center text-[0.86rem] text-fg3" data-testid="picker-empty">
              No analysis has been completed in this browser tab yet. Run one on the AI Movie Researcher page, or paste an analysis id below.
            </p>
          )}

          <form onSubmit={submit} className="mt-5 border-t border-line pt-4">
            <label htmlFor={`${tid}-id`} className="mb-1.5 block text-[0.82rem] font-medium text-fg">Or paste an analysis id</label>
            <div className="flex gap-2">
              <input id={`${tid}-id`} value={pasted} onChange={(e) => { setPasted(e.target.value); setError(null) }} autoComplete="off" spellCheck={false} placeholder="12 or 32 hexadecimal characters" data-testid="picker-id"
                aria-invalid={!!pasted && !validPasted} aria-describedby={`${tid}-h`} className={cx('min-h-11 min-w-0 flex-1 rounded-2xl border bg-black/30 px-4 font-mono text-sm text-fg placeholder:text-fg3/70 focus:border-cyan focus:outline-none', pasted && !validPasted ? 'border-flop' : 'border-line-strong')} />
              <button type="submit" disabled={!validPasted || !!busy} data-testid="picker-use" className="inline-flex min-h-11 items-center rounded-full border border-line-strong px-4 text-sm font-semibold text-fg hover:border-white/50 disabled:opacity-50">Use</button>
            </div>
            <p id={`${tid}-h`} className="mt-1.5 text-[0.74rem] text-fg3">{pasted && !validPasted ? 'That is not a valid analysis id.' : 'The server checks that it exists and has finished.'}</p>
          </form>
          {error && <p role="alert" data-testid="picker-error" className="mt-3 rounded-2xl border border-flop/40 bg-flop/10 px-4 py-2.5 text-[0.82rem] text-fg2">{error}</p>}
        </div>
      </div>
    </div>
  )
}
