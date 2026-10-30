import { AlertTriangle, Bot, Clapperboard, Film, MessageSquarePlus, RotateCw, SendHorizontal, Sparkles, Trash2, User, X } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect, useId, useLayoutEffect, useRef, type KeyboardEvent, type RefObject } from 'react'
import { STARTER_PROMPTS, titleOf, type ChatError, type ChatMessage, type ChatState, type Conversation } from '../../lib/chat'
import { Button, cx } from '../ui'
import { Markdown } from './Markdown'

/** whole seconds for the countdown: rounded (the clock ticks every 500 ms) and never 0 while sending is still blocked */
const secs = (ms: number) => Math.max(1, Math.round(ms / 1000))
const time = (at: number) => new Date(at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

/* ---------------------------------------------------------------- one message */
export function MessageBubble({ m }: { m: ChatMessage }) {
  const mine = m.role === 'user'
  return (
    <motion.li initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }} data-testid="message" data-role={m.role}
      className={cx('flex items-end gap-2.5', mine && 'flex-row-reverse')}>
      <span aria-hidden className={cx('mb-5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border', mine ? 'border-pink/50 bg-pink/15 text-pink' : 'border-cyan/50 bg-cyan/10 text-cyan')}>
        {mine ? <User size={15} /> : <Bot size={15} />}
      </span>
      <div className={cx('flex max-w-[86%] flex-col sm:max-w-[78%]', mine && 'items-end')}>
        <div className={cx('rounded-[1.25rem] px-4 py-3 text-[0.92rem] leading-relaxed',
          mine ? 'rounded-br-md bg-[image:var(--grad-brand)] font-medium text-[#12082a]' : 'glass rounded-bl-md text-fg2')}>
          <span className="sr-only">{mine ? 'You said: ' : 'Assistant said: '}</span>
          {mine ? <p className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">{m.text}</p> : <Markdown text={m.text} />}
        </div>
        <p className="mt-1 flex items-center gap-2 px-1 font-mono text-[0.7rem] text-fg3">
          <time dateTime={new Date(m.at).toISOString()}>{time(m.at)}</time>
          {m.failed && <span className="text-danger" data-testid="not-delivered">Not answered</span>}
          {m.truncated && <span className="text-avg" data-testid="truncated">Cut short by the length limit</span>}
        </p>
      </div>
    </motion.li>
  )
}

export function Thinking() {
  return (
    <li className="flex items-end gap-2.5" data-testid="thinking">
      <span aria-hidden className="mb-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-cyan/50 bg-cyan/10 text-cyan"><Bot size={15} /></span>
      <div role="status" className="glass flex items-center gap-1.5 rounded-[1.25rem] rounded-bl-md px-4 py-3.5">
        <span className="sr-only">The assistant is thinking…</span>
        {[0, 1, 2].map((i) => <span key={i} aria-hidden className="h-2 w-2 animate-pulse rounded-full bg-cyan motion-reduce:animate-none" style={{ animationDelay: `${i * 160}ms` }} />)}
      </div>
    </li>
  )
}

/* ---------------------------------------------------------------- empty state with starter prompts */
export function EmptyChat({ onPick, disabled }: { onPick: (p: string) => void; disabled: boolean }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-5 px-4 py-8 text-center" data-testid="empty-chat">
      <span aria-hidden className="flex h-14 w-14 items-center justify-center rounded-full bg-[image:var(--grad-brand)] text-[#12082a] shadow-[0_10px_34px_-8px_rgb(255_79_163/.8)]"><Sparkles size={24} /></span>
      <div>
        <h2 className="font-display text-xl font-bold text-fg">Ask about cinema, data and movie success</h2>
        <p className="mx-auto mt-1.5 max-w-md text-[0.88rem] text-fg3">Pick a starter or type your own question below.</p>
      </div>
      <ul className="flex max-w-2xl flex-wrap justify-center gap-2" aria-label="Suggested prompts">
        {STARTER_PROMPTS.map((p) => (
          <li key={p}>
            <button type="button" disabled={disabled} onClick={() => onPick(p)} data-testid="starter"
              className="min-h-11 rounded-full border border-line-strong bg-white/[0.04] px-4 py-2 text-left text-[0.84rem] font-medium text-fg2 transition-all duration-200 hover:-translate-y-0.5 hover:border-white/50 hover:bg-white/[0.09] hover:text-fg active:scale-95 disabled:opacity-50 disabled:hover:translate-y-0">
              {p}
            </button>
          </li>
        ))}
      </ul>
      <p className="max-w-md text-[0.76rem] leading-snug text-fg3">
        General assistant: it answers from Google Gemini and can be wrong. It cannot see your films, predictions or database, and its answers are not model predictions.
      </p>
    </div>
  )
}

/* ---------------------------------------------------------------- error banner with retry / countdown */
const ERROR_TITLE: Record<string, string> = {
  validation: 'That message can’t be sent', rate_limited: 'Slow down a moment', busy: 'The assistant is busy', not_configured: 'Assistant not set up',
  unavailable: 'Assistant unavailable', run_unavailable: 'Analysis unavailable', timeout: 'No answer in time', server: 'The assistant had a problem', declined: 'The assistant declined', network: 'You appear to be offline', unknown: 'Something went wrong',
}
export function ErrorBanner({ error, waitMs, onRetry, onDismiss }: { error: ChatError; waitMs: number; onRetry: () => void; onDismiss: () => void }) {
  const waiting = waitMs > 0
  return (
    <div role="alert" data-testid="chat-error" data-kind={error.kind} className="mx-4 mb-3 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-2xl border border-flop/40 bg-flop/10 px-4 py-3">
      <AlertTriangle size={17} aria-hidden className="shrink-0 text-danger" />
      <div className="min-w-0 flex-1 basis-56">
        <p className="text-[0.88rem] font-semibold text-fg">{ERROR_TITLE[error.kind] ?? ERROR_TITLE.unknown}</p>
        <p className="text-[0.82rem] text-fg2">{error.message}{waiting && <> <span data-testid="cooldown">You can try again in {secs(waitMs)}s.</span></>}</p>
      </div>
      {error.retryable && (
        <Button variant="ghost" icon={<RotateCw size={14} aria-hidden />} onClick={onRetry} disabled={waiting} data-testid="retry" className="!min-h-10">
          {waiting ? `Retry in ${secs(waitMs)}s` : 'Retry'}
        </Button>
      )}
      <button type="button" onClick={onDismiss} aria-label="Dismiss error" className="flex h-10 w-10 items-center justify-center rounded-full text-fg3 hover:bg-white/10 hover:text-fg"><X size={16} aria-hidden /></button>
    </div>
  )
}

/* ---------------------------------------------------------------- composer */
export function Composer({ conv, maxChars, waitMs, onChange, onSend, inputRef, placeholder }: {
  conv: Conversation; maxChars: number; waitMs: number; placeholder?: string
  onChange: (v: string) => void; onSend: () => void; inputRef: RefObject<HTMLTextAreaElement | null>
}) {
  const id = useId()
  const len = conv.draft.trim().length
  const over = len > maxChars
  const pending = !!conv.pending
  const canSend = !pending && waitMs === 0 && len > 0 && !over

  // grow with the text, up to about six lines
  useLayoutEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`
  }, [conv.draft, inputRef])

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (canSend) onSend() }
  }
  return (
    <form onSubmit={(e) => { e.preventDefault(); if (canSend) onSend() }} className="border-t border-line p-3 sm:p-4" aria-label="Send a message">
      <div className="flex items-end gap-2.5">
        <div className="min-w-0 flex-1">
          <label htmlFor={id} className="sr-only">Message to the assistant</label>
          <textarea id={id} ref={inputRef} rows={1} value={conv.draft} onChange={(e) => onChange(e.target.value)} onKeyDown={onKey}
            placeholder={placeholder ?? 'Ask a question…'} aria-describedby={`${id}-hint`} aria-invalid={over || undefined} data-testid="composer"
            className={cx('block max-h-[168px] min-h-12 w-full resize-none rounded-2xl border bg-black/30 px-4 py-3 text-[0.92rem] text-fg placeholder:text-fg3/70 hover:border-white/40 focus:border-cyan focus:outline-none focus:shadow-[0_0_0_4px_rgb(37_232_211/.16)] disabled:opacity-60', over ? 'border-flop' : 'border-line-strong')} />
        </div>
        <Button type="submit" variant="primary" disabled={!canSend} aria-label="Send message" data-testid="send" className="!h-12 !min-h-12 w-12 shrink-0 !px-0 sm:w-auto sm:!px-5" icon={<SendHorizontal size={17} aria-hidden />}>
          <span className="hidden sm:inline">Send</span>
        </Button>
      </div>
      <p id={`${id}-hint`} className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 px-1 text-[0.72rem] text-fg3">
        <span>{waitMs > 0 ? `You can send again in ${secs(waitMs)}s.` : pending ? 'Waiting for the answer…' : 'Enter to send · Shift+Enter for a new line'}</span>
        <span data-testid="counter" className={cx('font-mono tnum', over && 'font-semibold text-danger')} aria-live={over ? 'polite' : 'off'}>
          {len > maxChars * 0.8 || over ? `${len.toLocaleString('en-US')} / ${maxChars.toLocaleString('en-US')}${over ? ' — too long' : ''}` : ''}
        </span>
      </p>
    </form>
  )
}

/* ---------------------------------------------------------------- conversation switcher: tabs on small screens, a list on large ones */
export function ConversationRail({ state, onSelect, onNew, onNewMovie, canAdd }: { state: ChatState; onSelect: (id: string) => void; onNew: () => void; onNewMovie: () => void; canAdd: boolean }) {
  return (
    <nav aria-label="Conversations" className="min-w-0">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 lg:mb-3 lg:flex-col lg:items-stretch">
        <h2 className="eyebrow">Conversations</h2>
        <div className="flex gap-1.5 lg:grid lg:grid-cols-2">
          <button type="button" onClick={onNew} disabled={!canAdd} title={canAdd ? 'Start a new general conversation' : 'Delete a conversation to start another'} data-testid="new-conversation"
            className="inline-flex min-h-10 items-center justify-center gap-1.5 rounded-full border border-line-strong px-3.5 text-[0.8rem] font-semibold text-fg2 hover:border-white/50 hover:text-fg disabled:opacity-50">
            <MessageSquarePlus size={15} aria-hidden />New
          </button>
          <button type="button" onClick={onNewMovie} disabled={!canAdd} title={canAdd ? 'Discuss a completed movie analysis' : 'Delete a conversation to start another'} data-testid="new-movie-conversation"
            className="inline-flex min-h-10 items-center justify-center gap-1.5 rounded-full border border-cyan/50 bg-cyan/10 px-3.5 text-[0.8rem] font-semibold text-cyan hover:bg-cyan/20 disabled:opacity-50">
            <Clapperboard size={15} aria-hidden />Movie
          </button>
        </div>
      </div>
      <ul className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-2 lg:mx-0 lg:flex-col lg:overflow-visible lg:px-0 lg:pb-0" data-testid="conversation-list">
        {state.conversations.map((c) => {
          const on = c.id === state.activeId
          return (
            <li key={c.id} className="shrink-0 lg:shrink">
              <button type="button" onClick={() => onSelect(c.id)} aria-current={on ? 'true' : undefined} data-testid="conversation" data-conversation-id={c.id}
                className={cx('flex min-h-11 w-full max-w-[14rem] items-center gap-2 rounded-2xl border px-3.5 py-2 text-left text-[0.83rem] transition-colors lg:max-w-none',
                  on ? 'border-transparent bg-[image:var(--grad-brand)] font-semibold text-[#12082a]' : 'border-line bg-white/[0.03] text-fg2 hover:border-white/40 hover:text-fg')}>
                {c.movie && <Film size={14} aria-label="Movie analysis" role="img" className="shrink-0" />}
                <span className="min-w-0 flex-1 truncate">{titleOf(c)}</span>
                {c.pending && <span aria-label="Waiting for an answer" className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-cyan motion-reduce:animate-none" />}
                {!!c.messages.length && <span className={cx('shrink-0 font-mono text-[0.7rem]', on ? 'text-[#12082a]/70' : 'text-fg3')}>{c.messages.length}</span>}
              </button>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

/* ---------------------------------------------------------------- confirmation dialog (focus is trapped; Escape and Cancel close it) */
export function ConfirmDialog({ title, body, confirmLabel, onConfirm, onCancel }: { title: string; body: string; confirmLabel: string; onConfirm: () => void; onCancel: () => void }) {
  const tid = useId()
  const cancel = useRef<HTMLButtonElement>(null)
  const ok = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    cancel.current?.focus()
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onCancel() }
      if (e.key === 'Tab') {                                   // two controls: keep focus between them
        const a = document.activeElement
        if (e.shiftKey && a === cancel.current) { e.preventDefault(); ok.current?.focus() }
        else if (!e.shiftKey && a === ok.current) { e.preventDefault(); cancel.current?.focus() }
      }
    }
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('keydown', onKey); opener?.focus?.() }
  }, [onCancel])
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/75 backdrop-blur-sm" onClick={onCancel} aria-hidden />
      <div role="alertdialog" aria-modal="true" aria-labelledby={tid} aria-describedby={`${tid}-d`} className="glass relative w-full max-w-sm rounded-[1.5rem] p-6 shadow-2xl">
        <h2 id={tid} className="font-display text-lg font-bold text-fg">{title}</h2>
        <p id={`${tid}-d`} className="mt-2 text-[0.88rem] text-fg2">{body}</p>
        <div className="mt-5 flex justify-end gap-2.5">
          <button ref={cancel} type="button" onClick={onCancel} data-testid="confirm-cancel" className="inline-flex min-h-11 items-center justify-center rounded-full border border-line-strong bg-white/[0.04] px-5 text-sm font-semibold text-fg hover:border-white/40 hover:bg-white/[0.09] sm:min-h-10">Cancel</button>
          <button ref={ok} type="button" onClick={onConfirm} data-testid="confirm-ok" className="inline-flex min-h-11 items-center justify-center gap-2 rounded-full border border-flop/60 bg-flop/20 px-5 text-sm font-semibold text-fg hover:bg-flop/35 sm:min-h-10">
            <Trash2 size={15} aria-hidden />{confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
