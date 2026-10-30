import { Eraser, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Composer, ConfirmDialog, ConversationRail, EmptyChat, ErrorBanner, MessageBubble, Thinking } from '../components/assistant/Chat'
import { MovieChip, MovieEmptyChat, MoviePicker, OfferBanner } from '../components/assistant/Movie'
import { Disclosure, ErrorState, Note, PageHeader } from '../components/ui'
import { parseSummary, RUN_ID_RE } from '../lib/analyses'
import { api } from '../lib/api'
import { activeOf, cooldownMs, MAX_CONVERSATIONS, titleOf, type MovieRef } from '../lib/chat'
import { analysisRegistry, useAnalyses } from '../lib/useAnalyses'
import { chatStore, useChatState, useNow } from '../lib/useChat'
import { useApi } from '../lib/useApi'

interface ChatStatus { configured: boolean; model: string | null; missing: string[]; limits: { max_input_chars: number } }

/** Ask the server whether an analysis exists and is finished, and get its safe summary. Throws the API error otherwise. */
async function lookupAnalysis(runId: string): Promise<MovieRef> {
  const ref = parseSummary(await api.get<unknown>(`/api/chat/movie/${runId}`))
  if (!ref) throw Object.assign(new Error('unreadable'), { status: 502, code: 'bad_response' })
  return ref
}

export default function Assistant() {
  const store = chatStore()
  const state = useChatState()
  const status = useApi<ChatStatus>('/api/chat/status')
  const analyses = useAnalyses()
  const location = useLocation()
  const navigate = useNavigate()
  const conv = activeOf(state)
  const [confirm, setConfirm] = useState<'clear' | 'delete' | null>(null)
  const [picker, setPicker] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const log = useRef<HTMLOListElement>(null)

  // Availability informs the banner, but the server response handles failures when a message is sent.
  useEffect(() => {
    if (status.data) store.configure({ maxChars: status.data.limits.max_input_chars, available: status.data.configured })
    else if (status.error) store.configure({ available: status.error.status === 404 ? false : null })
  }, [status.data, status.error, store])

  const isPending = !!conv.pending
  const now = useNow(!!conv.error?.until)
  const waitMs = cooldownMs(conv, now)
  const unavailable = state.limits.available === false
  const focusInput = () => requestAnimationFrame(() => input.current?.focus())

  // keep the newest message in view
  useEffect(() => {
    const el = log.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })
  }, [conv.id, conv.messages.length, isPending])

  /* ---- hand-off from the Researcher: an OFFER, never an automatic switch */
  const offeredId = (location.state as { runId?: unknown } | null)?.runId
  const offerId = typeof offeredId === 'string' && RUN_ID_RE.test(offeredId) ? offeredId : null
  const [fetched, setFetched] = useState<MovieRef | null>(null)
  const offer = offerId ? analyses.find((a) => a.runId === offerId) ?? (fetched?.runId === offerId ? fetched : null) : null
  useEffect(() => {
    if (!offerId || analyses.some((a) => a.runId === offerId)) return
    let live = true                                          // not in this tab's list (e.g. after a refresh): ask the server, which also confirms it finished
    lookupAnalysis(offerId).then((r) => { if (live) setFetched(r) }, () => {})
    return () => { live = false }
  }, [offerId, analyses])
  const clearOffer = useCallback(() => navigate(location.pathname, { replace: true, state: null }), [navigate, location.pathname])

  /** Start (or reopen) the conversation about one analysis. A different film always gets a NEW conversation. */
  const startMovie = (ref: MovieRef) => {
    analysisRegistry().add(ref)
    const existing = state.conversations.find((c) => c.movie?.runId === ref.runId)
    if (existing) store.select(existing.id)
    else if (!store.newConversation(ref)) { setNotice('You have the maximum number of conversations. Delete one first.'); return }
    setNotice(null); setPicker(false); clearOffer(); focusInput()
  }

  const send = (text?: string) => { store.send(conv.id, text); focusInput() }
  const doConfirm = () => {
    if (confirm === 'clear') store.clear(conv.id)
    if (confirm === 'delete') store.remove(conv.id)
    setConfirm(null); focusInput()
  }
  const canAdd = state.conversations.length < MAX_CONVERSATIONS
  const mode = conv.movie ? 'Movie analysis' : 'General mode'

  return (
    <>
      <PageHeader eyebrow="AI Assistant" title="AI Assistant" highlight="Assistant">
        A cinema and data-science assistant powered by Google Gemini. In <b className="font-semibold text-fg">general</b> mode it answers general questions and does not see your films.
        In <b className="font-semibold text-fg">movie analysis</b> mode it explains one stored analysis from the AI Movie Researcher: it never makes or changes a prediction.
      </PageHeader>

      {offer && <OfferBanner movie={offer} existing={state.conversations.some((c) => c.movie?.runId === offer.runId)} onStart={() => startMovie(offer)}
        onOpen={() => startMovie(offer)} onDismiss={clearOffer} />}

      {unavailable && (
        <div className="mb-5" data-testid="assistant-unavailable">
          <Note tone="warn" title={status.error?.status === 404 ? 'The AI assistant is not available on this server' : 'The AI assistant is not set up yet'}>
            <p>You can still write a message. Sending will show an error until the assistant is available. Everything else in FRAME keeps working.</p>
            <Disclosure title="For the administrator">
              <p>{status.data?.missing?.length ? <>Missing settings: <span className="font-mono text-fg">{status.data.missing.join(', ')}</span>. </> : null}
                Set the Gemini key and model in the backend’s <span className="font-mono text-fg">.env</span> file (see <span className="font-mono text-fg">.env.example</span>), then restart the backend. The key never reaches the browser.</p>
            </Disclosure>
          </Note>
        </div>
      )}
      {status.error && status.error.status !== 404 && (
        <div className="mb-5"><ErrorState title="Could not check the assistant" message="The status check failed. You can still try sending a message." onRetry={status.retry} /></div>
      )}
      {notice && <p role="alert" data-testid="page-notice" className="mb-4 rounded-2xl border border-avg/40 bg-avg/10 px-4 py-2.5 text-[0.84rem] text-fg2">{notice}</p>}

      <div className="grid items-start gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <ConversationRail state={state} onSelect={(id) => { store.select(id); focusInput() }} canAdd={canAdd}
          onNew={() => { if (store.newConversation()) focusInput() }} onNewMovie={() => setPicker(true)} />

        <section aria-label="Chat" className="glass flex h-[clamp(28rem,calc(100dvh-21rem),52rem)] min-w-0 flex-col overflow-hidden rounded-[1.75rem]" data-testid="chat-panel" data-conversation-id={conv.id} data-mode={conv.movie ? 'movie' : 'general'}>
          <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3 sm:px-5">
            <div className="min-w-0">
              <h2 className="truncate font-display text-[1.02rem] font-bold text-fg" data-testid="conversation-title">{titleOf(conv)}</h2>
              <p className="font-mono text-[0.7rem] text-fg3" data-testid="conversation-mode">{mode}{status.data?.model ? ` · ${status.data.model}` : ''}</p>
            </div>
            <div className="flex items-center gap-1.5">
              <button type="button" onClick={() => setConfirm('clear')} disabled={!conv.messages.length && !conv.pending} data-testid="clear"
                className="inline-flex min-h-10 items-center gap-1.5 rounded-full border border-line-strong px-3.5 text-[0.8rem] font-semibold text-fg2 hover:border-white/50 hover:text-fg disabled:opacity-50">
                <Eraser size={15} aria-hidden /><span className="hidden sm:inline">Clear</span><span className="sr-only sm:hidden">Clear conversation</span>
              </button>
              {state.conversations.length > 1 && (
                <button type="button" onClick={() => setConfirm('delete')} aria-label="Delete conversation" title="Delete conversation" data-testid="delete"
                  className="flex h-10 w-10 items-center justify-center rounded-full border border-line-strong text-fg2 hover:border-flop/60 hover:text-danger"><Trash2 size={15} aria-hidden /></button>
              )}
            </div>
          </header>
          {conv.movie && <MovieChip movie={conv.movie} onChange={() => setPicker(true)} disabled={false} />}

          {conv.messages.length === 0 && !conv.pending ? (
            <div className="min-h-0 flex-1 overflow-y-auto">
              {conv.movie
                ? <MovieEmptyChat movie={conv.movie} onPick={(p) => send(p)} disabled={unavailable || waitMs > 0} />
                : <EmptyChat onPick={(p) => send(p)} disabled={unavailable || !!conv.pending || waitMs > 0} />}
            </div>
          ) : (
            <ol ref={log} role="log" aria-live="polite" aria-relevant="additions" aria-label="Conversation messages" tabIndex={0} data-testid="message-log"
              className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-5 sm:px-5">
              {conv.messages.map((m) => <MessageBubble key={m.id} m={m} />)}
              {conv.pending && <Thinking />}
            </ol>
          )}

          {conv.error && <ErrorBanner error={conv.error} waitMs={waitMs} onRetry={() => { store.retry(conv.id); focusInput() }} onDismiss={() => store.dismissError(conv.id)} />}
          <Composer conv={conv} maxChars={state.limits.maxChars} waitMs={waitMs} inputRef={input} placeholder={conv.movie ? 'Ask about this analysis…' : undefined}
            onChange={(v) => store.setDraft(conv.id, v)} onSend={() => send()} />
        </section>
      </div>

      {picker && <MoviePicker analyses={analyses} lookup={lookupAnalysis} onPick={startMovie} onClose={() => setPicker(false)} />}
      {confirm && (
        <ConfirmDialog
          title={confirm === 'clear' ? 'Clear this conversation?' : 'Delete this conversation?'}
          body={confirm === 'clear' ? 'Every message in this conversation will be removed from this page. This cannot be undone.' : 'This conversation and its messages will be removed from this page. This cannot be undone.'}
          confirmLabel={confirm === 'clear' ? 'Clear messages' : 'Delete'} onConfirm={doConfirm} onCancel={() => setConfirm(null)} />
      )}
    </>
  )
}
