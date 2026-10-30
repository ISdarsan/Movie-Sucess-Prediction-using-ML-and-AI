/* In-memory chat store for the AI Assistant. Pure TypeScript with no runtime imports, so Node can unit-test it (frontend/tests/chat.test.ts).
   Talks to the EXISTING backend only: POST /api/chat -> {reply, mode, provider, model, truncated, context?}.
     general conversation:  {message, mode:"general"}                      only the current message is sent (unchanged since Phase 2)
     movie conversation:    {message, mode:"movie", run_id, history}       the server resolves the film's context from run_id; the browser sends
                                                                          NO prediction, score or film data, only the id and the last few turns

   A movie conversation is bound to ONE analysis for its whole life (`movie` is set at creation and never changes). Discussing a different
   film always means a NEW conversation, so two films can never share a thread or context.

   Privacy: conversations live in memory only. Nothing is written to localStorage, sessionStorage, cookies or the server's database, and no
   credential exists in the browser.
   Safety rules enforced here, not just in the UI:
     * a conversation with a request in flight cannot send again (no duplicate submissions)
     * a response is applied to the conversation that asked, and only if that exact request is still pending (clearing/deleting drops it)
     * a rate-limit error blocks sending until its Retry-After has elapsed
     * error text shown to people is this file's own wording; raw provider or server text is never displayed (except a 422 field message,
       which the server writes and validates). */

export type Role = 'user' | 'assistant'
export interface ChatMessage { id: string; role: Role; text: string; at: number; truncated?: boolean; failed?: boolean }

export type ErrorKind = 'validation' | 'rate_limited' | 'busy' | 'not_configured' | 'unavailable' | 'run_unavailable' | 'timeout' | 'server' | 'declined' | 'network' | 'unknown'
export interface ChatError { kind: ErrorKind; message: string; retryable: boolean; until: number | null; at: number }

/** The analysis a movie conversation is about. Display data only: the server re-reads everything from the stored run. */
export interface MovieRef { runId: string; title: string | null; predictedClass: string; year: number | null; genres: string | null }

export interface Conversation {
  id: string
  createdAt: number
  /** null = general conversation. Fixed at creation. */
  movie: MovieRef | null
  messages: ChatMessage[]
  pending: { requestId: string; since: number } | null
  error: ChatError | null
  draft: string
}
export interface Limits { maxChars: number; available: boolean | null }
export interface ChatState { conversations: Conversation[]; activeId: string; limits: Limits }

export const STARTER_PROMPTS = [
  'Explain how Random Forest works.',
  'What makes a movie commercially successful?',
  'Explain the difference between Hit, Average and Flop.',
  'How does our Movie Intelligence System work?',
  'What factors affect movie performance?',
]
export const MOVIE_STARTER_PROMPTS = [
  'Why did the model predict this class?',
  'Explain the prediction in simple language.',
  'What are the biggest risks?',
  'What did the research agent discover?',
  'Explain the comparable movies.',
  'What are the limitations of this prediction?',
  'Summarize the entire analysis.',
]
export const MAX_HISTORY_SENT = 6          // earlier messages sent with a movie question (the server bounds them again)
export const HISTORY_TURN_CHARS = 1500
export const MAX_CONVERSATIONS = 20
export const MAX_MESSAGES = 200
export const DEFAULT_MAX_CHARS = 2000
export const REQUEST_TIMEOUT_MS = 45_000

export type SendResult = 'sent' | 'empty' | 'too_long' | 'pending' | 'cooldown' | 'full' | 'unknown_conversation'

/* ------------------------------------------------------------------ wording (ours, not the provider's) */
export const ERROR_TEXT: Record<Exclude<ErrorKind, 'validation'>, string> = {
  rate_limited: 'The assistant is getting a lot of requests right now.',
  busy: 'The assistant is handling other requests.',
  not_configured: 'The AI assistant is not set up on this server yet.',
  unavailable: 'The assistant is temporarily unavailable. The administrator may need to check its configuration.',
  run_unavailable: 'That analysis cannot be discussed right now. Start a new conversation with another analysis.',
  timeout: 'The assistant took too long to answer.',
  server: 'The assistant had a problem answering. Nothing was lost.',
  declined: 'The assistant declined to answer that. Try rephrasing your question.',
  network: 'Could not reach the server. Check your connection.',
  unknown: 'Something went wrong, and the message was not answered.',
}

interface ApiErrorLike { status?: unknown; code?: unknown; message?: unknown; fields?: unknown; retryAfter?: unknown }
const isRec = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

function retrySeconds(e: ApiErrorLike, fallback: number): number {
  let s = typeof e.retryAfter === 'number' && Number.isFinite(e.retryAfter) ? e.retryAfter : NaN
  if (!(s > 0) && typeof e.message === 'string') {            // header unreadable (e.g. cross-origin): the server also says it in words
    const m = /(\d{1,4})\s*seconds?/i.exec(e.message)
    if (m) s = Number(m[1])
  }
  if (!(s > 0)) s = fallback
  return Math.min(300, Math.max(1, Math.round(s)))
}

/** Turn whatever the API client threw into a safe, typed error. */
export function mapError(err: unknown, now: number): ChatError {
  const e: ApiErrorLike = isRec(err) || err instanceof Error ? (err as ApiErrorLike) : {}
  const status = typeof e.status === 'number' ? e.status : 0
  const code = typeof e.code === 'string' ? e.code : ''
  const mk = (kind: ErrorKind, retryable: boolean, message?: string, waitS?: number): ChatError =>
    ({ kind, retryable, message: message ?? ERROR_TEXT[kind as Exclude<ErrorKind, 'validation'>], at: now, until: waitS ? now + waitS * 1000 : null })

  if (code === 'assistant_declined') return mk('declined', false)
  if (code === 'assistant_not_configured') return mk('not_configured', false)
  if (code === 'run_not_found') return mk('run_unavailable', false, 'That analysis was not found on the server. Run it again, then start a new conversation about it.')
  if (code === 'run_failed' || code === 'run_incomplete') return mk('run_unavailable', false, 'That analysis did not produce a prediction, so there is nothing to discuss. Run it again.')
  if (code === 'run_not_ready') return mk('run_unavailable', true, 'That analysis has not finished yet. Wait for it to complete, then retry.')
  if (code === 'context_unavailable') return mk('run_unavailable', true, 'The saved analyses could not be read right now. Please retry in a moment.')
  if (status === 0 || code === 'network') return mk('network', true)
  if (status === 422 || code === 'validation_error') {
    const f = Array.isArray(e.fields) && isRec(e.fields[0]) && typeof e.fields[0].message === 'string' ? (e.fields[0].message as string) : ''
    return mk('validation', false, f && f.length < 200 ? f : "That message can't be sent. Try changing it.")
  }
  if (status === 429 || code === 'rate_limited' || code === 'assistant_rate_limited' || code === 'busy') {
    const busy = code === 'busy'
    const wait = retrySeconds(e, busy ? 5 : 30)
    return mk(busy ? 'busy' : 'rate_limited', true, undefined, wait)
  }
  if (status === 504 || code === 'assistant_timeout') return mk('timeout', true)
  if (status === 503 || code === 'assistant_misconfigured' || code === 'assistant_model_unavailable' || status === 404) return mk('unavailable', true)
  if (status >= 500) return mk('server', true)
  return mk('unknown', true)
}

/* ------------------------------------------------------------------ store */
export interface ChatDeps {
  post: (path: string, body: unknown) => Promise<unknown>
  now: () => number
  newId: () => string
  /** resolves after ms; used for the client-side request timeout */
  sleep: (ms: number) => Promise<void>
  timeoutMs?: number
}

const blank = (id: string, now: number, movie: MovieRef | null = null): Conversation => ({ id, createdAt: now, movie, messages: [], pending: null, error: null, draft: '' })

/** What to call a film in the UI: its title if the person gave one, otherwise its traits. Never invented. */
export const movieName = (m: MovieRef) => m.title ?? ([m.genres, m.year].filter(Boolean).join(' · ') || 'Untitled film')

/** Short label for a film, e.g. "Inception · Hit" or "Action / Sci-Fi · 2010 · Hit" when no title was given. */
export function movieLabel(m: MovieRef): string {
  const head = (m.title ?? [m.genres, m.year].filter(Boolean).join(' · ')) || 'Untitled film'
  return `${head} · ${m.predictedClass}`
}

export function titleOf(c: Conversation): string {
  if (c.movie) return movieLabel(c.movie)
  const first = c.messages.find((m) => m.role === 'user')?.text.replace(/\s+/g, ' ').trim()
  return first ? (first.length > 38 ? first.slice(0, 37).trimEnd() + '…' : first) : 'New conversation'
}

/** The exact request for one message. General conversations send ONLY the message; movie conversations send the run id and the recent
 *  answered turns that came BEFORE this message. Nothing about the film itself is ever sent from the browser. */
export function bodyFor(c: Conversation, userMsgId: string, text: string): Record<string, unknown> {
  if (!c.movie) return { message: text, mode: 'general' }
  const upTo = c.messages.findIndex((m) => m.id === userMsgId)
  const earlier = upTo < 0 ? c.messages : c.messages.slice(0, upTo)
  // only complete exchanges are history: a user message that never got an answer (it failed, or was abandoned) is not replayed
  const before = earlier.filter((m, i) => m.role === 'assistant' || (!m.failed && earlier[i + 1]?.role === 'assistant'))
  const history = before.slice(-MAX_HISTORY_SENT).map((m) => ({ role: m.role, text: m.text.slice(0, HISTORY_TURN_CHARS) }))
  return { message: text, mode: 'movie', run_id: c.movie.runId, history }
}

export function createChatStore(deps: ChatDeps) {
  const first = blank(deps.newId(), deps.now())
  let state: ChatState = { conversations: [first], activeId: first.id, limits: { maxChars: DEFAULT_MAX_CHARS, available: null } }
  const listeners = new Set<() => void>()
  const set = (next: ChatState) => { state = next; listeners.forEach((l) => l()) }
  const find = (id: string) => state.conversations.find((c) => c.id === id)
  const patch = (id: string, fn: (c: Conversation) => Conversation) => set({ ...state, conversations: state.conversations.map((c) => (c.id === id ? fn(c) : c)) })

  /** Start the network request for a user message that is already in the conversation. */
  function dispatch(convId: string, requestId: string, userMsgId: string, text: string) {
    const guard = (): Conversation | null => {
      const c = find(convId)
      return c && c.pending?.requestId === requestId ? c : null    // cleared, deleted or superseded meanwhile: the answer is dropped
    }
    const timeout = deps.sleep(deps.timeoutMs ?? REQUEST_TIMEOUT_MS).then(() => { throw { status: 504, code: 'assistant_timeout' } })
    const conv = find(convId)
    if (!conv) return
    Promise.race([deps.post('/api/chat', bodyFor(conv, userMsgId, text)), timeout]).then(
      (res) => {
        if (!guard()) return
        const ok = isRec(res) && typeof res.reply === 'string' && res.reply.trim() !== ''
        if (!ok) { fail(convId, requestId, userMsgId, text, { status: 502, code: 'bad_response' }); return }
        const r = res as Record<string, unknown>
        patch(convId, (c) => ({ ...c, pending: null, error: null, messages: [...c.messages, { id: deps.newId(), role: 'assistant', text: String(r.reply).trim(), at: deps.now(), truncated: r.truncated === true }] }))
      },
      (err) => fail(convId, requestId, userMsgId, text, err),
    )
  }

  function fail(convId: string, requestId: string, userMsgId: string, text: string, err: unknown) {
    const c = find(convId)
    if (!c || c.pending?.requestId !== requestId) return
    const e = mapError(err, deps.now())
    if (e.kind === 'validation') {              // the server rejected the text itself: take it back out of the thread and into the box to edit
      patch(convId, (x) => ({ ...x, pending: null, error: e, draft: text, messages: x.messages.filter((m) => m.id !== userMsgId) }))
    } else {
      patch(convId, (x) => ({ ...x, pending: null, error: e, messages: x.messages.map((m) => (m.id === userMsgId ? { ...m, failed: e.retryable } : m)) }))
    }
  }

  return {
    getState: () => state,
    subscribe(fn: () => void) { listeners.add(fn); return () => { listeners.delete(fn) } },

    /** Tell the store what the status endpoint said (null = not known yet). */
    configure(limits: Partial<Limits>) {
      const next = { ...state.limits, ...limits }
      if (next.maxChars !== state.limits.maxChars || next.available !== state.limits.available) set({ ...state, limits: next })
    },

    /** Start an empty conversation. Pass a film to make it a movie conversation (fixed for its whole life). */
    newConversation(movie: MovieRef | null = null): string | null {
      if (state.conversations.length >= MAX_CONVERSATIONS) return null
      const c = blank(deps.newId(), deps.now(), movie)
      set({ ...state, conversations: [c, ...state.conversations], activeId: c.id })
      return c.id
    },
    select(id: string) { if (find(id) && state.activeId !== id) set({ ...state, activeId: id }) },
    setDraft(id: string, text: string) { if (find(id)) patch(id, (c) => ({ ...c, draft: text })) },
    dismissError(id: string) { patch(id, (c) => ({ ...c, error: null })) },

    /** Remove every message of a conversation and drop any answer still on its way. The conversation itself stays. */
    clear(id: string) { if (find(id)) patch(id, (c) => ({ ...c, messages: [], pending: null, error: null, draft: '' })) },
    remove(id: string) {
      if (!find(id)) return
      let rest = state.conversations.filter((c) => c.id !== id)
      let active = state.activeId
      if (!rest.length) { const c = blank(deps.newId(), deps.now()); rest = [c] }
      if (active === id || !rest.some((c) => c.id === active)) active = rest[0].id
      set({ ...state, conversations: rest, activeId: active })
    },

    /** Send `text` (or the conversation's draft). Returns why nothing was sent, if nothing was. */
    send(id: string, text?: string): SendResult {
      const c = find(id)
      if (!c) return 'unknown_conversation'
      if (c.pending) return 'pending'
      if (c.error?.until && deps.now() < c.error.until) return 'cooldown'
      const msg = (text ?? c.draft).trim()
      if (!msg) return 'empty'
      if (msg.length > state.limits.maxChars) return 'too_long'
      if (c.messages.length >= MAX_MESSAGES) return 'full'
      const user: ChatMessage = { id: deps.newId(), role: 'user', text: msg, at: deps.now() }
      const requestId = deps.newId()
      patch(id, (x) => ({ ...x, messages: [...x.messages.map((m) => (m.failed ? { ...m, failed: false } : m)), user], pending: { requestId, since: deps.now() }, error: null, draft: '' }))
      dispatch(id, requestId, user.id, msg)
      return 'sent'
    },

    /** Re-send the last failed message WITHOUT adding it to the thread a second time. */
    retry(id: string): SendResult {
      const c = find(id)
      if (!c) return 'unknown_conversation'
      if (c.pending) return 'pending'
      if (!c.error?.retryable) return 'empty'
      if (c.error.until && deps.now() < c.error.until) return 'cooldown'
      const failed = [...c.messages].reverse().find((m) => m.role === 'user' && m.failed)
      if (!failed) return 'empty'
      const requestId = deps.newId()
      patch(id, (x) => ({ ...x, messages: x.messages.map((m) => (m.id === failed.id ? { ...m, failed: false } : m)), pending: { requestId, since: deps.now() }, error: null }))
      dispatch(id, requestId, failed.id, failed.text)
      return 'sent'
    },
  }
}

export type ChatStore = ReturnType<typeof createChatStore>

/* ------------------------------------------------------------------ small selectors the UI shares with the tests */
export const activeOf = (s: ChatState): Conversation => s.conversations.find((c) => c.id === s.activeId) ?? s.conversations[0]
/** Milliseconds until sending is allowed again after a rate-limit error (0 when free). */
export const cooldownMs = (c: Conversation, now: number): number => (c.error?.until ? Math.max(0, c.error.until - now) : 0)
export const canSend = (c: Conversation, _s: ChatState, now: number): boolean => !c.pending && cooldownMs(c, now) === 0
