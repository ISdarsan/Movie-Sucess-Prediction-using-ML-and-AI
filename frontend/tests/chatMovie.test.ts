/* Movie-aware conversations in the chat store (Phase 3). Fake network and clock: nothing here calls a server or Gemini. Run: npm test */
import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { bodyFor, createChatStore, HISTORY_TURN_CHARS, MAX_CONVERSATIONS, MAX_HISTORY_SENT, movieLabel, MOVIE_STARTER_PROMPTS, titleOf, type MovieRef } from '../src/lib/chat.ts'

class FakeApiError extends Error {
  status: number; code: string; fields: { field: string; message: string }[]; retryAfter?: number
  constructor(status: number, code: string, message = 'RAW PROVIDER TEXT: AIzaSy-do-not-show', retryAfter?: number) { super(message); this.status = status; this.code = code; this.fields = []; this.retryAfter = retryAfter }
}

function rig() {
  let t = 1_000_000, n = 0
  const calls: { path: string; body: any }[] = []
  const waiting: { res: (v: unknown) => void; rej: (e: unknown) => void }[] = []
  const store = createChatStore({
    post: (path, body) => { calls.push({ path, body: body as any }); return new Promise((res, rej) => waiting.push({ res, rej })) },
    now: () => t, newId: () => `id${++n}`, sleep: () => new Promise<void>(() => {}),
  })
  const flush = async () => { for (let i = 0; i < 6; i++) await Promise.resolve() }
  return {
    store, calls,
    conv: (id: string) => store.getState().conversations.find((c) => c.id === id)!,
    answer: async (i: number, reply = 'An answer.') => { waiting[i].res({ reply, mode: 'movie', provider: 'gemini', model: 'm', truncated: false }); await flush() },
    fail: async (i: number, e: unknown) => { waiting[i].rej(e); await flush() },
  }
}

const RUN_A = 'a'.repeat(32), RUN_B = 'b'.repeat(32)
const filmA: MovieRef = { runId: RUN_A, title: 'Alpha Film', predictedClass: 'Hit', year: 2010, genres: 'Action / Sci-Fi' }
const filmB: MovieRef = { runId: RUN_B, title: null, predictedClass: 'Flop', year: 1999, genres: 'Horror' }
const keys = (o: object) => Object.keys(o).sort()

describe('movie conversations: creation and labels', () => {
  it('is bound to its film at creation, starts empty, and becomes active', () => {
    const r = rig(); const id = r.store.newConversation(filmA)!
    assert.equal(r.store.getState().activeId, id); assert.deepEqual(r.conv(id).messages, []); assert.deepEqual(r.conv(id).movie, filmA)
    assert.equal(r.store.getState().conversations[1].movie, null, 'the original conversation stays general')
  })
  it('titles name the film (or its traits when no title was given) and the class', () => {
    assert.equal(movieLabel(filmA), 'Alpha Film · Hit'); assert.equal(movieLabel(filmB), 'Horror · 1999 · Flop'); assert.equal(movieLabel({ ...filmB, genres: null, year: null }), 'Untitled film · Flop')
    const r = rig(); const id = r.store.newConversation(filmA)!; r.store.send(id, 'a question'); assert.equal(titleOf(r.conv(id)), 'Alpha Film · Hit')
  })
  it('the film cannot change afterwards: no store operation rewrites it', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!; const ref = r.conv(id).movie
    r.store.send(id, 'q'); await r.answer(0); r.store.setDraft(id, 'x'); r.store.clear(id); r.store.select(id); r.store.dismissError(id)
    assert.equal(r.conv(id).movie, ref)
  })
  it('offers seven movie starter prompts covering the documented questions', () => {
    assert.equal(MOVIE_STARTER_PROMPTS.length, 7)
    for (const k of ['risks', 'research agent', 'comparable', 'limitations', 'Summarize', 'simple language']) assert.ok(MOVIE_STARTER_PROMPTS.some((p) => p.includes(k)), k)
  })
  it('respects the conversation cap for movie conversations too', () => {
    const r = rig(); for (let i = 1; i < MAX_CONVERSATIONS; i++) assert.ok(r.store.newConversation(i % 2 ? filmA : null)); assert.equal(r.store.newConversation(filmB), null)
  })
})

describe('movie conversations: what is sent', () => {
  it('sends the run id and history and NOTHING about the film', () => {
    const r = rig(); const id = r.store.newConversation(filmA)!
    assert.equal(r.store.send(id, '  Why Hit?  '), 'sent')
    assert.deepEqual(r.calls[0], { path: '/api/chat', body: { message: 'Why Hit?', mode: 'movie', run_id: RUN_A, history: [] } })
    assert.deepEqual(keys(r.calls[0].body), ['history', 'message', 'mode', 'run_id'])
    const blob = JSON.stringify(r.calls[0].body)
    for (const leak of ['Alpha', 'Action', '2010', 'predictedClass', 'probabilit', 'score', 'class_scores']) assert.ok(!blob.includes(leak), leak)
  })
  it('a follow-up carries the answered earlier turns, oldest first, as role and text only', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!
    r.store.send(id, 'first question'); await r.answer(0, 'first answer'); r.store.send(id, 'follow up')
    assert.deepEqual(r.calls[1].body, { message: 'follow up', mode: 'movie', run_id: RUN_A, history: [{ role: 'user', text: 'first question' }, { role: 'assistant', text: 'first answer' }] })
    assert.deepEqual(keys(r.calls[1].body.history[0]), ['role', 'text'])
  })
  it('history is bounded by turn count and turn length, keeping the most recent', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!
    for (let i = 0; i < 6; i++) { r.store.send(id, `q${i}`); await r.answer(i, `a${i} ` + 'x'.repeat(3000)) }
    r.store.send(id, 'last'); const h = r.calls[6].body.history
    assert.equal(h.length, MAX_HISTORY_SENT); assert.equal(h.at(-1).role, 'assistant'); assert.ok(h.at(-1).text.startsWith('a5 ')); assert.ok(h.every((t: any) => t.text.length <= HISTORY_TURN_CHARS)); assert.ok(!h.some((t: any) => t.text === 'q0'))
  })
  it('an unanswered (failed) message is not sent as history', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!
    r.store.send(id, 'lost one'); await r.fail(0, new FakeApiError(502, 'assistant_unreachable')); r.store.send(id, 'new one')
    assert.deepEqual(r.calls[1].body.history, [])
  })
  it('retry re-sends the same run id and history without duplicating the message', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!
    r.store.send(id, 'one'); await r.answer(0, 'ans'); r.store.send(id, 'two'); await r.fail(1, new FakeApiError(504, 'assistant_timeout'))
    assert.equal(r.store.retry(id), 'sent'); assert.deepEqual(r.calls[2].body, r.calls[1].body)
    assert.deepEqual(r.calls[2].body.history, [{ role: 'user', text: 'one' }, { role: 'assistant', text: 'ans' }]); assert.equal(r.conv(id).messages.filter((m) => m.role === 'user').length, 2)
  })
  it('clear removes the thread (so no history follows) but keeps the film', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!
    r.store.send(id, 'q'); await r.answer(0); r.store.clear(id); assert.deepEqual(r.conv(id).movie, filmA); r.store.send(id, 'fresh')
    assert.deepEqual(r.calls[1].body, { message: 'fresh', mode: 'movie', run_id: RUN_A, history: [] })
  })
  it('general conversations are exactly as before: two keys, never a run id or history', async () => {
    const r = rig(); r.store.newConversation(filmA); const g = r.store.newConversation(null)!
    r.store.send(g, 'general q'); await r.answer(0); r.store.send(g, 'general q2')
    for (const c of r.calls) assert.deepEqual(c.body, { message: c.body.message, mode: 'general' })
    assert.deepEqual(keys(r.calls[1].body), ['message', 'mode'])
  })
  it('bodyFor never mutates the conversation', () => {
    const r = rig(); const id = r.store.newConversation(filmA)!; const before = JSON.stringify(r.conv(id)); bodyFor(r.conv(id), 'x', 'hello'); assert.equal(JSON.stringify(r.conv(id)), before)
  })
  it('still sends movie-aware requests when status reports unavailable', () => {
    const r = rig(); const id = r.store.newConversation(filmA)!; r.store.configure({ available: false })
    assert.equal(r.store.send(id, 'x'), 'sent'); assert.equal(r.calls.length, 1)
    assert.deepEqual(r.calls[0].body, { message: 'x', mode: 'movie', run_id: filmA.runId, history: [] })
  })
})

describe('movie conversations never mix', () => {
  it('two films keep separate threads, run ids and histories', async () => {
    const r = rig(); const a = r.store.newConversation(filmA)!; const b = r.store.newConversation(filmB)!
    r.store.send(a, 'about A'); await r.answer(0, 'A answer'); r.store.send(b, 'about B'); await r.answer(1, 'B answer')
    r.store.send(a, 'more A'); r.store.send(b, 'more B')
    assert.equal(r.calls[2].body.run_id, RUN_A); assert.deepEqual(r.calls[2].body.history.map((t: any) => t.text), ['about A', 'A answer'])
    assert.equal(r.calls[3].body.run_id, RUN_B); assert.deepEqual(r.calls[3].body.history.map((t: any) => t.text), ['about B', 'B answer'])
    assert.ok(!JSON.stringify(r.calls[2]).includes('B answer') && !JSON.stringify(r.calls[3]).includes('A answer'))
  })
  it('an answer lands only in the conversation (and film) that asked, even after switching', async () => {
    const r = rig(); const a = r.store.newConversation(filmA)!; r.store.send(a, 'slow A'); const b = r.store.newConversation(filmB)!; r.store.send(b, 'quick B')
    await r.answer(1, 'B reply'); r.store.select(a); await r.answer(0, 'A reply')
    assert.deepEqual(r.conv(a).messages.map((m) => m.text), ['slow A', 'A reply']); assert.deepEqual(r.conv(b).messages.map((m) => m.text), ['quick B', 'B reply'])
  })
  it('starting another film adds a conversation and leaves the first untouched', async () => {
    const r = rig(); const a = r.store.newConversation(filmA)!; r.store.send(a, 'q'); await r.answer(0); const snapshot = JSON.stringify(r.conv(a))
    r.store.newConversation(filmB); assert.equal(JSON.stringify(r.conv(a)), snapshot); assert.equal(r.store.getState().conversations.filter((c) => c.movie).length, 2)
  })
  it('a general conversation next to a movie one never receives film context', async () => {
    const r = rig(); const m = r.store.newConversation(filmA)!; r.store.send(m, 'about the film'); await r.answer(0, 'film answer'); const g = r.store.newConversation()!; r.store.send(g, 'general')
    assert.deepEqual(r.calls[1].body, { message: 'general', mode: 'general' })
  })
})

describe('movie conversations: errors from the server', () => {
  it('analysis problems get their own wording, never the raw server text', async () => {
    const table: [number, string, boolean, RegExp][] = [
      [404, 'run_not_found', false, /not found on the server/], [409, 'run_failed', false, /did not produce a prediction/], [409, 'run_incomplete', false, /did not produce a prediction/],
      [409, 'run_not_ready', true, /not finished yet/], [503, 'context_unavailable', true, /could not be read/],
    ]
    for (const [status, code, retryable, text] of table) {
      const r = rig(); const id = r.store.newConversation(filmA)!; r.store.send(id, 'q'); await r.fail(0, new FakeApiError(status, code))
      const e = r.conv(id).error!; assert.equal(e.kind, 'run_unavailable', code); assert.equal(e.retryable, retryable, code); assert.match(e.message, text, code); assert.ok(!/RAW PROVIDER|AIza/.test(e.message))
      assert.equal(r.conv(id).movie, filmA)
    }
  })
  it('an unavailable analysis is never turned into a general conversation: nothing is re-sent', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!; r.store.send(id, 'q'); await r.fail(0, new FakeApiError(404, 'run_not_found'))
    assert.equal(r.store.retry(id), 'empty'); assert.equal(r.calls.length, 1); assert.equal(r.calls[0].body.mode, 'movie')
  })
  it('existing error handling still applies (rate limit with Retry-After)', async () => {
    const r = rig(); const id = r.store.newConversation(filmA)!; r.store.send(id, 'q'); await r.fail(0, new FakeApiError(429, 'rate_limited', 'x', 9))
    assert.equal(r.conv(id).error!.kind, 'rate_limited'); assert.equal(r.store.send(id, 'again'), 'cooldown')
  })
})
