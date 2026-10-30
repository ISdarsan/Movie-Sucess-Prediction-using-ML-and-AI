/* Chat store tests. The network and the clock are fakes: nothing here calls Gemini or any server. Run: npm test */
import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, it } from 'node:test'
import { activeOf, canSend, cooldownMs, createChatStore, DEFAULT_MAX_CHARS, ERROR_TEXT, mapError, MAX_CONVERSATIONS, MAX_MESSAGES, STARTER_PROMPTS, titleOf } from '../src/lib/chat.ts'

class FakeApiError extends Error {
  status: number; code: string; fields: { field: string; message: string }[]; retryAfter?: number
  constructor(status: number, code: string, message = 'RAW PROVIDER TEXT: AIzaSy-do-not-show', fields: { field: string; message: string }[] = [], retryAfter?: number) {
    super(message); this.status = status; this.code = code; this.fields = fields; this.retryAfter = retryAfter
  }
}

/** A store wired to fakes: every POST waits until the test resolves or rejects it; time only moves when the test says so. */
function rig(over: { timeoutMs?: number } = {}) {
  let t = 1_000_000, n = 0
  const calls: { path: string; body: any }[] = []
  const waiting: { res: (v: unknown) => void; rej: (e: unknown) => void }[] = []
  const sleepers: (() => void)[] = []
  const store = createChatStore({
    post: (path, body) => { calls.push({ path, body: body as any }); return new Promise((res, rej) => waiting.push({ res, rej })) },
    now: () => t, newId: () => `id${++n}`,
    sleep: () => new Promise<void>((r) => sleepers.push(r)), timeoutMs: over.timeoutMs,
  })
  const flush = async () => { for (let i = 0; i < 6; i++) await Promise.resolve() }
  return {
    store, calls, flush,
    get id() { return store.getState().activeId },
    conv: (id?: string) => store.getState().conversations.find((c) => c.id === (id ?? store.getState().activeId))!,
    tick: (ms: number) => { t += ms },
    answer: async (i: number, reply = 'An answer.', extra: object = {}) => { waiting[i].res({ reply, mode: 'general', provider: 'gemini', model: 'm', truncated: false, ...extra }); await flush() },
    fail: async (i: number, e: unknown) => { waiting[i].rej(e); await flush() },
    timeout: async (i = 0) => { sleepers[i](); await flush() },
  }
}
const roles = (c: { messages: { role: string }[] }) => c.messages.map((m) => m.role)

describe('initial state', () => {
  it('starts with one empty conversation, nothing pending, nothing sent', () => {
    const r = rig(), s = r.store.getState()
    assert.equal(s.conversations.length, 1); assert.equal(s.activeId, s.conversations[0].id)
    assert.deepEqual(s.conversations[0].messages, []); assert.equal(s.conversations[0].pending, null); assert.equal(s.conversations[0].error, null)
    assert.equal(s.limits.maxChars, DEFAULT_MAX_CHARS); assert.equal(s.limits.available, null); assert.equal(r.calls.length, 0)
    assert.equal(titleOf(s.conversations[0]), 'New conversation')
  })
  it('offers exactly the five starter prompts', () => {
    assert.deepEqual(STARTER_PROMPTS, ['Explain how Random Forest works.', 'What makes a movie commercially successful?', 'Explain the difference between Hit, Average and Flop.', 'How does our Movie Intelligence System work?', 'What factors affect movie performance?'])
  })
})

describe('sending and answers', () => {
  it('posts the existing general-mode schema with only the current message, then shows the reply', async () => {
    const r = rig()
    assert.equal(r.store.send(r.id, '  What is a Random Forest?  '), 'sent')
    assert.deepEqual(r.calls, [{ path: '/api/chat', body: { message: 'What is a Random Forest?', mode: 'general' } }])
    assert.deepEqual(Object.keys(r.calls[0].body).sort(), ['message', 'mode'], 'no history, no extra fields')
    await r.answer(0, 'Many trees voting.')
    const c = r.conv()
    assert.deepEqual(roles(c), ['user', 'assistant']); assert.equal(c.messages[1].text, 'Many trees voting.'); assert.equal(c.pending, null); assert.equal(c.error, null)
  })
  it('a second message sends ONLY itself, never the earlier turns', async () => {
    const r = rig(); r.store.send(r.id, 'first'); await r.answer(0); r.store.send(r.id, 'second')
    assert.deepEqual(r.calls[1].body, { message: 'second', mode: 'general' })
  })
  it('a starter prompt is sent as the message', async () => {
    const r = rig(); assert.equal(r.store.send(r.id, STARTER_PROMPTS[0]), 'sent')
    assert.equal(r.calls[0].body.message, STARTER_PROMPTS[0]); assert.equal(r.conv().messages[0].text, STARTER_PROMPTS[0])
  })
  it('sends the draft when no text is given, and clears it', async () => {
    const r = rig(); r.store.setDraft(r.id, 'from the box'); r.store.send(r.id)
    assert.equal(r.calls[0].body.message, 'from the box'); assert.equal(r.conv().draft, '')
  })
  it('shows a loading state while waiting', async () => {
    const r = rig(); r.store.send(r.id, 'hi')
    assert.ok(r.conv().pending); assert.equal(r.conv().pending!.since, 1_000_000); assert.equal(canSend(r.conv(), r.store.getState(), 1_000_000), false)
    await r.answer(0); assert.equal(r.conv().pending, null); assert.equal(canSend(r.conv(), r.store.getState(), 1_000_000), true)
  })
  it('messages get stable unique ids and timestamps', async () => {
    const r = rig(); r.store.send(r.id, 'a'); await r.answer(0); r.tick(5000); r.store.send(r.id, 'b'); await r.answer(1)
    const ms = r.conv().messages
    assert.equal(new Set(ms.map((m) => m.id)).size, 4); assert.ok(ms.every((m) => m.at >= 1_000_000)); assert.ok(ms[2].at > ms[0].at)
    const before = ms.map((m) => m.id); r.store.select(r.id); assert.deepEqual(r.conv().messages.map((m) => m.id), before)
  })
  it('records when a reply was cut short by the length limit', async () => {
    const r = rig(); r.store.send(r.id, 'x'); await r.answer(0, 'partial', { truncated: true }); assert.equal(r.conv().messages[1].truncated, true)
  })
  it('an empty or malformed API reply becomes an error, not a blank bubble', async () => {
    const r = rig(); r.store.send(r.id, 'x'); await r.answer(0, '   ')
    assert.equal(roles(r.conv()).join(), 'user'); assert.ok(r.conv().error)
  })
})

describe('validation before anything is sent', () => {
  it('empty and whitespace-only messages are not sent', () => {
    const r = rig()
    for (const t of ['', '   ', '\n\t ']) assert.equal(r.store.send(r.id, t), 'empty')
    assert.equal(r.calls.length, 0); assert.equal(r.conv().messages.length, 0)
  })
  it('the length limit is inclusive, follows the server status, and blocks before any request', () => {
    const r = rig()
    assert.equal(r.store.send(r.id, 'x'.repeat(DEFAULT_MAX_CHARS + 1)), 'too_long'); assert.equal(r.calls.length, 0)
    r.store.configure({ maxChars: 100 }); assert.equal(r.store.send(r.id, 'x'.repeat(101)), 'too_long'); assert.equal(r.store.send(r.id, 'x'.repeat(100)), 'sent')
  })
  it('still sends when status reports unavailable so the server can provide a clear error', async () => {
    const r = rig(); r.store.configure({ available: false }); r.store.setDraft(r.id, 'hello')
    assert.equal(canSend(r.conv(), r.store.getState(), 0), true)
    assert.equal(r.store.send(r.id), 'sent'); assert.equal(r.calls.length, 1)
    await r.fail(0, new FakeApiError(404, 'not_found'))
    assert.equal(r.conv().error!.kind, 'unavailable'); assert.match(r.conv().error!.message, /temporarily unavailable/)
    assert.equal(r.store.send(r.id, 'try again'), 'sent')
  })
  it('stops growing a conversation at the message cap', async () => {
    const r = rig(); const id = r.id
    for (let i = 0; i < MAX_MESSAGES / 2; i++) { r.store.send(id, `q${i}`); await r.answer(i) }
    assert.equal(r.conv().messages.length, MAX_MESSAGES); assert.equal(r.store.send(id, 'one more'), 'full')
  })
})

describe('duplicate submissions', () => {
  it('a conversation with a request in flight refuses another send, from any entry point', async () => {
    const r = rig(); assert.equal(r.store.send(r.id, 'first'), 'sent')
    for (const t of ['first', 'second', STARTER_PROMPTS[1]]) assert.equal(r.store.send(r.id, t), 'pending')
    assert.equal(r.store.retry(r.id), 'pending'); assert.equal(r.calls.length, 1); assert.equal(r.conv().messages.length, 1)
    await r.answer(0); assert.equal(r.store.send(r.id, 'now fine'), 'sent'); assert.equal(r.calls.length, 2)
  })
  it('ten rapid sends make exactly one request', () => {
    const r = rig(); r.store.setDraft(r.id, 'hello'); for (let i = 0; i < 10; i++) r.store.send(r.id)
    assert.equal(r.calls.length, 1); assert.equal(roles(r.conv()).join(), 'user')
  })
})

describe('failures and retry', () => {
  const E = (status: number, code: string, ra?: number) => new FakeApiError(status, code, undefined, [], ra)
  it('maps every documented status to a typed, safe error (never the raw text)', async () => {
    const table: [number, string, string, boolean][] = [
      [422, 'validation_error', 'validation', false], [429, 'rate_limited', 'rate_limited', true], [429, 'assistant_rate_limited', 'rate_limited', true], [429, 'busy', 'busy', true],
      [502, 'assistant_unreachable', 'server', true], [502, 'assistant_upstream_error', 'server', true], [502, 'assistant_no_reply', 'server', true],
      [503, 'assistant_not_configured', 'not_configured', false], [503, 'assistant_misconfigured', 'unavailable', true], [503, 'assistant_model_unavailable', 'unavailable', true],
      [504, 'assistant_timeout', 'timeout', true], [422, 'assistant_declined', 'declined', false], [0, 'network', 'network', true], [404, 'not_found', 'unavailable', true], [500, 'x', 'server', true], [418, 'teapot', 'unknown', true],
    ]
    for (const [status, code, kind, retryable] of table) {
      const r = rig(); r.store.send(r.id, 'hi'); await r.fail(0, E(status, code))
      const e = r.conv().error!
      assert.equal(e.kind, kind, `${status} ${code}`); assert.equal(e.retryable, retryable, `${status} ${code}`)
      if (kind !== 'validation') assert.equal(e.message, ERROR_TEXT[kind as keyof typeof ERROR_TEXT])
      assert.ok(!/RAW PROVIDER|AIza/.test(JSON.stringify(r.store.getState())), `raw text leaked for ${status} ${code}`)
    }
  })
  it('a failed message stays in the thread marked "not answered", and Retry re-sends it WITHOUT adding a duplicate', async () => {
    const r = rig(); r.store.send(r.id, 'tell me more'); await r.fail(0, E(502, 'assistant_unreachable'))
    assert.equal(r.conv().messages[0].failed, true); assert.equal(r.conv().pending, null)
    assert.equal(r.store.retry(r.id), 'sent'); assert.equal(r.calls.length, 2); assert.deepEqual(r.calls[1].body, r.calls[0].body)
    assert.equal(r.conv().messages.filter((m) => m.role === 'user').length, 1); assert.equal(r.conv().messages[0].failed, false); assert.equal(r.conv().error, null)
    await r.answer(1, 'Here you go.'); assert.deepEqual(roles(r.conv()), ['user', 'assistant'])
  })
  it('a retry that fails again keeps the message retryable', async () => {
    const r = rig(); r.store.send(r.id, 'q'); await r.fail(0, E(504, 'assistant_timeout')); r.store.retry(r.id); await r.fail(1, E(504, 'assistant_timeout'))
    assert.equal(r.conv().error!.kind, 'timeout'); assert.equal(r.conv().messages[0].failed, true); assert.equal(r.store.retry(r.id), 'sent')
  })
  it('errors that cannot be fixed by retrying offer no retry', async () => {
    const r = rig(); r.store.send(r.id, 'q'); await r.fail(0, E(422, 'assistant_declined'))
    assert.equal(r.store.retry(r.id), 'empty'); assert.equal(r.calls.length, 1); assert.equal(r.conv().messages[0].failed, false)
  })
  it('a server 422 puts the text back in the box to edit and removes it from the thread', async () => {
    const r = rig(); r.store.send(r.id, 'bad\u0000text'); await r.fail(0, new FakeApiError(422, 'validation_error', 'x', [{ field: 'message', message: 'The message contains characters that are not allowed.' }]))
    const c = r.conv(); assert.equal(c.messages.length, 0); assert.equal(c.draft, 'bad\u0000text'); assert.equal(c.error!.message, 'The message contains characters that are not allowed.')
  })
  it('sending a new message after a failure keeps the old one visible and clears the error', async () => {
    const r = rig(); r.store.send(r.id, 'old'); await r.fail(0, E(502, 'assistant_unreachable')); r.store.send(r.id, 'new')
    assert.deepEqual(r.conv().messages.map((m) => [m.text, !!m.failed]), [['old', false], ['new', false]]); assert.equal(r.conv().error, null)
  })
  it('dismissing an error removes only the banner', async () => {
    const r = rig(); r.store.send(r.id, 'q'); await r.fail(0, E(502, 'assistant_unreachable')); r.store.dismissError(r.id)
    assert.equal(r.conv().error, null); assert.equal(r.conv().messages.length, 1)
  })
  it('a request that never answers times out on the client, and a late answer is ignored', async () => {
    const r = rig({ timeoutMs: 1000 }); r.store.send(r.id, 'q'); await r.timeout(0)
    assert.equal(r.conv().error!.kind, 'timeout'); assert.equal(r.conv().pending, null)
    await r.answer(0, 'too late'); assert.equal(roles(r.conv()).join(), 'user', 'the late reply must not appear')
  })
})

describe('rate limits and Retry-After', () => {
  it('uses the Retry-After header, blocks send and retry until it has passed, then allows them', async () => {
    const r = rig(); r.store.send(r.id, 'q'); await r.fail(0, new FakeApiError(429, 'rate_limited', undefined, [], 12))
    const e = r.conv().error!
    assert.equal(e.kind, 'rate_limited'); assert.equal(e.until, 1_000_000 + 12_000); assert.equal(cooldownMs(r.conv(), 1_000_000), 12_000)
    assert.equal(r.store.send(r.id, 'again'), 'cooldown'); assert.equal(r.store.retry(r.id), 'cooldown'); assert.equal(r.calls.length, 1)
    r.tick(11_999); assert.equal(r.store.retry(r.id), 'cooldown')
    r.tick(1); assert.equal(cooldownMs(r.conv(), 1_012_000), 0); assert.equal(r.store.retry(r.id), 'sent'); assert.equal(r.calls.length, 2)
  })
  it('falls back to the seconds in the server message, then to safe defaults, and clamps silly values', () => {
    const wait = (e: unknown) => (mapError(e, 0).until ?? 0) / 1000
    assert.equal(wait(new FakeApiError(429, 'rate_limited', 'Too many messages. Try again in 7 seconds.')), 7)
    assert.equal(wait(new FakeApiError(429, 'assistant_rate_limited', 'busy')), 30)
    assert.equal(wait(new FakeApiError(429, 'busy', 'busy')), 5)
    assert.equal(wait(new FakeApiError(429, 'rate_limited', 'x', [], 99999)), 300)
    assert.equal(wait(new FakeApiError(429, 'rate_limited', 'x', [], 0.2)), 1)
  })
  it('the cooldown is per conversation: another conversation can still send', async () => {
    const r = rig(); const a = r.id; r.store.send(a, 'q'); await r.fail(0, new FakeApiError(429, 'rate_limited', undefined, [], 60))
    const b = r.store.newConversation()!; assert.equal(r.store.send(b, 'hello from B'), 'sent')
  })
})

describe('conversations', () => {
  it('a new conversation starts empty and becomes active; the old one keeps its messages', async () => {
    const r = rig(); const a = r.id; r.store.send(a, 'in A'); await r.answer(0, 'reply A')
    const b = r.store.newConversation()!
    assert.notEqual(a, b); assert.equal(r.store.getState().activeId, b); assert.deepEqual(r.conv(b).messages, []); assert.equal(r.conv(b).draft, '')
    assert.equal(r.conv(a).messages.length, 2); assert.equal(r.store.getState().conversations[0].id, b, 'newest first')
  })
  it('switching preserves each conversation, including drafts, and never mixes them', async () => {
    const r = rig(); const a = r.id; r.store.send(a, 'question A'); await r.answer(0, 'answer A'); r.store.setDraft(a, 'half-typed A')
    const b = r.store.newConversation()!; r.store.send(b, 'question B'); await r.answer(1, 'answer B')
    r.store.select(a)
    assert.deepEqual(r.conv(a).messages.map((m) => m.text), ['question A', 'answer A']); assert.equal(r.conv(a).draft, 'half-typed A')
    assert.deepEqual(r.conv(b).messages.map((m) => m.text), ['question B', 'answer B']); assert.equal(r.store.getState().activeId, a)
    assert.equal(titleOf(r.conv(a)), 'question A'); assert.equal(titleOf(r.conv(b)), 'question B')
  })
  it('an answer lands in the conversation that asked, even if another is open by then', async () => {
    const r = rig(); const a = r.id; r.store.send(a, 'slow question')
    const b = r.store.newConversation()!; r.store.send(b, 'quick question')
    await r.answer(1, 'quick answer'); r.store.select(a); await r.answer(0, 'slow answer')
    assert.deepEqual(r.conv(a).messages.map((m) => m.text), ['slow question', 'slow answer']); assert.deepEqual(r.conv(b).messages.map((m) => m.text), ['quick question', 'quick answer'])
    assert.equal(r.calls.length, 2)
  })
  it('an error in one conversation does not touch another', async () => {
    const r = rig(); const a = r.id; r.store.send(a, 'qa'); const b = r.store.newConversation()!; r.store.send(b, 'qb')
    await r.fail(0, new FakeApiError(502, 'assistant_unreachable')); await r.answer(1, 'ok')
    assert.ok(r.conv(a).error); assert.equal(r.conv(b).error, null); assert.equal(r.conv(b).messages.length, 2)
  })
  it('long first messages make a short title', () => {
    const r = rig(); r.store.send(r.id, 'word '.repeat(40)); assert.ok(titleOf(r.conv()).length <= 38 && titleOf(r.conv()).endsWith('…'))
  })
  it('caps the number of conversations and says so', () => {
    const r = rig()
    for (let i = 1; i < MAX_CONVERSATIONS; i++) assert.ok(r.store.newConversation())
    assert.equal(r.store.getState().conversations.length, MAX_CONVERSATIONS); assert.equal(r.store.newConversation(), null)
  })
  it('selecting an unknown id changes nothing', () => {
    const r = rig(); const before = r.store.getState(); r.store.select('nope'); assert.equal(r.store.getState(), before)
  })
})

describe('clear and delete', () => {
  it('clear removes every message and keeps the conversation', async () => {
    const r = rig(); const id = r.id; r.store.send(id, 'q'); await r.answer(0); r.store.setDraft(id, 'draft')
    r.store.clear(id); const c = r.conv(id)
    assert.deepEqual(c.messages, []); assert.equal(c.draft, ''); assert.equal(c.pending, null); assert.equal(c.error, null); assert.equal(r.store.getState().conversations.length, 1)
  })
  it('an answer that arrives after Clear is dropped, not shown in the emptied conversation', async () => {
    const r = rig(); r.store.send(r.id, 'q'); r.store.clear(r.id); await r.answer(0, 'ghost reply')
    assert.deepEqual(r.conv().messages, []); assert.equal(r.conv().pending, null)
    assert.equal(r.store.send(r.id, 'fresh'), 'sent', 'a cleared conversation can send again immediately')
  })
  it('clearing one conversation leaves the others alone', async () => {
    const r = rig(); const a = r.id; r.store.send(a, 'qa'); await r.answer(0); const b = r.store.newConversation()!; r.store.send(b, 'qb'); await r.answer(1)
    r.store.clear(b); assert.equal(r.conv(a).messages.length, 2); assert.equal(r.conv(b).messages.length, 0)
  })
  it('deleting the active conversation moves to a neighbour; deleting the last one leaves a fresh empty one', async () => {
    const r = rig(); const a = r.id; const b = r.store.newConversation()!
    r.store.remove(b); assert.equal(r.store.getState().activeId, a); assert.equal(r.store.getState().conversations.length, 1)
    r.store.remove(a); const s = r.store.getState(); assert.equal(s.conversations.length, 1); assert.notEqual(s.conversations[0].id, a); assert.deepEqual(s.conversations[0].messages, [])
  })
  it('an answer for a deleted conversation is ignored', async () => {
    const r = rig(); const a = r.id; const b = r.store.newConversation()!; r.store.send(a, 'q'); r.store.remove(a); await r.answer(0, 'orphan')
    assert.ok(!r.store.getState().conversations.some((c) => c.id === a)); assert.equal(r.conv(b).messages.length, 0)
  })
})

describe('store contract', () => {
  it('notifies subscribers on change and returns a stable snapshot between changes', () => {
    const r = rig(); let n = 0; const off = r.store.subscribe(() => n++)
    const s1 = r.store.getState(); assert.equal(r.store.getState(), s1)
    r.store.setDraft(r.id, 'x'); assert.equal(n, 1); assert.notEqual(r.store.getState(), s1); off(); r.store.setDraft(r.id, 'y'); assert.equal(n, 1)
  })
  it('activeOf falls back safely', () => { const r = rig(); assert.equal(activeOf({ ...r.store.getState(), activeId: 'gone' }).id, r.id) })
})

describe('privacy: nothing is stored and no secret or provider call exists in the browser code', () => {
  const walk = (dir: string): string[] => readdirSync(dir).flatMap((f) => { const p = join(dir, f); return statSync(p).isDirectory() ? walk(p) : [p] })
  const code = (path: string) => readFileSync(path, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`])\/\/.*$/gm, '$1')   // comments may MENTION storage to say there is none
  const files = ['src/lib/chat.ts', 'src/lib/useChat.ts', 'src/lib/miniMarkdown.ts', 'src/pages/Assistant.tsx', ...walk('src/components/assistant')]
  it('no web storage, cookies or IndexedDB in the assistant code', () => {
    for (const f of files) assert.ok(!/localStorage|sessionStorage|document\.cookie|indexedDB/.test(code(f)), f)
  })
  it('no provider URL, key variable or key-shaped text in the whole frontend source', () => {
    for (const f of walk('src').filter((p) => /\.(ts|tsx)$/.test(p))) {
      const t = readFileSync(f, 'utf8')
      assert.ok(!/generativelanguage|googleapis|x-goog|GEMINI_API_KEY|@google\/genai/.test(t), `${f} mentions the provider or key`)
      assert.ok(!/AIza[0-9A-Za-z_-]{20,}|\bAQ\.[0-9A-Za-z_-]{20,}/.test(t), `${f} contains key-shaped text`)
    }
  })
  it('the only network path used for chat is the existing same-origin /api/chat', () => {
    const t = readFileSync('src/lib/chat.ts', 'utf8'); assert.deepEqual([...t.matchAll(/'(\/api\/[^']+)'/g)].map((m) => m[1]), ['/api/chat'])
  })
})
