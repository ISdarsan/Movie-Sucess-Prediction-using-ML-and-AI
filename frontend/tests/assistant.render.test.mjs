/* Rendering tests for the AI Assistant: the REAL components, loaded through the project's Vite toolchain and rendered to static HTML.
   They cover content, structure, escaping and accessibility attributes. Interaction (Enter to send, focus, mobile layout) is covered by the
   real-browser suite in tests/browser/assistant.browser.mjs. Run: npm test */
import assert from 'node:assert/strict'
import { after, before, describe, it } from 'node:test'
import { createElement as h } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { createServer } from 'vite'

let vite, C, P, Chat, Shell
const render = (el) => renderToStaticMarkup(el)
const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/&#x27;/g, "'").replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/\s+/g, ' ')
const msg = (over = {}) => ({ id: 'm1', role: 'assistant', text: 'Hello', at: Date.UTC(2026, 9, 30, 12, 0), ...over })
const conv = (over = {}) => ({ id: 'c1', createdAt: 0, messages: [], pending: null, error: null, draft: '', ...over })
const noop = () => {}
/** the opening tag of the element with this data-testid (attribute order does not matter) */
const tag = (html, id) => (html.match(new RegExp(`<[a-z]+[^>]*data-testid="${id}"[^>]*>`)) ?? [''])[0]
const isDisabled = (html, id) => /\bdisabled=""/.test(tag(html, id))

before(async () => {
  vite = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'), server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' })
  C = await vite.ssrLoadModule('/src/components/assistant/Chat.tsx')
  P = await vite.ssrLoadModule('/src/pages/Assistant.tsx')
  Chat = await vite.ssrLoadModule('/src/lib/chat.ts')
  Shell = await vite.ssrLoadModule('/src/components/Shell.tsx')
})
after(async () => { await vite?.close() })

describe('messages', () => {
  it('user and assistant bubbles are distinguishable visually and for screen readers', () => {
    const u = render(h(C.MessageBubble, { m: msg({ role: 'user', text: 'What is a forest?' }) })), a = render(h(C.MessageBubble, { m: msg() }))
    assert.ok(u.includes('data-role="user"') && a.includes('data-role="assistant"')); assert.match(text(u), /You said: What is a forest\?/); assert.match(text(a), /Assistant said: Hello/)
    assert.ok(u.includes('flex-row-reverse') && !a.includes('flex-row-reverse'))
    assert.match(u, /<time dateTime="2026-10-30T12:00:00.000Z"/)
  })
  it('assistant markdown is rendered as structure: lists, bold, code', () => {
    const html = render(h(C.MessageBubble, { m: msg({ text: 'Intro **bold** and `code`\n\n- one\n- two\n\n1. first\n2. second\n\n```\nx = 1\n```' }) }))
    assert.ok(html.includes('<strong') && html.includes('<code') && html.includes('<ul') && html.includes('<ol') && html.includes('<pre'))
    assert.equal((html.match(/<li/g) ?? []).length, 5, 'the message bubble itself is an <li>, plus four list items')
  })
  it('hostile model text becomes inert text: no script, image, link or event handler can be produced', () => {
    const evil = '<img src=x onerror=alert(1)> <script>steal()</script> [click me](javascript:alert(1)) <a href="http://evil">x</a>'
    for (const role of ['assistant', 'user']) {
      const html = render(h(C.MessageBubble, { m: msg({ role, text: evil }) }))
      assert.ok(!/<(img|script|a)\b/i.test(html), role); assert.ok(!/<[a-z][^>]*\bhref=/i.test(html), role); assert.ok(!/<[a-z][^>]*\bon[a-z]+=/i.test(html), role); assert.ok(!/onerror=alert/.test(html.replace(/&lt;[^&]*&gt;/g, '')), role)
      assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'), role)
    }
  })
  it('flags a failed (unanswered) message and a truncated reply', () => {
    assert.match(text(render(h(C.MessageBubble, { m: msg({ role: 'user', failed: true }) }))), /Not answered/)
    assert.match(text(render(h(C.MessageBubble, { m: msg({ truncated: true }) }))), /Cut short by the length limit/)
    assert.ok(!/Not answered|Cut short/.test(text(render(h(C.MessageBubble, { m: msg() })))))
  })
  it('the loading indicator is an announced status', () => {
    const html = render(h(C.Thinking)); assert.ok(html.includes('role="status"') && /assistant is thinking/.test(text(html)) && html.includes('data-testid="thinking"'))
    assert.ok(html.includes('motion-reduce:animate-none'), 'the pulse stops under reduced motion')
  })
})

describe('empty state', () => {
  it('shows the five starter prompts as buttons, and says it is a general assistant', () => {
    const html = render(h(C.EmptyChat, { onPick: noop, disabled: false }))
    assert.equal((html.match(/data-testid="starter"/g) ?? []).length, 5)
    for (const p of Chat.STARTER_PROMPTS) assert.ok(text(html).includes(p), p)
    assert.match(text(html), /cannot see your films, predictions or database/); assert.match(text(html), /not model predictions/); assert.ok(!/disabled=""/.test(html))
  })
  it('starter prompts are disabled while sending is not possible', () => assert.equal((render(h(C.EmptyChat, { onPick: noop, disabled: true })).match(/disabled=""/g) ?? []).length, 5))
})

describe('errors', () => {
  const err = (kind, over = {}) => ({ kind, message: Chat.ERROR_TEXT[kind] ?? 'x', retryable: true, until: null, at: 0, ...over })
  it('is an alert with a Retry button for retryable errors', () => {
    const html = render(h(C.ErrorBanner, { error: err('server'), waitMs: 0, onRetry: noop, onDismiss: noop }))
    assert.ok(html.includes('role="alert"') && html.includes('data-kind="server"') && html.includes('data-testid="retry"') && !isDisabled(html, 'retry'))
    assert.match(text(html), /The assistant had a problem/); assert.ok(html.includes('aria-label="Dismiss error"'))
  })
  it('a rate limit shows the countdown and disables Retry until it ends', () => {
    const html = render(h(C.ErrorBanner, { error: err('rate_limited', { until: 12_000 }), waitMs: 12_000, onRetry: noop, onDismiss: noop }))
    assert.match(text(html), /You can try again in 12s/); assert.match(text(html), /Retry in 12s/); assert.ok(isDisabled(html, 'retry'), 'Retry is disabled during the cooldown')
  })
  it('non-retryable errors offer no Retry', () => {
    for (const kind of ['validation', 'declined', 'not_configured']) {
      const html = render(h(C.ErrorBanner, { error: err(kind, { retryable: false }), waitMs: 0, onRetry: noop, onDismiss: noop }))
      assert.ok(!html.includes('data-testid="retry"'), kind); assert.ok(html.includes('role="alert"'), kind)
    }
  })
  it('never shows provider or server text, only our wording', () => assert.ok(!/AIza|RAW PROVIDER|Traceback/.test(render(h(C.ErrorBanner, { error: err('unavailable'), waitMs: 0, onRetry: noop, onDismiss: noop })))))
})

describe('composer', () => {
  const comp = (over = {}, c = {}) => render(h(C.Composer, { conv: conv(c), maxChars: 2000, waitMs: 0, onChange: noop, onSend: noop, inputRef: { current: null }, ...over }))
  it('has a labelled textarea, a Send button and the keyboard hint', () => {
    const html = comp()
    assert.ok(html.includes('<textarea') && /<label[^>]*class="sr-only"[^>]*>Message to the assistant<\/label>/.test(html)); assert.match(text(html), /Enter to send · Shift\+Enter for a new line/)
    assert.ok(html.includes('aria-label="Send message"') && html.includes('aria-describedby'))
  })
  it('Send is disabled when empty, while pending and during a cooldown; enabled otherwise', () => {
    const disabled = (h2) => isDisabled(h2, 'send')
    assert.ok(disabled(comp())); assert.ok(!disabled(comp({}, { draft: 'hi' })))
    assert.ok(disabled(comp({}, { draft: 'hi', pending: { requestId: 'r', since: 0 } }))); assert.ok(disabled(comp({ waitMs: 5000 }, { draft: 'hi' })))
    assert.match(text(comp({ waitMs: 5000 }, { draft: 'hi' })), /You can send again in 5s/); assert.match(text(comp({}, { draft: 'hi', pending: { requestId: 'r', since: 0 } })), /Waiting for the answer/)
  })
  it('shows a counter near the limit and marks an over-long message invalid', () => {
    assert.ok(!text(comp({}, { draft: 'short' })).includes('/ 2,000'))
    assert.match(text(comp({}, { draft: 'x'.repeat(1700) })), /1,700 \/ 2,000/)
    const over = comp({}, { draft: 'x'.repeat(2001) }); assert.match(text(over), /2,001 \/ 2,000 — too long/); assert.ok(over.includes('aria-invalid="true"')); assert.ok(isDisabled(over, 'send'))
  })
  it('the limit comes from the server status', () => assert.match(text(comp({ maxChars: 100 }, { draft: 'x'.repeat(90) })), /90 \/ 100/))
  it('keeps the composer available for writing when the server is unavailable', () => {
    const html = comp({}, { draft: 'try again' })
    assert.ok(!/<textarea[^>]*\sdisabled(?:\s|=|>)/.test(html)); assert.ok(!isDisabled(html, 'send'))
  })
})

describe('conversation switcher and dialog', () => {
  const state = (cs, active) => ({ conversations: cs, activeId: active, limits: { maxChars: 2000, available: null } })
  it('marks the active conversation, shows titles and counts, and flags a pending one', () => {
    const a = conv({ id: 'a', messages: [msg({ role: 'user', text: 'Explain forests' })] }), b = conv({ id: 'b', pending: { requestId: 'r', since: 0 } })
    const html = render(h(C.ConversationRail, { state: state([b, a], 'a'), onSelect: noop, onNew: noop, canAdd: true }))
    assert.ok(html.includes('aria-label="Conversations"') && /data-conversation-id="a"[^>]*>|aria-current="true"/.test(html))
    assert.equal((html.match(/aria-current="true"/g) ?? []).length, 1); assert.match(text(html), /Explain forests/); assert.match(text(html), /New conversation/); assert.ok(html.includes('aria-label="Waiting for an answer"'))
  })
  it('New is disabled at the conversation limit, with a reason', () => {
    const html = render(h(C.ConversationRail, { state: state([conv()], 'c1'), onSelect: noop, onNew: noop, canAdd: false }))
    assert.ok(isDisabled(html, 'new-conversation')); assert.ok(html.includes('Delete a conversation to start another'))
  })
  it('the confirmation dialog is a labelled, modal alertdialog with Cancel and a destructive action', () => {
    const html = render(h(C.ConfirmDialog, { title: 'Clear this conversation?', body: 'Gone for good.', confirmLabel: 'Clear messages', onConfirm: noop, onCancel: noop }))
    assert.ok(html.includes('role="alertdialog"') && html.includes('aria-modal="true"') && /aria-labelledby="[^"]+"/.test(html) && /aria-describedby="[^"]+"/.test(html))
    assert.match(text(html), /Clear this conversation\?/); assert.ok(html.includes('data-testid="confirm-cancel"') && html.includes('data-testid="confirm-ok"'))
  })
})

describe('the page', () => {
  const page = () => text(render(h(MemoryRouter, null, h(P.default))))
  it('renders the heading, the empty state, the starters, the composer and the privacy note', () => {
    const html = render(h(MemoryRouter, null, h(P.default))), t = text(html)
    assert.ok(/<h1[^>]*aria-label="AI Assistant"/.test(html)); assert.match(t, /Ask about cinema, data and movie success/); assert.equal((html.match(/data-testid="starter"/g) ?? []).length, 5)
    assert.ok(html.includes('data-testid="composer"')); assert.match(t, /does not see your films/); assert.match(t, /never makes or changes a prediction|it never makes or changes a prediction/i); assert.ok(html.includes('data-testid="chat-panel"'))
    assert.ok(!html.includes('data-testid="assistant-unavailable"') && !html.includes('role="alert"'))
  })
  it('shows the unavailable state while keeping the composer usable', async () => {
    const { chatStore } = await vite.ssrLoadModule('/src/lib/useChat.ts')
    const store = chatStore()
    store.configure({ available: false })
    store.setDraft(store.getState().activeId, 'hello')
    try {
      const html = render(h(MemoryRouter, null, h(P.default)))
      assert.ok(html.includes('data-testid="assistant-unavailable"')); assert.match(text(html), /The AI assistant is not set up yet/); assert.match(text(html), /everything else in FRAME keeps working/i)
      assert.ok(!/<textarea[^>]*\sdisabled(?:\s|=|>)/.test(html), 'the message box remains editable')
      assert.ok(!isDisabled(html, 'send'), 'a non-empty draft can be sent to receive the server error')
      const starters = html.match(/<button[^>]*data-testid="starter"[^>]*>/g) ?? []
      assert.equal(starters.length, 5); assert.ok(starters.every((b) => /disabled=""/.test(b)), 'every starter prompt is disabled')
    } finally { store.setDraft(store.getState().activeId, ''); store.configure({ available: null }) }
  })
  it('registers exactly one navigation entry for /assistant and keeps the other seven', () => {
    const to = Shell.NAV.map((n) => n.to)
    assert.equal(Shell.NAV.filter((n) => n.to === '/assistant').length, 1); assert.deepEqual(to.filter((t) => t !== '/assistant'), ['/', '/predict', '/performance', '/explain', '/explore', '/researcher', '/about'])
    assert.equal(Shell.NAV.find((n) => n.to === '/assistant').label, 'AI Assistant'); assert.equal(Shell.NAV.length, 8)
  })
  it('the page text never contains a credential or provider address', () => assert.ok(!/AIza|AQ\.|generativelanguage|googleapis/.test(page())))
})
