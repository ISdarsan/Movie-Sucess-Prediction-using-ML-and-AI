/* Rendering tests for the movie-aware assistant UI: the REAL components loaded through the project's Vite toolchain, rendered to static HTML.
   Interaction (picking, switching, keyboard, mobile) is covered by the real-browser suite in tests/browser/assistant.browser.mjs. Run: npm test */
import assert from 'node:assert/strict'
import { after, before, describe, it } from 'node:test'
import { createElement as h } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { createServer } from 'vite'
import * as fx from './fixtures.ts'

let vite, M, C, P, RV, A, Reg, Chat
const render = (el) => renderToStaticMarkup(el)
const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/&#x27;/g, "'").replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/\s+/g, ' ')
const tag = (html, id) => (html.match(new RegExp(`<[a-z]+[^>]*data-testid="${id}"[^>]*>`)) ?? [''])[0]
const isDisabled = (html, id) => /\bdisabled=""/.test(tag(html, id))
const noop = () => {}
const filmA = { runId: 'a'.repeat(32), title: 'Alpha Film', predictedClass: 'Hit', year: 2010, genres: 'Action / Sci-Fi' }
const untitled = { runId: 'b'.repeat(32), title: null, predictedClass: 'Flop', year: 1999, genres: 'Horror' }

before(async () => {
  vite = await createServer({ root: new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'), server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' })
  M = await vite.ssrLoadModule('/src/components/assistant/Movie.tsx')
  C = await vite.ssrLoadModule('/src/components/assistant/Chat.tsx')
  P = await vite.ssrLoadModule('/src/pages/Assistant.tsx')
  RV = await vite.ssrLoadModule('/src/components/researcher/RunView.tsx')
  A = await vite.ssrLoadModule('/src/lib/agentRun.ts')
  Reg = await vite.ssrLoadModule('/src/lib/useAnalyses.ts')
  Chat = await vite.ssrLoadModule('/src/lib/chat.ts')
})
after(async () => { await vite?.close() })

describe('which film is being discussed', () => {
  it('the context chip names the film, shows its class, and says the assistant cannot change the prediction', () => {
    const html = render(h(M.MovieChip, { movie: filmA, onChange: noop, disabled: false })), t = text(html)
    assert.match(t, /Discussing Alpha Film/); assert.match(t, /HIT/i); assert.match(t, /cannot change the prediction/); assert.ok(html.includes('data-testid="change-movie"') && !isDisabled(html, 'change-movie'))
  })
  it('an untitled film is named by its traits, never by an invented title', () => {
    const t = text(render(h(M.MovieChip, { movie: untitled, onChange: noop, disabled: false }))); assert.match(t, /Discussing Horror · 1999/); assert.ok(!/Untitled/.test(t))
    assert.equal(Chat.movieName({ ...untitled, genres: null, year: null }), 'Untitled film')
  })
})

describe('movie empty state', () => {
  it('offers the seven suggested questions, the first naming the predicted class', () => {
    const html = render(h(M.MovieEmptyChat, { movie: filmA, onPick: noop, disabled: false }))
    assert.equal((html.match(/data-testid="starter"/g) ?? []).length, 7); assert.match(text(html), /Why did the model predict Hit\?/)
    for (const q of ['What are the biggest risks?', 'What did the research agent discover?', 'Explain the comparable movies.', 'What are the limitations of this prediction?', 'Summarize the entire analysis.', 'Explain the prediction in simple language.']) assert.ok(text(html).includes(q), q)
  })
  it('states the guarantees: cannot change the prediction, three separate kinds of information, classes are not financial success', () => {
    const t = text(render(h(M.MovieEmptyChat, { movie: filmA, onPick: noop, disabled: false })))
    assert.match(t, /cannot change the predicted class or its scores/); assert.match(t, /model’s prediction, historical dataset evidence about other films, and web research/); assert.match(t, /not financial success/); assert.match(t, /uncalibrated/)
  })
  it('starters are disabled when sending is not possible', () => assert.equal((render(h(M.MovieEmptyChat, { movie: filmA, onPick: noop, disabled: true })).match(/disabled=""/g) ?? []).length, 7))
})

describe('the hand-off is an offer', () => {
  it('asks first, names the film, and says the current conversation is not changed', () => {
    const html = render(h(M.OfferBanner, { movie: filmA, existing: false, onStart: noop, onOpen: noop, onDismiss: noop })), t = text(html)
    assert.ok(html.includes('aria-label="Offer: discuss an analysis"')); assert.match(t, /Talk about Alpha Film ?\?/); assert.match(t, /current conversation is not changed/); assert.match(t, /Start conversation/); assert.match(t, /Not now/)
  })
  it('offers to reopen the existing conversation instead of making a duplicate', () => {
    const t = text(render(h(M.OfferBanner, { movie: filmA, existing: true, onStart: noop, onOpen: noop, onDismiss: noop }))); assert.match(t, /already have a conversation/); assert.match(t, /Open it/); assert.ok(!/Start conversation/.test(t))
  })
})

describe('analysis picker', () => {
  const an = (over = {}) => ({ ...filmA, at: Date.UTC(2026, 9, 30, 10, 0), ...over })
  const dlg = (analyses) => render(h(M.MoviePicker, { analyses, lookup: async () => filmA, onPick: noop, onClose: noop }))
  it('is a labelled modal dialog', () => {
    const html = dlg([an()]); assert.ok(html.includes('role="dialog"') && html.includes('aria-modal="true"') && /aria-labelledby="[^"]+"/.test(html)); assert.match(text(html), /Discuss an analysis/); assert.match(text(html), /about that film only/)
  })
  it('lists this tab’s analyses with their class, newest first as given', () => {
    const html = dlg([an({ runId: 'c'.repeat(32), title: 'Gamma Film', predictedClass: 'Flop' }), an()]), t = text(html)
    assert.equal((html.match(/data-testid="picker-item"/g) ?? []).length, 2); assert.ok(t.indexOf('Gamma Film') < t.indexOf('Alpha Film')); assert.match(t, /FLOP/i); assert.match(t, /HIT/i)
  })
  it('says plainly when nothing has been analysed in this tab', () => {
    const html = dlg([]); assert.ok(html.includes('data-testid="picker-empty"')); assert.match(text(html), /No analysis has been completed in this browser tab yet/); assert.match(text(html), /AI Movie Researcher/)
  })
  it('has a labelled id field whose button is disabled until the id is valid', () => {
    const html = dlg([]); assert.ok(/<label[^>]*>Or paste an analysis id<\/label>/.test(html)); assert.ok(isDisabled(html, 'picker-use')); assert.match(text(html), /server checks that it exists and has finished/)
  })
})

describe('conversation list', () => {
  const state = (cs, active) => ({ conversations: cs, activeId: active, limits: { maxChars: 2000, available: null } })
  const conv = (id, movie) => ({ id, createdAt: 0, movie, messages: [], pending: null, error: null, draft: '' })
  it('marks movie conversations with a labelled film icon and keeps the other ones plain', () => {
    const html = render(h(C.ConversationRail, { state: state([conv('m', filmA), conv('g', null)], 'm'), onSelect: noop, onNew: noop, onNewMovie: noop, canAdd: true }))
    assert.equal((html.match(/aria-label="Movie analysis"/g) ?? []).length, 1); assert.match(text(html), /Alpha Film · Hit/); assert.match(text(html), /New conversation/); assert.equal((html.match(/aria-current="true"/g) ?? []).length, 1)
  })
  it('has separate New (general) and Movie buttons, both disabled at the conversation cap', () => {
    const open = render(h(C.ConversationRail, { state: state([conv('g', null)], 'g'), onSelect: noop, onNew: noop, onNewMovie: noop, canAdd: true }))
    assert.ok(open.includes('data-testid="new-conversation"') && open.includes('data-testid="new-movie-conversation"') && !isDisabled(open, 'new-movie-conversation'))
    const full = render(h(C.ConversationRail, { state: state([conv('g', null)], 'g'), onSelect: noop, onNew: noop, onNewMovie: noop, canAdd: false }))
    assert.ok(isDisabled(full, 'new-conversation') && isDisabled(full, 'new-movie-conversation'))
  })
  it('run problems are announced with their own title', () => {
    const html = render(h(C.ErrorBanner, { error: { kind: 'run_unavailable', message: Chat.ERROR_TEXT.run_unavailable, retryable: false, until: null, at: 0 }, waitMs: 0, onRetry: noop, onDismiss: noop }))
    assert.match(text(html), /Analysis unavailable/); assert.ok(html.includes('role="alert"') && !html.includes('data-testid="retry"'))
  })
})

describe('the page', () => {
  const page = (entry = '/assistant') => render(h(MemoryRouter, { initialEntries: [entry] }, h(P.default)))
  it('starts as a general conversation: no film chip, general mode, the general starters', () => {
    const html = page(); assert.ok(html.includes('data-mode="general"') && !html.includes('data-testid="movie-chip"') && html.includes('data-testid="empty-chat"')); assert.match(text(html), /General mode/)
    assert.match(text(html), /explains one stored analysis/); assert.ok(!html.includes('data-testid="offer"'))
  })
  it('an offer from the Researcher is shown but nothing is switched automatically', () => {
    Reg.analysisRegistry().add(filmA)
    const html = render(h(MemoryRouter, { initialEntries: [{ pathname: '/assistant', state: { runId: filmA.runId } }] }, h(P.default)))
    assert.ok(html.includes('data-testid="offer"')); assert.match(text(html), /Talk about Alpha Film ?\?/); assert.ok(html.includes('data-mode="general"'), 'the active conversation is still the general one'); assert.ok(!html.includes('data-testid="movie-chip"'))
  })
  it('a malformed or unknown hand-off id shows no offer', () => {
    for (const runId of ['../../etc/passwd', 'zz', 42, 'd'.repeat(32)]) assert.ok(!render(h(MemoryRouter, { initialEntries: [{ pathname: '/assistant', state: { runId } }] }, h(P.default))).includes('data-testid="offer"'), String(runId))
  })
  it('with a movie conversation active it shows the chip, movie mode, movie starters and the right placeholder', () => {
    return vite.ssrLoadModule('/src/lib/useChat.ts').then(({ chatStore }) => {
      const s = chatStore(); const before = s.getState().activeId; const id = s.newConversation(filmA)
      try {
        const html = page(), t = text(html)
        assert.ok(html.includes('data-mode="movie"') && html.includes('data-testid="movie-chip"') && html.includes('data-testid="empty-movie-chat"')); assert.match(t, /Movie analysis/); assert.match(t, /Discussing Alpha Film/)
        assert.equal((html.match(/data-testid="starter"/g) ?? []).length, 7); assert.ok(html.includes('placeholder="Ask about this analysis…"')); assert.ok(!html.includes('data-testid="empty-chat"'))
      } finally { s.remove(id); s.select(before) }
    })
  })
})

describe('Researcher result', () => {
  const run = () => A.normalizeRun(fx.completed())
  it('offers "Discuss in AI Assistant" only when a handler is given, and says it cannot change the prediction', () => {
    const withCta = render(h(RV.ResultView, { run: run(), onDiscuss: noop })), without = render(h(RV.ResultView, { run: run() }))
    assert.ok(withCta.includes('data-testid="discuss-cta"') && withCta.includes('data-testid="discuss"')); assert.match(text(withCta), /Discuss in AI Assistant/); assert.match(text(withCta), /cannot change the prediction/)
    assert.ok(!without.includes('data-testid="discuss-cta"'))
  })
  it('the call to action sits before the prediction panel and does not alter it', () => {
    const html = render(h(RV.ResultView, { run: run(), onDiscuss: noop })); assert.ok(html.indexOf('data-testid="discuss-cta"') < html.indexOf('id="agent-prediction"')); assert.ok(html.includes('data-testid="predicted-label"'))
  })
})
