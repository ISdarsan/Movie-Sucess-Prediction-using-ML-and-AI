/* Real-browser checks for the movie-aware AI Assistant (installed Edge/Chrome through Playwright; Playwright is NOT a project dependency).
     PLAYWRIGHT_ENTRY=.../playwright/index.mjs BROWSER_EXE=".../msedge.exe" BASE=http://localhost:4174 \
     node --experimental-strip-types --disable-warning=ExperimentalWarning frontend/tests/browser/movie.browser.mjs
   Every API answer comes from fixtures (chat, chat/movie summaries, agent runs), so NO Gemini call is made and no database is touched.
   SHOTS_DIR receives screenshots. Exit code 1 if any check fails. */
import assert from 'node:assert/strict'
import { mkdirSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import * as fx from '../fixtures.ts'

const BASE = process.env.BASE ?? 'http://localhost:4173'
const SHOTS = process.env.SHOTS_DIR ?? './shots'
const ONLY = process.env.ONLY ? new RegExp(process.env.ONLY) : null
mkdirSync(SHOTS, { recursive: true })
const { chromium } = await import(pathToFileURL(process.env.PLAYWRIGHT_ENTRY).href)
const browser = await chromium.launch({ executablePath: process.env.BROWSER_EXE, headless: true })
const VIEWS = { desktop: { width: 1440, height: 900 }, tablet: { width: 820, height: 1180 }, mobile: { width: 390, height: 844 } }
const results = [], problems = [], foreign = []
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const ok = (c, m) => { if (!c) throw new Error(m) }
const eq = (a, b, m) => { try { assert.deepStrictEqual(a, b) } catch { throw new Error(`${m}: got ${JSON.stringify(a)} expected ${JSON.stringify(b)}`) } }
const json = (route, status, obj, headers = {}) => route.fulfill({ status, contentType: 'application/json', headers, body: JSON.stringify(obj) })
async function check(name, fn) {
  if (ONLY && !ONLY.test(name)) return
  const t = Date.now()
  try { const d = await fn(); results.push({ name, ok: true }); console.log(`PASS  ${name}${d ? '  — ' + d : ''}  (${((Date.now() - t) / 1000).toFixed(1)}s)`) }
  catch (e) { results.push({ name, ok: false }); console.log(`FAIL  ${name}  — ${String(e.message ?? e).split('\n')[0].slice(0, 400)}`) }
}

const STATUS_OK = { provider: 'gemini', configured: true, model: 'test-model', missing: [], problems: [], limits: { max_input_chars: 2000, max_output_tokens: 512, movie_max_output_tokens: 1024 }, modes: { general: 'available', movie: 'available' } }
const RUN_A = 'a'.repeat(32), RUN_B = 'b'.repeat(32), RUN_MISSING = 'c'.repeat(32), RUN_PENDING = 'd'.repeat(32)
const summary = (runId, o = {}) => ({ run_id: runId, title: null, predicted_class: 'Hit', class_scores: { Flop: 0.02, Average: 0.34, Hit: 0.64 }, year: 2010, genres: 'Action / Sci-Fi', country: 'USA', content_rating: 'PG-13', has_written_analysis: true, research_sources: 2, comparable_films: 3, risk_items: 1, ...o })
const SUMMARIES = {
  [RUN_A]: summary(RUN_A, { title: 'Alpha Film' }),
  [RUN_B]: summary(RUN_B, { title: null, predicted_class: 'Flop', year: 1999, genres: 'Horror' }),
  [fx.RUN_ID]: summary(fx.RUN_ID, { title: 'Inception', year: 2010, genres: 'Drama / Romance' }),
}

async function newPage(view = 'desktop', o = {}) {
  const ctx = await browser.newContext({ viewport: VIEWS[view], reducedMotion: o.reducedMotion ? 'reduce' : 'no-preference' })
  const page = await ctx.newPage(); page.setDefaultTimeout(15000)
  page.on('console', (m) => { if (m.type() === 'error' && !/ERR_ABORTED|ERR_CONNECTION_RESET|net::ERR_FAILED|Failed to load resource: the server responded with a status of (4\d\d|5\d\d)/.test(m.text())) problems.push(`${view}: ${m.text().slice(0, 200)}`) })
  page.on('pageerror', (e) => problems.push(`${view}: pageerror ${String(e).slice(0, 200)}`))
  page.on('request', (r) => { const u = new URL(r.url()); if (u.origin !== new URL(BASE).origin && !/^(data|blob):/.test(u.protocol)) foreign.push(r.url().slice(0, 120)) })
  return { ctx, page }
}

/** Fixtures for everything the assistant and Researcher pages call. */
async function mock(page, o = {}) {
  const log = { chat: [], summaries: [], agentPosts: 0 }
  await page.route('**/api/chat/status', (r) => (o.status ? o.status(r) : json(r, 200, STATUS_OK)))
  await page.route('**/api/chat/movie/*', (r) => {
    const id = new URL(r.request().url()).pathname.split('/').pop(); log.summaries.push(id)
    if (id === RUN_PENDING) return json(r, 409, { error: 'run_not_ready', message: 'x' })
    return SUMMARIES[id] ? json(r, 200, SUMMARIES[id]) : json(r, 404, { error: 'run_not_found', message: 'No analysis was found for that run.' })
  })
  await page.route('**/api/chat', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    const body = JSON.parse(route.request().postData() || '{}'); log.chat.push(body)
    if (o.delay) await sleep(o.delay)
    const custom = o.reply ? await o.reply(log.chat.length, body, route) : undefined
    if (custom === null) return
    if (custom) return json(route, custom.status, custom.body, custom.headers)
    return json(route, 200, { reply: `Reply ${log.chat.length} (${body.mode}) to: ${body.message}`, mode: body.mode, provider: 'gemini', model: 'test-model', truncated: false, ...(body.mode === 'movie' ? { context: SUMMARIES[body.run_id] ?? {} } : {}) })
  })
  await page.route('**/api/agent/runs**', (route) => {
    const req = route.request(), p = new URL(req.url()).pathname
    if (req.method() === 'POST') { log.agentPosts++; return json(route, 202, fx.ACCEPTED) }
    if (p.includes('/by-key/')) return json(route, 404, { error: 'not_found', message: 'x' })
    return json(route, 200, fx.withResearch(fx.research({ query: { title: 'Inception', genres: ['Drama', 'Romance'] } })))
  })
  return log
}
const open = async (page, path = '/assistant') => { await page.goto(BASE + path, { waitUntil: 'networkidle' }); await page.waitForSelector('main#main') }
const send = async (page, text) => { await page.fill('[data-testid=composer]', text); await page.keyboard.press('Enter') }
const shot = (page, name, full = false) => page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: full })
const noOverflow = (page) => page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: innerWidth }))
const waitAnswers = (page, n) => page.waitForFunction((n) => document.querySelectorAll('[data-testid=message][data-role=assistant]').length === n, n)
const mode = (page) => page.locator('[data-testid=chat-panel]').getAttribute('data-mode')

/** Open the picker and start a conversation about a pasted analysis id. */
async function startByPaste(page, runId) {
  await page.locator('[data-testid=new-movie-conversation]').click(); await page.getByRole('dialog').waitFor()
  await page.fill('[data-testid=picker-id]', runId); await page.locator('[data-testid=picker-use]').click(); await page.getByRole('dialog').waitFor({ state: 'detached' })
}

/* =========================================================== layout geometry (a screenshot once showed the rail buttons overlapping the chat panel) */
for (const view of ['desktop', 'tablet', 'mobile']) {
  await check(`[layout] the conversation controls stay inside their column and never overlap the chat at ${view}`, async () => {
    const { ctx, page } = await newPage(view); await mock(page); await open(page); await startByPaste(page, RUN_A); await send(page, 'q'); await waitAnswers(page, 1)
    const rect = (sel) => page.locator(sel).first().boundingBox(); const panel = await rect('[data-testid=chat-panel]')
    const intersects = (a, b) => a.x < b.x + b.width - 1 && a.x + a.width > b.x + 1 && a.y < b.y + b.height - 1 && a.y + a.height > b.y + 1
    for (const sel of ['[data-testid=new-conversation]', '[data-testid=new-movie-conversation]', '[data-testid=conversation]']) {
      const b = await rect(sel); ok(b.x >= 0 && b.x + b.width <= VIEWS[view].width + 1, `${sel} outside the viewport ${JSON.stringify(b)}`); ok(!intersects(b, panel), `${sel} overlaps the chat panel: ${JSON.stringify(b)} vs ${JSON.stringify(panel)}`)
    }
    const rail = await rect('nav[aria-label=Conversations]'); const nm = await rect('[data-testid=new-movie-conversation]'); ok(nm.x + nm.width <= rail.x + rail.width + 1 || view === 'mobile', `Movie button leaves its column: ${JSON.stringify({ rail, nm })}`)
    await shot(page, `m-rail-${view}`, false); await ctx.close()
  })
}

/* =========================================================== picking an analysis */
await check('[picker] empty list, id validation, and starting a movie conversation from a pasted analysis id', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page); await open(page)
  eq(await mode(page), 'general', 'starts general'); await page.locator('[data-testid=new-movie-conversation]').click()
  const dlg = page.getByRole('dialog'); await dlg.waitFor(); eq(await dlg.getAttribute('aria-modal'), 'true', 'modal'); ok(/No analysis has been completed in this browser tab yet/.test(await dlg.innerText()), 'empty text')
  ok(await page.locator('[data-testid=picker-use]').isDisabled(), 'Use disabled when empty'); await page.fill('[data-testid=picker-id]', 'not-an-id'); eq(await page.locator('[data-testid=picker-id]').getAttribute('aria-invalid'), 'true', 'invalid flagged'); ok(await page.locator('[data-testid=picker-use]').isDisabled(), 'Use disabled for a bad id')
  await shot(page, 'm-picker-empty')
  await page.fill('[data-testid=picker-id]', RUN_MISSING); await page.locator('[data-testid=picker-use]').click(); const err = page.locator('[data-testid=picker-error]'); await err.waitFor(); ok(/not found on the server/.test(await err.innerText()), `error: ${await err.innerText()}`)
  eq(await page.locator('[data-testid=conversation]').count(), 1, 'no conversation was created for an unknown analysis'); eq(log.chat.length, 0, 'nothing sent')
  await page.fill('[data-testid=picker-id]', RUN_A.toUpperCase()); await page.locator('[data-testid=picker-use]').click(); await dlg.waitFor({ state: 'detached' })
  eq(await mode(page), 'movie', 'movie conversation active'); eq(await page.locator('[data-testid=conversation]').count(), 2, 'the general conversation is kept'); eq(log.summaries.at(-1), RUN_A, 'validated against the server')
  ok(/Discussing\s*Alpha Film/.test((await page.locator('[data-testid=movie-chip]').innerText()).replace(/\s+/g, ' ')), 'chip names the film'); eq(await page.locator('[data-testid=conversation-title]').innerText(), 'Alpha Film · Hit', 'title')
  ok(/Movie analysis/.test(await page.locator('[data-testid=conversation-mode]').innerText()), 'mode label'); eq(await page.locator('[data-testid=starter]').count(), 7, 'seven movie starters'); eq(await page.locator('[data-testid=composer]').getAttribute('placeholder'), 'Ask about this analysis…', 'placeholder')
  ok(await page.evaluate(() => document.activeElement?.getAttribute('data-testid') === 'composer'), 'composer focused'); await shot(page, 'm-movie-empty'); await ctx.close()
})

/* =========================================================== asking, follow-ups, history */
await check('[ask] a suggested question sends run id + empty history; a follow-up carries the earlier turns; nothing about the film leaves the browser', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page, { delay: 400 }); await open(page); await startByPaste(page, RUN_A)
  await page.getByRole('button', { name: 'What are the biggest risks?' }).click(); await page.waitForSelector('[data-testid=thinking]'); await waitAnswers(page, 1)
  eq(log.chat[0], { message: 'What are the biggest risks?', mode: 'movie', run_id: RUN_A, history: [] }, 'first request')
  await send(page, 'And what did the research find?'); await waitAnswers(page, 2)
  eq(log.chat[1], { message: 'And what did the research find?', mode: 'movie', run_id: RUN_A, history: [{ role: 'user', text: 'What are the biggest risks?' }, { role: 'assistant', text: 'Reply 1 (movie) to: What are the biggest risks?' }] }, 'follow-up carries the earlier exchange')
  const blob = JSON.stringify(log.chat); for (const leak of ['Alpha', 'Hit', 'Flop', 'probabilit', 'class_scores', 'Action', '2010', 'predicted']) ok(!blob.includes(leak) || leak === 'Hit' && !/"Hit"/.test(blob), `film data in a request: ${leak}`)
  ok(!/"(title|label|prediction|scores|inputs|context)"/.test(blob), 'the browser sends no film fields'); await shot(page, 'm-conversation', false); await ctx.close()
})
await check('[ask] general conversations are unchanged next to a movie conversation (exact two-key body, no run id, no history)', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page); await open(page); await startByPaste(page, RUN_A)
  await send(page, 'movie question'); await waitAnswers(page, 1)
  await page.locator('[data-testid=new-conversation]').click(); eq(await mode(page), 'general', 'new general conversation'); ok(!(await page.locator('[data-testid=movie-chip]').count()), 'no chip in general mode')
  await send(page, 'general question'); await waitAnswers(page, 1); eq(log.chat[1], { message: 'general question', mode: 'general' }, 'general body'); eq(Object.keys(log.chat[1]).sort(), ['message', 'mode'], 'keys')
  ok(!(await page.locator('[data-testid=message-log]').innerText()).includes('movie question'), 'the movie thread did not leak into the general one'); await ctx.close()
})

/* =========================================================== switching movies */
await check('[switch] changing movie starts a NEW conversation: threads, run ids and histories never mix', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page); await open(page); await startByPaste(page, RUN_A)
  await send(page, 'about A'); await waitAnswers(page, 1)
  await page.locator('[data-testid=change-movie]').click(); await page.getByRole('dialog').waitFor(); await page.fill('[data-testid=picker-id]', RUN_B); await page.locator('[data-testid=picker-use]').click(); await page.getByRole('dialog').waitFor({ state: 'detached' })
  eq(await page.locator('[data-testid=conversation]').count(), 3, 'general + A + B'); eq(await page.locator('[data-testid=message]').count(), 0, 'the new conversation is empty'); ok(/Horror · 1999/.test(await page.locator('[data-testid=movie-chip]').innerText()), 'chip now shows B')
  await send(page, 'about B'); await waitAnswers(page, 1); eq(log.chat[1], { message: 'about B', mode: 'movie', run_id: RUN_B, history: [] }, "B starts with no history from A")
  const items = page.locator('[data-testid=conversation]'); const titles = await items.allInnerTexts(); ok(titles.some((t) => t.includes('Alpha Film')) && titles.some((t) => t.includes('Horror · 1999')), `titles ${JSON.stringify(titles)}`)
  await page.locator('[data-testid=conversation]', { hasText: 'Alpha Film' }).click(); ok(/Alpha Film/.test(await page.locator('[data-testid=movie-chip]').innerText()), 'back on A'); ok((await page.locator('[data-testid=message-log]').innerText()).includes('about A') && !(await page.locator('[data-testid=message-log]').innerText()).includes('about B'), 'A thread intact, no B')
  await send(page, 'more on A'); await waitAnswers(page, 2); eq(log.chat[2].run_id, RUN_A, 'A follow-up uses A'); eq(log.chat[2].history.map((t) => t.text).join('|').includes('about B'), false, 'no B text in A history'); eq(log.chat[2].history.length, 2, 'only A turns')
  await shot(page, 'm-switched', false); await ctx.close()
})
await check('[switch] an answer that arrives after switching films lands in the conversation that asked', async () => {
  const { ctx, page } = await newPage('desktop'); await mock(page, { reply: (n) => (n === 1 ? new Promise((res) => setTimeout(() => res(undefined), 1500)) : undefined) }); await open(page); await startByPaste(page, RUN_A)
  await send(page, 'slow question on A'); await page.waitForSelector('[data-testid=thinking]'); await startByPaste(page, RUN_B); await send(page, 'quick on B'); await waitAnswers(page, 1)
  await sleep(1700); ok(!(await page.locator('[data-testid=message-log]').innerText()).includes('slow question'), "A's late answer must not appear in B"); await page.locator('[data-testid=conversation]', { hasText: 'Alpha Film' }).click()
  const t = await page.locator('[data-testid=message-log]').innerText(); ok(t.includes('slow question on A') && /Reply \d \(movie\) to: slow question on A/.test(t) && !t.includes('quick on B'), 'A got its own answer'); await ctx.close()
})
await check('[switch] reopening a film that already has a conversation does not create a duplicate', async () => {
  const { ctx, page } = await newPage('desktop'); await mock(page); await open(page); await startByPaste(page, RUN_A); await send(page, 'hello'); await waitAnswers(page, 1)
  await startByPaste(page, RUN_A); eq(await page.locator('[data-testid=conversation]').count(), 2, 'still general + one A'); ok((await page.locator('[data-testid=message-log]').innerText()).includes('hello'), 'the existing thread is shown'); await ctx.close()
})

/* =========================================================== the hand-off from the Researcher */
await check('[hand-off] "Discuss in AI Assistant" offers a conversation; Not now dismisses it; Start begins it; discussing again reopens it', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page); await open(page, '/researcher'); await page.waitForSelector('#r-title_year')
  await page.fill('#r-title_year', '2010'); await page.fill('#r-duration', '120'); for (const g of ['Drama', 'Romance']) await page.getByRole('button', { name: g, exact: true }).click()
  await page.selectOption('#r-country', 'USA'); await page.selectOption('#r-content_rating', 'PG-13'); await page.fill('#r-research_title', 'Inception'); await page.getByRole('button', { name: /Research this film/ }).click()
  await page.waitForSelector('[aria-label="Research result"][data-phase="completed"]', { timeout: 20000 }); ok(await page.locator('[data-testid=discuss-cta]').isVisible(), 'CTA visible'); await shot(page, 'm-researcher-cta', false)
  await page.locator('[data-testid=discuss]').click(); await page.waitForURL(/\/assistant/); await page.waitForSelector('[data-testid=offer]')
  eq(await mode(page), 'general', 'NOT switched automatically'); eq(await page.locator('[data-testid=conversation]').count(), 1, 'no conversation created yet'); ok(/Inception/.test(await page.locator('[data-testid=offer-title]').innerText()), 'offer names the film'); await shot(page, 'm-offer', false)
  await page.locator('[data-testid=offer-dismiss]').click(); await page.waitForSelector('[data-testid=offer]', { state: 'detached' }); eq(await mode(page), 'general', 'still general after Not now'); eq(await page.locator('[data-testid=conversation]').count(), 1, 'nothing created')
  await page.goBack(); await page.waitForURL(/\/researcher/); await page.locator('[data-testid=discuss]').click(); await page.waitForSelector('[data-testid=offer]')
  await page.locator('[data-testid=offer-accept]').click(); await page.waitForSelector('[data-testid=offer]', { state: 'detached' }); eq(await mode(page), 'movie', 'movie conversation started'); ok(/Inception/.test(await page.locator('[data-testid=movie-chip]').innerText()), 'chip'); eq(await page.locator('[data-testid=conversation]').count(), 2, 'general + movie')
  await send(page, 'Summarize the entire analysis.'); await waitAnswers(page, 1); eq(log.chat[0].run_id, fx.RUN_ID, 'the run id from the Researcher result')
  await page.locator('header nav[aria-label=Primary] a[title="AI Movie Researcher"]').click(); await page.waitForURL(/\/researcher/); await page.locator('[data-testid=discuss]').click().catch(() => {})
  if (!(await page.locator('[data-testid=offer]').count())) { await page.waitForURL(/\/assistant/) }
  await page.waitForSelector('[data-testid=offer]'); ok(/already have a conversation/.test(await page.locator('[data-testid=offer]').innerText()), 'offers the existing one'); await page.locator('[data-testid=offer-accept]').click()
  eq(await page.locator('[data-testid=conversation]').count(), 2, 'no duplicate conversation'); ok((await page.locator('[data-testid=message-log]').innerText()).includes('Summarize the entire analysis.'), 'the existing thread is reopened'); eq(log.agentPosts, 1, 'one analysis run'); await ctx.close()
})
await check('[hand-off] analyses finished in this tab appear in the picker', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page); await open(page, '/researcher'); await page.waitForSelector('#r-title_year')
  await page.fill('#r-title_year', '2010'); await page.fill('#r-duration', '120'); await page.getByRole('button', { name: 'Drama', exact: true }).click(); await page.selectOption('#r-country', 'USA'); await page.selectOption('#r-content_rating', 'PG-13'); await page.fill('#r-research_title', 'Inception')
  await page.getByRole('button', { name: /Research this film/ }).click(); await page.waitForSelector('[aria-label="Research result"][data-phase="completed"]', { timeout: 20000 })
  await page.locator('header nav[aria-label=Primary] a[title="AI Assistant"]').click(); await page.waitForSelector('[data-testid=composer]'); await page.locator('[data-testid=new-movie-conversation]').click()
  const items = page.locator('[data-testid=picker-item]'); eq(await items.count(), 1, 'one analysis listed'); ok((await items.first().innerText()).includes('Inception') && /HIT/i.test(await items.first().innerText()), 'named and classed')
  await items.first().click(); await page.getByRole('dialog').waitFor({ state: 'detached' }); eq(await mode(page), 'movie', 'started'); ok(log.summaries.includes(fx.RUN_ID), 'confirmed with the server before starting'); await ctx.close()
})

/* =========================================================== errors, clear, unavailable */
await check('[errors] an unavailable analysis is explained, offers no retry, and never falls back to a general answer', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page, { reply: () => ({ status: 404, body: { error: 'run_not_found', message: 'RAW SERVER TEXT AIzaSyNOTSHOWN' } }) }); await open(page); await startByPaste(page, RUN_A)
  await send(page, 'hello'); const err = page.locator('[data-testid=chat-error]'); await err.waitFor(); eq(await err.getAttribute('data-kind'), 'run_unavailable', 'kind'); ok(/not found on the server/.test(await err.innerText()), 'wording'); eq(await page.locator('[data-testid=retry]').count(), 0, 'no retry')
  ok(!/RAW SERVER|AIza/.test(await page.locator('body').innerText()), 'raw text shown'); eq(log.chat.length, 1, 'exactly one request'); eq(log.chat[0].mode, 'movie', 'it was a movie request'); eq(await mode(page), 'movie', 'still a movie conversation'); await shot(page, 'm-run-unavailable', false); await ctx.close()
})
await check('[errors] an analysis that has not finished can be retried later, with the same run id', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page, { reply: (n) => (n === 1 ? { status: 409, body: { error: 'run_not_ready', message: 'x' } } : undefined) }); await open(page); await startByPaste(page, RUN_A)
  await send(page, 'too early'); await page.locator('[data-testid=chat-error]').waitFor(); ok(/not finished yet/.test(await page.locator('[data-testid=chat-error]').innerText()), 'wording'); await page.locator('[data-testid=retry]').click(); await waitAnswers(page, 1)
  eq(log.chat.length, 2, 'two requests'); eq(log.chat[1], log.chat[0], 'identical retry'); eq(await page.locator('[data-testid=message][data-role=user]').count(), 1, 'one user bubble'); await ctx.close()
})
await check('[errors] picking an analysis that is still running or missing is refused before any conversation exists', async () => {
  const { ctx, page } = await newPage('desktop'); await mock(page); await open(page); await page.locator('[data-testid=new-movie-conversation]').click(); await page.fill('[data-testid=picker-id]', RUN_PENDING); await page.locator('[data-testid=picker-use]').click()
  const err = page.locator('[data-testid=picker-error]'); await err.waitFor(); ok(/not finished yet/.test(await err.innerText()), await err.innerText()); await page.keyboard.press('Escape'); eq(await page.locator('[data-testid=conversation]').count(), 1, 'nothing created'); await ctx.close()
})
await check('[clear] clearing a movie conversation keeps its film, removes the thread, and sends no old history afterwards', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page); await open(page); await startByPaste(page, RUN_A); await send(page, 'one'); await waitAnswers(page, 1)
  await page.locator('[data-testid=clear]').click(); await page.getByTestId('confirm-ok').click(); await page.waitForSelector('[data-testid=empty-movie-chat]'); ok(/Alpha Film/.test(await page.locator('[data-testid=movie-chip]').innerText()), 'chip survives Clear')
  await send(page, 'two'); await waitAnswers(page, 1); eq(log.chat[1], { message: 'two', mode: 'movie', run_id: RUN_A, history: [] }, 'no history after Clear'); await ctx.close()
})
await check('[unavailable] when the assistant is not configured, movie conversations cannot send either', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mock(page, { status: (r) => json(r, 200, { ...STATUS_OK, configured: false, model: null, missing: ['GEMINI_API_KEY'] }) }); await open(page); await startByPaste(page, RUN_A)
  ok(await page.locator('[data-testid=composer]').isDisabled(), 'composer disabled'); for (const s of await page.locator('[data-testid=starter]').all()) ok(await s.isDisabled(), 'starter disabled'); eq(log.chat.length, 0, 'nothing sent'); await ctx.close()
})

/* =========================================================== layout, keyboard, privacy, a11y */
for (const view of ['tablet', 'mobile']) {
  await check(`[layout] movie conversation and picker at ${view}: chip wraps, nothing overflows, targets are big enough`, async () => {
    const { ctx, page } = await newPage(view); await mock(page); await open(page); await page.locator('[data-testid=new-movie-conversation]').click(); await page.getByRole('dialog').waitFor()
    const d = await page.getByRole('dialog').boundingBox(); ok(d.x >= 0 && d.x + d.width <= VIEWS[view].width && d.y >= 0 && d.y + d.height <= VIEWS[view].height + 1, `dialog box ${JSON.stringify(d)}`); await shot(page, `m-picker-${view}`)
    await page.fill('[data-testid=picker-id]', RUN_A); await page.locator('[data-testid=picker-use]').click(); await page.getByRole('dialog').waitFor({ state: 'detached' })
    const o = await noOverflow(page); ok(o.sw <= o.iw + 1, `overflow ${o.sw}>${o.iw}`); const chip = await page.locator('[data-testid=movie-chip]').boundingBox(); ok(chip.x >= 0 && chip.x + chip.width <= VIEWS[view].width + 1, 'chip inside viewport')
    const cm = await page.locator('[data-testid=change-movie]').boundingBox(); ok(cm.height >= 40, `Change movie target ${cm.height}px`); await send(page, 'question'); await waitAnswers(page, 1); const o2 = await noOverflow(page); ok(o2.sw <= o2.iw + 1, 'no overflow after answering'); await shot(page, `m-conversation-${view}`); await ctx.close()
  })
}
await check('[keyboard] the picker traps focus, closes with Escape and returns focus; items are operable with Enter', async () => {
  const { ctx, page } = await newPage('desktop'); await mock(page); await open(page); await startByPaste(page, RUN_A); await send(page, 'q'); await waitAnswers(page, 1)
  await page.locator('[data-testid=change-movie]').focus(); await page.keyboard.press('Enter'); const dlg = page.getByRole('dialog'); await dlg.waitFor()
  const inside = () => page.evaluate(() => document.querySelector('[role=dialog]')?.contains(document.activeElement)); ok(await inside(), 'focus starts inside the dialog')
  for (let i = 0; i < 8; i++) { await page.keyboard.press('Tab'); ok(await inside(), `Tab ${i + 1} escaped the dialog`) }
  for (let i = 0; i < 8; i++) { await page.keyboard.press('Shift+Tab'); ok(await inside(), `Shift+Tab ${i + 1} escaped the dialog`) }
  await page.keyboard.press('Escape'); await dlg.waitFor({ state: 'detached' }); ok(await page.evaluate(() => document.activeElement?.getAttribute('data-testid') === 'change-movie'), 'focus returns to the opener')
  await page.locator('[data-testid=change-movie]').focus(); await page.keyboard.press('Enter'); await dlg.waitFor(); await page.locator('[data-testid=picker-item]').first().focus(); await page.keyboard.press('Enter'); await dlg.waitFor({ state: 'detached' }); await ctx.close()
})
await check('[privacy] chat adds no browser storage, sends only chat endpoints and ids, and talks to no other origin', async () => {
  const before = foreign.length
  const { ctx, page } = await newPage('desktop'); const log = await mock(page); const urls = []; page.on('request', (r) => { if (r.url().includes('/api/')) urls.push(new URL(r.url()).pathname) })
  await open(page); await startByPaste(page, RUN_A); await send(page, 'a private question about my script'); await waitAnswers(page, 1)
  const store = await page.evaluate(() => ({ ls: JSON.stringify({ ...localStorage }), ss: Object.keys(sessionStorage), cookie: document.cookie }))
  eq(store.ls, '{}', 'localStorage empty'); eq(store.ss, [], 'sessionStorage empty'); eq(store.cookie, '', 'no cookies')
  ok(urls.every((u) => u === '/api/health' || /^\/api\/chat(\/status|\/movie\/[0-9a-f]{12,32})?$/.test(u)), `unexpected API calls: ${[...new Set(urls)]}`); eq(foreign.slice(before), [], 'other origins')
  await page.reload({ waitUntil: 'networkidle' }); eq(await page.locator('[data-testid=message]').count(), 0, 'a refresh clears the conversation'); eq(await page.locator('[data-testid=conversation]').count(), 1, 'and the movie conversation'); eq(log.chat.length, 1, 'no request on reload'); await ctx.close()
})
await check('[a11y] chip, dialog and starters expose proper roles, names and live updates', async () => {
  const { ctx, page } = await newPage('desktop'); await mock(page); await open(page); await startByPaste(page, RUN_A)
  const a = await page.evaluate(() => ({ chipBtn: document.querySelector('[data-testid=change-movie]')?.textContent, starters: document.querySelector('[aria-label="Suggested questions"]')?.tagName, rail: [...document.querySelectorAll('[data-testid=conversation] [role=img]')].map((e) => e.getAttribute('aria-label')), current: document.querySelectorAll('[data-testid=conversation][aria-current=true]').length, h2: document.querySelector('[data-testid=empty-movie-chat] h2')?.textContent }))
  eq(a.chipBtn, 'Change movie', 'button text'); eq(a.starters, 'UL', 'list'); eq(a.rail, ['Movie analysis'], 'film icon is labelled'); eq(a.current, 1, 'one current'); eq(a.h2, 'Ask about this analysis', 'heading')
  await send(page, 'hi'); await waitAnswers(page, 1); eq(await page.locator('[data-testid=message-log]').getAttribute('aria-live'), 'polite', 'live region'); await ctx.close()
})
await check('[contrast] the new movie UI meets WCAG AA (approximate: gradients and blur are not modelled)', async () => {
  const { ctx, page } = await newPage('desktop'); await mock(page); await open(page); await startByPaste(page, RUN_A)
  const measure = () => {
    const cv = document.createElement('canvas'); cv.width = cv.height = 1; const cx = cv.getContext('2d', { willReadFrequently: true })
    const rgba = (css) => { cx.clearRect(0, 0, 1, 1); cx.fillStyle = '#000'; cx.fillStyle = css; cx.fillRect(0, 0, 1, 1); const d = cx.getImageData(0, 0, 1, 1).data; return [d[0], d[1], d[2], d[3] / 255] }
    const over = (f, b) => [0, 1, 2].map((i) => f[i] * f[3] + b[i] * (1 - f[3]))
    const lum = (c) => { const [r, g, b] = c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4 }); return 0.2126 * r + 0.7152 * g + 0.0722 * b }
    const bgOf = (el) => { const st = []; for (let e = el; e; e = e.parentElement) { const c = rgba(getComputedStyle(e).backgroundColor); if (c[3] > 0) st.push(c); if (c[3] === 1) break } let base = [6, 6, 26]; for (const c of st.reverse()) base = over(c, base).concat(1); return base.slice(0, 3) }
    return [['[data-testid=movie-chip-title]', 'chip title'], ['[data-testid=movie-chip] p', 'chip text'], ['[data-testid=change-movie]', 'change button'], ['[data-testid=empty-movie-chat] h2', 'empty title'], ['[data-testid=empty-movie-chat] li', 'disclaimer / starter'], ['[data-testid=starter]', 'starter'], ['[data-testid=new-movie-conversation]', 'Movie button']]
      .map(([sel, label]) => { const el = document.querySelector(sel); if (!el) return { label, missing: true }; const st = getComputedStyle(el), bg = bgOf(el), fg = over(rgba(st.color), bg), l1 = lum(fg), l2 = lum(bg), ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05), size = parseFloat(st.fontSize), large = size >= 24 || (size >= 18.66 && parseInt(st.fontWeight) >= 700); return { label, ratio: +ratio.toFixed(2), need: large ? 3 : 4.5 } })
  }
  const rows = (await page.evaluate(measure)).filter((r) => !r.missing); const fails = rows.filter((r) => r.ratio < r.need); ok(fails.length === 0, `below AA: ${JSON.stringify(fails)}`); await ctx.close(); return `min ratio ${Math.min(...rows.map((r) => r.ratio))} over ${rows.length} samples`
})
await check('[routes] all pages still load without console errors after the movie changes', async () => {
  const { ctx, page } = await newPage('desktop'); await mock(page); const before = problems.length
  for (const p of ['/', '/predict', '/performance', '/explain', '/explore', '/researcher', '/assistant', '/about']) { await page.goto(BASE + p, { waitUntil: 'networkidle' }); ok((await page.locator('h1').first().innerText()).length > 2, p) }
  ok(problems.length === before, `console problems: ${JSON.stringify(problems.slice(before))}`); await ctx.close()
})

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} checks passed`)
const distinct = [...new Set(problems)]
console.log(`console errors / page errors: ${distinct.length}`, distinct.slice(0, 6)); console.log(`requests to other origins: ${foreign.length}`, foreign.slice(0, 4))
process.exit(failed.length || distinct.length || foreign.length ? 1 : 0)
