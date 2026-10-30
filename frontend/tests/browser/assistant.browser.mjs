/* Real-browser checks for /assistant (installed Edge/Chrome through Playwright). Playwright is NOT a project dependency: point PLAYWRIGHT_ENTRY at an
   existing copy and BROWSER_EXE at an installed browser. Needs the built app on BASE (vite preview proxying /api) and a backend behind it.
     node --experimental-strip-types --disable-warning=ExperimentalWarning frontend/tests/browser/assistant.browser.mjs
   POST /api/chat and /api/chat/status are answered from fixtures, so NO Gemini call is made. Set REAL_GEMINI=1 to add ONE opt-in live message
   (a few tokens) through the real backend. SHOTS_DIR receives screenshots. Exit code 1 if any check fails. */
import assert from 'node:assert/strict'
import { mkdirSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const BASE = process.env.BASE ?? 'http://localhost:4173'
const SHOTS = process.env.SHOTS_DIR ?? './shots'
const ONLY = process.env.ONLY ? new RegExp(process.env.ONLY) : null
mkdirSync(SHOTS, { recursive: true })
const { chromium } = await import(pathToFileURL(process.env.PLAYWRIGHT_ENTRY).href)
const browser = await chromium.launch({ executablePath: process.env.BROWSER_EXE, headless: true })

const VIEWS = { desktop: { width: 1440, height: 900 }, 'laptop-1366': { width: 1366, height: 768 }, 'laptop-1280': { width: 1280, height: 720 }, laptop: { width: 1024, height: 768 }, tablet: { width: 820, height: 1180 }, 'tablet-768': { width: 768, height: 1024 }, mobile: { width: 390, height: 844 } }
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

const STATUS_OK = { provider: 'gemini', configured: true, model: 'test-model', missing: [], problems: [], limits: { max_input_chars: 2000, max_output_tokens: 512, timeout_seconds: 20, rate_limit_per_minute: 10 }, modes: { general: 'available', movie: 'not_implemented' } }

async function newPage(view = 'desktop', o = {}) {
  const ctx = await browser.newContext({ viewport: VIEWS[view], reducedMotion: o.reducedMotion ? 'reduce' : 'no-preference' })
  const page = await ctx.newPage(); page.setDefaultTimeout(o.timeout ?? 15000)
  page.on('console', (m) => { if (m.type() === 'error' && !/ERR_ABORTED|ERR_CONNECTION_RESET|net::ERR_FAILED|Failed to load resource: the server responded with a status of (4\d\d|5\d\d)/.test(m.text())) problems.push(`${view}: ${m.text().slice(0, 200)}`) })
  page.on('pageerror', (e) => problems.push(`${view}: pageerror ${String(e).slice(0, 200)}`))
  page.on('request', (r) => { const u = new URL(r.url()); if (u.origin !== new URL(BASE).origin && !u.protocol.startsWith('data') && !u.protocol.startsWith('blob')) foreign.push(r.url().slice(0, 120)) })
  return { ctx, page }
}

/** Answer /api/chat and /api/chat/status from fixtures. `reply(n, body)` may return {status, body, headers} or a function(route). */
async function mockChat(page, o = {}) {
  const log = { posts: [], statusCalls: 0 }
  await page.route('**/api/chat/status', (r) => { log.statusCalls++; return o.status ? o.status(r) : json(r, 200, STATUS_OK) })
  await page.route('**/api/chat', async (route) => {
    const req = route.request()
    if (req.method() !== 'POST') return route.fallback()
    const body = JSON.parse(req.postData() || '{}'); log.posts.push(body)
    const n = log.posts.length
    if (o.delay) await sleep(typeof o.delay === 'function' ? o.delay(n) : o.delay)
    const r = o.reply ? await o.reply(n, body, route) : { status: 200, body: { reply: `Answer ${n} to: ${body.message}`, mode: 'general', provider: 'gemini', model: 'test-model', truncated: false } }
    if (r === undefined) return                                       // the handler already responded or aborted
    return json(route, r.status, r.body, r.headers)
  })
  return log
}
const open = async (page) => { await page.goto(BASE + '/assistant', { waitUntil: 'networkidle' }); await page.waitForSelector('[data-testid=composer]') }
const send = async (page, text) => { await page.fill('[data-testid=composer]', text); await page.keyboard.press('Enter') }
const texts = (page) => page.locator('[data-testid=message]').allInnerTexts().then((a) => a.map((t) => t.replace(/\s+/g, ' ').trim()))
const isDisabled = (page, sel) => page.locator(sel).first().isDisabled()
const shot = (page, name, full = false) => page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: full })
const noOverflow = (page) => page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: innerWidth }))
const errBody = (code, message = 'RAW PROVIDER TEXT AIzaSyDONOTSHOW') => ({ error: code, message })

/* =========================================================== navigation and layout */
for (const view of ['desktop', 'laptop-1366', 'laptop-1280', 'laptop', 'tablet', 'tablet-768']) {
  await check(`[nav] eight items fit without overlap at ${view} (${VIEWS[view].width}px); AI Assistant is active`, async () => {
    const { ctx, page } = await newPage(view); await mockChat(page); await open(page)
    const links = page.locator('header nav[aria-label="Primary"] a'); eq(await links.count(), 8, 'nav items')
    const boxes = []
    for (let i = 0; i < 8; i++) { const b = await links.nth(i).boundingBox(); boxes.push(b); ok(await links.nth(i).isVisible(), `link ${i} hidden`); ok(b.x >= 0 && b.x + b.width <= VIEWS[view].width, `link ${i} outside viewport ${JSON.stringify(b)}`); ok(b.height >= 40, `link ${i} only ${b.height}px tall`) }
    for (let i = 1; i < 8; i++) ok(boxes[i].x >= boxes[i - 1].x + boxes[i - 1].width - 1, `links ${i - 1}/${i} overlap`)
    const nav = await page.locator('header nav[aria-label="Primary"]').boundingBox(), logo = await page.locator('header a[aria-label="FRAME home"]').boundingBox()
    ok(logo.x + logo.width <= nav.x + 1, 'logo overlaps nav')
    const right = await page.locator('header .glass > div').last().boundingBox(); ok(nav.x + nav.width <= right.x + 2, `nav (${Math.round(nav.x + nav.width)}) runs into the status cluster (${Math.round(right.x)})`)
    eq(await page.locator('header nav a[aria-current="page"]').getAttribute('title'), 'AI Assistant', 'active item')
    const o = await noOverflow(page); ok(o.sw <= o.iw + 1, `overflow ${o.sw}>${o.iw}`)
    await shot(page, `nav-${view}`); await ctx.close(); return `nav ${Math.round(nav.width)}px wide`
  })
}
await check('[nav] mobile menu lists eight items and links to the assistant', async () => {
  const { ctx, page } = await newPage('mobile'); await mockChat(page); await page.goto(BASE + '/', { waitUntil: 'networkidle' })
  await page.getByRole('button', { name: 'Open navigation' }).click(); const dlg = page.getByRole('dialog', { name: 'Navigation' }); eq(await dlg.locator('nav a').count(), 8, 'items')
  await dlg.getByRole('link', { name: 'AI Assistant' }).click(); await page.waitForURL(/\/assistant/); await page.waitForSelector('[data-testid=composer]'); await ctx.close()
})
for (const view of ['desktop', 'laptop', 'tablet', 'mobile']) {
  await check(`[layout] /assistant at ${view}: no horizontal scroll, composer usable, conversations arranged sensibly`, async () => {
    const { ctx, page } = await newPage(view); await mockChat(page); await open(page)
    const o = await noOverflow(page); ok(o.sw <= o.iw + 1, `overflow ${o.sw}>${o.iw}`)
    const rail = await page.locator('nav[aria-label=Conversations]').boundingBox(), panel = await page.locator('[data-testid=chat-panel]').boundingBox(), c = await page.locator('[data-testid=composer]').boundingBox()
    const side = panel.x > rail.x + rail.width - 10; ok(view === 'desktop' || view === 'laptop' ? side : !side, `rail ${JSON.stringify(rail)} panel ${JSON.stringify(panel)}`)
    ok(c.width >= 180 && c.x >= 0 && c.x + c.width <= VIEWS[view].width, `composer box ${JSON.stringify(c)}`)
    const send = await page.locator('[data-testid=send]').boundingBox(); ok(send.width >= 44 && send.height >= 44, `send target ${send.width}x${send.height}`)
    if (view === 'desktop') { const c2 = await page.locator('[data-testid=composer]').boundingBox(); ok(c2.y + c2.height <= VIEWS.desktop.height, `composer is below the fold at 1440x900 (bottom ${Math.round(c2.y + c2.height)})`) }
    await shot(page, `empty-${view}`, true); await ctx.close(); return side ? 'rail beside chat' : 'rail above chat'
  })
}

/* =========================================================== empty state, status and unavailable [mock] */
await check('[empty] starter prompts, disclaimer, disabled Send, status chip and "New conversation" title', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page); await open(page)
  const starters = await page.locator('[data-testid=starter]').allInnerTexts()
  eq(starters, ['Explain how Random Forest works.', 'What makes a movie commercially successful?', 'Explain the difference between Hit, Average and Flop.', 'How does our Movie Intelligence System work?', 'What factors affect movie performance?'], 'starters')
  ok(/cannot see your films, predictions or database/.test(await page.locator('[data-testid=empty-chat]').innerText()), 'general-mode disclaimer')
  ok(await isDisabled(page, '[data-testid=send]'), 'Send disabled when empty'); eq(await page.locator('[data-testid=conversation-title]').innerText(), 'New conversation', 'title')
  ok(/General mode · test-model/.test(await page.locator('[data-testid=chat-panel] header').innerText()), 'model chip from status')
  eq(await page.locator('[data-testid=message]').count(), 0, 'no fake messages'); await ctx.close()
})
await check('[unavailable] not configured: clear banner, box and starters disabled, nothing is sent', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page, { status: (r) => json(r, 200, { ...STATUS_OK, configured: false, model: null, missing: ['GEMINI_API_KEY', 'GEMINI_MODEL'] }) }); await open(page)
  const banner = page.locator('[data-testid=assistant-unavailable]'); await banner.waitFor(); ok(/not set up yet/.test(await banner.innerText()), 'banner text')
  ok(await isDisabled(page, '[data-testid=composer]'), 'composer disabled'); ok(await isDisabled(page, '[data-testid=send]'), 'send disabled')
  for (const s of await page.locator('[data-testid=starter]').all()) ok(await s.isDisabled(), 'starter disabled')
  await page.getByRole('button', { name: /For the administrator/ }).click(); ok(/GEMINI_API_KEY, GEMINI_MODEL/.test(await banner.innerText()), 'admin hint names the missing settings')
  ok(!/AIza|AQ\./.test(await page.locator('main').innerText()), 'no key-shaped text'); eq(log.posts.length, 0, 'POSTs'); await shot(page, 'unavailable', false); await ctx.close()
})
await check('[unavailable] an older backend without /api/chat/status (404) is reported as unavailable, not as a crash', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page, { status: (r) => json(r, 404, { error: 'http_error', message: 'Not Found' }) }); await open(page)
  await page.locator('[data-testid=assistant-unavailable]').waitFor(); ok(/not available on this server/.test(await page.locator('main').innerText()), 'text'); ok(await isDisabled(page, '[data-testid=composer]'), 'composer disabled'); await ctx.close()
})

/* =========================================================== sending, loading, markdown [mock] */
await check('[send] Enter sends the exact general-mode body, shows loading, then a formatted reply, and refocuses the box', async () => {
  const { ctx, page } = await newPage('desktop')
  const reply = '**Random Forest** is an ensemble.\n\n- many trees\n- majority vote\n\nUse `n_estimators`.\n\n<img src=x onerror=alert(1)> <script>window.__pwned=1</script>'
  const log = await mockChat(page, { delay: 900, reply: () => ({ status: 200, body: { reply, mode: 'general', provider: 'gemini', model: 'test-model', truncated: false } }) }); await open(page)
  await send(page, '  What is a Random Forest?  ')
  eq(log.posts.length >= 0, true, ''); await page.waitForSelector('[data-testid=thinking]'); ok(await isDisabled(page, '[data-testid=send]'), 'Send disabled while waiting')
  ok(/Waiting for the answer/.test(await page.locator('[data-testid=chat-panel] form').innerText()), 'status hint'); await shot(page, 'loading', false)
  await page.waitForSelector('[data-testid=message][data-role=assistant]'); await page.waitForSelector('[data-testid=thinking]', { state: 'detached' })
  eq(log.posts, [{ message: 'What is a Random Forest?', mode: 'general' }], 'request body (trimmed, general mode, no history)')
  const bubble = page.locator('[data-testid=message][data-role=assistant]')
  ok((await bubble.locator('strong').count()) === 1 && (await bubble.locator('li').count()) === 2 && (await bubble.locator('code').count()) === 1, 'markdown rendered as structure')
  eq(await bubble.locator('img, script, a').count(), 0, 'no img/script/link elements from model text'); eq(await page.evaluate(() => window.__pwned), undefined, 'script did not run')
  ok((await bubble.innerText()).includes('<img src=x onerror=alert(1)>'), 'hostile text shown as plain text')
  ok(await page.evaluate(() => document.activeElement?.getAttribute('data-testid') === 'composer'), 'focus back in the message box')
  eq(await page.locator('[data-testid=composer]').inputValue(), '', 'box cleared'); eq(await page.locator('[data-testid=conversation-title]').innerText(), 'What is a Random Forest?', 'title from first message')
  await shot(page, 'conversation-desktop', false); await ctx.close()
})
await check('[send] a starter prompt sends immediately and leaves the empty state', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page); await open(page)
  await page.getByRole('button', { name: 'What makes a movie commercially successful?' }).click(); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  eq(log.posts, [{ message: 'What makes a movie commercially successful?', mode: 'general' }], 'body'); eq(await page.locator('[data-testid=starter]').count(), 0, 'starters gone'); eq((await texts(page)).length, 2, 'two bubbles'); await ctx.close()
})
await check('[send] only the current message is sent, never the earlier turns', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page); await open(page)
  await send(page, 'first'); await page.waitForSelector('[data-testid=message][data-role=assistant]'); await send(page, 'second'); await page.waitForFunction(() => document.querySelectorAll('[data-testid=message][data-role=assistant]').length === 2)
  eq(log.posts, [{ message: 'first', mode: 'general' }, { message: 'second', mode: 'general' }], 'bodies'); await ctx.close()
})

/* =========================================================== keyboard [mock] */
await check('[keyboard] Shift+Enter makes a new line, Enter sends, Enter on an empty box does nothing', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page); await open(page)
  await page.click('[data-testid=composer]'); await page.keyboard.press('Enter'); await sleep(200); eq(log.posts.length, 0, 'empty Enter sent something')
  await page.keyboard.type('line one'); await page.keyboard.press('Shift+Enter'); await page.keyboard.type('line two')
  eq(await page.locator('[data-testid=composer]').inputValue(), 'line one\nline two', 'newline inserted'); eq(log.posts.length, 0, 'Shift+Enter must not send')
  const h = await page.locator('[data-testid=composer]').evaluate((e) => e.getBoundingClientRect().height); ok(h > 48, 'box grew for two lines')
  await page.keyboard.press('Enter'); await page.waitForSelector('[data-testid=message][data-role=assistant]'); eq(log.posts[0].message, 'line one\nline two', 'multi-line message sent intact'); await ctx.close()
})
await check('[keyboard] everything is reachable and operable with the keyboard, with visible focus', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page); await open(page)
  await page.locator('[data-testid=starter]').first().focus(); await page.keyboard.press('Enter'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  eq(log.posts.length, 1, 'starter via keyboard')
  await page.focus('[data-testid=composer]'); await page.keyboard.type('hello'); await page.keyboard.press('Tab')
  eq(await page.evaluate(() => document.activeElement?.getAttribute('data-testid')), 'send', 'Tab from the box reaches Send'); await page.keyboard.press('Space'); await page.waitForFunction(() => document.querySelectorAll('[data-testid=message]').length === 4)
  const ring = await page.locator('[data-testid=new-conversation]').evaluate((e) => { e.focus(); const s = getComputedStyle(e); return { outline: s.outlineStyle, width: parseFloat(s.outlineWidth) } }); ok(ring.outline !== 'none' && ring.width > 0, `focus ring ${JSON.stringify(ring)}`)
  await page.keyboard.press('Enter'); eq(await page.locator('[data-testid=conversation]').count(), 2, 'New conversation via keyboard'); await ctx.close()
})

/* =========================================================== duplicates [mock] */
await check('[duplicates] Enter x5, Send double-click and a double-clicked starter each make exactly ONE request', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page, { delay: 1200 }); await open(page)
  await page.fill('[data-testid=composer]', 'only once'); for (let i = 0; i < 5; i++) await page.keyboard.press('Enter'); await page.waitForSelector('[data-testid=thinking]')
  await page.waitForSelector('[data-testid=message][data-role=assistant]'); eq(log.posts.length, 1, 'Enter x5'); eq(await page.locator('[data-testid=message][data-role=user]').count(), 1, 'one user bubble')
  await page.fill('[data-testid=composer]', 'button twice'); await page.locator('[data-testid=send]').dblclick(); await page.waitForFunction(() => document.querySelectorAll('[data-testid=message][data-role=assistant]').length === 2); eq(log.posts.length, 2, 'Send double-click')
  const { ctx: c2, page: p2 } = await newPage('desktop'); const l2 = await mockChat(p2, { delay: 1000 }); await open(p2); await p2.locator('[data-testid=starter]').first().dblclick(); await p2.waitForSelector('[data-testid=message][data-role=assistant]'); eq(l2.posts.length, 1, 'starter double-click')
  await ctx.close(); await c2.close()
})
await check('[duplicates] while an answer is pending the box accepts typing but Send stays disabled, and a second conversation can still send', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page, { delay: 1500 }); await open(page)
  await send(page, 'slow one'); await page.waitForSelector('[data-testid=thinking]'); await page.fill('[data-testid=composer]', 'next thought'); ok(await isDisabled(page, '[data-testid=send]'), 'Send disabled'); await page.keyboard.press('Enter'); await sleep(150); eq(log.posts.length, 1, 'Enter while pending')
  eq(await page.locator('[data-testid=composer]').inputValue(), 'next thought', 'typed text kept')
  await page.locator('[data-testid=new-conversation]').click(); await send(page, 'in B'); await sleep(200); eq(log.posts.length, 2, 'other conversation sends'); await ctx.close()
})

/* =========================================================== failures, retry, rate limit [mock] */
await check('[errors] 502: our wording only, message kept as "not answered", Retry re-sends the SAME text without a duplicate bubble', async () => {
  const { ctx, page } = await newPage('desktop')
  const log = await mockChat(page, { reply: (n) => (n === 1 ? { status: 502, body: errBody('assistant_upstream_error') } : undefined) , });
  // second call: succeed. (the handler above returns undefined for n>1, so install a fresh route for that case)
  await page.unroute('**/api/chat'); let n = 0
  await page.route('**/api/chat', (route) => { if (route.request().method() !== 'POST') return route.fallback(); n++; log.posts.push(JSON.parse(route.request().postData())); return n === 1 ? json(route, 502, errBody('assistant_upstream_error')) : json(route, 200, { reply: 'Back again.', mode: 'general', provider: 'gemini', model: 'test-model', truncated: false }) })
  await open(page); await send(page, 'will fail once'); const err = page.locator('[data-testid=chat-error]'); await err.waitFor()
  eq(await err.getAttribute('data-kind'), 'server', 'kind'); ok(/had a problem/.test(await err.innerText()), 'our wording'); ok(!/RAW PROVIDER|AIza/.test(await page.locator('body').innerText()), 'raw provider text must never be shown')
  ok(await page.locator('[data-testid=not-delivered]').isVisible(), 'message flagged'); eq(await page.locator('[data-testid=message][data-role=user]').count(), 1, 'one user bubble'); await shot(page, 'error-retry', false)
  await page.locator('[data-testid=retry]').click(); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  eq(log.posts.length, 2, 'two requests'); eq(log.posts[1], log.posts[0], 'same body'); eq(await page.locator('[data-testid=message][data-role=user]').count(), 1, 'still ONE user bubble'); eq(await page.locator('[data-testid=chat-error]').count(), 0, 'banner gone')
  ok(await page.evaluate(() => document.activeElement?.getAttribute('data-testid') === 'composer'), 'focus returns to the box'); await ctx.close()
})
for (const [status, code, kind, retry] of [[503, 'assistant_misconfigured', 'unavailable', true], [503, 'assistant_not_configured', 'not_configured', false], [504, 'assistant_timeout', 'timeout', true], [502, 'assistant_unreachable', 'server', true], [422, 'assistant_declined', 'declined', false]]) {
  await check(`[errors] HTTP ${status} ${code} shows a "${kind}" alert ${retry ? 'with' : 'without'} Retry`, async () => {
    const { ctx, page } = await newPage('desktop'); await mockChat(page, { reply: () => ({ status, body: errBody(code) }) }); await open(page); await send(page, 'hello')
    const err = page.locator('[data-testid=chat-error]'); await err.waitFor(); eq(await err.getAttribute('data-kind'), kind, 'kind'); eq(await err.getAttribute('role'), 'alert', 'role')
    eq((await page.locator('[data-testid=retry]').count()) === 1, retry, 'retry button'); ok(!/RAW PROVIDER|AIza/.test(await page.locator('body').innerText()), 'raw text shown'); await ctx.close()
  })
}
await check('[errors] status 404 keeps the composer editable and reports the failed send', async () => {
  const { ctx, page } = await newPage('desktop')
  const log = await mockChat(page, {
    status: (route) => json(route, 404, errBody('not_found')),
    reply: () => ({ status: 404, body: errBody('not_found') }),
  })
  await open(page)
  const composer = page.locator('[data-testid=composer]')
  await composer.fill('Can I still type?')
  ok(!(await composer.isDisabled()), 'the message box stays editable')
  ok(!(await isDisabled(page, '[data-testid=send]')), 'Send stays enabled for a non-empty draft')
  await page.keyboard.press('Enter')
  const error = page.locator('[data-testid=chat-error]')
  await error.waitFor()
  eq(await error.getAttribute('data-kind'), 'unavailable', 'error kind')
  ok(/temporarily unavailable/.test(await error.innerText()), 'clear unavailable message')
  eq(log.posts[0].message, 'Can I still type?', 'submitted text')
  await ctx.close()
})
await check('[errors] a lost connection shows an offline message and can be retried', async () => {
  const { ctx, page } = await newPage('desktop'); let n = 0
  await page.route('**/api/chat/status', (r) => json(r, 200, STATUS_OK))
  await page.route('**/api/chat', (route) => { if (route.request().method() !== 'POST') return route.fallback(); n++; return n === 1 ? route.abort('connectionreset') : json(route, 200, { reply: 'Reconnected.', mode: 'general', provider: 'gemini', model: 'm', truncated: false }) })
  await open(page); await send(page, 'offline?'); const err = page.locator('[data-testid=chat-error]'); await err.waitFor(); eq(await err.getAttribute('data-kind'), 'network', 'kind')
  await page.locator('[data-testid=retry]').click(); await page.waitForSelector('[data-testid=message][data-role=assistant]'); eq(n, 2, 'requests'); await ctx.close()
})
await check('[errors] server-side 422: the text goes back into the box to edit and out of the conversation', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page, { reply: () => ({ status: 422, body: { error: 'validation_error', message: 'Some inputs are invalid.', fields: [{ field: 'message', message: 'Keep the message under 2000 characters.' }] } }) }); await open(page)
  await send(page, 'rejected text'); const err = page.locator('[data-testid=chat-error]'); await err.waitFor()
  ok(/Keep the message under 2000 characters\./.test(await err.innerText()), 'server field message shown'); eq(await page.locator('[data-testid=composer]').inputValue(), 'rejected text', 'text restored'); eq(await page.locator('[data-testid=message]').count(), 0, 'not left in the thread'); eq(await page.locator('[data-testid=retry]').count(), 0, 'no retry for validation'); await ctx.close()
})
await check('[rate-limit] 429 + Retry-After: countdown shown, Send and Retry blocked until it ends, then Retry works', async () => {
  const { ctx, page } = await newPage('desktop'); let n = 0
  const log = { posts: [] }
  await page.route('**/api/chat/status', (r) => json(r, 200, STATUS_OK))
  await page.route('**/api/chat', (route) => { if (route.request().method() !== 'POST') return route.fallback(); n++; log.posts.push(1); return n === 1 ? json(route, 429, errBody('rate_limited', 'Too many messages. Try again in 3 seconds.'), { 'Retry-After': '3' }) : json(route, 200, { reply: 'Calmer now.', mode: 'general', provider: 'gemini', model: 'm', truncated: false }) })
  await open(page); await send(page, 'too fast'); const err = page.locator('[data-testid=chat-error]'); await err.waitFor(); eq(await err.getAttribute('data-kind'), 'rate_limited', 'kind')
  ok(/You can try again in [1-3]s/.test(await err.innerText()), `countdown text: ${await err.innerText()}`); ok(/Retry in [1-3]s/.test(await page.locator('[data-testid=retry]').innerText()), 'retry label'); ok(await isDisabled(page, '[data-testid=retry]'), 'Retry disabled')
  await page.fill('[data-testid=composer]', 'trying anyway'); ok(await isDisabled(page, '[data-testid=send]'), 'Send disabled'); await page.keyboard.press('Enter'); await sleep(200); eq(log.posts.length, 1, 'Enter during cooldown must not send')
  ok(/You can send again in/.test(await page.locator('[data-testid=chat-panel] form').innerText()), 'composer hint'); await shot(page, 'rate-limited', false)
  await page.waitForFunction(() => { const b = document.querySelector('[data-testid=retry]'); return b && !b.disabled }, null, { timeout: 8000 })
  await page.locator('[data-testid=retry]').click(); await page.waitForSelector('[data-testid=message][data-role=assistant]'); eq(log.posts.length, 2, 'requests'); await ctx.close()
})
await check('[rate-limit] without a readable header the countdown still comes from the server message', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page, { reply: () => ({ status: 429, body: errBody('rate_limited', 'Too many messages. Try again in 6 seconds.') }) }); await open(page); await send(page, 'x')
  const err = page.locator('[data-testid=chat-error]'); await err.waitFor(); ok(/try again in [4-6]s/i.test(await err.innerText()), await err.innerText()); await ctx.close()
})

/* =========================================================== conversations [mock] */
await check('[conversations] New starts empty; switching preserves each thread and draft; threads never mix', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page); await open(page)
  await send(page, 'question for A'); await page.waitForSelector('[data-testid=message][data-role=assistant]'); await page.fill('[data-testid=composer]', 'unsent draft in A')
  await page.locator('[data-testid=new-conversation]').click()
  eq(await page.locator('[data-testid=message]').count(), 0, 'new conversation is empty'); eq(await page.locator('[data-testid=composer]').inputValue(), '', 'new draft is empty'); eq(await page.locator('[data-testid=conversation]').count(), 2, 'two conversations')
  ok(await page.evaluate(() => document.activeElement?.getAttribute('data-testid') === 'composer'), 'composer focused'); await send(page, 'question for B'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  const items = page.locator('[data-testid=conversation]'); const titles = await items.allInnerTexts(); ok(titles[0].includes('question for B') && titles[1].includes('question for A'), `titles ${JSON.stringify(titles)}`)
  eq(await items.nth(0).getAttribute('aria-current'), 'true', 'active marked')
  await items.nth(1).click(); eq((await texts(page)).map((t) => t.replace(/\d\d:\d\d.*$/, '').trim()).filter(Boolean).length, 2, 'A has its two bubbles')
  ok((await page.locator('[data-testid=message]').first().innerText()).includes('question for A'), 'A shows A'); ok(!(await page.locator('[data-testid=message-log]').innerText()).includes('question for B'), 'B leaked into A')
  eq(await page.locator('[data-testid=composer]').inputValue(), 'unsent draft in A', 'draft preserved'); await items.nth(0).click(); ok((await page.locator('[data-testid=message-log]').innerText()).includes('question for B') && !(await page.locator('[data-testid=message-log]').innerText()).includes('question for A'), 'B intact')
  await shot(page, 'conversations-desktop', false); await ctx.close()
})
await check('[conversations] an answer that arrives after switching lands in the conversation that asked', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page, { delay: (n) => (n === 1 ? 1800 : 100) }); await open(page)
  await send(page, 'slow question A'); await page.waitForSelector('[data-testid=thinking]'); await page.locator('[data-testid=new-conversation]').click(); await send(page, 'quick question B'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  ok(!(await page.locator('[data-testid=message-log]').innerText()).includes('slow question A'), 'A leaked into B'); await sleep(1900)
  ok(!(await page.locator('[data-testid=message-log]').innerText()).includes('Answer 1'), "A's late answer must not appear in B"); await page.locator('[data-testid=conversation]').nth(1).click()
  const t = await page.locator('[data-testid=message-log]').innerText(); ok(t.includes('slow question A') && t.includes('Answer 1 to: slow question A'), 'A received its own answer'); await ctx.close()
})
await check('[conversations] on mobile the conversations are a scrollable row and switching works', async () => {
  const { ctx, page } = await newPage('mobile'); await mockChat(page); await open(page)
  await send(page, 'mobile question A'); await page.waitForSelector('[data-testid=message][data-role=assistant]'); await page.locator('[data-testid=new-conversation]').click(); await send(page, 'mobile question B'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  await page.locator('[data-testid=conversation]').nth(1).click(); ok((await page.locator('[data-testid=message-log]').innerText()).includes('mobile question A'), 'switched'); const o = await noOverflow(page); ok(o.sw <= o.iw + 1, 'no page overflow')
  await shot(page, 'conversation-mobile', false); await ctx.close()
})
await check('[clear] needs confirmation: Escape and Cancel keep everything, confirming empties only this conversation, and a late answer is dropped', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page, { delay: (n) => (n === 3 ? 1500 : 80) }); await open(page)
  await send(page, 'keep me in A'); await page.waitForSelector('[data-testid=message][data-role=assistant]'); await page.locator('[data-testid=new-conversation]').click(); await send(page, 'clear me in B'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  await page.locator('[data-testid=clear]').click(); const dlg = page.getByRole('alertdialog'); await dlg.waitFor(); ok(await page.evaluate(() => document.activeElement?.getAttribute('data-testid') === 'confirm-cancel'), 'focus starts on Cancel (the safe choice)')
  ok(/cannot be undone/.test(await dlg.innerText()), 'warns'); eq(await dlg.getAttribute('aria-modal'), 'true', 'modal')
  await page.keyboard.press('Tab'); eq(await page.evaluate(() => document.activeElement?.getAttribute('data-testid')), 'confirm-ok', 'Tab to Clear'); await page.keyboard.press('Tab'); eq(await page.evaluate(() => document.activeElement?.getAttribute('data-testid')), 'confirm-cancel', 'focus is trapped inside the dialog')
  await page.keyboard.press('Escape'); await dlg.waitFor({ state: 'detached' }); eq(await page.locator('[data-testid=message]').count(), 2, 'Escape keeps messages'); ok(await page.evaluate(() => document.activeElement?.getAttribute('data-testid') === 'clear'), 'focus returns to the Clear button')
  await page.locator('[data-testid=clear]').click(); await page.getByTestId('confirm-cancel').click(); eq(await page.locator('[data-testid=message]').count(), 2, 'Cancel keeps messages')
  await send(page, 'will be answered late'); await page.waitForSelector('[data-testid=thinking]'); await page.locator('[data-testid=clear]').click(); await page.getByTestId('confirm-ok').click()
  eq(await page.locator('[data-testid=message]').count(), 0, 'cleared'); await page.waitForSelector('[data-testid=empty-chat]'); await sleep(1700); eq(await page.locator('[data-testid=message]').count(), 0, 'the late answer must not reappear'); ok(await isDisabled(page, '[data-testid=clear]'), 'Clear disabled when empty')
  await page.locator('[data-testid=conversation]').nth(1).click(); ok((await page.locator('[data-testid=message-log]').innerText()).includes('keep me in A'), 'the other conversation is untouched'); await shot(page, 'after-clear', false); await ctx.close()
})
await check('[delete] removes only the chosen conversation after confirmation; the last one cannot be deleted', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page); await open(page); eq(await page.locator('[data-testid=delete]').count(), 0, 'no Delete with a single conversation')
  await send(page, 'first thread'); await page.waitForSelector('[data-testid=message][data-role=assistant]'); await page.locator('[data-testid=new-conversation]').click(); await send(page, 'second thread'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  await page.locator('[data-testid=delete]').click(); await page.getByTestId('confirm-ok').click(); eq(await page.locator('[data-testid=conversation]').count(), 1, 'one left'); ok((await page.locator('[data-testid=message-log]').innerText()).includes('first thread'), 'the other thread is shown'); eq(await page.locator('[data-testid=delete]').count(), 0, 'Delete hidden again'); await ctx.close()
})

/* =========================================================== limits, privacy, motion, a11y [mock] */
await check('[limits] an over-long message is flagged, cannot be sent, and the limit comes from the server', async () => {
  const { ctx, page } = await newPage('desktop'); const log = await mockChat(page, { status: (r) => json(r, 200, { ...STATUS_OK, limits: { ...STATUS_OK.limits, max_input_chars: 120 } }) }); await open(page)
  await page.fill('[data-testid=composer]', 'x'.repeat(121)); ok(/121 \/ 120 — too long/.test(await page.locator('[data-testid=counter]').innerText()), 'counter'); eq(await page.locator('[data-testid=composer]').getAttribute('aria-invalid'), 'true', 'aria-invalid')
  ok(await isDisabled(page, '[data-testid=send]'), 'Send disabled'); await page.keyboard.press('Enter'); await sleep(150); eq(log.posts.length, 0, 'nothing sent')
  await page.fill('[data-testid=composer]', 'x'.repeat(120)); ok(!(await isDisabled(page, '[data-testid=send]')), 'exactly the limit is allowed'); await page.fill('[data-testid=composer]', '   '); ok(await isDisabled(page, '[data-testid=send]'), 'whitespace only is not sendable'); await ctx.close()
})
await check('[privacy] nothing is stored in the browser, no foreign request is made, and conversations are in-memory only', async () => {
  const before = foreign.length
  const { ctx, page } = await newPage('desktop'); await mockChat(page); await open(page); await send(page, 'a private question about my script'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  const store = await page.evaluate(() => ({ ls: JSON.stringify({ ...localStorage }), ss: JSON.stringify({ ...sessionStorage }), cookie: document.cookie, idb: indexedDB.databases ? null : null }))
  ok(!/private question|Answer 1|AIza|AQ\./.test(store.ls + store.ss + store.cookie), `chat content or key in browser storage: ${JSON.stringify(store).slice(0, 200)}`)
  await page.locator('header nav[aria-label=Primary] a[title="Methodology"]').click(); await page.waitForURL(/\/about/); await page.locator('header nav[aria-label=Primary] a[title="AI Assistant"]').click(); await page.waitForSelector('[data-testid=composer]'); ok((await page.locator('[data-testid=message-log]').innerText()).includes('private question'), 'in-app navigation keeps the in-memory conversation')
  await page.reload({ waitUntil: 'networkidle' }); await page.waitForSelector('[data-testid=empty-chat]'); eq(await page.locator('[data-testid=message]').count(), 0, 'a refresh clears the conversation (nothing persisted)')
  eq(foreign.slice(before), [], 'requests to other origins'); ok(!/generativelanguage|googleapis/.test((await page.content())), 'no provider address in the page'); await ctx.close()
})
await check('[reduced-motion] the loading dots stop animating when the user prefers reduced motion', async () => {
  const probe = async (reduce) => { const { ctx, page } = await newPage('desktop', { reducedMotion: reduce }); await mockChat(page, { delay: 1500 }); await open(page); await send(page, 'hi'); await page.waitForSelector('[data-testid=thinking]'); await sleep(300)
    const r = await page.evaluate(() => { const d = document.querySelector('[data-testid=thinking] [role=status] span[aria-hidden]'); return { anim: getComputedStyle(d).animationName, running: document.getAnimations().filter((a) => a.effect?.getComputedTiming().iterations === Infinity && document.querySelector('[data-testid=thinking]').contains(a.effect.target)).length } }); await ctx.close(); return r }
  const normal = await probe(false), reduced = await probe(true)
  ok(normal.anim !== 'none' && normal.running >= 1, `baseline should animate ${JSON.stringify(normal)}`); ok(reduced.anim === 'none' && reduced.running === 0, `still animating under reduce ${JSON.stringify(reduced)}`); return JSON.stringify({ normal, reduced })
})
await check('[a11y] landmarks, labels and live regions are present', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page); await open(page); await send(page, 'hello'); await page.waitForSelector('[data-testid=message][data-role=assistant]')
  const a = await page.evaluate(() => ({ log: document.querySelector('[data-testid=message-log]')?.getAttribute('role'), live: document.querySelector('[data-testid=message-log]')?.getAttribute('aria-live'), logLabel: document.querySelector('[data-testid=message-log]')?.getAttribute('aria-label'),
    label: document.querySelector('label[for="' + document.querySelector('[data-testid=composer]').id + '"]')?.textContent, rail: document.querySelector('nav[aria-label=Conversations]') !== null, h1: document.querySelector('h1')?.getAttribute('aria-label'), described: !!document.querySelector('#' + document.querySelector('[data-testid=composer]').getAttribute('aria-describedby')),
    sendName: document.querySelector('[data-testid=send]').getAttribute('aria-label'), times: document.querySelectorAll('[data-testid=message] time[datetime]').length, srLabels: [...document.querySelectorAll('[data-testid=message] .sr-only')].map((e) => e.textContent) }))
  eq(a.log, 'log', 'role'); eq(a.live, 'polite', 'aria-live'); ok(a.logLabel, 'log label'); eq(a.label, 'Message to the assistant', 'textarea label'); ok(a.rail && a.described, 'rail + describedby'); eq(a.h1, 'AI Assistant', 'h1'); eq(a.sendName, 'Send message', 'send name'); eq(a.times, 2, 'timestamps'); eq(a.srLabels, ['You said: ', 'Assistant said: '], 'speaker labels')
  await ctx.close()
})
await check('[contrast] key chat text meets WCAG AA against its effective background (approximate: gradients and blur are not modelled)', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page, { reply: () => ({ status: 429, body: errBody('rate_limited', 'Try again in 30 seconds.'), headers: { 'Retry-After': '30' } }) }); await open(page)
  await page.evaluate(() => 0); const empty = await page.evaluate(measure); await send(page, 'hello'); await page.waitForSelector('[data-testid=chat-error]'); const err = await page.evaluate(measure)
  const rows = [...empty, ...err].filter((r) => !r.missing); const fails = rows.filter((r) => r.ratio < r.need); ok(fails.length === 0, `below AA: ${JSON.stringify(fails)}`); await ctx.close(); return `min ratio ${Math.min(...rows.map((r) => r.ratio))} over ${rows.length} samples`
  function measure() {
    const cv = document.createElement('canvas'); cv.width = cv.height = 1; const cx = cv.getContext('2d', { willReadFrequently: true })
    const rgba = (css) => { cx.clearRect(0, 0, 1, 1); cx.fillStyle = '#000'; cx.fillStyle = css; cx.fillRect(0, 0, 1, 1); const d = cx.getImageData(0, 0, 1, 1).data; return [d[0], d[1], d[2], d[3] / 255] }
    const over = (f, b) => [0, 1, 2].map((i) => f[i] * f[3] + b[i] * (1 - f[3]))
    const lum = (c) => { const [r, g, b] = c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4 }); return 0.2126 * r + 0.7152 * g + 0.0722 * b }
    const bgOf = (el) => { const st = []; for (let e = el; e; e = e.parentElement) { const c = rgba(getComputedStyle(e).backgroundColor); if (c[3] > 0) st.push(c); if (c[3] === 1) break } let base = [6, 6, 26]; for (const c of st.reverse()) base = over(c, base).concat(1); return base.slice(0, 3) }
    const targets = [['[data-testid=empty-chat] h2', 'empty title'], ['[data-testid=empty-chat] p', 'empty text'], ['[data-testid=starter]', 'starter'], ['[data-testid=conversation-title]', 'title'], ['[data-testid=message][data-role=assistant] p, [data-testid=message][data-role=assistant] .glass', 'assistant text'], ['[data-testid=message] time', 'timestamp'],
      ['[data-testid=chat-error] p', 'error text'], ['[data-testid=chat-panel] form p', 'composer hint'], ['[data-testid=conversation]:not([aria-current])', 'inactive conversation'], ['label.sr-only', 'x']].filter(([s]) => s !== 'label.sr-only')
    return targets.map(([sel, label]) => { const el = document.querySelector(sel); if (!el) return { label, missing: true }; const st = getComputedStyle(el), bg = bgOf(el), fg = over(rgba(st.color), bg), l1 = lum(fg), l2 = lum(bg), ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05), size = parseFloat(st.fontSize), large = size >= 24 || (size >= 18.66 && parseInt(st.fontWeight) >= 700); return { label, ratio: +ratio.toFixed(2), need: large ? 3 : 4.5 } })
  }
})
await check('[routes] the other eight routes still load without console errors', async () => {
  const { ctx, page } = await newPage('desktop'); await mockChat(page); const before = problems.length
  for (const p of ['/', '/predict', '/performance', '/explain', '/explore', '/researcher', '/about']) { await page.goto(BASE + p, { waitUntil: 'networkidle' }); ok((await page.locator('h1').first().innerText()).length > 2, p) }
  ok(problems.length === before, `console problems: ${JSON.stringify(problems.slice(before))}`); await ctx.close()
})

/* =========================================================== [real] one opt-in live message through the real backend */
if (process.env.REAL_GEMINI === '1') {
  await check('[real] one live message through the UI, the real backend and Gemini', async () => {
    const { ctx, page } = await newPage('desktop', { timeout: 40000 }); await open(page)
    const st = await page.locator('[data-testid=chat-panel] header').innerText(); ok(!/not set up/.test(await page.locator('main').innerText()), 'assistant configured')
    const t = Date.now(); await send(page, 'In one short sentence, what is a Random Forest?'); await page.waitForSelector('[data-testid=message][data-role=assistant]', { timeout: 40000 })
    const reply = (await page.locator('[data-testid=message][data-role=assistant]').innerText()).replace(/\s+/g, ' '); ok(reply.length > 20, 'a real reply'); await shot(page, 'real-reply', false); await ctx.close()
    return `${((Date.now() - t) / 1000).toFixed(1)}s, header "${st.split('\n').pop()}", reply ${reply.length} chars`
  })
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} checks passed`)
const distinct = [...new Set(problems)]
console.log(`console errors / page errors: ${distinct.length}`, distinct.slice(0, 6)); console.log(`requests to other origins: ${foreign.length}`, foreign.slice(0, 4))
process.exit(failed.length || distinct.length || foreign.length ? 1 : 0)
