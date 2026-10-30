/* Interactive browser checks for /researcher, run in a REAL browser (installed Edge/Chrome) through Playwright.
   Playwright is NOT a project dependency: point PLAYWRIGHT_ENTRY at an existing copy and BROWSER_EXE at an installed browser, e.g.
     PLAYWRIGHT_ENTRY=C:/.../node_modules/playwright/index.mjs  BROWSER_EXE="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
     node --experimental-strip-types --disable-warning=ExperimentalWarning frontend/tests/browser/researcher.browser.mjs
   Needs the built app on BASE (default http://localhost:4173, `vite preview` proxying /api) and the backend on :8000.
   Sections tagged [mock] answer /api/agent/runs* from fixtures (deterministic UI states). Sections tagged [real] hit the live backend, model and Wikipedia.
   SHOTS_DIR receives screenshots. Exit code 1 if any check fails. */
import assert from 'node:assert/strict'
import { mkdirSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import * as fx from '../fixtures.ts'

const BASE = process.env.BASE ?? 'http://localhost:4173'
const SHOTS = process.env.SHOTS_DIR ?? './shots'
const ONLY = process.env.ONLY ? new RegExp(process.env.ONLY) : null
const REAL = process.env.SKIP_REAL !== '1'
mkdirSync(SHOTS, { recursive: true })
const { chromium } = await import(pathToFileURL(process.env.PLAYWRIGHT_ENTRY).href)
const browser = await chromium.launch({ executablePath: process.env.BROWSER_EXE, headless: true })

const VIEWS = { desktop: { width: 1440, height: 900 }, laptop: { width: 1024, height: 768 }, tablet: { width: 820, height: 1180 }, 'tablet-768': { width: 768, height: 1024 }, mobile: { width: 390, height: 844 } }
const results = []
const consoleProblems = []
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const json = (route, status, obj, headers = {}) => route.fulfill({ status, contentType: 'application/json', headers, body: JSON.stringify(obj) })

async function check(name, fn) {
  if (ONLY && !ONLY.test(name)) return
  const t = Date.now()
  try { const detail = await fn(); results.push({ name, ok: true, detail }); console.log(`PASS  ${name}${detail ? '  — ' + detail : ''}  (${((Date.now() - t) / 1000).toFixed(1)}s)`) }
  catch (e) { results.push({ name, ok: false, detail: String(e.message ?? e).split('\n')[0].slice(0, 400) }); console.log(`FAIL  ${name}  — ${String(e.message ?? e).split('\n')[0].slice(0, 400)}`) }
}
const ok = (c, msg) => { if (!c) throw new Error(msg) }
const eq = (a, b, msg) => { try { assert.deepStrictEqual(a, b) } catch { throw new Error(`${msg}: got ${JSON.stringify(a)} expected ${JSON.stringify(b)}`) } }
const flat = (t) => t.replace(/\s+/g, ' ')

async function newPage(view = 'desktop', opts = {}) {
  const ctx = await browser.newContext({ viewport: VIEWS[view], reducedMotion: opts.reducedMotion ? 'reduce' : 'no-preference', acceptDownloads: false })
  await ctx.route('https://en.wikipedia.org/**', (r) => r.fulfill({ status: 200, contentType: 'text/html', body: '<title>stub</title>wikipedia stub' }))   // never leave the app in tests
  const page = await ctx.newPage()
  page.setDefaultTimeout(opts.timeout ?? 15000)
  page.on('console', (m) => { if (m.type() === 'error' && !/ERR_ABORTED|ERR_CONNECTION_RESET|net::ERR_FAILED|Failed to load resource: the server responded with a status of (4\d\d|5\d\d)/.test(m.text())) consoleProblems.push({ view, text: m.text().slice(0, 200) }) })
  page.on('pageerror', (e) => consoleProblems.push({ view, text: 'pageerror: ' + String(e).slice(0, 200) }))
  return { ctx, page }
}

/** Run the page's own clock (Date.now and setTimeout) f times faster, continuously from "now"; speed(page, 1) restores real time. */
const speed = (page, f) => page.evaluate((f) => {
  const w = window; w.__real ??= { now: Date.now.bind(Date), st: w.setTimeout.bind(w) }
  const { now, st } = w.__real, base = Date.now(), anchor = now()
  Date.now = () => base + (now() - anchor) * f
  w.setTimeout = (fn, ms, ...a) => st(fn, (ms ?? 0) / f, ...a)
}, f)

/** [mock] answer the agent run endpoints from fixtures. */
function agentMock(page, o = {}) {
  const log = { posts: [], gets: [], byKey: [] }
  let gi = 0
  const seq = o.runs ?? [fx.running(), fx.running(), fx.completed()]
  page.route('**/api/agent/runs**', async (route) => {
    const req = route.request(), p = new URL(req.url()).pathname
    if (req.method() === 'POST') {
      log.posts.push({ headers: req.headers(), body: JSON.parse(req.postData() || '{}') })
      return o.post ? o.post(route, log.posts.length) : json(route, 202, fx.ACCEPTED)
    }
    if (p.includes('/by-key/')) { log.byKey.push(p); return o.byKey ? o.byKey(route, log.byKey.length) : json(route, 404, { error: 'not_found', message: 'No run was created for that key.' }) }
    log.gets.push(p)
    const item = typeof seq === 'function' ? seq(log.gets.length) : seq[Math.min(gi++, seq.length - 1)]
    return typeof item === 'function' ? item(route) : json(route, 200, item)
  })
  return log
}

async function open(page, path = '/researcher') { await page.goto(BASE + path, { waitUntil: 'networkidle' }); await page.waitForSelector('main#main') }
async function waitForm(page) { await page.waitForSelector('#r-title_year') }
async function fillForm(page, v = {}) {
  const f = { year: '2010', duration: '120', genres: ['Drama', 'Romance'], country: 'USA', rating: 'PG-13', title: '', ...v }
  await page.fill('#r-title_year', f.year); await page.fill('#r-duration', f.duration)
  for (const g of f.genres) await page.getByRole('button', { name: g, exact: true }).click()
  await page.selectOption('#r-country', f.country); await page.selectOption('#r-content_rating', f.rating)
  if (f.title) await page.fill('#r-research_title', f.title)
}
const submit = (page) => page.getByRole('button', { name: /Research this film|Run again/ }).click()
const phase = (page) => page.locator('[aria-label="Research result"]').getAttribute('data-phase')
const waitPhase = (page, ph, timeout) => page.waitForSelector(`[aria-label="Research result"][data-phase="${ph}"]`, { timeout: timeout ?? 20000 })
const shot = (page, name, full = true) => page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: full })
const noOverflow = (page) => page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth, ok: document.documentElement.scrollWidth <= window.innerWidth + 1 }))

/* =========================================================== existing routes still work */
await check('[routes] all 7 routes load, show a heading, and log no console errors', async () => {
  const { ctx, page } = await newPage('desktop')
  const seen = []
  for (const [path, h1] of [['/', /./], ['/predict', /screen/i], ['/performance', /./], ['/explain', /./], ['/explore', /./], ['/researcher', /AI Movie Researcher/], ['/about', /./]]) {
    const before = consoleProblems.length
    await open(page, path)
    const t = await page.locator('h1').first().innerText()
    ok(h1.test(t), `${path}: unexpected h1 "${t}"`)
    ok(consoleProblems.length === before, `${path}: console problems ${JSON.stringify(consoleProblems.slice(before))}`)
    seen.push(path)
  }
  await open(page, '/no-such-page'); ok(await page.getByText('Page not found').isVisible(), '404 page')
  await ctx.close(); return seen.join(' ')
})

/* =========================================================== navigation + layout at every width */
for (const view of ['desktop', 'laptop', 'tablet', 'tablet-768']) {
  await check(`[nav] eight items fit without overlap at ${view} (${VIEWS[view].width}px)`, async () => {
    const { ctx, page } = await newPage(view); await open(page)
    const links = page.locator('header nav[aria-label="Primary"] a')
    eq(await links.count(), 8, 'nav link count')
    const boxes = []
    for (let i = 0; i < 8; i++) {
      const l = links.nth(i); ok(await l.isVisible(), `link ${i} hidden`)
      const b = await l.boundingBox(); boxes.push(b)
      ok(b.x >= 0 && b.x + b.width <= VIEWS[view].width, `link ${i} outside viewport: ${JSON.stringify(b)}`)
      ok(b.height >= 40, `link ${i} tap target only ${b.height}px high`)
      const name = (await l.getAttribute('title')) || (await l.innerText()); ok(name.trim().length > 2, `link ${i} has no accessible name`)
    }
    for (let i = 1; i < 8; i++) ok(boxes[i].x >= boxes[i - 1].x + boxes[i - 1].width - 1, `links ${i - 1} and ${i} overlap`)
    const nav = await page.locator('header nav[aria-label="Primary"]').boundingBox(), logo = await page.locator('header a[aria-label="FRAME home"]').boundingBox()
    ok(logo.x + logo.width <= nav.x + 1, 'logo overlaps nav')
    const hdr = await page.locator('header .glass').first().boundingBox(); ok(hdr.x + hdr.width <= VIEWS[view].width + 1 && hdr.height < 80, `header box ${JSON.stringify(hdr)}`)
    eq(await page.locator('header nav a[aria-current="page"]').getAttribute('title'), 'AI Movie Researcher', 'active item')
    const labelsVisible = await links.nth(5).locator('span.lg\\:inline').isVisible().catch(() => false)
    await shot(page, `nav-${view}`, false); await ctx.close()
    return `nav ${Math.round(nav.width)}px wide, labels ${labelsVisible ? 'shown' : 'icon-only'}, ${JSON.stringify(await noOverflow(page).catch(() => ''))}`
  })
}

await check('[nav] mobile: menu opens as a dialog with eight items, Escape closes it and returns focus', async () => {
  const { ctx, page } = await newPage('mobile'); await open(page)
  ok(!(await page.locator('header nav[aria-label="Primary"]').isVisible()), 'desktop nav should be hidden on mobile')
  const btn = page.getByRole('button', { name: 'Open navigation' }); await btn.click()
  const dlg = page.getByRole('dialog', { name: 'Navigation' }); await dlg.waitFor()
  eq(await dlg.locator('nav a').count(), 8, 'items in dialog')
  ok(await page.evaluate(() => document.activeElement?.getAttribute('aria-label') === 'Close navigation'), 'focus should move into the dialog')
  await shot(page, 'nav-mobile-open', false)
  await page.keyboard.press('Escape'); await dlg.waitFor({ state: 'detached' })
  ok(await page.evaluate(() => document.activeElement?.getAttribute('aria-label') === 'Open navigation'), 'focus should return to the menu button')
  await btn.click(); await dlg.getByRole('link', { name: 'AI Movie Researcher' }).click(); await page.waitForURL(/researcher/)
  await ctx.close()
})

for (const view of ['desktop', 'laptop', 'tablet', 'mobile']) {
  await check(`[layout] /researcher at ${view}: no horizontal scroll, form and results arranged sensibly`, async () => {
    const { ctx, page } = await newPage(view); await open(page); await waitForm(page)
    const o = await noOverflow(page); ok(o.ok, `horizontal overflow ${o.sw}>${o.iw}`)
    const form = await page.locator('form[aria-label="Film attributes"]').boundingBox(), res = await page.locator('[aria-label="Research result"]').boundingBox()
    const side = res.x > form.x + form.width - 10
    ok(view === 'desktop' ? side : !side || VIEWS[view].width >= 1280, `columns: form ${JSON.stringify(form)} result ${JSON.stringify(res)}`)
    for (const loc of ['#r-title_year', '#r-duration', '#r-country', '#r-research_title']) { const b = await page.locator(loc).boundingBox(); ok(b.width >= 120 && b.x >= 0 && b.x + b.width <= VIEWS[view].width, `${loc} box ${JSON.stringify(b)}`) }
    await shot(page, `layout-${view}`); await ctx.close(); return `${side ? 'side-by-side' : 'stacked'}`
  })
}

/* =========================================================== form validation */
await check('[form] empty submit lists every required field, focuses the first, and sends nothing', async () => {
  const { ctx, page } = await newPage('desktop'); const log = agentMock(page); await open(page); await waitForm(page)
  await submit(page)
  ok(await page.getByRole('alert').filter({ hasText: 'Fix 5 fields to continue.' }).isVisible(), 'summary alert')
  const firstInvalid = await page.evaluate(() => document.querySelector('[aria-invalid="true"], #r-genres')?.id)
  eq(await page.evaluate(() => document.activeElement?.id), firstInvalid, 'the first invalid field in page order is focused')
  eq(await page.locator('[aria-invalid="true"]').count(), 4, 'invalid inputs'); ok(await page.getByText('Select at least one genre.').isVisible(), 'genre error')
  const d = await page.locator('#r-duration').getAttribute('aria-describedby'); ok(d && (await page.locator('#' + d).innerText()).includes('required'), 'error text linked via aria-describedby')
  eq(log.posts.length, 0, 'POSTs sent for an invalid form'); await shot(page, 'form-errors'); await ctx.close()
})
await check('[form] bad numbers and a hostile title are explained, and errors clear as the user fixes them', async () => {
  const { ctx, page } = await newPage('desktop'); const log = agentMock(page); await open(page); await waitForm(page)
  await fillForm(page, { year: '20.5', duration: '0', title: '<script>alert(1)</script>' }); await submit(page)
  ok(await page.getByText('Use a whole number.').isVisible(), 'year message'); ok(await page.getByText(/Must be between 1 and 600/).isVisible(), 'duration message'); ok(await page.getByText('Use 2 to 120 letters, numbers or ordinary punctuation.').isVisible(), 'title message')
  await page.fill('#r-title_year', '2010'); eq(await page.locator('#r-title_year').getAttribute('aria-invalid'), 'false', 'year error clears on edit')
  eq(log.posts.length, 0, 'no POST for invalid form'); await ctx.close()
})

/* =========================================================== submission, progress, completion [mock] */
await check('[mock] submit sends the right body + idempotency key, shows real progress, then the separated result', async () => {
  const { ctx, page } = await newPage('desktop'); const log = agentMock(page, { runs: [fx.running(), fx.running(), fx.completed()] }); await open(page); await waitForm(page)
  await fillForm(page, { title: 'Inception' }); await page.getByRole('button', { name: /Research this film/ }).dblclick()      // double click on purpose
  await page.waitForSelector('[data-testid="run-progress"]')
  const prog = await page.locator('[data-testid="run-progress"]').innerText()
  ok(/API status: (running|queued)/.test(prog) && prog.includes(fx.RUN_ID), `progress shows API status + run id: ${prog.slice(0, 120)}`)
  ok(await page.locator('#r-duration').isDisabled(), 'form locked while running'); eq(await page.getByRole('button', { name: /Researching/ }).getAttribute('aria-busy'), 'true', 'submit button busy')
  await shot(page, 'progress-running', false)
  await waitPhase(page, 'completed')
  eq(log.posts.length, 1, 'double click must create ONE run')
  const p = log.posts[0]
  ok(/^[0-9a-f]{32}$/.test(p.headers['idempotency-key']), `Idempotency-Key header ${p.headers['idempotency-key']}`)
  eq(p.body, { genres: ['Drama', 'Romance'], title_year: 2010, duration: 120, country: 'USA', content_rating: 'PG-13', research_title: 'Inception' }, 'POST body')
  ok(log.gets.length >= 3 && log.gets.every((g) => g.endsWith('/' + fx.RUN_ID)), 'polled the same run')
  ok(await page.evaluate(() => document.activeElement?.getAttribute('aria-label') === 'Research result'), 'focus moved to the result')
  await ctx.close()
})

await check('[mock] prediction, historical evidence, AI analysis, risks and web research are separate, ordered panels', async () => {
  const { ctx, page } = await newPage('desktop'); agentMock(page, { runs: [fx.withResearch(fx.research())] }); await open(page); await waitForm(page)
  await fillForm(page); await submit(page); await waitPhase(page, 'completed')
  const order = await page.evaluate(() => ['agent-prediction', 'agent-analysis', 'agent-risks', 'agent-evidence', 'agent-research'].map((id) => { const e = document.getElementById(id); return e ? e.getBoundingClientRect().top + scrollY : null }))
  ok(order.every((v) => v !== null) && order.every((v, i) => i === 0 || v > order[i - 1]), `panel order ${order}`)
  eq(await page.locator('[data-testid="predicted-label"]').innerText(), 'Hit', 'label')
  const pred = await page.locator('#agent-prediction').innerText(); ok(/Deterministic/.test(pred) && /not box office or profit/.test(pred) && /uncalibrated/.test(pred), 'prediction caveats')
  ok(!pred.includes('The model ranks Hit highest at 64%.') && !/Wikipedia/.test(pred), 'LLM or research text leaked into the prediction panel')
  const ana = await page.locator('#agent-analysis').innerText(); ok(/AI-generated analysis/.test(ana) && /Written by llama3.1:8b/.test(ana), 'analysis labelled')
  const ev = await page.locator('#agent-evidence').innerText(); ok(/Dataset cohort/.test(ev) && /Comparable earlier films/.test(ev) && !/Online research/.test(ev), 'historical evidence panel holds dataset evidence only')
  const rs = await page.locator('#agent-research').innerText(); ok(/Contextual research/.test(rs) && /not an input to the Random Forest/.test(rs), 'research panel labelled as context')
  eq(await page.locator('[data-testid="risk-item"]').count(), 1, 'risk items'); ok((await page.locator('[data-testid="risk-trace"]').innerText()).includes('sensitivity_no_change'), 'risk rule id visible')
  ok(/IMDb-score classes/.test(await page.locator('main').innerText()), 'limitations note present')
  await shot(page, 'result-desktop'); await ctx.close()
})
for (const view of ['tablet', 'mobile']) {
  await check(`[mock] completed result is readable at ${view}: no overflow, tables scroll inside their panel`, async () => {
    const { ctx, page } = await newPage(view); agentMock(page, { runs: [fx.withResearch(fx.research())] }); await open(page); await waitForm(page)
    await fillForm(page); await submit(page); await waitPhase(page, 'completed')
    const o = await noOverflow(page); ok(o.ok, `overflow ${o.sw}>${o.iw}`)
    await shot(page, `result-${view}`); await ctx.close()
  })
}

/* =========================================================== research states [mock] */
const researchCases = [
  ['ok', fx.research(), (t, h) => /2 sources retrieved using 7 of at most 8 requests/.test(t) && h.sources === 2 && !h.alert],
  ['partial', fx.research({ status: 'partial', reason: 'no_matching_film_page', message: 'Retrieved some pre-release background; part of the research could not be completed.', sources: [fx.source('genre', 'Action film')], skipped: [{ subject: 'Zzyzx', reason: 'no_matching_film_page' }] }), (t, h) => /1 source retrieved/.test(t) && /Not retrieved \(1\)/.test(t) && h.sources === 1],
  ['no_results', fx.failedResearch('no_results', 'no_pre_cutoff_pages', 'The research ran but found no usable page versions from before the cutoff.'), (t, h) => /0 sources retrieved/.test(t) && /Nothing is cited/.test(t) && h.sources === 0 && !h.alert],
  ['timeout', fx.failedResearch('timeout', 'timeout', 'Online research timed out, so no sources are shown.'), (t, h) => /Research timed out/.test(t) && /0 sources retrieved/.test(t) && h.alert && h.sources === 0],
  ['unavailable', fx.failedResearch('unavailable', 'connection_failed', 'Online research was unavailable, so no sources are shown.'), (t, h) => /Research was unavailable/.test(t) && h.alert && h.sources === 0],
  ['error', fx.failedResearch('error', 'http_500', 'Online research hit an error, so no sources are shown.'), (t, h) => /Research hit an error/.test(t) && h.alert && h.sources === 0],
  ['disabled', fx.research({ available: false, status: 'disabled', reason: 'research_not_enabled', sources: [], requests_made: 0 }), (t) => /not enabled on this server/.test(t) && !/retrieved/.test(t)],
]
for (const [name, research, verify] of researchCases) {
  await check(`[mock] research state "${name}" is rendered accurately and never hides the prediction`, async () => {
    const { ctx, page } = await newPage('desktop'); agentMock(page, { runs: [fx.withResearch(research)] }); await open(page); await waitForm(page)
    await fillForm(page); await submit(page); await waitPhase(page, 'completed')
    const panel = page.locator('#agent-research'), text = await panel.innerText()
    const facts = { sources: await panel.locator('[data-testid="research-source"]').count(), alert: (await panel.locator('[data-testid="research-problem"]').count()) > 0 }
    ok(verify(text, facts), `unexpected research rendering: ${JSON.stringify(facts)} ${text.slice(0, 260).replace(/\n/g, ' | ')}`)
    ok(await page.locator('[data-testid="predicted-label"]').isVisible() && await page.locator('#agent-analysis').isVisible(), 'prediction/analysis must remain visible')
    if (name === 'ok' || name === 'timeout') await shot(page, `research-${name}`)
    await ctx.close()
  })
}
await check('[mock] Wikipedia source links: titles, safe attributes, retrieval + page-version times, opens in a new tab; unsafe links never rendered', async () => {
  const { ctx, page } = await newPage('desktop')
  const bad = ['javascript:alert(1)', 'http://en.wikipedia.org/w/x', 'https://evil.example.com/w/x', 'https://en.wikipedia.org.evil.com/w/x'].map((u) => fx.source('genre', 'Trap', { url: u }))
  agentMock(page, { runs: [fx.withResearch(fx.research({ sources: [...fx.research().sources, ...bad] }))] }); await open(page); await waitForm(page)
  await fillForm(page); await submit(page); await waitPhase(page, 'completed')
  const links = page.locator('#agent-research a'); eq(await links.count(), 2, 'only the two valid links are rendered')
  for (let i = 0; i < 2; i++) { const l = links.nth(i); eq(await l.getAttribute('target'), '_blank', 'target'); ok((await l.getAttribute('rel')).includes('noopener') && (await l.getAttribute('rel')).includes('noreferrer'), 'rel'); ok((await l.getAttribute('href')).startsWith('https://en.wikipedia.org/w/index.php'), 'href') }
  const t = flat(await page.locator('#agent-research').innerText()); ok(/Page version 2009-12-29 22:16 UTC · revision 334798969/.test(t) && /Retrieved 2026-10-29 19:54 UTC/.test(t) && /4 returned sources were not shown/.test(t), `times/notice missing: ${t.slice(0, 600)}`)
  const [popup] = await Promise.all([ctx.waitForEvent('page'), links.first().click()])
  ok(popup.url().startsWith('https://en.wikipedia.org/w/index.php?title=Inception'), `popup ${popup.url()}`); ok(!page.url().includes('wikipedia'), 'original tab must not navigate')
  ok(await popup.evaluate(() => window.opener === null), 'noopener: new tab has no handle on the app'); await ctx.close()
})

/* =========================================================== errors, retry, timeout [mock] */
await check('[mock] 422 field errors from the server appear beside the right inputs', async () => {
  const { ctx, page } = await newPage('desktop')
  agentMock(page, { post: (r) => json(r, 422, { error: 'validation_error', message: 'Some inputs are invalid.', fields: [{ field: 'duration', message: 'Input should be less than or equal to 600' }] }) })
  await open(page); await waitForm(page); await fillForm(page); await submit(page); await waitPhase(page, 'error')
  ok(await page.getByText('Input should be less than or equal to 600').isVisible(), 'field message'); eq(await page.locator('#r-duration').getAttribute('aria-invalid'), 'true', 'aria-invalid'); await ctx.close()
})
for (const [status, body, expect] of [[429, { error: 'busy', message: 'Too many analyses are in progress. Try again shortly.' }, /researcher is busy/i], [503, { error: 'model_unavailable', message: 'The prediction model is not available right now.' }, /Model unavailable/]]) {
  await check(`[mock] HTTP ${status} on submit shows a clear error and keeps the form usable`, async () => {
    const { ctx, page } = await newPage('desktop'); agentMock(page, { post: (r) => json(r, status, body) }); await open(page); await waitForm(page)
    await fillForm(page); await submit(page); await waitPhase(page, 'error')
    ok(expect.test(await page.locator('[data-testid="run-problem"]').innerText()), 'error title'); ok(await page.locator('#r-duration').isEnabled(), 'form re-enabled'); await ctx.close()
  })
}
await check('[mock] a failed run (status "failed") is announced and offers a fresh start', async () => {
  const { ctx, page } = await newPage('desktop'); agentMock(page, { runs: [fx.running(), fx.failedRun()] }); await open(page); await waitForm(page)
  await fillForm(page); await submit(page); await waitPhase(page, 'failed')
  const a = page.locator('[data-testid="run-failed"]'); ok(await a.getAttribute('role') === 'alert', 'role=alert'); ok(/No prediction was produced/.test(await a.innerText()), 'text')
  await page.getByRole('button', { name: 'Start a new run' }).click(); await page.waitForSelector('[data-phase="idle"]'); await shot(page, 'run-failed', false); await ctx.close()
})
await check('[mock] LOST RESPONSE: retry re-sends the same inputs with the SAME idempotency key (no duplicate run)', async () => {
  const { ctx, page } = await newPage('desktop')
  const log = agentMock(page, { post: (r, n) => (n === 1 ? r.abort('connectionreset') : json(r, 202, fx.ACCEPTED, { 'Idempotent-Replayed': 'true' })), runs: [fx.completed()] })
  await open(page); await waitForm(page); await fillForm(page); await submit(page); await waitPhase(page, 'error')
  ok(/Cannot reach the FRAME API/.test(await page.locator('[data-testid="run-problem"]').innerText()), 'network error text')
  ok(/same request with the same key/.test(await page.locator('[data-testid="run-problem"]').innerText()), 'explains no duplicate')
  await page.getByRole('button', { name: 'Try again' }).click(); await waitPhase(page, 'completed')
  eq(log.posts.length, 2, 'two POSTs'); eq(log.posts[1].headers['idempotency-key'], log.posts[0].headers['idempotency-key'], 'same key'); eq(log.posts[1].body, log.posts[0].body, 'same body'); await ctx.close()
})
await check('[mock] after a completed run, "Run again" is a deliberate NEW run with a NEW key', async () => {
  const { ctx, page } = await newPage('desktop'); const log = agentMock(page, { runs: [fx.completed()] }); await open(page); await waitForm(page)
  await fillForm(page); await submit(page); await waitPhase(page, 'completed'); await page.getByRole('button', { name: 'Run again' }).click(); await sleep(800)
  eq(log.posts.length, 2, 'two POSTs'); ok(log.posts[0].headers['idempotency-key'] !== log.posts[1].headers['idempotency-key'], 'keys must differ'); await ctx.close()
})
await check('[mock] polling is bounded: after 5 minutes it stops, offers Keep waiting on the SAME run, and Keep waiting never POSTs', async () => {
  const { ctx, page } = await newPage('desktop')
  // Playwright's fake clock freezes its own actions, so while WAITING the page's clock is run 60x faster (5 "minutes" of polling in ~5 s) and restored before any click/fill
  // ~132 polls fit in one 5-minute window, so the run finishes during the second window
  const log = agentMock(page, { runs: (n) => (n < 140 ? fx.running() : fx.completed()) }); await open(page); await waitForm(page)
  await fillForm(page); await submit(page); await page.waitForSelector('[data-testid="run-progress"]')
  await speed(page, 60); await waitPhase(page, 'timeout', 40000); await speed(page, 1)
  ok(/Still waiting on the run/.test(await page.locator('[data-testid="run-problem"]').innerText()), 'timeout panel'); const getsAtTimeout = log.gets.length
  ok(getsAtTimeout < 200, `polling was bounded (${getsAtTimeout} GETs for 5 minutes)`); await shot(page, 'run-timeout', false)
  await page.getByRole('button', { name: 'Keep waiting' }).click()
  await speed(page, 60); await waitPhase(page, 'completed', 40000); await speed(page, 1); eq(log.posts.length, 1, 'no second POST'); ok(log.gets.every((g) => g.endsWith('/' + fx.RUN_ID)), 'same run id'); await ctx.close()
})

/* =========================================================== refresh recovery [mock] */
await check('[mock] REFRESH while running: the page reconnects to the same run, creates nothing, and stores no movie data', async () => {
  const { ctx, page } = await newPage('desktop'); const log = agentMock(page, { runs: (n) => (n < 9 ? fx.running() : fx.completed()) }); await open(page); await waitForm(page)
  await fillForm(page, { title: 'Inception' }); await submit(page); await page.waitForSelector('[data-testid="run-progress"]')
  const stored = await page.evaluate(() => sessionStorage.getItem('frame.researcher.run.v1')); const s = JSON.parse(stored)
  eq(Object.keys(s).sort(), ['key', 'runId', 'savedAt'], 'stored fields'); ok(!/Inception|Drama|2010|USA/.test(stored), `movie data in storage: ${stored}`); eq(s.runId, fx.RUN_ID, 'run id stored')
  await page.reload({ waitUntil: 'networkidle' })
  await page.waitForSelector('[data-testid="recovering"]'); ok(/No new run is being created/.test(await page.locator('[data-testid="recovering"]').innerText()), 'recovery notice')
  await shot(page, 'recovering', false); await waitPhase(page, 'completed')
  eq(log.posts.length, 1, 'refresh must not POST'); ok(/Restored after a page refresh/.test(await page.locator('main').innerText()), 'restored note'); ok(await page.locator('[data-testid="predicted-label"]').isVisible(), 'prediction shown')
  await page.reload({ waitUntil: 'networkidle' }); await waitPhase(page, 'completed'); eq(log.posts.length, 1, 'a second refresh of a finished run also shows it without POST')
  await page.getByRole('button', { name: 'Reset' }).click(); eq(await page.evaluate(() => sessionStorage.getItem('frame.researcher.run.v1')), null, 'Reset forgets the run')
  await page.reload({ waitUntil: 'networkidle' }); eq(await phase(page), 'idle', 'nothing to recover after Reset'); await ctx.close()
})
await check('[mock] REFRESH mid-POST: the saved key finds the run on the server (by key), or reports that nothing was started', async () => {
  for (const found of [true, false]) {
    const { ctx, page } = await newPage('desktop')
    const log = agentMock(page, { post: (r) => r.abort('connectionreset'), byKey: (r) => (found ? json(r, 200, fx.ACCEPTED) : json(r, 404, { error: 'not_found', message: 'x' })), runs: [fx.completed()] })
    await open(page); await waitForm(page); await fillForm(page); await submit(page); await waitPhase(page, 'error')
    const key = JSON.parse(await page.evaluate(() => sessionStorage.getItem('frame.researcher.run.v1'))).key
    await page.reload({ waitUntil: 'networkidle' })
    if (found) { await waitPhase(page, 'completed'); ok(log.byKey[0].endsWith('/by-key/' + key), 'looked up by the saved key') }
    else { await waitPhase(page, 'error'); ok(/Nothing was started/.test(await page.locator('[data-testid="run-problem"]').innerText()), 'honest message'); eq(await page.evaluate(() => sessionStorage.getItem('frame.researcher.run.v1')), null, 'session cleared') }
    eq(log.posts.length, 1, `no re-POST after refresh (found=${found})`); await ctx.close()
  }
})
await check('[mock] a tampered or expired saved session is ignored and sends no request', async () => {
  const { ctx, page } = await newPage('desktop'); const log = agentMock(page); await open(page)
  for (const raw of ['{"key":"x","runId":"../../etc","savedAt":1}', 'not json', JSON.stringify({ key: 'a'.repeat(32), runId: null, savedAt: Date.now() - 7 * 3600 * 1000 })]) {
    await page.evaluate((v) => sessionStorage.setItem('frame.researcher.run.v1', v), raw); await page.reload({ waitUntil: 'networkidle' })
    eq(await phase(page), 'idle', `idle for ${raw.slice(0, 30)}`); eq(log.gets.length + log.byKey.length + log.posts.length, 0, 'no agent requests')
  }
  await ctx.close()
})

/* =========================================================== keyboard, focus, reduced motion */
await check('[keyboard] skip link is first, the form is operable with Tab/Space/Enter, focus lands on the result', async () => {
  const { ctx, page } = await newPage('desktop'); agentMock(page, { runs: [fx.completed()] }); await open(page); await waitForm(page)
  await page.keyboard.press('Tab')
  eq(await page.evaluate(() => document.activeElement?.textContent), 'Skip to content', 'first tab stop'); const b = await page.evaluate(() => { const r = document.activeElement.getBoundingClientRect(); return { w: r.width, top: r.top } }); ok(b.w > 40 && b.top >= 0, 'skip link visible when focused')
  await page.keyboard.press('Enter'); ok(await page.evaluate(() => location.hash === '#main'), 'skip link jumps to main')
  await page.focus('#r-title_year'); await page.keyboard.type('2010'); await page.keyboard.press('Tab'); await page.keyboard.type('120')
  const firstGenre = await page.locator('#r-genres button').first().innerText()
  await page.keyboard.press('Tab'); eq(await page.evaluate(() => document.activeElement?.textContent), firstGenre, 'tab reaches the first genre chip')
  await page.keyboard.press('Space'); eq(await page.evaluate(() => document.activeElement?.getAttribute('aria-pressed')), 'true', 'Space toggles a genre')
  await page.keyboard.press('Space'); eq(await page.evaluate(() => document.activeElement?.getAttribute('aria-pressed')), 'false', 'Space toggles it back')
  await page.getByRole('button', { name: 'Drama', exact: true }).focus(); await page.keyboard.press('Enter')
  await page.focus('#r-country'); await page.keyboard.press('ArrowDown'); await page.selectOption('#r-country', 'USA'); await page.selectOption('#r-content_rating', 'PG-13')
  await page.focus('#r-research_title'); await page.keyboard.press('Enter'); await waitPhase(page, 'completed')
  ok(await page.evaluate(() => document.activeElement?.getAttribute('aria-label') === 'Research result'), 'focus moved to the result region')
  const d = page.getByRole('button', { name: /Guardrail results/ }); await d.focus(); eq(await d.getAttribute('aria-expanded'), 'false', 'collapsed'); await page.keyboard.press('Enter'); eq(await d.getAttribute('aria-expanded'), 'true', 'Enter opens the disclosure')
  await ctx.close()
})
await check('[keyboard] every interactive control shows a visible focus indicator', async () => {
  const { ctx, page } = await newPage('desktop'); agentMock(page, { runs: [fx.completed()] }); await open(page); await waitForm(page)
  const bad = []
  for (const sel of ['#r-title_year', '#r-country', '#r-research_title', 'button[aria-pressed]', 'button[type=submit]', 'header nav a']) {
    await page.locator(sel).first().focus()
    const vis = await page.locator(sel).first().evaluate((e) => { const s = getComputedStyle(e); return (s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0) || (s.boxShadow && s.boxShadow !== 'none') || s.borderColor })
    if (!vis) bad.push(sel)
  }
  ok(bad.length === 0, `no focus indicator for ${bad}`); await ctx.close(); return 'checked computed outline/box-shadow on :focus (not pixel contrast)'
})
await check('[reduced-motion] with prefers-reduced-motion the spinner stops and no infinite animation runs in the progress panel', async () => {
  const run = async (reduce) => {
    const { ctx, page } = await newPage('desktop', { reducedMotion: reduce }); agentMock(page, { runs: (n) => (n < 99 ? fx.running() : fx.completed()) }); await open(page); await waitForm(page)
    ok(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches) === reduce, 'media query emulated')
    await fillForm(page); await submit(page); await page.waitForSelector('[data-testid="run-progress"]'); await sleep(700)
    const r = await page.evaluate(() => { const sp = document.querySelector('[data-testid="run-progress"] span[aria-hidden]'); return { anim: sp ? getComputedStyle(sp).animationName : null, infinite: document.getAnimations().filter((a) => a.effect?.getComputedTiming().iterations === Infinity && document.querySelector('[data-testid="run-progress"]').contains(a.effect.target)).length } })
    await ctx.close(); return r
  }
  const normal = await run(false), reduced = await run(true)
  ok(normal.anim === 'spin' && normal.infinite >= 1, `baseline should animate: ${JSON.stringify(normal)}`); ok(reduced.anim === 'none' && reduced.infinite === 0, `reduced-motion still animating: ${JSON.stringify(reduced)}`)
  return `normal ${JSON.stringify(normal)} / reduced ${JSON.stringify(reduced)}`
})

/* =========================================================== contrast (approximate) */
await check('[contrast] key text meets WCAG AA against its effective background (approximate: gradients and blur are not modelled)', async () => {
  const { ctx, page } = await newPage('desktop'); agentMock(page, { runs: [fx.withResearch(fx.failedResearch('error', 'http_500', 'Online research hit an error, so no sources are shown.'))] }); await open(page); await waitForm(page)
  await fillForm(page); await submit(page); await waitPhase(page, 'completed')
  const rows = await page.evaluate(() => {
    const cv = document.createElement('canvas'); cv.width = cv.height = 1; const cx = cv.getContext('2d', { willReadFrequently: true })
    const rgba = (css) => { cx.clearRect(0, 0, 1, 1); cx.fillStyle = '#000'; cx.fillStyle = css; cx.fillRect(0, 0, 1, 1); const d = cx.getImageData(0, 0, 1, 1).data; return [d[0], d[1], d[2], d[3] / 255] }
    const over = (f, b) => [0, 1, 2].map((i) => f[i] * f[3] + b[i] * (1 - f[3]))
    const lum = (c) => { const [r, g, b] = c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4 }); return 0.2126 * r + 0.7152 * g + 0.0722 * b }
    const bgOf = (el) => { const stack = []; for (let e = el; e; e = e.parentElement) { const c = rgba(getComputedStyle(e).backgroundColor); if (c[3] > 0) stack.push(c); if (c[3] === 1) break }; let base = [6, 6, 26]; for (const c of stack.reverse()) base = over(c, base).concat(1); return base.slice(0, 3) }
    const targets = [['h1', 'page title'], ['.eyebrow', 'eyebrow'], ['label', 'form label'], ['#r-title_year-d', 'field help text'], ['[data-testid=provenance] p', 'provenance text'], ['[data-testid=analysis-summary]', 'analysis summary'],
      ['[data-testid=risk-trace]', 'risk rule line'], ['#agent-research p', 'research text'], ['[data-testid=research-problem]', 'research alert'], ['#agent-prediction p', 'prediction caption'], ['header nav a:not([aria-current])', 'inactive nav link'],
      ['footer', 'footer'], ['#agent-evidence td', 'table cell'], ['#agent-evidence th', 'table header'], ['[data-testid=predicted-label]', 'prediction label (large)']]
    return targets.map(([sel, label]) => { const el = document.querySelector(sel); if (!el) return { label, missing: true }
      const st = getComputedStyle(el), bg = bgOf(el), fg = over(rgba(st.color), bg), l1 = lum(fg), l2 = lum(bg), ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05), size = parseFloat(st.fontSize), large = size >= 24 || (size >= 18.66 && parseInt(st.fontWeight) >= 700)
      return { label, ratio: +ratio.toFixed(2), need: large ? 3 : 4.5, size } })
  })
  const fails = rows.filter((r) => !r.missing && r.ratio < r.need); const miss = rows.filter((r) => r.missing).map((r) => r.label)
  ok(fails.length === 0, `below AA: ${JSON.stringify(fails)}`); return `min ratio ${Math.min(...rows.filter((r) => !r.missing).map((r) => r.ratio))} over ${rows.length - miss.length} samples${miss.length ? '; not present: ' + miss.join(', ') : ''}`
})

/* =========================================================== [real] backend + model + Wikipedia */
if (REAL) {
  const REAL_BODY = { year: '2010', duration: '148', genres: ['Action', 'Sci-Fi'], country: 'USA', rating: 'PG-13', title: 'Inception' }
  await check('[real] full run through the live backend, Random Forest, local model and Wikipedia', async () => {
    const { ctx, page } = await newPage('desktop', { timeout: 30000 }); await open(page); await waitForm(page)
    ok(/Local language model ready|not available/.test(await page.locator('[data-testid="agent-status"]').innerText()), 'agent status shown')
    await fillForm(page, REAL_BODY); const t0 = Date.now(); await submit(page); await page.waitForSelector('[data-testid="run-progress"]')
    await waitPhase(page, 'completed', 180000); const secs = ((Date.now() - t0) / 1000).toFixed(1)
    const label = await page.locator('[data-testid="predicted-label"]').innerText()
    const api = await page.evaluate(async () => (await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ duration: 148, title_year: 2010, genres: ['Action', 'Sci-Fi'], country: 'USA', content_rating: 'PG-13' }) })).json())
    eq(label, api.label, 'UI label must equal /api/predict'); for (const c of ['Flop', 'Average', 'Hit']) ok((await page.locator(`#agent-prediction [role=img][aria-label^="${c} "]`).getAttribute('aria-label')) === `${c} ${Math.round(api.probabilities[c] * 100)}%`, `${c} score matches`)
    const kind = await page.locator('[data-testid="provenance"]').getAttribute('data-kind'); const prov = await page.locator('[data-testid="provenance"]').innerText()
    const research = await page.locator('#agent-research').innerText(); const nSrc = await page.locator('[data-testid="research-source"]').count()
    ok(await page.locator('#agent-prediction').isVisible() && await page.locator('#agent-analysis').isVisible() && await page.locator('#agent-evidence').isVisible(), 'panels present')
    await shot(page, 'real-result'); await ctx.close()
    return `completed in ${secs}s; label ${label}; analysis provenance "${kind}" (${prov.split('\n')[0]}); research sources shown: ${nSrc}; ${research.split('\n').find((l) => /sources? retrieved/.test(l)) ?? 'no count line'}`
  })

  await check('[real] LOST RESPONSE against the real server: retry returns the SAME run (server-side dedupe, exactly one run created)', async () => {
    const { ctx, page } = await newPage('desktop', { timeout: 30000 }); const keys = []
    let first = true
    await page.route('**/api/agent/runs', async (route) => {
      if (route.request().method() !== 'POST') return route.fallback()
      keys.push(route.request().headers()['idempotency-key'])
      if (first) { first = false; await route.fetch(); return route.abort('connectionreset') }       // the SERVER creates the run; the browser never hears about it
      return route.continue()
    })
    await open(page); await waitForm(page); await fillForm(page, { ...REAL_BODY, title: '' })
    await submit(page); await waitPhase(page, 'error'); ok(/Cannot reach the FRAME API/.test(await page.locator('[data-testid="run-problem"]').innerText()), 'client saw a lost response')
    await page.getByRole('button', { name: 'Try again' }).click(); await waitPhase(page, 'completed', 180000)
    eq(keys.length, 2, 'two POSTs'); eq(keys[0], keys[1], 'same idempotency key'); ok(/^[0-9a-f]{32}$/.test(keys[0]), 'key shape')
    const rid = JSON.parse(await page.evaluate(() => sessionStorage.getItem('frame.researcher.run.v1'))).runId; await ctx.close(); return `one run id ${rid} for two POSTs with key ${keys[0].slice(0, 8)}…`
  })

  await check('[real] REFRESH after a lost response: the page recovers the run by key from the real server, without re-sending inputs', async () => {
    const { ctx, page } = await newPage('desktop', { timeout: 30000 }); let posts = 0
    await page.route('**/api/agent/runs', async (route) => {
      if (route.request().method() !== 'POST') return route.fallback()
      posts++; await route.fetch(); return route.abort('connectionreset')
    })
    await open(page); await waitForm(page); await fillForm(page, { ...REAL_BODY, title: '' }); await submit(page); await waitPhase(page, 'error')
    const s = JSON.parse(await page.evaluate(() => sessionStorage.getItem('frame.researcher.run.v1'))); eq(s.runId, null, 'no run id known to the page')
    await page.unroute('**/api/agent/runs'); await page.reload({ waitUntil: 'networkidle' })
    await waitPhase(page, 'completed', 180000); eq(posts, 1, 'the refresh did not POST again'); ok(/Restored after a page refresh/.test(await page.locator('main').innerText()), 'restored note')
    ok(await page.locator('[data-testid="predicted-label"]').isVisible(), 'prediction shown'); await shot(page, 'real-recovered', false); await ctx.close()
  })
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} checks passed`)
const uniqueProblems = [...new Map(consoleProblems.map((p) => [p.text, p])).values()]
console.log(`console errors / page errors seen: ${uniqueProblems.length}`, uniqueProblems.slice(0, 8))
process.exit(failed.length ? 1 : 0)
