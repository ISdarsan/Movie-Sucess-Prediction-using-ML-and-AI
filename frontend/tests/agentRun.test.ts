/* Logic tests for the submit-and-poll controller, the response normaliser and the provenance labels. No browser, no network, no real waiting:
   a fake clock advances inside sleep(). Run: npm test */
import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { createRunController, DEFAULT_POLL, IDEMPOTENCY_HEADER, normalizeRun, provenance, reasonText, type RunState } from '../src/lib/agentRun.ts'
import { loadSession, saveSession, SESSION_KEY, SESSION_TTL_MS, type Session } from '../src/lib/runSession.ts'
import { ACCEPTED, analysis, completed, evidence, failedResearch, failedRun, fallbackRun, guardrails, partialRun, research, risk, running, RUN_ID, source, withResearch } from './fixtures.ts'

class FakeApiError extends Error {
  status: number; code: string; fields: { field: string; message: string }[]
  constructor(status: number, code: string, message: string, fields: { field: string; message: string }[] = []) { super(message); this.status = status; this.code = code; this.fields = fields }
}

type Reply = unknown | (() => unknown)
const K1 = 'a'.repeat(32), K2 = 'b'.repeat(32), K3 = 'c'.repeat(32)

/** in-memory stand-in for the controller's tab-scoped session store, plus the raw map so tests can inspect exactly what was written */
function memorySession(initial?: Session | null) {
  const mem = new Map<string, string>()
  const store = { getItem: (k: string) => mem.get(k) ?? null, setItem: (k: string, v: string) => void mem.set(k, v), removeItem: (k: string) => void mem.delete(k) }
  if (initial) saveSession(store, initial)
  return { mem, store, session: { load: () => loadSession(store, T0 + 1000), save: (s: Session) => saveSession(store, s), clear: () => store.removeItem(SESSION_KEY) } }
}
const T0 = 1_000_000

function harness(opts: { post?: Reply[]; get?: Reply[]; config?: Partial<typeof DEFAULT_POLL>; session?: ReturnType<typeof memorySession> } = {}) {
  let t = 0
  const posts: { path: string; body: unknown; headers?: Record<string, string> }[] = []
  const keys = [K1, K2, K3]; let ki = 0
  const gets: string[] = []
  const sleeps: number[] = []
  const states: RunState[] = []
  const postQ = [...(opts.post ?? [ACCEPTED])], getQ = [...(opts.get ?? [completed()])]
  const take = (q: Reply[]) => { const r = q.length > 1 ? q.shift() : q[0]; const v = typeof r === 'function' ? (r as () => unknown)() : r; if (v instanceof Error) throw v; return v }
  const ctl = createRunController({
    post: async (path, body, headers) => { posts.push({ path, body, headers }); return take(postQ) },
    get: async (path) => { gets.push(path); return take(getQ) },
    now: () => T0 + t, sleep: async (ms) => { sleeps.push(ms); t += ms }, onChange: (s) => states.push(s), config: opts.config, newKey: () => keys[ki++] ?? K3, session: opts.session?.session,
  })
  return { ctl, posts, gets, sleeps, states, phases: () => states.map((s) => s.phase).filter((p, i, a) => a.indexOf(p) === i || a[i - 1] !== p) }
}

const BODY = { duration: 120, title_year: 2010, genres: ['Drama'], country: 'USA', content_rating: 'PG-13' }

describe('submission and polling', () => {
  it('POSTs the body once to /api/agent/runs, then polls that run until completed', async () => {
    const h = harness({ get: [running(), running(), completed()] })
    await h.ctl.start(BODY)
    assert.equal(h.posts.length, 1); assert.equal(h.posts[0].path, '/api/agent/runs'); assert.deepEqual(h.posts[0].body, BODY)
    assert.deepEqual(h.posts[0].headers, { [IDEMPOTENCY_HEADER]: K1 }, 'every submission carries an idempotency key')
    assert.deepEqual(h.gets, Array(3).fill(`/api/agent/runs/${RUN_ID}`))
    assert.deepEqual(h.phases(), ['submitting', 'running', 'completed'])
    const end = h.ctl.getState()
    assert.equal(end.runId, RUN_ID); assert.equal(end.run?.prediction?.label, 'Hit'); assert.equal(end.polls, 3)
  })

  it('reports the API status as given while running', async () => {
    const h = harness({ get: [{ run_id: RUN_ID, status: 'queued' }, running(), completed()] })
    await h.ctl.start(BODY)
    const seen = h.states.map((s) => s.apiStatus).filter(Boolean)
    assert.ok(seen.includes('queued') && seen.includes('running') && seen.includes('completed'))
  })

  it('a second start while a run is in flight is ignored, so a double click cannot create two runs', async () => {
    const h = harness({ get: [running(), completed()] })
    const first = h.ctl.start(BODY)
    const second = await h.ctl.start(BODY)
    await first
    assert.equal(second, false); assert.equal(h.posts.length, 1)
  })

  it('polls quickly at first, then backs off, and stays inside the wait budget', async () => {
    const h = harness({ get: [running()], config: { maxWaitMs: 60_000 } })
    await h.ctl.start(BODY)
    assert.equal(h.sleeps[0], DEFAULT_POLL.intervalMs)
    assert.equal(h.sleeps.at(-1), DEFAULT_POLL.slowIntervalMs)
    assert.ok(h.gets.length < 60_000 / DEFAULT_POLL.intervalMs, 'polling is bounded by the budget, not unbounded')
    assert.equal(h.ctl.getState().phase, 'timeout')
  })
})

describe('terminal states', () => {
  it('completed: prediction and analysis are available', async () => {
    const h = harness(); await h.ctl.start(BODY)
    const s = h.ctl.getState()
    assert.equal(s.phase, 'completed'); assert.ok(s.run?.analysis); assert.equal(s.problem, null)
  })

  it('failed: surfaces the API error and keeps the trace', async () => {
    const h = harness({ get: [running(), failedRun()] }); await h.ctl.start(BODY)
    const s = h.ctl.getState()
    assert.equal(s.phase, 'failed'); assert.equal(s.run?.error?.message, 'The run could not produce a prediction.'); assert.equal(s.run?.trace[0].detail, 'model_unavailable')
  })

  it('failed with an empty body (e.g. a restart marked it failed) is still a failed run, not a crash', async () => {
    const h = harness({ get: [{ run_id: RUN_ID, status: 'failed', created: 1 }] }); await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().phase, 'failed'); assert.equal(h.ctl.getState().run?.error, null)
  })

  it('completed without a usable prediction is an error, never a blank result', async () => {
    const h = harness({ get: [completed({ prediction: { label: 'Blockbuster' } })] }); await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().phase, 'error'); assert.equal(h.ctl.getState().problem?.code, 'bad_response')
  })

  it('an unrecognisable poll body is an error', async () => {
    const h = harness({ get: [{ hello: 'world' }] }); await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().problem?.code, 'bad_response')
  })
})

describe('API errors', () => {
  it('422 on submit keeps the per-field messages and creates no run', async () => {
    const e = new FakeApiError(422, 'validation_error', 'Some inputs are invalid.', [{ field: 'duration', message: 'Input should be greater than or equal to 1' }])
    const h = harness({ post: [e] }); await h.ctl.start(BODY)
    const s = h.ctl.getState()
    assert.equal(s.phase, 'error'); assert.equal(s.runId, null); assert.deepEqual(s.problem?.fields, [{ field: 'duration', message: 'Input should be greater than or equal to 1' }]); assert.equal(h.gets.length, 0)
  })

  it('429 busy and 503 model_unavailable surface their codes', async () => {
    for (const [status, code] of [[429, 'busy'], [503, 'model_unavailable']] as const) {
      const h = harness({ post: [new FakeApiError(status, code, 'nope')] }); await h.ctl.start(BODY)
      assert.equal(h.ctl.getState().problem?.code, code)
    }
  })

  it('network failure on submit is an error with no run id, and an explicit retry is the only thing that POSTs again', async () => {
    const h = harness({ post: [new FakeApiError(0, 'network', 'Cannot reach the FRAME API.'), ACCEPTED], get: [completed()] })
    await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().phase, 'error'); assert.equal(h.posts.length, 1)
    assert.equal(await h.ctl.resume(), false, 'resume must not POST when there is no run to resume'); assert.equal(h.posts.length, 1)
    await h.ctl.start(BODY)
    assert.equal(h.posts.length, 2); assert.equal(h.ctl.getState().phase, 'completed')
  })

  it('a submit response without a valid run id is an error', async () => {
    for (const bad of [{}, { run_id: 'x' }, { run_id: 123 }, null, 'oops']) {
      const h = harness({ post: [bad] }); await h.ctl.start(BODY)
      assert.equal(h.ctl.getState().problem?.code, 'bad_response', JSON.stringify(bad))
      assert.equal(h.gets.length, 0)
    }
  })

  it('a few transient poll failures are tolerated; the run still completes', async () => {
    const down = () => new FakeApiError(0, 'network', 'offline')
    const h = harness({ get: [down, down, down, completed()], config: { maxConsecutiveErrors: 4 } }); await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().phase, 'completed')
  })

  it('too many consecutive poll failures stop polling with the run id kept', async () => {
    const down = () => new FakeApiError(502, 'http_error', 'bad gateway')
    const h = harness({ get: [down], config: { maxConsecutiveErrors: 3 } }); await h.ctl.start(BODY)
    const s = h.ctl.getState()
    assert.equal(s.phase, 'error'); assert.equal(s.runId, RUN_ID); assert.equal(h.gets.length, 3)
  })

  it('404 while polling is terminal: the run is gone', async () => {
    const h = harness({ get: [new FakeApiError(404, 'not_found', 'No such run.')] }); await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().problem?.code, 'run_not_found'); assert.equal(h.gets.length, 1)
  })

  it('other 4xx while polling are not retried', async () => {
    const h = harness({ get: [new FakeApiError(400, 'bad', 'bad request')] }); await h.ctl.start(BODY)
    assert.equal(h.gets.length, 1); assert.equal(h.ctl.getState().phase, 'error')
  })
})

describe('timeout and retry without duplicates', () => {
  it('times out at the wait budget, keeps the run id, and resume() keeps polling the SAME run', async () => {
    const h = harness({ get: [running(), running(), running(), running(), running(), completed()], config: { maxWaitMs: 3000, intervalMs: 1000, slowAfterMs: 99_999 } })
    await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().phase, 'timeout'); assert.equal(h.ctl.getState().runId, RUN_ID); assert.equal(h.ctl.getState().problem?.code, 'poll_timeout')
    assert.equal(h.posts.length, 1)
    assert.equal(await h.ctl.resume(), true)
    assert.equal(h.posts.length, 1, 'resuming never creates a second run')
    assert.ok(h.gets.every((g) => g === `/api/agent/runs/${RUN_ID}`))
    assert.equal(h.ctl.getState().phase, 'completed')
  })

  it('resume after a lost connection also re-reads the same run', async () => {
    const down = () => new FakeApiError(0, 'network', 'offline')
    const h = harness({ get: [down, down, completed()], config: { maxConsecutiveErrors: 2 } })
    await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().phase, 'error')
    await h.ctl.resume()
    assert.equal(h.posts.length, 1); assert.equal(h.ctl.getState().phase, 'completed')
  })

  it('resume is refused while running or after completion', async () => {
    const h = harness(); await h.ctl.start(BODY)
    assert.equal(await h.ctl.resume(), false)
  })

  it('a new start after a terminal state is a deliberate new run', async () => {
    const h = harness({ post: [ACCEPTED, { ...ACCEPTED, run_id: 'aaaaaaaaaaaa' }], get: [completed(), completed({ run_id: 'aaaaaaaaaaaa' })] })
    await h.ctl.start(BODY); await h.ctl.start(BODY)
    assert.equal(h.posts.length, 2); assert.equal(h.ctl.getState().runId, 'aaaaaaaaaaaa')
    assert.notEqual(h.posts[0].headers![IDEMPOTENCY_HEADER], h.posts[1].headers![IDEMPOTENCY_HEADER], 'a deliberate new run gets a new key')
  })

  it('cancel() (unmount) stops writing state even if a response arrives later', async () => {
    let release!: (v: unknown) => void
    const h = harness({ get: [() => new Promise((r) => { release = r })] })
    const p = h.ctl.start(BODY)
    await Promise.resolve(); await Promise.resolve(); await Promise.resolve()
    const before = h.states.length
    h.ctl.cancel(); release(completed()); await p
    assert.equal(h.states.length, before)
  })
})

describe('normalizeRun: what the API supplies', () => {
  it('reads structured risk items with every field', () => {
    const r = normalizeRun(completed())!
    assert.deepEqual(r.analysis!.risks, [{ id: 'sensitivity_no_change', category: 'sensitivity', severity: 'info', text: risk().text, rule: 'one top-class segment across the sweep', evidence: { 'sensitivity.segments': 1, 'sensitivity.top_class': 'Hit' } }])
    assert.equal(r.analysis!.risksMalformed, 0)
  })

  it('does NOT accept the old string-array risk schema', () => {
    const r = normalizeRun(completed({ analysis: analysis({ risks: ['The cohort is small.', 'Scores are uncalibrated.', risk()] }) }))!
    assert.equal(r.analysis!.risks.length, 1); assert.equal(r.analysis!.risksMalformed, 2)
    assert.ok(!r.analysis!.risks.some((x) => x.text === 'The cohort is small.'))
  })

  it('drops risk objects that miss required fields instead of inventing them', () => {
    const r = normalizeRun(completed({ analysis: analysis({ risks: [{ id: 'x', text: 'no category' }, risk({ severity: 5 })] }) }))!
    assert.equal(r.analysis!.risks.length, 0); assert.equal(r.analysis!.risksMalformed, 2)
  })

  it('comparables: explicit null is UNAVAILABLE (the tool failed), [] is "none found", absent is missing', () => {
    const ev = (c: unknown, drop = false) => { const e: Record<string, unknown> = { ...evidence(), comparables: c }; if (drop) delete e.comparables; return normalizeRun(completed({ evidence: e }))!.comparables }
    assert.deepEqual(ev(null), { state: 'unavailable' })
    assert.deepEqual(ev([]), { state: 'ok', value: [] })
    assert.deepEqual(ev(undefined, true), { state: 'missing' })
    assert.deepEqual(ev('junk'), { state: 'unavailable' })
    assert.equal((ev(evidence().comparables) as { value: unknown[] }).value.length, 2)
  })

  it('cohort and sensitivity null are unavailable; a malformed cohort is unavailable, not guessed', () => {
    const r = normalizeRun(completed({ evidence: { ...evidence(), cohort: null, sensitivity: null } }))!
    assert.deepEqual([r.cohort, r.sensitivity], [{ state: 'unavailable' }, { state: 'unavailable' }])
    assert.deepEqual(normalizeRun(completed({ evidence: { ...evidence(), cohort: { final: { n: 'x' } } } }))!.cohort, { state: 'unavailable' })
  })

  it('research stub is reported as not available, with no sources', () => {
    const r = normalizeRun(completed())!.research
    assert.equal(r.state, 'ok'); if (r.state === 'ok') { assert.equal(r.value.available, false); assert.deepEqual(r.value.sources, []) }
  })

  it('missing generated_by is treated as a fallback, never as model-written', () => {
    const a = analysis(); delete (a as Record<string, unknown>).generated_by
    assert.equal(normalizeRun(completed({ analysis: a }))!.analysis!.generatedBy.fallback, true)
  })

  it('rejects things that are not runs', () => {
    for (const bad of [null, [], 'x', {}, { run_id: 1, status: 'x' }, { run_id: 'a' }]) assert.equal(normalizeRun(bad), null)
  })

  it('analysis may be absent (a run can finish without one)', () => {
    const r = normalizeRun(completed({ analysis: undefined }))!
    assert.equal(r.analysis, null); assert.equal(r.prediction?.label, 'Hit')
  })
})

describe('provenance: LLM, partial replacement and fallback are visible', () => {
  const p = (run: Record<string, unknown>) => { const r = normalizeRun(run)!; return provenance(r.analysis!, r.guardrails) }

  it('model-written, nothing replaced', () => {
    const v = p(completed()); assert.equal(v.kind, 'llm'); assert.match(v.headline, /Written by llama3\.1:8b/)
  })
  it('partial replacement lists what was replaced', () => {
    const v = p(partialRun()); assert.equal(v.kind, 'llm_partial'); assert.match(v.detail, /driver 2/)
  })
  it('replaced items alone make it partial even if the partial flag is missing', () => {
    const v = p(completed({ guardrails: guardrails({ replaced: ['summary'] }) })); assert.equal(v.kind, 'llm_partial')
  })
  it('fallback says it was NOT written by a language model, with the reason', () => {
    const v = p(fallbackRun('timeout')); assert.equal(v.kind, 'template'); assert.match(v.headline, /not written by a language model/); assert.match(v.detail, /timed out/)
  })
  it('reason codes map to plain words; unknown codes are shown, not hidden', () => {
    assert.match(reasonText('ollama_unreachable')!, /not reachable/); assert.equal(reasonText('some_new_reason'), 'some new reason'); assert.equal(reasonText(null), null)
  })
})

describe('normalizeRun: contextual research block', () => {
  const read = (r: unknown) => normalizeRun(withResearch(r))!.research

  it('reads status, counts, cutoff, request usage and every source field', () => {
    const r = read(research())
    assert.equal(r.state, 'ok'); if (r.state !== 'ok') return
    assert.equal(r.value.status, 'ok'); assert.equal(r.value.available, true); assert.equal(r.value.sources.length, 2)
    assert.equal(r.value.cutoff, '2010-01-01T00:00:00Z'); assert.equal(r.value.requestsMade, 7); assert.equal(r.value.maxRequests, 8)
    assert.deepEqual(r.value.query, { title: 'Inception', genres: ['Action', 'Sci-Fi'] })
    const s0 = r.value.sources[0]
    assert.equal(s0.title, 'Inception'); assert.equal(s0.kind, 'film'); assert.equal(s0.revisionId, 334798969)
    assert.equal(s0.revisionTimestamp, '2009-12-29T22:16:08Z'); assert.equal(s0.retrievedAt, '2026-10-29T19:54:02Z'); assert.match(s0.url, /^https:\/\/en\.wikipedia\.org\/w\/index\.php/)
  })

  it('failed research keeps its status and reports zero sources, never an invented one', () => {
    for (const [st, reason] of [['timeout', 'timeout'], ['unavailable', 'connection_failed'], ['error', 'http_500'], ['no_results', 'no_pre_cutoff_pages']]) {
      const r = read(failedResearch(st, reason, 'msg'))
      assert.equal(r.state, 'ok'); if (r.state === 'ok') { assert.equal(r.value.status, st); assert.equal(r.value.available, false); assert.deepEqual(r.value.sources, []) }
    }
  })

  it('the old stub shape still reads, as disabled', () => {
    const r = read({ available: false, reason: 'research_not_enabled', sources: [] })
    assert.equal(r.state, 'ok'); if (r.state === 'ok') { assert.equal(r.value.status, 'disabled'); assert.equal(r.value.sources.length, 0) }
  })

  it('links that are not https Wikipedia pages are dropped and counted', () => {
    const bad = ['javascript:alert(1)', 'http://en.wikipedia.org/w/index.php?title=X', 'https://evil.example.com/w/index.php', 'https://en.wikipedia.org.evil.com/w/x',
      'https://user:pw@en.wikipedia.org/w/x', 'https://en.wikipedia.org:8443/w/x', '//evil.com/x', 'not a url', 42 as never]
    const r = read(research({ sources: [...bad.map((u) => source('genre', 'Bad', { url: u })), source('genre', 'Good')] }))
    assert.equal(r.state, 'ok'); if (r.state === 'ok') { assert.deepEqual(r.value.sources.map((x) => x.title), ['Good']); assert.equal(r.value.sourcesRejected, bad.length) }
  })

  it('the source count comes from sources that survive the checks, not from the available flag', () => {
    const r = read(research({ available: true, sources: [source('genre', 'Bad', { url: 'http://evil.example.com' })] }))
    assert.equal(r.state, 'ok'); if (r.state === 'ok') { assert.equal(r.value.available, false); assert.equal(r.value.sources.length, 0) }
  })

  it('skipped subjects are kept; missing and malformed research are distinguished', () => {
    const r = read(research({ skipped: [{ subject: 'Drama (film and television)', reason: 'redirect_page_at_cutoff' }] }))
    if (r.state === 'ok') assert.deepEqual(r.value.skipped, [{ subject: 'Drama (film and television)', reason: 'redirect_page_at_cutoff' }])
    const ev = evidence() as Record<string, unknown>; delete ev.research
    assert.deepEqual(normalizeRun(completed({ evidence: ev }))!.research, { state: 'missing' })
    assert.deepEqual(read('junk'), { state: 'unavailable' }); assert.deepEqual(read({ status: 'ok' }), { state: 'unavailable' }); assert.deepEqual(read(null), { state: 'unavailable' })
  })

  it('reason codes for research are worded in plain language', () => {
    assert.match(reasonText('no_matching_film_page')!, /exact film title/); assert.match(reasonText('connection_failed')!, /connection/); assert.match(reasonText('redirect_page_at_cutoff')!, /redirect/)
  })
})

describe('duplicate-run protection: the same submission always carries the same key', () => {
  it('retry() after a failed submission re-sends the same inputs with the SAME key', async () => {
    const lost = new FakeApiError(0, 'network', 'Cannot reach the FRAME API.')
    const h = harness({ post: [lost, ACCEPTED], get: [completed()] })
    await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().phase, 'error')
    assert.equal(await h.ctl.retry(), true)
    assert.equal(h.posts.length, 2); assert.deepEqual(h.posts[1].body, BODY)
    assert.equal(h.posts[1].headers![IDEMPOTENCY_HEADER], h.posts[0].headers![IDEMPOTENCY_HEADER], 'same key, so the server can return the run it already made')
    assert.equal(h.ctl.getState().phase, 'completed')
  })

  it('retry() is refused when there is nothing to retry or a run exists', async () => {
    const h = harness(); assert.equal(await h.ctl.retry(), false)
    await h.ctl.start(BODY); assert.equal(await h.ctl.retry(), false); assert.equal(h.posts.length, 1)
  })

  it('a server 409 for a reused key surfaces as an error, never as a silent second run', async () => {
    const h = harness({ post: [new FakeApiError(409, 'idempotency_conflict', 'That key was used with different inputs.')] })
    await h.ctl.start(BODY)
    assert.equal(h.ctl.getState().problem?.code, 'idempotency_conflict'); assert.equal(h.gets.length, 0)
  })

  it('a replayed response (same run id as before) is polled like any other', async () => {
    const h = harness({ post: [{ ...ACCEPTED, status: 'completed' }], get: [completed()] })
    await h.ctl.start(BODY); assert.equal(h.ctl.getState().phase, 'completed')
  })
})

describe('refresh recovery', () => {
  const NOW_SESSION = { key: K1, savedAt: T0 }

  it('start() saves the key BEFORE the request leaves, then the run id, and stores nothing else', async () => {
    const mem = memorySession()
    let beforePost: string | null = null
    const h = harness({ session: mem, post: [() => { beforePost = mem.mem.get(SESSION_KEY) ?? null; return ACCEPTED }] })
    await h.ctl.start(BODY)
    assert.deepEqual(JSON.parse(beforePost!), { key: K1, runId: null, savedAt: T0 })
    const saved = JSON.parse(mem.mem.get(SESSION_KEY)!)
    assert.deepEqual(Object.keys(saved).sort(), ['key', 'runId', 'savedAt']); assert.equal(saved.runId, RUN_ID)
    assert.ok(!mem.mem.get(SESSION_KEY)!.includes('Drama') && !mem.mem.get(SESSION_KEY)!.includes('2010'), 'no movie inputs are stored')
  })

  it('recover() with a saved run id polls that run and never POSTs', async () => {
    const mem = memorySession({ ...NOW_SESSION, runId: RUN_ID })
    const h = harness({ session: mem, get: [running(), completed()] })
    assert.equal(await h.ctl.recover(), true)
    assert.equal(h.posts.length, 0); assert.deepEqual(h.gets, Array(2).fill(`/api/agent/runs/${RUN_ID}`))
    const s = h.ctl.getState(); assert.equal(s.phase, 'completed'); assert.equal(s.recovered, true); assert.equal(s.run?.prediction?.label, 'Hit')
  })

  it('a refresh mid-POST (key saved, no run id) is resolved by asking the server by key: found', async () => {
    const mem = memorySession({ ...NOW_SESSION, runId: null })
    const h = harness({ session: mem, get: [{ run_id: RUN_ID, status: 'running', poll_url: 'x' }, completed()] })
    await h.ctl.recover()
    assert.equal(h.posts.length, 0); assert.equal(h.gets[0], `/api/agent/runs/by-key/${K1}`); assert.equal(h.gets[1], `/api/agent/runs/${RUN_ID}`)
    assert.equal(h.ctl.getState().phase, 'completed'); assert.equal(JSON.parse(mem.mem.get(SESSION_KEY)!).runId, RUN_ID)
  })

  it('...and when the server never saw the submission it says so, clears the session and creates nothing', async () => {
    const mem = memorySession({ ...NOW_SESSION, runId: null })
    const h = harness({ session: mem, get: [new FakeApiError(404, 'not_found', 'No run was created for that key.')] })
    await h.ctl.recover()
    const s = h.ctl.getState()
    assert.equal(s.phase, 'error'); assert.equal(s.problem?.code, 'submission_not_received'); assert.equal(h.posts.length, 0); assert.equal(mem.mem.size, 0)
  })

  it('a network failure during recovery keeps the session so "check again" works, and still never POSTs', async () => {
    const mem = memorySession({ ...NOW_SESSION, runId: null })
    const h = harness({ session: mem, get: [new FakeApiError(0, 'network', 'offline'), { run_id: RUN_ID, status: 'completed' }, completed()] })
    await h.ctl.recover()
    assert.equal(h.ctl.getState().problem?.code, 'recover_failed'); assert.ok(mem.mem.has(SESSION_KEY))
    assert.equal(await h.ctl.recover(), true)
    assert.equal(h.ctl.getState().phase, 'completed'); assert.equal(h.posts.length, 0)
  })

  it('a saved run that the server no longer has ends cleanly and forgets the session', async () => {
    const mem = memorySession({ ...NOW_SESSION, runId: RUN_ID })
    const h = harness({ session: mem, get: [new FakeApiError(404, 'not_found', 'No such run.')] })
    await h.ctl.recover()
    assert.equal(h.ctl.getState().problem?.code, 'run_not_found'); assert.equal(mem.mem.size, 0)
  })

  it('nothing saved, expired, malformed or tampered: recover() does nothing and sends no request', async () => {
    const cases: (string | null)[] = [null, 'not json', '{}', JSON.stringify({ key: 'short', runId: null, savedAt: T0 }), JSON.stringify({ key: K1, runId: '../etc/passwd', savedAt: T0 }),
      JSON.stringify({ key: K1, runId: null, savedAt: T0 - SESSION_TTL_MS - 1 }), JSON.stringify({ key: K1, runId: null, savedAt: T0 + 10 ** 9 }), JSON.stringify([K1])]
    for (const raw of cases) {
      const mem = memorySession(); if (raw !== null) mem.mem.set(SESSION_KEY, raw)
      const h = harness({ session: mem }); assert.equal(await h.ctl.recover(), false, String(raw))
      assert.equal(h.gets.length + h.posts.length, 0, String(raw)); assert.equal(h.ctl.getState().phase, 'idle')
    }
  })

  it('works without any session store (storage blocked): no recovery, no crash', async () => {
    const h = harness(); assert.equal(await h.ctl.recover(), false)
    await h.ctl.start(BODY); assert.equal(h.ctl.getState().phase, 'completed')
  })

  it('a session store that throws never breaks submission, polling, reset or recover', async () => {
    const boom = { load: () => { throw new Error('blocked') }, save: () => { throw new Error('blocked') }, clear: () => { throw new Error('blocked') } }
    const h = harness({ session: { session: boom } as never, get: [running(), completed()] })
    assert.equal(await h.ctl.recover(), false)
    await h.ctl.start(BODY); assert.equal(h.ctl.getState().phase, 'completed')
    assert.doesNotThrow(() => h.ctl.reset())
  })

  it('recover() is ignored while a submission or run is in flight, so it cannot start a second poll', async () => {
    const mem = memorySession({ ...NOW_SESSION, runId: RUN_ID })
    const h = harness({ session: mem, get: [running(), completed()] })
    const p = h.ctl.start(BODY)
    assert.equal(await h.ctl.recover(), false); await p
    assert.equal(h.posts.length, 1)
  })

  it('reset() forgets the run: the saved session is cleared and a later recover() finds nothing', async () => {
    const mem = memorySession(); const h = harness({ session: mem })
    await h.ctl.start(BODY); assert.ok(mem.mem.has(SESSION_KEY))
    h.ctl.reset(); assert.equal(mem.mem.size, 0); assert.equal(await h.ctl.recover(), false)
  })

  it('leaving the page mid-POST still records the run id, so a reload can pick it up', async () => {
    const mem = memorySession(); let release!: (v: unknown) => void
    const h = harness({ session: mem, post: [() => new Promise((r) => { release = r })] })
    const p = h.ctl.start(BODY); await Promise.resolve()
    h.ctl.cancel(); release(ACCEPTED); await p
    assert.equal(JSON.parse(mem.mem.get(SESSION_KEY)!).runId, RUN_ID); assert.equal(h.gets.length, 0, 'but it stopped polling')
  })

  it('after cancel() (unmount) recover() can start again, as React StrictMode remounts require', async () => {
    const mem = memorySession({ ...NOW_SESSION, runId: RUN_ID })
    const h = harness({ session: mem, get: [completed()] })
    const first = h.ctl.recover(); h.ctl.cancel(); await first
    assert.equal(await h.ctl.recover(), true); assert.equal(h.ctl.getState().phase, 'completed')
  })
})
