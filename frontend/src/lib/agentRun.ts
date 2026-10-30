/* Agent run client logic: response normalising, provenance labels, and a bounded submit-and-poll controller.
   Deliberately free of runtime imports (no api.ts, no React) so it can be unit-tested with plain Node: see frontend/tests/agentRun.test.ts.
   Contract (backend/agent/router.py): POST /api/agent/runs -> 202 {run_id, status, poll_url};  GET /api/agent/runs/{id} -> {run_id, status, ...result}. */

import { RUN_ID_RE, type Session } from './runSession.ts'

export type Cls = 'Flop' | 'Average' | 'Hit'
const CLASSES: Cls[] = ['Flop', 'Average', 'Hit']

/* ------------------------------------------------------------------ response shapes (what the API actually sends) */
export interface TraceStep { step: string; status: string; seconds?: number; detail?: string }
export interface Prediction {
  label: Cls
  probabilities: Record<Cls, number>
  inputs: Record<string, string | number>
  imputed: { field: string; value: number | string; rule: string }[]
  warnings: { field: string; message: string }[]
  caveats: string[]
  model: { name: string; source?: string; stage?: string }
}
export interface CohortStage { label: string; n: number; counts: Record<Cls, number>; mean_score: number | null }
export interface Cohort { stages: CohortStage[]; final: CohortStage; thin: boolean; empty: boolean; note: string }
export interface SensitivityEvidence { label: string; current_value: number | null; current_scores: Record<Cls, number> | null; top_class_by_range: { from: number; to: number; label: Cls }[]; note: string }
export interface Comparable { title: string; year: number; genres: string; country: string; content_rating: string; duration: number | null; Classify: Cls }
export interface ResearchSource {
  title: string; url: string; kind: string; excerpt: string
  retrievedAt: string | null; revisionTimestamp: string | null; revisionId: number | null
}
/** status: ok | partial | no_results | unavailable | timeout | error | disabled. Older responses had only available/reason/sources. */
export interface Research {
  available: boolean; status: string; reason: string | null; message: string | null
  cutoff: string | null; retrievedAt: string | null; requestsMade: number | null; maxRequests: number | null
  query: { title: string | null; genres: string[] } | null
  skipped: { subject: string; reason: string }[]
  sources: ResearchSource[]; sourcesRejected: number
}
/** Links are only rendered for https Wikipedia URLs: defence in depth on top of the server's allow-list. */
export const isAllowedSourceUrl = (u: string) => {
  try { const x = new URL(u); return x.protocol === 'https:' && x.hostname === 'en.wikipedia.org' && !x.username && !x.password && (x.port === '' || x.port === '443') }
  catch { return false }
}
export interface Driver { claim: string; evidence_ref: string; fact_ids: string[] }
export interface RiskItem { id: string; category: string; severity: string; text: string; rule: string; evidence: Record<string, unknown> }
export interface GeneratedBy { model: string | null; fallback: boolean; partial: boolean; reason: string | null; seconds: number | null }
export interface Analysis { summary: string; drivers: Driver[]; risks: RiskItem[]; limitations: string[]; generatedBy: GeneratedBy; risksMalformed: number }
export interface Guardrails { schemaValid: boolean | null; numbersVerified: boolean | null; unsupportedClaims: string[]; forbiddenTerms: string[]; replaced: string[]; passed: boolean | null; llmRisksIgnored: boolean }

/** A part of the evidence the API may supply, or not: absent key, explicit null (the tool failed), or a usable value. */
export type Supplied<T> = { state: 'ok'; value: T } | { state: 'unavailable' } | { state: 'missing' }

export interface AgentRun {
  runId: string
  status: string
  prediction: Prediction | null
  cohort: Supplied<Cohort>
  sensitivity: Supplied<SensitivityEvidence>
  comparables: Supplied<Comparable[]>
  research: Supplied<Research>
  analysis: Analysis | null
  guardrails: Guardrails | null
  disclaimers: string[]
  trace: TraceStep[]
  toolCalls: number | null
  totalSeconds: number | null
  error: { error: string; message: string } | null
}

/* ------------------------------------------------------------------ defensive readers */
type Rec = Record<string, unknown>
const isRec = (v: unknown): v is Rec => typeof v === 'object' && v !== null && !Array.isArray(v)
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [])
const isCls = (v: unknown): v is Cls => typeof v === 'string' && (CLASSES as string[]).includes(v)

function supplied<T>(parent: Rec | null, key: string, read: (v: unknown) => T | null): Supplied<T> {
  if (!parent || !(key in parent)) return { state: 'missing' }
  const raw = parent[key]
  if (raw === null) return { state: 'unavailable' }     // an explicit null means the tool failed: not "nothing found"
  const value = read(raw)
  return value === null ? { state: 'unavailable' } : { state: 'ok', value }
}

function readScores(v: unknown): Record<Cls, number> | null {
  if (!isRec(v)) return null
  const out = {} as Record<Cls, number>
  for (const c of CLASSES) { const n = num(v[c]); if (n === null) return null; out[c] = n }
  return out
}

function readPrediction(v: unknown): Prediction | null {
  if (!isRec(v) || !isCls(v.label)) return null
  const probabilities = readScores(v.probabilities)
  if (!probabilities) return null
  const model = isRec(v.model) ? v.model : {}
  return {
    label: v.label, probabilities,
    inputs: isRec(v.inputs) ? (v.inputs as Prediction['inputs']) : {},
    imputed: Array.isArray(v.imputed) ? v.imputed.filter(isRec).map((i) => ({ field: String(i.field), value: i.value as number | string, rule: String(i.rule ?? '') })) : [],
    warnings: Array.isArray(v.warnings) ? v.warnings.filter(isRec).map((w) => ({ field: String(w.field ?? ''), message: String(w.message ?? '') })) : [],
    caveats: strs(v.caveats),
    model: { name: str(model.name) ?? 'Pre-release model', source: str(model.source) ?? undefined, stage: str(model.stage) ?? undefined },
  }
}

function readStage(v: unknown): CohortStage | null {
  if (!isRec(v)) return null
  const n = num(v.n), counts = readScores(v.counts)
  if (n === null || !counts) return null
  return { label: str(v.label) ?? '', n, counts, mean_score: num(v.mean_score) }
}

function readCohort(v: unknown): Cohort | null {
  if (!isRec(v)) return null
  const final = readStage(v.final)
  if (!final) return null
  const stages = Array.isArray(v.stages) ? v.stages.map(readStage).filter((s): s is CohortStage => s !== null) : [final]
  return { stages, final, thin: v.thin === true, empty: v.empty === true, note: str(v.note) ?? '' }
}

function readSensitivity(v: unknown): SensitivityEvidence | null {
  if (!isRec(v) || !Array.isArray(v.top_class_by_range)) return null
  const segs = v.top_class_by_range.filter(isRec).map((s) => ({ from: num(s.from), to: num(s.to), label: s.label })).filter((s): s is { from: number; to: number; label: Cls } => s.from !== null && s.to !== null && isCls(s.label))
  if (!segs.length || !str(v.label)) return null
  return { label: str(v.label)!, current_value: num(v.current_value), current_scores: readScores(v.current_scores), top_class_by_range: segs, note: str(v.note) ?? '' }
}

function readComparables(v: unknown): Comparable[] | null {
  if (!Array.isArray(v)) return null
  return v.filter(isRec).filter((c) => typeof c.title === 'string' && num(c.year) !== null && isCls(c.Classify)).map((c) => ({
    title: c.title as string, year: c.year as number, genres: str(c.genres) ?? '', country: str(c.country) ?? '', content_rating: str(c.content_rating) ?? '',
    duration: num(c.duration), Classify: c.Classify as Cls,
  }))
}

function readResearch(v: unknown): Research | null {
  if (!isRec(v) || typeof v.available !== 'boolean') return null
  const raw = Array.isArray(v.sources) ? v.sources.filter(isRec) : []
  const sources: ResearchSource[] = raw.filter((s) => typeof s.url === 'string' && isAllowedSourceUrl(s.url as string)).map((s) => ({
    title: str(s.title) ?? String(s.url), url: String(s.url), kind: str(s.kind) ?? 'page', excerpt: str(s.excerpt) ?? '',
    retrievedAt: str(s.retrieved_at), revisionTimestamp: str(s.revision_timestamp), revisionId: num(s.revision_id),
  }))
  const reason = str(v.reason)
  // "available" is only believed when sources survive our own checks; a count is never taken from a flag
  const status = str(v.status) ?? (sources.length ? 'ok' : reason === 'research_not_enabled' ? 'disabled' : 'unavailable')
  const q = isRec(v.query) ? v.query : null
  return {
    available: v.available && sources.length > 0, status, reason, message: str(v.message), cutoff: str(v.cutoff), retrievedAt: str(v.retrieved_at),
    requestsMade: num(v.requests_made), maxRequests: num(v.max_requests),
    query: q ? { title: str(q.title), genres: strs(q.genres) } : null,
    skipped: Array.isArray(v.skipped) ? v.skipped.filter(isRec).map((x) => ({ subject: str(x.subject) ?? '', reason: str(x.reason) ?? '' })) : [],
    sources, sourcesRejected: raw.length - sources.length,
  }
}

/** Structured risk items only. Anything else (e.g. the old string array) is counted as malformed and not shown as a risk. */
function readRisks(v: unknown): { items: RiskItem[]; malformed: number } {
  if (!Array.isArray(v)) return { items: [], malformed: 0 }
  const items: RiskItem[] = []
  let malformed = 0
  for (const r of v) {
    if (isRec(r) && str(r.id) && str(r.category) && str(r.severity) && str(r.text)) {
      items.push({ id: r.id as string, category: r.category as string, severity: r.severity as string, text: r.text as string, rule: str(r.rule) ?? '', evidence: isRec(r.evidence) ? r.evidence : {} })
    } else malformed++
  }
  return { items, malformed }
}

function readAnalysis(v: unknown): Analysis | null {
  if (!isRec(v) || typeof v.summary !== 'string') return null
  const gb = isRec(v.generated_by) ? v.generated_by : {}
  const { items, malformed } = readRisks(v.risks)
  return {
    summary: v.summary,
    drivers: Array.isArray(v.drivers) ? v.drivers.filter(isRec).filter((d) => typeof d.claim === 'string').map((d) => ({ claim: d.claim as string, evidence_ref: str(d.evidence_ref) ?? '', fact_ids: strs(d.fact_ids) })) : [],
    risks: items, risksMalformed: malformed, limitations: strs(v.limitations),
    // fallback defaults to true: if the API does not say the model wrote it, we must not claim that it did
    generatedBy: { model: str(gb.model), fallback: gb.fallback === false ? false : true, partial: gb.partial === true, reason: str(gb.reason), seconds: num(gb.seconds) },
  }
}

function readGuardrails(v: unknown): Guardrails | null {
  if (!isRec(v)) return null
  const b = (x: unknown) => (typeof x === 'boolean' ? x : null)
  return { schemaValid: b(v.schema_valid), numbersVerified: b(v.numbers_verified), unsupportedClaims: strs(v.unsupported_claims), forbiddenTerms: strs(v.forbidden_terms),
    replaced: strs(v.replaced), passed: b(v.passed), llmRisksIgnored: v.llm_risks_ignored === true }
}

/** Turn a GET /api/agent/runs/{id} body into a typed run, or null if it is not recognisably a run. */
export function normalizeRun(raw: unknown): AgentRun | null {
  if (!isRec(raw) || !str(raw.run_id) || !str(raw.status)) return null
  const evidence = isRec(raw.evidence) ? raw.evidence : null
  const timings = isRec(raw.timings) ? raw.timings : {}
  return {
    runId: raw.run_id as string, status: raw.status as string,
    prediction: readPrediction(raw.prediction),
    cohort: supplied(evidence, 'cohort', readCohort),
    sensitivity: supplied(evidence, 'sensitivity', readSensitivity),
    comparables: supplied(evidence, 'comparables', readComparables),
    research: supplied(evidence, 'research', readResearch),
    analysis: readAnalysis(raw.analysis), guardrails: readGuardrails(raw.guardrails),
    disclaimers: strs(raw.disclaimers),
    trace: Array.isArray(raw.trace) ? raw.trace.filter(isRec).filter((t) => typeof t.step === 'string').map((t) => ({ step: t.step as string, status: str(t.status) ?? '', seconds: num(t.seconds) ?? undefined, detail: str(t.detail) ?? undefined })) : [],
    toolCalls: num(raw.tool_calls), totalSeconds: num(timings.total_seconds),
    error: isRec(raw.error) && str(raw.error.message) ? { error: str(raw.error.error) ?? 'run_failed', message: raw.error.message as string } : null,
  }
}

/* ------------------------------------------------------------------ provenance: who wrote the analysis, stated from the API's own fields */
const REASONS: Record<string, string> = {
  ollama_unreachable: 'the local language model was not reachable', model_not_installed: 'the language model is not installed', unreachable: 'the local language model was not reachable',
  timeout: 'the language model timed out', http_error: 'the language model returned an error', invalid_json: 'the model returned malformed JSON', truncated: 'the model output was cut off',
  invalid_schema: 'the model output did not match the required structure', no_matching_film_page: 'no Wikipedia page matched that exact film title',
  no_revision_before_cutoff: 'the page did not exist before the cutoff', redirect_page_at_cutoff: 'the page was only a redirect before the cutoff',
  revision_not_before_cutoff: 'the page version was not dated before the cutoff', no_reference_article_for_genre: 'there is no reference article for this genre',
  source_limit: 'the source limit was reached', request_limit: 'the request limit was reached', time_budget: 'the time budget ran out', connection_failed: 'the connection failed',
  rate_limited: 'the source asked us to slow down', response_too_large: 'the response was too large', redirect_limit: 'a redirect was not allowed', url_not_allowed: 'a URL was not on the allow-list',
  malformed_response: 'the response was not in the expected format', empty_excerpt: 'the page had no usable text', no_pre_cutoff_pages: 'no usable page versions from before the cutoff', internal_error: 'an internal error', failed_verification: 'every model claim failed verification', not_specific: 'the model output quoted no figures from the evidence',
  verification_error: 'the checker hit an internal error', analysis_error: 'the analysis step hit an internal error', run_deadline: 'the run was out of time',
}
export const reasonText = (r: string | null) => (r ? REASONS[r] ?? r.replaceAll('_', ' ') : null)

export interface Provenance { kind: 'llm' | 'llm_partial' | 'template'; headline: string; detail: string }
export function provenance(a: Analysis, g: Guardrails | null): Provenance {
  const by = a.generatedBy
  if (by.fallback) {
    return { kind: 'template', headline: 'Template analysis (not written by a language model)',
      detail: `Built by fixed rules from the evidence${by.reason ? `, because ${reasonText(by.reason)}` : ''}. The prediction above is unaffected.` }
  }
  const who = by.model ?? 'the local language model'
  if (by.partial || (g?.replaced.length ?? 0) > 0) {
    return { kind: 'llm_partial', headline: `Written by ${who}, with parts replaced`,
      detail: `Some model text failed verification and was swapped for safe template wording: ${(g?.replaced ?? []).join(', ') || 'see guardrails'}.` }
  }
  return { kind: 'llm', headline: `Written by ${who}`, detail: 'Checked against the evidence, but pattern checks cannot prove every sentence correct.' }
}

/* ------------------------------------------------------------------ bounded submit-and-poll controller */
export interface PollConfig { intervalMs: number; slowIntervalMs: number; slowAfterMs: number; maxWaitMs: number; maxConsecutiveErrors: number }
export const DEFAULT_POLL: PollConfig = { intervalMs: 1000, slowIntervalMs: 2500, slowAfterMs: 20_000, maxWaitMs: 300_000, maxConsecutiveErrors: 4 }

export type Phase = 'idle' | 'submitting' | 'running' | 'completed' | 'failed' | 'timeout' | 'error'
export interface RunProblem { code: string; message: string; fields: { field: string; message: string }[] }
export interface RunState {
  phase: Phase
  runId: string | null
  apiStatus: string | null
  run: AgentRun | null
  problem: RunProblem | null
  polls: number
  elapsedMs: number
  /** true when this state came from reconnecting after a refresh rather than from a submission made in this page session */
  recovered: boolean
}
export const IDLE: RunState = { phase: 'idle', runId: null, apiStatus: null, run: null, problem: null, polls: 0, elapsedMs: 0, recovered: false }

export interface SessionStore { load(): Session | null; save(s: Session): void; clear(): void }
export const IDEMPOTENCY_HEADER = 'Idempotency-Key'

export interface Deps {
  post: (path: string, body: unknown, headers?: Record<string, string>) => Promise<unknown>
  get: (path: string) => Promise<unknown>
  now: () => number
  sleep: (ms: number) => Promise<void>
  onChange: (s: RunState) => void
  /** fresh random submission key (32 hex chars) */
  newKey: () => string
  /** optional tab-scoped memory of {key, runId}; without it nothing is persisted and recover() has nothing to find */
  session?: SessionStore
  config?: Partial<PollConfig>
}

const problemOf = (e: unknown, fallback = 'Something went wrong.'): RunProblem & { status: number } => {
  const x = (isRec(e) || e instanceof Error ? e : {}) as { status?: unknown; code?: unknown; message?: unknown; fields?: unknown }
  return {
    status: typeof x.status === 'number' ? x.status : 0, code: typeof x.code === 'string' ? x.code : 'unknown',
    message: typeof x.message === 'string' && x.message ? x.message : fallback,
    fields: Array.isArray(x.fields) ? (x.fields as { field: string; message: string }[]) : [],
  }
}

export function createRunController(deps: Deps) {
  const cfg = { ...DEFAULT_POLL, ...deps.config }
  let state: RunState = IDLE
  let generation = 0                    // bumped by cancel/start/recover so a superseded loop can never write state
  let last: { body: unknown; key: string } | null = null
  const set = (patch: Partial<RunState>) => { state = { ...state, ...patch }; deps.onChange(state) }
  // the session store is a convenience: if it throws, submission and polling carry on without recovery
  const guard = <T,>(fn: () => T): T | null => { try { return fn() } catch { return null } }
  const remember = (key: string, runId: string | null) => { guard(() => deps.session?.save({ key, runId, savedAt: deps.now() })) }
  const forget = () => { guard(() => deps.session?.clear()) }

  async function poll(runId: string, gen: number) {
    const began = deps.now()
    let errors = 0
    set({ phase: 'running', runId, problem: null, elapsedMs: 0 })
    while (gen === generation) {
      const elapsed = deps.now() - began
      if (elapsed >= cfg.maxWaitMs) {
        set({ phase: 'timeout', elapsedMs: elapsed, problem: { code: 'poll_timeout', message: 'The run is taking longer than expected. It may still finish on the server.', fields: [] } })
        return
      }
      try {
        const raw = await deps.get(`/api/agent/runs/${runId}`)
        if (gen !== generation) return
        errors = 0
        const run = normalizeRun(raw)
        if (!run) { set({ phase: 'error', problem: { code: 'bad_response', message: 'The server sent a response this page does not understand.', fields: [] } }); return }
        const now = deps.now() - began
        if (run.status === 'completed') {
          if (!run.prediction) { set({ phase: 'error', run, apiStatus: run.status, elapsedMs: now, problem: { code: 'bad_response', message: 'The run finished without a prediction.', fields: [] } }); return }
          set({ phase: 'completed', run, apiStatus: run.status, elapsedMs: now, polls: state.polls + 1 }); return
        }
        if (run.status === 'failed') { set({ phase: 'failed', run, apiStatus: run.status, elapsedMs: now, polls: state.polls + 1 }); return }
        set({ apiStatus: run.status, elapsedMs: now, polls: state.polls + 1 })
      } catch (e) {
        if (gen !== generation) return
        const p = problemOf(e)
        if (p.status === 404) { forget(); set({ phase: 'error', problem: { code: 'run_not_found', message: 'The server no longer has this run.', fields: [] } }); return }
        if (p.status >= 400 && p.status < 500) { set({ phase: 'error', problem: { code: p.code, message: p.message, fields: p.fields } }); return }
        if (++errors >= cfg.maxConsecutiveErrors) { set({ phase: 'error', problem: { code: p.code, message: `Lost contact with the server while waiting. ${p.message}`, fields: [] } }); return }
      }
      await deps.sleep(deps.now() - began >= cfg.slowAfterMs ? cfg.slowIntervalMs : cfg.intervalMs)
    }
  }

  const validId = (id: unknown): id is string => typeof id === 'string' && RUN_ID_RE.test(id)

  async function start(body: unknown, key?: string): Promise<boolean> {
    if (state.phase === 'submitting' || state.phase === 'running') return false
    const k = key ?? deps.newKey()
    const gen = ++generation
    last = { body, key: k }
    state = { ...IDLE, phase: 'submitting' }
    deps.onChange(state)
    remember(k, null)                    // before the request leaves: if the page dies mid-POST, the key lets us ask the server what happened
    let created: unknown
    try {
      created = await deps.post('/api/agent/runs', body, { [IDEMPOTENCY_HEADER]: k })
    } catch (e) {
      if (gen !== generation) return true
      const p = problemOf(e)
      set({ phase: 'error', problem: { code: p.code, message: p.message, fields: p.fields } })
      return true
    }
    const id = isRec(created) ? str(created.run_id) : null
    if (validId(id)) remember(k, id)     // kept even if this page was left meanwhile, so a reload can pick the run up
    if (gen !== generation) return true
    if (!validId(id)) { set({ phase: 'error', problem: { code: 'bad_response', message: 'The server did not return a run id.', fields: [] } }); return true }
    set({ runId: id, apiStatus: str((created as Rec).status) })
    await poll(id, gen)
    return true
  }

  return {
    getState: () => state,
    /** Start a NEW run with a fresh submission key. Ignored (returns false) while one is already submitting or running, so a double click
     *  cannot create two. Pass `key` only to repeat the SAME submission; the server then returns the run it already created. */
    start,
    /** After a failed submission (no run id): send the SAME inputs with the SAME key. If the first request had in fact reached the server,
     *  the server answers with that run instead of creating a second one. */
    async retry(): Promise<boolean> {
      if (state.phase !== 'error' || state.runId !== null || !last) return false
      return start(last.body, last.key)
    },
    /** Keep waiting on the SAME run after a timeout or lost connection. Never creates a new run. */
    async resume(): Promise<boolean> {
      const id = state.runId
      if (!id || (state.phase !== 'timeout' && state.phase !== 'error')) return false
      const gen = ++generation
      await poll(id, gen)
      return true
    },
    /** After a page refresh: reconnect to the run this tab started. Never POSTs. Returns false when there is nothing to recover. */
    async recover(): Promise<boolean> {
      const retryable = state.phase === 'error' && state.runId === null && !last
      if (state.phase !== 'idle' && !retryable) return false
      const s = guard(() => deps.session?.load() ?? null)
      if (!s) return false
      const gen = ++generation
      state = { ...IDLE, phase: 'running', recovered: true, runId: s.runId }
      deps.onChange(state)
      let id = s.runId
      if (!id) {
        try {
          const found = await deps.get(`/api/agent/runs/by-key/${s.key}`)
          id = isRec(found) ? str(found.run_id) : null
          if (gen !== generation) return true
          if (!validId(id)) throw Object.assign(new Error('The server did not return a run id.'), { status: 0, code: 'bad_response' })
        } catch (e) {
          if (gen !== generation) return true
          const p = problemOf(e)
          if (p.status === 404) {         // the server never saw that submission: nothing was created
            forget()
            set({ phase: 'error', problem: { code: 'submission_not_received', message: 'Your last submission did not reach the server, so nothing was started. You can submit again.', fields: [] } })
          } else {
            set({ phase: 'error', problem: { code: 'recover_failed', message: `Could not reconnect to your earlier run. ${p.message}`, fields: [] } })
          }
          return true
        }
        remember(s.key, id)
      }
      await poll(id!, gen)
      return true
    },
    /** Stop polling (e.g. on unmount). The run itself continues on the server and stays recoverable through the saved session. */
    cancel() { generation++; state = IDLE },
    /** The user chose to forget this run: stop, clear the saved session and go back to the empty state. */
    reset() { generation++; last = null; forget(); state = IDLE; deps.onChange(state) },
  }
}
