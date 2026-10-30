/* Realistic GET /api/agent/runs/{id} bodies, shaped exactly like backend/agent/orchestrator.py's result. Plain data, no imports. */

export const RUN_ID = 'b6e93ea6a0ed'

export const prediction = () => ({
  label: 'Hit',
  probabilities: { Flop: 0.0213, Average: 0.3412, Hit: 0.6375 },
  inputs: { duration: 120, title_year: 2010, genres: 'Drama|Romance', country: 'USA', content_rating: 'PG-13', budget: 30000000 },
  imputed: [{ field: 'aspect_ratio', value: 2.35, rule: 'training median' }],
  warnings: [],
  model: { name: 'Random Forest', source: 'pre_release_movie_classifier.joblib', stage: 'pre-release' },
  caveats: ['Class scores are the model\'s estimated class probabilities. They have not been calibrated.',
    'The label is an IMDb-rating band (Flop below 3, Average 3 to 6, Hit 6 or above), not box office or profit.'],
})

const stage = (label: string, n: number, h: number, a: number, f: number, mean: number) => ({ label, n, counts: { Flop: f, Average: a, Hit: h }, mean_score: mean })

export const evidence = () => ({
  cohort: {
    stages: [stage('All films', 4919, 1000, 3500, 419, 6.4), stage('Drama + Romance', 125, 88, 36, 1, 6.31)],
    final: stage('Drama + Romance', 125, 88, 36, 1, 6.31), thin: false, empty: false,
    note: 'Descriptive counts from the dataset\'s films (1916–2016). Not a model output.',
  },
  sensitivity: { feature: 'duration', label: 'Runtime (min)', current_value: 120, current_scores: { Flop: 0.02, Average: 0.34, Hit: 0.64 },
    top_class_by_range: [{ from: 75, to: 157, label: 'Hit' }], note: 'Each point is a real run of the saved Random Forest with only this input changed.' },
  comparables: [
    { id: 1, title: 'The City of Your Final Destination', year: 2009, genres: 'Drama|Romance', country: 'USA', content_rating: 'R', duration: 117, Classify: 'Hit' },
    { id: 2, title: 'Another Film', year: 2004, genres: 'Drama', country: 'USA', content_rating: 'PG-13', duration: 110, Classify: 'Average' },
  ],
  research: { available: false, reason: 'research_not_enabled', sources: [] },
})

export const risk = (over: Record<string, unknown> = {}) => ({
  id: 'sensitivity_no_change', category: 'sensitivity', severity: 'info',
  text: 'Sweeping Runtime (min) from 75 to 157 did not change the top class. That holds for this tested range with every other input unchanged; it does not show that Runtime (min) is unimportant.',
  rule: 'one top-class segment across the sweep', evidence: { 'sensitivity.segments': 1, 'sensitivity.top_class': 'Hit' }, ...over,
})

export const analysis = (over: Record<string, unknown> = {}) => ({
  summary: 'The model ranks Hit highest at 64%.',
  drivers: [
    { claim: 'The saved Random Forest ranks Hit highest at 64%, then Average at 34% and Flop at 2%.', evidence_ref: 'prediction', fact_ids: ['F1'] },
    { claim: '125 films in the dataset match: 88 Hit, 36 Average, 1 Flop.', evidence_ref: 'cohort', fact_ids: ['F2'] },
  ],
  risks: [risk()],
  limitations: [...prediction().caveats, 'The AI analysis is written by a local language model from the evidence above.'],
  generated_by: { model: 'llama3.1:8b', fallback: false, partial: false, seconds: 24.1 },
  ...over,
})

export const guardrails = (over: Record<string, unknown> = {}) => ({
  schema_valid: true, numbers_verified: true, unsupported_claims: [], forbidden_terms: [], replaced: [], passed: true, llm_risks_ignored: false, ...over,
})

export const trace = () => [
  { step: 'prediction', status: 'ok', seconds: 0.06 }, { step: 'analyst.cohort', status: 'ok', seconds: 0.01 },
  { step: 'analyst.sensitivity', status: 'ok', seconds: 0.05 }, { step: 'analyst.comparables', status: 'ok', seconds: 0.01 },
  { step: 'research', status: 'ok', seconds: 0 }, { step: 'report.llm', status: 'ok', seconds: 24.05 }, { step: 'report.verify', status: 'ok' }, { step: 'risks', status: 'ok', seconds: 0 },
]

export const completed = (over: Record<string, unknown> = {}): Record<string, any> => ({
  run_id: RUN_ID, status: 'completed', created: 1761830000, inputs: prediction().inputs,
  prediction: prediction(), evidence: evidence(), analysis: analysis(), guardrails: guardrails(),
  disclaimers: prediction().caveats, trace: trace(), tool_calls: 4, timings: { total_seconds: 27.1 }, ...over,
})

export const running = () => ({ run_id: RUN_ID, status: 'running', created: 1761830000 })
export const failedRun = () => ({ run_id: RUN_ID, status: 'failed', trace: [{ step: 'prediction', status: 'failed', seconds: 0.01, detail: 'model_unavailable' }],
  error: { error: 'model_unavailable', message: 'The run could not produce a prediction.' } })

/** The same completed run, but every deterministic fallback signal the API can send. */
export const fallbackRun = (reason = 'ollama_unreachable') => completed({
  analysis: analysis({ generated_by: { model: null, fallback: true, reason } }), guardrails: guardrails(),
})
export const partialRun = () => completed({
  analysis: analysis({ generated_by: { model: 'llama3.1:8b', fallback: false, partial: true, seconds: 20.5 } }),
  guardrails: guardrails({ numbers_verified: false, unsupported_claims: ['driver 2: 104 is not a Hit figure in the cited facts'], replaced: ['driver 2'], passed: false }),
})

export const ACCEPTED = { run_id: RUN_ID, status: 'queued', poll_url: `/api/agent/runs/${RUN_ID}` }

/* ---- research block as returned by backend/agent/research.py ---- */
export const source = (kind: string, title: string, over: Record<string, unknown> = {}) => ({
  title, url: `https://en.wikipedia.org/w/index.php?title=${title.replaceAll(' ', '_')}&oldid=334798969`, host: 'en.wikipedia.org', kind,
  revision_id: 334798969, revision_timestamp: '2009-12-29T22:16:08Z', retrieved_at: '2026-10-29T19:54:02Z',
  excerpt: `${title} is described here as it was before release.`, temporal_status: 'version_before_cutoff', use: 'context_only', ...over,
})
export const research = (over: Record<string, unknown> = {}): Record<string, any> => ({
  provider: 'wikipedia', available: true, status: 'ok', reason: null, message: 'Retrieved pre-release background from Wikipedia.',
  cutoff: '2010-01-01T00:00:00Z', retrieved_at: '2026-10-29T19:54:01Z', requests_made: 7, max_requests: 8,
  query: { title: 'Inception', genres: ['Action', 'Sci-Fi'] }, skipped: [], sources: [source('film', 'Inception'), source('genre', 'Action film')],
  use: 'Contextual background only.', ...over,
})
export const failedResearch = (status: string, reason: string, message: string) => research({ available: false, status, reason, message, sources: [], requests_made: 1 })
export const withResearch = (r: unknown) => completed({ evidence: { ...evidence(), research: r } })
