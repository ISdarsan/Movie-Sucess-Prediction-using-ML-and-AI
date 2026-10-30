/* Analyses finished in THIS tab, kept in memory so the AI Assistant can offer them for a movie conversation. Pure TypeScript (type imports only).

   What is kept per analysis: the run id and a few display words (title if one was given, predicted class, year, genres). No scores, no
   evidence, no research text: the server re-reads everything from the stored run when a question is asked. Nothing is persisted: no web
   storage, no cookies; a refresh clears the list. Runs cannot be listed by the server, so this registry (plus a pasted run id) is the only way
   a person reaches an analysis, which is intentional.
   Single-user, local assumption: a run id is the capability to read that run; there are no user accounts. */
import type { AgentRun } from './agentRun'
import type { MovieRef } from './chat'

export const MAX_ANALYSES = 10
export const RUN_ID_RE = /^(?:[0-9a-f]{12}|[0-9a-f]{32})$/

export interface Analysis extends MovieRef { at: number }

/** Display reference for a COMPLETED run that has a prediction; null for anything else. */
export function fromRun(run: AgentRun | null | undefined): MovieRef | null {
  if (!run || run.status !== 'completed' || !run.prediction || !RUN_ID_RE.test(run.runId)) return null
  const title = run.research.state === 'ok' ? run.research.value.query?.title ?? null : null
  const inputs = run.prediction.inputs
  const genres = typeof inputs.genres === 'string' ? inputs.genres.replaceAll('|', ' / ') : null
  const year = typeof inputs.title_year === 'number' ? inputs.title_year : null
  return { runId: run.runId, title: title && title.trim() ? title.trim().slice(0, 120) : null, predictedClass: run.prediction.label, year, genres }
}

/** The server's small summary of an analysis (GET /api/chat/movie/{id}) as a display reference; null if it is not recognisable. */
export function parseSummary(raw: unknown): MovieRef | null {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null
  const r = raw as Record<string, unknown>
  if (typeof r.run_id !== 'string' || !RUN_ID_RE.test(r.run_id) || !['Flop', 'Average', 'Hit'].includes(String(r.predicted_class))) return null
  const s = (v: unknown, n: number) => (typeof v === 'string' && v.trim() ? v.trim().slice(0, n) : null)
  return { runId: r.run_id, title: s(r.title, 120), predictedClass: String(r.predicted_class), year: typeof r.year === 'number' ? r.year : null, genres: s(r.genres, 80) }
}

export function createAnalysisRegistry(now: () => number = () => Date.now()) {
  let list: Analysis[] = []
  const listeners = new Set<() => void>()
  return {
    getState: () => list,
    subscribe(fn: () => void) { listeners.add(fn); return () => { listeners.delete(fn) } },
    /** Newest first; the same run is only ever listed once; the list is capped. */
    add(ref: MovieRef | null) {
      if (!ref || !RUN_ID_RE.test(ref.runId)) return
      list = [{ ...ref, at: now() }, ...list.filter((a) => a.runId !== ref.runId)].slice(0, MAX_ANALYSES)
      listeners.forEach((l) => l())
    },
    get: (runId: string) => list.find((a) => a.runId === runId) ?? null,
  }
}
export type AnalysisRegistry = ReturnType<typeof createAnalysisRegistry>
