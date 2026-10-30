import { Bot, RotateCcw, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FilmForm } from '../components/researcher/FilmForm'
import { FailedRun, ResultView } from '../components/researcher/RunView'
import { Button, EmptyState, ErrorState, LoadingBlock, Note, PageHeader, Panel } from '../components/ui'
import { fromRun } from '../lib/analyses'
import { DEFAULT_POLL, type RunState } from '../lib/agentRun'
import { buildBody, emptyValues, validateForm, type FormValues } from '../lib/filmForm'
import type { PredictSchema } from '../lib/types'
import { useAgentRun } from '../lib/useAgentRun'
import { analysisRegistry } from '../lib/useAnalyses'
import { useApi } from '../lib/useApi'

interface AgentStatus { enabled: boolean; llm: { available: boolean; model: string; reason?: string }; max_tool_calls: number; research?: { enabled: boolean; provider: string } }

export function RunProgress({ s }: { s: RunState }) {
  const submitting = s.phase === 'submitting'
  return (
    <Panel title={submitting ? 'Sending the request' : 'The agent is running'} tone="#25e8d3" subtitle="Status below is exactly what the API reports. The step-by-step trace is only available once the run finishes.">
      <div role="status" aria-live="polite" data-testid="run-progress" className="space-y-3">
        <p className="flex items-center gap-3 text-[0.95rem] text-fg">
          <span aria-hidden className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-cyan border-t-transparent motion-reduce:animate-none" />
          {submitting ? (s.recovered ? 'Checking whether the server received your last submission…' : 'Waiting for the server to accept the run…') : `API status: ${s.apiStatus ?? 'queued'}`}
        </p>
        {s.recovered && <p className="text-[0.8rem] text-fg2" data-testid="recovering">Reconnecting to the run this tab started before the page was refreshed. No new run is being created.</p>}
        {!submitting && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-[0.78rem]">
            <dt className="text-fg3">Run id</dt><dd className="text-fg2">{s.runId}</dd>
            <dt className="text-fg3">Waiting</dt><dd className="text-fg2">{Math.floor(s.elapsedMs / 1000)}s of at most {DEFAULT_POLL.maxWaitMs / 60000} min</dd>
            <dt className="text-fg3">Status checks</dt><dd className="text-fg2">{s.polls}</dd>
          </dl>
        )}
        <p className="text-[0.78rem] text-fg3">A run usually takes 20 to 60 seconds, mostly the local language model writing the analysis. The prediction itself takes under a second.</p>
      </div>
    </Panel>
  )
}

export function Problem({ s, onResume, onRetry, onRecover, onNew }: { s: RunState; onResume: () => void; onRetry: () => void; onRecover: () => void; onNew: () => void }) {
  const p = s.problem!
  const timeout = s.phase === 'timeout'
  if (s.runId) {
    // A run exists on the server: offer to keep checking it. Starting another run is a separate, explicit choice.
    return (
      <div role="alert" className="space-y-4 rounded-[1.25rem] border border-avg/40 bg-avg/10 p-5" data-testid="run-problem" data-code={p.code}>
        <h3 className="font-display text-base font-bold text-fg">{timeout ? 'Still waiting on the run' : 'Could not check the run'}</h3>
        <p className="max-w-prose text-sm text-fg2">{p.message}</p>
        <p className="font-mono text-[0.75rem] text-fg3">Run id {s.runId}</p>
        <div className="flex flex-wrap gap-2">
          {p.code !== 'run_not_found' && <Button variant="primary" onClick={onResume}>{timeout ? 'Keep waiting' : 'Check again'}</Button>}
          <Button variant="ghost" onClick={onNew}>Start a new run</Button>
        </div>
        <p className="text-[0.75rem] text-fg3">Checking again re-reads this same run and never creates a second one. “Start a new run” queues another analysis.</p>
      </div>
    )
  }
  if (p.code === 'submission_not_received') {
    return (
      <div role="alert" className="space-y-3 rounded-[1.25rem] border border-avg/40 bg-avg/10 p-5" data-testid="run-problem" data-code={p.code}>
        <h3 className="font-display text-base font-bold text-fg">Nothing was started</h3>
        <p className="max-w-prose text-sm text-fg2">{p.message}</p>
      </div>
    )
  }
  return (
    <div data-testid="run-problem" data-code={p.code}>
      <ErrorState title={p.code === 'recover_failed' ? 'Could not reconnect' : p.code === 'busy' ? 'The researcher is busy' : p.code === 'validation_error' ? 'Please check the inputs' : p.code === 'model_unavailable' ? 'Model unavailable' : 'The run could not be started'}
        message={p.message} onRetry={p.code === 'recover_failed' ? onRecover : onRetry} />
      <p className="mt-2 text-[0.75rem] text-fg3">{p.code === 'recover_failed' ? 'Checking again only looks up your earlier submission; it never starts a new run.'
        : 'Trying again sends the same request with the same key, so if the server already received it you get that run back, not a duplicate.'}</p>
    </div>
  )
}

export default function Researcher() {
  const schema = useApi<PredictSchema>('/api/predict/schema', { cache: true })
  const agent = useApi<AgentStatus>('/api/agent/status')
  const { state, start, retry: retrySubmit, resume, recover, reset, peek } = useAgentRun()
  const navigate = useNavigate()

  const [values, setValues] = useState<FormValues>(emptyValues)
  const [genres, setGenres] = useState<string[]>([])
  const [errors, setErrors] = useState<Record<string, string>>({})
  const resultRef = useRef<HTMLDivElement>(null)

  const busy = state.phase === 'submitting' || state.phase === 'running'
  const agentOff = agent.error?.status === 404
  const llm = agent.data?.llm

  // after a refresh: reconnect to the run this tab started (reads only the saved key and run id; never submits anything)
  useEffect(() => { void recover() }, [recover])

  // a finished analysis becomes available to the AI Assistant (in memory for this tab only)
  useEffect(() => { if (state.phase === 'completed') analysisRegistry().add(fromRun(state.run)) }, [state.phase, state.run])

  // move keyboard and screen-reader focus to the outcome when a run reaches a resting state
  useEffect(() => {
    if (!['completed', 'failed', 'timeout', 'error'].includes(state.phase)) return
    resultRef.current?.focus({ preventScroll: true })
    resultRef.current?.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })
  }, [state.phase])

  const setValue = (k: string, v: string) => { setValues((x) => ({ ...x, [k]: v })); setErrors((e) => { if (!e[k]) return e; const n = { ...e }; delete n[k]; return n }) }

  function submit(ev: React.FormEvent) {
    ev.preventDefault()
    if (busy || !schema.data) return
    const e = validateForm(schema.data, values, genres)
    setErrors(e)
    if (Object.keys(e).length) { document.getElementById(e.genres && Object.keys(e)[0] === 'genres' ? 'r-genres' : `r-${Object.keys(e)[0]}`)?.focus(); return }
    run(buildBody(schema.data, values, genres))
  }
  // server-side validation (422) comes back per field: show it against the matching inputs once the run settles
  const showFieldErrors = () => {
    const fields = peek().problem?.fields ?? []
    if (fields.length) setErrors(Object.fromEntries(fields.map((f) => [f.field, f.message])))
  }
  function run(body: Record<string, unknown>) { void start(body).then(showFieldErrors) }
  const retry = () => { void retrySubmit().then(showFieldErrors) }
  function clear() { reset(); setValues(emptyValues()); setGenres([]); setErrors({}) }

  return (
    <>
      <PageHeader eyebrow="AI Movie Researcher" title="AI Movie Researcher" highlight="Researcher"
        actions={<Button variant="ghost" icon={<RotateCcw size={15} aria-hidden />} onClick={clear} disabled={busy}>Reset</Button>}>
        Describe a film using only what is known <b className="font-semibold text-fg">before release</b>. The saved Random Forest makes the prediction; local agents then gather dataset evidence and a language model explains it, with every claim checked and every risk produced by fixed rules.
      </PageHeader>

      <div className="mb-5 grid gap-4 lg:grid-cols-2">
        <Note tone="info" title="Two separate things on this page">
          <p><b className="font-semibold text-fg">The prediction</b> comes only from the saved model. <b className="font-semibold text-fg">The analysis</b> is written afterwards and is labelled with who wrote it. Hit, Average and Flop are IMDb-score classes, not financial success, and the class scores are uncalibrated. Web research, when shown, is background only and never feeds the model.</p>
        </Note>
        <aside className="glass flex items-start gap-3 rounded-[1.25rem] p-5" aria-label="Language model status" data-testid="agent-status">
          <Bot size={20} aria-hidden className={llm?.available ? 'mt-0.5 text-hit' : 'mt-0.5 text-fg3'} />
          <div className="text-[0.85rem] leading-snug text-fg2">
            {agent.loading && !agent.data ? 'Checking the local language model…'
              : agentOff ? <><b className="font-semibold text-fg">The AI researcher is not enabled on this server.</b> The rest of the app is unaffected.</>
              : agent.error ? <><b className="font-semibold text-fg">Could not reach the agent service.</b> <button type="button" className="text-accent underline underline-offset-4" onClick={agent.retry}>Try again</button></>
              : llm?.available ? <><b className="font-semibold text-fg">Local language model ready</b> ({llm.model}). {agent.data?.research?.enabled ? 'Web research reads Wikipedia only, as page versions from before release.' : 'Web research is not enabled on this server.'}</>
              : <><b className="font-semibold text-fg">Local language model not available.</b> Runs still finish, with a template analysis instead of model-written text.</>}
          </div>
        </aside>
      </div>

      {schema.error && <ErrorState title="The form is unavailable" message={schema.error.message} onRetry={schema.retry} />}
      {schema.loading && !schema.data && <LoadingBlock height="h-96" label="Loading model schema" />}

      {schema.data && (
        <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <form onSubmit={submit} noValidate className="space-y-4" aria-label="Film attributes">
            <FilmForm schema={schema.data} values={values} genres={genres} errors={errors} disabled={busy}
              onValue={setValue} onGenres={(g) => { setGenres(g); setErrors((e) => { const n = { ...e }; delete n.genres; return n }) }} />
            <div className="glass sticky bottom-3 z-10 flex flex-wrap items-center gap-3 rounded-full p-2 pr-4">
              <Button type="submit" variant="primary" loading={busy} disabled={agentOff} icon={!busy && <Sparkles size={16} aria-hidden />} className="flex-1 !min-h-12 text-base sm:flex-none sm:min-w-60">
                {busy ? 'Researching…' : state.phase === 'completed' ? 'Run again' : 'Research this film'}
              </Button>
              {Object.keys(errors).length > 0 && <p role="alert" className="text-sm text-danger">Fix {Object.keys(errors).length} field{Object.keys(errors).length > 1 ? 's' : ''} to continue.</p>}
            </div>
          </form>

          <div ref={resultRef} tabIndex={-1} aria-label="Research result" className="space-y-5 outline-none" data-phase={state.phase}>
            {state.phase === 'idle' && <EmptyState title="No run yet" message="Fill in the film and choose Research this film. The deterministic prediction appears first, then the written analysis, evidence and risks." />}
            {busy && <RunProgress s={state} />}
            {state.recovered && ['completed', 'failed'].includes(state.phase) && (
              <Note tone="info" title="Restored after a page refresh">
                <p>This is the run your tab started earlier. The form was not restored; the inputs the model received are listed under the prediction.</p>
              </Note>
            )}
            {state.phase === 'completed' && state.run && <ResultView run={state.run} onDiscuss={() => navigate('/assistant', { state: { runId: state.run!.runId } })} />}
            {state.phase === 'failed' && <FailedRun run={state.run} onNew={() => { reset() }} />}
            {(state.phase === 'timeout' || state.phase === 'error') && state.problem && (
              <Problem s={state} onResume={() => void resume()} onRetry={retry} onRecover={() => void recover()} onNew={() => { reset() }} />
            )}
          </div>
        </div>
      )}
    </>
  )
}
