import { AlertTriangle, Bot, Layers, ListChecks, MessagesSquare, Telescope } from 'lucide-react'
import { motion } from 'motion/react'
import { CLASS_COLOR, CLASSES, CLASS_RANGE, fmtInt, fmtPct, type MovieClass } from '../../lib/format'
import { provenance, reasonText, type AgentRun, type Analysis, type Cohort, type Comparable, type Guardrails, type Prediction, type Research, type ResearchSource, type RiskItem, type SensitivityEvidence, type Supplied } from '../../lib/agentRun'
import { ClassBadge, Disclosure, Note, Panel, cx } from '../ui'

const SHOWN = [...CLASSES].reverse()

/** Shown where the API sent an explicit null: the evidence tool failed, which is different from "none found". */
function Unavailable({ what }: { what: string }) {
  return (
    <p role="status" className="rounded-2xl border border-avg/40 bg-avg/10 px-4 py-3 text-[0.84rem] text-fg2">
      <b className="font-semibold text-fg">{what} unavailable.</b> The step that gathers it failed for this run. That is not the same as finding none.
    </p>
  )
}

/* ---------------------------------------------------------------- 1. deterministic prediction */
export function PredictionPanel({ p }: { p: Prediction }) {
  const c = CLASS_COLOR[p.label]
  return (
    <Panel id="agent-prediction" title="Random Forest prediction" tone={c} marks
      subtitle="Deterministic: produced by the saved pre-release model. A language model never produces or changes these numbers.">
      <p className="eyebrow text-center">The model says</p>
      <p className="mt-2 text-center font-display text-[4.2rem] font-extrabold leading-none sm:text-[5.2rem]" style={{ color: c, textShadow: `0 0 40px ${c}99` }} data-testid="predicted-label">{p.label}</p>
      <p className="mx-auto mt-3 max-w-md text-center text-[0.84rem] text-fg2">
        {p.label} means an IMDb score band of <span className="font-mono text-fg">{CLASS_RANGE[p.label]}</span>. It is a rating category, <b className="font-semibold text-fg">not box office or profit</b>.
      </p>

      <h3 className="mt-7 font-display text-[0.95rem] font-bold">Class scores</h3>
      <ul className="mt-3 space-y-3" data-testid="class-scores">
        {SHOWN.map((k) => (
          <li key={k} className="grid grid-cols-[4.5rem_1fr_3.4rem] items-center gap-3 text-sm">
            <span className="font-medium" style={{ color: CLASS_COLOR[k] }}>{k}</span>
            <span className="h-3 overflow-hidden rounded-full bg-white/[0.08]" role="img" aria-label={`${k} ${fmtPct(p.probabilities[k], 0)}`}>
              <motion.span className="block h-full rounded-full" style={{ background: CLASS_COLOR[k] }} initial={{ width: 0 }} animate={{ width: `${Math.max(p.probabilities[k] * 100, 0.8)}%` }} transition={{ duration: 0.8 }} />
            </span>
            <span className="text-right font-mono text-[0.85rem] text-fg">{fmtPct(p.probabilities[k], 0)}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[0.75rem] leading-snug text-fg3">These scores are uncalibrated: read them as a ranking of the three classes, not as the real-world chance of an outcome.</p>

      {p.warnings.length > 0 && <ul className="mt-4 space-y-2">{p.warnings.map((w) => <li key={w.message} className="rounded-2xl border border-avg/40 bg-avg/10 px-3.5 py-2 text-[0.8rem] text-fg2">{w.message}</li>)}</ul>}

      <Disclosure title="Inputs the model received">
        <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-[0.8rem]">
          {Object.entries(p.inputs).map(([k, v]) => <div key={k} className="contents"><dt className="font-mono text-fg3">{k}</dt><dd className="tnum text-right font-mono text-fg">{typeof v === 'number' ? fmtInt(v) : String(v).replaceAll('|', ' / ')}</dd></div>)}
        </dl>
        {p.imputed.length > 0 && (
          <div className="mt-4">
            <p className="mb-1.5 text-[0.8rem] font-semibold text-avg">Filled in by the pipeline ({p.imputed.length})</p>
            <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-[0.8rem]">
              {p.imputed.map((i) => <div key={i.field} className="contents"><dt className="font-mono text-fg3">{i.field} <span className="text-[0.72rem]">({i.rule})</span></dt><dd className="tnum text-right font-mono text-fg2">{typeof i.value === 'number' ? fmtInt(i.value) : i.value}</dd></div>)}
            </dl>
          </div>
        )}
        <p className="mt-3 text-[0.75rem] text-fg3">Model: {p.model.name}{p.model.stage ? ` · ${p.model.stage}` : ''}</p>
      </Disclosure>
    </Panel>
  )
}

/* ---------------------------------------------------------------- 2. LLM / template analysis */
const KIND_STYLE = {
  llm: { color: 'var(--color-cyan)', icon: Bot },
  llm_partial: { color: 'var(--color-avg)', icon: Layers },
  template: { color: 'var(--color-accent)', icon: ListChecks },
} as const

export function GuardrailsView({ g }: { g: Guardrails }) {
  const yn = (v: boolean | null) => (v === null ? 'not reported' : v ? 'yes' : 'no')
  return (
    <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-[0.8rem]" data-testid="guardrails">
      <dt className="text-fg3">Output matched the required structure</dt><dd className="text-right font-mono text-fg">{yn(g.schemaValid)}</dd>
      <dt className="text-fg3">Numbers matched the evidence</dt><dd className="text-right font-mono text-fg">{yn(g.numbersVerified)}</dd>
      <dt className="text-fg3">All checks passed</dt><dd className="text-right font-mono text-fg">{yn(g.passed)}</dd>
      <dt className="text-fg3">Replaced with template wording</dt><dd className="text-right font-mono text-fg">{g.replaced.length ? g.replaced.join(', ') : 'none'}</dd>
      <dt className="text-fg3">Financial wording rejected</dt><dd className="text-right font-mono text-fg">{g.forbiddenTerms.length ? g.forbiddenTerms.join(', ') : 'none found'}</dd>
      {g.llmRisksIgnored && <><dt className="text-fg3">Model-written risks</dt><dd className="text-right font-mono text-fg">ignored</dd></>}
      {g.unsupportedClaims.length > 0 && (
        <div className="col-span-2 mt-2"><dt className="mb-1 text-fg3">Rejected claims</dt><dd><ul className="list-disc space-y-1 pl-4 text-fg2">{g.unsupportedClaims.map((c) => <li key={c}>{c}</li>)}</ul></dd></div>
      )}
    </dl>
  )
}

export function AnalysisPanel({ a, g }: { a: Analysis; g: Guardrails | null }) {
  const pv = provenance(a, g)
  const { color, icon: Icon } = KIND_STYLE[pv.kind]
  return (
    <Panel id="agent-analysis" title="AI-generated analysis" tone={color}
      subtitle="Explanation written from the evidence below. It is separate from the prediction and cannot change it.">
      <div data-testid="provenance" data-kind={pv.kind} className="flex items-start gap-3 rounded-2xl border px-4 py-3" style={{ borderColor: `color-mix(in srgb, ${color} 45%, transparent)`, background: `color-mix(in srgb, ${color} 10%, transparent)` }}>
        <Icon size={18} aria-hidden className="mt-0.5 shrink-0" style={{ color }} />
        <div><p className="text-[0.9rem] font-semibold text-fg">{pv.headline}</p><p className="mt-0.5 text-[0.8rem] leading-snug text-fg2">{pv.detail}</p></div>
      </div>

      <p className="mt-5 text-[1rem] leading-relaxed text-fg" data-testid="analysis-summary">{a.summary}</p>

      {a.drivers.length > 0 ? (
        <ul className="mt-5 space-y-3" data-testid="drivers">
          {a.drivers.map((d, i) => (
            <li key={`${d.evidence_ref}-${i}`} className="rounded-2xl border border-line bg-black/20 px-4 py-3">
              <p className="text-[0.9rem] leading-snug text-fg">{d.claim}</p>
              <p className="mt-1.5 font-mono text-[0.72rem] text-fg3">evidence: {d.evidence_ref}{d.fact_ids.length ? ` · ${d.fact_ids.join(', ')}` : ''}</p>
            </li>
          ))}
        </ul>
      ) : <p className="mt-5 text-[0.84rem] text-fg3">No written drivers were produced for this run.</p>}

      {g && <Disclosure title="Guardrail results"><GuardrailsView g={g} /></Disclosure>}
    </Panel>
  )
}

/* ---------------------------------------------------------------- 3. structured risks */
const SEV: Record<string, { label: string; color: string }> = {
  caution: { label: 'Caution', color: 'var(--color-avg)' },
  info: { label: 'Note', color: 'var(--color-cyan)' },
}

function RiskCard({ r }: { r: RiskItem }) {
  const s = SEV[r.severity] ?? { label: r.severity, color: 'var(--color-fg2)' }
  return (
    <li className="rounded-2xl border bg-black/20 px-4 py-3" style={{ borderColor: `color-mix(in srgb, ${s.color} 40%, transparent)` }} data-testid="risk-item" data-risk-id={r.id}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border px-2 py-0.5 font-mono text-[0.72rem] font-medium uppercase tracking-wider" style={{ color: s.color, borderColor: `color-mix(in srgb, ${s.color} 50%, transparent)` }}>{s.label}</span>
        <span className="font-mono text-[0.72rem] uppercase tracking-wider text-fg3">{r.category}</span>
      </div>
      <p className="mt-2 text-[0.9rem] leading-snug text-fg">{r.text}</p>
      <p className="mt-2 font-mono text-[0.72rem] leading-relaxed text-fg3" data-testid="risk-trace">
        rule <span className="text-fg2">{r.id}</span>{r.rule ? <> · <span className="text-fg2">{r.rule}</span></> : null}
        {Object.entries(r.evidence).map(([k, v]) => <span key={k}> · {k} = <span className="text-fg2">{String(v)}</span></span>)}
      </p>
    </li>
  )
}

export function RisksPanel({ a }: { a: Analysis }) {
  return (
    <Panel id="agent-risks" title="Risks and cautions" subtitle="Generated by fixed rules from the evidence, never by the language model. Each one names the rule that fired.">
      {a.risks.length > 0
        ? <ul className="space-y-3" data-testid="risks">{a.risks.map((r) => <RiskCard key={r.id} r={r} />)}</ul>
        : <p className="text-[0.88rem] text-fg2" data-testid="no-risks">No evidence-specific risks were triggered for this run. That does not mean there are none: see the limitations below.</p>}
      {a.risksMalformed > 0 && <p role="status" className="mt-3 text-[0.78rem] text-fg3">{a.risksMalformed} risk {a.risksMalformed === 1 ? 'entry was' : 'entries were'} in an unrecognised format and {a.risksMalformed === 1 ? 'is' : 'are'} not shown.</p>}
    </Panel>
  )
}

/* ---------------------------------------------------------------- 4. evidence */
function CohortView({ s }: { s: Supplied<Cohort> }) {
  if (s.state === 'missing') return null
  return (
    <div data-testid="evidence-cohort">
      <h3 className="font-display text-[0.95rem] font-bold">Dataset cohort</h3>
      {s.state === 'unavailable' ? <div className="mt-2"><Unavailable what="Cohort evidence" /></div> : (
        <>
          <p className="mt-1 text-[0.78rem] text-fg3">{s.value.note}</p>
          {s.value.empty && <p className="mt-2 text-[0.84rem] text-fg2">No dataset films match all of these traits.</p>}
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[30rem] text-left text-[0.8rem]">
              <caption className="sr-only">Films remaining as each trait narrows the dataset, with their class counts</caption>
              <thead className="text-fg3"><tr><th scope="col" className="py-1.5 pr-3 font-medium">Narrowed by</th><th scope="col" className="px-2 text-right font-medium">Films</th>{SHOWN.map((k) => <th key={k} scope="col" className="px-2 text-right font-medium" style={{ color: CLASS_COLOR[k] }}>{k}</th>)}<th scope="col" className="pl-2 text-right font-medium">Mean IMDb</th></tr></thead>
              <tbody>
                {s.value.stages.map((st, i) => (
                  <tr key={`${st.label}-${i}`} className="border-t border-line text-fg2">
                    <th scope="row" className="py-1.5 pr-3 font-normal text-fg">{st.label}</th><td className="tnum px-2 text-right font-mono">{fmtInt(st.n)}</td>
                    {SHOWN.map((k) => <td key={k} className="tnum px-2 text-right font-mono">{fmtInt(st.counts[k])}</td>)}
                    <td className="tnum pl-2 text-right font-mono">{st.mean_score ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

function SensitivityView({ s }: { s: Supplied<SensitivityEvidence> }) {
  if (s.state === 'missing') return null
  return (
    <div data-testid="evidence-sensitivity">
      <h3 className="font-display text-[0.95rem] font-bold">Sensitivity sweep</h3>
      {s.state === 'unavailable' ? <div className="mt-2"><Unavailable what="Sensitivity evidence" /></div> : (
        <>
          <p className="mt-1 text-[0.8rem] text-fg2">Top class across the tested range of <b className="font-semibold text-fg">{s.value.label}</b>, every other input unchanged:</p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {s.value.top_class_by_range.map((seg, i) => (
              <li key={i} className="flex items-center gap-2 rounded-full border border-line bg-black/20 px-3 py-1 text-[0.8rem]"><ClassBadge value={seg.label as MovieClass} /><span className="font-mono text-fg2">{seg.from} to {seg.to}</span></li>
            ))}
          </ul>
          <p className="mt-2 text-[0.75rem] text-fg3">{s.value.note}</p>
        </>
      )}
    </div>
  )
}

function ComparablesView({ s }: { s: Supplied<Comparable[]> }) {
  if (s.state === 'missing') return null
  return (
    <div data-testid="evidence-comparables">
      <h3 className="font-display text-[0.95rem] font-bold">Comparable earlier films</h3>
      {s.state === 'unavailable' ? <div className="mt-2"><Unavailable what="Comparable films" /></div>
        : s.value.length === 0 ? <p className="mt-2 text-[0.84rem] text-fg2" data-testid="comparables-none">The search ran and found no similar earlier films.</p> : (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[26rem] text-left text-[0.8rem]">
              <caption className="sr-only">Earlier films similar to this one, with their dataset class</caption>
              <thead className="text-fg3"><tr><th scope="col" className="py-1.5 pr-3 font-medium">Title</th><th scope="col" className="px-2 font-medium">Year</th><th scope="col" className="px-2 font-medium">Genres</th><th scope="col" className="pl-2 text-right font-medium">Class</th></tr></thead>
              <tbody>{s.value.map((c, i) => (
                <tr key={`${c.title}-${i}`} className="border-t border-line text-fg2">
                  <th scope="row" className="py-1.5 pr-3 font-normal text-fg">{c.title}</th><td className="tnum px-2 font-mono">{c.year}</td>
                  <td className="px-2">{c.genres.replaceAll('|', ' / ')}</td><td className="pl-2 text-right"><ClassBadge value={c.Classify} /></td>
                </tr>))}</tbody>
            </table>
            <p className="mt-2 text-[0.75rem] text-fg3">Each class is that film's own IMDb-score band in the dataset. It is context about other films, never an input to this prediction.</p>
          </div>
        )}
    </div>
  )
}

/* ---------------------------------------------------------------- 4b. contextual research (its own panel: it is not model input and not evidence for the prediction) */
const utc = (iso: string | null) => {
  if (!iso) return null
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? null : `${d.toISOString().slice(0, 16).replace('T', ' ')} UTC`
}
const KIND_LABEL: Record<string, string> = { film: 'Film page', genre: 'Genre background' }
const FAILED = ['unavailable', 'timeout', 'error']

function SourceCard({ x }: { x: ResearchSource }) {
  return (
    <li className="rounded-2xl border border-line bg-black/20 px-4 py-3" data-testid="research-source">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-line-strong px-2 py-0.5 font-mono text-[0.72rem] uppercase tracking-wider text-fg3">{KIND_LABEL[x.kind] ?? x.kind}</span>
        <a className="text-[0.95rem] font-semibold text-accent underline underline-offset-4" href={x.url} rel="noreferrer noopener" target="_blank">
          {x.title}<span className="sr-only"> (opens Wikipedia in a new tab)</span>
        </a>
      </div>
      {x.excerpt && <p className="mt-2 text-[0.84rem] leading-snug text-fg2">“{x.excerpt}”</p>}
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[0.72rem] text-fg3">
        {utc(x.revisionTimestamp) && <><dt>Page version</dt><dd className="text-fg2">{utc(x.revisionTimestamp)}{x.revisionId !== null ? ` · revision ${x.revisionId}` : ''}</dd></>}
        {utc(x.retrievedAt) && <><dt>Retrieved</dt><dd className="text-fg2">{utc(x.retrievedAt)}</dd></>}
      </dl>
    </li>
  )
}

export function ResearchView({ s }: { s: Supplied<Research> }) {
  const head = <h3 className="flex items-center gap-2 font-display text-[0.95rem] font-bold"><Telescope size={15} aria-hidden className="text-fg3" />Online research (Wikipedia)</h3>
  if (s.state !== 'ok') {
    return (
      <div data-testid="evidence-research" data-status={s.state === 'missing' ? 'missing' : 'unreadable'}>
        {head}
        <p className="mt-1 text-[0.84rem] text-fg2">{s.state === 'missing' ? 'This run did not report a research status. No web sources are shown.' : 'The research result could not be read for this run. No web sources are shown.'}</p>
      </div>
    )
  }
  const r = s.value
  const failed = FAILED.includes(r.status)
  const count = r.sources.length
  const cutoff = utc(r.cutoff)
  return (
    <div data-testid="evidence-research" data-status={r.status}>
      {head}
      {r.status === 'disabled' ? (
        <p className="mt-1 text-[0.84rem] text-fg2"><b className="font-semibold text-fg">Online research is not enabled on this server.</b> No web pages were searched, and no sources are cited.</p>
      ) : (
        <>
          <p className="mt-1 text-[0.84rem] text-fg2" data-testid="research-count"><b className="font-semibold text-fg">{count} {count === 1 ? 'source' : 'sources'} retrieved</b>
            {r.requestsMade !== null && r.maxRequests !== null ? ` using ${r.requestsMade} of at most ${r.maxRequests} requests` : ''}.</p>
          {failed && (
            <p role="alert" className="mt-2 rounded-2xl border border-avg/40 bg-avg/10 px-4 py-3 text-[0.84rem] text-fg2" data-testid="research-problem">
              <b className="font-semibold text-fg">{r.status === 'timeout' ? 'Research timed out.' : r.status === 'error' ? 'Research hit an error.' : 'Research was unavailable.'}</b> {r.message ?? ''}
              {r.reason ? <> ({reasonText(r.reason)}).</> : null} The prediction and analysis are unaffected.
            </p>
          )}
          {!failed && r.status === 'no_results' && <p className="mt-2 text-[0.84rem] text-fg2" data-testid="research-none">{r.message ?? 'No usable page versions were found.'} Nothing is cited.</p>}
          {r.status === 'partial' && <p role="status" className="mt-2 text-[0.84rem] text-fg2" data-testid="research-partial">{r.message}{r.reason ? ` (${reasonText(r.reason)})` : ''}</p>}
          {count > 0 && <ul className="mt-3 space-y-3" data-testid="research-sources">{r.sources.map((x) => <SourceCard key={x.url} x={x} />)}</ul>}
          {r.sourcesRejected > 0 && <p role="status" className="mt-2 text-[0.78rem] text-fg3">{r.sourcesRejected} returned {r.sourcesRejected === 1 ? 'source was' : 'sources were'} not shown because the link was not an approved Wikipedia address.</p>}
          {cutoff && <p className="mt-3 text-[0.78rem] leading-snug text-fg3">Each page is shown as it stood before <span className="font-mono text-fg2">{cutoff}</span>, so it reflects what could be known before release. A page that did not exist yet is left out, never replaced by today’s version.</p>}
          {r.skipped.length > 0 && (
            <Disclosure title={`Not retrieved (${r.skipped.length})`}>
              <ul className="list-disc space-y-1 pl-4" data-testid="research-skipped">{r.skipped.map((k, i) => <li key={`${k.subject}-${i}`}><span className="text-fg">{k.subject}</span>: {reasonText(k.reason) ?? k.reason}</li>)}</ul>
            </Disclosure>
          )}
        </>
      )}
    </div>
  )
}

export function ResearchPanel({ s }: { s: Supplied<Research> }) {
  return (
    <Panel id="agent-research" title="Contextual research" tone="var(--color-cyan)"
      subtitle="Background from the web. It is not an input to the Random Forest, is not shown to the language model, and is not evidence that the prediction is right or wrong.">
      <ResearchView s={s} />
    </Panel>
  )
}

export function EvidencePanel({ run }: { run: AgentRun }) {
  return (
    <Panel id="agent-evidence" title="Evidence the analysis was built from" subtitle="Gathered by code from the dataset and the saved model using pre-release fields only. Web research is shown separately below.">
      <div className="space-y-6">
        <CohortView s={run.cohort} />
        <SensitivityView s={run.sensitivity} />
        <ComparablesView s={run.comparables} />
      </div>
    </Panel>
  )
}

/* ---------------------------------------------------------------- 5. limitations + trace */
export function LimitationsPanel({ run }: { run: AgentRun }) {
  const items = Array.from(new Set([...(run.analysis?.limitations ?? []), ...run.disclaimers]))
  return (
    <Note tone="info" title="What this result does and does not mean">
      <ul className="list-disc space-y-1.5 pl-4">
        <li>Hit, Average and Flop are <b className="font-semibold text-fg">IMDb-score classes</b> (Flop below 3, Average 3 to under 6, Hit 6 or above). They are not predictions of financial success.</li>
        <li>The class scores are <b className="font-semibold text-fg">uncalibrated</b>, so they rank the three classes and are not probabilities of real-world outcomes.</li>
        {items.map((x) => <li key={x}>{x}</li>)}
      </ul>
    </Note>
  )
}

const STEP_LABEL: Record<string, string> = {
  prediction: 'Prediction (existing model)', 'analyst.cohort': 'Analyst: cohort', 'analyst.sensitivity': 'Analyst: sensitivity', 'analyst.comparables': 'Analyst: comparables',
  research: 'Research', 'report.llm': 'Report: language model', 'report.verify': 'Report: verification', risks: 'Risk rules', report: 'Report',
}

export function TraceTable({ run }: { run: AgentRun }) {
  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[22rem] text-left text-[0.8rem]" data-testid="trace">
          <caption className="sr-only">Steps the agent ran, as reported by the API</caption>
          <thead className="text-fg3"><tr><th scope="col" className="py-1 pr-3 font-medium">Step</th><th scope="col" className="px-2 font-medium">Status</th><th scope="col" className="px-2 text-right font-medium">Seconds</th><th scope="col" className="pl-2 font-medium">Detail</th></tr></thead>
          <tbody>{run.trace.map((t, i) => (
            <tr key={`${t.step}-${i}`} className="border-t border-line text-fg2">
              <th scope="row" className="py-1 pr-3 font-normal text-fg">{STEP_LABEL[t.step] ?? t.step}</th>
              <td className={cx('px-2 font-mono', t.status === 'failed' && 'text-danger')}>{t.status}</td>
              <td className="tnum px-2 text-right font-mono">{t.seconds ?? '—'}</td><td className="pl-2">{t.detail ? (reasonText(t.detail) ?? t.detail) : ''}</td>
            </tr>))}</tbody>
        </table>
      </div>
      {run.toolCalls !== null && <p className="mt-2 text-[0.75rem] text-fg3">Tool calls used: {run.toolCalls}</p>}
    </>
  )
}

export function TracePanel({ run }: { run: AgentRun }) {
  if (!run.trace.length) return null
  return (
    <Disclosure title={`Run trace (${run.trace.length} steps${run.totalSeconds !== null ? `, ${run.totalSeconds}s total` : ''})`}>
      <TraceTable run={run} />
    </Disclosure>
  )
}

/* ---------------------------------------------------------------- composed result */
export function ResultView({ run, onDiscuss }: { run: AgentRun; onDiscuss?: () => void }) {
  if (!run.prediction) return null
  return (
    <div className="space-y-5" data-testid="run-result">
      {onDiscuss && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-[1.25rem] border border-cyan/40 bg-cyan/[0.07] px-5 py-3.5" data-testid="discuss-cta">
          <p className="min-w-0 flex-1 basis-56 text-[0.86rem] text-fg2"><b className="font-semibold text-fg">Have questions about this result?</b> The AI Assistant can explain it in plain language. It reads this stored analysis and cannot change the prediction.</p>
          <button type="button" onClick={onDiscuss} data-testid="discuss" className="inline-flex min-h-11 items-center gap-2 rounded-full border border-cyan/60 bg-cyan/15 px-5 text-sm font-semibold text-fg hover:bg-cyan/25 sm:min-h-10">
            <MessagesSquare size={16} aria-hidden />Discuss in AI Assistant
          </button>
        </div>
      )}
      <PredictionPanel p={run.prediction} />
      {run.analysis
        ? <><AnalysisPanel a={run.analysis} g={run.guardrails} /><RisksPanel a={run.analysis} /></>
        : <Note tone="info" title="No written analysis"><p>The run finished without an analysis section. The prediction above is unaffected.</p></Note>}
      <EvidencePanel run={run} />
      <ResearchPanel s={run.research} />
      <LimitationsPanel run={run} />
      <TracePanel run={run} />
    </div>
  )
}

/* ---------------------------------------------------------------- failed run */
export function FailedRun({ run, onNew }: { run: AgentRun | null; onNew: () => void }) {
  return (
    <div role="alert" className="space-y-4 rounded-[1.25rem] border border-flop/40 bg-flop/10 p-5" data-testid="run-failed">
      <div className="flex items-center gap-2 text-danger"><AlertTriangle size={18} aria-hidden /><h3 className="font-display text-base font-bold">The run failed</h3></div>
      <p className="max-w-prose text-sm text-fg2">{run?.error?.message ?? 'The server marked this run as failed, for example because it was interrupted by a restart.'} No prediction was produced.</p>
      {run && <TracePanel run={run} />}
      <button type="button" onClick={onNew} className="inline-flex min-h-11 items-center gap-2 rounded-full border border-line-strong bg-white/[0.04] px-5 text-sm font-semibold hover:bg-white/[0.09]">Start a new run</button>
    </div>
  )
}

