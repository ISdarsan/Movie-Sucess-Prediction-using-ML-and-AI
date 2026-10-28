import { Check, ChevronDown, Clapperboard, ClipboardCopy, History, RotateCcw, Sparkles, Trash2 } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { lazy, Suspense, useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import { Counter, ProjectorLoader } from '../components/fx'
import { Button, ClassBadge, Disclosure, ErrorState, LoadingBlock, Note, PageHeader, Panel, cx } from '../components/ui'
import { ApiError, api, qs, type FieldProblem } from '../lib/api'
import { CLASS_COLOR, CLASSES, fmtInt, fmtPct, type MovieClass } from '../lib/format'
import type { Explain, ModelMetrics, MovieDetail, MovieList, PredictResult, PredictSchema } from '../lib/types'
import { useApi } from '../lib/useApi'

/* ---------- field definitions: ranges mirror backend/app/schemas.py ---------- */
interface NumField { key: string; label: string; unit?: string; help: string; min: number; max: number; int?: boolean; required?: boolean; step?: string; post?: boolean }
const CORE: NumField[] = [
  { key: 'title_year', label: 'Release year', help: 'Training data spans 1916–2016.', min: 1888, max: 2030, int: true, required: true },
  { key: 'duration', label: 'Duration', unit: 'min', help: 'Runtime in minutes.', min: 1, max: 600, required: true },
]
const PRODUCTION: NumField[] = [
  { key: 'budget', label: 'Budget', unit: 'USD', help: 'Production budget. US films are in dollars; the training data holds local currency for other countries.', min: 0, max: 10_000_000_000, int: true },
  { key: 'aspect_ratio', label: 'Aspect ratio', help: 'e.g. 2.35 or 1.85.', min: 0.5, max: 10, step: '0.01' },
]
const ALL_NUM = [...CORE, ...PRODUCTION]
const loadInsight = () => import('../components/insight')
const WhatIf = lazy(() => loadInsight().then((m) => ({ default: m.WhatIf })))
const CohortLens = lazy(() => loadInsight().then((m) => ({ default: m.CohortLens })))
const InsightFallback = () => <div aria-hidden className="skeleton h-72 w-full" />

const N_REQUIRED = 5   // year, runtime, genres, country, content rating
type Form = Record<string, string> & { genres: string }

const emptyForm = (): Form => ({ genres: '', country: '', content_rating: '', language: '', color: '' } as Form)

interface HistoryItem { id: number; at: string; label: MovieClass; probs: Record<MovieClass, number>; summary: string; form: Form; actual?: { score: number; cls: MovieClass; title: string } }
const HKEY = 'frame.predict.history.v1'
const loadHistory = (): HistoryItem[] => { try { return JSON.parse(sessionStorage.getItem(HKEY) ?? '[]') } catch { return [] } }

export default function Predictor() {
  const schema = useApi<PredictSchema>('/api/predict/schema', { cache: true })
  const loc = useLocation()
  const seeded = (loc.state as { movie?: MovieDetail } | null)?.movie

  const [form, setForm] = useState<Form>(emptyForm)
  const [genres, setGenres] = useState<string[]>([])
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [failure, setFailure] = useState<ApiError | null>(null)
  const [actual, setActual] = useState<HistoryItem['actual']>()
  const [history, setHistory] = useState<HistoryItem[]>(loadHistory)
  const [lastBody, setLastBody] = useState<Record<string, unknown> | null>(null)
  const resultRef = useRef<HTMLDivElement>(null)
  const seedApplied = useRef(false)

  useEffect(() => { const go = () => { void loadInsight() }; if ('requestIdleCallback' in window) { const h = window.requestIdleCallback(go, { timeout: 4000 }); return () => window.cancelIdleCallback(h) } const t = setTimeout(go, 2000); return () => clearTimeout(t) }, [])
  useEffect(() => { try { sessionStorage.setItem(HKEY, JSON.stringify(history.slice(0, 12))) } catch { /* storage unavailable */ } }, [history])

  // Prefill from "Use in predictor" on the Explore page (or a demo shortcut): real values from a dataset record.
  function applyMovie(m: MovieDetail) {
    const next = emptyForm()
    for (const f of ALL_NUM) { const v = (m as unknown as Record<string, unknown>)[f.key === 'title_year' ? 'year' : f.key]; if (typeof v === 'number') next[f.key] = String(v) }
    next.country = m.country ?? ''; next.content_rating = m.content_rating ?? ''; next.language = m.language ?? ''; next.color = m.color ?? ''
    setForm(next); setGenres(m.genre_list ?? []); setErrors({}); setResult(null); setFailure(null)
    setActual({ score: m.imdb_score, cls: m.Classify, title: m.title })
  }
  useEffect(() => {
    if (!seeded || seedApplied.current) return
    seedApplied.current = true
    applyMovie(seeded)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seeded])

  // Demo shortcut: pull a random REAL film of the chosen class from the dataset API
  const [demoBusy, setDemoBusy] = useState<MovieClass | null>(null)
  const [demoErr, setDemoErr] = useState<string | null>(null)
  async function loadReal(cls: MovieClass) {
    setDemoBusy(cls); setDemoErr(null)
    try {
      const size = 5
      const first = await api.get<MovieList>(`/api/movies${qs({ classify: cls, page_size: size, page: 1 })}`)
      // pick a random film that has every REQUIRED field (about 6% of films have no content rating), so the form can be submitted as loaded
      let row = null
      for (let attempt = 0; attempt < 12 && !row; attempt++) {
        const idx = Math.floor(Math.random() * first.total)
        const list = idx < size ? first : await api.get<MovieList>(`/api/movies${qs({ classify: cls, page_size: size, page: Math.floor(idx / size) + 1 })}`)
        const cand = list.rows[idx % size]
        if (cand && cand.year && cand.duration && cand.country && cand.content_rating && cand.genres) row = cand
      }
      if (!row) throw new ApiError(0, 'no_complete_film', 'Could not find a complete film of that class. Try again.')
      applyMovie(await api.get<MovieDetail>(`/api/movies/${row.id}`))
    } catch (e) { setDemoErr(e instanceof ApiError ? e.message : 'Could not load a film.') } finally { setDemoBusy(null) }
  }
  const yr = Number(form.title_year), dur = Number(form.duration)
  const cohortProps = { genres, country: form.country, contentRating: form.content_rating, year: Number.isInteger(yr) && yr >= 1888 && yr <= 2030 ? yr : null, duration: Number.isFinite(dur) && dur >= 1 && dur <= 600 && form.duration?.trim() ? dur : null }
  const requiredFilled = ['title_year', 'duration'].filter((k) => form[k]?.trim()).length + (genres.length ? 1 : 0) + (form.country ? 1 : 0) + (form.content_rating ? 1 : 0)

  const set = (k: string, v: string) => { setForm((f) => ({ ...f, [k]: v })); if (errors[k]) setErrors((e) => { const n = { ...e }; delete n[k]; return n }) }

  function validate(): Record<string, string> {
    const e: Record<string, string> = {}
    for (const f of ALL_NUM) {
      const raw = form[f.key]?.trim()
      if (!raw) { if (f.required) e[f.key] = `${f.label} is required.`; continue }
      const n = Number(raw)
      if (!Number.isFinite(n)) e[f.key] = 'Enter a number.'
      else if (f.int && !Number.isInteger(n)) e[f.key] = 'Use a whole number.'
      else if (n < f.min || n > f.max) e[f.key] = `Must be between ${fmtInt(f.min)} and ${fmtInt(f.max)}.`
    }
    if (!genres.length) e.genres = 'Select at least one genre.'
    if (!form.country) e.country = 'Country is required.'
    if (!form.content_rating) e.content_rating = 'Content rating is required.'
    return e
  }

  async function submit(ev: React.FormEvent) {
    ev.preventDefault()
    const e = validate()
    setErrors(e); setFailure(null)
    if (Object.keys(e).length) { document.getElementById(`f-${Object.keys(e)[0]}`)?.focus(); return }
    const body: Record<string, unknown> = { genres }
    for (const f of ALL_NUM) if (form[f.key]?.trim()) body[f.key] = Number(form[f.key])
    for (const k of ['country', 'content_rating', 'language', 'color']) if (form[k]) body[k] = form[k]
    setBusy(true)
    try {
      const [r] = await Promise.all([api.post<PredictResult>('/api/predict', body), new Promise((res) => setTimeout(res, matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 1400))])
      setResult(r); setLastBody(body)
      const snap = { ...form, genres: genres.join('|') } as Form
      setHistory((h) => [{ id: Date.now(), at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), label: r.label, probs: r.probabilities,
        summary: `${genres.join(' / ')} · ${form.title_year}`, form: snap, actual }, ...h].slice(0, 12))
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }))
    } catch (err) {
      const ae = err instanceof ApiError ? err : new ApiError(0, 'unknown', 'Unexpected error.')
      if (ae.fields.length) {
        const m: Record<string, string> = {}
        ae.fields.forEach((f: FieldProblem) => { m[f.field] = f.message })
        setErrors(m)
      }
      setResult(null); setFailure(ae)
    } finally { setBusy(false) }
  }

  function reset() { setForm(emptyForm()); setGenres([]); setErrors({}); setResult(null); setFailure(null); setActual(undefined) }
  function restore(h: HistoryItem) { setForm(h.form); setGenres(h.form.genres.split('|').filter(Boolean)); setActual(h.actual); setResult(null); setErrors({}); window.scrollTo({ top: 0, behavior: 'smooth' }) }

  const med = schema.data?.defaults_when_blank
  const blankNote = (k: string) => (med && med.numeric_median[k] != null ? `Blank → training median ${fmtInt(med.numeric_median[k])}` : undefined)

  return (
    <>
      <PageHeader eyebrow="Predictor" title="Put a film on the screen" highlight="screen"
        actions={<Button variant="ghost" icon={<RotateCcw size={15} aria-hidden />} onClick={reset}>Reset form</Button>}>
        Describe a film using only what is known <b className="font-semibold text-fg">before release</b>, and the saved model (<span className="font-mono text-[0.85em] text-fg">pre_release_movie_classifier.joblib</span>) assigns a category. Leave optional fields blank and the pipeline substitutes the training median or an “Unknown” category — every substitution is listed in the result.
      </PageHeader>

      <div className="mb-5">
        <Note tone="info" title="Pre-release inputs only — and a modest signal">
          <p>No votes, reviews, gross or social-media numbers are used, because none of them exist before a film is released. That makes this a genuine pre-release estimate, but also a weaker one: pre-release information explains only part of how a film ends up rated. Read the class scores as a ranking, not a promise.</p>
        </Note>
      </div>

      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35, duration: 0.5 }} className="mb-6 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border border-line bg-white/[0.03] px-4 py-3">
        <p className="flex items-center gap-2 text-[0.82rem] font-medium text-fg"><Clapperboard size={16} className="text-cyan" aria-hidden />Try a real film</p>
        <div className="flex flex-wrap gap-2">
          {([...CLASSES].reverse()).map((c) => (
            <button key={c} type="button" disabled={demoBusy !== null || !schema.data} onClick={() => loadReal(c)} aria-busy={demoBusy === c}
              className="inline-flex min-h-10 items-center gap-2 rounded-full border px-4 text-[0.8rem] font-semibold transition-all duration-200 hover:-translate-y-0.5 active:scale-95 disabled:opacity-50"
              style={{ color: CLASS_COLOR[c], borderColor: `color-mix(in srgb, ${CLASS_COLOR[c]} 50%, transparent)`, background: `color-mix(in srgb, ${CLASS_COLOR[c]} 10%, transparent)` }}>
              {demoBusy === c ? <span aria-hidden className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" /> : <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: CLASS_COLOR[c] }} />}
              A real {c}
            </button>
          ))}
        </div>
        <p className="min-w-0 flex-1 basis-60 text-[0.75rem] leading-snug text-fg3" role={demoErr ? 'alert' : undefined}>{demoErr ?? 'Loads a random film from the dataset, then you run the model and compare it with the actual class. Most dataset films were also in the training data, so agreement is not a fair test.'}</p>
      </motion.div>

      {schema.error && <ErrorState title="The predictor is unavailable" message={schema.error.message} onRetry={schema.retry} />}
      {schema.loading && !schema.data && <LoadingBlock height="h-96" label="Loading model schema" />}

      {schema.data && (
        <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          <form onSubmit={submit} noValidate className="space-y-4" aria-label="Movie attributes">
            {seeded && actual && <p className="rounded-2xl border border-violet/50 bg-violet/15 px-4 py-2.5 text-[0.82rem] text-fg2">Prefilled from the dataset record <b className="font-medium text-fg">{actual.title}</b>. Its actual IMDb score ({actual.score.toFixed(1)}, {actual.cls}) will be shown beside the model’s answer.</p>}

            <Panel title="01 · The film" subtitle="Core description" tone="#ff4fa3">
              <div className="grid gap-4 sm:grid-cols-2">
                {CORE.map((f) => <NumberInput key={f.key} f={f} value={form[f.key] ?? ''} onChange={(v) => set(f.key, v)} error={errors[f.key]} hint={blankNote(f.key)} />)}
                <div className="sm:col-span-2"><GenrePicker all={schema.data.genres} value={genres} onChange={(g) => { setGenres(g); setErrors((e) => { const n = { ...e }; delete n.genres; return n }) }} error={errors.genres} /></div>
                <SelectInput id="country" label="Country" required options={schema.data.categorical.country} value={form.country} onChange={(v) => set('country', v)} error={errors.country} />
                <SelectInput id="content_rating" label="Content rating" required options={schema.data.categorical.content_rating} value={form.content_rating} onChange={(v) => set('content_rating', v)} error={errors.content_rating} />
                <SelectInput id="language" label="Language" options={schema.data.categorical.language} value={form.language} onChange={(v) => set('language', v)} error={errors.language} hint="Blank → its own “Unknown” category" />
                <SelectInput id="color" label="Colour" options={schema.data.categorical.color} value={form.color} onChange={(v) => set('color', v)} error={errors.color} hint="Blank → its own “Unknown” category" />
              </div>
            </Panel>

            <Panel title="02 · Production" subtitle="Optional. Leave blank if it isn’t decided yet — the pipeline then uses the training median." tone="#ffb23e">
              <div className="grid gap-4 sm:grid-cols-2">
                {PRODUCTION.map((f) => <NumberInput key={f.key} f={f} value={form[f.key] ?? ''} onChange={(v) => set(f.key, v)} error={errors[f.key]} hint={blankNote(f.key)} />)}
              </div>
            </Panel>

            <Disclosure title={`Not asked, on purpose: ${schema.data.excluded.length} columns that would leak the answer`}>
              <p className="mb-3 text-fg3">This model is strictly pre-release. These columns exist in the dataset but are never given to the model, because they are only known after release or their timing cannot be established.</p>
              <ul className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
                {schema.data.excluded.map((e) => <li key={e.column} className="text-[0.8rem]"><span className="font-mono text-fg">{e.column}</span><span className="block text-fg3">{e.reason}</span></li>)}
              </ul>
            </Disclosure>

            <div className="glass sticky bottom-3 z-10 flex items-center gap-3 rounded-full p-2 pr-4">
              <Button type="submit" variant="primary" loading={busy} icon={!busy && <Sparkles size={16} aria-hidden />} className="flex-1 !min-h-12 text-base sm:flex-none sm:min-w-60">{busy ? 'Scoring…' : 'Predict category'}</Button>
              <div className="ml-auto hidden items-center gap-2.5 sm:flex" role="status" aria-label={`${requiredFilled} of ${N_REQUIRED} required fields filled`}>
                <span aria-hidden className="flex gap-1">{Array.from({ length: N_REQUIRED }, (_, i) => <motion.span key={i} className="h-1.5 w-5 rounded-full" animate={{ backgroundColor: i < requiredFilled ? '#2ee6c8' : 'rgb(255 255 255 / .14)', boxShadow: i < requiredFilled ? '0 0 10px #2ee6c8aa' : '0 0 0 transparent' }} transition={{ duration: 0.3 }} />)}</span>
                <span className="font-mono text-[0.72rem] text-fg3">{requiredFilled}/{N_REQUIRED} required</span>
              </div>
              {Object.keys(errors).length > 0 && <p role="alert" className="text-sm text-danger">Fix {Object.keys(errors).length} field{Object.keys(errors).length > 1 ? 's' : ''} to continue.</p>}
            </div>
          </form>

          <div ref={resultRef} className="space-y-5 xl:sticky xl:top-24" aria-live="polite">
            {!failure && (
              <Screen tone={!busy && result ? result.label : undefined}>
                {busy ? <ProjectorLoader label="Scoring the film" steps={['Encoding the 9 pre-release inputs', 'Scoring with the saved model', 'Reading the class scores']} /> : result ? <ResultCard key={result.label + JSON.stringify(result.probabilities)} r={result} actual={actual} body={lastBody} /> : <CohortLens {...cohortProps} />}
              </Screen>
            )}
            {!busy && failure && <ErrorState title={failure.code === 'model_unavailable' || failure.status === 503 ? 'Model unavailable' : failure.status === 422 ? 'Please check the inputs' : 'Prediction failed'} message={failure.message} />}
            <HistoryList items={history} onRestore={restore} onClear={() => setHistory([])} />
          </div>
        </div>
      )}
    </>
  )
}

/* ---------- inputs ---------- */
function FieldShell({ id, label, required, unit, help, hint, error, post, children }: { id: string; label: string; required?: boolean; unit?: string; help?: string; hint?: string; error?: string; post?: boolean; children: ReactNode }) {
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 flex items-baseline justify-between gap-2 text-[0.82rem] font-medium text-fg">
        <span>{label}{unit && <span className="ml-1.5 font-mono text-[0.72rem] font-normal text-fg3">{unit}</span>}{required && <span className="ml-1 text-pink" aria-label="required">*</span>}</span>
        {post && <span className="font-mono text-[0.72rem] uppercase tracking-wider text-fg3">post-release</span>}
      </label>
      {children}
      <p id={`${id}-d`} className={cx('mt-1.5 text-[0.75rem] leading-snug', error ? 'text-danger' : 'text-fg3')}>{error ?? [help, hint].filter(Boolean).join(' · ')}</p>
    </div>
  )
}
const inputCls = (err?: string) => cx('min-h-11 w-full rounded-2xl border bg-black/30 px-4 text-sm text-fg transition-[border-color,box-shadow] duration-200 placeholder:text-fg3/70 hover:border-white/40 focus:outline-none focus:border-cyan focus:shadow-[0_0_0_4px_rgb(37_232_211/.16),0_0_26px_-6px_rgb(37_232_211/.6)] sm:min-h-10', err ? 'border-flop' : 'border-line-strong')

function NumberInput({ f, value, onChange, error, hint }: { f: NumField; value: string; onChange: (v: string) => void; error?: string; hint?: string }) {
  const id = `f-${f.key}`
  return (
    <FieldShell id={id} label={f.label} required={f.required} unit={f.unit} help={f.help} hint={f.required ? undefined : hint} error={error} post={f.post}>
      <input id={id} inputMode={f.int ? 'numeric' : 'decimal'} autoComplete="off" value={value} onChange={(e) => onChange(e.target.value)} placeholder={f.required ? '' : 'optional'}
        aria-invalid={!!error} aria-describedby={`${id}-d`} step={f.step} className={cx(inputCls(error), 'tnum font-mono')} />
    </FieldShell>
  )
}

function SelectInput({ id, label, options, value, onChange, required, error, hint }: { id: string; label: string; options: string[]; value: string; onChange: (v: string) => void; required?: boolean; error?: string; hint?: string }) {
  const fid = `f-${id}`
  return (
    <FieldShell id={fid} label={label} required={required} hint={hint} error={error}>
      <select id={fid} value={value} onChange={(e) => onChange(e.target.value)} aria-invalid={!!error} aria-describedby={`${fid}-d`} className={inputCls(error)}>
        <option value="">{required ? 'Select…' : 'Not specified'}</option>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </FieldShell>
  )
}

function GenrePicker({ all, value, onChange, error }: { all: string[]; value: string[]; onChange: (g: string[]) => void; error?: string }) {
  const gid = useId()
  const toggle = (g: string) => onChange(value.includes(g) ? value.filter((x) => x !== g) : value.length >= 6 ? value : [...value, g])
  const sorted = useMemo(() => [...value].sort().join(' / '), [value])
  return (
    <fieldset aria-describedby={`${gid}-d`} id="f-genres" tabIndex={-1}>
      <legend className="mb-1.5 text-[0.82rem] font-medium text-fg">Genres<span className="ml-1 text-pink" aria-label="required">*</span> <span className="font-normal text-fg3">pick up to 6</span></legend>
      <div className="flex flex-wrap gap-1.5">
        {all.map((g) => {
          const on = value.includes(g)
          return (
            <button key={g} type="button" aria-pressed={on} onClick={() => toggle(g)}
              className={cx('min-h-9 rounded-full border px-3.5 text-[0.8rem] font-medium transition-all duration-200 active:scale-95', on ? 'border-transparent bg-[image:var(--grad-brand)] font-bold text-[#12082a] shadow-[0_6px_20px_-6px_rgb(255_79_163/.8)]' : 'border-line-strong text-fg2 hover:-translate-y-0.5 hover:border-white/50 hover:text-fg')}>{g}</button>
          )
        })}
      </div>
      <p id={`${gid}-d`} className={cx('mt-2 text-[0.75rem] leading-snug', error ? 'text-danger' : 'text-fg3')}>
        {error ?? (value.length ? <>Sent as <span className="font-mono text-fg2">{sorted}</span>. Each genre is its own flag in the model, so any mix is valid.</> : 'Each genre is a separate flag, so any combination works.')}
      </p>
    </fieldset>
  )
}

/* ---------- result: the cinema screen ---------- */
const TAGLINE: Record<MovieClass, string> = {
  Hit: 'This profile looks like past films rated 6 or higher.',
  Average: 'This profile looks like past films rated between 3 and 6.',
  Flop: 'This profile looks like past films rated below 3.',
}

function Screen({ tone, children }: { tone?: MovieClass; children: ReactNode }) {
  const c = tone ? CLASS_COLOR[tone] : '#8b5cff'
  return (
    <div className="relative">
      <div aria-hidden className="pointer-events-none absolute -inset-3 -z-10 sm:-inset-6 rounded-[3rem] opacity-60 blur-3xl transition-colors duration-700" style={{ background: `radial-gradient(60% 60% at 50% 35%, ${c}55, transparent 70%)` }} />
      <div className="glass glow-edge is-on relative overflow-hidden rounded-[2rem]" style={{ ['--edge' as string]: c }}>
        <div aria-hidden className="perf-edge absolute inset-x-0 top-0 h-3.5 opacity-70" />
        <div aria-hidden className="perf-edge absolute inset-x-0 bottom-0 h-3.5 opacity-70" />
        <div className="px-1 py-4">{children}</div>
      </div>
    </div>
  )
}

function ResultCard({ r, actual, body }: { r: PredictResult; actual?: HistoryItem['actual']; body: Record<string, unknown> | null }) {
  const shown = [...CLASSES].reverse()
  const c = CLASS_COLOR[r.label]
  const [openIn, setOpenIn] = useState(false)
  const explain = useApi<Explain>('/api/explain', { cache: true }).data
  const metrics = useApi<ModelMetrics>('/api/model-metrics', { cache: true }).data
  const drivers = explain?.shap[r.label].bar.slice(0, 5)
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="relative px-5 pb-5 pt-6 sm:px-7">
      <motion.span aria-hidden className="pointer-events-none absolute left-1/2 top-24 h-40 w-40 -translate-x-1/2 rounded-full border-2" style={{ borderColor: c }}
        initial={{ scale: 0.2, opacity: 0.9 }} animate={{ scale: 5, opacity: 0 }} transition={{ duration: 1.1, ease: 'easeOut' }} />
      <CopyButton text={`FRAME prediction: ${r.label} (average class probability: ${[...CLASSES].reverse().map((k) => `${k} ${fmtPct(r.probabilities[k], 0)}`).join(', ')}). Inputs: ${Object.entries(r.inputs).map(([k, v]) => `${k}=${String(v).replaceAll('|', '/')}`).join(', ')}. Model: ${r.model.name} (${r.model.source}). Estimate only, not a guarantee. ${metrics ? `On the held-out test set the model found ${metrics.held_out.confusion_matrix[0][0]} of ${metrics.held_out.per_class.Flop.support} Flops.` : ''}`} />
      <p className="eyebrow text-center">The model says</p>
      <p className="mt-2 text-center font-display text-[5rem] font-extrabold leading-none sm:text-[6.5rem]" style={{ color: c, textShadow: `0 0 50px ${c}aa, 0 0 12px ${c}66` }} data-testid="predicted-label" aria-label={r.label}>
        {r.label.split('').map((ch, i) => (
          <motion.span key={i} aria-hidden className="inline-block" initial={{ opacity: 0, y: 36, filter: 'blur(8px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            transition={{ delay: 0.08 * i + 0.1, duration: 0.55, ease: [0.16, 1, 0.3, 1] }}>{ch}</motion.span>
        ))}
      </p>
      <p className="mx-auto mt-3 max-w-sm text-center text-sm text-fg2">{TAGLINE[r.label]}</p>
      {actual && (
        <p className="mt-4 flex flex-wrap items-center justify-center gap-2 text-sm text-fg2">Actual in dataset: <ClassBadge value={actual.cls} /> <span className="font-mono text-fg">IMDb {actual.score.toFixed(1)}</span>
          <span className={cx('font-semibold', actual.cls === r.label ? 'text-hit' : 'text-danger')}>{actual.cls === r.label ? '· model agrees' : '· model disagrees'}</span>
          <span className="basis-full text-center text-[0.72rem] leading-snug text-fg3">About 80% of dataset films were in the training split, so agreement here is not a fair out-of-sample test.</span></p>
      )}

      <h3 className="mt-10 font-display text-[0.95rem] font-bold">Class scores from the model</h3>
      <ul className="mt-3 space-y-3.5">
        {shown.map((k) => (
          <li key={k} className="grid grid-cols-[4.5rem_1fr_3.4rem] items-center gap-3 text-sm">
            <span className="font-medium" style={{ color: CLASS_COLOR[k] }}>{k}</span>
            <span className="h-3 overflow-hidden rounded-full bg-white/[0.08]" role="img" aria-label={`${k} ${fmtPct(r.probabilities[k])}`}>
              <motion.span className="block h-full rounded-full" style={{ background: `linear-gradient(90deg, color-mix(in srgb, ${CLASS_COLOR[k]} 55%, #06061a), ${CLASS_COLOR[k]})`, boxShadow: `0 0 16px ${CLASS_COLOR[k]}99` }}
                initial={{ width: 0 }} animate={{ width: `${Math.max(r.probabilities[k] * 100, 0.8)}%` }} transition={{ duration: 0.9, delay: 0.5, ease: [0.16, 1, 0.3, 1] }} />
            </span>
            <span className="text-right font-mono text-[0.85rem] text-fg"><Counter value={r.probabilities[k] * 100} suffix="%" duration={1} /></span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[0.75rem] leading-snug text-fg3">Each score is the model’s estimated probability for that class — uncalibrated, so a ranking rather than the real-world chance of this outcome.</p>

      {r.warnings.length > 0 && <ul className="mt-4 space-y-2">{r.warnings.map((w) => <li key={w.message} className="rounded-2xl border border-avg/40 bg-avg/10 px-3.5 py-2 text-[0.8rem] text-fg2">{w.message}</li>)}</ul>}

      {drivers && (
        <div className="mt-6 rounded-2xl border border-line bg-black/20 p-4">
          <h3 className="font-display text-[0.95rem] font-bold">What the model leans on for <span style={{ color: c }}>{r.label}</span></h3>
          <p className="mt-0.5 text-[0.72rem] text-fg3">Global SHAP drivers over 200 test films — not an explanation of this one film.</p>
          <ol className="mt-3 space-y-1.5">
            {drivers.map((d, i) => (
              <li key={d.feature} className="grid grid-cols-[1fr_5rem] items-center gap-3 text-[0.8rem]">
                <span className="truncate font-mono text-fg2">{i + 1}. {d.feature.replace(/_/g, ' ')}</span>
                <span className="h-1.5 overflow-hidden rounded-full bg-white/[0.08]"><motion.span className="block h-full rounded-full" style={{ background: c }} initial={{ width: 0 }} animate={{ width: `${(d.mean_abs_shap / drivers[0].mean_abs_shap) * 100}%` }} transition={{ duration: 0.8, delay: 0.7 + i * 0.07 }} /></span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {body && <Suspense fallback={<InsightFallback />}><WhatIf body={body} /></Suspense>}
      {body && <Suspense fallback={<InsightFallback />}><CohortLens compact genres={(body.genres as string[]) ?? []} country={String(body.country ?? '')} contentRating={String(body.content_rating ?? '')} year={Number(body.title_year) || null} duration={Number(body.duration) || null} /></Suspense>}

      <div className="mt-5 border-t border-line pt-3">
        <button type="button" onClick={() => setOpenIn((o) => !o)} aria-expanded={openIn} className="flex min-h-10 w-full items-center justify-between text-left text-[0.88rem] font-semibold">
          Inputs the model received <ChevronDown size={16} aria-hidden className={cx('text-fg3 transition-transform', openIn && 'rotate-180')} />
        </button>
        {openIn && (
          <div className="mt-2 space-y-4">
            <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-[0.8rem]">
              {Object.entries(r.inputs).map(([k, v]) => <div key={k} className="contents"><dt className="font-mono text-fg3">{k}</dt><dd className="tnum text-right font-mono text-fg">{typeof v === 'number' ? fmtInt(v) : String(v).replaceAll('|', ' / ')}</dd></div>)}
            </dl>
            {r.imputed.length > 0 && (
              <div>
                <p className="mb-1.5 text-[0.8rem] font-semibold text-avg">Filled in by the pipeline ({r.imputed.length})</p>
                <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-[0.8rem]">
                  {r.imputed.map((i) => <div key={i.field} className="contents"><dt className="font-mono text-fg3">{i.field} <span className="text-[0.72rem]">({i.rule})</span></dt><dd className="tnum text-right font-mono text-fg2">{typeof i.value === 'number' ? fmtInt(i.value) : i.value}</dd></div>)}
                </dl>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="mt-5"><Note title="An estimate, not a guarantee"><ul className="list-disc space-y-1 pl-4">{r.caveats.map((x) => <li key={x}>{x}</li>)}</ul></Note></div>
    </motion.div>
  )
}

function HistoryList({ items, onRestore, onClear }: { items: HistoryItem[]; onRestore: (h: HistoryItem) => void; onClear: () => void }) {
  if (!items.length) return null
  return (
    <Panel title="Session history" subtitle="Stored in this browser tab only; cleared when the tab closes. Select an entry to reload its inputs."
      actions={<Button variant="quiet" className="!min-h-9 px-2 text-xs" icon={<Trash2 size={14} aria-hidden />} onClick={onClear}>Clear</Button>} bodyClass="pt-3">
      <ul className="divide-y divide-line">
        {items.map((h) => (
          <li key={h.id}>
            <button type="button" onClick={() => onRestore(h)} className="flex min-h-12 w-full items-center gap-3 py-2 text-left hover:bg-s2/60">
              <History size={15} aria-hidden className="shrink-0 text-fg3" />
              <span className="min-w-0 flex-1"><span className="block truncate text-[0.82rem] text-fg">{h.summary}</span><span className="font-mono text-[0.72rem] text-fg3">{h.at}{h.actual ? ` · actual ${h.actual.cls}` : ''}</span></span>
              <ClassBadge value={h.label} />
            </button>
          </li>
        ))}
      </ul>
    </Panel>
  )
}


function CopyButton({ text }: { text: string }) {
  const [state, setState] = useState<'idle' | 'ok' | 'fail'>('idle')
  async function copy() {
    try { await navigator.clipboard.writeText(text); setState('ok') } catch { setState('fail') }
    setTimeout(() => setState('idle'), 2200)
  }
  return (
    <button type="button" onClick={copy} className="absolute right-4 top-4 z-10 flex min-h-9 items-center gap-1.5 rounded-full border border-line-strong bg-black/30 px-3 text-[0.72rem] font-semibold text-fg2 transition-all duration-200 hover:border-white/50 hover:text-fg active:scale-95" aria-live="polite">
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={state} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: 0.15 }} className="flex items-center gap-1.5">
          {state === 'ok' ? <><Check size={13} className="text-hit" aria-hidden />Copied</> : state === 'fail' ? <>Copy blocked by browser</> : <><ClipboardCopy size={13} aria-hidden />Copy summary</>}
        </motion.span>
      </AnimatePresence>
    </button>
  )
}
