import type { PredictSchema } from './types'

/* The request body is POST /api/predict's body (plus one optional research-only title), so the server (PredictRequest, extra="forbid") stays the single authority:
   anything this form gets wrong comes back as a 422 with per-field messages, which the page shows against the matching input.
   Field lists come from GET /api/predict/schema; the labels and the bounds below are display hints only. */

export interface NumMeta { label: string; unit?: string; help: string; min?: number; max?: number; int?: boolean; required?: boolean; step?: string }
const NUM_META: Record<string, NumMeta> = {
  title_year: { label: 'Release year', help: 'Training data spans 1916–2016.', min: 1888, max: 2030, int: true, required: true },
  duration: { label: 'Duration', unit: 'min', help: 'Runtime in minutes.', min: 1, max: 600, required: true },
  budget: { label: 'Budget', unit: 'USD', help: 'Optional. US films are in dollars; other countries are local currency in the training data.', min: 0, max: 10_000_000_000, int: true },
  aspect_ratio: { label: 'Aspect ratio', help: 'Optional, e.g. 2.35 or 1.85.', min: 0.5, max: 10, step: '0.01' },
}
export const CATEGORICAL = [
  { key: 'country', label: 'Country', required: true }, { key: 'content_rating', label: 'Content rating', required: true },
  { key: 'language', label: 'Language' }, { key: 'color', label: 'Colour' },
]

const int = (n: number) => Math.round(n).toLocaleString('en-US')

export type FormValues = Record<string, string>
export const emptyValues = (): FormValues => ({ country: '', content_rating: '', language: '', color: '', research_title: '' })
/** Mirrors backend/agent/schemas.py: letters, numbers and ordinary punctuation, 2 to 120 characters. */
export const TITLE_OK = /^[\p{L}\p{N}_\s.,:;'’&!?()\-–—/]+$/u

/** Same order as the Predictor page (year, runtime, then optional fields); fields the schema adds later keep their schema order after these. */
const FIELD_ORDER = ['title_year', 'duration', 'budget', 'aspect_ratio']
const rank = (k: string) => { const i = FIELD_ORDER.indexOf(k); return i < 0 ? FIELD_ORDER.length : i }
export const numericFields = (schema: PredictSchema) => schema.numeric.map((key) => ({ key, ...(NUM_META[key] ?? { label: key.replaceAll('_', ' '), help: '' }) })).sort((a, b) => rank(a.key) - rank(b.key))

/** Client-side hints only. Returns field -> message. */
export function validateForm(schema: PredictSchema, values: FormValues, genres: string[]): Record<string, string> {
  const e: Record<string, string> = {}
  for (const f of numericFields(schema)) {
    const raw = values[f.key]?.trim()
    if (!raw) { if (f.required) e[f.key] = `${f.label} is required.`; continue }
    const n = Number(raw)
    if (!Number.isFinite(n)) e[f.key] = 'Enter a number.'
    else if (f.int && !Number.isInteger(n)) e[f.key] = 'Use a whole number.'
    else if ((f.min !== undefined && n < f.min) || (f.max !== undefined && n > f.max)) e[f.key] = `Must be between ${int(f.min ?? 0)} and ${int(f.max ?? 0)}.`
  }
  const title = values.research_title?.trim()
  if (title && (title.length < 2 || title.length > 120 || !TITLE_OK.test(title))) e.research_title = 'Use 2 to 120 letters, numbers or ordinary punctuation.'
  if (!genres.length) e.genres = 'Select at least one genre.'
  for (const c of CATEGORICAL) if (c.required && !values[c.key]) e[c.key] = `${c.label} is required.`
  return e
}

export function buildBody(schema: PredictSchema, values: FormValues, genres: string[]): Record<string, unknown> {
  const body: Record<string, unknown> = { genres }
  for (const f of numericFields(schema)) if (values[f.key]?.trim()) body[f.key] = Number(values[f.key])
  for (const c of CATEGORICAL) if (values[c.key]) body[c.key] = values[c.key]
  // research-only field: the server strips it before prediction, so it can never reach the model
  const title = values.research_title?.replace(/\s+/g, ' ').trim()
  if (title) body.research_title = title
  return body
}

