import { useId, useMemo, type ReactNode } from 'react'
import { numericFields, CATEGORICAL, type FormValues } from '../../lib/filmForm'
import type { PredictSchema } from '../../lib/types'
import { Panel, cx } from '../ui'

const inputCls = (err?: string) => cx('min-h-11 w-full rounded-2xl border bg-black/30 px-4 text-sm text-fg transition-[border-color,box-shadow] duration-200 placeholder:text-fg3/70 hover:border-white/40 focus:outline-none focus:border-cyan focus:shadow-[0_0_0_4px_rgb(37_232_211/.16)] disabled:opacity-60 sm:min-h-10', err ? 'border-flop' : 'border-line-strong')

function Field({ id, label, required, unit, help, error, children }: { id: string; label: string; required?: boolean; unit?: string; help?: string; error?: string; children: ReactNode }) {
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-[0.82rem] font-medium text-fg">
        {label}{unit && <span className="ml-1.5 font-mono text-[0.72rem] font-normal text-fg3">{unit}</span>}{required && <span className="ml-1 text-pink" aria-label="required">*</span>}
      </label>
      {children}
      <p id={`${id}-d`} className={cx('mt-1.5 text-[0.75rem] leading-snug', error ? 'text-danger' : 'text-fg3')}>{error ?? help}</p>
    </div>
  )
}

export function FilmForm({ schema, values, genres, errors, disabled, onValue, onGenres }: {
  schema: PredictSchema; values: FormValues; genres: string[]; errors: Record<string, string>; disabled: boolean
  onValue: (key: string, v: string) => void; onGenres: (g: string[]) => void
}) {
  const gid = useId()
  const nums = useMemo(() => numericFields(schema), [schema])
  const core = nums.filter((f) => f.required), optional = nums.filter((f) => !f.required)
  const toggle = (g: string) => onGenres(genres.includes(g) ? genres.filter((x) => x !== g) : genres.length >= 6 ? genres : [...genres, g])
  const num = (f: (typeof nums)[number]) => {
    const id = `r-${f.key}`
    return (
      <Field key={f.key} id={id} label={f.label} unit={f.unit} required={f.required} help={f.help} error={errors[f.key]}>
        <input id={id} inputMode={f.int ? 'numeric' : 'decimal'} autoComplete="off" disabled={disabled} value={values[f.key] ?? ''} onChange={(e) => onValue(f.key, e.target.value)}
          placeholder={f.required ? '' : 'optional'} aria-invalid={!!errors[f.key]} aria-describedby={`${id}-d`} className={cx(inputCls(errors[f.key]), 'tnum font-mono')} />
      </Field>
    )
  }
  return (
    <>
      <Panel title="01 · The film" subtitle="Only what is known before release." tone="#ff4fa3">
        <div className="grid gap-4 sm:grid-cols-2">
          {core.map(num)}
          <div className="sm:col-span-2">
            <fieldset id="r-genres" tabIndex={-1} aria-describedby={`${gid}-d`} disabled={disabled}>
              <legend className="mb-1.5 text-[0.82rem] font-medium text-fg">Genres<span className="ml-1 text-pink" aria-label="required">*</span> <span className="font-normal text-fg3">pick up to 6</span></legend>
              <div className="flex flex-wrap gap-1.5">
                {schema.genres.map((g) => {
                  const on = genres.includes(g)
                  return <button key={g} type="button" aria-pressed={on} onClick={() => toggle(g)}
                    className={cx('min-h-9 rounded-full border px-3.5 text-[0.8rem] font-medium transition-all duration-200 active:scale-95 disabled:opacity-60', on ? 'border-transparent bg-[image:var(--grad-brand)] font-bold text-[#12082a]' : 'border-line-strong text-fg2 hover:border-white/50 hover:text-fg')}>{g}</button>
                })}
              </div>
              <p id={`${gid}-d`} className={cx('mt-2 text-[0.75rem] leading-snug', errors.genres ? 'text-danger' : 'text-fg3')}>{errors.genres ?? 'Each genre is a separate flag, so any combination works.'}</p>
            </fieldset>
          </div>
          {CATEGORICAL.map((c) => {
            const id = `r-${c.key}`
            return (
              <Field key={c.key} id={id} label={c.label} required={c.required} help={c.required ? undefined : 'Blank uses its own “Unknown” category.'} error={errors[c.key]}>
                <select id={id} disabled={disabled} value={values[c.key] ?? ''} onChange={(e) => onValue(c.key, e.target.value)} aria-invalid={!!errors[c.key]} aria-describedby={`${id}-d`} className={inputCls(errors[c.key])}>
                  <option value="">{c.required ? 'Select…' : 'Not specified'}</option>
                  {(schema.categorical[c.key] ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              </Field>
            )
          })}
        </div>
      </Panel>
      {optional.length > 0 && (
        <Panel title="02 · Production" subtitle="Optional. Left blank, the pipeline substitutes the training median." tone="#ffb23e">
          <div className="grid gap-4 sm:grid-cols-2">{optional.map(num)}</div>
        </Panel>
      )}
      <Panel title="03 · Research title (optional)" subtitle="Used only to look up Wikipedia background as it stood before release. Never given to the model." tone="#25e8d3">
        <Field id="r-research_title" label="Film title" help="Leave blank to research the genres only. Only an exact title match is used." error={errors.research_title}>
          <input id="r-research_title" autoComplete="off" maxLength={120} disabled={disabled} value={values.research_title ?? ''} onChange={(e) => onValue('research_title', e.target.value)}
            aria-invalid={!!errors.research_title} aria-describedby="r-research_title-d" className={inputCls(errors.research_title)} />
        </Field>
      </Panel>
    </>
  )
}
