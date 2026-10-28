import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, Download, ExternalLink, FilterX, Search, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Button, ClassBadge, EmptyState, ErrorState, LoadingBlock, PageHeader, Segmented, cx } from '../components/ui'
import { apiUrl, qs } from '../lib/api'
import { CLASS_COLOR, CLASSES, fmtInt, fmtUSD } from '../lib/format'
import type { FilterOptions, MovieDetail, MovieList, MovieRow } from '../lib/types'
import { useApi } from '../lib/useApi'

const sel = 'min-h-11 w-full rounded-full border border-line-strong bg-black/25 px-4 text-sm text-fg transition-colors hover:border-white/40 focus:border-cyan sm:min-h-10'
type SortKey = 'title' | 'year' | 'imdb_score' | 'duration' | 'budget' | 'gross' | 'num_voted_users'

export default function Explore() {
  const opts = useApi<FilterOptions>('/api/filters', { cache: true })
  const [sp] = useSearchParams()
  const [q, setQ] = useState(sp.get('q') ?? '')
  const [dq, setDq] = useState((sp.get('q') ?? '').trim())
  const [f, setF] = useState({ genre: '', country: '', content_rating: '', classify: '', year_min: '', year_max: '' })
  const [sort, setSort] = useState<SortKey>('imdb_score')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')
  const [page, setPage] = useState(1)
  const [openId, setOpenId] = useState<number | null>(null)
  const [view, setView] = useState<'cards' | 'table'>('cards')

  useEffect(() => { const t = setTimeout(() => { setDq(q.trim()); setPage(1) }, 300); return () => clearTimeout(t) }, [q])

  const filterParams = { ...f, q: dq }
  const list = useApi<MovieList>(`/api/movies${qs({ ...filterParams, sort, order, page, page_size: 25 })}`)
  const d = list.data
  const active = Object.values(f).some(Boolean) || !!dq
  const setFilter = (k: keyof typeof f, v: string) => { setF((s) => ({ ...s, [k]: v })); setPage(1) }
  const reset = () => { setF({ genre: '', country: '', content_rating: '', classify: '', year_min: '', year_max: '' }); setQ(''); setDq(''); setPage(1) }
  const sortBy = (k: SortKey) => { if (k === sort) setOrder((o) => (o === 'asc' ? 'desc' : 'asc')); else { setSort(k); setOrder(k === 'title' ? 'asc' : 'desc') } setPage(1) }
  const years = opts.data ? Array.from({ length: opts.data.year_max - opts.data.year_min + 1 }, (_, i) => opts.data!.year_min + i) : []

  return (
    <>
      <PageHeader eyebrow="Explore movies" title="Browse the whole archive" highlight="whole archive"
        actions={<a href={apiUrl(`/api/movies/export.csv${qs(filterParams)}`)} download className="inline-flex min-h-11 items-center gap-2 rounded-full border border-line-strong bg-white/[0.04] px-5 text-sm font-semibold hover:bg-white/10 sm:min-h-10"><Download size={15} aria-hidden />Download filtered CSV</a>}>
        Every row is a real record from the IMDb 5000 file (duplicates removed). Search by title, director or actor; filter; sort; open a film for its full metadata.
      </PageHeader>

      <section aria-label="Search and filters" className="glass mb-5 space-y-3 rounded-[1.75rem] p-4 sm:p-5">
        <label className="relative block">
          <span className="sr-only">Search titles, directors and actors</span>
          <Search size={17} aria-hidden className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-fg3" />
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search a title, director or actor…" className={cx(sel, '!min-h-12 pl-11 text-base')} />
        </label>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-[repeat(6,1fr)_auto] xl:items-end">
          <Sel label="Genre" v={f.genre} on={(v) => setFilter('genre', v)} all="All genres" items={opts.data?.genres} />
          <Sel label="Country" v={f.country} on={(v) => setFilter('country', v)} all="All countries" items={opts.data?.countries} />
          <Sel label="Content rating" v={f.content_rating} on={(v) => setFilter('content_rating', v)} all="Any rating" items={opts.data?.content_ratings} />
          <label className="block"><span className="eyebrow mb-1.5 block">Class</span>
            <select className={sel} value={f.classify} onChange={(e) => setFilter('classify', e.target.value)}><option value="">All classes</option>{[...CLASSES].reverse().map((c) => <option key={c}>{c}</option>)}</select></label>
          <label className="block"><span className="eyebrow mb-1.5 block">From year</span>
            <select className={sel} value={f.year_min} onChange={(e) => setFilter('year_min', e.target.value)}><option value="">Any</option>{years.map((y) => <option key={y}>{y}</option>)}</select></label>
          <label className="block"><span className="eyebrow mb-1.5 block">To year</span>
            <select className={sel} value={f.year_max} onChange={(e) => setFilter('year_max', e.target.value)}><option value="">Any</option>{years.map((y) => <option key={y}>{y}</option>)}</select></label>
          <Button className="col-span-2 md:col-span-1" disabled={!active} onClick={reset} icon={<FilterX size={15} aria-hidden />}>Reset</Button>
        </div>
      </section>

      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <Segmented label="View" value={view} onChange={setView} options={[{ value: 'cards', label: 'Cards' }, { value: 'table', label: 'Table' }]} />
        {view === 'cards' && (
          <label className="flex items-center gap-2 text-[0.78rem] text-fg3">Sort
            <select className={cx(sel, '!min-h-10 w-auto')} value={`${sort}:${order}`} onChange={(e) => { const [k, o] = e.target.value.split(':'); setSort(k as SortKey); setOrder(o as 'asc' | 'desc'); setPage(1) }}>
              <option value="imdb_score:desc">IMDb score ↓</option><option value="imdb_score:asc">IMDb score ↑</option><option value="year:desc">Newest</option><option value="year:asc">Oldest</option>
              <option value="num_voted_users:desc">Most votes</option><option value="gross:desc">Highest gross</option><option value="title:asc">Title A–Z</option>
            </select></label>
        )}
      </div>
      <p className="mb-3 text-[0.8rem] text-fg3" aria-live="polite">
        {d ? <><b className="tnum font-medium text-fg">{fmtInt(d.total)}</b> film{d.total === 1 ? '' : 's'}{active ? ' match' : ''} · page {d.page} of {fmtInt(d.pages)}</> : 'Loading…'}
      </p>

      {list.error && !d && <ErrorState title="Couldn’t load movies" message={list.error.message} onRetry={list.retry} />}
      {!d && !list.error && <LoadingBlock height="h-[28rem]" label="Loading movies" />}
      {d && d.total === 0 && <EmptyState title="No films match" message="Nothing in the dataset fits that search and filter combination." action={<Button onClick={reset}>Clear search and filters</Button>} />}
      {d && d.total > 0 && (
        <div className={cx('transition-opacity duration-150', list.loading && 'opacity-60')} aria-busy={list.loading}>
          {view === 'table' && <Table rows={d.rows} sort={sort} order={order} onSort={sortBy} onOpen={setOpenId} />}
          {view === 'cards' && <CardGrid rows={d.rows} onOpen={setOpenId} />}
          <nav aria-label="Pagination" className="mt-4 flex items-center justify-between gap-3">
            <Button disabled={page <= 1} onClick={() => setPage((p) => p - 1)} icon={<ChevronLeft size={16} aria-hidden />}>Previous</Button>
            <span className="tnum font-mono text-[0.8rem] text-fg3">{(d.page - 1) * d.page_size + 1}–{Math.min(d.page * d.page_size, d.total)} of {fmtInt(d.total)}</span>
            <Button disabled={page >= d.pages} onClick={() => setPage((p) => p + 1)}>Next<ChevronRight size={16} aria-hidden /></Button>
          </nav>
        </div>
      )}

      <AnimatePresence>{openId !== null && <Detail id={openId} onClose={() => setOpenId(null)} />}</AnimatePresence>
    </>
  )
}

function Sel({ label, v, on, all, items }: { label: string; v: string; on: (v: string) => void; all: string; items?: { value: string; count: number }[] }) {
  return (
    <label className="block"><span className="eyebrow mb-1.5 block">{label}</span>
      <select className={sel} value={v} onChange={(e) => on(e.target.value)}><option value="">{all}</option>{items?.map((i) => <option key={i.value} value={i.value}>{i.value} ({fmtInt(i.count)})</option>)}</select></label>
  )
}

function Table({ rows, sort, order, onSort, onOpen }: { rows: MovieRow[]; sort: SortKey; order: 'asc' | 'desc'; onSort: (k: SortKey) => void; onOpen: (id: number) => void }) {
  const head = (k: SortKey, label: string, right?: boolean) => (
    <th scope="col" aria-sort={sort === k ? (order === 'asc' ? 'ascending' : 'descending') : 'none'} className={cx('px-3 py-2 font-medium', right ? 'text-right' : 'text-left')}>
      <button onClick={() => onSort(k)} className={cx('inline-flex min-h-9 items-center gap-1 rounded-xs uppercase tracking-wider hover:text-fg', sort === k ? 'text-accent' : 'text-fg3')}>
        {label}{sort === k && (order === 'asc' ? <ArrowUp size={12} aria-hidden /> : <ArrowDown size={12} aria-hidden />)}</button>
    </th>
  )
  return (
    <div className="glass relative overflow-x-auto rounded-[1.5rem]">
      <table className="w-full min-w-[56rem] text-sm">
        <caption className="sr-only">Movies</caption>
        <thead className="border-b border-line font-mono text-[0.72rem]"><tr>
          {head('title', 'Title')}{head('year', 'Year', true)}<th scope="col" className="px-3 py-2 text-left font-mono text-[0.72rem] font-medium uppercase tracking-wider text-fg3">Genres</th>
          {head('imdb_score', 'IMDb', true)}<th scope="col" className="px-3 py-2 text-left font-medium uppercase tracking-wider text-fg3">Class</th>
          {head('duration', 'Min', true)}{head('budget', 'Budget', true)}{head('gross', 'Gross', true)}{head('num_voted_users', 'Votes', true)}
        </tr></thead>
        <tbody className="divide-y divide-line">
          {rows.map((r) => (
            <tr key={r.id} className="group hover:bg-s2/70">
              <td className="max-w-[18rem] px-3 py-0">
                <button onClick={() => onOpen(r.id)} className="block min-h-11 w-full truncate rounded-xs text-left font-medium text-fg group-hover:text-accent" title={r.title}>{r.title}</button>
              </td>
              <td className="tnum px-3 text-right font-mono text-[0.8rem] text-fg2">{r.year ?? '—'}</td>
              <td className="max-w-[13rem] truncate px-3 text-[0.8rem] text-fg3" title={r.genres?.replaceAll('|', ' · ')}>{r.genres?.replaceAll('|', ' · ') ?? '—'}</td>
              <td className="tnum px-3 text-right font-mono text-fg">{r.imdb_score.toFixed(1)}</td>
              <td className="px-3"><ClassBadge value={r.Classify} /></td>
              <td className="tnum px-3 text-right font-mono text-[0.8rem] text-fg2">{r.duration ?? '—'}</td>
              <td className="tnum px-3 text-right font-mono text-[0.8rem] text-fg2" title={r.country && r.country !== 'USA' ? 'Local currency — not comparable to USD' : undefined}>{r.country === 'USA' ? fmtUSD(r.budget) : r.budget != null ? <span className="text-fg3">local</span> : '—'}</td>
              <td className="tnum px-3 text-right font-mono text-[0.8rem] text-fg2">{fmtUSD(r.gross)}</td>
              <td className="tnum px-3 text-right font-mono text-[0.8rem] text-fg2">{fmtInt(r.num_voted_users)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* Record cards: a class-coloured rule and a faint genre tint, no invented artwork. Fixed height keeps the grid even. */
const hueOf = (g: string | null) => { const k = (g ?? '').split('|')[0]; let h = 0; for (const ch of k) h = (h * 31 + ch.charCodeAt(0)) % 360; return h }

function CardGrid({ rows, onOpen }: { rows: MovieRow[]; onOpen: (id: number) => void }) {
  return (
    <ul className="grid grid-cols-1 gap-3 min-[560px]:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {rows.map((r) => {
        const c = CLASS_COLOR[r.Classify]; const h = hueOf(r.genres)
        return (
          <li key={r.id}>
            <button onClick={() => onOpen(r.id)}
              className="group relative flex h-[8.75rem] w-full flex-col justify-between overflow-hidden rounded-2xl border border-line p-4 pl-5 text-left transition-[transform,border-color] duration-200 hover:-translate-y-0.5 hover:border-line-strong active:scale-[0.99] motion-reduce:transition-none motion-reduce:hover:translate-y-0"
              style={{ background: `linear-gradient(135deg, hsl(${h} 45% 18% / .75), rgb(14 12 44 / .7) 70%)` }}>
              <span aria-hidden className="absolute inset-y-0 left-0 w-1" style={{ background: c }} />
              <div className="flex items-start justify-between gap-3">
                <h2 className="line-clamp-2 min-w-0 font-display text-[1.05rem] font-bold leading-snug text-fg">{r.title}</h2>
                <span className="font-display text-[1.75rem] font-extrabold leading-none tabular-nums" style={{ color: c }}>{r.imdb_score.toFixed(1)}</span>
              </div>
              <div className="min-w-0">
                <p className="truncate text-[0.8rem] text-fg2">{[r.year, r.country, r.duration ? `${r.duration} min` : null].filter(Boolean).join(' · ')}</p>
                <div className="mt-1.5 flex items-center justify-between gap-2">
                  <p className="min-w-0 truncate font-mono text-[0.72rem] uppercase tracking-wide text-fg3">{r.genres?.replaceAll('|', ' · ') ?? '—'}</p>
                  <ClassBadge value={r.Classify} />
                </div>
              </div>
            </button>
          </li>
        )
      })}
    </ul>
  )
}

function Detail({ id, onClose }: { id: number; onClose: () => void }) {
  const { data, error, retry } = useApi<MovieDetail>(`/api/movies/${id}`, { cache: true })
  const nav = useNavigate()
  const closeRef = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = ''; prev?.focus?.() }
  }, [onClose])
  const dash = (v: unknown) => (v == null || v === '' ? '—' : String(v))
  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label={data ? `Details for ${data.title}` : 'Movie details'}>
      <motion.div className="absolute inset-0 bg-black/75 backdrop-blur-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
      <motion.aside className="glass absolute inset-y-0 right-0 flex w-full max-w-md flex-col overflow-y-auto sm:inset-y-3 sm:right-3 sm:w-[28rem] sm:rounded-[2rem]"
        initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}>
        <div className="flex items-center justify-between border-b border-line px-5 py-3"><span className="eyebrow">Film record</span>
          <button ref={closeRef} onClick={onClose} aria-label="Close details" className="flex h-11 w-11 items-center justify-center rounded-sm text-fg2 hover:bg-s2"><X size={20} aria-hidden /></button></div>
        {error && <div className="p-5"><ErrorState message={error.message} onRetry={retry} /></div>}
        {!data && !error && <div className="p-5"><LoadingBlock height="h-80" /></div>}
        {data && (
          <div className="space-y-5 p-5">
            <div>
              <h2 className="font-display text-3xl font-extrabold leading-tight text-fg">{data.title}</h2>
              <p className="mt-2 text-sm text-fg3">{[data.year, data.country, data.language, data.content_rating, data.duration ? `${data.duration} min` : null].filter(Boolean).join(' · ')}</p>
              <div className="mt-3 flex items-center gap-3"><ClassBadge size="lg" value={data.Classify} /><span className="tnum font-mono text-xl text-fg">IMDb {data.imdb_score.toFixed(1)}</span></div>
            </div>
            <div className="flex flex-wrap gap-1.5">{data.genre_list.map((g) => <span key={g} className="rounded-full border border-line-strong bg-white/[0.04] px-3 py-1 text-[0.75rem] text-fg2">{g}</span>)}</div>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-4 border-t border-line pt-4 text-sm">
              <Meta k="Director" v={dash(data.director_name)} /><Meta k="Lead actors" v={[data.actor_1_name, data.actor_2_name, data.actor_3_name].filter(Boolean).join(', ') || '—'} />
              <Meta k="Budget" v={data.country === 'USA' ? fmtUSD(data.budget) : data.budget != null ? `${fmtInt(data.budget)} (local currency)` : '—'} /><Meta k="Gross" v={fmtUSD(data.gross)} />
              <Meta k="IMDb votes" v={fmtInt(data.num_voted_users)} /><Meta k="Reviews" v={`${fmtInt(data.num_critic_for_reviews)} critic · ${fmtInt(data.num_user_for_reviews)} user`} />
              <Meta k="Movie page likes" v={fmtInt(data.movie_facebook_likes)} /><Meta k="Cast likes" v={fmtInt(data.cast_total_facebook_likes)} />
            </dl>
            {data.budget_currency_note && <p className="text-[0.75rem] text-fg3">{data.budget_currency_note}</p>}
            {data.keywords.length > 0 && <div><p className="eyebrow mb-2">Plot keywords</p><p className="text-[0.85rem] text-fg2">{data.keywords.join(' · ')}</p></div>}
            <div className="flex flex-wrap gap-2 border-t border-line pt-4">
              <Button variant="primary" onClick={() => nav('/predict', { state: { movie: data } })}>Use in predictor</Button>
              {data.movie_imdb_link && <a href={data.movie_imdb_link} target="_blank" rel="noreferrer noopener" className="inline-flex min-h-11 items-center gap-2 rounded-full border border-line-strong px-5 text-sm font-semibold hover:bg-white/10 sm:min-h-10">IMDb<ExternalLink size={14} aria-hidden /></a>}
            </div>
            <p className="text-[0.75rem] leading-snug text-fg3">“Use in predictor” fills the form with this record’s real values and shows its actual class beside the model’s answer. This film may have been in the training data, so agreement is not an out-of-sample test.</p>
          </div>
        )}
      </motion.aside>
    </div>
  )
}
const Meta = ({ k, v }: { k: string; v: string }) => <div><dt className="eyebrow mb-1">{k}</dt><dd className="tnum text-fg">{v}</dd></div>
