export const fmtInt = (n: number | null | undefined) => (n == null ? '—' : Math.round(n).toLocaleString('en-US'))
export const fmtPct = (x: number | null | undefined, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`)
export const fmtScore = (x: number | null | undefined) => (x == null ? '—' : x.toFixed(1))
export function fmtUSD(n: number | null | undefined): string {
  if (n == null) return '—'
  const a = Math.abs(n)
  if (a >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
  if (a >= 1e6) return `$${(n / 1e6).toFixed(a >= 1e8 ? 0 : 1)}M`
  if (a >= 1e3) return `$${(n / 1e3).toFixed(0)}K`
  return `$${n.toFixed(0)}`
}
export const CLASS_COLOR = { Flop: 'var(--color-flop)', Average: 'var(--color-avg)', Hit: 'var(--color-hit)' } as const
export type MovieClass = keyof typeof CLASS_COLOR
export const CLASSES: MovieClass[] = ['Flop', 'Average', 'Hit']
export const CLASS_RANGE: Record<MovieClass, string> = { Flop: 'IMDb 0 – <3', Average: 'IMDb 3 – <6', Hit: 'IMDb 6 – 10' }
