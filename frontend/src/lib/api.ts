const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? ''

export interface FieldProblem { field: string; message: string }
export class ApiError extends Error {
  status: number
  code: string
  fields: FieldProblem[]
  /** Seconds from the Retry-After response header, when the server sent one. */
  retryAfter?: number
  constructor(status: number, code: string, message: string, fields: FieldProblem[] = [], retryAfter?: number) {
    super(message)
    this.status = status
    this.code = code
    this.fields = fields
    this.retryAfter = retryAfter
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...(init?.headers as Record<string, string> | undefined) },
    })
  } catch {
    throw new ApiError(0, 'network', 'Cannot reach the FRAME API. Check that the backend is running.')
  }
  if (!res.ok) {
    let body: { error?: string; message?: string; fields?: FieldProblem[] } = {}
    try { body = await res.json() } catch { /* non-JSON error body */ }
    const ra = Number(res.headers.get('Retry-After'))
    throw new ApiError(res.status, body.error ?? 'http_error', body.message ?? `Request failed (${res.status}).`, body.fields, Number.isFinite(ra) && ra > 0 ? ra : undefined)
  }
  return res.json() as Promise<T>
}

/** Concurrent identical GETs share one request (effects can fire twice on mount); each settles and is forgotten, so retries refetch. */
const inflight = new Map<string, Promise<unknown>>()
function dedupedGet<T>(path: string): Promise<T> {
  const hit = inflight.get(path)
  if (hit) return hit as Promise<T>
  const p = request<T>(path).finally(() => inflight.delete(path))
  inflight.set(path, p)
  return p
}

export const api = {
  get: <T,>(path: string) => dedupedGet<T>(path),
  post: <T,>(path: string, body: unknown, headers?: Record<string, string>) => request<T>(path, { method: 'POST', body: JSON.stringify(body), headers }),
}

export const qs = (params: Record<string, string | number | null | undefined>) => {
  const u = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== '') u.set(k, String(v))
  const s = u.toString()
  return s ? `?${s}` : ''
}
export const apiUrl = (path: string) => `${BASE}${path}`
