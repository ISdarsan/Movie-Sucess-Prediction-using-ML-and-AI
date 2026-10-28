import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from './api'

interface State<T> { data: T | null; error: ApiError | null; loading: boolean }

const cache = new Map<string, unknown>()

/** GET hook: keeps previous data visible while a refetch runs (so filters feel instant); optional per-URL cache. */
export function useApi<T>(path: string | null, opts: { cache?: boolean } = {}) {
  const useCache = !!opts.cache
  const [state, setState] = useState<State<T>>(() => ({
    data: path && useCache && cache.has(path) ? (cache.get(path) as T) : null, error: null, loading: !!path,
  }))
  const seq = useRef(0)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    const id = ++seq.current
    if (!path) { setState({ data: null, error: null, loading: false }); return }
    const hit = useCache && cache.has(path)
    setState((s) => ({ data: hit ? (cache.get(path) as T) : s.data, error: null, loading: !hit }))
    if (hit) return
    api.get<T>(path).then(
      (data) => { if (id === seq.current) { if (useCache) cache.set(path, data); setState({ data, error: null, loading: false }) } },
      (error: ApiError) => { if (id === seq.current) setState((s) => ({ data: s.data, error, loading: false })) },
    )
  }, [path, nonce, useCache])

  const retry = useCallback(() => { if (path) cache.delete(path); setNonce((n) => n + 1) }, [path])
  return { ...state, retry }
}
