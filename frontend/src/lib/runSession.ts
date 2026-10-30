/* Tab-scoped memory of the run this tab started, so a refresh can reconnect instead of forgetting it.
   Stores ONLY {key, runId, savedAt}: no movie inputs and no results. sessionStorage is per tab and is cleared when the tab closes; nothing
   here is shared across tabs, windows or users. Pure (storage is injected) so it can be unit-tested with plain Node. */

export interface Session { key: string; runId: string | null; savedAt: number }
export interface KeyValueStore { getItem(k: string): string | null; setItem(k: string, v: string): void; removeItem(k: string): void }

export const SESSION_KEY = 'frame.researcher.run.v1'
export const SESSION_TTL_MS = 6 * 60 * 60 * 1000
export const KEY_RE = /^[A-Za-z0-9_-]{16,64}$/
export const RUN_ID_RE = /^(?:[0-9a-f]{12}|[0-9a-f]{32})$/

export function loadSession(store: KeyValueStore | null, now: number): Session | null {
  if (!store) return null
  try {
    const raw = store.getItem(SESSION_KEY)
    if (!raw) return null
    const v = JSON.parse(raw) as Partial<Session>
    const ok = v && typeof v === 'object' && typeof v.key === 'string' && KEY_RE.test(v.key)
      && (v.runId === null || (typeof v.runId === 'string' && RUN_ID_RE.test(v.runId)))
      && typeof v.savedAt === 'number' && Number.isFinite(v.savedAt) && now - v.savedAt >= 0 && now - v.savedAt <= SESSION_TTL_MS
    if (!ok) { store.removeItem(SESSION_KEY); return null }       // tampered, malformed or expired: forget it
    return { key: v.key!, runId: v.runId as string | null, savedAt: v.savedAt! }
  } catch {
    try { store.removeItem(SESSION_KEY) } catch { /* storage unavailable */ }
    return null
  }
}

export function saveSession(store: KeyValueStore | null, s: Session): void {
  try { store?.setItem(SESSION_KEY, JSON.stringify({ key: s.key, runId: s.runId, savedAt: s.savedAt })) } catch { /* private mode or quota: recovery is a convenience */ }
}

export function clearSession(store: KeyValueStore | null): void {
  try { store?.removeItem(SESSION_KEY) } catch { /* ignore */ }
}

/** 128 random bits as 32 hex characters, from the browser's CSPRNG. */
export function newKey(c: { randomUUID?: () => string; getRandomValues?: (a: Uint8Array<ArrayBuffer>) => unknown } | undefined): string {
  if (c?.randomUUID) return c.randomUUID().replaceAll('-', '')
  if (c?.getRandomValues) { const bytes = new Uint8Array(16); c.getRandomValues(bytes); return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('') }
  throw new Error('No secure random source is available in this browser.')
}
