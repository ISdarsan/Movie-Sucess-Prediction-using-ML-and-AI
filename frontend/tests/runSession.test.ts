import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { clearSession, KEY_RE, loadSession, newKey, saveSession, SESSION_KEY, SESSION_TTL_MS } from '../src/lib/runSession.ts'

const mem = () => { const m = new Map<string, string>(); return { m, s: { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), removeItem: (k: string) => void m.delete(k) } } }
const K = 'f'.repeat(32), NOW = 5_000_000

describe('run session (tab storage)', () => {
  it('round-trips key, run id and time, and nothing else', () => {
    const { m, s } = mem(); saveSession(s, { key: K, runId: 'abcdef012345', savedAt: NOW })
    assert.deepEqual(JSON.parse(m.get(SESSION_KEY)!), { key: K, runId: 'abcdef012345', savedAt: NOW })
    assert.deepEqual(loadSession(s, NOW + 5), { key: K, runId: 'abcdef012345', savedAt: NOW })
  })
  it('accepts both 12- and 32-character run ids, and a null run id', () => {
    for (const runId of ['abcdef012345', 'a'.repeat(32), null]) { const { s } = mem(); saveSession(s, { key: K, runId, savedAt: NOW }); assert.equal(loadSession(s, NOW)?.runId, runId) }
  })
  it('expires after the TTL and deletes the stale entry', () => {
    const { m, s } = mem(); saveSession(s, { key: K, runId: null, savedAt: NOW })
    assert.ok(loadSession(s, NOW + SESSION_TTL_MS)); assert.equal(loadSession(s, NOW + SESSION_TTL_MS + 1), null); assert.equal(m.size, 0)
  })
  it('rejects and deletes anything malformed or tampered', () => {
    for (const raw of ['', 'nope', 'null', '[]', '{"key":1}', JSON.stringify({ key: K, runId: 'XYZ', savedAt: NOW }), JSON.stringify({ key: 'bad key!!!!!!!!!!!!', runId: null, savedAt: NOW }), JSON.stringify({ key: K, runId: null, savedAt: 'now' })]) {
      const { m, s } = mem(); m.set(SESSION_KEY, raw); assert.equal(loadSession(s, NOW), null, raw)
    }
  })
  it('never throws when storage is missing or throws', () => {
    const boom = { getItem: () => { throw new Error('blocked') }, setItem: () => { throw new Error('blocked') }, removeItem: () => { throw new Error('blocked') } }
    assert.equal(loadSession(null, NOW), null); assert.equal(loadSession(boom, NOW), null)
    assert.doesNotThrow(() => { saveSession(null, { key: K, runId: null, savedAt: NOW }); saveSession(boom, { key: K, runId: null, savedAt: NOW }); clearSession(null); clearSession(boom) })
  })
  it('clearSession removes the entry', () => { const { m, s } = mem(); saveSession(s, { key: K, runId: null, savedAt: NOW }); clearSession(s); assert.equal(m.size, 0) })
})

describe('submission keys', () => {
  it('are 32 hex characters (128 bits) and accepted by the server-side key pattern', () => {
    const k = newKey(globalThis.crypto); assert.match(k, /^[0-9a-f]{32}$/); assert.ok(KEY_RE.test(k))
  })
  it('are different every time', () => assert.equal(new Set(Array.from({ length: 200 }, () => newKey(globalThis.crypto))).size, 200))
  it('fall back to getRandomValues when randomUUID is unavailable (plain-http pages)', () => {
    const k = newKey({ getRandomValues: (a: Uint8Array) => { a.fill(171); return a } }); assert.equal(k, 'ab'.repeat(16))
  })
  it('refuse to invent a weak key when there is no secure randomness', () => assert.throws(() => newKey(undefined), /secure random/))
})
