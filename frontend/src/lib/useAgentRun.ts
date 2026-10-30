import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import { createRunController, IDLE, type RunState, type SessionStore } from './agentRun'
import { clearSession, loadSession, newKey, saveSession, type KeyValueStore } from './runSession'

const clock = () => Date.now()
const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))
/** Tab-scoped storage; null when the browser blocks it (private mode, policy). Recovery then simply does nothing. */
const tabStore = (): KeyValueStore | null => { try { return window.sessionStorage } catch { return null } }
const session: SessionStore = {
  load: () => loadSession(tabStore(), Date.now()),
  save: (s) => saveSession(tabStore(), s),
  clear: () => clearSession(tabStore()),
}

/** React binding for the bounded submit-and-poll controller. Polling stops on unmount; the run itself keeps going on the server
 *  and can be picked up again with recover(). */
export function useAgentRun() {
  const [state, setState] = useState<RunState>(IDLE)
  const [ctl] = useState(() => createRunController({
    post: (path, body, headers) => api.post(path, body, headers), get: (path) => api.get(path), now: clock, sleep: wait,
    onChange: (s) => setState(s), newKey: () => newKey(globalThis.crypto), session,
  }))
  useEffect(() => () => ctl.cancel(), [ctl])
  const actions = useMemo(() => ({
    start: (body: unknown) => ctl.start(body),
    retry: () => ctl.retry(),
    resume: () => ctl.resume(),
    recover: () => ctl.recover(),
    reset: () => ctl.reset(),
    peek: () => ctl.getState(),
  }), [ctl])
  return { state, ...actions }
}
