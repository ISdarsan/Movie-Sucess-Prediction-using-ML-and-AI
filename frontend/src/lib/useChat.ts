import { useCallback, useSyncExternalStore } from 'react'
import { api } from './api'
import { createChatStore, type ChatStore } from './chat'
import { newKey } from './runSession'

/** One in-memory store for the whole tab: leaving /assistant and coming back keeps the conversations, a refresh clears them.
 *  Nothing is persisted anywhere. */
let store: ChatStore | null = null
export function chatStore(): ChatStore {
  return (store ??= createChatStore({
    post: (path, body) => api.post(path, body),
    now: () => Date.now(),
    newId: () => newKey(globalThis.crypto),
    sleep: (ms) => new Promise<void>((r) => setTimeout(r, ms)),
  }))
}

export function useChatState() {
  const s = chatStore()
  return useSyncExternalStore(s.subscribe, s.getState, s.getState)
}

const idle = () => () => {}

/** The wall clock, re-read every `everyMs` while `active` (for the rate-limit countdown). The value is always fresh when a render happens;
 *  the interval only schedules the re-renders. */
export function useNow(active: boolean, everyMs = 500): number {
  const subscribe = useCallback((cb: () => void) => {
    if (!active) return idle()
    const t = setInterval(cb, everyMs)
    return () => clearInterval(t)
  }, [active, everyMs])
  return useSyncExternalStore(subscribe, () => Math.floor(Date.now() / everyMs) * everyMs, () => 0)
}
