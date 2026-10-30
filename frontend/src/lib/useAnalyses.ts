import { useSyncExternalStore } from 'react'
import { createAnalysisRegistry, type AnalysisRegistry } from './analyses'

/** One in-memory registry for the tab (shared by the Researcher and Assistant pages). Cleared by a refresh; never persisted. */
let registry: AnalysisRegistry | null = null
export const analysisRegistry = (): AnalysisRegistry => (registry ??= createAnalysisRegistry())

export function useAnalyses() {
  const r = analysisRegistry()
  return useSyncExternalStore(r.subscribe, r.getState, r.getState)
}
