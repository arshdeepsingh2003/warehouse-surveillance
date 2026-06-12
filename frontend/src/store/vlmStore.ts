// store/vlmStore.ts
// Zustand store for real-time VLM insights pushed via WebSocket.

import { create } from 'zustand'
import type { VLMInsight } from '../types'

// HARD RULE: block any insight whose description matches known fallback patterns
function _isFallback(insight: VLMInsight): boolean {
  if (insight.backend_used === 'fallback') return true
  const desc = insight.description ?? ''
  return desc.includes('VLM analysis unavailable') || desc.includes('Person detected in')
}

interface VLMState {
  insights:          VLMInsight[]          // all insights (newest first, capped at 100)
  vlmInsights:       VLMInsight[]          // actual VLM insights (newest first, capped at 100)
  pendingPersons:    Record<string, boolean>  // persons waiting for VLM analysis
  latestByPerson:    Record<string, VLMInsight>  // person_id -> latest insight
  addInsight:        (insight: VLMInsight) => void
  setInsights:       (insights: VLMInsight[]) => void
  clearInsights:     () => void
}

export const useVLMStore = create<VLMState>((set) => ({
  insights: [],
  vlmInsights: [],
  pendingPersons: {},
  latestByPerson: {},

  addInsight: (insight) => {
    // HARD RULE: silently drop fallback insights at the store level
    if (_isFallback(insight)) {
      console.warn(`[FALLBACK-TRACE] store blocked insight person_id=${insight.person_id} backend=${insight.backend_used} desc="${(insight.description ?? '').slice(0, 60)}"`)
      return
    }

    set((state) => {
      const isVlm = insight.source === 'vlm' || insight.source === 'hybrid'

      // Main insights list (capped at 100)
      const allNext = [insight, ...state.insights].slice(0, 100)

      // VLM-only list
      const vlmNext = isVlm
        ? [insight, ...state.vlmInsights].slice(0, 100)
        : state.vlmInsights

      // Pending: if a VLM insight arrives, remove from pending
      const pending = { ...state.pendingPersons }
      if (isVlm) {
        delete pending[insight.person_id]
      }

      return {
        insights: allNext,
        vlmInsights: vlmNext,
        pendingPersons: pending,
        latestByPerson: {
          ...state.latestByPerson,
          [insight.person_id]: insight,
        },
      }
    })
  },

  setInsights: (insights) => {
    const filtered = insights.filter(i => !_isFallback(i))
    set({
      insights: filtered,
      vlmInsights: filtered.filter(i => i.source === 'vlm' || i.source === 'hybrid'),
      latestByPerson: Object.fromEntries(
        filtered.map(i => [i.person_id, i])
      ),
    })
  },

  clearInsights: () => {
    set({ insights: [], vlmInsights: [], pendingPersons: {}, latestByPerson: {} })
  },
}))
