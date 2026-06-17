// store/vlmStore.ts
// Zustand store for real-time VLM insights pushed via WebSocket.

import { create } from 'zustand'
import type { VLMInsight } from '../types'

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
    set({
      insights,
      vlmInsights: insights.filter(i => i.source === 'vlm' || i.source === 'hybrid'),
      latestByPerson: Object.fromEntries(
        insights.map(i => [i.person_id, i])
      ),
    })
  },

  clearInsights: () => {
    set({ insights: [], vlmInsights: [], pendingPersons: {}, latestByPerson: {} })
  },
}))
