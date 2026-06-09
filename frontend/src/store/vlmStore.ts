// store/vlmStore.ts
// Zustand store for real-time VLM insights pushed via WebSocket.

import { create } from 'zustand'
import type { VLMInsight } from '../types'

interface VLMState {
  insights:     VLMInsight[]          // newest first, capped at 100
  latestByPerson: Record<string, VLMInsight>  // person_id → latest insight
  addInsight:   (insight: VLMInsight) => void
  clearInsights: () => void
}

export const useVLMStore = create<VLMState>((set, get) => ({
  insights: [],
  latestByPerson: {},

  addInsight: (insight) => {
    set((state) => {
      const next = [insight, ...state.insights].slice(0, 100)
      return {
        insights: next,
        latestByPerson: {
          ...state.latestByPerson,
          [insight.person_id]: insight,
        },
      }
    })
  },

  clearInsights: () => {
    set({ insights: [], latestByPerson: {} })
  },
}))
