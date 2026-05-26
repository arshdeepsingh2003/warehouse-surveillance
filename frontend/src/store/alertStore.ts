// store/alertStore.ts
// Zustand store for alerts.
// Holds live alerts pushed via WebSocket so the Alert Panel
// updates in real time without polling.

import { create } from 'zustand'
import type { Alert } from '../types'

interface AlertState {
  liveAlerts:   Alert[]          // WebSocket-pushed alerts (newest first, capped at 50)
  addLiveAlert: (a: Alert) => void
  resolveAlert: (id: string, by: string) => void
  clearLive:    () => void
}

export const useAlertStore = create<AlertState>((set) => ({
  liveAlerts: [],

  addLiveAlert: (alert) =>
    set((state) => ({
      // Prepend and cap at 50 to avoid unbounded growth
      liveAlerts: [alert, ...state.liveAlerts].slice(0, 50),
    })),

  resolveAlert: (id, by) =>
    set((state) => ({
      liveAlerts: state.liveAlerts.map((a) =>
        a.id === id
          ? { ...a, status: 'resolved', resolved_by: by, resolved_at: new Date().toISOString() }
          : a
      ),
    })),

  clearLive: () => set({ liveAlerts: [] }),
}))
