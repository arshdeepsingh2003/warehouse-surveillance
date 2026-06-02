// store/intelligenceStore.ts
// Zustand store for VLM/LLM intelligence panel data.
// Receives zone_summary, shift_report, and alert_explanation
// events from the WebSocket and surfaces them on the dashboard.

import { create } from 'zustand'

export interface ZoneSummary {
  zone_id:      string
  zone_name:    string
  summary:      string
  risk_level:   'low' | 'medium' | 'high' | 'unknown'
  key_events:   string[]
  person_count: number
  alert_count:  number
  generated_at: string
}

export interface ShiftReport {
  shift:           string
  summary:         string
  total_alerts:    number
  high_severity:   number
  recommendations: string[]
  generated_at:    string
  report_date?:    string
}

export interface AlertExplanation {
  alert_type:     string
  person_id:      string
  camera_id:      string
  explanation:    string
  recommendation: string
  false_positive: string
  timestamp:      string
}

interface IntelligenceState {
  zoneSummaries:     Record<string, ZoneSummary>
  shiftReport:       ShiftReport | null
  explanations:      AlertExplanation[]  // newest first, max 20

  setZoneSummary:    (s: ZoneSummary) => void
  setShiftReport:    (r: ShiftReport) => void
  addExplanation:    (e: AlertExplanation) => void
}

export const useIntelligenceStore = create<IntelligenceState>((set) => ({
  zoneSummaries: {},
  shiftReport:   null,
  explanations:  [],

  setZoneSummary: (s) => set(state => ({
    zoneSummaries: { ...state.zoneSummaries, [s.zone_id]: s },
  })),

  setShiftReport: (r) => set({ shiftReport: r }),

  addExplanation: (e) => set(state => ({
    explanations: [e, ...state.explanations].slice(0, 20),
  })),
}))
