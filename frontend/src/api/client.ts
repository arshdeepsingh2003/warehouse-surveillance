// api/client.ts
// Typed API client. All components import from here — never fetch() directly.
// Change BASE_URL once and everything updates.

import type { Camera, Alert, Activity, PersonTimeline, AnalyticsSummary } from '../types'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1'
export const WS_URL  = import.meta.env.VITE_WS_URL  ?? 'ws://localhost:8000/ws'

async function get<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  const url = new URL(`${BASE_URL}${path}`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v))
    })
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json() as Promise<T>
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json() as Promise<T>
}

// ── Cameras ───────────────────────────────────────────────────────────────────
export const api = {
  cameras: {
    list: (params?: { status?: string; zone?: string }) =>
      get<Camera[]>('/cameras', params),
    get: (id: string) =>
      get<Camera>(`/cameras/${id}`),
  },

  // ── Alerts ──────────────────────────────────────────────────────────────────
  alerts: {
    list: (params?: { status?: string; severity?: string; zone?: string; limit?: number }) =>
      get<Alert[]>('/alerts', params),
    live: () =>
      get<Alert[]>('/alerts/live'),
    get: (id: string) =>
      get<Alert>(`/alerts/${id}`),
    resolve: (id: string, resolved_by: string) =>
      patch<Alert>(`/alerts/${id}/resolve`, { resolved_by }),
  },

  // ── Activities ───────────────────────────────────────────────────────────────
  activities: {
    list: (params?: { camera_id?: string; zone?: string; person_id?: string; anomaly_only?: boolean; limit?: number }) =>
      get<Activity[]>('/activities', params),
    personTimeline: (personId: string) =>
      get<PersonTimeline>(`/activities/persons/${personId}/timeline`),
  },

  // ── Analytics ────────────────────────────────────────────────────────────────
  analytics: {
    summary: () => get<AnalyticsSummary>('/analytics/summary'),
  },
}
