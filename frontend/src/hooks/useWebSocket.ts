// hooks/useWebSocket.ts
// Manages the single WebSocket connection to the backend.
// Automatically reconnects on disconnect. Dispatches events to the alert store.

import { useEffect, useRef, useCallback } from 'react'
import { WS_URL } from '../api/client'
import { useAlertStore } from '../store/alertStore'
import { useCameraStore } from '../store/cameraStore'
import { useTrackingStore } from '../store/trackingStore'
import { useIntelligenceStore } from '../store/intelligenceStore'
import type { WSEvent } from '../types'

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const addLiveAlert    = useAlertStore(s => s.addLiveAlert)
  const updateCamStatus = useCameraStore(s => s.updateCameraStatus)
  const updateTracking  = useTrackingStore(s => s.updateTracking)
  const setZoneSummary  = useIntelligenceStore(s => s.setZoneSummary)
  const setShiftReport  = useIntelligenceStore(s => s.setShiftReport)
  const addExplanation  = useIntelligenceStore(s => s.addExplanation)

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return

    ws.current = new WebSocket(WS_URL)

    ws.current.onopen = () => {
      console.log('[WS] Connected to', WS_URL)
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
    }

    ws.current.onmessage = (event) => {
      try {
        const data: WSEvent = JSON.parse(event.data)

        switch (data.type) {
          case 'frame_update':
            if (data.camera_id && data.persons) {
              console.log(`[WS] frame_update for ${data.camera_id}: ${data.persons.length} persons`)
              updateTracking(data.camera_id, data.persons, data.timestamp)
            }
            break
          case 'alert_triggered':
            // Push alert into the live alert store so Alert Panel updates instantly
            if (data.alert_id) {
              addLiveAlert({
                id:           data.alert_id,
                camera_id:    data.camera_id ?? '',
                zone:         data.zone ?? '',
                alert_type:   data.alert_type ?? 'unknown',
                severity:     data.severity ?? 'low',
                description:  data.description ?? '',
                person_id:    data.person_id ?? null,
                snapshot_url: data.snapshot_url ?? null,
                status:       'active',
                confidence:   data.confidence ?? 0,
                triggered_at: data.timestamp,
                resolved_at:  null,
                resolved_by:  null,
              })
            }
            break

          case 'camera_status':
            if (data.camera_id && data.status) {
              updateCamStatus(data.camera_id, data.status, data.fps ?? 0, data.latency_ms ?? 0)
            }
            break

          case 'ping':
            // keep-alive — no action needed
            break

          case 'zone_summary':
            if (data.payload) {
              setZoneSummary(data.payload as any)
            }
            break

          case 'shift_report':
            if (data.payload) {
              setShiftReport(data.payload as any)
            }
            break

          case 'anomaly_explanation':
          case 'alert_explanation':
            if (data.payload) {
              addExplanation(data.payload as any)
            } else {
              addExplanation(data as any)
            }
            break

          default:
            break
        }
      } catch (e) {
        console.warn('[WS] Failed to parse message', e)
      }
    }

    ws.current.onclose = () => {
      console.log('[WS] Disconnected. Reconnecting in 3 s…')
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.current.onerror = () => {
      ws.current?.close()
    }
  }, [addLiveAlert, updateCamStatus, updateTracking, setZoneSummary, setShiftReport, addExplanation])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      ws.current?.close()
    }
  }, [connect])

  return { connected: ws.current?.readyState === WebSocket.OPEN }
}
