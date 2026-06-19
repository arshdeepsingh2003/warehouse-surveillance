// hooks/useWebSocket.ts
// Manages the single WebSocket connection to the backend.
// Automatically reconnects on disconnect. Dispatches events to the alert store.

import { useEffect, useRef, useCallback } from 'react'
import { WS_URL } from '../api/client'
import { useAlertStore } from '../store/alertStore'
import { useCameraStore } from '../store/cameraStore'
import { useTrackingStore } from '../store/trackingStore'
import { useVLMStore } from '../store/vlmStore'
import type { VLMInsight, WSEvent } from '../types'

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const addLiveAlert    = useAlertStore(s => s.addLiveAlert)
  const updateCamStatus = useCameraStore(s => s.updateCameraStatus)
  const updateTracking  = useTrackingStore(s => s.updateTracking)
  const addVLMInsight   = useVLMStore(s => s.addInsight)

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
              const vlmCount = data.persons.filter((p: any) => p.vlm_description).length
              console.log(`🔍 TRACE[ws-receive] camera=${data.camera_id} persons=${data.persons.length} ids=[${data.persons.map((p: any) => p.person_id).join(', ')}] bboxes=[${data.persons.map((p: any) => JSON.stringify(p.bbox)).join('; ')}] vlm_in_frame=${vlmCount}`)
              for (const p of data.persons) {
                const vlmDesc = p.vlm_description ?? ''
                console.log(`[VLM-TRACE] ${p.person_id} WS_RECEIVED has_vlm=${Boolean(vlmDesc)} overlay_summary="${vlmDesc.slice(0, 80)}"`)
              }
              updateTracking(data.camera_id, data.persons, data.timestamp)
            } else {
              console.log(`🔍 TRACE[ws-receive] INCOMPLETE frame_update camera=${data.camera_id} has_persons=${Boolean(data.persons)}`)
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

          case 'activity_update':
            // Activity records are now consumed by the Activity Log page
            // via REST API polling. VLM insights go through 'vlm_insight' events.
            break

          case 'vlm_insight':
            if (data.person_id) {
              const vlmInsight: VLMInsight = {
                id:               data.insight_id ?? `vlm-${data.person_id}-${data.timestamp}-${Date.now()}`,
                person_id:        data.person_id,
                camera_id:        data.camera_id ?? '',
                zone:             data.zone ?? '',
                activity_type:    data.activity_type ?? '',
                anomaly_label:    data.anomaly_label ?? 'normal',
                description:      data.description ?? '',
                confidence:       data.confidence ?? 0,
                objects_detected: data.objects_detected ?? [],
                backend_used:     data.backend_used ?? 'moondream',
                latency_ms:       data.latency_ms ?? 0,
                source:           data.source ?? 'vlm',
                timestamp:        data.timestamp,
              }
              console.log(
                `[VLM-TRACE] ${data.person_id} WS_RECEIVED (vlm_insight) `
                + `latency=${vlmInsight.latency_ms}ms `
                + `backend=${vlmInsight.backend_used} `
                + `isFallback=${vlmInsight.backend_used === 'fallback'} `
                + `overlay_summary="${vlmInsight.description.slice(0, 80)}"`
              )
              addVLMInsight(vlmInsight)
            } else {
              console.warn(`[VLM-TRACE] WS_RECEIVED missing person_id`)
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
  }, [addLiveAlert, updateCamStatus, updateTracking, addVLMInsight])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      ws.current?.close()
    }
  }, [connect])

  return { connected: ws.current?.readyState === WebSocket.OPEN }
}
