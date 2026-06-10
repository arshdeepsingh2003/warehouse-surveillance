// components/cameras/CameraGrid.tsx
// Live MJPEG camera grid with real-time person tracking overlays.
// The <img> tag streams MJPEG from the AI service (already contains
// YOLO boxes drawn by Python). The SVG overlay layer adds interactive
// React elements (person badges, zone info, alert indicators).

import { useEffect, useState, useRef, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Maximize2, WifiOff, Wifi, AlertTriangle, X } from 'lucide-react'
import { api } from '../../api/client'
import { useCameraStore } from '../../store/cameraStore'
import { useAlertStore } from '../../store/alertStore'
import { useTrackingStore } from '../../store/trackingStore'
import { BoundingBoxOverlay } from './BoundingBoxOverlay'
import { PersonDetailPanel } from './PersonDetailPanel'
import { ErrorBoundary } from '../common/ErrorBoundary'
import type { Camera } from '../../types'

const STREAM_BASE = (import.meta as any).env?.VITE_STREAM_URL ?? 'http://localhost:8002'

type GridLayout = '1x1' | '2x2' | '2x3'
const LAYOUT_COLS: Record<GridLayout, number> = { '1x1': 1, '2x2': 2, '2x3': 3 }

// ── Single camera card ────────────────────────────────────────────────────────

interface CardProps {
  camera:   Camera
  onExpand: () => void
  expanded: boolean
  onPersonClick?: (personId: string) => void
}

function CameraCard({ camera, onExpand, expanded, onPersonClick }: CardProps) {
  const online   = camera.status === 'online'
  const [imgErr, setImgErr]   = useState(false)
  const [loading, setLoading] = useState(true)
  const imgRef = useRef<HTMLDivElement>(null)

  // Reset on status change
  useEffect(() => {
    if (online) { setImgErr(false); setLoading(true) }
  }, [online])

  // Get tracked persons for this camera (from WS)
  const persons = useTrackingStore(s => s.getPersonsForCamera(camera.id))
  console.log(`🔍 TRACE[grid] camera=${camera.id} persons=${persons.length} ids=[${persons.map(p => p.person_id).join(', ')}]`)

  // Check if this camera has active alerts
  const liveAlerts = useAlertStore(s => s.liveAlerts)
  const hasAlert   = liveAlerts.some(a => a.camera_id === camera.id && a.status === 'active')
  const alertSev   = liveAlerts.find(a => a.camera_id === camera.id && a.status === 'active')?.severity

  const borderColor = hasAlert
    ? alertSev === 'high' ? 'border-red-600 glow-red' : 'border-amber-700'
    : online && !imgErr   ? 'border-surface-600 hover:border-accent-cyan/40' : 'border-red-900/30'

  return (
    <div className={`flex flex-col rounded-xl overflow-hidden border transition-all duration-200 group ${borderColor}`}>
      {/* ── Video + overlay container ── */}
      <div ref={imgRef} className="relative bg-surface-900 flex-shrink-0" style={{ aspectRatio: '16/9' }}>

        {online && !imgErr ? (
          <>
            {/* MJPEG stream — AI service already draws YOLO boxes on these frames */}
            <img
              src={`${STREAM_BASE}/stream/${camera.id}`}
              alt={camera.name}
              className={`w-full h-full object-cover transition-opacity duration-500 ${loading ? 'opacity-0' : 'opacity-100'}`}
              onLoad={() => setLoading(false)}
              onError={() => { setImgErr(true); setLoading(false) }}
              crossOrigin="anonymous"
            />

            {/* SVG overlay layer (interactive) — draws boxes from WS events */}
            {!loading && (
              <BoundingBoxOverlay cameraId={camera.id} persons={persons} onPersonClick={onPersonClick} />
            )}

            {/* Loading spinner */}
            {loading && (
              <div className="absolute inset-0 cam-feed flex items-center justify-center">
                <div className="flex flex-col items-center gap-2 text-text-muted">
                  <div className="w-5 h-5 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin" />
                  <span className="text-[10px] font-mono">Connecting…</span>
                </div>
              </div>
            )}

            {/* Active alert badge (top-right flash) */}
            {hasAlert && !loading && (
              <div className={`absolute top-2 left-2 flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold
                ${alertSev === 'high' ? 'bg-red-900/80 text-red-300 border border-red-700' : 'bg-amber-900/80 text-amber-300 border border-amber-700'}
                animate-pulse`}>
                ⚠ ALERT
              </div>
            )}

            {/* Person count badge — disabled (tracking store commented out) */}
            {/* {persons.length > 0 && !loading && (
              <div className="absolute top-2 right-8 px-1.5 py-0.5 rounded bg-surface-900/70 text-[10px] font-mono text-accent-cyan">
                👤 {persons.length}
              </div>
            )} */}

            {/* Corner decorations */}
            {!loading && (
              <>
                <div className="absolute top-1.5 left-1.5 w-3 h-3 border-t border-l border-accent-cyan/40 pointer-events-none" />
                <div className="absolute top-1.5 right-8 w-3 h-3 border-t border-r border-accent-cyan/40 pointer-events-none" />
                <div className="absolute bottom-7 left-1.5 w-3 h-3 border-b border-l border-accent-cyan/40 pointer-events-none" />
                <div className="absolute bottom-7 right-1.5 w-3 h-3 border-b border-r border-accent-cyan/40 pointer-events-none" />
              </>
            )}
          </>
        ) : online && imgErr ? (
          <div className="cam-feed absolute inset-0 flex flex-col items-center justify-center gap-2 text-accent-amber">
            <AlertTriangle size={22} />
            <span className="text-xs font-mono">Stream unavailable</span>
            <span className="text-[10px] text-text-muted">Start AI service: python main.py</span>
            <button onClick={() => { setImgErr(false); setLoading(true) }}
              className="mt-1 text-[10px] px-2 py-1 rounded border border-amber-800 text-accent-amber hover:bg-amber-950">
              Retry
            </button>
          </div>
        ) : (
          <div className="cam-feed absolute inset-0 flex flex-col items-center justify-center gap-2 text-red-800">
            <WifiOff size={20} />
            <span className="text-xs font-mono">SIGNAL LOST</span>
          </div>
        )}

        {/* Expand toggle */}
        <button onClick={onExpand}
          className="absolute top-2 right-2 p-1.5 bg-surface-900/70 rounded text-text-muted hover:text-accent-cyan opacity-0 group-hover:opacity-100 transition-opacity z-10">
          {expanded ? <X size={11} /> : <Maximize2 size={11} />}
        </button>
      </div>

      {/* ── Person tracking badges row ── */}
      {/* <PersonTrackingBadge persons={persons} cameraId={camera.id} /> */}

      {/* ── Caption bar ── */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-surface-800 border-t border-surface-700 gap-2 min-w-0">
        {/* Left: status dot + name/zone stacked */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {online ? (
            <span className="live-dot flex-shrink-0" />
          ) : (
            <span className="offline-dot flex-shrink-0" />
          )}
          <div className="min-w-0 leading-tight">
            <div className="text-xs font-medium text-text-primary truncate">
              {camera.name}
            </div>
            <div className="text-[10px] text-text-muted truncate">
              {camera.zone.replace(/_/g, ' ')}
            </div>
          </div>
        </div>
        {/* Right: alert dot + FPS + camera ID */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {hasAlert && (
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                alertSev === 'high'
                  ? 'bg-red-500 animate-pulse'
                  : 'bg-amber-500'
              }`}
              title={alertSev === 'high' ? 'High severity alert' : 'Active alert'}
            />
          )}
          {online && (
            <span className="text-[10px] font-mono text-text-muted whitespace-nowrap">{camera.fps}fps</span>
          )}
          <span className="text-[10px] font-mono text-text-muted whitespace-nowrap">{camera.id}</span>
        </div>
      </div>
    </div>
  )
}

// ── Grid layout ───────────────────────────────────────────────────────────────

interface SelectedPerson {
  personId: string
  cameraId: string
}

export function CameraGrid() {
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [layout, setLayout]         = useState<GridLayout>('2x3')
  const [selectedPerson, setSelectedPerson] = useState<SelectedPerson | null>(null)
  const { cameras: stored, setCameras } = useCameraStore()

  const handlePersonClick = useCallback((cameraId: string, personId: string) => {
    setSelectedPerson({ cameraId, personId })
  }, [])

  const { data: cameras, isLoading } = useQuery({
    queryKey: ['cameras'],
    queryFn:  () => api.cameras.list(),
    refetchInterval: 15_000,
  })

  useEffect(() => { if (cameras) setCameras(cameras) }, [cameras, setCameras])

  const display = stored.length > 0 ? stored : (cameras ?? [])
  const cols    = expandedId ? 1 : LAYOUT_COLS[layout]

  const shown   = expandedId
    ? display.filter(c => c.id === expandedId)
    : display

  if (isLoading) return (
    <div className="gap-3 animate-pulse" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)' }}>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="rounded-xl bg-surface-800" style={{ aspectRatio: '16/9' }} />
      ))}
    </div>
  )

  return (
    <div className="space-y-3">
      {/* Controls bar */}
      <div className="flex items-center gap-2">
        {expandedId ? (
          <button onClick={() => setExpandedId(null)}
            className="text-xs px-3 py-1 rounded-md border border-surface-600 text-text-secondary hover:text-text-primary flex items-center gap-1.5">
            <X size={11} /> Exit fullscreen
          </button>
        ) : (
          <>
            <span className="text-xs text-text-muted">Layout:</span>
            {(['2x3', '2x2', '1x1'] as GridLayout[]).map(l => (
              <button key={l} onClick={() => setLayout(l)}
                className={`text-xs px-2.5 py-1 rounded-md border transition-colors ${
                  layout === l
                    ? 'bg-accent-cyan/10 border-accent-cyan/30 text-accent-cyan'
                    : 'border-surface-600 text-text-muted hover:text-text-secondary'
                }`}>{l}</button>
            ))}
          </>
        )}
        <div className="ml-auto flex items-center gap-1.5 text-xs text-text-muted">
          <Wifi size={11} />
          <span>{display.filter(c => c.status === 'online').length}/{display.length} online</span>
        </div>
      </div>

      {/* Grid */}
      <div className="gap-3 transition-all" style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {shown.map(cam => (
          <ErrorBoundary key={cam.id}>
            <CameraCard camera={cam}
              expanded={expandedId === cam.id}
              onExpand={() => setExpandedId(expandedId === cam.id ? null : cam.id)}
              onPersonClick={(personId) => handlePersonClick(cam.id, personId)} />
          </ErrorBoundary>
        ))}
      </div>

      <p className="text-[11px] text-text-muted text-center">
        AI-annotated streams at <code className="font-mono text-accent-cyan/70">{STREAM_BASE}/stream/cam-0N</code>
      </p>

      {selectedPerson && (
        <PersonDetailPanel
          personId={selectedPerson.personId}
          cameraId={selectedPerson.cameraId}
          onClose={() => setSelectedPerson(null)}
        />
      )}
    </div>
  )
}