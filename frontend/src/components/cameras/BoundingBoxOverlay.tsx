// components/cameras/BoundingBoxOverlay.tsx
// Renders tracked person bounding boxes on top of the MJPEG stream.
//
// The overlay is an absolutely-positioned SVG that sits on top of the
// <img> element showing the MJPEG stream. Bounding box coordinates
// come from WebSocket frame_update events (real pixel coordinates from the AI).
//
// IMPORTANT: The AI service already draws boxes on the JPEG frames themselves
// (via FrameOverlay). This React component adds a second, interactive layer
// — useful for click-to-inspect, tooltips, and highlighting specific persons.
// You can disable it if the JPEG overlay alone is sufficient.

import type { WSFramePerson } from '../../types'
import { useAlertStore } from '../../store/alertStore'

// Colors matching the dark surveillance theme
const COLOR_NORMAL  = '#00e5ff'   // cyan
const COLOR_ANOMALY = '#ff1744'   // red

interface Props {
  cameraId:   string
  persons:    WSFramePerson[]
  frameW?:    number   // original frame width  (default 640)
  frameH?:    number   // original frame height (default 360)
}

export function BoundingBoxOverlay({ cameraId, persons, frameW = 640, frameH = 360 }: Props) {
  const liveAlerts = useAlertStore(s => s.liveAlerts)
  const alertPersonIds = new Set(
    liveAlerts
      .filter(a => a.status === 'active' && a.camera_id === cameraId)
      .map(a => a.person_id)
      .filter(Boolean)
  )

  if (!persons || persons.length === 0) return null

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox={`0 0 ${frameW} ${frameH}`}
      preserveAspectRatio="xMidYMid slice"
    >
      {persons.map((person) => {
        const isAlert = alertPersonIds.has(person.person_id)
        const color   = isAlert ? COLOR_ANOMALY : COLOR_NORMAL

        // Draw bounding box if available, otherwise use center point with fixed size
        const [x1, y1, x2, y2] = person.bbox ? person.bbox : 
          person.center ? [person.center[0] - 40, person.center[1] - 40, person.center[0] + 40, person.center[1] + 40] :
          [0, 0, 0, 0]

        if (x2 === 0) return null // Skip if no position data

        const width = x2 - x1
        const height = y2 - y1

        return (
          <g key={person.person_id} opacity={0.85}>
            {/* Bounding box rectangle */}
            <rect
              x={x1}
              y={y1}
              width={width}
              height={height}
              fill="none"
              stroke={color}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />

            {/* Corner markers */}
            <rect x={x1} y={y1} width={8} height={8} fill={color} />
            <rect x={x2 - 8} y={y1} width={8} height={8} fill={color} />
            <rect x={x1} y={y2 - 8} width={8} height={8} fill={color} />
            <rect x={x2 - 8} y={y2 - 8} width={8} height={8} fill={color} />

            {/* Person ID label with background */}
            <rect
              x={x1}
              y={Math.max(0, y1 - 24)}
              width={Math.max(70, width)}
              height={20}
              fill={`rgba(10, 14, 26, 0.9)`}
              stroke={color}
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={x1 + 4}
              y={Math.max(14, y1 - 8)}
              fill={color}
              fontSize={11}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight={600}
              vectorEffect="non-scaling-stroke"
            >
              {person.person_id} · {person.activity}
            </text>
          </g>
        )
      })}
    </svg>
  )
}


// ── PersonTrackingBadge: shows tracked persons list below the camera ──────────
interface BadgeProps {
  persons:  WSFramePerson[]
  cameraId: string
}

export function PersonTrackingBadge({ persons }: BadgeProps) {
  if (persons.length === 0) return null
  return (
    <div className="flex gap-1.5 flex-wrap px-3 py-1.5 bg-surface-900/80 border-t border-surface-700">
      {persons.map(p => (
        <span
          key={p.person_id}
          className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-accent-cyan/30 text-accent-cyan bg-cyan-950/30"
        >
          {p.person_id} · {p.activity}
        </span>
      ))}
    </div>
  )
}
