import type { WSFramePerson } from '../../types'
import { useAlertStore } from '../../store/alertStore'
import { useVLMStore } from '../../store/vlmStore'

const COLOR_NORMAL  = '#00e676'
const COLOR_SUSPICIOUS = '#ffab00'
const COLOR_ALERT   = '#ff1744'

const ORIGINAL_FRAME_W = 640
const ORIGINAL_FRAME_H = 360

const MAX_DESC_LEN = 50

function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return text.slice(0, max) + '...'
}

interface Props {
  cameraId:   string
  persons:    WSFramePerson[]
  onPersonClick?: (personId: string) => void
}

export function BoundingBoxOverlay({ cameraId, persons, onPersonClick }: Props) {
  const liveAlerts = useAlertStore(s => s.liveAlerts)
  const latestByPerson = useVLMStore(s => s.latestByPerson)

  const highAlertPersonIds = new Set(
    liveAlerts
      .filter(a => a.status === 'active' && a.camera_id === cameraId && a.severity === 'high')
      .map(a => a.person_id)
      .filter(Boolean)
  )

  if (!persons || persons.length === 0) return null

  return (
    <svg
      className="absolute inset-0 w-full h-full"
      viewBox={`0 0 ${ORIGINAL_FRAME_W} ${ORIGINAL_FRAME_H}`}
      preserveAspectRatio="xMidYMid slice"
      style={{ pointerEvents: 'none' }}
    >
      {persons.map((person) => {
        const activeHighAlert = highAlertPersonIds.has(person.person_id)
        const vlm = latestByPerson[person.person_id]
        const isAnomaly = vlm?.anomaly_label === 'anomaly'

        let color: string
        if (activeHighAlert) {
          color = COLOR_ALERT
        } else if (isAnomaly) {
          color = COLOR_SUSPICIOUS
        } else {
          color = COLOR_NORMAL
        }

        const [x1, y1, x2, y2] = person.bbox
          ? person.bbox
          : person.center
            ? [person.center[0] - 40, person.center[1] - 40, person.center[0] + 40, person.center[1] + 40]
            : [0, 0, 0, 0]

        if (x2 === 0) return null

        const width = x2 - x1

        const descText = vlm ? `"${truncate(vlm.description, MAX_DESC_LEN)}"` : 'Analyzing...'
        const labelH = 38
        const labelY = Math.max(0, y1 - labelH - 2)

        return (
          <g key={person.person_id} opacity={0.85}>
            <rect
              x={x1}
              y={y1}
              width={width}
              height={y2 - y1}
              fill="none"
              stroke={color}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />

            <rect x={x1} y={y1} width={8} height={8} fill={color} />
            <rect x={x2 - 8} y={y1} width={8} height={8} fill={color} />
            <rect x={x1} y={y2 - 8} width={8} height={8} fill={color} />
            <rect x={x2 - 8} y={y2 - 8} width={8} height={8} fill={color} />

            <rect
              x={x1}
              y={labelY}
              width={Math.max(100, width)}
              height={labelH}
              fill={`rgba(10, 14, 26, 0.92)`}
              stroke={color}
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
              rx={2}
              style={{ pointerEvents: 'auto', cursor: 'pointer' }}
              onClick={() => onPersonClick?.(person.person_id)}
            />

            <text
              x={x1 + 5}
              y={labelY + 13}
              fill={color}
              fontSize={11}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight={600}
              vectorEffect="non-scaling-stroke"
              style={{ pointerEvents: 'none' }}
            >
              {person.person_id} · {person.activity}
            </text>

            <text
              x={x1 + 5}
              y={labelY + 28}
              fill={color}
              fontSize={10}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight={400}
              opacity={0.85}
              vectorEffect="non-scaling-stroke"
              style={{ pointerEvents: 'none' }}
            >
              {descText}
            </text>

            <text
              x={x1 + Math.max(100, width) - 12}
              y={labelY + 13}
              fill={color}
              fontSize={9}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight={600}
              opacity={0.4}
              vectorEffect="non-scaling-stroke"
              style={{ pointerEvents: 'none' }}
            >
              ›
            </text>
          </g>
        )
      })}
    </svg>
  )
}

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
