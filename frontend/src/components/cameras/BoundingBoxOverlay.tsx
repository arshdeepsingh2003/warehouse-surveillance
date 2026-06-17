import { useMemo } from 'react'
import type { WSFramePerson } from '../../types'
import { useAlertStore } from '../../store/alertStore'
import { useVLMStore } from '../../store/vlmStore'

const COLOR_NORMAL     = '#00e676'
const COLOR_SUSPICIOUS = '#ffab00'
const COLOR_ALERT      = '#ff1744'

const ORIGINAL_FRAME_W = 640
const ORIGINAL_FRAME_H = 360

const LABEL_GAP      = 10
const LABEL_PAD_X    = 8
const LABEL_PAD_Y    = 6
const LINE_H         = 14
const DESC_GAP       = 2
const MAX_LABEL_W    = 280
const MIN_LABEL_W    = 100
const COLLISION_GAP  = 8
const MAX_DESC_CHARS = 120
const APPROX_CHAR_W  = 5.5

function getBackendSuffix(backend_used?: string): string {
  if (backend_used === 'groq') return ' - by groq'
  if (backend_used && backend_used !== 'mock' && backend_used !== 'fallback') return ' - by no groq'
  return ''
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return text.slice(0, max) + '...'
}

function wrapText(text: string, maxChars: number): string[] {
  const lines: string[] = []
  const words = text.split(' ')
  let line = ''
  for (const word of words) {
    const added = line ? line.length + 1 + word.length : word.length
    if (added <= maxChars) {
      line = line ? `${line} ${word}` : word
    } else {
      if (line) lines.push(line)
      line = word
      while (line.length > maxChars) {
        lines.push(line.slice(0, maxChars))
        line = line.slice(maxChars)
      }
    }
  }
  if (line) lines.push(line)
  return lines.length ? lines : ['']
}

function estimateLabelW(bboxW: number, headerLen: number, descLen: number): number {
  const neededByText = Math.max(headerLen * 6.6, descLen * 6) + LABEL_PAD_X * 2
  return Math.min(Math.max(neededByText, MIN_LABEL_W, bboxW), MAX_LABEL_W)
}

function calcLabelH(labelW: number, descText: string): number {
  const charsPerLine = Math.max(1, Math.floor((labelW - LABEL_PAD_X * 2) / APPROX_CHAR_W))
  const descLines = Math.max(1, Math.ceil(descText.length / charsPerLine))
  return LABEL_PAD_Y + LINE_H + DESC_GAP + descLines * LINE_H + LABEL_PAD_Y
}

interface PersonLabelItem {
  person: WSFramePerson
  x1: number; y1: number; x2: number; y2: number
  bboxW: number; bboxH: number
  color: string; displayDesc: string
}

interface PlacedLabel {
  personId: string
  x: number
  y: number
  w: number
  h: number
  anchorX: number
  anchorY: number
  displaced: boolean
}

function rectsOverlap(a: PlacedLabel, b: PlacedLabel): boolean {
  return !(
    a.x + a.w + COLLISION_GAP <= b.x ||
    b.x + b.w + COLLISION_GAP <= a.x ||
    a.y + a.h + COLLISION_GAP <= b.y ||
    b.y + b.h + COLLISION_GAP <= a.y
  )
}

interface Props {
  cameraId:   string
  persons:    WSFramePerson[]
  onPersonClick?: (personId: string) => void
}

export function BoundingBoxOverlay({ cameraId, persons, onPersonClick }: Props) {
  const liveAlerts = useAlertStore(s => s.liveAlerts)
  const latestByPerson = useVLMStore(s => s.latestByPerson)

  const highAlertPersonIds = useMemo(
    () => new Set(
      liveAlerts
        .filter(a => a.status === 'active' && a.camera_id === cameraId && a.severity === 'high')
        .map(a => a.person_id)
        .filter(Boolean)
    ),
    [liveAlerts, cameraId]
  )

  const positionedLabels = useMemo(() => {
    if (!persons || persons.length === 0) return []

    const items = persons
      .map(person => {
        const [x1, y1, x2, y2] = person.bbox
          ? person.bbox
          : person.center
            ? [person.center[0] - 40, person.center[1] - 40, person.center[0] + 40, person.center[1] + 40]
            : [0, 0, 0, 0]
        if (x2 === 0) return null

        const vlmFromFrame = person.vlm_description
          ? { description: person.vlm_description, anomaly_label: person.vlm_anomaly_label ?? 'normal', backend_used: person.vlm_backend_used }
          : null
        const vlmFromStore = latestByPerson[person.person_id]
        const vlm = vlmFromFrame ?? vlmFromStore
        const isAnomaly = vlm?.anomaly_label === 'anomaly'
        const backendSuffix = getBackendSuffix(vlm?.backend_used)
        const vlmDesc = vlm?.description ?? ''
        const truncated = vlm ? `"${truncate(vlmDesc, MAX_DESC_CHARS)}${backendSuffix}"` : null
        const displayDesc = truncated ?? 'AI analysis pending...'

        const activeHighAlert = highAlertPersonIds.has(person.person_id)
        let color = COLOR_NORMAL
        if (activeHighAlert) color = COLOR_ALERT
        else if (isAnomaly) color = COLOR_SUSPICIOUS

        return { person, x1, y1, x2, y2, bboxW: x2 - x1, bboxH: y2 - y1, color, displayDesc }
      })
      .filter(Boolean) as PersonLabelItem[]

    const sorted = [...items].sort((a, b) => a.y1 - b.y1)
    const placed: PlacedLabel[] = []

    return sorted.map(d => {
      const headerText = `${d.person.person_id} · ${d.person.activity}`
      const labelW = estimateLabelW(d.bboxW, headerText.length, d.displayDesc.length)
      const labelH = calcLabelH(labelW, d.displayDesc)

      let labelX = d.x1
      let labelY = d.y1 - labelH - LABEL_GAP
      let displaced = false

      if (labelY < 0) {
        labelY = d.y2 + LABEL_GAP
        displaced = true
      }

      if (labelX + labelW > ORIGINAL_FRAME_W) {
        labelX = ORIGINAL_FRAME_W - labelW
        displaced = true
      }
      if (labelX < 0) {
        labelX = 0
        displaced = true
      }
      if (labelY + labelH > ORIGINAL_FRAME_H) {
        labelY = ORIGINAL_FRAME_H - labelH
        displaced = true
      }
      if (labelY < 0) {
        labelY = 0
        displaced = true
      }

      const candidate: PlacedLabel = {
        personId: d.person.person_id,
        x: labelX, y: labelY, w: labelW, h: labelH,
        anchorX: d.x1 + d.bboxW / 2,
        anchorY: labelY < d.y1 ? d.y1 : d.y2,
        displaced,
      }

      let iterations = 0
      while (iterations < 20) {
        let collided = false
        for (const p of placed) {
          if (rectsOverlap(candidate, p)) {
            collided = true
            candidate.y = p.y + p.h + COLLISION_GAP
            candidate.displaced = true
            if (candidate.y + labelH > ORIGINAL_FRAME_H) {
              candidate.y = p.y - labelH - COLLISION_GAP
              if (candidate.y < 0) {
                candidate.y = p.y
                candidate.x = p.x + p.w + COLLISION_GAP
                if (candidate.x + labelW > ORIGINAL_FRAME_W) {
                  candidate.x = p.x - labelW - COLLISION_GAP
                  if (candidate.x < 0) {
                    candidate.x = 0
                    candidate.y = Math.max(0, d.y1 < ORIGINAL_FRAME_H / 2 ? d.y1 - labelH - LABEL_GAP : d.y2 + LABEL_GAP)
                  }
                }
              }
            }
            break
          }
        }
        if (!collided) break
        iterations++
      }

      candidate.x = Math.max(0, Math.min(candidate.x, ORIGINAL_FRAME_W - labelW))
      candidate.y = Math.max(0, Math.min(candidate.y, ORIGINAL_FRAME_H - labelH))

      placed.push(candidate)
      return { ...d, label: candidate, headerText }
    })
  }, [persons, latestByPerson, highAlertPersonIds])

  if (positionedLabels.length === 0) return null

  return (
    <svg
      className="absolute inset-0 w-full h-full"
      viewBox={`0 0 ${ORIGINAL_FRAME_W} ${ORIGINAL_FRAME_H}`}
      preserveAspectRatio="xMidYMid slice"
      style={{ pointerEvents: 'none' }}
    >
      {positionedLabels.map(d => {
        const { label } = d
        const charsPerLine = Math.max(1, Math.floor((label.w - LABEL_PAD_X * 2) / APPROX_CHAR_W))
        const descLines = wrapText(d.displayDesc, charsPerLine)
        const descStartY = label.y + LABEL_PAD_Y + LINE_H + DESC_GAP

        return (
          <g key={d.person.person_id} opacity={0.85}>
            <rect
              x={d.x1} y={d.y1}
              width={d.bboxW} height={d.bboxH}
              fill="none" stroke={d.color}
              strokeWidth={2} vectorEffect="non-scaling-stroke"
            />

            <rect x={d.x1} y={d.y1} width={8} height={8} fill={d.color} />
            <rect x={d.x2 - 8} y={d.y1} width={8} height={8} fill={d.color} />
            <rect x={d.x1} y={d.y2 - 8} width={8} height={8} fill={d.color} />
            <rect x={d.x2 - 8} y={d.y2 - 8} width={8} height={8} fill={d.color} />

            {label.displaced && (
              <line
                x1={d.x1 + d.bboxW / 2}
                y1={d.y1}
                x2={label.x + label.w / 2}
                y2={label.y + label.h}
                stroke={d.color}
                strokeWidth={1}
                strokeDasharray="3 2"
                opacity={0.5}
                vectorEffect="non-scaling-stroke"
              />
            )}

            <rect
              x={label.x} y={label.y}
              width={label.w} height={label.h}
              fill="rgba(10, 14, 26, 0.92)"
              stroke={d.color} strokeWidth={1}
              vectorEffect="non-scaling-stroke"
              rx={2}
              style={{ pointerEvents: 'auto', cursor: 'pointer' }}
              onClick={() => onPersonClick?.(d.person.person_id)}
            />

            <text
              x={label.x + LABEL_PAD_X}
              y={label.y + LABEL_PAD_Y + LINE_H - 3}
              fill={d.color}
              fontSize={11}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight={600}
              vectorEffect="non-scaling-stroke"
              style={{ pointerEvents: 'none' }}
            >
              {d.headerText}
            </text>

            {descLines.map((line, i) => (
              <text
                key={i}
                x={label.x + LABEL_PAD_X}
                y={descStartY + i * LINE_H}
                fill={d.color}
                fontSize={10}
                fontFamily="'JetBrains Mono', monospace"
                fontWeight={400}
                opacity={0.85}
                vectorEffect="non-scaling-stroke"
                style={{ pointerEvents: 'none' }}
              >
                {line}
              </text>
            ))}

            <text
              x={label.x + label.w - 12}
              y={label.y + LABEL_PAD_Y + LINE_H - 3}
              fill={d.color}
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
