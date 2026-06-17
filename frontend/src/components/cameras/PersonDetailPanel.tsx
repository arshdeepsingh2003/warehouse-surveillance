import { useMemo } from 'react'
import { Camera, Clock, Cpu, Layers, User, X, Sparkles } from 'lucide-react'
import { useVLMStore } from '../../store/vlmStore'
import { ConfidenceBar } from '../common/SeverityBadge'

function getBackendSuffix(backend_used?: string): string {
  if (backend_used === 'groq') return ' - by groq'
  if (backend_used && backend_used !== 'mock' && backend_used !== 'fallback') return ' - by no groq'
  return ''
}

interface Props {
  personId: string
  cameraId: string
  onClose: () => void
}

const STREAM_BASE = (import.meta as any).env?.VITE_STREAM_URL ?? 'http://localhost:8002'

export function PersonDetailPanel({ personId, cameraId, onClose }: Props) {
  const latestByPerson = useVLMStore(s => s.latestByPerson)
  const insight = latestByPerson[personId] ?? null

  const ts = useMemo(() => {
    if (!insight?.timestamp) return null
    return new Date(insight.timestamp)
  }, [insight?.timestamp])

  const isAnomaly = insight?.anomaly_label === 'anomaly'

  const cropUrl = `${STREAM_BASE}/crop/${cameraId}/${personId}`

  return (
    <div className="detail-panel-overlay" onClick={onClose}>
      <div className="detail-panel" onClick={e => e.stopPropagation()}>
        <div className="detail-panel-header">
          <h2 className="detail-panel-title">
            <User size={16} />
            Person Details
          </h2>
          <button className="detail-panel-close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="detail-panel-body">
          <div className="detail-sector">
            <div className="detail-id-row">
              <div className="detail-id-group">
                <span className="detail-label">Person</span>
                <span className="detail-value">{personId}</span>
              </div>
              <div className="detail-id-group">
                <Camera size={12} />
                <span className="detail-label">Camera</span>
                <span className="detail-value">{cameraId}</span>
              </div>
              {ts && (
                <div className="detail-id-group">
                  <Clock size={12} />
                  <span className="detail-label">Time</span>
                  <span className="detail-value">{ts.toLocaleTimeString()}</span>
                </div>
              )}
            </div>
          </div>

          <div className="detail-sector">
            <img
              src={cropUrl}
              alt={`${personId} crop`}
              className="detail-crop"
              onError={e => {
                (e.target as HTMLImageElement).style.display = 'none'
              }}
            />
          </div>

          {insight ? (
            <>
              <div className="detail-sector">
                <h3 className="detail-sector-title">
                  <Sparkles size={13} />
                  Activity Description
                </h3>
                <div className={`detail-desc ${isAnomaly ? 'detail-desc-anomaly' : ''}`}>
                  {insight.description}{getBackendSuffix(insight.backend_used)}
                </div>
              </div>

              <div className="detail-grid">
                <div className="detail-card">
                  <span className="detail-card-label">Activity</span>
                  <span className="detail-card-value">{insight.activity_type}</span>
                </div>
                <div className="detail-card">
                  <span className="detail-card-label">Risk</span>
                  <span className={`detail-card-value ${isAnomaly ? 'text-accent-amber' : 'text-accent-green'}`}>
                    {isAnomaly ? 'Suspicious' : 'Normal'}
                  </span>
                </div>
                <div className="detail-card">
                  <span className="detail-card-label">Confidence</span>
                  <div className="detail-card-value" style={{ width: '100%' }}>
                    <ConfidenceBar value={insight.confidence} />
                  </div>
                </div>
                <div className="detail-card">
                  <span className="detail-card-label">Source</span>
                  <span className="detail-card-value">
                    <Cpu size={12} />
                    {insight.backend_used || 'N/A'}
                  </span>
                </div>
              </div>

              <div className="detail-sector">
                <h3 className="detail-sector-title">
                  <Layers size={13} />
                  Objects Detected
                </h3>
                <div className="detail-objects">
                  {insight.objects_detected.length > 0 ? (
                    insight.objects_detected.map(obj => (
                      <span key={obj} className="detail-object-tag">{obj}</span>
                    ))
                  ) : (
                    <span className="detail-empty">No objects detected</span>
                  )}
                </div>
              </div>

              <div className="detail-sector detail-perf">
                <div className="detail-perf-row">
                  <span className="detail-perf-label">Latency</span>
                  <span className="detail-perf-value">{insight.latency_ms}ms</span>
                  <span className="detail-perf-label">Zone</span>
                  <span className="detail-perf-value">{insight.zone || '—'}</span>
                  <span className="detail-perf-label">Timestamp</span>
                  <span className="detail-perf-value">{ts?.toISOString() || '—'}</span>
                </div>
              </div>
            </>
          ) : (
            <div className="detail-empty-state">
              <span className="detail-empty-text">No description available for this person.</span>
              <span className="detail-empty-sub">Waiting for analysis...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
