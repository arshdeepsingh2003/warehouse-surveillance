// components/common/SeverityBadge.tsx
import type { AlertSeverity, AlertStatus, AnomalyLabel } from '../../types'
import './SeverityBadge.css'

export function SeverityBadge({ severity }: { severity: AlertSeverity }) {
  const cls = {
    high:   'severity-high',
    medium: 'severity-medium',
    low:    'severity-low',
  }[severity]
  return <span className={`severity-badge ${cls}`}>{severity.toUpperCase()}</span>
}

export function StatusBadge({ status }: { status: AlertStatus }) {
  return status === 'active'
    ? <span className="severity-badge severity-high"><span className="live-dot scale-75" />ACTIVE</span>
    : <span className="severity-badge severity-ok">RESOLVED</span>
}

export function AnomalyBadge({ label }: { label: AnomalyLabel }) {
  return label === 'anomaly'
    ? <span className="severity-badge severity-high">ANOMALY</span>
    : <span className="severity-badge severity-ok">NORMAL</span>
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct  = Math.round(value * 100)
  const color = pct >= 90 ? '#ff1744' : pct >= 75 ? '#ffab00' : '#00e5ff'
  return (
    <div className="confidence-bar">
      <div className="confidence-bar-track">
        <div className="confidence-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="confidence-bar-label">{pct}%</span>
    </div>
  )
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">⬡</div>
      <p className="empty-state-message">{message}</p>
    </div>
  )
}
