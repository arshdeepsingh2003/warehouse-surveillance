// components/alerts/AlertPanel.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import { ShieldAlert, CheckCircle, ChevronRight, Camera } from 'lucide-react'
import { api } from '../../api/client'
import { useAlertStore } from '../../store/alertStore'
import { SeverityBadge, StatusBadge, ConfidenceBar, EmptyState } from '../common/SeverityBadge'
import type { Alert } from '../../types'
import './AlertPanel.css'

function AlertRow({ alert, onResolve }: { alert: Alert; onResolve: (id: string) => void }) {
  const [expanded, setExpanded] = useState(false)

  const sevClass = alert.severity === 'high' ? 'alert-severity-high'
    : alert.severity === 'medium' ? 'alert-severity-medium'
    : 'alert-severity-low'

  return (
    <div className={`alert-row ${alert.status === 'active' ? 'alert-row-active' : 'alert-row-resolved'}`}>
      <div className="alert-row-header" onClick={() => setExpanded(!expanded)}>
        <div className={`alert-severity-icon ${sevClass}`}>
          <ShieldAlert size={15} />
        </div>

        <div className="alert-row-main">
          <div className="alert-row-top">
            <SeverityBadge severity={alert.severity} />
            <span className="alert-type">{alert.alert_type.replace(/_/g, ' ')}</span>
            <span className="alert-time">
              {formatDistanceToNow(new Date(alert.triggered_at), { addSuffix: true })}
            </span>
          </div>
          <p className="alert-description">{alert.description}</p>
          <div className="alert-meta">
            <span className="alert-meta-item">
              <Camera size={10} />{alert.camera_id}
            </span>
            <span className="alert-meta-item">{alert.zone.replace(/_/g, ' ')}</span>
          </div>
        </div>

        <ChevronRight size={14} className={`alert-chevron ${expanded ? 'alert-chevron-open' : ''}`} />
      </div>

      {expanded && (
        <div className="alert-detail">
          {alert.snapshot_url && (
            <img
              src={alert.snapshot_url}
              alt="Alert snapshot"
              className="alert-detail-snapshot"
            />
          )}

          <div className="alert-detail-grid">
            <div>
              <span className="alert-detail-label">Person ID</span>
              <div className="alert-detail-value">{alert.person_id ?? '—'}</div>
            </div>
            <div>
              <span className="alert-detail-label">Confidence</span>
              <div><ConfidenceBar value={alert.confidence} /></div>
            </div>
            <div>
              <span className="alert-detail-label">Status</span>
              <div><StatusBadge status={alert.status} /></div>
            </div>
            {alert.resolved_by && (
              <div>
                <span className="alert-detail-label">Resolved by</span>
                <div className="alert-detail-value-plain">{alert.resolved_by}</div>
              </div>
            )}
          </div>

          {alert.status === 'active' && (
            <button onClick={() => onResolve(alert.id)} className="alert-resolve-btn">
              <CheckCircle size={13} /> Mark Resolved
            </button>
          )}
        </div>
      )}
    </div>
  )
}

interface Props { compact?: boolean }

export function AlertPanel({ compact = false }: Props) {
  const [tab, setTab] = useState<'live' | 'history'>('live')
  const qc = useQueryClient()
  const liveAlerts = useAlertStore(s => s.liveAlerts)
  const resolveInStore = useAlertStore(s => s.resolveAlert)

  const { data: historicAlerts = [] } = useQuery({
    queryKey: ['alerts', 'all'],
    queryFn:  () => api.alerts.list({ limit: 50 }),
    refetchInterval: 60_000,
  })

  const resolveMutation = useMutation({
    mutationFn: (id: string) => api.alerts.resolve(id, 'operator@warehouse.com'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts'] }),
  })

  const handleResolve = (id: string) => {
    resolveMutation.mutate(id)
    resolveInStore(id, 'operator@warehouse.com')
  }

  const shown: Alert[] = tab === 'live'
    ? (liveAlerts.length > 0 ? liveAlerts : historicAlerts.filter(a => a.status === 'active'))
    : historicAlerts

  const activeCount = liveAlerts.filter(a => a.status === 'active').length

  return (
    <div className="card alert-panel">
      <div className="alert-panel-header">
        <div className="alert-panel-header-left">
          <ShieldAlert size={15} className="alert-panel-icon" />
          <span className="alert-panel-title">Alert Panel</span>
          {activeCount > 0 && (
            <span className="alert-panel-count">{activeCount}</span>
          )}
        </div>
        <div className="alert-panel-tabs">
          {(['live', 'history'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`alert-panel-tab ${tab === t ? 'alert-panel-tab-active' : 'alert-panel-tab-inactive'}`}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className={`alert-list ${compact ? 'alert-list-compact' : ''}`}>
        {shown.length === 0
          ? <EmptyState message="No alerts" />
          : shown.map(alert => (
              <AlertRow key={alert.id} alert={alert} onResolve={handleResolve} />
            ))
        }
      </div>
    </div>
  )
}
