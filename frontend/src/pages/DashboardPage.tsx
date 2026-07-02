// pages/DashboardPage.tsx
import { useQuery } from '@tanstack/react-query'
import { Activity, AlertTriangle, Camera, Users, TrendingUp, Zap, Eye, Sparkles } from 'lucide-react'
import { api } from '../api/client'
import { AlertPanel } from '../components/alerts/AlertPanel'
import { AlertTrendChart, ZoneRiskChart } from '../components/analytics/AnalyticsCharts'
import { useAlertStore } from '../store/alertStore'
import { useVLMStore } from '../store/vlmStore'
import './Dashboard.css'

const ACCENT_MAP: Record<string, string> = {
  cyan: 'rgba(0,229,255,0.08)',
  red: 'rgba(255,23,68,0.08)',
  amber: 'rgba(255,171,0,0.08)',
  green: 'rgba(0,230,118,0.08)',
  purple: 'rgba(213,0,249,0.08)',
}

function KpiCard({
  label, value, sub, icon, accent,
}: {
  label: string; value: string | number; sub?: string
  icon: React.ReactNode; accent: string
}) {
  return (
    <div className="card kpi-card">
      <div className="kpi-icon" style={{ background: ACCENT_MAP[accent] }}>{icon}</div>
      <div className="kpi-body">
        <div className="kpi-value">{value}</div>
        <div className="kpi-label">{label}</div>
        {sub && <div className="kpi-sub">{sub}</div>}
      </div>
    </div>
  )
}

export function DashboardPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['analytics', 'summary'],
    queryFn:  api.analytics.summary,
    refetchInterval: 60_000,
  })
  const liveAlerts = useAlertStore(s => s.liveAlerts)
  const liveActive = liveAlerts.filter(a => a.status === 'active').length
  const vlmInsights = useVLMStore(s => s.insights)
  const recentVLM = vlmInsights.slice(0, 6)

  const s = data?.summary

  return (
    <div className="dashboard">
      <div className="kpi-grid dashboard-section">
        <KpiCard
          label="Cameras"
          value={isLoading ? '—' : `${s?.cameras_online}/${s?.total_cameras}`}
          sub="online"
          icon={<Camera size={16} style={{ color: 'var(--accent-cyan)' }} />}
          accent="cyan"
        />
        <KpiCard
          label="Alerts Today"
          value={isLoading ? '—' : s?.total_alerts_today ?? 0}
          sub="total triggered"
          icon={<AlertTriangle size={16} style={{ color: 'var(--accent-red)' }} />}
          accent="red"
        />
        <KpiCard
          label="Active Alerts"
          value={liveActive > 0 ? liveActive : (s?.active_alerts ?? 0)}
          sub="unresolved"
          icon={<Zap size={16} style={{ color: 'var(--accent-amber)' }} />}
          accent="amber"
        />
        <KpiCard
          label="High Severity"
          value={isLoading ? '—' : s?.high_severity_alerts ?? 0}
          sub="critical events"
          icon={<TrendingUp size={16} style={{ color: 'var(--accent-red)' }} />}
          accent="red"
        />
        <KpiCard
          label="People"
          value={isLoading ? '—' : s?.people_detected ?? 0}
          sub="detected today"
          icon={<Users size={16} style={{ color: 'var(--accent-green)' }} />}
          accent="green"
        />
        <KpiCard
          label="Peak Hour"
          value={isLoading ? '—' : (s?.peak_activity_hour?.split('–')[0].trim() ?? '—')}
          sub={s?.most_risky_zone ?? ''}
          icon={<Activity size={16} style={{ color: 'var(--accent-purple)' }} />}
          accent="purple"
        />
      </div>

      <div className="middle-grid dashboard-section">
        <div className="min-h-0">
          <AlertPanel />
        </div>
        <div>
          <ZoneRiskChart />
        </div>
      </div>

      <div className="dashboard-section vlm-stream-section">
        <div className="vlm-stream-header">
          <Sparkles size={14} />
          <span>Latest VLM Activity</span>
          {recentVLM.length > 0 && (
            <span className="vlm-stream-badge">{vlmInsights.length} insights</span>
          )}
        </div>
        <div className="vlm-stream-body">
          {recentVLM.length > 0 ? (
            recentVLM.map((insight) => (
              <div
                key={insight.id}
                className={`vlm-stream-item ${insight.anomaly_label === 'anomaly' ? 'vlm-stream-item-anomaly' : ''}`}
              >
                <div className="vlm-stream-item-left">
                  <span className={`vlm-stream-dot ${insight.anomaly_label === 'anomaly' ? 'vlm-stream-dot-anomaly' : 'vlm-stream-dot-normal'}`} />
                  <span className="vlm-stream-person">{insight.person_id}</span>
                  <span className="vlm-stream-camera">{insight.camera_id}</span>
                </div>
                <span className="vlm-stream-desc">{insight.description.slice(0, 60)}</span>
                <div className="vlm-stream-item-right">
                  <span className="vlm-stream-activity">{insight.activity_type}</span>
                  <span className="vlm-stream-confidence">{Math.round(insight.confidence * 100)}%</span>
                  <span className="vlm-stream-time">{new Date(insight.timestamp).toLocaleTimeString()}</span>
                </div>
              </div>
            ))
          ) : (
            <div className="vlm-stream-empty">
              <Eye size={18} />
              <span>Waiting for VLM insights…</span>
            </div>
          )}
        </div>
      </div>

      <div className="bottom-grid dashboard-section">
        <AlertTrendChart />

        <div className="card">
          <p className="section-title">System Status</p>
          {[
            { label: 'AI Pipeline',     ok: true  },
            { label: 'Frame Extractor', ok: true  },
            { label: 'Person Tracker',  ok: true  },
            { label: 'Alert Engine',    ok: true  },
            { label: 'WebSocket Hub',   ok: true  },
            { label: 'Notification Svc',ok: true  },
          ].map(({ label, ok }) => (
            <div key={label} className="system-status-item">
              <span className="system-status-label">{label}</span>
              <div className="system-status-indicator">
                <div className={`system-status-dot ${ok ? 'system-status-dot-ok' : 'system-status-dot-fail'}`} />
                <span className={`system-status-text ${ok ? 'system-status-text-ok' : 'system-status-text-fail'}`}>
                  {ok ? 'OPERATIONAL' : 'DEGRADED'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
