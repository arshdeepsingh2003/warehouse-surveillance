// pages/AIInsightsPage.tsx
// AI Insights panel — shows VLM-generated insights only.
// Consumes VLMInsight records (persisted) from the REST API,
// with real-time updates via WebSocket.

import { useState, useMemo, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Eye, User, Camera, Clock, Cpu, Layers, AlertTriangle,
  ChevronRight, X, Sparkles, Zap, Activity as ActivityIcon,
  Hourglass, BrainCircuit, Bot
} from 'lucide-react'
import { api } from '../api/client'
import { useVLMStore } from '../store/vlmStore'
import { ConfidenceBar } from '../components/common/SeverityBadge'
import type { VLMInsight } from '../types'
import './AIInsights.css'

// ── Backend badge mapping ─────────────────────────────────────────────────────

const BACKEND_BADGES: Record<string, { label: string; className: string; icon: React.ReactNode }> = {
  'moondream':{ label: 'Moondream',  className: 'backend-badge-moondream',icon: <BrainCircuit size={12} /> },
  'qwen_vl':  { label: 'Qwen',       className: 'backend-badge-qwen',     icon: <Bot size={12} /> },
  'mock':     { label: 'Mock',       className: 'backend-badge-mock',     icon: <Cpu size={12} /> },
}

function BackendBadge({ backendUsed }: { backendUsed: string }) {
  const badge = BACKEND_BADGES[backendUsed] ?? {
    label: backendUsed || 'Unknown',
    className: 'backend-badge-unknown',
    icon: <Cpu size={12} />,
  }
  return (
    <span className={`backend-badge ${badge.className}`}>
      {badge.icon}
      {badge.label}
    </span>
  )
}

// ── Risk-level badge ──────────────────────────────────────────────────────────

function RiskBadge({ label }: { label: string }) {
  const isAnomaly = label === 'anomaly'
  return (
    <span className={`risk-badge ${isAnomaly ? 'risk-badge-anomaly' : 'risk-badge-normal'}`}>
      <span className={`risk-dot ${isAnomaly ? 'risk-dot-anomaly' : 'risk-dot-normal'}`} />
      {isAnomaly ? 'SUSPICIOUS' : 'NORMAL'}
    </span>
  )
}

function SeverityIndicator({ label }: { label: string }) {
  if (label === 'anomaly') {
    return <span className="severity-dot severity-dot-high" title="Suspicious activity" />
  }
  return <span className="severity-dot severity-dot-normal" title="Normal activity" />
}

// ── Insight Card ──────────────────────────────────────────────────────────────

function InsightCard({
  insight,
  onClick,
}: {
  insight: VLMInsight
  onClick: () => void
}) {
  const ts = new Date(insight.timestamp)
  const timeStr = ts.toLocaleTimeString()
  const isAnomaly = insight.anomaly_label === 'anomaly'

  return (
    <div
      className={`insight-card ${isAnomaly ? 'insight-card-anomaly' : 'insight-card-normal'}`}
      onClick={onClick}
    >
      <div className="insight-card-header">
        <div className="insight-card-person">
          <SeverityIndicator label={insight.anomaly_label} />
          <User size={14} />
          <span className="insight-person-id">{insight.person_id}</span>
          <Camera size={12} className="insight-icon-spacer" />
          <span className="insight-camera-id">{insight.camera_id}</span>
          <BackendBadge backendUsed={insight.backend_used} />
        </div>
        <RiskBadge label={insight.anomaly_label} />
      </div>

      <p className="insight-description">{insight.description}</p>

      <div className="insight-meta-row">
        <div className="insight-meta-tags">
          <span className="insight-activity-tag">{insight.activity_type}</span>
          {insight.objects_detected.slice(0, 3).map((obj) => (
            <span key={obj} className="insight-object-tag">{obj}</span>
          ))}
          {insight.objects_detected.length > 3 && (
            <span className="insight-object-tag insight-object-tag-more">
              +{insight.objects_detected.length - 3}
            </span>
          )}
        </div>
        <div className="insight-meta-time">
          <Clock size={12} />
          <span>{timeStr}</span>
        </div>
      </div>

      <div className="insight-footer">
        <ConfidenceBar value={insight.confidence} />
        <ChevronRight size={14} className="insight-chevron" />
      </div>
    </div>
  )
}

// ── Detail Drawer ─────────────────────────────────────────────────────────────

function InsightDetailDrawer({
  insight,
  onClose,
}: {
  insight: VLMInsight
  onClose: () => void
}) {
  const ts = new Date(insight.timestamp)
  const isAnomaly = insight.anomaly_label === 'anomaly'

  return (
    <div className="insight-drawer-overlay" onClick={onClose}>
      <div className="insight-drawer" onClick={(e) => e.stopPropagation()}>
        <div className="insight-drawer-header">
          <h2 className="insight-drawer-title">
            <Sparkles size={18} />
            VLM Insight Details
          </h2>
          <button className="insight-drawer-close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="insight-drawer-body">
          {/* Person identity */}
          <div className="detail-section">
            <div className="detail-id-row">
              <div className="detail-id-group">
                <User size={14} />
                <span className="detail-label">Person</span>
                <span className="detail-value">{insight.person_id}</span>
              </div>
              <div className="detail-id-group">
                <Camera size={14} />
                <span className="detail-label">Camera</span>
                <span className="detail-value">{insight.camera_id}</span>
              </div>
              <div className="detail-id-group">
                <Clock size={14} />
                <span className="detail-label">Time</span>
                <span className="detail-value">{ts.toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Activity description */}
          <div className="detail-section">
            <h3 className="detail-section-title">Activity Description</h3>
            <div className={`detail-description ${isAnomaly ? 'detail-description-anomaly' : ''}`}>
              <p>{insight.description}</p>
            </div>
          </div>

          {/* Activity category + risk */}
          <div className="detail-grid">
            <div className="detail-card">
              <span className="detail-card-label">Activity Category</span>
              <span className="detail-card-value">
                <ActivityIcon size={14} />
                {insight.activity_type}
              </span>
            </div>
            <div className="detail-card">
              <span className="detail-card-label">Risk Assessment</span>
              <span className="detail-card-value">
                <RiskBadge label={insight.anomaly_label} />
              </span>
            </div>
            <div className="detail-card">
              <span className="detail-card-label">Confidence</span>
              <span className="detail-card-value">
                <ConfidenceBar value={insight.confidence} />
              </span>
            </div>
            <div className="detail-card">
              <span className="detail-card-label">Source</span>
              <span className="detail-card-value detail-backend">
                <Cpu size={14} />
                <BackendBadge backendUsed={insight.backend_used} />
              </span>
            </div>
          </div>

          {/* Objects detected */}
          <div className="detail-section">
            <h3 className="detail-section-title">
              <Layers size={14} />
              Objects Detected
            </h3>
            <div className="detail-objects">
              {insight.objects_detected.length > 0 ? (
                insight.objects_detected.map((obj) => (
                  <span key={obj} className="detail-object-tag">{obj}</span>
                ))
              ) : (
                <span className="detail-empty">No objects detected</span>
              )}
            </div>
          </div>

          {/* Performance */}
          <div className="detail-section">
            <h3 className="detail-section-title">
              <Zap size={14} />
              Inference Performance
            </h3>
            <div className="detail-perf-row">
              <span className="detail-perf-label">Latency</span>
              <span className="detail-perf-value">{insight.latency_ms} ms</span>
              <span className="detail-perf-label">Zone</span>
              <span className="detail-perf-value">{insight.zone}</span>
              <span className="detail-perf-label">Timestamp</span>
              <span className="detail-perf-value">{ts.toISOString()}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Real-time Stream Strip ────────────────────────────────────────────────────

function RealtimeStream({ insights }: { insights: VLMInsight[] }) {
  const recent = insights.slice(0, 8)

  return (
    <div className="realtime-stream">
      <div className="realtime-stream-header">
        <div className="realtime-stream-title">
          <Zap size={14} className="realtime-stream-icon" />
          <span>Latest VLM Insights</span>
        </div>
        <span className="realtime-stream-count">{insights.length} insights</span>
      </div>
      <div className="realtime-stream-items">
        {recent.length > 0 ? (
          recent.map((insight) => (
            <div key={insight.id} className="realtime-stream-item">
              <SeverityIndicator label={insight.anomaly_label} />
              <span className="realtime-stream-person">{insight.person_id}</span>
              <span className="realtime-stream-desc">{insight.description.slice(0, 50)}</span>
              <BackendBadge backendUsed={insight.backend_used} />
              <span className="realtime-stream-time">
                {new Date(insight.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))
        ) : (
          <div className="realtime-stream-empty">
            <Eye size={20} />
            <span>Waiting for VLM insights...</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Section component ─────────────────────────────────────────────────────────

function InsightSection({
  title,
  icon,
  insights,
  onSelectInsight,
}: {
  title: string
  icon: React.ReactNode
  insights: VLMInsight[]
  onSelectInsight: (insight: VLMInsight) => void
}) {
  if (insights.length === 0) return null

  return (
    <div className="insight-section">
      <div className="insight-section-header">
        {icon}
        <span className="insight-section-title">{title}</span>
        <span className="insight-section-count">{insights.length}</span>
      </div>

      {insights.map((insight) => (
        <InsightCard
          key={insight.id}
          insight={insight}
          onClick={() => onSelectInsight(insight)}
        />
      ))}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export function AIInsightsPage() {
  const [selectedInsight, setSelectedInsight] = useState<VLMInsight | null>(null)
  const [filterAnomaly, setFilterAnomaly] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  // Live WebSocket VLM insights (real-time updates)
  const liveInsights = useVLMStore((s) => s.vlmInsights)
  const setInsights = useVLMStore((s) => s.setInsights)

  const queryClient = useQueryClient()

  // Fetch persisted VLM insights from API (polling)
  const { data: apiInsights = [] } = useQuery({
    queryKey: ['vlm-insights', filterAnomaly],
    queryFn: () => api.vlmInsights.list({ anomaly_only: filterAnomaly, limit: 100 }),
    refetchInterval: 30_000,
  })

  // Merge API data into store on fetch (so WS updates still layer on top)
  useEffect(() => {
    if (apiInsights.length > 0) {
      setInsights(apiInsights)
    }
  }, [apiInsights, setInsights])

  // Use live insights from store (which includes API + WS merged)
  const allInsights = useVLMStore((s) => s.insights)
  const vlmInsights = useVLMStore((s) => s.vlmInsights)
  const latestByPerson = useVLMStore((s) => s.latestByPerson)

  // Filter + search
  const filteredVlm = useMemo(() => {
    let list = vlmInsights
    if (filterAnomaly) list = list.filter((i) => i.anomaly_label === 'anomaly')
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      list = list.filter(
        (i) =>
          i.person_id.toLowerCase().includes(q) ||
          i.description.toLowerCase().includes(q) ||
          i.activity_type.toLowerCase().includes(q) ||
          i.camera_id.toLowerCase().includes(q)
      )
    }
    return list
  }, [vlmInsights, filterAnomaly, searchQuery])

  const hasAnyContent = filteredVlm.length > 0

  return (
    <div className="ai-insights-page">
      {/* Top bar */}
      <div className="ai-insights-topbar">
        <div className="ai-insights-topbar-left">
          <h1 className="ai-insights-title">
            <Sparkles size={20} />
            AI Insights
          </h1>
          <span className="ai-insights-subtitle">
            VLM-powered activity understanding
          </span>
        </div>
        <div className="ai-insights-topbar-right">
          <input
            className="ai-insights-search"
            type="text"
            placeholder="Search person, description, camera..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <label className="ai-insights-filter">
            <input
              type="checkbox"
              checked={filterAnomaly}
              onChange={(e) => setFilterAnomaly(e.target.checked)}
            />
            <AlertTriangle size={14} />
            Anomalies only
          </label>
        </div>
      </div>

      {/* Stats bar */}
      <div className="ai-insights-stats">
        <div className="stats-card">
          <span className="stats-value">{allInsights.length}</span>
          <span className="stats-label">Total VLM Insights</span>
        </div>
        <div className="stats-card">
          <span className="stats-value">{vlmInsights.length}</span>
          <span className="stats-label">VLM Records</span>
        </div>
        <div className="stats-card">
          <span className="stats-value">
            {Object.keys(latestByPerson).length}
          </span>
          <span className="stats-label">Unique Persons</span>
        </div>
        <div className="stats-card">
          <span className="stats-value stats-value-anomaly">
            {allInsights.filter((i) => i.anomaly_label === 'anomaly').length}
          </span>
          <span className="stats-label">Suspicious</span>
        </div>
      </div>

      {/* Real-time stream */}
      <RealtimeStream insights={allInsights} />

      {/* VLM Insight Section */}
      <InsightSection
        title="VLM Insights"
        icon={<BrainCircuit size={14} className="insight-section-icon vlm-icon" />}
        insights={filteredVlm}
        onSelectInsight={setSelectedInsight}
      />

      {/* Empty state */}
      {!hasAnyContent && (
        <div className="insight-empty">
          <Eye size={48} />
          <h3>No VLM Insights Yet</h3>
          <p>
            Waiting for VLM to analyze person crops. Make sure the AI pipeline is running and persons are detected.
          </p>
        </div>
      )}

      {/* Detail drawer */}
      {selectedInsight && (
        <InsightDetailDrawer
          insight={selectedInsight}
          onClose={() => setSelectedInsight(null)}
        />
      )}
    </div>
  )
}
