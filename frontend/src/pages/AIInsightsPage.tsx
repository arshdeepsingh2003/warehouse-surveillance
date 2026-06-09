// pages/AIInsightsPage.tsx
// AI Insights panel — shows real-time VLM activity understanding from Qwen2.5-VL.

import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Eye, User, Camera, Clock, Cpu, Layers, AlertTriangle,
  ChevronRight, X, ExternalLink, Sparkles, Zap, Activity as ActivityIcon
} from 'lucide-react'
import { api } from '../api/client'
import { useVLMStore } from '../store/vlmStore'
import { AnomalyBadge, ConfidenceBar } from '../components/common/SeverityBadge'
import type { VLMInsight, Activity } from '../types'
import './AIInsights.css'

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
  const dateStr = ts.toLocaleDateString()
  const isAnomaly = insight.label === 'anomaly'

  return (
    <div
      className={`insight-card ${isAnomaly ? 'insight-card-anomaly' : 'insight-card-normal'}`}
      onClick={onClick}
    >
      <div className="insight-card-header">
        <div className="insight-card-person">
          <SeverityIndicator label={insight.label} />
          <User size={14} />
          <span className="insight-person-id">{insight.person_id}</span>
          <Camera size={12} className="insight-icon-spacer" />
          <span className="insight-camera-id">{insight.camera_id}</span>
        </div>
        <RiskBadge label={insight.label} />
      </div>

      <p className="insight-description">{insight.description}</p>

      <div className="insight-meta-row">
        <div className="insight-meta-tags">
          <span className="insight-activity-tag">{insight.activity}</span>
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
  personActivities,
}: {
  insight: VLMInsight
  onClose: () => void
  personActivities: Activity[]
}) {
  const ts = new Date(insight.timestamp)
  const isAnomaly = insight.label === 'anomaly'

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
                {insight.activity}
              </span>
            </div>
            <div className="detail-card">
              <span className="detail-card-label">Risk Assessment</span>
              <span className="detail-card-value">
                <RiskBadge label={insight.label} />
              </span>
            </div>
            <div className="detail-card">
              <span className="detail-card-label">Confidence</span>
              <span className="detail-card-value">
                <ConfidenceBar value={insight.confidence} />
              </span>
            </div>
            <div className="detail-card">
              <span className="detail-card-label">VLM Backend</span>
              <span className="detail-card-value detail-backend">
                <Cpu size={14} />
                {insight.backend_used || 'Qwen2.5-VL'}
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

          {/* Latest crop placeholder */}
          <div className="detail-section">
            <h3 className="detail-section-title">Latest Crop</h3>
            <div className="detail-crop-placeholder">
              <Eye size={24} />
              <span>Crop image for {insight.person_id} would appear here</span>
              <span className="detail-crop-sub">
                (available at ./crops/{insight.camera_id}/{insight.person_id}_*.jpg)
              </span>
            </div>
          </div>

          {/* Person activity history */}
          {personActivities.length > 0 && (
            <div className="detail-section">
              <h3 className="detail-section-title">Person Timeline</h3>
              <div className="detail-timeline">
                {personActivities.slice(0, 5).map((act) => (
                  <div key={act.id} className="detail-timeline-item">
                    <div className="detail-timeline-dot" />
                    <div className="detail-timeline-content">
                      <span className="detail-timeline-act">{act.activity_type}</span>
                      <span className="detail-timeline-desc">{act.description.slice(0, 60)}</span>
                      <span className="detail-timeline-time">
                        {new Date(act.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
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
          <span>Latest VLM Activity</span>
        </div>
        <span className="realtime-stream-count">{insights.length} insights</span>
      </div>
      <div className="realtime-stream-items">
        {recent.length > 0 ? (
          recent.map((insight) => (
            <div key={insight.id} className="realtime-stream-item">
              <SeverityIndicator label={insight.label} />
              <span className="realtime-stream-person">{insight.person_id}</span>
              <span className="realtime-stream-desc">{insight.description.slice(0, 50)}</span>
              <span className="realtime-stream-time">
                {new Date(insight.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))
        ) : (
          <div className="realtime-stream-empty">
            <Eye size={20} />
            <span>Waiting for VLM insights from Qwen2.5-VL…</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export function AIInsightsPage() {
  const [selectedInsight, setSelectedInsight] = useState<VLMInsight | null>(null)
  const [filterAnomaly, setFilterAnomaly] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  const insights = useVLMStore((s) => s.insights)
  const latestByPerson = useVLMStore((s) => s.latestByPerson)

  // Fetch full activities for selected person's timeline
  const { data: personActivities } = useQuery({
    queryKey: ['activities', 'person', selectedInsight?.person_id],
    queryFn: () =>
      api.activities.list({ person_id: selectedInsight!.person_id, limit: 20 }),
    enabled: !!selectedInsight,
    refetchInterval: 30_000,
  })

  // Filter + search
  const filtered = useMemo(() => {
    let list = insights
    if (filterAnomaly) list = list.filter((i) => i.label === 'anomaly')
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      list = list.filter(
        (i) =>
          i.person_id.toLowerCase().includes(q) ||
          i.description.toLowerCase().includes(q) ||
          i.activity.toLowerCase().includes(q) ||
          i.camera_id.toLowerCase().includes(q)
      )
    }
    return list
  }, [insights, filterAnomaly, searchQuery])

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
            Real-time VLM activity understanding from Qwen2.5-VL
          </span>
        </div>
        <div className="ai-insights-topbar-right">
          <input
            className="ai-insights-search"
            type="text"
            placeholder="Search person, description, camera…"
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
          <span className="stats-value">{insights.length}</span>
          <span className="stats-label">Total Insights</span>
        </div>
        <div className="stats-card">
          <span className="stats-value">
            {Object.keys(latestByPerson).length}
          </span>
          <span className="stats-label">Unique Persons</span>
        </div>
        <div className="stats-card">
          <span className="stats-value stats-value-anomaly">
            {insights.filter((i) => i.label === 'anomaly').length}
          </span>
          <span className="stats-label">Suspicious</span>
        </div>
        <div className="stats-card">
          <span className="stats-value">
            {insights.length > 0
              ? `${(insights.filter((i) => i.confidence > 0.8).length / insights.length * 100).toFixed(0)}%`
              : '—'}
          </span>
          <span className="stats-label">High Confidence</span>
        </div>
      </div>

      {/* Real-time stream */}
      <RealtimeStream insights={insights} />

      {/* Insight cards */}
      <div className="insight-list">
        {filtered.length > 0 ? (
          filtered.map((insight) => (
            <InsightCard
              key={insight.id}
              insight={insight}
              onClick={() => setSelectedInsight(insight)}
            />
          ))
        ) : (
          <div className="insight-empty">
            <Eye size={48} />
            <h3>No VLM Insights Yet</h3>
            <p>
              {insights.length === 0
                ? 'Waiting for Qwen2.5-VL to analyze person crops. Make sure the AI pipeline is running and persons are detected.'
                : 'No insights match your current filters.'}
            </p>
          </div>
        )}
      </div>

      {/* Detail drawer */}
      {selectedInsight && (
        <InsightDetailDrawer
          insight={selectedInsight}
          onClose={() => setSelectedInsight(null)}
          personActivities={personActivities ?? []}
        />
      )}
    </div>
  )
}
