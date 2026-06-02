// components/intelligence/ZoneIntelligencePanel.tsx
//
// Displays LLM-generated zone summaries in real time.
// Updates whenever the backend broadcasts a "zone_summary" WS event.
//
// This is the "AI brain" view — instead of raw alerts,
// the operator sees a synthesized understanding of each zone.

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Brain, TrendingUp, Users, AlertTriangle, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react'

// Extend API client inline for summary endpoints
const BASE = (import.meta as any).env?.VITE_API_URL ?? 'http://localhost:8000/api/v1'

async function fetchSummaries() {
  const res = await fetch(`${BASE}/summaries/`)
  if (!res.ok) throw new Error('fetch failed')
  return res.json() as Promise<{ summaries: ZoneSummary[]; total: number }>
}

async function fetchShiftReport() {
  const res = await fetch(`${BASE}/summaries/shift-report`)
  if (!res.ok) throw new Error('fetch failed')
  return res.json() as Promise<ShiftReport>
}

// ── Types ─────────────────────────────────────────────────────────────────────
interface ZoneSummary {
  id:           string
  zone_id:      string
  zone_name:    string
  summary:      string
  risk_level:   'low' | 'medium' | 'high'
  key_events:   string[]
  person_count: number
  alert_count:  number
  generated_at: string
  time_window_min?: number
}

interface ShiftReport {
  report_date:     string
  shift:           string
  summary:         string
  total_alerts:    number
  high_severity:   number
  recommendations: string[]
  generated_at:    string
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function RiskBadge({ level }: { level: string }) {
  const map: Record<string, string> = {
    high:   'badge-high',
    medium: 'badge-medium',
    low:    'badge-ok',
  }
  return <span className={map[level] ?? 'badge-ok'}>{level.toUpperCase()}</span>
}

function TimeAgo({ iso }: { iso: string }) {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  const label = diff < 60 ? `${diff}s ago`
    : diff < 3600 ? `${Math.floor(diff/60)}m ago`
    : `${Math.floor(diff/3600)}h ago`
  return <span className="text-[10px] font-mono text-text-muted">{label}</span>
}

// ── Zone card ─────────────────────────────────────────────────────────────────
function ZoneCard({ summary }: { summary: ZoneSummary }) {
  const [expanded, setExpanded] = useState(false)

  const riskBorder = {
    high:   'border-red-900/60',
    medium: 'border-amber-900/40',
    low:    'border-surface-600',
  }[summary.risk_level] ?? 'border-surface-600'

  return (
    <div className={`card border ${riskBorder} transition-all duration-200`}>
      {/* Header */}
      <div
        className="flex items-start gap-3 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className={`mt-0.5 flex-shrink-0 ${
          summary.risk_level === 'high' ? 'text-accent-red'
          : summary.risk_level === 'medium' ? 'text-accent-amber'
          : 'text-accent-green'
        }`}>
          <Brain size={15} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-sm font-semibold text-text-primary">{summary.zone_name}</span>
            <RiskBadge level={summary.risk_level} />
            <TimeAgo iso={summary.generated_at} />
          </div>

          {/* Quick stats row */}
          <div className="flex items-center gap-4 text-[11px] text-text-muted font-mono">
            <span className="flex items-center gap-1">
              <Users size={10} />{summary.person_count} persons
            </span>
            <span className="flex items-center gap-1">
              <AlertTriangle size={10} />{summary.alert_count} alerts
            </span>
          </div>

          {/* Summary preview (truncated) */}
          <p className={`text-xs text-text-secondary mt-1.5 ${expanded ? '' : 'line-clamp-2'}`}>
            {summary.summary}
          </p>
        </div>

        <div className="flex-shrink-0 mt-0.5 text-text-muted">
          {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        </div>
      </div>

      {/* Expanded: key events */}
      {expanded && summary.key_events.length > 0 && (
        <div className="mt-3 pt-3 border-t border-surface-700 animate-fade-in">
          <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">
            Key Events
          </p>
          <ul className="space-y-1">
            {summary.key_events.map((evt, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                <span className="text-accent-cyan mt-0.5 flex-shrink-0">›</span>
                {evt}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── Shift report card ─────────────────────────────────────────────────────────
function ShiftReportCard({ report }: { report: ShiftReport }) {
  return (
    <div className="card border border-accent-cyan/20 glow-cyan">
      <div className="flex items-center gap-2 mb-3 pb-3 border-b border-surface-600">
        <TrendingUp size={14} className="text-accent-cyan" />
        <span className="text-sm font-semibold text-text-primary">Shift Intelligence Report</span>
        <span className="text-[10px] font-mono text-text-muted ml-auto">
          {report.shift} · {report.report_date}
        </span>
      </div>

      {/* Summary */}
      <p className="text-xs text-text-secondary leading-relaxed mb-4">{report.summary}</p>

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="card-sm">
          <div className="text-xl font-bold font-mono text-text-primary">{report.total_alerts}</div>
          <div className="text-[10px] text-text-muted uppercase tracking-wider">Total Alerts</div>
        </div>
        <div className="card-sm">
          <div className={`text-xl font-bold font-mono ${report.high_severity > 0 ? 'text-accent-red' : 'text-accent-green'}`}>
            {report.high_severity}
          </div>
          <div className="text-[10px] text-text-muted uppercase tracking-wider">High Severity</div>
        </div>
      </div>

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">
            AI Recommendations
          </p>
          <ul className="space-y-1.5">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                <span className="text-accent-cyan font-bold flex-shrink-0">{i + 1}.</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export function ZoneIntelligencePanel() {
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const { data: summaryData, isLoading, refetch } = useQuery({
    queryKey: ['summaries'],
    queryFn:  fetchSummaries,
    refetchInterval: 30_000,
  })

  const { data: shiftReport } = useQuery({
    queryKey: ['shift-report'],
    queryFn:  fetchShiftReport,
    refetchInterval: 60_000,
  })

  // Listen for real-time zone_summary WS events
  useEffect(() => {
    const handler = (e: CustomEvent) => {
      if (e.detail?.type === 'zone_summary' || e.detail?.type === 'shift_report') {
        refetch()
        setLastUpdate(new Date())
      }
    }
    window.addEventListener('ws_frame_update', handler as EventListener)
    return () => window.removeEventListener('ws_frame_update', handler as EventListener)
  }, [refetch])

  const summaries = summaryData?.summaries ?? []

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Brain size={15} className="text-accent-cyan" />
        <span className="text-sm font-semibold text-text-primary">Zone Intelligence</span>
        <span className="text-[10px] text-text-muted font-mono ml-1">
          LLM-powered · updates every 30s
        </span>
        <button
          onClick={() => refetch()}
          className="ml-auto text-text-muted hover:text-text-secondary transition-colors"
          title="Refresh"
        >
          <RefreshCw size={12} />
        </button>
        {lastUpdate && (
          <TimeAgo iso={lastUpdate.toISOString()} />
        )}
      </div>

      {/* Shift report */}
      {shiftReport && <ShiftReportCard report={shiftReport} />}

      {/* Zone cards */}
      {isLoading ? (
        <div className="space-y-3">
          {[1,2,3].map(i => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 bg-surface-600 rounded w-1/3 mb-2" />
              <div className="h-3 bg-surface-700 rounded w-full mb-1" />
              <div className="h-3 bg-surface-700 rounded w-4/5" />
            </div>
          ))}
        </div>
      ) : summaries.length === 0 ? (
        <div className="card text-center py-8">
          <Brain size={24} className="text-text-muted mx-auto mb-2 opacity-40" />
          <p className="text-sm text-text-muted">Zone summaries will appear here.</p>
          <p className="text-xs text-text-muted mt-1">
            The LLM generates summaries every {' '}
            <span className="font-mono text-accent-cyan/70">30 seconds</span>
            {' '}from AI detections.
          </p>
          <p className="text-xs text-text-muted mt-1">
            Set <span className="font-mono text-accent-cyan/70">VLM_BACKEND=openai</span> for real summaries.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {summaries.map(s => (
            <ZoneCard key={s.id} summary={s} />
          ))}
        </div>
      )}
    </div>
  )
}
