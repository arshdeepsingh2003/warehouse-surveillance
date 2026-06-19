// pages/IntelligencePage.tsx
// AI Intelligence Dashboard — shows LLM-generated zone summaries,
// shift reports, anomaly explanations, and VLM activity descriptions.
//
// Data sources:
//   • GET /api/v1/summaries/zones        — zone risk + LLM summaries
//   • GET /api/v1/summaries/shift-report — consolidated shift report
//   • WS  zone_summary events            — real-time updates
//   • WS  shift_report events
//   • WS  anomaly_explanation events

import { useState, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Brain, TrendingUp, FileText,
  Zap, CheckCircle, ChevronDown, ChevronUp, Sparkles
} from 'lucide-react'

// ── Type defs ─────────────────────────────────────────────────────────────────
interface ZoneSummary {
  zone_id:      string
  zone_name:    string
  summary:      string
  risk_level:   'low' | 'medium' | 'high' | 'unknown'
  key_events:   string[]
  person_count: number
  alert_count:  number
  generated_at: string
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

interface ZoneSummaryResponse {
  summaries:   ZoneSummary[]
  zone_count:  number
}

// ── Risk level styling ────────────────────────────────────────────────────────
function riskBg(level: string): string {
  return { high: 'bg-red-950/40 border-red-800/60', medium: 'bg-amber-950/40 border-amber-800/60', low: 'bg-green-950/40 border-green-800/60' }[level] ?? 'bg-surface-800 border-surface-600'
}
function riskBadge(level: string) {
  const cls = { high: 'badge-high', medium: 'badge-medium', low: 'badge-ok' }[level] ?? 'badge-low'
  return <span className={cls}>{level.toUpperCase()}</span>
}

// ── Zone Summary Card ─────────────────────────────────────────────────────────
function ZoneSummaryCard({ summary }: { summary: ZoneSummary }) {
  const [expanded, setExpanded] = useState(false)
  const age = summary.generated_at
    ? Math.round((Date.now() - new Date(summary.generated_at).getTime()) / 60000)
    : null

  return (
    <div className={`card border rounded-xl transition-all ${riskBg(summary.risk_level)}`}>
      <div className="flex items-start justify-between gap-3 cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-sm font-semibold text-text-primary">{summary.zone_name}</span>
            {riskBadge(summary.risk_level)}
            {age !== null && (
              <span className="text-[10px] font-mono text-text-muted ml-auto">
                {age}m ago
              </span>
            )}
          </div>
          <p className={`text-xs leading-relaxed ${expanded ? 'text-text-secondary' : 'text-text-muted line-clamp-2'}`}>
            {summary.summary}
          </p>
        </div>
        <button className="text-text-muted flex-shrink-0 mt-0.5">
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-surface-600 space-y-3 animate-fade-in">
          {/* Key events */}
          {summary.key_events.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">Key Events</p>
              <ul className="space-y-1">
                {summary.key_events.map((ev, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                    <span className="text-accent-amber flex-shrink-0 mt-0.5">•</span>
                    {ev}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Stats row */}
          <div className="flex gap-4 text-xs font-mono">
            <span className="text-text-muted">
              Persons: <span className="text-accent-cyan">{summary.person_count}</span>
            </span>
            <span className="text-text-muted">
              Alerts: <span className={summary.alert_count > 0 ? 'text-accent-red' : 'text-accent-green'}>{summary.alert_count}</span>
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Shift report card ─────────────────────────────────────────────────────────
function ShiftReportCard({ report }: { report: ShiftReport }) {
  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText size={15} className="text-accent-cyan" />
          <span className="text-sm font-semibold text-text-primary">Shift Report</span>
          <span className="text-xs text-text-muted font-mono capitalize">{report.shift}</span>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="text-accent-red">{report.high_severity} HIGH</span>
          <span className="text-text-muted">{report.total_alerts} total alerts</span>
        </div>
      </div>

      {/* Summary paragraph */}
      <div className="bg-surface-700 rounded-lg p-3 border border-surface-600">
        <div className="flex items-center gap-1.5 mb-2">
          <Sparkles size={11} className="text-accent-cyan" />
          <span className="text-[10px] font-semibold text-accent-cyan uppercase tracking-wider">AI Summary</span>
        </div>
        <p className="text-sm text-text-secondary leading-relaxed">{report.summary}</p>
      </div>

      {/* Recommendations */}
      {report.recommendations?.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">Recommendations</p>
          <ul className="space-y-2">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                <CheckCircle size={12} className="text-accent-green flex-shrink-0 mt-0.5" />
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── VLM model comparison table ────────────────────────────────────────────────
function VLMComparisonTable() {
  const models = [
    { name: 'Groq (Llama 4 Scout)', cost: 'Free tier', latency: '~300ms', accuracy: '★★★★☆', privacy: '☁ Cloud', recommended: true, note: 'Fastest, best value' },
    { name: 'GPT-4o (OpenAI)',      cost: '$0.003/img', latency: '1–3s',   accuracy: '★★★★★', privacy: '☁ Cloud', recommended: false, note: 'Best accuracy' },
    { name: 'Claude Haiku',         cost: '$0.001/img', latency: '1–2s',   accuracy: '★★★★☆', privacy: '☁ Cloud', recommended: false, note: 'Structured output' },
    { name: 'Gemini 1.5 Flash',     cost: 'Free tier',  latency: '0.5–2s', accuracy: '★★★★☆', privacy: '☁ Cloud', recommended: false, note: 'Fast + free' },
    { name: 'LLaVA (Ollama)',       cost: 'Free',       latency: '2–8s',   accuracy: '★★★☆☆', privacy: '✅ Local', recommended: false, note: 'Full privacy' },
    { name: 'Qwen-VL (Ollama)',     cost: 'Free',       latency: '1–5s',   accuracy: '★★★★☆', privacy: '✅ Local', recommended: false, note: 'Best local VLM' },
  ]
  return (
    <div className="card">
      <p className="section-title">VLM Model Comparison</p>
      <div className="overflow-x-auto">
        <table className="data-table text-xs">
          <thead>
            <tr>
              <th>Model</th>
              <th>Cost</th>
              <th>Latency</th>
              <th>Accuracy</th>
              <th>Privacy</th>
              <th>Best for</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m.name}>
                <td>
                  <span className="font-medium text-text-primary">{m.name}</span>
                  {m.recommended && (
                    <span className="ml-2 text-[9px] px-1.5 py-0.5 rounded bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20 font-mono">
                      RECOMMENDED
                    </span>
                  )}
                </td>
                <td className="font-mono text-accent-green">{m.cost}</td>
                <td className="font-mono">{m.latency}</td>
                <td>{m.accuracy}</td>
                <td>{m.privacy}</td>
                <td className="text-text-muted">{m.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Pipeline architecture diagram ─────────────────────────────────────────────
function PipelineFlow() {
  const steps = [
    { icon: '📹', label: 'Camera Frame',   sub: 'RTSP / mp4', color: 'border-accent-cyan/30 bg-cyan-950/20' },
    { icon: '🔍', label: 'YOLO Detect',    sub: 'Person bbox', color: 'border-purple-500/30 bg-purple-950/20' },
    { icon: '🏷️', label: 'Track + Zone',   sub: '01-P1025, zone', color: 'border-purple-500/30 bg-purple-950/20' },
    { icon: '👁️', label: 'VLM Analyze',    sub: 'Groq / GPT-4o', color: 'border-accent-amber/30 bg-amber-950/20' },
    { icon: '📋', label: 'Rules Engine',   sub: 'Anomaly?', color: 'border-red-500/30 bg-red-950/20' },
    { icon: '🧠', label: 'LLM Explain',    sub: 'Groq Llama 3', color: 'border-accent-amber/30 bg-amber-950/20' },
    { icon: '📊', label: 'Zone Summary',   sub: 'Every 30s',   color: 'border-accent-green/30 bg-green-950/20' },
  ]
  return (
    <div className="card">
      <p className="section-title">Intelligence Pipeline</p>
      <div className="flex flex-wrap gap-1.5 items-center">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <div className={`flex flex-col items-center border rounded-lg px-3 py-2 min-w-[80px] ${step.color}`}>
              <span className="text-lg leading-none mb-1">{step.icon}</span>
              <span className="text-[11px] font-medium text-text-primary text-center">{step.label}</span>
              <span className="text-[9px] text-text-muted text-center">{step.sub}</span>
            </div>
            {i < steps.length - 1 && (
              <span className="text-text-muted text-sm">→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export function IntelligencePage() {
  const qc = useQueryClient()

  const { data: zonesData, isLoading: zonesLoading } = useQuery<ZoneSummaryResponse>({
    queryKey:       ['summaries', 'zones'],
    queryFn:        () => fetch('/api/v1/summaries/zones').then(r => r.json()),
    refetchInterval:60_000,
  })

  const { data: reportData } = useQuery<ShiftReport>({
    queryKey:       ['summaries', 'shift-report'],
    queryFn:        () => fetch('/api/v1/summaries/shift-report').then(r => r.json()),
    refetchInterval:60_000,
  })

  // Listen for real-time WS summary updates
  useEffect(() => {
    const handler = (e: CustomEvent) => {
      const evt = e.detail
      if (evt?.type === 'zone_summary' || evt?.type === 'shift_report') {
        qc.invalidateQueries({ queryKey: ['summaries'] })
      }
    }
    window.addEventListener('ws_event', handler as EventListener)
    return () => window.removeEventListener('ws_event', handler as EventListener)
  }, [qc])

  const summaries: ZoneSummary[] = zonesData?.summaries ?? []
  const hasRealData = summaries.some(s => s.summary && !s.summary.includes('No summary'))

  return (
    <div className="p-5 h-full overflow-y-auto space-y-5">

      {/* Header banner */}
      <div className="card bg-gradient-to-r from-purple-950/40 to-cyan-950/40 border-purple-800/40">
        <div className="flex items-start gap-4">
          <div className="p-2.5 rounded-xl bg-purple-900/30 border border-purple-700/30">
            <Brain size={20} className="text-purple-400" />
          </div>
          <div className="flex-1">
            <h2 className="text-base font-semibold text-text-primary mb-1">AI Intelligence Layer</h2>
            <p className="text-xs text-text-secondary leading-relaxed max-w-2xl">
              Vision Language Models analyze individual worker crops; Large Language Models synthesize
              zone-level narratives and shift reports. Using{' '}
              <span className="font-mono text-accent-cyan">Groq (Llama 3.3 70B)</span> for text reasoning
              and{' '}
              <span className="font-mono text-accent-cyan">Llama 4 Scout</span> for visual analysis.
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {hasRealData
              ? <><span className="live-dot" /><span className="text-xs font-mono text-accent-green">LIVE</span></>
              : <span className="text-xs font-mono text-text-muted">MOCK MODE</span>
            }
          </div>
        </div>
      </div>

      {/* Pipeline flow */}
      <PipelineFlow />

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

        {/* Zone summaries — 2 cols */}
        <div className="lg:col-span-2 space-y-3">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={14} className="text-accent-cyan" />
            <span className="text-sm font-semibold text-text-primary">Zone Intelligence</span>
            <span className="text-xs text-text-muted font-mono ml-auto">
              {summaries.length} zones • updated every 30s
            </span>
          </div>

          {zonesLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="card animate-pulse h-20 bg-surface-700" />
              ))}
            </div>
          ) : summaries.length === 0 ? (
            <div className="card text-center py-10 space-y-3">
              <Brain size={32} className="mx-auto text-text-muted opacity-30" />
              <p className="text-sm text-text-muted">No zone summaries yet.</p>
              <p className="text-xs text-text-muted">
                Start the AI service with{' '}
                <code className="font-mono text-accent-cyan">USE_LLM=true</code>{' '}
                to generate summaries every 30 seconds.
              </p>
              <div className="bg-surface-700 rounded-lg p-3 mt-2 text-left max-w-sm mx-auto">
                <p className="text-[10px] font-mono text-text-muted mb-1">For instant demo (mock data):</p>
                <p className="text-[10px] font-mono text-accent-cyan">LLM_BACKEND=mock python main.py</p>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              {summaries.map(s => (
                <ZoneSummaryCard key={s.zone_id} summary={s} />
              ))}
            </div>
          )}
        </div>

        {/* Shift report — 1 col */}
        <div className="space-y-4">
          {reportData && <ShiftReportCard report={reportData} />}

          {/* Setup guide */}
          <div className="card space-y-3">
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-accent-cyan" />
              <span className="text-sm font-semibold text-text-primary">Quick Setup (Groq)</span>
            </div>
            <div className="space-y-2 text-xs">
              {[
                { step: '1', text: 'Sign up free at console.groq.com' },
                { step: '2', text: 'Create an API key (takes 30s)' },
                { step: '3', text: 'Add to .env: GROQ_API_KEY=gsk_...' },
                { step: '4', text: 'Set: LLM_BACKEND=groq' },
                { step: '5', text: 'Set: VLM_BACKEND=groq' },
                { step: '6', text: 'Restart: python main.py' },
              ].map(({ step, text }) => (
                <div key={step} className="flex items-start gap-2">
                  <span className="w-5 h-5 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan text-[10px] font-bold flex items-center justify-center flex-shrink-0">
                    {step}
                  </span>
                  <span className="text-text-secondary">{text}</span>
                </div>
              ))}
            </div>
            <div className="bg-surface-700 rounded p-2 font-mono text-[10px] text-accent-cyan border border-surface-600">
              # Free tier: 14,400 req/day<br />
              # Llama-3.3-70B: 1000+ tok/s<br />
              # No credit card needed
            </div>
          </div>
        </div>
      </div>

      {/* VLM comparison */}
      <VLMComparisonTable />

      {/* Prompt examples */}
      <div className="card">
        <p className="section-title">VLM Prompt Examples</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            {
              title: 'Activity Detection',
              badge: 'Person crop',
              color: 'border-accent-cyan/30',
              prompt: `You are a warehouse safety AI.\nContext: Storage Area\n\nAnalyze this image:\nACTIVITY: [walking/standing/...]\nANOMALY: [normal/anomaly]\nSEVERITY: [none/low/medium/high]\nDESCRIPTION: [one sentence]`
            },
            {
              title: 'Scene Understanding',
              badge: 'Full frame',
              color: 'border-accent-amber/30',
              prompt: `Analyze this warehouse camera frame.\nCount visible workers.\nDescribe the primary activity.\nNote any safety concerns.\nRate operational status: normal/alert/critical.`
            },
            {
              title: 'Anomaly Explanation',
              badge: 'LLM text',
              color: 'border-accent-red/30',
               prompt: `Alert: unauthorized_access\nZone: Restricted Area\nPerson: 01-P1025, dwell: 45s\n\nExplain: why flagged, what normally happens here, recommended action, false positive probability.`
            },
          ].map(({ title, badge, color, prompt }) => (
            <div key={title} className={`border rounded-lg p-3 space-y-2 ${color} bg-surface-800`}>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-text-primary">{title}</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-surface-700 text-text-muted font-mono">{badge}</span>
              </div>
              <pre className="text-[10px] font-mono text-text-muted whitespace-pre-wrap leading-relaxed bg-surface-900 rounded p-2">
                {prompt}
              </pre>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
