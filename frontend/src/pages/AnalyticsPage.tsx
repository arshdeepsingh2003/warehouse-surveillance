// pages/AnalyticsPage.tsx
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { AlertTrendChart, ZoneRiskChart } from '../components/analytics/AnalyticsCharts'
import { TrendingUp, AlertTriangle, MapPin, Clock } from 'lucide-react'
import './Analytics.css'

const ZONE_BAR_COLORS = ['#ff1744', '#ffab00', '#00e5ff']

export function AnalyticsPage() {
  const { data } = useQuery({
    queryKey: ['analytics', 'summary'],
    queryFn:  api.analytics.summary,
    refetchInterval: 120_000,
  })

  const zones = data?.zone_risk ?? []

  return (
    <div className="analytics-page">
      <div className="analytics-chart-grid analytics-section">
        <AlertTrendChart />
        <ZoneRiskChart />
      </div>

      <div className="card analytics-section">
        <p className="analytics-zone-title">Zone Risk Breakdown</p>
        <table className="data-table">
          <thead>
            <tr>
              <th><MapPin size={11} className="inline mr-1" />Zone</th>
              <th><AlertTriangle size={11} className="inline mr-1" />Incidents</th>
              <th><TrendingUp size={11} className="inline mr-1" />Share</th>
              <th>Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {zones.map((z, i) => (
              <tr key={i}>
                <td className="analytics-zone-cell">{z.zone}</td>
                <td className="analytics-incident-cell">{z.incidents}</td>
                <td>
                  <div className="analytics-bar-wrap">
                    <div className="analytics-bar-track">
                      <div
                        className="analytics-bar-fill"
                        style={{
                          width: `${z.percentage}%`,
                          background: ZONE_BAR_COLORS[i] ?? '#00e5ff'
                        }}
                      />
                    </div>
                    <span className="analytics-bar-pct">{z.percentage}%</span>
                  </div>
                </td>
                <td>
                  <span className={
                    z.percentage >= 35 ? 'badge-high' :
                    z.percentage >= 15 ? 'badge-medium' : 'badge-low'
                  }>
                    {z.percentage >= 35 ? 'HIGH' : z.percentage >= 15 ? 'MEDIUM' : 'LOW'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data?.summary && (
        <div className="analytics-stat-grid analytics-section">
          {[
            { label: 'Most Risky Zone',   value: data.summary.most_risky_zone,    icon: <MapPin size={14} /> },
            { label: 'Peak Activity',     value: data.summary.peak_activity_hour, icon: <Clock size={14} /> },
            { label: 'High Severity',     value: data.summary.high_severity_alerts, icon: <AlertTriangle size={14} /> },
            { label: 'People Detected',   value: data.summary.people_detected,    icon: <TrendingUp size={14} /> },
          ].map(({ label, value, icon }) => (
            <div key={label} className="card analytics-stat-card">
              <div className="analytics-stat-icon">{icon}</div>
              <div>
                <div className="analytics-stat-value">{value}</div>
                <div className="analytics-stat-label">{label}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
