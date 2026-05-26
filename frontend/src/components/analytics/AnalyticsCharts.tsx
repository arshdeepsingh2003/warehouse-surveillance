// components/analytics/AnalyticsCharts.tsx
import { useQuery } from '@tanstack/react-query'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'
import { api } from '../../api/client'
import './AnalyticsCharts.css'

const ZONE_COLORS = ['#ff1744', '#ffab00', '#00e5ff', '#00e676', '#d500f9']

const chartStyle = {
  background:  'transparent',
  fontFamily:  "'JetBrains Mono', monospace",
  fontSize:    10,
  fill:        '#8892a4',
}

export function AlertTrendChart() {
  const { data } = useQuery({
    queryKey: ['analytics', 'summary'],
    queryFn:  api.analytics.summary,
    refetchInterval: 120_000,
  })

  const trend = data?.alert_trend ?? []

  return (
    <div className="card chart-card">
      <p className="chart-title">Alert Trend — Last 24 h</p>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={trend} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="alertGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#ff1744" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#ff1744" stopOpacity={0}    />
            </linearGradient>
          </defs>
          <XAxis dataKey="hour" tick={chartStyle} axisLine={false} tickLine={false} interval={2} />
          <YAxis tick={chartStyle} axisLine={false} tickLine={false} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: '#151d30', border: '1px solid #253050', borderRadius: 6, fontSize: 11 }}
            labelStyle={{ color: '#8892a4' }}
            itemStyle={{ color: '#ff1744' }}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke="#ff1744"
            strokeWidth={1.5}
            fill="url(#alertGrad)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export function ZoneRiskChart() {
  const { data } = useQuery({
    queryKey: ['analytics', 'summary'],
    queryFn:  api.analytics.summary,
    refetchInterval: 120_000,
  })

  const zones = data?.zone_risk ?? []

  return (
    <div className="card chart-card">
      <p className="chart-title">Zone-wise Incidents</p>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={zones}
            dataKey="incidents"
            nameKey="zone"
            cx="50%"
            cy="50%"
            outerRadius={70}
            innerRadius={40}
            strokeWidth={0}
          >
            {zones.map((_, i) => (
              <Cell key={i} fill={ZONE_COLORS[i % ZONE_COLORS.length]} opacity={0.85} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#151d30', border: '1px solid #253050', borderRadius: 6, fontSize: 11 }}
            formatter={(v: unknown) => [`${v} incidents`]}
          />
          <Legend
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: 10, color: '#8892a4', fontFamily: "'JetBrains Mono', monospace" }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
