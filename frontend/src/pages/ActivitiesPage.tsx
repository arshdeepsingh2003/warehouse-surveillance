// pages/ActivitiesPage.tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import { Search, Filter } from 'lucide-react'
import { api } from '../api/client'
import { AnomalyBadge, EmptyState } from '../components/common/SeverityBadge'
import './Activities.css'

export function ActivitiesPage() {
  const [anomalyOnly, setAnomalyOnly] = useState(false)
  const [search, setSearch]           = useState('')

  const { data: activities = [], isLoading } = useQuery({
    queryKey: ['activities', anomalyOnly],
    queryFn:  () => api.activities.list({ anomaly_only: anomalyOnly, limit: 100 }),
    refetchInterval: 30_000,
  })

  const filtered = activities.filter(a =>
    !search ||
    a.person_id.toLowerCase().includes(search.toLowerCase()) ||
    a.zone.toLowerCase().includes(search.toLowerCase()) ||
    a.camera_id.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="activities-page">
      <div className="activities-controls">
        <div className="activities-search">
          <Search size={13} className="activities-search-icon" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search person, zone, camera…"
            className="activities-search-input"
          />
        </div>
        <button
          onClick={() => setAnomalyOnly(!anomalyOnly)}
          className={`activities-filter-btn ${anomalyOnly ? 'activities-filter-on' : 'activities-filter-off'}`}
        >
          <Filter size={12} /> Anomalies only
        </button>
        <span className="activities-count">{filtered.length} records</span>
      </div>

      <div className="card activities-table-wrap">
        <div className="activities-table-scroll">
          {isLoading ? (
            <div className="activities-loading">Loading activity log…</div>
          ) : filtered.length === 0 ? (
            <EmptyState message="No activities found" />
          ) : (
            <table className="data-table">
              <thead className="sticky top-0 z-10" style={{ background: 'var(--surface-800)' }}>
                <tr>
                  <th>Person</th>
                  <th>Camera</th>
                  <th>Zone</th>
                  <th>Activity</th>
                  <th>Label</th>
                  <th>Dwell</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(act => (
                  <tr key={act.id}>
                    <td>
                      <span className="activities-person-id">{act.person_id}</span>
                    </td>
                    <td>
                      <span className="activities-camera-id">{act.camera_id}</span>
                    </td>
                    <td>
                      <span className="activities-zone">{act.zone.replace(/_/g, ' ')}</span>
                    </td>
                    <td>
                      <span className="activities-activity-type">{act.activity_type.replace(/_/g, ' ')}</span>
                      <p className="activities-description">{act.description}</p>
                    </td>
                    <td><AnomalyBadge label={act.anomaly_label} /></td>
                    <td>
                      <span className="activities-dwell">
                        {act.dwell_seconds >= 60
                          ? `${Math.floor(act.dwell_seconds / 60)}m ${act.dwell_seconds % 60}s`
                          : `${act.dwell_seconds}s`}
                      </span>
                    </td>
                    <td>
                      <span className="activities-timestamp">
                        {formatDistanceToNow(new Date(act.timestamp), { addSuffix: true })}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
