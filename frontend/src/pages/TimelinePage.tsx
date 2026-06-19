// pages/TimelinePage.tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { format } from 'date-fns'
import { Search, MapPin, Clock, Camera } from 'lucide-react'
import { api } from '../api/client'
import { AnomalyBadge, EmptyState } from '../components/common/SeverityBadge'
import './Timeline.css'

const ANOMALY_ACTIVITIES = ['unauthorized_entry', 'falling', 'loitering']

export function TimelinePage() {
  const [personId, setPersonId] = useState('01-P1025')
  const [input,    setInput]    = useState('01-P1025')

  const { data: timeline, isLoading, error } = useQuery({
    queryKey: ['timeline', personId],
    queryFn:  () => api.activities.personTimeline(personId),
    enabled:  !!personId,
  })

  const handleSearch = () => setPersonId(input.trim())

  return (
    <div className="timeline-page">
      <div className="timeline-section timeline-search">
        <div className="timeline-search-input-wrap">
          <Search size={13} className="timeline-search-icon" />
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            placeholder="Enter Person ID (e.g. 01-P1025)"
            className="timeline-search-input"
          />
        </div>
        <button onClick={handleSearch} className="timeline-track-btn">
          Track
        </button>
      </div>

      {isLoading && <div className="timeline-loading">Loading timeline…</div>}
      {error    && <div className="timeline-error">Person not found or no activity recorded.</div>}

      {timeline && (
        <div className="card timeline-card">
          <div className="timeline-person-header">
            <div className="timeline-person-avatar">
              <span className="timeline-person-avatar-text">
                {timeline.person_id}
              </span>
            </div>
            <div>
              <div className="timeline-person-name">{timeline.person_id}</div>
              <div className="timeline-person-meta">{timeline.timeline.length} zone transitions recorded</div>
            </div>
          </div>

          {timeline.timeline.length === 0
            ? <EmptyState message="No timeline data for this person" />
            : (
              <div className="timeline-list">
                <div className="timeline-line" />

                {timeline.timeline.map((step, i) => {
                  const isAnomaly = ANOMALY_ACTIVITIES.includes(step.activity_type)
                  const isLast    = i === timeline.timeline.length - 1
                  const dotClass = isAnomaly ? 'timeline-dot-anomaly'
                    : isLast ? 'timeline-dot-last' : 'timeline-dot-normal'

                  return (
                    <div key={i} className="timeline-entry animate-fade-in" style={{ animationDelay: `${i * 60}ms` }}>
                      <div className={`timeline-dot ${dotClass}`}
                        style={isAnomaly ? { boxShadow: '0 0 8px #ff1744' } : {}}
                      />

                      <div className={`timeline-entry-card ${isAnomaly ? 'timeline-entry-card-anomaly' : ''}`}>
                        <div className="timeline-entry-body">
                          <div className="timeline-entry-main">
                            <div className="timeline-entry-header">
                              <span className="timeline-entry-zone">{step.zone.replace(/_/g, ' ')}</span>
                              <AnomalyBadge label={isAnomaly ? 'anomaly' : 'normal'} />
                            </div>
                            <p className="timeline-entry-desc">{step.description}</p>
                            <div className="timeline-entry-meta">
                              <span className="timeline-entry-meta-item">
                                <Camera size={10} />{step.camera_id}
                              </span>
                              <span className="timeline-entry-meta-item">
                                <Clock size={10} />
                                {format(new Date(step.entry_time), 'HH:mm:ss')}
                                {step.exit_time && ` → ${format(new Date(step.exit_time), 'HH:mm:ss')}`}
                              </span>
                              <span className="timeline-entry-meta-item">
                                <MapPin size={10} />{step.activity_type.replace(/_/g, ' ')}
                              </span>
                              <span className="timeline-entry-meta-item">
                                {step.dwell_seconds >= 60
                                  ? `${Math.floor(step.dwell_seconds / 60)}m ${step.dwell_seconds % 60}s dwell`
                                  : `${step.dwell_seconds}s dwell`}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )
          }
        </div>
      )}
    </div>
  )
}
