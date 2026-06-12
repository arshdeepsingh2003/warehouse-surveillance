// types/index.ts
// These mirror the backend Pydantic schemas exactly.

// ── Cameras ───────────────────────────────────────────────────────────────────
export type CameraStatus = 'online' | 'offline' | 'unknown'
export type CameraType   = 'rtsp' | 'file' | 'mock'

export interface Camera {
  id:          string
  name:        string
  location:    string
  zone:        string
  stream_url:  string
  camera_type: CameraType
  status:      CameraStatus
  fps:         number
  latency_ms:  number
  created_at:  string
  updated_at:  string
}

// ── Alerts ────────────────────────────────────────────────────────────────────
export type AlertSeverity = 'low' | 'medium' | 'high'
export type AlertStatus   = 'active' | 'resolved'
export type AlertType     =
  | 'unauthorized_access'
  | 'loitering'
  | 'ppe_violation'
  | 'worker_fall'
  | 'restricted_zone_entry'
  | 'suspicious_activity'
  | 'theft_attempt'
  | 'unknown'

export interface Alert {
  id:           string
  camera_id:    string
  zone:         string
  alert_type:   AlertType
  severity:     AlertSeverity
  description:  string
  person_id:    string | null
  snapshot_url: string | null
  status:       AlertStatus
  confidence:   number
  triggered_at: string
  resolved_at:  string | null
  resolved_by:  string | null
}

// ── Activities ────────────────────────────────────────────────────────────────
export type ActivityType  = 'walking' | 'standing' | 'carrying_object' | 'loitering' |
                            'running' | 'falling'  | 'handling_items'  | 'unauthorized_entry' | 'unknown'
export type AnomalyLabel  = 'normal' | 'anomaly'

export interface Activity {
  id:              string
  person_id:       string
  camera_id:       string
  zone:            string
  activity_type:   ActivityType
  description:     string
  anomaly_label:   AnomalyLabel
  dwell_seconds:   number
  confidence:      number
  timestamp:       string
  objects_detected: string[]
  backend_used:    string
  latency_ms:      number
}

// ── VLM Insights ──────────────────────────────────────────────────────────────
export interface VLMInsight {
  id:               string
  person_id:        string
  camera_id:        string
  zone:             string
  activity_type:    string
  anomaly_label:    string
  description:      string
  confidence:       number
  objects_detected: string[]
  backend_used:     string
  latency_ms:       number
  source:           string    // "vlm" | "hybrid"
  timestamp:        string
}

export interface VLMInsightEvent {
  type:             'vlm_insight'
  insight_id:       string
  person_id:        string
  camera_id:        string
  zone:             string
  activity_type:    string
  anomaly_label:    string
  description:      string
  confidence:       number
  objects_detected: string[]
  backend_used:     string
  latency_ms:       number
  source:           string
  timestamp:        string
}

export interface WSVLMInsightEvent {
  type:             'vlm_insight'
  person_id:        string
  camera_id:        string
  zone:             string
  description:      string
  activity_type:    string
  anomaly_label:    string
  confidence:       number
  objects_detected: string[]
  backend_used:     string
  latency_ms:       number
  timestamp:        string
}

export interface PersonTimelineEntry {
  zone:          string
  camera_id:     string
  activity_type: ActivityType
  description:   string
  entry_time:    string
  exit_time:     string | null
  dwell_seconds: number
}

export interface PersonTimeline {
  person_id: string
  timeline:  PersonTimelineEntry[]
}

// ── Analytics ─────────────────────────────────────────────────────────────────
export interface DashboardSummary {
  total_cameras:        number
  cameras_online:       number
  total_alerts_today:   number
  active_alerts:        number
  high_severity_alerts: number
  people_detected:      number
  most_risky_zone:      string
  peak_activity_hour:   string
  system_status:        string
}

export interface AlertTrendPoint { hour: string; count: number }
export interface ZoneRisk        { zone: string; incidents: number; percentage: number }

export interface AnalyticsSummary {
  summary:     DashboardSummary
  alert_trend: AlertTrendPoint[]
  zone_risk:   ZoneRisk[]
}

// ── WebSocket events ──────────────────────────────────────────────────────────
export type WSEventType = 'connected' | 'alert_triggered' | 'alert_resolved' |
                          'frame_update' | 'camera_status' | 'ping' |
                          'activity_update' | 'vlm_insight'

export interface WSFramePerson {
  person_id:     string
  zone:          string
  activity:      string
  dwell_seconds: number
  // Optional visual fields supplied by the AI service
  bbox?:         number[]   // [x1,y1,x2,y2]
  center?:       number[]   // [cx,cy]
  // VLM enrichment from AI service (included in frame_update payload)
  vlm_description?:   string
  vlm_anomaly_label?: string
  vlm_latency_ms?:    number
  vlm_backend_used?:  string
}

export interface WSEvent {
  type:        WSEventType
  timestamp:   string
  // alert_triggered / alert_resolved
  alert_id?:   string
  alert?:      Alert
  camera_id?:  string
  zone?:       string
  alert_type?: AlertType
  severity?:   AlertSeverity
  description?: string
  person_id?:  string
  confidence?: number
  snapshot_url?: string
  // frame_update
  persons?:    WSFramePerson[]
  // camera_status
  status?:     CameraStatus
  fps?:        number
  latency_ms?: number
  // connected
  message?:    string
  // activity_update fields
  activity_id?:   string
  activity?:      string
  label?:         string
  // vlm_insight fields
  insight_id?:    string
  activity_type?: string
  anomaly_label?: string
  source?:        string
  // shared
  objects_detected?: string[]
  backend_used?:  string
  // generic payload passthrough
  payload?:      unknown
}
