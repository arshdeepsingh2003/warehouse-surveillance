export interface Camera {
  id: string;
  name: string;
  rtsp_url: string;
  zone_id: string;
  status: 'online' | 'offline';
}

export interface Alert {
  id: string;
  camera_id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface Activity {
  id: string;
  camera_id: string;
  person_id: string;
  activity_type: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface Person {
  id: string;
  track_id: string;
  appearance_count: number;
  first_seen: string;
  last_seen: string;
}

export interface Zone {
  id: string;
  name: string;
  cameras: Camera[];
}
