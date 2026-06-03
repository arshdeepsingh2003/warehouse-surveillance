// store/trackingStore.ts
// Zustand store for real-time person tracking data.
// Receives frame_update events from WebSocket and stores the latest
// tracked persons per camera — used by the Live Feed page.

import { create } from 'zustand'
import type { WSFramePerson } from '../types'

const EMPTY_PERSONS: WSFramePerson[] = []

interface CameraTracking {
  persons:   WSFramePerson[]
  updatedAt: string
}

interface TrackingState {
  // Map of camera_id → latest tracking snapshot
  cameraTracking: Record<string, CameraTracking>
  updateTracking: (cameraId: string, persons: WSFramePerson[], timestamp: string) => void
  getPersonsForCamera: (cameraId: string) => WSFramePerson[]
}

export const useTrackingStore = create<TrackingState>((set, get) => ({
  cameraTracking: {},

  updateTracking: (cameraId, persons, timestamp) =>
    set(state => ({
      cameraTracking: {
        ...state.cameraTracking,
        [cameraId]: { persons, updatedAt: timestamp },
      },
    })),

  getPersonsForCamera: (cameraId) =>
    get().cameraTracking[cameraId]?.persons ?? EMPTY_PERSONS,
}))
