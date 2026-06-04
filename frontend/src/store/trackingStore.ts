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

  updateTracking: (cameraId, persons, timestamp) => {
    console.log(`🔍 TRACE[store-update] camera=${cameraId} persons=${persons.length} ids=[${persons.map(p => p.person_id).join(', ')}] bboxes=[${persons.map(p => JSON.stringify(p.bbox)).join('; ')}]`)
    return set(state => ({
      cameraTracking: {
        ...state.cameraTracking,
        [cameraId]: { persons, updatedAt: timestamp },
      },
    }))
  },

  getPersonsForCamera: (cameraId) => {
    const result = get().cameraTracking[cameraId]?.persons ?? EMPTY_PERSONS
    console.log(`🔍 TRACE[store-get] camera=${cameraId} persons=${result.length} ids=[${result.map(p => p.person_id).join(', ')}]`)
    return result
  },
}))
