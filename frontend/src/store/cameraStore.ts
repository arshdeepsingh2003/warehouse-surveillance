// store/cameraStore.ts
// Zustand store for camera status.
// Real-time status updates from WebSocket are applied here
// so the camera grid reflects live online/offline state.

import { create } from 'zustand'
import type { Camera, CameraStatus } from '../types'

interface CameraState {
  cameras: Camera[]
  setCameras: (cameras: Camera[]) => void
  updateCameraStatus: (id: string, status: CameraStatus, fps: number, latency_ms: number) => void
}

export const useCameraStore = create<CameraState>((set) => ({
  cameras: [],

  setCameras: (cameras) => set({ cameras }),

  updateCameraStatus: (id, status, fps, latency_ms) =>
    set((state) => ({
      cameras: state.cameras.map((c) =>
        c.id === id ? { ...c, status, fps, latency_ms } : c
      ),
    })),
}))
