// store/navigationStore.ts
import { create } from 'zustand'

interface NavigationState {
  page: string
  selectedCameraId: string | null
  setPage: (page: string) => void
  navigateToCamera: (cameraId: string) => void
  clearSelectedCamera: () => void
}

export const useNavigationStore = create<NavigationState>((set) => ({
  page: 'dashboard',
  selectedCameraId: null,
  setPage: (page) => set({ page, selectedCameraId: null }),
  navigateToCamera: (cameraId) => set({ page: 'livefeed', selectedCameraId: cameraId }),
  clearSelectedCamera: () => set({ selectedCameraId: null }),
}))
