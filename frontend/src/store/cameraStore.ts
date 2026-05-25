import { create } from 'zustand';

interface CameraStore {
  cameras: any[];
  setCameras: (cameras: any[]) => void;
}

export const useCameraStore = create<CameraStore>((set) => ({
  cameras: [],
  setCameras: (cameras) => set({ cameras }),
}));
