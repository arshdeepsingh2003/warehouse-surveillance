import { create } from 'zustand';

interface AnalyticsStore {
  data: any;
  setData: (data: any) => void;
}

export const useAnalyticsStore = create<AnalyticsStore>((set) => ({
  data: null,
  setData: (data) => set({ data }),
}));
