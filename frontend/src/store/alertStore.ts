import { create } from 'zustand';

interface AlertStore {
  alerts: any[];
  setAlerts: (alerts: any[]) => void;
}

export const useAlertStore = create<AlertStore>((set) => ({
  alerts: [],
  setAlerts: (alerts) => set({ alerts }),
}));
