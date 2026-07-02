// store/notificationStore.ts
import { create } from 'zustand'
import type { Alert } from '../types'
import { playAlarm } from '../utils/sound'

export interface ToastItem {
  id: string
  alert: Alert
  isTheft: boolean
  duration: number // ms
}

interface NotificationState {
  toasts: ToastItem[]
  soundEnabled: boolean
  desktopEnabled: boolean
  addToast: (alert: Alert) => void
  removeToast: (id: string) => void
  toggleSound: () => void
  toggleDesktop: () => void
  syncDesktopPermission: () => void
}

const STORAGE_SOUND_KEY = 'surveillance_sound_enabled'
const STORAGE_DESKTOP_KEY = 'surveillance_desktop_enabled'

export const useNotificationStore = create<NotificationState>((set, get) => {
  // Read initial configuration from localStorage
  const savedSound = localStorage.getItem(STORAGE_SOUND_KEY)
  const initialSound = savedSound !== null ? savedSound === 'true' : true

  const savedDesktop = localStorage.getItem(STORAGE_DESKTOP_KEY)
  const initialDesktop = savedDesktop === 'true'

  return {
    toasts: [],
    soundEnabled: initialSound,
    desktopEnabled: initialDesktop && ('Notification' in window) && Notification.permission === 'granted',

    addToast: (alert) => {
      const isTheft = alert.alert_type === 'theft_attempt'
      if (!isTheft) return // Only notify on theft attempts

      const id = `${alert.id}-${Date.now()}`
      const duration = 12000 // 12s for theft

      // 1. Play synthesized audio if enabled
      if (get().soundEnabled) {
        playAlarm()
      }

      // 2. Trigger native browser notification if enabled
      if (get().desktopEnabled && ('Notification' in window) && Notification.permission === 'granted') {
        const title = '🚨 CRITICAL: Theft Attempt Detected!'
        const body = `Camera: ${alert.camera_id} (${alert.zone.replace(/_/g, ' ')})\n${alert.description}`
        
        try {
          new Notification(title, {
            body,
            icon: '/vite.svg',
            tag: alert.id,
            requireInteraction: true // Keep native notification visible for theft
          })
        } catch (e) {
          console.warn('Native notification failed:', e)
        }
      }

      const newToast: ToastItem = { id, alert, isTheft, duration }

      set((state) => ({
        toasts: [...state.toasts, newToast],
      }))

      // Auto-remove toast
      setTimeout(() => {
        get().removeToast(id)
      }, duration)
    },

    removeToast: (id) =>
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      })),

    toggleSound: () =>
      set((state) => {
        const next = !state.soundEnabled
        localStorage.setItem(STORAGE_SOUND_KEY, String(next))
        return { soundEnabled: next }
      }),

    toggleDesktop: () => {
      const enabled = get().desktopEnabled
      if (!enabled) {
        // Requesting permission
        if (!('Notification' in window)) {
          alert('Desktop notifications are not supported in this browser.')
          return
        }

        Notification.requestPermission().then((permission) => {
          const granted = permission === 'granted'
          localStorage.setItem(STORAGE_DESKTOP_KEY, String(granted))
          set({ desktopEnabled: granted })
        })
      } else {
        localStorage.setItem(STORAGE_DESKTOP_KEY, 'false')
        set({ desktopEnabled: false })
      }
    },

    syncDesktopPermission: () => {
      if (!('Notification' in window)) return
      const hasPermission = Notification.permission === 'granted'
      const savedPref = localStorage.getItem(STORAGE_DESKTOP_KEY) === 'true'
      set({ desktopEnabled: hasPermission && savedPref })
    }
  }
})
