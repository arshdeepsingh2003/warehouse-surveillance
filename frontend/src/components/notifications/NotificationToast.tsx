// components/notifications/NotificationToast.tsx
import { useQueryClient } from '@tanstack/react-query'
import { ShieldAlert, Eye, CheckCircle, X, Camera } from 'lucide-react'
import { api } from '../../api/client'
import { useAlertStore } from '../../store/alertStore'
import { useNavigationStore } from '../../store/navigationStore'
import { useNotificationStore } from '../../store/notificationStore'
import type { ToastItem } from '../../store/notificationStore'
import { SeverityBadge } from '../common/SeverityBadge'
import './NotificationToast.css'

interface ToastProps {
  toast: ToastItem
  onClose: () => void
}

function Toast({ toast, onClose }: ToastProps) {
  const { alert, isTheft } = toast
  const qc = useQueryClient()
  const resolveInStore = useAlertStore((s) => s.resolveAlert)
  const navigateToCamera = useNavigationStore((s) => s.navigateToCamera)

  const handleResolve = async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await api.alerts.resolve(alert.id, 'operator@warehouse.com')
      resolveInStore(alert.id, 'operator@warehouse.com')
      qc.invalidateQueries({ queryKey: ['alerts'] })
      onClose()
    } catch (err) {
      console.error('Failed to resolve alert from toast:', err)
    }
  }

  const handleViewFeed = (e: React.MouseEvent) => {
    e.stopPropagation()
    navigateToCamera(alert.camera_id)
    onClose()
  }

  const sevClass =
    alert.severity === 'high'
      ? 'toast-high'
      : alert.severity === 'medium'
      ? 'toast-medium'
      : 'toast-low'

  const formattedType = alert.alert_type.replace(/_/g, ' ')

  return (
    <div className={`toast-card ${isTheft ? 'toast-theft-critical' : sevClass}`} onClick={handleViewFeed}>
      {/* Visual pulse glow for theft */}
      {isTheft && <div className="toast-theft-pulse-border" />}

      <div className="toast-header">
        <div className="toast-title-area">
          <div className="toast-icon-container">
            <ShieldAlert size={14} className={isTheft ? 'text-red-500 animate-pulse' : ''} />
          </div>
          <span className="toast-alert-title">
            {isTheft ? 'CRITICAL THEFT DETECTED' : formattedType}
          </span>
        </div>
        <button className="toast-close-btn" onClick={(e) => { e.stopPropagation(); onClose(); }} title="Dismiss">
          <X size={14} />
        </button>
      </div>

      <div className="toast-body">
        <div className="toast-content-wrapper">
          <p className="toast-desc">{alert.description}</p>
          <div className="toast-meta">
            <span className="toast-meta-item">
              <Camera size={10} />
              {alert.camera_id}
            </span>
            <span className="toast-meta-item">{alert.zone.replace(/_/g, ' ')}</span>
            <SeverityBadge severity={alert.severity} />
          </div>
        </div>

        {alert.snapshot_url && (
          <div className="toast-snapshot-wrapper">
            <img src={alert.snapshot_url} alt="Alert crop" className="toast-snapshot" />
          </div>
        )}
      </div>

      <div className="toast-actions">
        <button className="toast-action-btn toast-btn-primary" onClick={handleViewFeed}>
          <Eye size={12} /> View Feed
        </button>
        {alert.status === 'active' && (
          <button className="toast-action-btn toast-btn-success" onClick={handleResolve}>
            <CheckCircle size={12} /> Resolve
          </button>
        )}
      </div>
    </div>
  )
}

export function NotificationToastContainer() {
  const toasts = useNotificationStore((s) => s.toasts)
  const removeToast = useNotificationStore((s) => s.removeToast)

  if (toasts.length === 0) return null

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <Toast key={toast.id} toast={toast} onClose={() => removeToast(toast.id)} />
      ))}
    </div>
  )
}
