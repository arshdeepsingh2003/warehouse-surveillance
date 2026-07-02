// components/layout/Header.tsx
import { useState, useRef, useEffect } from 'react'
import { Bell, Wifi, WifiOff, Volume2, VolumeX, Monitor, Settings } from 'lucide-react'
import { useAlertStore } from '../../store/alertStore'
import { useNotificationStore } from '../../store/notificationStore'
import './Header.css'

interface Props { wsConnected: boolean; pageTitle: string }

export function Header({ wsConnected, pageTitle }: Props) {
  const liveAlerts  = useAlertStore(s => s.liveAlerts)
  const activeCount = liveAlerts.filter(a => a.status === 'active').length
  const now = new Date()

  const { soundEnabled, desktopEnabled, toggleSound, toggleDesktop, syncDesktopPermission } = useNotificationStore()
  const [popoverOpen, setPopoverOpen] = useState(false)
  const popoverRef = useRef<HTMLDivElement>(null)

  // Sync browser notification permission state on mount
  useEffect(() => {
    syncDesktopPermission()
  }, [syncDesktopPermission])

  // Handle clicking outside the popover to close it
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setPopoverOpen(false)
      }
    }
    if (popoverOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [popoverOpen])

  return (
    <header className="header">
      <h1 className="header-title">{pageTitle}</h1>

      <div className="header-right">
        <span className="header-timestamp header-timestamp-visible">
          {now.toLocaleTimeString('en-IN', { hour12: false })} &nbsp;·&nbsp; {now.toLocaleDateString('en-IN')}
        </span>

        <div className={`header-ws ${wsConnected ? 'header-ws-online' : 'header-ws-offline'}`}>
          {wsConnected
            ? <><Wifi size={12} /><span className="header-ws-label">LIVE</span></>
            : <><WifiOff size={12} /><span className="header-ws-label">OFFLINE</span></>
          }
        </div>

        {/* Interactive Bell and Notification settings popover */}
        <div className="header-alert-bell-container" ref={popoverRef}>
          <div 
            className={`header-alert-bell ${popoverOpen ? 'header-alert-bell-active' : ''}`}
            onClick={() => setPopoverOpen(!popoverOpen)}
          >
            <Bell size={16} />
            {activeCount > 0 && (
              <span className="header-alert-count">
                {activeCount > 9 ? '9+' : activeCount}
              </span>
            )}
          </div>

          {popoverOpen && (
            <div className="header-popover card">
              <div className="popover-header">
                <Settings size={12} className="popover-header-icon" />
                <span className="popover-title">Surveillance Alerts</span>
              </div>
              
              <div className="popover-stats">
                <span className="popover-stat-label">Active Anomaly Alerts:</span>
                <span className={`popover-stat-value ${activeCount > 0 ? 'text-accent-red font-bold' : ''}`}>
                  {activeCount}
                </span>
              </div>

              <div className="popover-divider" />

              <div className="popover-settings-list">
                <div className="popover-setting-item">
                  <div className="popover-setting-label-block">
                    <span className="popover-setting-title">Audio Warnings</span>
                    <span className="popover-setting-desc">Synthesized sirens on theft events</span>
                  </div>
                  <button 
                    onClick={toggleSound} 
                    className={`popover-toggle-btn ${soundEnabled ? 'btn-enabled' : 'btn-disabled'}`}
                  >
                    {soundEnabled ? <Volume2 size={12} /> : <VolumeX size={12} />}
                    {soundEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>

                <div className="popover-setting-item">
                  <div className="popover-setting-label-block">
                    <span className="popover-setting-title">Desktop Push</span>
                    <span className="popover-setting-desc">Browser desktop notifications</span>
                  </div>
                  <button 
                    onClick={toggleDesktop} 
                    className={`popover-toggle-btn ${desktopEnabled ? 'btn-enabled' : 'btn-disabled'}`}
                  >
                    <Monitor size={12} />
                    {desktopEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="header-system-status">
          <span className="live-dot" />
          <span className="header-system-label">ALL SYSTEMS</span>
        </div>
      </div>
    </header>
  )
}
