// components/layout/Header.tsx
import { Bell, Wifi, WifiOff } from 'lucide-react'
import { useAlertStore } from '../../store/alertStore'
import './Header.css'

interface Props { wsConnected: boolean; pageTitle: string }

export function Header({ wsConnected, pageTitle }: Props) {
  const liveAlerts  = useAlertStore(s => s.liveAlerts)
  const activeCount = liveAlerts.filter(a => a.status === 'active').length
  const now = new Date()

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

        <div className="header-alert-bell">
          <Bell size={16} />
          {activeCount > 0 && (
            <span className="header-alert-count">
              {activeCount > 9 ? '9+' : activeCount}
            </span>
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
