// components/layout/Sidebar.tsx
import { useState } from 'react'
import {
  LayoutDashboard, Video, Bell, ClipboardList,
  BarChart3, User, BrainCircuit, Settings, Shield, ChevronLeft, ChevronRight
} from 'lucide-react'
import './Sidebar.css'

interface NavItem { label: string; icon: React.ReactNode; page: string }

const NAV: NavItem[] = [
  { label: 'Dashboard',      icon: <LayoutDashboard size={16} />, page: 'dashboard'  },
  { label: 'Live Feed',      icon: <Video           size={16} />, page: 'livefeed'   },
  { label: 'Alerts',         icon: <Bell            size={16} />, page: 'alerts'     },
  { label: 'Activity Log',   icon: <ClipboardList   size={16} />, page: 'activities' },
  { label: 'Analytics',      icon: <BarChart3       size={16} />, page: 'analytics'  },
  { label: 'Person Timeline',icon: <User            size={16} />, page: 'timeline'   },
  { label: 'Intelligence',   icon: <BrainCircuit    size={16} />, page: 'intelligence' },
]

interface Props { activePage: string; onNavigate: (page: string) => void }

export function Sidebar({ activePage, onNavigate }: Props) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar-collapsed' : 'sidebar-expanded'}`}>
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <Shield size={16} />
        </div>
        {!collapsed && (
          <div className="sidebar-logo-text">
            <div className="sidebar-logo-title">WAREHOUSE AI</div>
            <div className="sidebar-logo-sub">SURVEILLANCE</div>
          </div>
        )}
      </div>

      <nav className="sidebar-nav">
        {!collapsed && (
          <div className="sidebar-nav-section">Monitoring</div>
        )}
        {NAV.map((item) => (
          <button
            key={item.page}
            onClick={() => onNavigate(item.page)}
            className={`sidebar-nav-item ${activePage === item.page ? 'sidebar-nav-item-active' : ''}`}
            title={collapsed ? item.label : undefined}
          >
            <span className="sidebar-nav-icon">{item.icon}</span>
            {!collapsed && <span className="sidebar-nav-label">{item.label}</span>}
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <button
          className="sidebar-bottom-btn"
          title={collapsed ? 'Settings' : undefined}
        >
          <Settings size={16} className="flex-shrink-0" />
          {!collapsed && <span>Settings</span>}
        </button>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="sidebar-collapse-btn"
        >
          {collapsed ? <ChevronRight size={14} /> : <><ChevronLeft size={14} /><span>Collapse</span></>}
        </button>
      </div>
    </aside>
  )
}
