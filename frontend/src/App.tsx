// App.tsx
import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/layout/Sidebar'
import { Header }  from './components/layout/Header'
import { DashboardPage }  from './pages/DashboardPage'
import { LiveFeedPage }   from './pages/LiveFeedPage'
import { AlertsPage }     from './pages/AlertsPage'
import { ActivitiesPage } from './pages/ActivitiesPage'
import { AnalyticsPage }  from './pages/AnalyticsPage'
import { TimelinePage }   from './pages/TimelinePage'
import { useWebSocket }   from './hooks/useWebSocket'

const qc = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry:     1,
    },
  },
})

const PAGE_TITLES: Record<string, string> = {
  dashboard:  'Dashboard Overview',
  livefeed:   'Live Camera Feed',
  alerts:     'Alert Management',
  activities: 'Activity Log',
  analytics:  'Analytics & Reports',
  timeline:   'Person Timeline',
}

function AppInner() {
  const [page, setPage] = useState('dashboard')
  const { connected }   = useWebSocket()

  return (
    <div className="flex h-screen overflow-hidden bg-surface-900">
      <Sidebar activePage={page} onNavigate={setPage} />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header wsConnected={connected} pageTitle={PAGE_TITLES[page] ?? page} />

        <main className="flex-1 overflow-auto min-h-0">
          {page === 'dashboard'  && <DashboardPage />}
          {page === 'livefeed'   && <LiveFeedPage  />}
          {page === 'alerts'     && <AlertsPage    />}
          {page === 'activities' && <ActivitiesPage />}
          {page === 'analytics'  && <AnalyticsPage />}
          {page === 'timeline'   && <TimelinePage  />}
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <AppInner />
    </QueryClientProvider>
  )
}
