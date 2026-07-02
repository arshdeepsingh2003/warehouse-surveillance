// App.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/layout/Sidebar'
import { Header }  from './components/layout/Header'
import { ErrorBoundary } from './components/common/ErrorBoundary'
import { AlertTriangle } from 'lucide-react'
import { DashboardPage }  from './pages/DashboardPage'
import { LiveFeedPage }   from './pages/LiveFeedPage'
import { AlertsPage }     from './pages/AlertsPage'
import { ActivitiesPage } from './pages/ActivitiesPage'
import { TimelinePage }   from './pages/TimelinePage'
import { useWebSocket }   from './hooks/useWebSocket'
import { useNavigationStore } from './store/navigationStore'
import { NotificationToastContainer } from './components/notifications/NotificationToast'

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
  timeline:   'Person Timeline',
}

function AppInner() {
  const page = useNavigationStore(s => s.page)
  const setPage = useNavigationStore(s => s.setPage)
  const { connected }   = useWebSocket()

  return (
    <div className="flex h-screen overflow-hidden bg-surface-900">
      <Sidebar activePage={page} onNavigate={setPage} />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header wsConnected={connected} pageTitle={PAGE_TITLES[page] ?? page} />

        <main className="flex-1 overflow-auto min-h-0">
          <ErrorBoundary fallback={
            <div className="flex flex-col items-center justify-center h-full gap-4 text-text-muted p-8">
              <AlertTriangle size={32} className="text-red-500" />
              <span className="text-sm font-mono text-red-400">Page Error</span>
              <span className="text-xs text-text-muted">Something went wrong rendering this page. Try navigating to another page and back.</span>
              <button onClick={() => window.location.reload()}
                className="text-xs px-3 py-1.5 rounded border border-red-800 text-red-400 hover:bg-red-950">
                Reload
              </button>
            </div>
          }>
            {page === 'dashboard'  && <DashboardPage />}
            {page === 'livefeed'   && <LiveFeedPage  />}
            {page === 'alerts'     && <AlertsPage    />}
            {page === 'activities' && <ActivitiesPage />}
            {page === 'timeline'   && <TimelinePage  />}
          </ErrorBoundary>
        </main>
      </div>

      <NotificationToastContainer />
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
