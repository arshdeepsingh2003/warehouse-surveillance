// pages/LiveFeedPage.tsx
import { CameraGrid } from '../components/cameras/CameraGrid'
import { ErrorBoundary } from '../components/common/ErrorBoundary'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { Wifi, WifiOff } from 'lucide-react'
import './LiveFeed.css'

export function LiveFeedPage() {
  const { data: cameras } = useQuery({
    queryKey: ['cameras'],
    queryFn:  () => api.cameras.list(),
    refetchInterval: 30_000,
  })

  const online  = cameras?.filter(c => c.status === 'online').length  ?? 0
  const offline = cameras?.filter(c => c.status === 'offline').length ?? 0

  return (
    <div className="livefeed">
      <div className="livefeed-section livefeed-status">
        <div className="livefeed-title-area">
          <span className="live-dot" />
          <span className="livefeed-title">Live Feed</span>
        </div>
        <div className="livefeed-count-online">
          <Wifi size={12} /> {online} online
        </div>
        {offline > 0 && (
          <div className="livefeed-count-offline">
            <WifiOff size={12} /> {offline} offline
          </div>
        )}
        <div className="livefeed-total">
          {cameras?.length ?? 0} cameras registered
        </div>
      </div>

      <div className="livefeed-grid-container">
        <ErrorBoundary>
          <CameraGrid />
        </ErrorBoundary>
      </div>

      <p className="livefeed-hint">
        Live video feeds will render here once RTSP streams are connected.
        Currently showing mock camera placeholders.
      </p>
    </div>
  )
}
