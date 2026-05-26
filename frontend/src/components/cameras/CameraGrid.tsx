// components/cameras/CameraGrid.tsx
import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Maximize2, WifiOff, Film } from 'lucide-react'
import { api } from '../../api/client'
import { useCameraStore } from '../../store/cameraStore'
import type { Camera } from '../../types'
import './CameraGrid.css'

function CameraFeedCard({ camera }: { camera: Camera }) {
  const online = camera.status === 'online'

  return (
    <div className={`camera-card ${online ? 'camera-card-online' : 'camera-card-offline'}`}>
      <div className="camera-feed scanline">
        {online ? (
          <>
            <div className="camera-feed-sim">
              <Film size={28} className="camera-feed-sim-icon" />
            </div>
            <div className="camera-feed-corner camera-feed-corner-tl" />
            <div className="camera-feed-corner camera-feed-corner-tr" />
            <div className="camera-feed-corner camera-feed-corner-bl" />
            <div className="camera-feed-corner camera-feed-corner-br" />
          </>
        ) : (
          <div className="camera-feed-placeholder">
            <WifiOff size={24} className="camera-feed-icon" />
            <span className="camera-feed-signal-lost">SIGNAL LOST</span>
          </div>
        )}

        {online && (
          <div className="camera-feed-timestamp">
            {new Date().toLocaleTimeString('en-IN', { hour12: false })}
          </div>
        )}

        <button className="camera-feed-expand">
          <Maximize2 size={11} />
        </button>
      </div>

      <div className="camera-caption">
        <div className="camera-caption-left">
          {online
            ? <span className="live-dot" />
            : <span className="offline-dot" />
          }
          <span className="camera-caption-name">{camera.name}</span>
        </div>
        <div className="camera-caption-right">
          {online && (
            <span className="camera-caption-fps">{camera.fps}fps</span>
          )}
          <span className="camera-caption-id">{camera.id}</span>
        </div>
      </div>
    </div>
  )
}

export function CameraGrid() {
  const { data: cameras, isLoading } = useQuery({
    queryKey: ['cameras'],
    queryFn:  () => api.cameras.list(),
    refetchInterval: 30_000,
  })
  const { cameras: stored, setCameras } = useCameraStore()

  useEffect(() => {
    if (cameras) setCameras(cameras)
  }, [cameras, setCameras])

  const display = stored.length > 0 ? stored : (cameras ?? [])

  if (isLoading) return (
    <div className="camera-grid-skeleton">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="camera-grid-skeleton-item" />
      ))}
    </div>
  )

  return (
    <div className="camera-grid">
      {display.map(cam => (
        <div key={cam.id}>
          <CameraFeedCard camera={cam} />
        </div>
      ))}
    </div>
  )
}
