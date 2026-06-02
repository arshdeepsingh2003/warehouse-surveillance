import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.props.onError?.(error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback

      return (
        <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-red-900/30 bg-surface-900 p-6 text-center"
          style={{ aspectRatio: '16/9' }}>
          <AlertTriangle size={22} className="text-red-500" />
          <span className="text-xs font-mono text-red-400">Component Error</span>
          <span className="text-[10px] text-text-muted max-w-[200px]">
            {this.state.error?.message || 'An unexpected error occurred'}
          </span>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-1 text-[10px] px-2 py-1 rounded border border-red-800 text-red-400 hover:bg-red-950">
            Retry
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
