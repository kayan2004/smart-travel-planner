import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

type ErrorBoundaryProps = {
  children: ReactNode
}

type ErrorBoundaryState = {
  hasError: boolean
}

// Error boundaries must be class components - React has no hook equivalent
// of getDerivedStateFromError/componentDidCatch as of React 19. Without
// this, any uncaught render error white-screens the whole app with no
// recovery path.
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false }

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Logged to the browser console only - not shown to the user, and not
    // sent anywhere. The raw error/component stack can carry internal
    // details (matches the same sanitize-before-surfacing approach used
    // for backend tool failures).
    console.error('Uncaught render error:', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="gt-auth-grid">
          <section className="gt-panel auth-page-panel">
            <p className="gt-eyebrow">Something went wrong</p>
            <h2>This page hit an unexpected error.</h2>
            <p className="byok-copy">
              Reloading usually fixes it. If it keeps happening, please let us know.
            </p>
            <button
              type="button"
              className="gt-btn gt-btn--primary"
              onClick={() => window.location.reload()}
            >
              Reload
            </button>
          </section>
        </main>
      )
    }

    return this.props.children
  }
}
