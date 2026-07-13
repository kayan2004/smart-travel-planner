import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ErrorBoundary } from './ErrorBoundary'

function Bomb(): never {
  throw new Error('boom')
}

describe('ErrorBoundary', () => {
  beforeEach(() => {
    // React logs the caught error to console.error too (in addition to our
    // own componentDidCatch call) - silence both so the test output stays
    // readable; this is expected noise, not a real failure.
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders children normally when nothing throws', () => {
    render(
      <ErrorBoundary>
        <p>all fine</p>
      </ErrorBoundary>,
    )

    expect(screen.getByText('all fine')).toBeInTheDocument()
  })

  it('shows a fallback UI instead of white-screening when a child throws', () => {
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>,
    )

    expect(screen.getByText('This page hit an unexpected error.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reload' })).toBeInTheDocument()
    // The raw error message must not be shown to the user.
    expect(screen.queryByText('boom')).not.toBeInTheDocument()
  })
})
