import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'

const sampleUser = {
  id: 1,
  email: 'traveler@test.com',
  full_name: 'Traveler',
  is_active: true,
  created_at: '2026-01-01T00:00:00Z',
}

function makeRun(freeRunsRemaining: number) {
  return {
    id: 5,
    user_id: 1,
    prompt: 'A trip please',
    response: 'Try Banff.',
    status: 'completed',
    created_at: '2026-07-16T10:00:00Z',
    tool_logs: [],
    recommendations: [],
    free_runs_remaining: freeRunsRemaining,
  }
}

type MockOptions = { freeRunsRemaining?: number; post402?: boolean }

function mockFetch(options: MockOptions = {}) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString()
    const method = (init?.method ?? 'GET').toUpperCase()
    if (url.includes('/llm-options')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve([{ provider: 'gemini', model: 'gemini-3.1-flash-lite' }]),
      })
    }
    if (url.includes('/auth/me')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(sampleUser) })
    }
    if (url.includes('/agent-runs') && method === 'POST') {
      if (options.post402) {
        return Promise.resolve({
          ok: false,
          status: 402,
          json: () =>
            Promise.resolve({
              detail: {
                reason: 'free_quota_exhausted',
                message: "You've used your free trip plan. Add your own API key below to keep planning.",
              },
            }),
        })
      }
      return Promise.resolve({
        ok: true,
        status: 201,
        json: () => Promise.resolve(makeRun(options.freeRunsRemaining ?? 0)),
      })
    }
    if (url.includes('/agent-runs')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) })
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

beforeEach(() => {
  window.sessionStorage.clear()
  window.history.pushState(null, '', '/app')
})

afterEach(() => {
  vi.unstubAllGlobals()
  window.sessionStorage.clear()
})

describe('BYOK reveal on free-tier exhaustion', () => {
  it('reveals the BYOK panel once the free run is used (free_runs_remaining=0)', async () => {
    mockFetch({ freeRunsRemaining: 0 })
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Run agent' }))

    expect(await screen.findByText(/That was your free trip plan/i)).toBeInTheDocument()
  })

  it('does not reveal the prompt while a free run still remains', async () => {
    mockFetch({ freeRunsRemaining: 1 })
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Run agent' }))

    expect(await screen.findByText('Try Banff.')).toBeInTheDocument()
    expect(screen.queryByText(/That was your free trip plan/i)).not.toBeInTheDocument()
  })

  it('shows the gate message and reveals BYOK on a 402', async () => {
    mockFetch({ post402: true })
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Run agent' }))

    expect(await screen.findByText(/used your free trip plan/i)).toBeInTheDocument()
  })
})
