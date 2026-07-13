import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, createAgentRun, isLocalhostApiUrlMismatch } from './api'

function mockFetchOnce(body: unknown, ok = true) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    status: ok ? 201 : 400,
    json: () => Promise.resolve(body),
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('createAgentRun', () => {
  it('sets the X-LLM-API-Key header when an apiKey is provided, and always sends credentials', async () => {
    const fetchMock = mockFetchOnce({ id: 1 })

    await createAgentRun(
      { prompt: 'a trip', retrieval_top_k: 3, llm_provider: 'openai', llm_model: 'gpt-5.4-nano' },
      'user-supplied-key',
    )

    const [, requestInit] = fetchMock.mock.calls[0]
    expect(requestInit.headers['X-LLM-API-Key']).toBe('user-supplied-key')
    // Auth is a cookie now, not a header this client attaches - 'include'
    // is what makes the browser send it.
    expect(requestInit.credentials).toBe('include')
  })

  it('omits the X-LLM-API-Key header when no apiKey is provided', async () => {
    const fetchMock = mockFetchOnce({ id: 1 })

    await createAgentRun({ prompt: 'a trip', retrieval_top_k: 3 })

    const [, requestInit] = fetchMock.mock.calls[0]
    expect(requestInit.headers['X-LLM-API-Key']).toBeUndefined()
  })

  it('never shows a 5xx body verbatim, even if it happens to contain a detail field', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'Traceback: something internal broke at line 42' }),
      }),
    )

    await expect(createAgentRun({ prompt: 'a trip', retrieval_top_k: 3 })).rejects.toSatisfy(
      (error: unknown) => {
        expect(error).toBeInstanceOf(ApiError)
        expect((error as ApiError).message).toBe(
          'Something went wrong on our end - please try again.',
        )
        expect((error as ApiError).message).not.toContain('Traceback')
        return true
      },
    )
  })

  it('still shows a 4xx detail message verbatim - those are deliberate, safe, user-facing strings', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ detail: 'llm_provider and llm_model are required.' }),
      }),
    )

    await expect(createAgentRun({ prompt: 'a trip', retrieval_top_k: 3 })).rejects.toSatisfy(
      (error: unknown) => {
        expect((error as ApiError).message).toBe('llm_provider and llm_model are required.')
        return true
      },
    )
  })
})

describe('isLocalhostApiUrlMismatch', () => {
  it('flags a deployed page whose API base URL still points at localhost', () => {
    expect(isLocalhostApiUrlMismatch('app.example.com', 'http://localhost:8000')).toBe(true)
    expect(isLocalhostApiUrlMismatch('app.example.com', 'http://127.0.0.1:8000')).toBe(true)
  })

  it('does not flag the current real deployment target (page and API both localhost)', () => {
    expect(isLocalhostApiUrlMismatch('localhost', 'http://localhost:8000')).toBe(false)
    expect(isLocalhostApiUrlMismatch('127.0.0.1', 'http://localhost:8000')).toBe(false)
  })

  it('does not flag a deployed page whose API base URL was correctly overridden', () => {
    expect(isLocalhostApiUrlMismatch('app.example.com', 'https://api.example.com')).toBe(false)
  })
})
