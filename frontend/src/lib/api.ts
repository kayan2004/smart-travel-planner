import type {
  AgentRunRead,
  AgentRunSummary,
  FeedbackRead,
  FeedbackVerdict,
  LlmOption,
  PlannerRequest,
  TokenResponse,
  UserRead,
} from '../types'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ??
  'http://localhost:8000'

// VITE_API_BASE_URL is baked in at build time (Vite inlines it into the
// bundle, it can't be changed at container runtime) - the current
// deployment target is a single-host docker-compose setup where
// http://localhost:8000 is correct, but if this build ever gets deployed
// somewhere else without overriding the build arg, every API call would
// silently fail against the wrong host with nothing but a generic network
// error to debug from. This makes that misconfiguration loud and
// immediately diagnosable instead: if the page itself isn't being served
// from localhost but API_BASE_URL still points there, something's wrong
// with how this image was built. Exported as a pure function (rather than
// just a module-load-time side effect) so the logic is directly testable
// without fighting ESM module-caching semantics.
export function isLocalhostApiUrlMismatch(hostname: string, apiBaseUrl: string): boolean {
  const pageIsLocalhost = hostname === 'localhost' || hostname === '127.0.0.1'
  const apiIsLocalhost = /^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(apiBaseUrl)
  return !pageIsLocalhost && apiIsLocalhost
}

if (
  typeof window !== 'undefined' &&
  isLocalhostApiUrlMismatch(window.location.hostname, API_BASE_URL)
) {
  console.warn(
    `[config] This app is running on ${window.location.hostname}, but its API base URL ` +
      `(${API_BASE_URL}) still points at localhost. VITE_API_BASE_URL was likely not set ` +
      'at build time for this deployment - API calls will fail. See frontend/Dockerfile.',
  )
}

type RequestOptions = {
  method?: 'GET' | 'POST'
  body?: unknown
  headers?: Record<string, string>
}

class ApiError extends Error {
  status: number
  // Machine-readable code from a structured error body (e.g. the 402
  // free-tier gates return {reason, message}). Lets callers branch on the
  // cause without matching the human-facing message string.
  reason?: string

  constructor(message: string, status: number, reason?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.reason = reason
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? 'GET',
    // Auth is an httpOnly cookie set by POST /auth/login, not a token this
    // client ever sees - 'include' is required for the browser to send it
    // cross-origin (frontend/backend are different ports) and to accept
    // the Set-Cookie response on login.
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })

  if (!response.ok) {
    // 5xx bodies are never shown verbatim, regardless of shape - a server
    // error response is exactly where an accidental future change (a new
    // exception handler, a debug flag flipped) is most likely to leak
    // internal detail. Every 4xx `detail` in this API is a deliberately
    // written, safe, user-facing string (see backend/app/api/routes/ and
    // app/services/*.py's HTTPException call sites) - those are fine to
    // surface directly, that's what they're for.
    if (response.status >= 500) {
      throw new ApiError('Something went wrong on our end - please try again.', response.status)
    }

    const payload = await response.json().catch(() => null)
    const rawDetail = payload?.detail
    let detail: string
    let reason: string | undefined
    if (typeof rawDetail === 'string') {
      detail = rawDetail
    } else if (Array.isArray(rawDetail)) {
      detail = rawDetail.map((item: { msg?: string }) => item.msg).join(', ')
    } else if (rawDetail && typeof rawDetail === 'object') {
      // Structured error body, e.g. the free-tier gates' {reason, message}.
      detail =
        typeof rawDetail.message === 'string'
          ? rawDetail.message
          : `Request failed with status ${response.status}`
      reason = typeof rawDetail.reason === 'string' ? rawDetail.reason : undefined
    } else {
      detail = `Request failed with status ${response.status}`
    }
    throw new ApiError(detail, response.status, reason)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export async function signup(input: {
  email: string
  password: string
  full_name: string
}): Promise<UserRead> {
  return request<UserRead>('/auth/signup', {
    method: 'POST',
    body: input,
  })
}

export async function login(input: {
  email: string
  password: string
}): Promise<TokenResponse> {
  return request<TokenResponse>('/auth/login', {
    method: 'POST',
    body: input,
  })
}

export async function logout(): Promise<void> {
  return request<void>('/auth/logout', { method: 'POST' })
}

export async function fetchCurrentUser(): Promise<UserRead> {
  return request<UserRead>('/auth/me')
}

export async function createAgentRun(
  payload: PlannerRequest,
  apiKey?: string,
): Promise<AgentRunRead> {
  return request<AgentRunRead>('/agent-runs', {
    method: 'POST',
    body: payload,
    headers: apiKey ? { 'X-LLM-API-Key': apiKey } : undefined,
  })
}

export async function fetchLlmOptions(): Promise<LlmOption[]> {
  return request<LlmOption[]>('/llm-options')
}

export async function listAgentRuns(): Promise<AgentRunSummary[]> {
  return request<AgentRunSummary[]>('/agent-runs')
}

export async function fetchAgentRun(agentRunId: number): Promise<AgentRunRead> {
  return request<AgentRunRead>(`/agent-runs/${agentRunId}`)
}

export async function submitFeedback(payload: {
  recommendation_id: number
  session_uuid: string
  verdict: FeedbackVerdict
}): Promise<FeedbackRead> {
  return request<FeedbackRead>('/feedback', {
    method: 'POST',
    body: payload,
  })
}

export { ApiError, API_BASE_URL }
