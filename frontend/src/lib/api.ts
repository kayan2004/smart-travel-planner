import type {
  AgentRunRead,
  FeedbackRead,
  FeedbackVerdict,
  PlannerRequest,
  TokenResponse,
  UserRead,
} from '../types'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ??
  'http://localhost:8000'

type RequestOptions = {
  method?: 'GET' | 'POST'
  token?: string
  body?: unknown
}

class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...(options.token ? { Authorization: `Bearer ${options.token}` } : {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail =
      typeof payload?.detail === 'string'
        ? payload.detail
        : Array.isArray(payload?.detail)
          ? payload.detail.map((item: { msg?: string }) => item.msg).join(', ')
          : `Request failed with status ${response.status}`
    throw new ApiError(detail, response.status)
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

export async function fetchCurrentUser(token: string): Promise<UserRead> {
  return request<UserRead>('/auth/me', { token })
}

export async function createAgentRun(
  token: string,
  payload: PlannerRequest,
): Promise<AgentRunRead> {
  return request<AgentRunRead>('/agent-runs', {
    method: 'POST',
    token,
    body: payload,
  })
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
