export type AuthMode = 'login' | 'signup'

export interface UserRead {
  id: number
  email: string
  full_name: string | null
  is_active: boolean
  created_at: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface PlannerRequest {
  prompt: string
  retrieval_top_k: number
  llm_provider?: string
  llm_model?: string
}

export interface LlmOption {
  provider: string
  model: string
}

export interface ToolLogRead {
  id: number
  agent_run_id: number
  tool_name: string
  input_payload: string
  output_payload: string
  status: string
  created_at: string
}

export interface RecommendationFeatures {
  cosine_sim: number
  tag_match_count: number
  budget_delta: number | null
  region_match: boolean
}

export interface RecommendationRead {
  id: number
  destination_id: string
  destination_name: string
  country: string
  rank_position: number
  score: number
  features: RecommendationFeatures | null
  created_at: string
}

export type FeedbackVerdict = 1 | -1

export interface FeedbackRead {
  id: number
  recommendation_id: number
  session_uuid: string
  verdict: FeedbackVerdict
  channel: string
  created_at: string
}

export interface AgentRunRead {
  id: number
  user_id: number
  prompt: string
  response: string
  status: string
  created_at: string
  tool_logs: ToolLogRead[]
  recommendations: RecommendationRead[]
  // Free server-key runs left after this response. 0 means the next run
  // needs a BYOK key - the UI reveals the BYOK panel when it hits 0.
  free_runs_remaining?: number | null
}

export interface AgentRunSummary {
  id: number
  prompt: string
  status: string
  created_at: string
}

export interface SessionState {
  // No token field - auth is an httpOnly cookie the frontend never reads
  // or stores (see lib/api.ts). This just tracks who's logged in for UI
  // purposes, restored on mount via GET /auth/me (which succeeds/fails
  // based on the cookie the browser sends automatically).
  user: UserRead
}
