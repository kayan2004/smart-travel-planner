import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import './App.css'
import groundtripLogo from './assets/groundtrip-logo.svg'
import {
  ApiError,
  createAgentRun,
  fetchCurrentUser,
  fetchLlmOptions,
  login,
  logout,
  signup,
  submitFeedback,
} from './lib/api'
import { JsonPayload } from './JsonPayload'
import type { AgentRunRead, AuthMode, FeedbackVerdict, LlmOption, SessionState } from './types'
import { WhyThisPick } from './WhyThisPick'

type View = 'login' | 'signup' | 'app'

const APP_ROUTE = '/app'
const LOGIN_ROUTE = '/login'
const SIGNUP_ROUTE = '/signup'
const FEEDBACK_SESSION_STORAGE_KEY = 'smart-travel-feedback-session-uuid'
// Deliberately sessionStorage, not localStorage - a BYOK key should not
// outlive the browser tab. Unrelated to auth, which no longer touches
// client-side storage at all (an httpOnly cookie now, see lib/api.ts).
const BYOK_SESSION_STORAGE_KEY = 'smart-travel-byok'

type ByokSelection = {
  provider: string
  model: string
  apiKey: string
}

function loadByokSelection(): ByokSelection {
  const raw = window.sessionStorage.getItem(BYOK_SESSION_STORAGE_KEY)
  if (!raw) {
    return { provider: '', model: '', apiKey: '' }
  }
  try {
    const parsed = JSON.parse(raw) as Partial<ByokSelection>
    return {
      provider: parsed.provider ?? '',
      model: parsed.model ?? '',
      apiKey: parsed.apiKey ?? '',
    }
  } catch {
    return { provider: '', model: '', apiKey: '' }
  }
}

function getOrCreateFeedbackSessionUuid(): string {
  const existing = window.localStorage.getItem(FEEDBACK_SESSION_STORAGE_KEY)
  if (existing) {
    return existing
  }
  const created = crypto.randomUUID()
  window.localStorage.setItem(FEEDBACK_SESSION_STORAGE_KEY, created)
  return created
}

function getViewFromPath(pathname: string): View {
  if (pathname === SIGNUP_ROUTE) {
    return 'signup'
  }
  if (pathname === APP_ROUTE) {
    return 'app'
  }
  return 'login'
}

function navigateTo(view: View, replace = false) {
  const target =
    view === 'signup' ? SIGNUP_ROUTE : view === 'app' ? APP_ROUTE : LOGIN_ROUTE
  const nextUrl = `${target}${window.location.search}${window.location.hash}`

  if (replace) {
    window.history.replaceState(null, '', nextUrl)
    return
  }

  window.history.pushState(null, '', nextUrl)
}

function statusPillTone(status: string | undefined): string {
  if (status === 'completed') return 'gt-pill--positive'
  if (status === 'partial') return 'gt-pill--brass'
  if (status === 'failed') return 'gt-pill--negative'
  return ''
}

function renderInlineBoldText(text: string) {
  const parts = text.split(/(\*\*.*?\*\*)/g)

  return parts.map((part, index) => {
    const isBold = part.startsWith('**') && part.endsWith('**') && part.length > 4
    if (!isBold) {
      return <span key={`${part}-${index}`}>{part}</span>
    }

    return (
      <strong key={`${part}-${index}`} className="markdown-strong">
        {part.slice(2, -2)}
      </strong>
    )
  })
}

function App() {
  const [view, setView] = useState<View>(() => getViewFromPath(window.location.pathname))
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [prompt, setPrompt] = useState(
    'I have two weeks off in July and around $1,500. I want somewhere warm, not too touristy, and I like hiking. Where should I go, when should I book, and what should I expect?',
  )
  const [retrievalTopK, setRetrievalTopK] = useState(3)
  const [session, setSession] = useState<SessionState | null>(null)
  // Gates the initial render: restoring a session now means an async
  // GET /auth/me call (the cookie is invisible to JS, there's nothing to
  // read synchronously), so without this the login screen would flash
  // briefly even for an already-logged-in user on every page load.
  const [sessionChecked, setSessionChecked] = useState(false)
  const [result, setResult] = useState<AgentRunRead | null>(null)
  const [authError, setAuthError] = useState('')
  const [plannerError, setPlannerError] = useState('')
  const [authPending, setAuthPending] = useState(false)
  const [plannerPending, setPlannerPending] = useState(false)
  const [feedbackSessionUuid] = useState(getOrCreateFeedbackSessionUuid)
  const [feedbackByRecommendation, setFeedbackByRecommendation] = useState<
    Record<number, FeedbackVerdict>
  >({})
  const [feedbackError, setFeedbackError] = useState('')
  const [llmOptions, setLlmOptions] = useState<LlmOption[]>([])
  const [byokSelection, setByokSelection] = useState<ByokSelection>(loadByokSelection)

  const authMode: AuthMode = view === 'signup' ? 'signup' : 'login'

  useEffect(() => {
    const handlePopState = () => {
      setView(getViewFromPath(window.location.pathname))
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    // One-time cleanup: this key held a raw, still-valid JWT under the
    // pre-cookie auth scheme. The new code never reads or writes it, but a
    // browser that logged in before this migration landed still has it
    // sitting in localStorage - a real, usable credential an XSS could
    // read directly, doing nothing useful otherwise. Purge unconditionally
    // on every load until enough time has passed that no browser could
    // plausibly still have it.
    window.localStorage.removeItem('smart-travel-session')
  }, [])

  useEffect(() => {
    fetchLlmOptions()
      .then(setLlmOptions)
      .catch(() => setLlmOptions([]))
  }, [])

  useEffect(() => {
    if (byokSelection.provider || byokSelection.model || byokSelection.apiKey) {
      window.sessionStorage.setItem(BYOK_SESSION_STORAGE_KEY, JSON.stringify(byokSelection))
    } else {
      window.sessionStorage.removeItem(BYOK_SESSION_STORAGE_KEY)
    }
  }, [byokSelection])

  useEffect(() => {
    let cancelled = false

    fetchCurrentUser()
      .then((user) => {
        if (cancelled) return
        setSession({ user })
        if (getViewFromPath(window.location.pathname) !== 'app') {
          navigateTo('app', true)
          setView('app')
        }
      })
      .catch(() => {
        if (cancelled) return
        if (view === 'app') {
          navigateTo('login', true)
          setView('login')
        }
      })
      .finally(() => {
        if (!cancelled) setSessionChecked(true)
      })

    return () => {
      cancelled = true
    }
  }, [])

  function setSessionAndNavigate(nextSession: SessionState | null) {
    setSession(nextSession)

    if (nextSession) {
      navigateTo('app')
      setView('app')
      return
    }

    navigateTo('login')
    setView('login')
  }

  function handleAuthViewChange(nextMode: AuthMode) {
    setAuthError('')
    const nextView = nextMode === 'signup' ? 'signup' : 'login'
    navigateTo(nextView)
    setView(nextView)
  }

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setAuthPending(true)
    setAuthError('')

    try {
      if (authMode === 'signup') {
        await signup({
          email,
          password,
          full_name: fullName.trim(),
        })
      }

      await login({ email, password })
      const user = await fetchCurrentUser()
      setSessionAndNavigate({ user })
    } catch (error) {
      setAuthError(
        error instanceof ApiError ? error.message : 'Authentication failed.',
      )
    } finally {
      setAuthPending(false)
    }
  }

  async function handlePlanSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!session) {
      setPlannerError('Please log in first.')
      navigateTo('login')
      setView('login')
      return
    }

    setPlannerPending(true)
    setPlannerError('')

    try {
      const useByok = Boolean(
        byokSelection.apiKey && byokSelection.provider && byokSelection.model,
      )
      const agentRun = await createAgentRun(
        {
          prompt,
          retrieval_top_k: retrievalTopK,
          ...(useByok
            ? { llm_provider: byokSelection.provider, llm_model: byokSelection.model }
            : {}),
        },
        useByok ? byokSelection.apiKey : undefined,
      )
      setResult(agentRun)
    } catch (error) {
      setPlannerError(
        error instanceof ApiError ? error.message : 'Trip planning failed.',
      )
    } finally {
      setPlannerPending(false)
    }
  }

  async function handleFeedback(recommendationId: number, verdict: FeedbackVerdict) {
    setFeedbackError('')

    try {
      await submitFeedback({
        recommendation_id: recommendationId,
        session_uuid: feedbackSessionUuid,
        verdict,
      })
      setFeedbackByRecommendation((previous) => ({
        ...previous,
        [recommendationId]: verdict,
      }))
    } catch (error) {
      setFeedbackError(
        error instanceof ApiError ? error.message : 'Feedback could not be submitted.',
      )
    }
  }

  async function handleLogout() {
    try {
      await logout()
    } catch {
      // Best-effort - even if the network call fails, drop the client-side
      // session state so the UI doesn't strand the user on the app view.
    }
    setSessionAndNavigate(null)
    setResult(null)
  }

  function handleByokOptionChange(value: string) {
    const [provider, model] = value.split('::')
    setByokSelection((previous) => ({ ...previous, provider: provider ?? '', model: model ?? '' }))
  }

  function handleByokKeyChange(apiKey: string) {
    setByokSelection((previous) => ({ ...previous, apiKey }))
  }

  function handleRemoveByokKey() {
    setByokSelection({ provider: '', model: '', apiKey: '' })
  }

  if (!sessionChecked) {
    return (
      <main className="gt-auth-grid">
        <section className="gt-panel auth-hero">
          <img src={groundtripLogo} alt="GroundTrip" className="gt-logo" />
        </section>
      </main>
    )
  }

  if (view !== 'app' || !session) {
    return (
      <main className="gt-auth-grid">
        <section className="gt-panel auth-hero">
          <img src={groundtripLogo} alt="GroundTrip" className="gt-logo" />
          <h1 className="auth-hero-heading">Sign in before you ask the agent where to go.</h1>
          <div className="pipeline-stages">
            <span className="gt-pill gt-pill--brass">extract</span>
            <span className="gt-pill">recommend</span>
            <span className="gt-pill">RAG</span>
            <span className="gt-pill">weather</span>
            <span className="gt-pill gt-pill--positive">synthesize</span>
          </div>
        </section>

        <section className="gt-panel auth-page-panel">
          <div className="gt-panel-header gt-auth-header">
            <div>
              <p className="gt-eyebrow">Authentication</p>
              <h2>{authMode === 'login' ? 'Welcome back' : 'Create your account'}</h2>
            </div>
            <div className="gt-segmented" role="tablist" aria-label="Auth mode">
              <button
                type="button"
                role="tab"
                aria-selected={authMode === 'login'}
                className={authMode === 'login' ? 'gt-segmented-btn active' : 'gt-segmented-btn'}
                onClick={() => handleAuthViewChange('login')}
              >
                Login
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={authMode === 'signup'}
                className={authMode === 'signup' ? 'gt-segmented-btn active' : 'gt-segmented-btn'}
                onClick={() => handleAuthViewChange('signup')}
              >
                Sign up
              </button>
            </div>
          </div>

          <form className="form-grid" onSubmit={handleAuthSubmit}>
            {authMode === 'signup' ? (
              <label className="gt-field">
                <span>Full name</span>
                <input
                  className="gt-input"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  placeholder="Kayan"
                  autoComplete="name"
                />
              </label>
            ) : null}
            <label className="gt-field">
              <span>Email</span>
              <input
                className="gt-input"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </label>
            <label className="gt-field">
              <span>Password</span>
              <input
                className="gt-input"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="At least 8 characters"
                autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
                required
              />
            </label>
            {authError ? (
              <p className="error-text" role="alert">
                {authError}
              </p>
            ) : null}
            <button type="submit" className="gt-btn gt-btn--primary" disabled={authPending}>
              {authPending
                ? 'Working…'
                : authMode === 'login'
                  ? 'Login and continue'
                  : 'Create account and continue'}
            </button>
          </form>
        </section>
      </main>
    )
  }

  return (
    <main className="shell">
      <section className="gt-panel hero-panel">
        <div className="gt-planner-header">
          <div>
            <img src={groundtripLogo} alt="GroundTrip" className="gt-logo" />
            <h1 className="planner-heading">Prompt-first trip planning with your backend agent in the loop.</h1>
          </div>
          <div className="gt-panel gt-panel--raised gt-user-card">
            <div className="session-chip">
              <strong>{session.user.full_name || 'Traveler'}</strong>
              <span>{session.user.email}</span>
            </div>
            <button type="button" className="gt-btn gt-btn--ghost" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </div>

        <div className="gt-panel planner-inner">
          <div className="gt-panel-header">
            <div>
              <p className="gt-eyebrow">Planner</p>
              <h2>Ask for a trip recommendation</h2>
            </div>
            <span className="gt-pill gt-pill--positive">ready</span>
          </div>

          <form className="form-grid" onSubmit={handlePlanSubmit}>
            <label className="gt-field">
              <span>Prompt</span>
              <textarea
                className="gt-textarea"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={7}
                // Matches the backend's AgentRunCreate.prompt cap
                // (max_length=4000, app/schemas/agent_runs.py) - client-side
                // feedback instead of only finding out via a failed request.
                maxLength={4000}
                required
              />
            </label>
            <label className="gt-field gt-compact-field">
              <span>RAG top K</span>
              <input
                className="gt-input"
                type="number"
                min={1}
                max={8}
                value={retrievalTopK}
                onChange={(event) => setRetrievalTopK(Number(event.target.value))}
              />
            </label>

            <details className="gt-panel gt-panel--raised byok-panel">
              <summary>Use your own API key</summary>
              <p className="byok-copy">
                Your key stays in this browser tab only (sessionStorage) and is sent only with
                your own requests - it is never stored server-side or logged. Leave this empty to
                use the app's default model.
              </p>
              <label className="gt-field">
                <span>Provider / model</span>
                <select
                  className="gt-input"
                  value={
                    byokSelection.provider && byokSelection.model
                      ? `${byokSelection.provider}::${byokSelection.model}`
                      : ''
                  }
                  onChange={(event) => handleByokOptionChange(event.target.value)}
                >
                  <option value="">Select a provider and model…</option>
                  {llmOptions.map((option) => (
                    <option
                      key={`${option.provider}::${option.model}`}
                      value={`${option.provider}::${option.model}`}
                    >
                      {option.provider} / {option.model}
                    </option>
                  ))}
                </select>
              </label>
              <label className="gt-field">
                <span>API key</span>
                <input
                  className="gt-input"
                  type="password"
                  value={byokSelection.apiKey}
                  onChange={(event) => handleByokKeyChange(event.target.value)}
                  placeholder="sk-…"
                  autoComplete="off"
                />
              </label>
              {byokSelection.apiKey || byokSelection.provider || byokSelection.model ? (
                <button type="button" className="gt-btn gt-btn--ghost" onClick={handleRemoveByokKey}>
                  Remove key
                </button>
              ) : null}
            </details>

            {plannerError ? (
              <p className="error-text" role="alert">
                {plannerError}
              </p>
            ) : null}
            <button type="submit" className="gt-btn gt-btn--primary" disabled={plannerPending}>
              {plannerPending ? 'Planning trip…' : 'Run agent'}
            </button>
          </form>
        </div>
      </section>

      {result && (
        <section className="gt-results-grid">
          <article className="gt-panel">
            <div className="gt-panel-header">
              <div>
                <p className="gt-eyebrow">Final answer</p>
                <h2>Saved recommendation</h2>
              </div>
              <span className={`gt-pill ${statusPillTone(result.status)}`}>{result.status}</span>
            </div>

            <div className="result-meta gt-mono">
              <span>run #{result.id}</span>
              <span>{new Date(result.created_at).toLocaleString()}</span>
            </div>
            <p className="gt-panel gt-panel--paper prompt-preview">{result.prompt}</p>
            <div className="response-card">
              {result.response.split('\n').map((line, index) => (
                <p key={`${line}-${index}`}>{renderInlineBoldText(line)}</p>
              ))}
            </div>
          </article>

          <article className="gt-panel">
            <div className="gt-panel-header">
              <div>
                <p className="gt-eyebrow">Recommendations</p>
                <h2>Rate the ranked slate</h2>
              </div>
              <span className="gt-pill">{result.recommendations.length} destinations</span>
            </div>

            {feedbackError ? (
              <p className="error-text" role="alert">
                {feedbackError}
              </p>
            ) : null}

            {result.recommendations.length ? (
              <div className="logs-list">
                {result.recommendations.map((recommendation) => {
                  const activeVerdict = feedbackByRecommendation[recommendation.id]
                  return (
                    <article key={recommendation.id} className="gt-panel gt-panel--raised log-card">
                      <div className="log-header">
                        <strong>
                          #{recommendation.rank_position} {recommendation.destination_name}, {recommendation.country}
                        </strong>
                        <span className="gt-mono" style={{ color: 'var(--brass)' }}>
                          {recommendation.score.toFixed(4)}
                        </span>
                      </div>
                      <WhyThisPick features={recommendation.features} />
                      <div className="feedback-actions">
                        <button
                          type="button"
                          className={
                            activeVerdict === 1
                              ? 'gt-stamp gt-stamp--positive gt-stamp--active'
                              : 'gt-stamp gt-stamp--positive'
                          }
                          aria-pressed={activeVerdict === 1}
                          onClick={() => handleFeedback(recommendation.id, 1)}
                        >
                          Good match
                        </button>
                        <button
                          type="button"
                          className={
                            activeVerdict === -1
                              ? 'gt-stamp gt-stamp--negative gt-stamp--active'
                              : 'gt-stamp gt-stamp--negative'
                          }
                          aria-pressed={activeVerdict === -1}
                          onClick={() => handleFeedback(recommendation.id, -1)}
                        >
                          Not a fit
                        </button>
                      </div>
                    </article>
                  )
                })}
              </div>
            ) : (
              <p className="empty-state">
                This run didn't return any ranked destinations.
              </p>
            )}
          </article>

          <article className="gt-panel">
            <div className="gt-panel-header">
              <div>
                <p className="gt-eyebrow">Tool trail</p>
                <h2>What the agent used</h2>
              </div>
              <span className="gt-pill">{result.tool_logs.length} logs</span>
            </div>

            {result.tool_logs.length ? (
              <div className="logs-list">
                {result.tool_logs.map((log) => (
                  <article key={log.id} className="gt-panel gt-panel--raised log-card">
                    <div className="log-header">
                      <strong>{log.tool_name}</strong>
                      <span className={`gt-pill ${log.status === 'success' ? 'gt-pill--positive' : 'gt-pill--negative'}`}>
                        {log.status}
                      </span>
                    </div>
                    <p className="log-time gt-mono-sm" style={{ color: 'var(--text-tertiary)' }}>
                      {new Date(log.created_at).toLocaleString()}
                    </p>
                    <details>
                      <summary>Input payload</summary>
                      <JsonPayload value={log.input_payload} />
                    </details>
                    <details>
                      <summary>Output payload</summary>
                      <JsonPayload value={log.output_payload} />
                    </details>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-state">No tool logs were recorded for this run.</p>
            )}
          </article>
        </section>
      )}
    </main>
  )
}

export default App
