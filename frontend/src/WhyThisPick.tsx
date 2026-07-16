import type { RecommendationFeatures } from './types'

interface WhyThisPickProps {
  features: RecommendationFeatures | null
}

// budget_delta is an ordinal step between budget *tiers* (low=0/medium=1/
// high=2), not a dollar amount - there is no price data in this pipeline, so
// this deliberately renders "tiers," never a fabricated currency figure.
function formatBudgetFit(budgetDelta: number | null): string {
  if (budgetDelta === null) return 'Budget comparison unavailable'
  if (budgetDelta === 0) return 'Matches your budget tier'
  const steps = Math.abs(budgetDelta)
  const noun = steps === 1 ? 'tier' : 'tiers'
  return budgetDelta < 0
    ? `${steps} ${noun} under your budget ceiling`
    : `${steps} ${noun} over your budget ceiling`
}

function formatSemanticMatchPercent(cosineSim: number): number {
  return Math.round(Math.min(1, Math.max(0, cosineSim)) * 100)
}

export function WhyThisPick({ features }: WhyThisPickProps) {
  return (
    <details className="why-pick">
      <summary className="why-pick-summary">Why this pick?</summary>
      {features ? (
        <div className="why-pick-grid">
          <div className="why-pick-row">
            <span className="gt-eyebrow">Semantic match</span>
            <div className="why-pick-bar-track">
              <div
                className="why-pick-bar-fill"
                style={{ width: `${formatSemanticMatchPercent(features.cosine_sim)}%` }}
              />
            </div>
            <span className="gt-mono-sm why-pick-value">
              {formatSemanticMatchPercent(features.cosine_sim)}%
            </span>
          </div>

          <div className="why-pick-row">
            <span className="gt-eyebrow">Budget fit</span>
            <span className="gt-mono-sm why-pick-value">
              {formatBudgetFit(features.budget_delta)}
            </span>
          </div>

          <div className="why-pick-row">
            <span className="gt-eyebrow">Region match</span>
            <span
              className={`gt-pill ${features.region_match ? 'gt-pill--positive' : 'gt-pill--negative'}`}
            >
              {features.region_match ? 'Yes' : 'No'}
            </span>
          </div>
        </div>
      ) : (
        <p className="empty-state why-pick-empty">Ranking details unavailable for this result.</p>
      )}
    </details>
  )
}
