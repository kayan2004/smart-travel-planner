import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { WhyThisPick } from './WhyThisPick'
import type { RecommendationFeatures } from './types'

const sampleFeatures: RecommendationFeatures = {
  cosine_sim: 0.842,
  tag_match_count: 2,
  budget_delta: -1,
  region_match: true,
}

describe('WhyThisPick', () => {
  it('is collapsed by default', () => {
    render(<WhyThisPick features={sampleFeatures} />)

    const details = screen.getByText('Why this pick?').closest('details')
    expect(details).not.toBeNull()
    expect(details).not.toHaveAttribute('open')
  })

  it('reveals the ranking breakdown when expanded', async () => {
    const user = userEvent.setup()
    render(<WhyThisPick features={sampleFeatures} />)

    await user.click(screen.getByText('Why this pick?'))

    expect(screen.getByText('84%')).toBeInTheDocument()
    expect(screen.getByText('1 tier under your budget ceiling')).toBeInTheDocument()
    expect(screen.getByText('Yes')).toBeInTheDocument()
  })

  it('renders a "no match" chip and message when region/budget signals say so', async () => {
    const user = userEvent.setup()
    render(
      <WhyThisPick
        features={{ cosine_sim: 0.5, tag_match_count: 0, budget_delta: null, region_match: false }}
      />,
    )

    await user.click(screen.getByText('Why this pick?'))

    expect(screen.getByText('Budget comparison unavailable')).toBeInTheDocument()
    expect(screen.getByText('No')).toBeInTheDocument()
  })

  it('falls back to an unavailable message when features is null, without breaking layout', async () => {
    const user = userEvent.setup()
    render(<WhyThisPick features={null} />)

    await user.click(screen.getByText('Why this pick?'))

    expect(screen.getByText('Ranking details unavailable for this result.')).toBeInTheDocument()
  })
})
