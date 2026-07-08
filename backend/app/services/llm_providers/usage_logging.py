"""Per-call token usage + estimated cost logging for the LLM provider layer.

Real Python `logging` (structured, via `extra={}`), not LangSmith/OpenTelemetry
- no new external account/service needed (see backend/README.md's
"Provider-Agnostic LLM Layer" section for the tradeoff). `main.py` configures
`logging.basicConfig()` so these records are actually emitted, not silently
dropped by Python's default "no handler configured" root logger behavior.
"""

import logging

logger = logging.getLogger("app.llm_usage")

# USD per 1,000,000 tokens, as (input_price, output_price). Verified live
# against each provider's own current pricing page on 2026-07-06
# (ai.google.dev/gemini-api/docs/pricing, platform.claude.com/docs/en/about-claude/pricing)
# - not from memory, to avoid silently baking in stale numbers. A model
# string not in this table logs real token counts but skips the cost
# estimate rather than guessing at an unverified price.
MODEL_PRICING_USD_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "gemma-4-26b-a4b-it": (0.0, 0.0),  # free tier, confirmed live this session
    "gemma-4-31b-it": (0.0, 0.0),  # free tier, confirmed live this session
    "gemini-3.1-flash-lite": (0.25, 1.50),
    # gemini-3.1-pro-preview: 2.00/12.00 for prompts <=200k tokens (the
    # common case here); jumps to 4.00/18.00 beyond that - not modeled,
    # since this app's prompts are nowhere near 200k tokens.
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-haiku-latest": (1.00, 5.00),  # same pricing tier as claude-haiku-4-5
}


def estimate_cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float | None:
    """Returns None (not 0.0) for an unrecognized model - "unknown" and "free"
    must stay distinguishable, since silently reporting $0.00 for a model
    this table has never verified pricing for would be worse than reporting
    nothing.
    """
    pricing = MODEL_PRICING_USD_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def log_completion_usage(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    extra_tokens: dict[str, int] | None = None,
) -> None:
    """Logs one structured record per LLM call - token counts, an estimated
    dollar cost (when the model's pricing is known), and latency. This is
    the single place both providers report usage from, so every call site
    (extraction, synthesis, cluster naming) gets this for free.
    """
    estimated_cost_usd = estimate_cost_usd(
        model, input_tokens=input_tokens, output_tokens=output_tokens
    )
    cost_display = f"${estimated_cost_usd:.6f}" if estimated_cost_usd is not None else "unknown"
    # The message itself carries the key numbers (not just extra={}), so
    # they're visible in plain-text console output too, not only to a
    # structured/JSON log consumer that reads the extra fields.
    message = (
        f"llm_completion provider={provider} model={model} "
        f"input_tokens={input_tokens} output_tokens={output_tokens} "
        f"cost={cost_display} latency={latency_seconds:.3f}s"
    )
    logger.info(
        message,
        extra={
            "llm_provider": provider,
            "llm_model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "latency_seconds": round(latency_seconds, 3),
            **(extra_tokens or {}),
        },
    )
