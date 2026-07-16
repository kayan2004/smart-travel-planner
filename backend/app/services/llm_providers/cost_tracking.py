"""Request-scoped accumulation of estimated LLM cost.

A single trip-planner run makes several LLM calls (field extraction +
synthesis, at minimum). Each call's estimated dollar cost is computed down
in the provider layer (`usage_logging.estimate_cost_usd`) but was previously
only logged, never summed back up to the run. The server-key monthly budget
gate needs a per-run total, so this module accumulates it.

Mechanism: a `ContextVar` pointing at a *mutable* `CostAccumulator`.
`record_cost()` mutates the shared object rather than reassigning the var -
deliberately. Reassigning inside a child task/context (an `asyncio.gather`
or `create_task` somewhere in the pipeline) would not propagate back to the
reader, but mutating a shared object the parent already holds a reference to
always does. The request path is sequential today, so this is belt-and-
suspenders, but it costs nothing and won't silently break if a future
pipeline step gets parallelized.

Isolation across concurrent requests is the same property BYOK relies on:
each request runs in its own asyncio task with its own copied context, so
one request's `reset_cost_accumulator()` never touches another's total.
"""

import contextvars
from dataclasses import dataclass


@dataclass
class CostAccumulator:
    total_usd: float = 0.0

    def add(self, amount_usd: float) -> None:
        self.total_usd += amount_usd


_cost_accumulator: contextvars.ContextVar[CostAccumulator | None] = contextvars.ContextVar(
    "llm_cost_accumulator", default=None
)


def reset_cost_accumulator() -> CostAccumulator:
    """Installs a fresh accumulator for the current request context and
    returns it. The caller should hold the returned reference and read
    `.total_usd` after the run - reading the returned object directly is
    robust even if the ContextVar gets shadowed by nested contexts.
    """
    accumulator = CostAccumulator()
    _cost_accumulator.set(accumulator)
    return accumulator


def record_cost(amount_usd: float) -> None:
    """Adds one LLM call's estimated cost to the active accumulator, if any.

    A no-op when no accumulator is installed (e.g. an offline script's LLM
    call, or a call outside a tracked request) - never raises.
    """
    accumulator = _cost_accumulator.get()
    if accumulator is not None:
        accumulator.add(amount_usd)
