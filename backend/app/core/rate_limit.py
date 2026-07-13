import asyncio
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """Per-process, sliding-window request counter, keyed by an arbitrary
    hashable (an IP string or a user id).

    Deliberately not backed by a third-party library (slowapi et al.) or an
    external store (Redis) - this app targets a single-container
    docker-compose deployment, and a per-user limiter needs `current_user.id`
    which is only available after the auth dependency resolves, so a
    library keyed purely off the raw Request wouldn't cover that case
    cleanly anyway.

    Known limitation, stated plainly: this state is in-memory and
    per-process. It resets on every restart/redeploy, and does NOT
    coordinate across multiple uvicorn workers or container replicas - each
    process keeps its own counters, so the effective limit multiplies by
    worker/replica count under horizontal scaling. Fine for this app's
    current single-container target; would need a shared store (Redis) the
    moment it scales beyond that.
    """

    def __init__(self, *, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[object, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: object) -> bool:
        """Returns True and records a hit if `key` is under its limit,
        False (without recording) if it's already at the limit."""
        now = time.monotonic()
        async with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self._window_seconds:
                hits.popleft()
            if len(hits) >= self._max_requests:
                return False
            hits.append(now)
            return True


# Scoped to /agent-runs only - the one route that triggers an LLM call.
# Thresholds are placeholders, not derived from real traffic data - adjust
# based on actual usage/cost tolerance once this is live.
agent_run_ip_rate_limiter = InMemoryRateLimiter(max_requests=20, window_seconds=60.0)
agent_run_user_rate_limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60.0)

# Scoped to /auth/signup and /auth/login - both pre-auth, so IP is the only
# available key (no current_user yet). Tighter than the agent-run limiter on
# purpose: these are classic credential-stuffing/brute-force targets, not
# just cost-abuse targets.
auth_ip_rate_limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60.0)

# Scoped to /feedback - deliberately unauthenticated by design (session_uuid
# is client-generated, no login required), so this closes the spam/vote-
# stuffing abuse gap without changing that anonymous-by-design UX.
feedback_ip_rate_limiter = InMemoryRateLimiter(max_requests=30, window_seconds=60.0)
