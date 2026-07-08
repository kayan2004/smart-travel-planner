"""One shared logging setup for both the live app (main.py) and any offline
script that makes LLM calls (currently scripts/cluster_destinations.py's
`name` phase). Without calling this, `logger.info(...)` calls (see
app/services/tool_logs.py, app/services/llm_providers/usage_logging.py) are
silently dropped - Python's root logger has no handler configured by
default, and INFO-level records don't reach the "handler of last resort"
(which only fires at WARNING+).
"""

import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
