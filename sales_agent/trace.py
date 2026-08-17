from __future__ import annotations

import json
import logging
from typing import Any

from sales_agent.state import AgentState, NodeLog

logger = logging.getLogger("sales_agent.trace")


def log_node(
    state: AgentState,
    *,
    node: str,
    input_text: str = "",
    output_text: str = "",
    result: Any = None,
    note: str | None = None,
) -> NodeLog:
    """Build and emit a structured trace entry.

    Returns the NodeLog so callers can fold it into their returned state delta.
    The trace channel uses an ``operator.add`` reducer, so each node contributes
    only its own entries (and parallel branches merge cleanly).
    """
    entry: NodeLog = {
        "node": node,
        "path": state.get("path"),
        "input_chars": len(input_text),
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
    }
    if result is not None:
        entry["model"] = getattr(result, "model", None)
        entry["provider"] = getattr(result, "provider", None)
        entry["input_tokens"] = getattr(result, "input_tokens", 0)
        entry["output_tokens"] = getattr(result, "output_tokens", 0)
        entry["latency_ms"] = round(getattr(result, "latency_ms", 0.0), 2)
    if note is not None:
        entry["note"] = note
    logger.info(json.dumps({"event": "node", **entry}))
    return entry


def summarize_trace(state: AgentState) -> dict:
    """Aggregate token/latency totals from the trace for the cost/latency eval."""
    total_in = sum(e.get("input_tokens", 0) for e in state.get("trace", []))
    total_out = sum(e.get("output_tokens", 0) for e in state.get("trace", []))
    total_ms = sum(e.get("latency_ms", 0.0) for e in state.get("trace", []))
    return {
        "nodes": len(state.get("trace", [])),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_latency_ms": round(total_ms, 2),
        "wallclock_ms": round(state.get("_wallclock_ms", total_ms), 2),
    }
