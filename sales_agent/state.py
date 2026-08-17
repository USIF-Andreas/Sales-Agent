from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class NodeLog(TypedDict, total=False):
    node: str
    path: str | None
    model: str | None
    provider: str | None
    input_chars: int
    output_chars: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    note: str | None


class AgentState(TypedDict, total=False):
    user_message: str
    session_id: str
    path: Literal["simple", "complex"]
    domains: list[str]
    # simple-path outputs
    direct_response: str | None
    direct_agent: str | None
    # complex-path outputs
    product_context: dict | None
    profile_summary: dict | None
    sales_output: str | None
    lead_signals: dict | None
    lead_score: float | None
    crm_result: dict | None
    # transient (set by supervisor, not surfaced as a result)
    _branches: list[str]
    # wall-clock execution time of the whole graph, set by the runners
    _wallclock_ms: float
    # reducer channel: every node contributes only its own entries
    trace: Annotated[list[NodeLog], operator.add]
