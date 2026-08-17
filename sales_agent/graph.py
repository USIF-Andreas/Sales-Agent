from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sales_agent.nodes.complex_agents import (
    product_agent_complex,
    profile_agent,
    sales_agent_complex,
    supervisor,
    supervisor_fanout,
)
from sales_agent.nodes.intent_gateway import intent_gateway
from sales_agent.nodes.lead_pipeline import (
    crm,
    deterministic_score,
    lead_extractor,
)
from sales_agent.nodes.simple_agents import (
    product_agent,
    sales_agent,
    simple_router,
    support_agent,
)
from sales_agent.state import AgentState


# ---------------------------------------------------------------------------
# Context isolation at the graph-edge level.
#
# The plan requires that no node receives the full conversation history: "the
# state fields passed between nodes are the actual scoping mechanism." These
# wrappers construct each complex-path node's input from ONLY the state fields
# it needs, so a node physically cannot read what it was not given. LangGraph's
# reducer merges each node's returned delta back into the shared state, so the
# scoping is lossless.
# ---------------------------------------------------------------------------

_PRODUCT_SCOPE = ("user_message", "path")
_PROFILE_SCOPE = ("user_message", "path")
_SALES_SCOPE = ("product_context", "profile_summary", "path")


def _scoped(state: AgentState, fields: tuple[str, ...]) -> AgentState:
    return {k: state[k] for k in fields if k in state}  # type: ignore[misc]


def scoped_product_agent_complex(state: AgentState) -> AgentState:
    # Receives only the request slice — never history, profile, or sales output.
    return product_agent_complex(_scoped(state, _PRODUCT_SCOPE))


def scoped_profile_agent(state: AgentState) -> AgentState:
    return profile_agent(_scoped(state, _PROFILE_SCOPE))


def scoped_sales_agent_complex(state: AgentState) -> AgentState:
    # Consumes ONLY product_context + profile_summary (not raw user message).
    return sales_agent_complex(_scoped(state, _SALES_SCOPE))


def _route_after_gateway(state: AgentState) -> str:
    # Gateway blocked the request -> return the canned response and stop.
    if state.get("direct_response"):
        return END
    if state.get("path") == "complex":
        return "supervisor"
    return simple_router(state)


def _fanout_branches(state: AgentState) -> list:
    from langgraph.types import Send

    branches = supervisor_fanout(state)
    sends = []
    if "product" in branches:
        sends.append(Send("product_agent_complex", state))
    if "profile" in branches:
        sends.append(Send("profile_agent", state))
    return sends


def build_graph() -> CompiledStateGraph:
    g = StateGraph(AgentState)

    # Entry + gateway
    g.add_node("intent_gateway", intent_gateway)

    # Simple path
    g.add_node("product_agent", product_agent)
    g.add_node("sales_agent", sales_agent)
    g.add_node("support_agent", support_agent)

    # Complex path (scoped wrappers enforce the context-isolation rule)
    g.add_node("supervisor", supervisor)
    g.add_node("product_agent_complex", scoped_product_agent_complex)
    g.add_node("profile_agent", scoped_profile_agent)
    g.add_node("sales_agent_complex", scoped_sales_agent_complex)
    g.add_node("lead_extractor", lead_extractor)
    g.add_node("deterministic_score", deterministic_score)
    g.add_node("crm", crm)

    g.set_entry_point("intent_gateway")
    g.add_conditional_edges(
        "intent_gateway",
        _route_after_gateway,
        {
            "product_agent": "product_agent",
            "sales_agent": "sales_agent",
            "support_agent": "support_agent",
            "supervisor": "supervisor",
            END: END,
        },
    )

    # Simple: single specialist handles the request, then stop.
    g.add_edge("product_agent", END)
    g.add_edge("sales_agent", END)
    g.add_edge("support_agent", END)

    # Complex: supervisor -> parallel fan-out -> join -> sales -> lead -> score -> crm
    g.add_conditional_edges("supervisor", _fanout_branches)
    g.add_edge("product_agent_complex", "sales_agent_complex")
    g.add_edge("profile_agent", "sales_agent_complex")
    g.add_edge("sales_agent_complex", "lead_extractor")
    g.add_edge("lead_extractor", "deterministic_score")
    g.add_edge("deterministic_score", "crm")
    g.add_edge("crm", END)

    return g.compile()


GRAPH = build_graph()


def run(message: str, session_id: str = "default") -> AgentState:
    import time

    initial: AgentState = {
        "user_message": message,
        "session_id": session_id,
        "trace": [],
    }
    start = time.perf_counter()
    result = GRAPH.invoke(initial)
    result["_wallclock_ms"] = (time.perf_counter() - start) * 1000.0
    return result


def _merge(cur: AgentState, delta: AgentState) -> AgentState:
    """Merge a node's returned delta into the running state.

    Mirrors how LangGraph applies node updates (preserving keys not in the delta)
    while honoring the trace channel's extend reducer instead of clobbering it.
    """
    merged: AgentState = {**cur, **delta}
    merged["trace"] = cur.get("trace", []) + delta.get("trace", [])
    return merged


def run_complex_sequential(message: str, session_id: str = "seq") -> AgentState:
    """Naive baseline: run the complex pipeline strictly sequentially (no fan-out).

    Used by the cost/latency eval to confirm the parallel fan-out actually buys
    latency reduction. The node set and scoped inputs are identical to the graph's,
    so token cost matches; only the product/profile fan-out runs serially instead
    of in parallel.
    """
    import time

    state: AgentState = {
        "user_message": message,
        "session_id": session_id,
        "trace": [],
    }
    start = time.perf_counter()
    state = _merge(state, intent_gateway(state))
    state = _merge(state, supervisor(state))
    state = _merge(state, scoped_product_agent_complex(state))
    state = _merge(state, scoped_profile_agent(state))
    state = _merge(state, scoped_sales_agent_complex(state))
    state = _merge(state, lead_extractor(state))
    state = _merge(state, deterministic_score(state))
    state = _merge(state, crm(state))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    state["_wallclock_ms"] = elapsed_ms
    return state


def run_always_supervisor(message: str, session_id: str = "always") -> AgentState:
    """Baseline: an 'always run Supervisor' design on ANY request.

    The gateway still classifies (its real cost is counted) but its simple/complex
    decision is ignored — the full complex pipeline always runs. Compared against
    the adaptive `run()` this makes the case for the adaptive gate with data.
    """
    import time

    state: AgentState = {
        "user_message": message,
        "session_id": session_id,
        "trace": [],
    }
    start = time.perf_counter()
    state = _merge(state, intent_gateway(state))
    state["path"] = "complex"  # override the adaptive decision
    state = _merge(state, supervisor(state))
    state = _merge(state, scoped_product_agent_complex(state))
    state = _merge(state, scoped_profile_agent(state))
    state = _merge(state, scoped_sales_agent_complex(state))
    state = _merge(state, lead_extractor(state))
    state = _merge(state, deterministic_score(state))
    state = _merge(state, crm(state))
    state["_wallclock_ms"] = (time.perf_counter() - start) * 1000.0
    return state