from __future__ import annotations

from sales_agent.llm import FAST, MEDIUM, STRONG, LLM, parse_json
from sales_agent.rag import lookup
from sales_agent.state import AgentState
from sales_agent.trace import log_node

# ---------------------------------------------------------------------------
# Supervisor — decides which specialist branches are actually needed.
#
# The supervisor is the complex-path choke point: it receives the raw request
# (the gateway sends it through), runs the security guardrails once, and fans
# out only to safe downstream branches. Downstream nodes therefore never need
# the raw message, which preserves the plan's context-isolation rule.
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM = (
    "You are a Supervisor for a sales assistant. Given the user's request and the "
    "domains it touches, decide which specialist branches to run. Available: "
    "['product', 'profile']. Respond with JSON: "
    '{"branches": ["product", "profile"]}. Only include a branch if it is needed.'
)


def supervisor(state: AgentState) -> AgentState:
    message = state["user_message"]

    llm = LLM()
    domains = state.get("domains", [])
    res = llm.complete(
        STRONG, SUPERVISOR_SYSTEM,
        f"Request: {message}\nDomains: {domains}",
    )
    data = parse_json(res.text)
    branches = data.get("branches", ["product", "profile"])
    # Always include product on the complex path; profile only if profiling needed.
    if "product" not in branches:
        branches = ["product", *branches]
    entry = log_node(state, node="supervisor", input_text=message, output_text=res.text,
                     result=res, note=f"branches={branches}")
    return {"_branches": branches, "trace": [entry]}  # type: ignore[typeddict-item]


def supervisor_fanout(state: AgentState) -> list[str]:
    """Conditional edge: which branches to run in parallel."""
    return list(state.get("_branches", ["product", "profile"]))


# ---------------------------------------------------------------------------
# Product Agent (complex path) — scoped to the request, sets product_context.
# ---------------------------------------------------------------------------

PRODUCT_SYSTEM = (
    "You are the Product specialist in a multi-agent pipeline. Use only the "
    "catalog entry below. Return a concise factual summary of the relevant "
    "product facts (price, fit, key features)."
)


def product_agent_complex(state: AgentState) -> AgentState:
    # Context isolation: only the request slice drives the lookup, not history.
    message = state["user_message"]
    item = lookup(message)
    llm = LLM()
    if item is None:
        res = llm.complete(FAST, PRODUCT_SYSTEM, f"User asks: {message}\nNo catalog match.")
        ctx = {"found": False, "summary": res.text}
    else:
        res = llm.complete(
            FAST, PRODUCT_SYSTEM,
            f"User asks: {message}\nCatalog: {item['name']} ${item['price']} — {item['summary']}",
        )
        ctx = {"found": True, "name": item["name"], "price": item["price"], "summary": res.text}
    entry = log_node(state, node="product_agent_complex", input_text=message,
                     output_text=str(ctx), result=res)
    return {"product_context": ctx, "trace": [entry]}


# ---------------------------------------------------------------------------
# Profile Agent — only runs on complex path; extracts user context.
# ---------------------------------------------------------------------------

PROFILE_SYSTEM = (
    "You are the Profile Agent. From the user's request extract relevant context "
    "for a sales recommendation: background, goals, and constraints. Respond with "
    'JSON: {"background": str, "goals": [str], "constraints": [str]}.'
)


def profile_agent(state: AgentState) -> AgentState:
    message = state["user_message"]
    llm = LLM()
    res = llm.complete(MEDIUM, PROFILE_SYSTEM, message)
    data = parse_json(res.text) or {
        "background": "", "goals": [], "constraints": [],
    }
    entry = log_node(state, node="profile_agent", input_text=message, output_text=res.text, result=res)
    return {"profile_summary": data, "trace": [entry]}


# ---------------------------------------------------------------------------
# Sales Agent (complex path) — consumes ONLY product_context + profile_summary.
# ---------------------------------------------------------------------------

SALES_SYSTEM = (
    "You are the Sales Agent. Produce a recommendation/pitch grounded in the "
    "product facts and the user's profile context provided. Do not invent facts."
)


def sales_agent_complex(state: AgentState) -> AgentState:
    # Context isolation: explicitly constructed from scoped state fields only.
    # This node cannot see the raw user message or any conversation history.
    product_ctx = state.get("product_context") or {}
    profile = state.get("profile_summary") or {}
    prompt = (
        f"Product context: {product_ctx}\n"
        f"User profile: {profile}\n"
        "Write a recommendation that is grounded in both."
    )
    llm = LLM()
    res = llm.complete(MEDIUM, SALES_SYSTEM, prompt)
    entry = log_node(state, node="sales_agent_complex", input_text=prompt, output_text=res.text,
                     result=res, note="isolated: product_context + profile_summary only")
    return {"sales_output": res.text, "trace": [entry]}