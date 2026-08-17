from __future__ import annotations

from sales_agent.classify import Classification, rule_classify  # re-export for tests
from sales_agent.llm import GATEWAY, LLM, parse_json
from sales_agent.security import BLOCKED_RESPONSE, run_all_guardrails
from sales_agent.state import AgentState
from sales_agent.trace import log_node

GATEWAY_SYSTEM = (
    "You are an Intent Gateway for a sales assistant. Classify the user message "
    "as 'simple' (single-domain, answerable by one specialist) or 'complex' "
    "(cross-domain, needs Supervisor fan-out). Tag which domains it touches from "
    "[product, sales, support, profile]. Respond with JSON: "
    '{"path": "simple"|"complex", "domains": [...]}'
)


def intent_gateway(state: AgentState) -> AgentState:
    message = state["user_message"]

    # Security gate at the single entry point for BOTH paths. If the request is
    # unsafe, short-circuit with a safe canned response (no specialist runs).
    guardrail = run_all_guardrails(message)
    if not guardrail.is_safe:
        entry = log_node(state, node="intent_gateway", input_text=message, output_text="",
                         result=None, note=f"blocked: {guardrail.blocked_category}")
        return {
            "path": "simple",
            "domains": ["support"],
            "direct_response": BLOCKED_RESPONSE,
            "direct_agent": "gateway_blocked",
            "trace": [entry],
        }

    cls = rule_classify(message)

    if cls.confidence == "low" or cls.path is None:
        # Ambiguous -> delegate to a fast model call (mock returns rule-based too).
        llm = LLM()
        res = llm.complete(GATEWAY, GATEWAY_SYSTEM, message)
        data = parse_json(res.text)
        path = data.get("path") or "simple"
        domains = data.get("domains") or cls.domains or ["product"]
        entry = log_node(state, node="intent_gateway", input_text=message, output_text=res.text,
                         result=res, note=f"model fallback -> {path}")
    else:
        path, domains = cls.path, cls.domains
        entry = log_node(state, node="intent_gateway", input_text=message,
                         output_text=f"path={path} domains={domains}", note=f"rule ({cls.confidence})")

    return {"path": path, "domains": domains, "trace": [entry]}  # type: ignore[typeddict-item]


__all__ = ["intent_gateway", "rule_classify", "Classification"]