from __future__ import annotations

import re

from sales_agent.llm import FAST, LLM, parse_json
from sales_agent.state import AgentState
from sales_agent.trace import log_node
from sales_agent.security import run_all_guardrails

LEAD_SYSTEM = (
    "Extract structured lead signals from the sales recommendation. Respond with "
    'JSON: {"intent_to_buy": bool, "urgency": bool, "budget_mentioned": bool}.'
)


def lead_extractor(state: AgentState) -> AgentState:
    sales_output = state.get("sales_output") or ""

    # Security gate: check guardrails on the sales output before extracting signals.
    guardrail = run_all_guardrails(sales_output if sales_output else "")
    if not guardrail.is_safe:
        log_node(state, node="lead_extractor",
                 input_text=sales_output, output_text="",
                 result=None, note=f"blocked: {guardrail.blocked_category}")
        signals = {"intent_to_buy": False, "urgency": False, "budget_mentioned": False}
        state["lead_signals"] = signals
        return {"lead_signals": signals, "trace": [{"node": "lead_extractor", "note": f"blocked: {guardrail.blocked_category}"}]}

    # --- Normal lead extraction flow ---
    llm = LLM()
    res = llm.complete(FAST, LEAD_SYSTEM, sales_output)
    signals = parse_json(res.text) or {
        "intent_to_buy": False, "urgency": False, "budget_mentioned": False,
    }
    entry = log_node(state, node="lead_extractor", input_text=sales_output, output_text=res.text,
                     result=res)
    state["lead_signals"] = signals
    return {"lead_signals": signals, "trace": [entry]}


def _detect_signals_from_text(text: str) -> dict:
    t = text.lower()
    return {
        "intent_to_buy": any(w in t for w in ["buy", "purchase", "order", "subscribe", "upgrade"]),
        "urgency": any(w in t for w in ["today", "now", "asap", "urgent", "immediately"]),
        "budget_mentioned": bool(re.search(r"\$\s?\d|\d{3,}\s?(usd|dollars|k\b)", t)),
    }


# ---------------------------------------------------------------------------
# Deterministic Score — rule-based, must be identical for identical inputs.
# ---------------------------------------------------------------------------

WEIGHTS = {"intent_to_buy": 50.0, "urgency": 30.0, "budget_mentioned": 20.0}


def deterministic_score(state: AgentState) -> AgentState:
    signals = state.get("lead_signals") or {}
    score = 0.0
    for key, weight in WEIGHTS.items():
        if signals.get(key):
            score += weight
    # Deterministic rounding to keep outputs stable.
    score = round(score, 2)
    entry = log_node(state, node="deterministic_score",
             input_text=str(signals), output_text=str(score),
             note="deterministic, no LLM")
    state["lead_score"] = score
    return {"lead_score": score, "trace": [entry]}


def crm(state: AgentState) -> AgentState:
    payload = {
        "session_id": state.get("session_id"),
        "path": state.get("path"),
        "lead_signals": state.get("lead_signals"),
        "lead_score": state.get("lead_score"),
        "sales_output": state.get("sales_output"),
        "recorded": True,
    }
    entry = log_node(state, node="crm", input_text=str(state.get("lead_signals")),
             output_text=str(payload), note="lead/ticket written")
    return {"crm_result": payload, "trace": [entry]}
