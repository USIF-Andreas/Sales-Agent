from __future__ import annotations

from sales_agent.llm import FAST, LLM
from sales_agent.rag import lookup
from sales_agent.state import AgentState
from sales_agent.trace import log_node

PRODUCT_SYSTEM = (
    "You are a Product specialist. Answer using only the provided catalog entry. "
    "Be concise and factual. If no product matches, say so."
)


def product_agent(state: AgentState) -> AgentState:
    message = state["user_message"]
    item = lookup(message)
    llm = LLM()
    if item is None:
        res = llm.complete(FAST, PRODUCT_SYSTEM, f"User asks: {message}\nNo catalog match.")
        answer = res.text
    else:
        res = llm.complete(
            FAST, PRODUCT_SYSTEM,
            f"User asks: {message}\nCatalog entry: {item['name']} — "
            f"${item['price']} — {item['summary']}",
        )
        answer = res.text
    entry = log_node(state, node="product_agent", input_text=message, output_text=answer, result=res)
    return {"direct_response": answer, "direct_agent": "product", "trace": [entry]}


SALES_SYSTEM = (
    "You are a Sales specialist. Help the user with purchasing, discounts, plans, "
    "and checkout. Be friendly and concise."
)


def sales_agent(state: AgentState) -> AgentState:
    message = state["user_message"]
    llm = LLM()
    res = llm.complete(FAST, SALES_SYSTEM, message)
    entry = log_node(state, node="sales_agent", input_text=message, output_text=res.text, result=res)
    return {"direct_response": res.text, "direct_agent": "sales", "trace": [entry]}


SUPPORT_SYSTEM = (
    "You are a Support specialist. Help the user with orders, returns, warranties, "
    "shipping, and issues. Be clear and empathetic."
)


def support_agent(state: AgentState) -> AgentState:
    message = state["user_message"]
    llm = LLM()
    res = llm.complete(FAST, SUPPORT_SYSTEM, message)
    entry = log_node(state, node="support_agent", input_text=message, output_text=res.text, result=res)
    return {"direct_response": res.text, "direct_agent": "support", "trace": [entry]}


def simple_router(state: AgentState) -> str:
    """Pick which simple-path specialist to call based on tagged domains."""
    domains = state.get("domains", [])
    if "support" in domains:
        return "support_agent"
    if "sales" in domains:
        return "sales_agent"
    return "product_agent"