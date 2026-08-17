from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sales_agent.graph import run
from sales_agent.rag import lookup


SIMPLE_CASES = [
    ("What is the price of the Aurora headset?", "product", "Aurora"),
    ("How much does the Cascade chair cost?", "product", "Cascade"),
    ("I want a refund for my broken Solstice lamp.", "support", "refund"),
    ("Where is my order? It hasn't shipped.", "support", "order"),
    ("Any discount on the Vortex keyboard?", "sales", "discount"),
]


def test_simple_path_routes_to_correct_specialist():
    for message, agent, _ in SIMPLE_CASES:
        state = run(message)
        assert state["path"] == "simple"
        assert state["direct_agent"] == agent, (message, state["direct_agent"])
        assert state.get("direct_response")


def test_simple_product_mentions_catalog_fact():
    message = "What is the price of the Aurora headset?"
    state = run(message)
    item = lookup(message)
    assert item is not None
    # The direct response should reference the actual product name or price.
    resp = state["direct_response"].lower()
    assert item["name"].lower() in resp or str(item["price"]) in resp


def test_simple_path_does_not_run_supervisor():
    state = run("Tell me about the Nimbus desk specs.")
    nodes = {e["node"] for e in state.get("trace", [])}
    assert "supervisor" not in nodes
    assert "sales_agent_complex" not in nodes
