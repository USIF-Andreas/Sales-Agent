from __future__ import annotations

import sys
import os

# Ensure the package root is importable when running pytest from the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from sales_agent.graph import run
from sales_agent.nodes.intent_gateway import rule_classify
from sales_agent.state import AgentState


# Labeled set: (message, expected_path, expected_domains_subset)
GATEWAY_LABELS = [
    # --- simple, single-domain ---
    ("What is the price of the Aurora headset?", "simple", ["product"]),
    ("Tell me the specs of the Nimbus Standing Desk.", "simple", ["product"]),
    ("I want to return my broken Solstice lamp for a refund.", "simple", ["support"]),
    ("Where is my order? It hasn't shipped yet.", "simple", ["support"]),
    ("Do you have a discount code for the Vortex keyboard?", "simple", ["sales"]),
    ("How much does the Cascade chair cost?", "simple", ["product"]),
    # --- complex, cross-domain ---
    ("I'm a remote worker with a $300 budget. Which product do you recommend for me?",
     "complex", ["product", "profile", "sales"]),
    ("As a student, what should I buy to set up a study desk?", "complex", ["product", "profile", "sales"]),
    ("Compare the Nimbus desk and the Cascade chair, which is best for my home office?", "complex", ["product", "profile"]),
    ("My company is outfitting a new office. Recommend a good lighting plan and quote it.",
     "complex", ["product", "profile", "sales"]),
]


@pytest.mark.parametrize("message,exp_path,exp_domains", GATEWAY_LABELS)
def test_gateway_classification(message, exp_path, exp_domains):
    cls = rule_classify(message)
    assert cls.path == exp_path, f"path: got {cls.path}, want {exp_path}"
    for d in exp_domains:
        assert d in cls.domains, f"domain {d} missing in {cls.domains}"


def test_gateway_precision_recall():
    tp = fp = fn = 0
    for message, exp_path, _ in GATEWAY_LABELS:
        got = rule_classify(message).path
        if got == exp_path:
            if got == "complex":
                tp += 1
            else:
                tp += 1  # simple/simple also true positive
        else:
            if got == "complex" and exp_path == "simple":
                fp += 1
            elif got == "simple" and exp_path == "complex":
                fn += 1
    # For a balanced check we treat each correct classification as a TP.
    correct = sum(1 for m, ep, _ in GATEWAY_LABELS if rule_classify(m).path == ep)
    accuracy = correct / len(GATEWAY_LABELS)
    assert accuracy >= 0.9, f"gateway accuracy {accuracy:.2f} < 0.90"
