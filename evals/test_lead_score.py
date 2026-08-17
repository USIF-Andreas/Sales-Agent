from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sales_agent.nodes.lead_pipeline import deterministic_score


def _state_with(signals):
    return {"lead_signals": signals}


def test_deterministic_score_vanilla():
    st = _state_with({"intent_to_buy": False, "urgency": False, "budget_mentioned": False})
    deterministic_score(st)
    assert st["lead_score"] == 0.0


def test_deterministic_score_full():
    signals = {"intent_to_buy": True, "urgency": True, "budget_mentioned": True}
    st = _state_with(signals)
    deterministic_score(st)
    assert st["lead_score"] == 100.0


def test_deterministic_score_consistency():
    # Same inputs must always produce the same output (auditable, no drift).
    signals = {"intent_to_buy": True, "urgency": False, "budget_mentioned": True}
    scores = []
    for _ in range(5):
        st = _state_with(dict(signals))
        deterministic_score(st)
        scores.append(st["lead_score"])
    assert scores == [70.0, 70.0, 70.0, 70.0, 70.0]


def test_deterministic_score_in_pipeline():
    from sales_agent.graph import run

    # Intent + urgency + budget all present -> full 100.
    state = run("My company wants to buy 10 desks today. Quote me ASAP, budget is $5000.")
    assert state.get("lead_score") == 100.0
    assert state.get("crm_result", {}).get("recorded") is True

    # Budget only (no explicit buy/urgency) -> 20, still deterministic.
    state2 = run("I am a remote worker with a $300 budget. Which product do you recommend for me?")
    assert state2.get("lead_score") == 20.0
