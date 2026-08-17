from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sales_agent.graph import run, run_always_supervisor, run_complex_sequential
from sales_agent.trace import summarize_trace


def test_simple_path_cheaper_than_complex():
    simple = run("What is the price of the Aurora headset?")
    complex_state = run("I'm a remote worker with a $300 budget. Recommend a product for me.")

    s_simple = summarize_trace(simple)
    s_complex = summarize_trace(complex_state)

    # Simple path must invoke strictly fewer nodes than the complex pipeline.
    assert s_simple["nodes"] < s_complex["nodes"]
    # Simple path must not touch the Supervisor / CRM lead machinery.
    assert s_simple["total_input_tokens"] < s_complex["total_input_tokens"]


def test_adaptive_gate_saves_cost_vs_always_supervisor():
    message = "What is the price of the Aurora headset?"  # a SIMPLE request
    adaptive = run(message)
    always = run_always_supervisor(message)

    a = summarize_trace(adaptive)
    s = summarize_trace(always)

    # The adaptive gateway must run far fewer nodes and consume far fewer tokens
    # than an 'always run Supervisor' design on the same simple request.
    assert a["nodes"] < s["nodes"]
    assert a["total_input_tokens"] < s["total_input_tokens"]


def test_parallel_fanout_is_not_slower_than_sequential():
    message = "I'm a remote worker with a $300 budget. Recommend a product for me."

    parallel = run(message)
    sequential = run_complex_sequential(message)

    p = summarize_trace(parallel)
    s = summarize_trace(sequential)

    # Token cost is identical (both branches still run); only wall-clock differs.
    assert p["total_input_tokens"] == s["total_input_tokens"]
    assert p["total_output_tokens"] == s["total_output_tokens"]
    # The parallel fan-out must actually buy latency reduction: product + profile
    # run concurrently, so wall-clock is strictly lower than running them in series.
    assert p["wallclock_ms"] < s["wallclock_ms"]
