from __future__ import annotations

"""Phase 5 deliverable: cost/latency comparison table.

Produces the comparison the plan asks for:

  * simple vs complex path  (cost/latency of the adaptive design)
  * parallel fan-out vs naive sequential complex pipeline
  * this adaptive design vs a naive 'always run Supervisor' baseline

Run:  python -m scripts.compare_paths
"""

from sales_agent.graph import run, run_always_supervisor, run_complex_sequential
from sales_agent.trace import summarize_trace

SIMPLE_MSG = "What is the price of the Aurora headset?"
COMPLEX_MSG = "I'm a remote worker with a $300 budget. Which product do you recommend for me?"


def _row(label: str, state) -> dict:
    t = summarize_trace(state)
    return {
        "label": label,
        "nodes": t["nodes"],
        "in_tok": t["total_input_tokens"],
        "out_tok": t["total_output_tokens"],
        # Wall-clock, not summed model latency: this is what the fan-out actually saves.
        "lat_ms": t["wallclock_ms"],
        "path": state.get("path"),
    }


def build_rows() -> list[dict]:
    """Run each design and return comparable cost/latency rows (used by the API)."""
    return [
        _row("adaptive simple path", run(SIMPLE_MSG)),
        _row("adaptive complex path", run(COMPLEX_MSG)),
        _row("complex path (sequential baseline)", run_complex_sequential(COMPLEX_MSG)),
        _row("always-Supervisor on simple request", run_always_supervisor(SIMPLE_MSG)),
    ]


def main() -> None:
    rows = build_rows()

    header = f"{'design':<40} {'path':<8} {'nodes':>5} {'in_tok':>7} {'out_tok':>7} {'lat_ms':>8}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        print(
            f"{r['label']:<40} {str(r['path']):<8} {r['nodes']:>5} "
            f"{r['in_tok']:>7} {r['out_tok']:>7} {r['lat_ms']:>8.1f}"
        )
    print("=" * len(header))

    adaptive_simple = rows[0]
    always = rows[3]
    adaptive_complex = rows[1]
    sequential = rows[2]
    print()
    print(f"Adaptive gate saves {adaptive_simple['in_tok'] - always['in_tok']:+.0f} input tokens "
          f"and {adaptive_simple['nodes'] - always['nodes']:+.0f} node invocations on a simple request.")
    print(f"Parallel fan-out costs {adaptive_complex['in_tok'] - sequential['in_tok']:+.0f} tokens "
          f"vs sequential (should be ~0) and "
          f"{adaptive_complex['lat_ms'] - sequential['lat_ms']:+.1f} ms vs sequential.")


if __name__ == "__main__":
    main()