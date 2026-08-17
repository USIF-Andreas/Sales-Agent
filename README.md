# Sales-Agent

Adaptive multi-agent conversational assistant (product Q&A + sales + support)
that routes each request through a cheap **Intent Gateway**: simple, single-domain
requests are answered by one specialist; complex, cross-domain requests are routed
through a **Supervisor** that fans out to the right specialists and aggregates their
output before handing off to CRM.

Orchestration complexity scales with request complexity, not applied uniformly.

## Architecture

```
USER -> FastAPI/Chat -> Intent Gateway
                          |-- simple -> Product / Sales / Support agent (direct)
                          |-- complex -> Supervisor -> Product | Profile -> Sales -> Lead Extractor -> Deterministic Score -> CRM
```

See `adaptive-supervisor-plan.md` for the full design.

## Design notes

- **Adaptive gate**: the Intent Gateway classifies each message as `simple` or
  `complex` (rule-based first, a fast model call only for ambiguous requests) and
  tags which domains it touches. Simple requests pay only one specialist; complex
  requests pay the full fan-out pipeline.
- **Context isolation at the graph-edge level**: each complex-path node is fed a
  *scoped* state slice (only the fields it needs) — the Product/Profile branches
  get just the request, and the Sales Agent gets only `product_context` +
  `profile_summary`, never the raw message or history. Scoping is done by the
  graph wrappers in `sales_agent/graph.py`, not by prompt discipline.
- **Security gate**: a guardrail check (`sales_agent/security`) runs once at the
  Intent Gateway (the single entry for both paths). Unsafe/off-topic/jailbreak
  requests short-circuit with a canned response; no specialist runs.
- **Observability**: every node emits a structured JSON trace entry (node, path,
  input/output size, tokens, latency) via `sales_agent/trace.py`.

## Quickstart

```bash
pip install -e .
# optional: set ANTHROPIC_API_KEY (or GROQ_API_KEY) to use real models;
# otherwise a deterministic mock provider is used (zero external dependencies).
python -m sales_agent.api.app   # or: uvicorn sales_agent.api.app:app --reload
```

Try it:

```bash
curl -X POST http://localhost:8000/chat -H 'content-type: application/json' \
  -d '{"user_message":"What is the price of the Aurora headset?"}'
curl -X POST http://localhost:8000/chat -H 'content-type: application/json' \
  -d '{"user_message":"I\'m a remote worker with a $300 budget. Which product do you recommend for me?"}'
```

### Frontend

The API also serves a single-page web UI at the root (`http://localhost:8000/`):

- **Chat** — full conversational UI. Each assistant reply shows which path it took
  (`simple`/`complex`/`blocked`), the specialist agent, the tagged domains, the lead
  score (complex requests) and an expandable per-node trace (tokens + latency).
- **Cost / Latency** tab — the Phase 5 comparison table (adaptive vs complex vs
  sequential vs always-Supervisor) rendered live from the API, with savings
  highlights.

Frontend files live in `sales_agent/frontend/` (`index.html`, `styles.css`,
`app.js`) and are served by FastAPI's `StaticFiles` mount.

## Run the eval harness

```bash
pip install -e ".[dev]"
pytest
```

The eval suite covers all five evaluation points from the plan:
gateway precision/recall, simple-path regression, complex-path correctness,
cost/latency comparison, and deterministic lead-score consistency.

> **Harness note**: the plan lists DeepEval. This implementation uses
> pytest-style deterministic graders instead (DeepEval stays an optional dev
> dependency). The graders are fully deterministic — gateway classification is
> a labeled precision/recall check, and lead scoring is a same-input/same-output
> regression — so the suite is hermetic and runs in CI without any API key.
> An LLM-judge (e.g. DeepEval `G-Eval`) can be added on top once an API key is
> available; the deterministic graders remain the offline gate.

## Cost/latency comparison (Phase 5)

```bash
python -m scripts.compare_paths
```

Prints a table comparing the adaptive simple path vs the complex path, the
parallel fan-out vs a naive sequential baseline, and the adaptive gate vs an
"always run Supervisor" design. Latency is **wall-clock** execution time (the
metric the fan-out actually improves; summed per-node model latency is identical
either way). Representative result:

| design                                  | path    | nodes | in_tok | wall-clock (ms) |
|-----------------------------------------|---------|------:|-------:|----------------:|
| adaptive simple path                    | simple  |     2 |     95 |             109 |
| adaptive complex path                   | complex |     8 |    530 |             713 |
| complex path (sequential baseline)      | complex |     8 |    530 |             802 |
| always-Supervisor on simple request     | complex |     8 |    485 |             802 |

The adaptive gate on a simple request saves ~390 input tokens and 6 node
invocations vs an always-Supervisor design; the parallel fan-out costs **zero**
extra tokens versus the sequential baseline while cutting wall-clock latency by
~90 ms (product + profile run concurrently).

## Model tiers

| Task                          | Model tier  |
|-------------------------------|-------------|
| Intent Gateway classification | fast/rule   |
| Simple product/RAG query      | fast        |
| Lead extraction               | fast        |
| Sales Agent reasoning         | medium      |
| Supervisor branch decision    | medium+     |