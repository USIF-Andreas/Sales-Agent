# Adaptive Supervisor Agent System — Implementation Plan

## Goal

Build a multi-agent conversational assistant (product Q&A + sales + support) that only pays the cost/latency of a full multi-agent workflow on requests that actually need it. Simple, single-domain requests are answered directly by one specialist. Complex, cross-domain requests (product info + user profiling + sales strategy + lead qualification together) are routed through a Supervisor that fans out to the right specialists and aggregates their output before handing off to CRM.

The core design principle: **orchestration complexity should scale with request complexity, not be applied uniformly.** A cheap gateway decides which path a request takes before any expensive model is invoked.

## Architecture

```
                                  USER
                                    │
                                    ▼
                    ┌──────────────────────────┐
                    │ FASTAPI / Chat Service   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────┐
                      │ Intent Gateway     │
                      │ Cheap + Structured │
                      └─────────┬──────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
              SIMPLE PATH              COMPLEX PATH
                    │                        │
                    ▼                        ▼
             Product Agent             Supervisor
             Sales Agent                    │
             Support Agent          ┌────────┴─────────┐
                    │               │                  │
                    │               ▼                  ▼
                    │        Product Agent       Profile Agent
                    │               │                  │
                    │               └────────┬─────────┘
                    │                        ▼
                    │                   Sales Agent
                    │                        │
                    │                 Lead Extractor
                    │                        │
                    │              Deterministic Score
                    │                        │
                    └────────────────────────▼
                                          CRM
                                           │
                                           ▼
                                          END
```

## Component Responsibilities

- **FastAPI / Chat Service**: entry point. Receives the user message, manages session/conversation state, calls the graph, returns the response.
- **Intent Gateway**: the single most important node in this design. Classifies each incoming message as `simple` or `complex`, and tags which domain(s) it touches (product / sales / support / profile). Must be cheap — rule-based/heuristic for obvious cases (e.g. direct price or catalog lookups), a small/fast model call only for ambiguous ones. This node is what prevents the system from becoming "always run everything" — if it misclassifies, the cost savings and the quality of complex-request handling both collapse, so it needs its own precision/recall evaluation, not just end-to-end accuracy.
- **Simple Path — Product / Sales / Support Agents**: single-specialist handling, same shape as a conventional router-to-specialist system. Each of these can be called directly with no Supervisor overhead.
- **Complex Path — Supervisor**: receives requests tagged complex, decides which of the specialist branches are actually needed (not necessarily always both), and fans out.
- **Product Agent (complex path)**: same underlying capability as the simple-path Product Agent (product/catalog RAG), but scoped to receive only the relevant slice of context instead of full history.
- **Profile Agent**: extracts and summarizes the user's relevant context — stated background, goals, constraints — that Sales Agent needs to give a well-targeted answer. This does not exist on the simple path; it only runs when the request is complex enough to warrant profiling.
- **Sales Agent (complex path)**: consumes both Product Agent output and Profile Agent output (not raw conversation history) to produce a recommendation/pitch that's actually grounded in the user's situation.
- **Lead Extractor**: pulls structured lead signals (intent to buy, urgency, budget mentions) out of the Sales Agent's output.
- **Deterministic Score**: rule-based/deterministic lead scoring on top of the extracted signals — kept deterministic (not LLM-based) so it's auditable and doesn't drift between runs.
- **CRM**: writes the lead/ticket, triggers any downstream notifications.

## Tech Stack

- **Orchestration**: LangGraph (StateGraph; conditional edge for simple/complex branch, parallel edges for the Product/Profile fan-out)
- **API layer**: FastAPI
- **LLM**: Anthropic API, with model tiering (see below)
- **Eval harness**: DeepEval (pytest-style, custom metrics for gateway classification accuracy, task completion, and cost/latency per path)
- **Observability**: structured JSON trace log per node (node name, path taken, input/output size, tokens, latency)

## Model Tiering

| Task                          | Model Tier |
|-------------------------------|------------|
| Intent Gateway classification | Fast/cheap (or rule-based) |
| Simple product/RAG query      | Fast       |
| Lead extraction               | Fast       |
| Sales Agent reasoning          | Medium     |
| Supervisor branch decision     | Medium/Strong |

## State Schema (sketch)

```python
class AgentState(TypedDict):
    user_message: str
    path: Literal["simple", "complex"]
    domains: list[str]                # tagged by Intent Gateway
    product_context: dict | None
    profile_summary: dict | None      # only populated on complex path
    sales_output: str | None
    lead_signals: dict | None
    lead_score: float | None
    trace: list[NodeLog]
```

## Context Isolation Rule

No node receives the full conversation history by default. Each node's input is explicitly constructed from only the state fields it needs (e.g. Sales Agent gets `product_context` + `profile_summary`, not `user_message` history). This is enforced at the graph-edge level, not left to prompt discipline — the state fields passed between nodes are the actual scoping mechanism.

## Eval Plan

1. **Gateway classification eval** — labeled set of simple vs complex requests; measure precision/recall of the Gateway's classification independent of everything downstream. This is a deterministic grader (correct label or not).
2. **Simple path regression suite** — existing single-specialist behavior must not regress; deterministic + LLM-judge (calibrated) graders on response correctness.
3. **Complex path eval** — on requests that genuinely need cross-domain handling, measure whether the aggregated Sales Agent output correctly reflects both product facts and profile context (a grader that checks the response references both).
4. **Cost/latency comparison** — average tokens and wall-clock time for simple vs complex paths, and for complex path fan-out (parallel) vs a naive sequential version, to confirm the parallel execution is actually buying latency reduction (it will not reduce token cost — both branches still run).
5. **Lead scoring consistency** — since Deterministic Score is rule-based, this should be a straightforward regression test: same signals in, same score out, every time.

## Implementation Phases

### Phase 1 — Simple path skeleton
- Build FastAPI service + Intent Gateway (rule-based first) + the three simple-path specialists as direct calls, no Supervisor yet.
- Get the Gateway classification eval working before writing any complex-path code.

### Phase 2 — Complex path skeleton
- Add Supervisor node and the Product/Profile fan-out, stubbed outputs first, to validate the LangGraph wiring end to end.
- Add the structured trace logger to every node.

### Phase 3 — Real agent logic
- Implement Product Agent (shared RAG logic between simple and complex paths), Profile Agent, Sales Agent with the context-isolation rule enforced.

### Phase 4 — Lead pipeline
- Implement Lead Extractor and Deterministic Score, wire into CRM.

### Phase 5 — Eval harness + comparison
- Build the DeepEval suite covering all five eval points above.
- Produce a comparison table: simple vs complex path cost/latency, and (if useful) this adaptive design vs a naive "always run Supervisor" baseline, to make the case for the adaptive gate with data.

### Phase 6 — CI gate
- Wrap the eval suite in CI, gate merges on regression in Gateway precision/recall or simple-path correctness.

## Definition of Done
- [ ] Intent Gateway built and evaluated in isolation (precision/recall on labeled simple/complex set)
- [ ] Simple path fully functional with its own regression suite
- [ ] Complex path (Supervisor → fan-out → Sales Agent → Lead Extractor → Score → CRM) fully functional
- [ ] Context isolation enforced at the state/edge level, not just by prompt instruction
- [ ] Full eval suite (gateway accuracy, simple regression, complex correctness, cost/latency, lead score consistency) passing
- [ ] CI gate wired to the eval suite
- [ ] Every node emits structured trace logs
