from __future__ import annotations

import logging
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sales_agent.graph import run
from sales_agent.trace import summarize_trace
from scripts.compare_paths import build_rows

logging.basicConfig(level=logging.INFO)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = FastAPI(title="Adaptive Sales Agent", version="0.1.0")


class ChatRequest(BaseModel):
    user_message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    path: str | None
    domains: list[str]
    agent: str | None = None
    lead_score: float | None = None
    lead_signals: dict | None = None
    crm_result: dict | None = None
    trace: list[dict]
    trace_summary: dict


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.user_message or not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")
    session_id = req.session_id or str(uuid.uuid4())
    state = run(req.user_message, session_id=session_id)
    response = state.get("direct_response") or state.get("sales_output") or ""
    return ChatResponse(
        response=response,
        path=state.get("path"),
        domains=state.get("domains", []),
        agent=state.get("direct_agent"),
        lead_score=state.get("lead_score"),
        lead_signals=state.get("lead_signals"),
        crm_result=state.get("crm_result"),
        trace=state.get("trace", []),
        trace_summary=summarize_trace(state),
    )


@app.get("/compare")
def compare():
    rows = build_rows()
    adaptive_simple = rows[0]
    always = rows[3]
    adaptive_complex = rows[1]
    sequential = rows[2]
    return {
        "rows": rows,
        "savings": {
            "token_savings": adaptive_simple["in_tok"] - always["in_tok"],
            "node_savings": adaptive_simple["nodes"] - always["nodes"],
            "parallel_token_delta": adaptive_complex["in_tok"] - sequential["in_tok"],
            "parallel_latency_delta_ms": round(adaptive_complex["lat_ms"] - sequential["lat_ms"], 1),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the static frontend (routes above take precedence over the mount).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)