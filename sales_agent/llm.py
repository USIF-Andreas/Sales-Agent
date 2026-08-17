from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sales_agent.classify import rule_classify
from sales_agent.config import (
    FAST,
    GATEWAY,
    MEDIUM,
    STRONG,
    ModelTier,
    get_provider,
    anthropic_api_key,
    groq_api_key,
)


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _approx_tokens(text: str) -> int:
    # Cheap heuristic so trace logs have plausible numbers without a tokenizer.
    return max(1, len(text) // 4)


class LLM:
    """Thin LLM abstraction with model tiers.

    Uses the Anthropic SDK (or Groq) when the corresponding API key is present,
    otherwise falls back to a deterministic mock provider so the whole system is
    runnable offline and the eval suite is hermetic.
    """

    def __init__(self) -> None:
        self.provider = get_provider()
        self._client = None
        if self.provider == "anthropic" and anthropic_api_key():
            import anthropic

            self._client = ("anthropic", anthropic.Anthropic(api_key=anthropic_api_key()))
        elif self.provider == "groq" and groq_api_key():
            try:
                from groq import Groq

                self._client = ("groq", Groq(api_key=groq_api_key()))
            except ImportError:  # groq SDK not installed -> degrade to mock
                self.provider = "mock"

    def complete(self, tier: ModelTier, system: str, user: str) -> LLMResult:
        if self._client is not None:
            kind, client = self._client
            if kind == "anthropic":
                return self._anthropic(client, tier, system, user)
            if kind == "groq":
                return self._groq(client, tier, system, user)
        return self._mock(tier, system, user)

    @staticmethod
    def _anthropic(client, tier: ModelTier, system: str, user: str) -> LLMResult:
        import time

        start = time.perf_counter()
        resp = client.messages.create(
            model=tier.model_name,
            max_tokens=tier.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResult(
            text=text,
            model=tier.model_name,
            provider="anthropic",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _groq(client, tier: ModelTier, system: str, user: str) -> LLMResult:
        import time

        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=tier.model_name,
            max_tokens=tier.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = resp.choices[0].message.content or ""
        return LLMResult(
            text=text,
            model=tier.model_name,
            provider="groq",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=latency_ms,
        )

    def _mock(self, tier: ModelTier, system: str, user: str) -> LLMResult:
        import time

        start = time.perf_counter()
        # Deterministic, fast, no-network response derived from the task.
        text = _mock_dispatch(tier, system, user)
        # Simulate model latency proportional to tier cost, large enough that the
        # parallel product/profile fan-out measurably beats the sequential baseline
        # even after LangGraph's orchestration overhead (which is tens of ms).
        time.sleep(0.10 if tier.cheap else 0.20)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return LLMResult(
            text=text,
            model=f"mock-{tier.model_name}",
            provider="mock",
            input_tokens=_approx_tokens(system + user),
            output_tokens=_approx_tokens(text),
            latency_ms=latency_ms,
        )


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_RAW_JSON = re.compile(r"(\{.*\})", re.DOTALL)


def parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    for pat in (_JSON_BLOCK, _RAW_JSON):
        m = pat.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return {}


# ---------------------------------------------------------------------------
# Deterministic mock provider
# ---------------------------------------------------------------------------
# The mock provider does not call any model. It returns structured, deterministic
# text for each task so that the eval suite is reproducible and the graph wiring
# can be validated end to end without network access or API keys. When an
# ANTHROPIC_API_KEY is present the real Anthropic models are used instead.


def _mock_dispatch(tier: ModelTier, system: str, user: str) -> str:
    # Key off the distinctive system-prompt string (not request content), so the
    # complex Sales Agent's prompt — which itself mentions "profile" — is not
    # misrouted to the Profile Agent.
    s = system.lower()
    if "gateway" in s:
        return _mock_gateway(system, user)
    if "lead" in s:
        return _mock_lead_extract(user)
    if "profile agent" in s:
        return _mock_profile(user)
    if "supervisor" in s:
        return _mock_sales_or_supervisor(system, user)
    if "sales agent" in s:
        return _mock_sales_or_supervisor(system, user)
    if "product specialist" in s:
        return _mock_product(user)
    if "sales specialist" in s:
        return _mock_sales_or_supervisor(system, user)
    if "support specialist" in s:
        return "ACK (support response)"
    return "ACK"


def _mock_gateway(system: str, user: str) -> str:
    # The gateway prompt asks for JSON {path, domains}. Build it deterministically
    # from the same rule-based classifier the real gateway uses.
    cls = rule_classify(user)
    return json.dumps({"path": cls.path, "domains": cls.domains})


def _mock_lead_extract(user: str) -> str:
    u = user.lower()
    signals = {
        "intent_to_buy": any(w in u for w in ["buy", "purchase", "order", "subscribe"]),
        "urgency": any(w in u for w in ["today", "now", "asap", "urgent", "immediately"]),
        "budget_mentioned": bool(re.search(r"\$\s?\d|\d{3,}\s?(usd|dollars|k\b)", u)),
    }
    return json.dumps(signals)


def _mock_product(user: str) -> str:
    from sales_agent.rag import CATALOG

    q = user.lower()
    for item in CATALOG:
        if item["name"].lower() in q or item["category"].lower() in q:
            return json.dumps(
                {
                    "found": True,
                    "name": item["name"],
                    "price": item["price"],
                    "summary": item["summary"],
                }
            )
    return json.dumps({"found": False, "summary": "No matching product in catalog."})


_BUDGET_RE = re.compile(r"\$\s?(\d[\d,]*)\b")
_URGENT_WORDS = ("today", "now", "asap", "urgent", "immediately", "immediate")
_BUY_WORDS = ("buy", "purchase", "order", "subscribe", "upgrade")


def _mock_profile(user: str) -> str:
    # The Profile Agent sees the raw request, so it can surface real context:
    # goals, budget and urgency are pulled through to the Sales Agent (which is
    # context-isolated and cannot see the raw message itself).
    u = user.lower()
    budget = _BUDGET_RE.search(user)
    has_urgency = any(w in u for w in _URGENT_WORDS)
    has_buy = any(w in u for w in _BUY_WORDS)
    goals = ["identify best-fit product"]
    if has_buy:
        goals.append("purchase within budget")
    constraints: list[str] = []
    if budget:
        constraints.append(f"budget of ${budget.group(1)}")
    if has_urgency:
        constraints.append("immediate need")
    return json.dumps({
        "background": "extracted profile background",
        "goals": goals,
        "constraints": constraints,
    })


def _mock_sales_or_supervisor(system: str, user: str) -> str:
    if "supervisor" in system.lower() or "branch" in user.lower():
        return json.dumps({"branches": ["product", "profile"]})
    # Sales Agent input is the isolated product+profile context. The mock echoes
    # the signals it finds there so the Lead Extractor has real material to work
    # with (and the CRM handoff is demonstrably non-trivial).
    u = user.lower()
    budget = _BUDGET_RE.search(user)
    urgent = any(w in u for w in _URGENT_WORDS)
    buy = any(w in u for w in _BUY_WORDS)
    parts = ["Recommendation grounded in product facts and the user's profile context."]
    if buy:
        parts.append("Given your goals, we recommend you buy the best-fit product.")
    if urgent:
        parts.append("Given your timeline, we recommend acting today.")
    if budget:
        parts.append(f"It fits within your ${budget.group(1)} budget.")
    parts.append("It best matches your stated goals and constraints.")
    return " ".join(parts)


__all__ = ["LLM", "LLMResult", "parse_json"]


# Re-export so nodes can import the tier constants directly.
from sales_agent.config import (  # noqa: E402
    FAST,
    GATEWAY,
    MEDIUM,
    STRONG,
)
