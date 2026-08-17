from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTier:
    """Model tiers from the plan's model-tiering table."""

    name: str
    model_name: str
    max_tokens: int
    # Whether this tier is cheap enough to call on every request (gateway / lead).
    cheap: bool


# Default tiers using Anthropic (can be overridden via env)
GATEWAY = ModelTier("gateway", "claude-3-5-haiku-latest", 256, True)
FAST = ModelTier("fast", "claude-3-5-haiku-latest", 1024, True)
MEDIUM = ModelTier("medium", "claude-3-5-sonnet-latest", 1500, False)
STRONG = ModelTier("strong", "claude-3-5-sonnet-latest", 1500, False)


# Groq model overrides (optional)
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
GROQ_MODEL_MEDIUM = os.getenv("GROQ_MODEL_MEDIUM", "mixtral-8x7b-32768")
GROQ_MODEL_STRONG = os.getenv("GROQ_MODEL_STRONG", "gemma2-9b-it")

# Provider detection: "anthropic" | "groq" | "mock"
def get_provider() -> str:
    """Return the LLM provider to use.

    Order of precedence:
    1. LLM_PROVIDER env var (if set)
    2. ANTHROPIC_API_KEY present -> "anthropic"
    3. GROQ_API_KEY present -> "groq"
    4. Default "mock" (no API key needed, runs offline)
    """
    provider = os.getenv("LLM_PROVIDER")
    if provider:
        return provider.lower()

    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"

    if os.getenv("GROQ_API_KEY"):
        return "groq"

    return "mock"


def anthropic_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY")


def groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")
