from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Guardrail results
# ---------------------------------------------------------------------------

class GuardrailResult:
    """Result of running all guardrails on a user message."""
    def __init__(
        self,
        is_safe: bool,
        reason: str | None = None,
        blocked_category: str | None = None,
        sanitized_message: str | None = None,
    ) -> None:
        self.is_safe = is_safe
        self.reason = reason
        self.blocked_category = blocked_category
        self.sanitized_message = sanitized_message

    def __bool__(self) -> bool:
        return self.is_safe

    def __repr__(self) -> str:
        return f"GuardrailResult(is_safe={self.is_safe}, reason={self.reason!r})"


# ---------------------------------------------------------------------------
# Off-topic / domain guardrail
# ---------------------------------------------------------------------------

# Keywords that indicate the user is asking about things outside the sales domain:
# - direct system prompts, jailbreak attempts
# - unrelated topics (politics, religion, etc.)
# - attempts to extract the system prompt or instructions
OFF_TOPIC_KW = [
    "jailbreak", "ignore previous", "system prompt", "forget instructions",
    "override", "disregard", "you are a", "you are an",
    "politics", "election", "vote", "president", "prime minister",
    "religion", "god", "church", "bible", "quran",
    "hate", "violence", "kill", "murder", "terrorist",
    "drugs", "illegal", "weapon", "gun", "bomb",
    "scam", "hack", "exploit", "vulnerability",
]

_OFF_TOPIC_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in OFF_TOPIC_KW) + r")\b",
    re.IGNORECASE,
)


def check_off_topic(message: str) -> GuardrailResult:
    """Detect off-topic or jailbreak attempts.

    Returns a GuardrailResult that is False if the message contains
    disallowed keywords or patterns.
    """
    if _OFF_TOPIC_PATTERN.search(message):
        return GuardrailResult(
            is_safe=False,
            reason="Off-topic or jailbreak attempt detected",
            blocked_category="off_topic",
        )
    return GuardrailResult(is_safe=True)


# ---------------------------------------------------------------------------
# Prompt injection / content guardrail
# ---------------------------------------------------------------------------

# Patterns that indicate prompt injection or attempt to subvert the LLM.
PROMPT_INJECTION_KW = [
    "say ", "repeat after me", "print ", "output ",
    "you are", "as an ai", "as a language model",
    "disregard your instructions", "ignore above",
]

_PROMPT_INJECTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in PROMPT_INJECTION_KW) + r")\b",
    re.IGNORECASE,
)


def check_prompt_injection(message: str) -> GuardrailResult:
    """Detect prompt injection attempts.

    Returns GuardrailResult(is_safe=False) if the message contains
    patterns designed to make the LLM ignore its system prompt.
    """
    if _PROMPT_INJECTION_PATTERN.search(message):
        return GuardrailResult(
            is_safe=False,
            reason="Prompt injection attempt detected",
            blocked_category="prompt_injection",
        )
    return GuardrailResult(is_safe=True)


# ---------------------------------------------------------------------------
# Content safety / harmful content guardrail
# ---------------------------------------------------------------------------

# Very simple keyword-based filter for extreme content.
HARMFUL_KW = [
    "kill", "murder", "hurt", "assault", "attack",
    "bomb", "explosive", "weapon", "gun", "shoot",
    "drugs", "self-harm", "suicide",
]

_HARMFUL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in HARMFUL_KW) + r")\b",
    re.IGNORECASE,
)


def check_harmful_content(message: str) -> GuardrailResult:
    """Detect potentially harmful content.

    Returns GuardrailResult(is_safe=False) if the message contains
    keywords associated with violence, self-harm, etc.
    """
    if _HARMFUL_PATTERN.search(message):
        return GuardrailResult(
            is_safe=False,
            reason="Potentially harmful content detected",
            blocked_category="harmful_content",
        )
    return GuardrailResult(is_safe=True)


# ---------------------------------------------------------------------------
# Main guardrail runner
# ---------------------------------------------------------------------------

def run_all_guardrails(message: str) -> GuardrailResult:
    """Run the full suite of guardrails on a user message.

    Order matters: off-topic check first, then prompt injection,
    then harmful content. The first failure short-circuits.
    """
    # 1. Off-topic / jailbreak
    result = check_off_topic(message)
    if not result.is_safe:
        return result

    # 2. Prompt injection
    result = check_prompt_injection(message)
    if not result.is_safe:
        return result

    # 3. Harmful content
    result = check_harmful_content(message)
    if not result.is_safe:
        return result

    return GuardrailResult(is_safe=True)


# Canned, safe response returned when a request is blocked by the guardrails.
BLOCKED_RESPONSE = (
    "I'm here to help with product questions, sales, and support. "
    "I can't assist with that request. Let me know if you have any product-related questions!"
)


# ---------------------------------------------------------------------------
# Sanitization helper (strip dangerous patterns, keep intent)
# ---------------------------------------------------------------------------

def sanitize_message(message: str) -> str:
    """Light sanitization: remove obvious jailbreak / injection patterns.

    This is a best-effort strip; it does not change the user's core intent.
    """
    msg = message.strip()

    # Remove "say/repeat/print" commands that could be used for injection
    msg = re.sub(r"\b(say|repeat|print)\b\s+", "", msg, flags=re.IGNORECASE)

    # Remove direct system prompt references
    msg = re.sub(r"\b(system prompt|forget instructions|ignore previous)\b[.!?]*",
                 "", msg, flags=re.IGNORECASE)

    # Collapse whitespace
    msg = re.sub(r"\s+", " ", msg).strip()

    return msg if msg else message