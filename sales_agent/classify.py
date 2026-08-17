from __future__ import annotations

import re
from dataclasses import dataclass

from sales_agent.rag import CATALOG

SUPPORT_KW = [
    "refund", "return", "warranty", "broken", "issue", "problem", "not working",
    "help me fix", "ship", "shipping", "delivery", "track", "cancel", "defective",
]
PRODUCT_KW = ["price", "cost", "spec", "specifications", "what is", "tell me about",
              "features", "compare", "details", "how much", "waranty"]
SALES_KW = ["buy", "purchase", "discount", "coupon", "deal", "subscribe",
            "upgrade", "plan", "quote", "checkout"]
RECOMMEND_KW = ["recommend", "suggest", "which", "should i", "best for",
                "help me choose", "advice", "what should", "good for"]
PROFILE_KW = [
    "i'm a", "i am a", "i work", "my budget", "as a", "we are", "our team",
    "i'm looking", "i am looking", "my company", "my office", "my home",
    "i'm setting", "i have a", "remote worker", "student",
]

PRODUCT_TERMS = [i["name"].lower() for i in CATALOG] + [i["category"].lower() for i in CATALOG]


@dataclass
class Classification:
    path: str | None
    domains: list[str]
    confidence: str


def rule_classify(message: str) -> Classification:
    """Classify a message into a path (simple/complex) and the domains it touches.

    Cheap rule-based heuristic used by the Intent Gateway and the mock LLM
    provider. Low confidence means the request is ambiguous and should be sent
    to a fast model call instead.
    """
    msg = message.lower()
    domains: set[str] = set()

    if any(k in msg for k in SUPPORT_KW):
        domains.add("support")
    if any(term in msg for term in PRODUCT_TERMS) or any(k in msg for k in PRODUCT_KW):
        domains.add("product")
    if any(k in msg for k in SALES_KW):
        domains.add("sales")

    profile_signal = any(re.search(r"\b" + re.escape(k) + r"\b", msg) for k in PROFILE_KW)
    if profile_signal:
        domains.add("profile")
    recommend_signal = any(k in msg for k in RECOMMEND_KW)
    multi_product = sum(1 for term in PRODUCT_TERMS if term in msg) >= 2

    if recommend_signal and (profile_signal or multi_product or "product" in domains):
        domains.update(["product", "profile", "sales"])
        return Classification("complex", sorted(domains), "high")
    if len(domains) >= 2:
        return Classification("complex", sorted(domains), "medium")
    if not domains:
        return Classification(None, [], "low")
    return Classification("simple", sorted(domains), "high")