from __future__ import annotations

import json
import os

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "catalog.json")


def _load_catalog() -> list[dict]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CATALOG: list[dict] = _load_catalog()


def lookup(query: str) -> dict | None:
    """Simple keyword lookup over the product catalog (stand-in for RAG)."""
    q = query.lower()
    for item in CATALOG:
        if item["name"].lower() in q or item["category"].lower() in q:
            return item
    # Fall back to token overlap.
    best, best_score = None, 0
    for item in CATALOG:
        score = sum(1 for tok in q.split() if tok in item["summary"].lower())
        if score > best_score:
            best, best_score = item, score
    return best if best_score else None
