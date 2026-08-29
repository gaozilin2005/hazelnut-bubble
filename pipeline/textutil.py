"""Shared text normalization. Used by router, retriever, and reranker."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
WS_RE = re.compile(r"\s+")

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
})

SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


def flatten(value: object) -> str:
    """Mirror the evaluator's field flattening so our corpus matches what the
    simulator drew its constraint strings from."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def normalize(text: str) -> str:
    """Lowercase, strip non-alphanumerics to single spaces.

    The simulator's constraint strings are verbatim slices of a product's own
    features/details, differing only by whitespace collapse, punctuation strip,
    and a 180-char truncation. Normalizing both sides makes containment an
    exact test despite those transformations.
    """
    return WS_RE.sub(" ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def terms(text: str, keep_stopwords: bool = False) -> list[str]:
    found = TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return [token for token in found if len(token) > 1]
    return [token for token in found if len(token) > 1 and token not in STOPWORDS]


def product_corpus(product: dict) -> str:
    return " ".join(flatten(product.get(field)) for field in SEARCH_FIELDS)
