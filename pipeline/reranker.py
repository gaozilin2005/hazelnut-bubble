"""Reranking. Identity for now, per the Day-1 plan.

A target that is not in the shortlist cannot be reranked, so Hit@10 is settled
before this module matters. Day 3 swaps in a cross-encoder here IF measurement
shows headroom -- note the submission may be scored with network disabled under
CPU/timeout limits, so any model added here must be bundled and cheap.
"""
from __future__ import annotations

from pipeline.interfaces import SharedSessionState


class IdentityReranker:
    """Reranker protocol. Preserves retriever order."""

    def rerank(
        self, state: SharedSessionState, candidate_ids: list[str]
    ) -> list[tuple[str, float]]:
        count = len(candidate_ids)
        return [(asin, float(count - rank)) for rank, asin in enumerate(candidate_ids)]
