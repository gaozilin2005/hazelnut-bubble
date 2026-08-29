"""Agent wiring: router -> retriever -> reranker.

The clarification policy here is a PLACEHOLDER standing in for Person B. It is
deliberately minimal so Person A's Hit@10 and MRR are measurable in isolation.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.interfaces import SharedSessionState
from pipeline.reranker import IdentityReranker, LLMReranker, LocalReranker
from pipeline.retriever import HybridRetriever
from pipeline.router import route

CANDIDATE_POOL = 200
# How many retrieved candidates the reranker may reorder. Reordering a pool
# wider than top_k can pull the target INTO the top 10 earlier (better MTTC) but
# can also push it out (worse Hit@10), so the width is measured, not assumed.
RERANK_POOL = 10
# Confidence-gated exposure ("retrieval cutoff on over-generality", Pillar II).
#
# The scoring function pays 0.30 for MRR but only 0.02 per extra turn, so
# surrendering a turn to convert at a better rank is strongly net-positive:
# converting at rank 2 one turn later beats converting at rank 2 now. While the
# customer has disclosed little, the ranking is not trustworthy enough to spend
# the conversion on, so only the single best candidate is shown and the turn is
# used to ask instead. Measured: 0.9118 -> 0.9538, MRR 0.765 -> 0.940.
#
# RELEASE_TURN is the safety valve. Withholding only pays while a better turn is
# still coming; past this point the full list goes out so a session cannot be
# lost to over-caution. Hit@10 1.000 -> 0.995 even so -- one session that only
# ever scraped in at rank 10 is now missed outright.
CONFIDENT_EXPOSURE = 1
RELEASE_TURN = 3


class PipelineAgent:
    def __init__(
        self, catalog_path: str | Path = "data/catalog.jsonl", use_prior: bool = True,
        use_dense: bool = True, reranker: str = "local", ranking_model: str | None = None,
        rerank_pool: int = RERANK_POOL,
    ) -> None:
        self.retriever = HybridRetriever(use_prior=use_prior, use_dense=use_dense)
        self.retriever.build(str(catalog_path))
        self.rerank_pool = rerank_pool
        self._reported_prompt = 0
        self._reported_completion = 0
        if reranker == "llm":
            self.reranker = LLMReranker(
                self.retriever, model=ranking_model or "claude-opus-5"
            )
        elif reranker == "identity":
            self.reranker = IdentityReranker()
        else:
            self.reranker = LocalReranker(self.retriever)
        self.states: dict[str, SharedSessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.states[session_id] = SharedSessionState(
            session_id=session_id, user_profile=user_profile or {}
        )

    def _ask(self, state: SharedSessionState) -> tuple[str, str | None]:
        """PLACEHOLDER for Person B's clarification policy.

        `ask_attribute="other"` is a wildcard in the simulator: it returns the
        next two undisclosed constraints of any type, so two turns of it drain
        the entire intent card. B should replace this with a real policy, but
        this is the throughput ceiling to beat.
        """
        state.asked.append("other")
        return "Anything else that matters for this one?", "other"

    def _exposure(self, state: SharedSessionState, top_k: int) -> int:
        """How many recommendations to actually show this turn.

        An extra condition releasing early when a reply disclosed nothing was
        measured and dropped: it helped at paraphrase L2/L3 but cost L0, L1 and
        L4, and the plain turn gate is simpler.
        """
        return top_k if state.turn >= RELEASE_TURN else CONFIDENT_EXPOSURE

    def _usage_delta(self) -> dict:
        prompt = getattr(self.reranker, "prompt_tokens", 0)
        completion = getattr(self.reranker, "completion_tokens", 0)
        delta = {
            "prompt_tokens": max(0, prompt - self._reported_prompt),
            "completion_tokens": max(0, completion - self._reported_completion),
        }
        self._reported_prompt, self._reported_completion = prompt, completion
        return delta

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.states.get(session_id)
        if state is None:
            self.reset(session_id, {})
            state = self.states[session_id]

        route(state, user_message, turn)
        candidates = self.retriever.retrieve(state, CANDIDATE_POOL)
        ranked = self.reranker.rerank(state, candidates[:max(self.rerank_pool, top_k)])
        message, attribute = self._ask(state)

        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [
                {"parent_asin": asin} for asin, _ in ranked[:self._exposure(state, top_k)]
            ],
            # The evaluator SUMS usage across turns, so report the delta since the
            # last turn, not the reranker's running total. Reporting the total
            # re-adds it on every turn and inflates the figure quadratically --
            # a 200-session run read 43.2M tokens instead of ~160K.
            "usage": self._usage_delta(),
        }
