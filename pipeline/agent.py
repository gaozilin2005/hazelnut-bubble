"""Agent wiring: router -> retriever -> reranker.

The clarification policy here is a PLACEHOLDER standing in for Person B. It is
deliberately minimal so Person A's Hit@10 and MRR are measurable in isolation.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.interfaces import SharedSessionState
from pipeline.reranker import IdentityReranker
from pipeline.retriever import HybridRetriever
from pipeline.router import route

CANDIDATE_POOL = 200


class PipelineAgent:
    def __init__(
        self, catalog_path: str | Path = "data/catalog.jsonl", use_prior: bool = True
    ) -> None:
        self.retriever = HybridRetriever(use_prior=use_prior)
        self.retriever.build(str(catalog_path))
        self.reranker = IdentityReranker()
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

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.states.get(session_id)
        if state is None:
            self.reset(session_id, {})
            state = self.states[session_id]

        route(state, user_message, turn)
        candidates = self.retriever.retrieve(state, CANDIDATE_POOL)
        ranked = self.reranker.rerank(state, candidates[:top_k])
        message, attribute = self._ask(state)

        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin, _ in ranked[:top_k]],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
