"""Semantic reranking of the retriever's shortlist.

Two implementations behind one protocol:

  LocalReranker  -- no network, no credentials, always available.
  LLMReranker    -- Claude listwise rerank, falling back to LocalReranker on any
                    failure (missing SDK, missing credentials, network disabled,
                    malformed response).

The fallback is not decoration. Submission rules warn that official scoring may
run with network access disabled under CPU and timeout limits, and the organizer
supplies no credentials, so LocalReranker is the path that actually runs unless
a team supplies its own key. LLMReranker is opt-in via ranking_model=.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from pipeline.interfaces import SharedSessionState
from pipeline.textutil import normalize

RANKING_MODEL = "claude-opus-5"
MAX_LLM_CANDIDATES = 12
SUMMARY_CHARS = 150


class _Corpus(Protocol):
    asins: list[str]
    corpora: list[str]
    snippets: list[list[str]]


class LocalReranker:
    """Reorders by DISTINCT constraint coverage, then by retrieval score.

    The retriever sums a weighted score across constraints, so one constraint
    matched emphatically can outrank three matched quietly. For MRR the opposite
    is wanted: the target satisfies every constraint the customer stated, because
    the constraints were drawn from it. Counting satisfied constraints first, and
    using the retriever's score only to break ties, encodes that directly.
    """

    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self._position = {asin: i for i, asin in enumerate(retriever.asins)}

    def _covered(self, index: int, state: SharedSessionState) -> int:
        corpus = self.retriever.corpora[index]
        count = 0
        for constraint in state.constraints:
            text = normalize(constraint)
            if not text:
                continue
            tokens = [t for t in text.split() if t in self.retriever.idf]
            if not tokens:
                continue
            # Satisfied = the whole phrase appears, or every informative token does.
            if text in corpus or all(t in corpus for t in tokens):
                count += 1
        return count

    def rerank(
        self, state: SharedSessionState, candidate_ids: list[str]
    ) -> list[tuple[str, float]]:
        if not state.constraints or not candidate_ids:
            total = len(candidate_ids)
            return [(a, float(total - i)) for i, a in enumerate(candidate_ids)]
        blob = self.retriever._blob(state)
        scored: list[tuple[int, float, str]] = []
        for rank, asin in enumerate(candidate_ids):
            index = self._position.get(asin)
            if index is None:
                scored.append((-1, 0.0, asin))
                continue
            scored.append(
                (self._covered(index, state), self.retriever.score(index, state, blob), asin)
            )
        scored.sort(key=lambda row: (-row[0], -row[1]))
        return [(asin, float(cover) + score) for cover, score, asin in scored]


class LLMReranker:
    """Listwise rerank with Claude. Falls back to LocalReranker on any failure.

    Untested against the live API: no credentials were available in the build
    environment. The fallback path is the tested one.
    """

    def __init__(self, retriever, model: str = RANKING_MODEL) -> None:
        self.model = model
        self.fallback = LocalReranker(retriever)
        self.retriever = retriever
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._position = {asin: i for i, asin in enumerate(retriever.asins)}
        self._client = None
        self._disabled = False

    def _connect(self):
        if self._client is not None or self._disabled:
            return self._client
        try:
            import anthropic  # imported lazily so the offline path needs no SDK
        except ImportError:
            self._disabled = True
            return None
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # An `ant auth login` profile also authenticates, so try anyway and
            # let a failed first call disable this permanently.
            pass
        try:
            self._client = anthropic.Anthropic()
        except Exception:
            self._disabled = True
        return self._client

    def _summary(self, asin: str) -> str:
        index = self._position.get(asin)
        if index is None:
            return asin
        snippets = self.retriever.snippets[index][:2]
        text = " | ".join(snippets) if snippets else self.retriever.corpora[index]
        return text[:SUMMARY_CHARS]

    def rerank(
        self, state: SharedSessionState, candidate_ids: list[str]
    ) -> list[tuple[str, float]]:
        baseline = self.fallback.rerank(state, candidate_ids)
        client = self._connect()
        if client is None or not state.constraints or len(candidate_ids) < 2:
            return baseline

        shortlist = [asin for asin, _ in baseline][:MAX_LLM_CANDIDATES]
        catalogue = "\n".join(
            f"{i}. {self._summary(asin)}" for i, asin in enumerate(shortlist)
        )
        requirements = "\n".join(f"- {c}" for c in state.constraints)
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                output_config={"effort": "low"},
                system=(
                    "You rank shopping search results. The customer has one specific "
                    "product in mind. Given their stated requirements and a numbered "
                    "candidate list, return the candidate numbers ordered from best to "
                    "worst match. Reply with the numbers only, comma-separated, every "
                    "candidate included exactly once, and nothing else."
                ),
                messages=[{
                    "role": "user",
                    "content": f"Requirements:\n{requirements}\n\nCandidates:\n{catalogue}",
                }],
            )
        except Exception:
            # Any failure -- auth, network disabled, rate limit, timeout -- degrades
            # to the local ordering rather than costing the session a turn.
            self._disabled = True
            return baseline

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        text = " ".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        order: list[str] = []
        seen: set[int] = set()
        for token in re.findall(r"\d+", text):
            position = int(token)
            if position < len(shortlist) and position not in seen:
                seen.add(position)
                order.append(shortlist[position])
        if not order:
            return baseline
        # Anything the model dropped keeps its local ordering, appended.
        order.extend(asin for asin, _ in baseline if asin not in set(order))
        return [(asin, float(len(order) - i)) for i, asin in enumerate(order)]


class IdentityReranker:
    """Preserves retriever order. Kept as the ablation baseline."""

    def rerank(
        self, state: SharedSessionState, candidate_ids: list[str]
    ) -> list[tuple[str, float]]:
        count = len(candidate_ids)
        return [(asin, float(count - rank)) for rank, asin in enumerate(candidate_ids)]
