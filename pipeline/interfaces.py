"""Shared contracts between Person A (retrieval), B (dialog), and C (harness).

Provisional: the Block-0 contract was never committed, so this is written to the
shape the work-split describes. B and C should amend rather than fork.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Written by the router (A); read by the router and the dialog policy (B).
INTENT_BUYING = "buying"
INTENT_BROWSING = "browsing"
INTENT_OVERRIDE = "override"
INTENT_UNKNOWN = "unknown"


@dataclass
class SharedSessionState:
    session_id: str
    user_profile: dict = field(default_factory=dict)
    turn: int = 0

    # --- A writes ---
    intent: str = INTENT_UNKNOWN
    # Turn on which the customer actually replaced a preference. Distinct from
    # `intent`: an intent-override session is labelled OVERRIDE from its opening
    # message, but the override event itself lands on turn 3 or 4.
    override_turn: int | None = None
    override_value: str | None = None   # what the customer switched TO, verbatim
    category: str | None = None          # parsed coarse category, verbatim
    category_confident: bool = False     # False => retrieval must not hard-filter

    # --- B writes ---
    slots: dict[str, list[str]] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)

    # Attributes the customer has explicitly declined to constrain. The router
    # detects these (the simulator answers "I don't have a preference for X")
    # but until now discarded them; a question policy that re-asks a declined
    # attribute burns a turn for nothing.
    no_preference: set[str] = field(default_factory=set)

    # --- both write ---
    # Raw disclosed constraint strings, in disclosure order. These are verbatim
    # slices of the target's own metadata and are the strongest ranking signal
    # available, so they are kept unparsed alongside B's structured slots.
    constraints: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def add_constraint(self, value: str) -> None:
        value = value.strip()
        if value and value not in self.constraints:
            self.constraints.append(value)


@runtime_checkable
class Retriever(Protocol):
    def build(self, catalog_path: str) -> None: ...
    def retrieve(self, state: SharedSessionState, k: int) -> list[str]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, state: SharedSessionState, candidate_ids: list[str]
    ) -> list[tuple[str, float]]: ...
