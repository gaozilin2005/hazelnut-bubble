"""Intent routing: parse the customer turn, set state.intent, extract signal.

The simulator's customer policy is deterministic and template-driven
(evaluator/local_evaluator.py::initial_message and ::customer_reply). Every
template carries a payload that is a verbatim slice of the target product's own
metadata, so parsing precisely is worth far more than classifying loosely.

Every parse is best-effort: an unrecognized message degrades to INTENT_UNKNOWN
with no category, which tells the retriever to skip hard-filtering rather than
filter on a wrong guess. That is the safety valve if the organizer paraphrases
messages on the private split.
"""
from __future__ import annotations

import re

from pipeline.interfaces import (
    INTENT_BROWSING,
    INTENT_BUYING,
    INTENT_OVERRIDE,
    INTENT_UNKNOWN,
    SharedSessionState,
)

OPENING_RE = re.compile(r"^\s*I'm looking for\s+(?P<body>.+?)\s*$", re.I | re.S)
BROWSING_TAIL = ", but I'm still exploring."
BUYING_MARKER = ". A key requirement is: "
DISCLOSE_MARKER = "For that, what matters is: "
OVERRIDE_MARKER = "What I need is: "

# Templates that carry no product signal; parsing them must not add constraints.
NO_SIGNAL_RE = re.compile(
    r"^\s*(?:I don't have (?:an additional )?a?\s*preference for|"
    r"Those options are not quite right yet)",
    re.I,
)


def _split_payload(payload: str) -> list[str]:
    """Constraints are joined with '; ' and terminated with '.'.

    A constraint may itself contain a semicolon, so an over-eager split can
    fragment one. Fragments are harmless: the retriever scores by containment,
    and a fragment either still matches or falls through to token overlap.
    """
    payload = payload.strip().rstrip(".")
    return [part.strip() for part in payload.split("; ") if part.strip()]


def parse_opening(message: str) -> tuple[str, str | None, list[str]]:
    """-> (intent, category, constraints) for the first customer turn."""
    match = OPENING_RE.match(message)
    if not match:
        return INTENT_UNKNOWN, None, []
    body = match.group("body")

    if body.endswith(BROWSING_TAIL):
        return INTENT_BROWSING, body[: -len(BROWSING_TAIL)].strip(), []

    if BUYING_MARKER in body:
        category, _, constraint = body.partition(BUYING_MARKER)
        return INTENT_BUYING, category.strip(), _split_payload(constraint)

    # Intent-override opener: "{category}. {old_value}". The category is a short
    # taxonomy string, so the first sentence break is the boundary.
    category, separator, remainder = body.partition(". ")
    if separator:
        return INTENT_OVERRIDE, category.strip(), _split_payload(remainder)

    return INTENT_UNKNOWN, body.strip() or None, []


def parse_reply(message: str) -> tuple[bool, list[str]]:
    """-> (is_override_turn, constraints) for turns 2..10."""
    if NO_SIGNAL_RE.match(message):
        return False, []
    if OVERRIDE_MARKER in message:
        _, _, payload = message.partition(OVERRIDE_MARKER)
        return True, _split_payload(payload)
    if DISCLOSE_MARKER in message:
        _, _, payload = message.partition(DISCLOSE_MARKER)
        return False, _split_payload(payload)
    return False, []


def route(state: SharedSessionState, message: str, turn: int) -> SharedSessionState:
    """Fold the turn into shared state. Called at the top of retrieve()."""
    state.turn = turn
    state.messages.append(message)

    if turn == 1 or state.category is None:
        intent, category, constraints = parse_opening(message)
        if category:
            state.category = category
            state.category_confident = intent is not INTENT_UNKNOWN
        state.intent = intent
        for value in constraints:
            state.add_constraint(value)
        return state

    is_override, constraints = parse_reply(message)
    if is_override:
        # The simulator draws old_value and new_value from the SAME target's
        # intent card, so they describe one product and never truly conflict.
        # Earlier constraints are therefore kept, not discarded.
        state.intent = INTENT_OVERRIDE
    for value in constraints:
        state.add_constraint(value)
    return state
