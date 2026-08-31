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

from pipeline.textutil import classify_attribute
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

# The two decline templates name the attribute being declined:
#   "I don't have a preference for {attr}; please use your judgment."   (boundary)
#   "I don't have an additional preference for {attr}."                 (exhausted)
# The attribute carries no product signal, but it is a strong instruction to the
# question policy: never ask for it again.
NO_PREFERENCE_RE = re.compile(
    r"^\s*I don't have (?:an additional|a) preference for\s+(?P<attribute>[^.;]+)",
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


def fill_slots(state: SharedSessionState) -> None:
    """Project retrieval evidence onto ACTIVE conversational slots.

    `state.constraints` retains every disclosed value because historical
    constraints can still help retrieval.

    `state.slots`, however, represents the user's current intent, so values
    explicitly superseded by an Intent Override are excluded.
    """
    slots: dict[str, list[str]] = {}

    for value in state.constraints:

        if value in state.superseded_constraints:
            continue

        slots.setdefault(
            classify_attribute(value),
            []
        ).append(value)

    state.slots = slots

def parse_no_preference(message: str) -> str | None:
    """-> the attribute the customer just declined, if any.

    Kept separate from parse_reply rather than widening its return tuple:
    tools/robustness.py unpacks that as a 2-tuple.
    """
    match = NO_PREFERENCE_RE.match(message)
    return match.group("attribute").strip().lower() if match else None


def erase_superseded(state: SharedSessionState) -> None:
    """Drop the preference the customer just replaced (Pillar II slot rewriting).

    The simulator's intent-override opener discloses `old_value`, and the override
    message discloses `new_value` -- both drawn from the SAME target's intent card
    (evaluator::behavior_for). So the "conflict" is between two true descriptions
    of one product, and erasing throws away a valid retrieval signal.

    That is an argument, not evidence, which is why this is a flag rather than a
    default. Enable with --erase-on-override and compare.
    """
    if state.override_turn is None or not state.constraints:
        return
    superseded = state.constraints[0]
    if superseded != state.override_value:
        state.constraints.remove(superseded)
        state.slots = {}
        fill_slots(state)


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
        fill_slots(state)
        return state

    declined = parse_no_preference(message)
    if declined:
        state.no_preference.add(declined)

    is_override, constraints = parse_reply(message)
    if is_override:
        state.intent = INTENT_OVERRIDE

        if state.override_turn is None:
            state.override_turn = turn

            if constraints:
                state.override_value = constraints[0]

                # The first constraint in an Intent Override session is the
                # preference being replaced.
                #
                # Keep it in state.constraints because it remains useful retrieval
                # evidence, but mark it as superseded so the conversational state
                # knows it is no longer an active preference.
                if state.constraints:
                    old_value = state.constraints[0]

                    if old_value != state.override_value:
                        state.superseded_constraints.add(old_value)
    for value in constraints:
        state.add_constraint(value)
    fill_slots(state)
    return state
