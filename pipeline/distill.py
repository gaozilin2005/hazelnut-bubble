"""Pillar III, retrieval half: Personalized Context Distillation.

The brief asks the agent to "leverage accumulated dialog history to perform
Personalized Context Distillation, continuously updating short-term session
states"; the repo's own spec calls the same thing "dynamic context
construction". Before this module, `state.constraints` was an append-only log:
every disclosed string kept forever, each scored independently, none ever
merged or reweighted.

Two operations, following the ADD / MERGE / DELETE shape that agent-memory
systems converge on (Mem0 and successors; see the self-evolving-agent survey,
arXiv:2507.21046). ADD and DELETE already exist elsewhere -- the router
appends, and `router.erase_superseded` handles override deletion -- so what is
missing here is MERGE, plus the reweighting that makes a merged state useful.

    merge_redundant()      one fact stated twice is one fact
    live_discriminance()   a fact everything satisfies is not evidence

Both are measured, not assumed: context compaction is known to silently drop
information that mattered (arXiv:2606.22528), so each is behind its own flag
and each was compared against no-distillation on three datasets.
"""
from __future__ import annotations

import math

from pipeline.textutil import normalize, terms


def merge_redundant(constraints: list[str]) -> list[str]:
    """Collapse constraints that restate the same fact, keeping the specific one.

    The simulator draws constraints from a product's own metadata, which
    routinely says the same thing at two granularities -- a real traced
    session disclosed both "leather" and "100% Leather". Scored independently
    those are two matches for one fact, so a product mentioning leather twice
    outranks one mentioning it once as precisely.

    Redundant means one constraint's informative tokens are a subset of
    another's. The superset survives: it is strictly more specific, and the
    retriever's phrase-containment bonus can still fire on it.
    """
    # Positions are carried through so the result keeps the caller's ordering:
    # constraints[0] is the customer's first-stated requirement and other code
    # treats that position as meaningful.
    kept: list[tuple[int, str, frozenset[str]]] = []
    for position, constraint in enumerate(constraints):
        tokens = frozenset(terms(normalize(constraint)))
        if not tokens:
            # No indexable tokens (punctuation, a stray single character).
            # Keep it: it cannot match anything downstream, but silently
            # deleting state we cannot reason about is precisely the
            # compaction failure mode this module is supposed to avoid.
            kept.append((position, constraint, tokens))
            continue
        # Drop if an already-kept constraint covers every token of this one.
        # `<=` catches the strict-subset case AND exact duplicates, so a
        # repeated string is emitted once rather than twice.
        if any(other and tokens <= other for _, _, other in kept):
            continue
        # This one is strictly more specific than an earlier keeper: evict it.
        kept = [(p, c, o) for p, c, o in kept if not (o and o < tokens)]
        kept.append((position, constraint, tokens))

    kept.sort(key=lambda row: row[0])
    return [constraint for _, constraint, _ in kept]


def live_discriminance(
    constraints: list[str],
    satisfied_counts: dict[str, int],
    pool_size: int,
) -> dict[str, float]:
    """Weight each constraint by how much it narrows the LIVE candidate pool.

    Global IDF answers "how rare is this word in the catalog". That is the
    wrong question once a category filter has already run: inside "women's
    leather riding boots", *every* candidate says leather, so the word is
    globally rare but locally worthless. A traced miss had four disclosed
    constraints, all satisfied by essentially the whole surviving bucket --
    together they contributed score without contributing information, and the
    target sat at rank 11 inside a 0.019-wide stalemate.

    Weight is self-information, -log2(p), where p is the fraction of the live
    pool satisfying the constraint: a constraint everything satisfies scores 0,
    one that halves the pool scores 1, one that isolates 1-in-16 scores 4.
    Normalised so the mean weight is 1.0, which keeps overall score magnitude
    comparable to the undistilled scorer and leaves existing tuned constants
    (PHRASE_BONUS, SNIPPET_BONUS, the priors) on the same scale.
    """
    if not constraints or pool_size <= 0:
        return {}
    raw: dict[str, float] = {}
    for constraint in constraints:
        count = satisfied_counts.get(constraint, 0)
        if count <= 0:
            # Nothing in the live pool satisfies it. Treat as maximally
            # informative rather than dividing by zero -- it is the only
            # evidence that could still separate a candidate we have not
            # scored yet.
            raw[constraint] = math.log2(max(pool_size, 2))
            continue
        fraction = min(1.0, count / pool_size)
        raw[constraint] = -math.log2(fraction)
    total = sum(raw.values())
    if total <= 0:
        # Every constraint is satisfied by everything: no ordering information
        # survives. Return uniform weights rather than all-zero, which would
        # silently discard the constraint signal entirely.
        return {c: 1.0 for c in constraints}
    scale = len(raw) / total
    return {c: w * scale for c, w in raw.items()}
