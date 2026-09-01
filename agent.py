"""Submission entry point: exports `Agent` as the rules require.

    docs/submission_rules.md -> "one Python agent entry file exporting `Agent`"

The organizer's harness constructs `Agent(catalog_path)` and calls `reset` /
`respond`. `PipelineAgent` already implements that contract; this module exists
so the required name is exported from a single obvious place, without renaming
the class every other module and tool refers to.

Defaults here are the configuration all reported scores use: the integrated
dialog policy, the local reranker, the exposure gate, the single-item walk,
and aspect-level negative feedback (neg_aspects=1.0) on; no network access.
Every other experimental flag documented in the README (--rrf, --broad-pool,
--len-norm, --distill, --no-repeat, --tie-break-dense) is off. The walk
decouples presentation from ranking -- see README "Single-Item Walk
Disclosure"; disable it with walk=False for a full-page surface instead.

    from agent import Agent
    a = Agent("data/catalog.jsonl")
    a.reset("s1", {"preference_tags": ["fit"]})
    a.respond("s1", "I'm looking for shirts.", turn=1, top_k=10)
"""
from __future__ import annotations

from pipeline.agent import PipelineAgent

__all__ = ["Agent"]


class Agent(PipelineAgent):
    """The submitted agent. See README.md for method, results and limitations."""
