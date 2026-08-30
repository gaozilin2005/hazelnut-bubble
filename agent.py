"""Submission entry point: exports `Agent` as the rules require.

    docs/submission_rules.md -> "one Python agent entry file exporting `Agent`"

The organizer's harness constructs `Agent(catalog_path)` and calls `reset` /
`respond`. `PipelineAgent` already implements that contract; this module exists
so the required name is exported from a single obvious place, without renaming
the class every other module and tool refers to.

Defaults here are the configuration all reported scores use: the integrated
dialog policy, the local reranker, the exposure gate on, and no network access.
Every experimental flag documented in the README is off.

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
