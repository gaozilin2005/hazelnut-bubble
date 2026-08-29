"""Held-out generalization check: same evaluator, different targets.

The 200 public sessions are what every design decision in this repo was
measured and tuned against. This builds a second set of sessions on catalog
products that were NEVER used as a public target, in the official scenario
mix, and scores them with the SAME unmodified evaluator.

It reuses the evaluator's own session-generation path rather than inventing a
new one: a sample that omits intent_card/behavior is auto-derived by
evaluator.local_evaluator.materialize_hidden_fields() from the target
product's own fields, deterministically seeded by sample_id -- the identical
mechanism the organizer uses for the 800 private sessions. Only the target
selection and the user_profile are synthesized here.

This is a proxy, not the organizer's private set: their 800 sessions use
different users AND may include paraphrasing this harness does not model.
Treat it as a generalization sanity check, not an official score.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

# Official mix from docs/competition_specification.md: 40/40/15/5.
SCENARIO_MIX = (
    ("buying", 0.40), ("browsing", 0.40),
    ("intent_override", 0.15), ("boundary", 0.05),
)
# Empirical from the 200 public profiles (crosstab is exact, not sampled).
RATING_TO_STYLE = {
    1.0: "critical", 2.0: "critical", 3.0: "critical",
    4.0: "mixed", 5.0: "usually positive",
}
RATING_WEIGHTS = {5.0: 134, 1.0: 14, 4.0: 21, 3.0: 22, 2.0: 9}
TAG_POOL = ("fit", "comfort", "durability", "style", "material", "weather", "warmth", "performance")
TAG_WEIGHTS = (163, 144, 47, 101, 154, 12, 18, 26)


def synthetic_profile(rng: random.Random) -> dict:
    rating = rng.choices(list(RATING_WEIGHTS), weights=list(RATING_WEIGHTS.values()))[0]
    tags = rng.sample(TAG_POOL, k=rng.choice((2, 2, 3, 3, 4)), )
    # weighted sample without replacement, matching the observed tag frequency
    tags = list(dict.fromkeys(
        rng.choices(TAG_POOL, weights=TAG_WEIGHTS, k=8)
    ))[: rng.choice((2, 3, 3, 4))]
    style = RATING_TO_STYLE[rating]
    return {
        "average_prior_rating": rating,
        "preference_tags": tags,
        "purchase_frequency": "3-4 prior purchases",
        "rating_style": style,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {style}.",
    }


def build_sessions(
    catalog_ids: set[str], excluded: set[str], count: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    pool = sorted(catalog_ids - excluded)
    rng.shuffle(pool)
    if len(pool) < count:
        raise SystemExit(f"catalog only has {len(pool)} non-public products, need {count}")

    scenarios: list[str] = []
    for name, fraction in SCENARIO_MIX:
        scenarios.extend([name] * round(count * fraction))
    while len(scenarios) < count:
        scenarios.append("browsing")
    scenarios = scenarios[:count]
    rng.shuffle(scenarios)

    sessions = []
    for i, (target, scenario) in enumerate(zip(pool, scenarios), start=1):
        sessions.append({
            "sample_id": f"heldout_{i:04d}",
            "scenario_type": scenario,
            "category_bucket": "clothing",
            "difficulty_bucket": "medium",
            "ground_truth": {"parent_asin": target},
            "user_profile": synthetic_profile(rng),
            # intent_card / behavior deliberately omitted: the evaluator derives
            # them from `target`'s own catalog fields via materialize_hidden_fields.
        })
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="pipeline", choices=("pipeline", "baseline"))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-dataset", default="data/public_set.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    public_samples = load_jsonl(args.public_dataset)
    excluded = {str(s["ground_truth"]["parent_asin"]) for s in public_samples}
    catalog_ids, categories, products = catalog_index(args.catalog)

    sessions = build_sessions(catalog_ids, excluded, args.count, args.seed)
    overlap = excluded & {s["ground_truth"]["parent_asin"] for s in sessions}
    assert not overlap, f"held-out targets leaked into the public set: {overlap}"

    if args.agent == "pipeline":
        from pipeline.agent import PipelineAgent
        agent = PipelineAgent(args.catalog)
    else:
        from starter.agent import Agent
        agent = Agent(args.catalog)

    result = evaluate(agent, sessions, catalog_ids, categories, products)
    result["note"] = (
        f"{args.count} synthetic sessions on catalog products NOT among the "
        "200 public targets, generated via the evaluator's own "
        "materialize_hidden_fields(). Proxy for generalization, not an "
        "official score -- the organizer's 800 private sessions use "
        "different users and may include paraphrasing this does not model."
    )
    output = Path(args.output or f"results_heldout_{args.agent}.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
