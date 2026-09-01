"""Replay one real evaluation session turn by turn, printing both sides.

The scoring harness runs sessions silently and reports only aggregate metrics.
This prints the same session as a readable transcript -- the customer messages
the evaluator's own simulator generates, and what the agent does with each one --
so the behaviour behind a score can be inspected rather than inferred.

Nothing here is a mock: the customer side is the evaluator's `initial_message` /
`customer_reply`, and the agent side is the submitted `Agent`, on the same hidden
target the scorer uses. Only the printing is new.

    python3 tools/demo_session.py                      # first intent_override session
    python3 tools/demo_session.py --sample public_0002
    python3 tools/demo_session.py --scenario buying
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)

RULE = "=" * 78


def title(products: dict, asin: str, width: int = 58) -> str:
    text = str(products.get(asin, {}).get("title", asin)).strip()
    return text[:width] + ("…" if len(text) > width else "")


def show_routing(samples: list[dict], catalog: str) -> None:
    """Print how the router reads one opener of each scenario type.

    The opener's *shape* carries the intent: a browsing tail, a "key requirement"
    marker, or a bare "{category}. {value}" override opener. Routing happens
    before any retrieval, and decides which track the turn takes.
    """
    from pipeline.router import parse_opening

    catalog_ids, categories, products = catalog_index(catalog)
    print(RULE)
    print("PILLAR I — dual-track routing, one opener per scenario type")
    print(RULE)

    seen: set[str] = set()
    for sample in samples:
        scenario = sample["scenario_type"]
        if scenario in seen:
            continue
        seen.add(scenario)
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        message = initial_message(effective, coarse_category(categories.get(target, [])), set())
        intent, category, constraints = parse_opening(message)
        print(f"\n  {scenario}")
        print(f"    customer > {message}")
        print(f"    routed   > intent={intent!r}  category={category!r}")
        print(f"               constraints={constraints}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample", default=None, help="sample_id to replay")
    parser.add_argument("--scenario", default="intent_override",
                        choices=("buying", "browsing", "intent_override", "boundary"),
                        help="replay the first session of this type (if --sample is unset)")
    parser.add_argument("--show", type=int, default=3, help="recommendations to print per turn")
    parser.add_argument("--routing", action="store_true",
                        help="instead of replaying a session, show how the router classifies "
                             "one opener of each scenario type (Pillar I dual-track routing)")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)

    if args.routing:
        show_routing(samples, args.catalog)
        return
    if args.sample:
        chosen = next((s for s in samples if s["sample_id"] == args.sample), None)
        if chosen is None:
            raise SystemExit(f"no sample {args.sample!r} in {args.dataset}")
    else:
        chosen = next((s for s in samples if s["scenario_type"] == args.scenario), None)
        if chosen is None:
            raise SystemExit(f"no {args.scenario!r} session in {args.dataset}")

    catalog_ids, categories, products = catalog_index(args.catalog)
    from agent import Agent

    agent = Agent(args.catalog)

    target = str(chosen["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(chosen, products)
    effective = {**chosen, "intent_card": intent_card, "behavior": behavior}

    print(RULE)
    print(f"session {chosen['sample_id']}   scenario={chosen['scenario_type']}   "
          f"difficulty={chosen.get('difficulty_bucket', '?')}")
    print(f"hidden target  {target}  {title(products, target)}")
    print(f"hidden card    {intent_card}")
    print(RULE)

    agent.reset(chosen["sample_id"], chosen.get("user_profile", {}))
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = chosen["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"\nTURN {turn}")
        print(f"  customer > {message}")

        response = agent.respond(chosen["sample_id"], message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)

        print(f"  agent    > {response.get('message', '')}")
        print(f"             asks: {response.get('ask_attribute')!r}   "
              f"shows {len(ranked)} of top-{TOP_K}")
        for i, asin in enumerate(ranked[: args.show], 1):
            mark = "  <-- TARGET" if asin == target else ""
            print(f"             {i}. {asin}  {title(products, asin)}{mark}")

        if target in ranked and not override_applied:
            # Faithful to the scorer: in an override scenario the evaluator does
            # not count a hit until the override has fired, so the session keeps
            # going even though the target is on screen. Printed so that is not
            # mistaken for a bug.
            print("             (target shown, but the override hasn't fired yet —"
                  " the evaluator doesn't count a hit until it does)")

        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            print(f"\n{RULE}")
            print(f"CONVERTED on turn {turn} at rank {rank}   "
                  f"(RR = {1 / rank:.3f}, the evaluator ends the session here)")
            print(RULE)
            return

        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, ignore my earlier preference."))
            print(f"\n  ** INTENT OVERRIDE next turn: customer switches to {new_value!r} **")
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    print(f"\n{RULE}")
    print(f"no conversion within {MAX_TURNS} turns")
    print(RULE)


if __name__ == "__main__":
    main()
