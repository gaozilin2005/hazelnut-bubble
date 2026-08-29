"""Measure how much of the score survives if the organizer paraphrases messages.

The spec warns: "If natural-language paraphrasing is added by the organizer, it
cannot decide correctness." That means the private split's wording may not match
the templates pipeline/router.py parses. This wraps the evaluator's customer
policy (without editing it) and re-renders each message at increasing paraphrase
strength, then rescores.

    L0  templates verbatim (what the public evaluator emits today)
    L1  scaffolding reworded, constraint payload still verbatim
    L2  scaffolding reworded, payload lightly perturbed (case/punctuation/filler)
    L3  scaffolding reworded, payload heavily rewritten (containment destroyed)
    L4  scaffolding reworded, payload verbatim, but the CATEGORY NAME is also
        reworded -- probes the single point of failure, since the category
        bucket filter is doing most of the work

L1 is the realistic case: an organizer paraphrasing prose would rarely alter a
quoted product attribute. L3 is the pessimistic bound.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluator.local_evaluator as harness
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from pipeline.interfaces import INTENT_BROWSING, INTENT_BUYING, INTENT_OVERRIDE
from pipeline.router import parse_opening, parse_reply

OPENINGS = {
    INTENT_BUYING: [
        "I want to find {cat}. One thing I really need: {payload}.",
        "Hi! Shopping for {cat} today. It has to be {payload}.",
        "Do you carry {cat}? Important for me: {payload}.",
    ],
    INTENT_BROWSING: [
        "Just browsing {cat} for now, nothing specific in mind.",
        "I'm curious about {cat}. Haven't decided on anything yet.",
        "Show me some {cat}? Still figuring out what I want.",
    ],
    INTENT_OVERRIDE: [
        "Hi, I'm after {cat}. {payload}",
        "Shopping for {cat} today. {payload}",
    ],
}
DISCLOSE = [
    "What I care about is {payload}.",
    "Mainly this: {payload}.",
    "Here's the thing that matters -- {payload}.",
]
OVERRIDE = [
    "Scratch that, actually. What I really want is {payload}.",
    "Change of plan -- forget what I said. I need {payload}.",
]
NO_SIGNAL = [
    "No strong feelings there, honestly.",
    "That one's not important to me.",
    "Whatever you think is best on that.",
]
FILLER = ("really ", "quite ", "pretty much ")


def degrade_category(category: str, rng: random.Random) -> str:
    """Shorten a taxonomy string to how a person would actually say it."""
    words = [word for word in re.split(r"[^A-Za-z0-9]+", category) if word]
    if len(words) <= 1:
        return category.lower()
    keep = max(1, len(words) // 2)
    return " ".join(words[-keep:]).lower()


def perturb(payload: str, level: int, rng: random.Random) -> str:
    if level == 4:
        return payload
    if level <= 1:
        return payload
    if level == 2:
        text = re.sub(r"[,:;]", "", payload).lower()
        if rng.random() < 0.5:
            words = text.split()
            if len(words) > 3:
                position = rng.randrange(1, len(words))
                words.insert(position, rng.choice(FILLER).strip())
                text = " ".join(words)
        return text
    words = re.sub(r"[,:;]", " ", payload).lower().split()
    if len(words) <= 2:
        return " ".join(reversed(words))
    keep = max(2, int(len(words) * 0.6))
    chosen = sorted(rng.sample(range(len(words)), keep))
    kept = [words[i] for i in chosen]
    rng.shuffle(kept)
    return " ".join(kept)


# Captured once, at import, before anything is patched. install() must always
# wrap these -- wrapping whatever is currently installed compounds the levels,
# and a level-N wrapper cannot parse level-(N-1) output, so replies silently
# collapse to "no preference" and the run measures starvation, not paraphrase.
PRISTINE_INITIAL = harness.initial_message
PRISTINE_REPLY = harness.customer_reply


def install(level: int, seed: int = 11) -> None:
    """Monkeypatch the customer policy. The evaluator file itself is untouched."""
    original_initial = PRISTINE_INITIAL
    original_reply = PRISTINE_REPLY
    rng = random.Random(seed)

    def initial_message(sample, category, disclosed):
        text = original_initial(sample, category, disclosed)
        if level == 0:
            return text
        intent, parsed_category, constraints = parse_opening(text)
        if parsed_category is None or intent not in OPENINGS:
            return text
        payload = perturb("; ".join(constraints), level, rng) if constraints else ""
        if level == 4:
            parsed_category = degrade_category(parsed_category, rng)
        return rng.choice(OPENINGS[intent]).format(cat=parsed_category, payload=payload)

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        text, boundary = original_reply(sample, ask_attribute, disclosed, boundary_used)
        if level == 0:
            return text, boundary
        is_override, constraints = parse_reply(text)
        if not constraints:
            return rng.choice(NO_SIGNAL), boundary
        payload = perturb("; ".join(constraints), level, rng)
        pool = OVERRIDE if is_override else DISCLOSE
        return rng.choice(pool).format(payload=payload), boundary

    harness.initial_message = initial_message
    harness.customer_reply = customer_reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="pipeline", choices=("pipeline", "baseline"))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--levels", default="0,1,2,3,4")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    if args.agent == "pipeline":
        from pipeline.agent import PipelineAgent
        agent = PipelineAgent(args.catalog)
    else:
        from starter.agent import Agent
        agent = Agent(args.catalog)

    print(f"{'level':<7}{'Hit@10':>9}{'MRR':>9}{'MTTC':>8}{'Score':>9}")
    for level in [int(v) for v in args.levels.split(",")]:
        install(level)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        print(f"L{level:<6}{result['hit_rate_at_10']:>9.3f}{result['mrr']:>9.3f}"
              f"{result['mttc']:>8.2f}{result['recommended_technical_score']:>9.3f}")


if __name__ == "__main__":
    main()
