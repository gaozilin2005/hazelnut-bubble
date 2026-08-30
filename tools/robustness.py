"""Measure how much of the score survives if the organizer paraphrases messages.

The spec warns: "If natural-language paraphrasing is added by the organizer, it
cannot decide correctness." That means the private split's wording may not match
the templates pipeline/router.py parses. This wraps the evaluator's customer
policy (without editing it) and re-renders each message at increasing paraphrase
strength, then rescores.

These are TWO AXES, not one ladder. The pipeline has two independent signals --
verbatim phrase containment, and the coarse-category bucket filter -- and the
levels attack them separately:

    PAYLOAD AXIS (containment), category left intact:
      L0  templates verbatim (what the public evaluator emits today)
      L1  scaffolding reworded, constraint payload still verbatim
      L2  scaffolding reworded, payload lightly perturbed (case/punctuation/filler)
      L3  scaffolding reworded, payload heavily rewritten (containment destroyed)

    CATEGORY AXIS, payload left verbatim:
      L4  scaffolding reworded, payload VERBATIM, category name reworded

    BOTH:
      L5  payload rewritten AND category reworded -- the true pessimistic bound

Only L0..L3 form a monotone ladder, and `check_monotone` asserts that they do.
L4 is NOT "worse than L3": it keeps containment and removes the category filter,
so it usually scores HIGHER. Reading L4 > L3 as an inversion is a category error
that has cost this project time twice; it is why the axes are now labelled.

L1 is the realistic case: an organizer paraphrasing prose would rarely alter a
quoted product attribute. L5 is the pessimistic bound.
"""
from __future__ import annotations

import argparse
import hashlib
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


# Which signal each level attacks. Used for reporting and for the monotonicity
# check, so the two axes can never be silently compared against each other.
PAYLOAD_AXIS = (0, 1, 2, 3)
DEGRADES_CATEGORY = (4, 5)
AXIS_LABEL = {
    0: "payload verbatim, category intact",
    1: "payload verbatim, prose reworded",
    2: "payload lightly perturbed",
    3: "payload rewritten (containment destroyed)",
    4: "category reworded, payload verbatim",
    5: "payload rewritten AND category reworded",
}


def check_monotone(scores: dict[int, float]) -> list[str]:
    """Scores must not increase along the payload axis. Anything else is a bug.

    Deliberately scoped to L0..L3. L4 and L5 degrade a different signal, so
    comparing them to L3 is meaningless -- that comparison is what produced the
    phantom "L3/L4 inversion".
    """
    ladder = [level for level in PAYLOAD_AXIS if level in scores]
    return [
        f"L{a} ({scores[a]:.4f}) < L{b} ({scores[b]:.4f}) -- adding degradation "
        f"improved the score, which compounding perturbations cannot do"
        for a, b in zip(ladder, ladder[1:]) if scores[a] < scores[b] - 1e-9
    ]


def perturb(payload: str, level: int, rng: random.Random) -> str:
    if level == 4:
        return payload
    if level <= 1:
        return payload
    if level == 5:
        level = 3
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


def _message_rng(seed: int, level: int, text: str) -> random.Random:
    """Deterministic rng PER MESSAGE, derived from the template text itself.

    A single shared stream would make the realized paraphrase of session N
    depend on how many messages sessions 1..N-1 rendered -- i.e. on the
    AGENT'S behavior. Two agent configurations would then be scored against
    different paraphrases, and their delta would mix the code change with
    resampling noise. Hashing the pristine template text instead means the
    same session always gets the same paraphrase at a given level, whatever
    ran before it: cross-config comparisons are paired by construction.
    """
    digest = hashlib.sha256(f"{seed}:{level}:{text}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def install(level: int, seed: int = 11) -> None:
    """Monkeypatch the customer policy. The evaluator file itself is untouched."""
    original_initial = PRISTINE_INITIAL
    original_reply = PRISTINE_REPLY

    def initial_message(sample, category, disclosed):
        text = original_initial(sample, category, disclosed)
        if level == 0:
            return text
        intent, parsed_category, constraints = parse_opening(text)
        if parsed_category is None or intent not in OPENINGS:
            return text
        rng = _message_rng(seed, level, text)
        payload = perturb("; ".join(constraints), level, rng) if constraints else ""
        if level in DEGRADES_CATEGORY:
            parsed_category = degrade_category(parsed_category, rng)
        return rng.choice(OPENINGS[intent]).format(cat=parsed_category, payload=payload)

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        text, boundary = original_reply(sample, ask_attribute, disclosed, boundary_used)
        if level == 0:
            return text, boundary
        is_override, constraints = parse_reply(text)
        rng = _message_rng(seed, level, text)
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
    parser.add_argument("--levels", default="0,1,2,3,4,5")
    parser.add_argument("--no-dense", action="store_true",
                        help="disable the dense recall route (ablation)")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    if args.agent == "pipeline":
        from pipeline.agent import PipelineAgent
        agent = PipelineAgent(args.catalog, use_dense=not args.no_dense)
    else:
        from starter.agent import Agent
        agent = Agent(args.catalog)

    levels = [int(v) for v in args.levels.split(",")]
    print(f"{'level':<7}{'Hit@10':>9}{'MRR':>9}{'MTTC':>8}{'Score':>9}   what it degrades")
    scores: dict[int, float] = {}
    for level in levels:
        if level in DEGRADES_CATEGORY and PAYLOAD_AXIS[-1] in scores:
            print("       " + "-" * 44 + "   (different axis below)")
        install(level)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        scores[level] = result["recommended_technical_score"]
        print(f"L{level:<6}{result['hit_rate_at_10']:>9.3f}{result['mrr']:>9.3f}"
              f"{result['mttc']:>8.2f}{result['recommended_technical_score']:>9.3f}"
              f"   {AXIS_LABEL[level]}")

    violations = check_monotone(scores)
    print()
    if violations:
        print("FAIL: payload axis is not monotone --")
        for line in violations:
            print(f"   {line}")
        raise SystemExit(1)
    print("payload axis L0..L3 is monotone (as it must be).")
    if 4 in scores and 3 in scores:
        print(f"L4 ({scores[4]:.3f}) vs L3 ({scores[3]:.3f}): different axes, not comparable. "
              "L4 keeps containment and drops the category filter.")


if __name__ == "__main__":
    main()
