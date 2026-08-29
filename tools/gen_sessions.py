"""Generate held-out evaluation sessions using the official evaluator's machinery.

The evaluator derives `intent_card` and `behavior` itself (see
`materialize_hidden_fields`) whenever a sample omits them. The released public
set omits them too -- its rows carry only sample_id, scenario_type,
user_profile, ground_truth, category_bucket, difficulty_bucket -- so a
generated row built the same way traverses byte-identical customer-simulation
code. That is what makes these numbers comparable to the public-set numbers,
and it is the only reason this script is allowed to exist.

This script therefore simulates no customer behaviour. It samples held-out
targets, reuses real profiles, and lets the frozen evaluator build the
customer.

Usage:
    python3 tools/gen_sessions.py \
        --catalog data/catalog.jsonl \
        --public data/public_set.jsonl \
        --out data/holdout_1000.jsonl \
        --count 1000 --seed 20260829

Then:
    python3 tools/run_eval.py --dataset data/holdout_1000.jsonl

Contamination rule: the moment you tune against a draw, that draw is training
data. Draw a fresh --seed for every held-out measurement and record which seed
produced which number.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

# Running `python3 tools/gen_sessions.py` puts tools/ on sys.path, not the repo
# root, so `import evaluator` fails without this. Mirrors tools/run_eval.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import intent_card, load_jsonl

# Fixed scenario mix from the competition specification (40/40/15/5).
# Verified against the released public set: 80/80/30/10 of 200.
SCENARIO_MIX = [
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
]

# Every released session carries category_bucket == "clothing". Nothing in
# evaluator/, pipeline/ or starter/ reads it; it is copied for schema parity so
# a generated file and the public file can be concatenated or diffed.
CATEGORY_BUCKET = "clothing"

# Number of popularity strata used to match the public targets' rating_number
# profile. The organizer sampled targets from the 5-core split, so real targets
# carry review history; a uniform catalog draw would not be comparable.
POPULARITY_BINS = 5

# If fewer than this fraction of catalog products expose a numeric
# rating_number, the stratification below is meaningless and silently collapses
# into a uniform draw from one bucket. Fail loudly instead.
MIN_POPULATED_RATING_FRACTION = 0.50


def load_catalog(path: Path) -> dict[str, dict]:
    products: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                product = json.loads(line)
                products[str(product["parent_asin"])] = product
    return products


def popularity(product: dict) -> float | None:
    """Review count, or None when the field is absent or unparseable.

    None is distinct from 0.0 on purpose: a product with no reviews and a
    product with no *field* look identical once both become 0.0, and that
    collapse is exactly what hides a broken stratification.
    """
    value = product.get("rating_number")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def constraint_count(product: dict) -> int:
    """How many distinct constraints beyond the bare title the customer can disclose.

    A target whose intent card collapses to just its title tells the agent
    nothing across ten turns. This is the honest stand-in for the organizer's
    `difficulty_bucket`, whose labelling rule is not published and cannot be
    reproduced.
    """
    card = intent_card(product)
    constraints = {*card["hard_constraints"], *card["soft_preferences"]}
    constraints.discard(card["target_category"])
    constraints.discard(str(product.get("title") or "").strip())
    return len(constraints)


def build_strata(values: list[float], bins: int) -> list[float]:
    """Return bin edges from the observed public-target popularity profile."""
    ordered = sorted(values)
    if not ordered:
        return []
    return [
        ordered[min(len(ordered) - 1, int(len(ordered) * i / bins))]
        for i in range(1, bins)
    ]


def stratum_of(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def draw_stratum(rng: random.Random, weights: list[float],
                 pools: dict[int, list[str]]) -> int | None:
    """Draw a stratum by the public profile, falling back to the nearest non-empty."""
    choice = rng.choices(range(POPULARITY_BINS), weights=weights)[0]
    if pools[choice]:
        return choice
    for offset in range(1, POPULARITY_BINS):
        for candidate in (choice - offset, choice + offset):
            if 0 <= candidate < POPULARITY_BINS and pools[candidate]:
                return candidate
    return None


def scenario_schedule(count: int) -> list[str]:
    """Deterministic mix -- exact proportions, not sampled."""
    schedule: list[str] = []
    for name, share in SCENARIO_MIX:
        schedule.extend([name] * round(count * share))
    while len(schedule) < count:
        schedule.append("browsing")
    return schedule[:count]


def check_popularity_is_usable(products: dict[str, dict],
                               public_values: list[float],
                               edges: list[float]) -> float:
    """Refuse to run when popularity matching would degrade to a uniform draw."""
    populated = sum(1 for product in products.values() if popularity(product) is not None)
    fraction = populated / max(1, len(products))
    if fraction < MIN_POPULATED_RATING_FRACTION:
        raise SystemExit(
            f"FATAL: only {fraction:.1%} of catalog products expose a numeric "
            f"'rating_number' (need >= {MIN_POPULATED_RATING_FRACTION:.0%}). "
            "Popularity stratification would silently collapse to a uniform "
            "single-bucket draw. Check the catalog is the real one."
        )
    if not public_values:
        raise SystemExit(
            "FATAL: no public target resolved to a catalog product with a "
            "numeric 'rating_number'. Is --catalog the right file?"
        )
    if len(set(edges)) < len(edges):
        raise SystemExit(
            f"FATAL: popularity bin edges are degenerate ({edges}). Public "
            "targets do not span enough distinct review counts for "
            f"{POPULARITY_BINS} strata; lower POPULARITY_BINS."
        )
    return fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out sessions")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--out", default="data/holdout_1000.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--prefix", default="holdout")
    parser.add_argument(
        "--profiles", default="empirical", choices=("empirical", "synthetic"),
        help="empirical resamples whole profiles from the public set (default)",
    )
    parser.add_argument(
        "--min-constraints", type=int, default=0,
        help="drop targets with fewer usable constraints. DEFAULT 0: the public "
             "set was never filtered this way, and any nonzero value makes the "
             "held-out set systematically easier than the set you tuned on.",
    )
    parser.add_argument(
        "--match", default="popularity", choices=("popularity", "none"),
        help="popularity matches the public targets' review-count profile "
             "(default); none draws uniformly from the catalog, which is far "
             "more obscure and NOT comparable to public-set numbers",
    )
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="previously generated session file whose targets to also exclude; "
             "repeatable. Use for disjoint draws.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    catalog = load_catalog(Path(args.catalog))
    public = load_jsonl(args.public)

    # Exclude every target the pipeline was tuned against, plus any prior draw.
    used = {str(row["ground_truth"]["parent_asin"]) for row in public}
    for path in args.exclude:
        used |= {str(row["ground_truth"]["parent_asin"]) for row in load_jsonl(path)}

    public_popularity = [
        value for asin in used
        if asin in catalog and (value := popularity(catalog[asin])) is not None
    ]
    edges = build_strata(public_popularity, POPULARITY_BINS)
    populated_fraction = check_popularity_is_usable(catalog, public_popularity, edges)

    # The comparability check that matters: the public set applied no
    # constraint filter, so its own difficulty profile is the yardstick.
    public_constraints = [
        constraint_count(catalog[str(row["ground_truth"]["parent_asin"])])
        for row in public
        if str(row["ground_truth"]["parent_asin"]) in catalog
    ]
    public_degenerate = sum(1 for value in public_constraints if value < 2)

    if args.min_constraints > 0:
        print(
            f"WARNING: --min-constraints {args.min_constraints} filters the "
            f"held-out set, but {public_degenerate}/{len(public_constraints)} "
            "public targets would fail the same filter. Held-out scores will "
            "be optimistic unless you apply the filter to both sides.",
            file=sys.stderr,
        )

    # Bucket held-out candidates by the same popularity strata.
    pools: dict[int, list[str]] = {index: [] for index in range(POPULARITY_BINS)}
    counts_by_asin: dict[str, int] = {}
    skipped_degenerate = 0
    skipped_no_rating = 0
    for asin, product in catalog.items():
        if asin in used:
            continue
        value = popularity(product)
        if value is None:
            skipped_no_rating += 1
            continue
        count = constraint_count(product)
        if count < args.min_constraints:
            skipped_degenerate += 1
            continue
        counts_by_asin[asin] = count
        pools[stratum_of(value, edges)].append(asin)
    for pool in pools.values():
        pool.sort()
        rng.shuffle(pool)

    # Match the public targets' stratum proportions.
    public_strata = [stratum_of(value, edges) for value in public_popularity]
    weights = [
        public_strata.count(index) / max(1, len(public_strata))
        for index in range(POPULARITY_BINS)
    ]

    # The catalog cannot always supply a popularity-matched draw of arbitrary
    # size. Public targets come from the 5-core split and are far more popular
    # than the catalog at large (catalog median rating_number ~12; public target
    # median ~6846), so the top strata are tiny. The largest N for which every
    # stratum can still meet its quota is the real ceiling.
    feasible_count = min(
        (int(len(pools[index]) / weights[index]) for index in range(POPULARITY_BINS)
         if weights[index] > 0),
        default=0,
    )
    if args.match == "popularity" and args.count > feasible_count:
        print(
            f"WARNING: --count {args.count} exceeds the popularity-matched "
            f"ceiling of {feasible_count}. Scarce strata will empty and their "
            "draws will spill into denser, less popular strata, producing a "
            "held-out set materially more obscure than the public set. Use "
            f"--count {feasible_count}, or --match none to draw without matching.",
            file=sys.stderr,
        )

    profile_pool = [row["user_profile"] for row in public]

    if args.match == "popularity":
        draw_weights = weights
    else:
        draw_weights = [len(pools[index]) for index in range(POPULARITY_BINS)]

    schedule = scenario_schedule(args.count)
    rng.shuffle(schedule)

    rows: list[dict] = []
    exhausted = 0
    for position, scenario in enumerate(schedule, start=1):
        stratum = draw_stratum(rng, draw_weights, pools)
        if stratum is None:
            exhausted += 1
            continue
        asin = pools[stratum].pop()
        if args.profiles == "empirical":
            # Copy a whole real profile: preserves tag co-occurrence and the
            # rating/rating_style/summary agreement that per-field sampling breaks.
            profile = dict(rng.choice(profile_pool))
        else:
            profile = synthetic_profile(rng)
        rows.append({
            "sample_id": f"{args.prefix}_{position:05d}",
            "scenario_type": scenario,
            "category_bucket": CATEGORY_BUCKET,
            "user_profile": profile,
            "ground_truth": {"parent_asin": asin},
            # Extra keys are ignored by evaluate(); kept so held-out results can
            # be sliced by difficulty after the fact instead of pre-filtered.
            "constraint_count": counts_by_asin[asin],
            "rating_number": popularity(catalog[asin]),
        })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    scenario_counts: dict[str, int] = {}
    for row in rows:
        scenario_counts[row["scenario_type"]] = scenario_counts.get(row["scenario_type"], 0) + 1
    generated_popularity = [float(row["rating_number"]) for row in rows]
    generated_constraints = [row["constraint_count"] for row in rows]

    realized = [0] * POPULARITY_BINS
    for value in generated_popularity:
        realized[stratum_of(value, edges)] += 1
    realized_shares = [count / max(1, len(rows)) for count in realized]
    max_deviation = max(
        abs(realized_shares[index] - weights[index]) for index in range(POPULARITY_BINS)
    )

    meta = {
        "written": len(rows),
        "output": str(output),
        "seed": args.seed,
        "profiles": args.profiles,
        "min_constraints": args.min_constraints,
        "scenario_counts": scenario_counts,
        "catalog_products": len(catalog),
        "catalog_rating_number_populated": round(populated_fraction, 4),
        "excluded_targets": len(used),
        "skipped_no_rating": skipped_no_rating,
        "skipped_below_min_constraints": skipped_degenerate,
        "exhausted_strata_draws": exhausted,
        "popularity_edges": edges,
        "match_mode": args.match,
        "popularity_matched_ceiling": feasible_count,
        "strata_target_shares": [round(value, 4) for value in weights],
        "strata_realized_shares": [round(value, 4) for value in realized_shares],
        "strata_max_deviation": round(max_deviation, 4),
        "popularity_match_ok": bool(args.match == "popularity" and max_deviation <= 0.02),
        "median_rating_number_public": statistics.median(public_popularity),
        "median_rating_number_generated": statistics.median(generated_popularity or [0]),
        "mean_constraints_public": round(statistics.fmean(public_constraints or [0]), 3),
        "mean_constraints_generated": round(statistics.fmean(generated_constraints or [0]), 3),
        "degenerate_share_public": round(public_degenerate / max(1, len(public_constraints)), 4),
        "degenerate_share_generated": round(
            sum(1 for value in generated_constraints if value < 2) / max(1, len(rows)), 4
        ),
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


# Retained for --profiles synthetic. Hand-fitted marginals; the empirical path
# is preferred because these drift from the real joint distribution.
PREFERENCE_TAGS = [
    "fit", "comfort", "material", "style", "durability",
    "performance", "warmth", "weather",
]
RATING_STYLE = {5.0: "usually positive", 4.0: "mixed", 3.0: "critical",
                2.0: "critical", 1.0: "critical"}
RATING_WEIGHTS = [(5.0, 0.67), (4.0, 0.105), (3.0, 0.11), (2.0, 0.045), (1.0, 0.07)]
TAG_COUNT_WEIGHTS = [(1, 0.03), (2, 0.215), (3, 0.15), (4, 0.605)]


def synthetic_profile(rng: random.Random) -> dict:
    rating = rng.choices(
        [value for value, _ in RATING_WEIGHTS],
        weights=[weight for _, weight in RATING_WEIGHTS],
    )[0]
    size = rng.choices(
        [value for value, _ in TAG_COUNT_WEIGHTS],
        weights=[weight for _, weight in TAG_COUNT_WEIGHTS],
    )[0]
    tags = rng.sample(PREFERENCE_TAGS, size)
    style = RATING_STYLE[rating]
    return {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": rating,
        "rating_style": style,
        "preference_tags": tags,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {style}.",
    }


if __name__ == "__main__":
    main()
