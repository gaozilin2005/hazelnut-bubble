"""Build a DEVELOPMENT PROXY catalog from the upstream Amazon Reviews 2023 metadata.

This is NOT the organizer's frozen catalog.jsonl. It is a stand-in sampled from the
same upstream universe so the pipeline and evaluator can run before the official
release is available. Scores produced against it are not comparable to official
scores -- the distractor sample differs, and distractor quality is what sets
Hit@10. Replace with the real data/catalog.jsonl as soon as you have it.

Usage:
    wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/\
meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz -P data/releases/
    python3 tools/build_dev_catalog.py data/releases/meta_Clothing_Shoes_and_Jewelry.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import zlib
from pathlib import Path

# Participant-visible fields, per docs/competition_specification.md.
VISIBLE_FIELDS = (
    "parent_asin", "title", "features", "description", "price",
    "categories", "details", "average_rating", "rating_number", "store",
)
CATALOG_SIZE = 50_000
# Approximates the organizer's "Clothing 5-core" split. Sampling without this
# floor pulls long-tail items with no reviews, which are weak distractors and
# would inflate local Hit@10.
MIN_RATINGS = 5
STRATA = 10


def target_asins(dataset: Path) -> set[str]:
    with dataset.open(encoding="utf-8") as handle:
        return {
            str(json.loads(line)["ground_truth"]["parent_asin"])
            for line in handle if line.strip()
        }


def trim(product: dict) -> dict:
    return {field: product.get(field) for field in VISIBLE_FIELDS}


def popularity_strata(strata_path: Path, targets: set[str]) -> list[float]:
    """Quantile edges of the TARGETS' review counts.

    A uniform sample of "rating_number >= 5" items is ~236x less reviewed than
    the targets, which turns a popularity prior into a near-oracle and inflates
    every metric. Sampling distractors to match this distribution removes the
    giveaway. The organizer's catalog is drawn from the 5-core split, so its
    distractors are uniformly well-reviewed -- this approximates that.
    """
    counts: list[float] = []
    with strata_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            if str(product.get("parent_asin")) in targets:
                counts.append(float(product.get("rating_number") or 0))
    if not counts:
        return []
    counts.sort()
    return [counts[min(len(counts) - 1, int(len(counts) * i / STRATA))] for i in range(1, STRATA)]


def stratum_of(count: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if count < edge:
            return index
    return len(edges)


def build(
    meta_path: Path, dataset: Path, output: Path, size: int, seed: int,
    strata_from: Path | None = None,
) -> None:
    targets = target_asins(dataset)
    rng = random.Random(seed)
    found: dict[str, dict] = {}
    pool_size = size - len(targets)
    edges = popularity_strata(strata_from, targets) if strata_from else []
    # One reservoir per stratum (plus a spare for backfill when a stratum is
    # too rare upstream to fill its quota).
    bins = len(edges) + 1 if edges else 1
    quota = [pool_size // bins] * bins
    pools: list[list[dict]] = [[] for _ in range(bins)]
    seen = [0] * bins
    spare: list[dict] = []
    spare_seen = 0
    scanned = eligible = 0

    try:
        handle = gzip.open(meta_path, "rt", encoding="utf-8")
        stream = iter(handle)
        while True:
            try:
                line = next(stream)
            except StopIteration:
                break
            except (EOFError, zlib.error) as error:
                raise SystemExit(
                    f"\n{meta_path} is truncated or corrupt ({type(error).__name__}).\n"
                    "Resume the download with 'wget -c' and re-run; a partial scan would\n"
                    "silently drop targets and produce a catalog you cannot trust."
                ) from error
            try:
                product = json.loads(line)
            except json.JSONDecodeError:
                continue
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  scanned {scanned:,} | targets {len(found)}/{len(targets)}", file=sys.stderr)
            parent_asin = product.get("parent_asin")
            if not parent_asin:
                continue
            parent_asin = str(parent_asin)
            if parent_asin in targets:
                found.setdefault(parent_asin, trim(product))
                continue
            if not product.get("categories") or (product.get("rating_number") or 0) < MIN_RATINGS:
                continue
            eligible += 1
            row = trim(product)
            # Reservoir sample per stratum: one pass, no full materialization,
            # deterministic under seed.
            bucket = stratum_of(float(product.get("rating_number") or 0), edges) if edges else 0
            seen[bucket] += 1
            if len(pools[bucket]) < quota[bucket]:
                pools[bucket].append(row)
            else:
                index = rng.randrange(seen[bucket])
                if index < quota[bucket]:
                    pools[bucket][index] = row
            spare_seen += 1
            if len(spare) < pool_size:
                spare.append(row)
            else:
                index = rng.randrange(spare_seen)
                if index < pool_size:
                    spare[index] = row
    finally:
        handle.close()

    missing = targets - found.keys()
    pool = [row for bucket in pools for row in bucket]
    if len(pool) < pool_size:
        # A rare stratum could not fill its quota; backfill to keep the catalog
        # at full size so distractor density stays comparable.
        have = {row["parent_asin"] for row in pool}
        for row in spare:
            if len(pool) >= pool_size:
                break
            if row["parent_asin"] not in have:
                pool.append(row)
                have.add(row["parent_asin"])
    rows = list(found.values()) + pool[:pool_size]
    rng.shuffle(rows)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nscanned   {scanned:,} upstream items")
    print(f"eligible  {eligible:,} (categories set, rating_number >= {MIN_RATINGS})")
    print(f"targets   {len(found)}/{len(targets)} located")
    if edges:
        print(f"strata    {bins} popularity bins, edges {[int(e) for e in edges]}")
    print(f"wrote     {len(rows):,} rows -> {output}")
    if missing:
        print(f"\nWARNING: {len(missing)} target(s) absent from upstream metadata.")
        print("The evaluator raises KeyError on a missing target; those sessions cannot run.")
        print("Sample:", sorted(missing)[:5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meta", type=Path, help="meta_Clothing_Shoes_and_Jewelry.jsonl.gz")
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--size", type=int, default=CATALOG_SIZE)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--strata-from", type=Path, default=None,
        help="jsonl containing the targets with rating_number; distractors are "
             "sampled to match the targets' popularity distribution",
    )
    args = parser.parse_args()
    build(args.meta, args.dataset, args.output, args.size, args.seed, args.strata_from)


if __name__ == "__main__":
    main()
