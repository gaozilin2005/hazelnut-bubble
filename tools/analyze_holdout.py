"""Slice an eval result by the per-row properties gen_sessions.py recorded.

tools/run_eval.py reports metrics overall and by scenario. It cannot report by
popularity or difficulty, because those live on the session file, not the
result file. This joins the two on sample_id.

Usage:
    python3 tools/analyze_holdout.py \
        --dataset data/holdout_broad_1000.jsonl \
        --results results_pipeline_broad.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def technical_score(sessions: list[dict]) -> tuple[float, float, float, float]:
    hit_rate = sum(int(item["hit"]) for item in sessions) / len(sessions)
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else 11
        for item in sessions
    )
    efficiency = max(0.0, min(1.0, (11 - mttc) / 10))
    return hit_rate, mrr, mttc, 0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency


def report(title: str, groups: list[tuple[str, list[dict]]], extra: str = "") -> None:
    print(f"\n{title}")
    print(f"{'':<14}{'n':>6}{extra:>14}{'Hit@10':>9}{'MRR':>8}{'MTTC':>7}{'Score':>9}")
    for label, sessions in groups:
        if not sessions:
            continue
        hit_rate, mrr, mttc, score = technical_score(sessions)
        note = sessions[0].get("_note", "")
        print(f"{label:<14}{len(sessions):>6}{note:>14}"
              f"{hit_rate:>9.3f}{mrr:>8.3f}{mttc:>7.2f}{score:>9.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="generated .jsonl")
    parser.add_argument("--results", required=True, help="run_eval.py output")
    parser.add_argument("--bins", type=int, default=5, help="popularity quantiles")
    args = parser.parse_args()

    rows = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()
    }
    sessions = json.loads(Path(args.results).read_text(encoding="utf-8"))["sessions"]

    missing = [s["sample_id"] for s in sessions if s["sample_id"] not in rows]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} result rows are absent from --dataset "
            f"(first: {missing[0]}). The result file and session file do not match."
        )
    if "rating_number" not in next(iter(rows.values())):
        raise SystemExit(
            "FATAL: --dataset has no per-row 'rating_number'. Only files written "
            "by tools/gen_sessions.py carry it; the released public set does not."
        )

    for session in sessions:
        row = rows[session["sample_id"]]
        session["popularity"] = float(row["rating_number"])
        session["constraints"] = row["constraint_count"]

    hit_rate, mrr, mttc, score = technical_score(sessions)
    print(f"overall  n={len(sessions)}  Hit@10={hit_rate:.4f}  MRR={mrr:.4f}  "
          f"MTTC={mttc:.3f}  Score={score:.4f}")

    ordered = sorted(sessions, key=lambda s: s["popularity"])
    total = len(ordered)
    quantiles = []
    for index in range(args.bins):
        part = ordered[index * total // args.bins:(index + 1) * total // args.bins]
        if part:
            median = statistics.median(s["popularity"] for s in part)
            for session in part:
                session["_note"] = f"med {median:.0f} rev"
        quantiles.append((f"Q{index + 1}", part))
    report("by popularity", quantiles, extra="median")
    # The median note belongs to the popularity view only; leaving it set would
    # label the groups below with an unrelated quantile's median.
    for session in sessions:
        session.pop("_note", None)

    by_constraints = defaultdict(list)
    for session in sessions:
        by_constraints[session["constraints"]].append(session)
    report("by constraint count", [
        (str(key), by_constraints[key]) for key in sorted(by_constraints)
    ])

    by_scenario = defaultdict(list)
    for session in sessions:
        by_scenario[session["scenario_type"]].append(session)
    report("by scenario", [
        (key, by_scenario[key]) for key in sorted(by_scenario)
    ])


if __name__ == "__main__":
    main()
