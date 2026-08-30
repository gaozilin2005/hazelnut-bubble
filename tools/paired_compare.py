"""Paired per-session comparison of two run_eval result files.

The single-draw noise floor is +/-0.007 (README), so an aggregate delta smaller
than ~0.015 from two *different* draws is indistinguishable from sampling
variation. Comparing the SAME sessions under two configurations removes the
draw variance entirely: every session is its own control, and the sign test on
per-session score deltas is exact.

    python3 tools/paired_compare.py results_a.json results_b.json

Per-session score uses the evaluator's own composite weights on the session's
contributions. The evaluator computes efficiency from mean MTTC as
(11 - mttc)/10 with a missed session counted as turn 11, which is linear in the
per-session turn, so the mean of these per-session scores IS the composite
(before the evaluator's clamp of efficiency into [0, 1], which never binds in
practice: mttc <= 11 by construction).
"""
from __future__ import annotations

import json
import math
import sys

WEIGHTS = (0.50, 0.30, 0.20)
MAX_TURNS = 10


def session_score(row: dict) -> float:
    hit = float(row["hit"])
    rr = float(row["reciprocal_rank"])
    turn = row["first_hit_turn"] if row["first_hit_turn"] is not None else MAX_TURNS + 1
    efficiency = (11.0 - float(turn)) / 10.0
    return WEIGHTS[0] * hit + WEIGHTS[1] * rr + WEIGHTS[2] * efficiency


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    total = sum(math.comb(n, i) for i in range(k + 1)) * 2
    return min(1.0, total / 2**n)


def main() -> None:
    path_a, path_b = sys.argv[1], sys.argv[2]
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    rows_a = {row["sample_id"]: row for row in a["sessions"]}
    rows_b = {row["sample_id"]: row for row in b["sessions"]}
    shared = sorted(set(rows_a) & set(rows_b))
    if len(shared) != len(rows_a) or len(shared) != len(rows_b):
        print(f"WARNING: only {len(shared)} shared sessions "
              f"({len(rows_a)} in A, {len(rows_b)} in B)")

    deltas = []
    wins = losses = 0
    moved: list[tuple[float, str, dict, dict]] = []
    for sid in shared:
        delta = session_score(rows_b[sid]) - session_score(rows_a[sid])
        deltas.append(delta)
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        if abs(delta) > 1e-12:
            moved.append((delta, sid, rows_a[sid], rows_b[sid]))

    n = len(shared)
    mean = sum(deltas) / max(1, n)
    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print(f"sessions: {n}   B-A mean score delta: {mean:+.6f}")
    print(f"B wins: {wins}   B losses: {losses}   ties: {n - wins - losses}")
    print(f"exact sign test (two-sided): p = {sign_test_p(wins, losses):.4f}")
    for key in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            print(f"  {key:<28} {va:>9.6f} -> {vb:>9.6f}  ({vb - va:+.6f})")
    moved.sort(key=lambda item: item[0])
    if moved:
        print(f"\nlargest movers (of {len(moved)} changed sessions):")
        for delta, sid, ra, rb in moved[:5] + moved[-5:][::-1]:
            print(f"  {delta:+.4f}  {sid} ({ra.get('scenario_type')})  "
                  f"rank {ra.get('best_rank')} -> {rb.get('best_rank')}  "
                  f"turn {ra.get('first_hit_turn')} -> {rb.get('first_hit_turn')}")


if __name__ == "__main__":
    main()
