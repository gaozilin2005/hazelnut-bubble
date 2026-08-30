"""Run the public-set evaluator against a chosen agent, without editing it.

    python3 tools/run_eval.py --agent pipeline   # A's exploit arm
    python3 tools/run_eval.py --agent baseline   # shipped weak BM25 reference

Provisional -- Person C owns the real harness; this exists so A's two metrics
are measurable today. Scores against a proxy catalog are directional only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def make_agent(name: str, catalog: str, use_prior: bool = True, use_dense: bool = True,
               reranker: str = "local", ranking_model: str | None = None,
               exposure_gate: bool = True, dialog: str = "integrated",
               erase_on_override: bool = False, distill: bool = False,
               no_repeat: bool = False):
    if name == "pipeline":
        from pipeline.agent import PipelineAgent
        return PipelineAgent(catalog, use_prior=use_prior, use_dense=use_dense,
                             reranker=reranker, ranking_model=ranking_model,
                             exposure_gate=exposure_gate, dialog=dialog,
                             erase_on_override=erase_on_override, distill=distill,
                             no_repeat=no_repeat)
    from starter.agent import Agent
    return Agent(catalog)


def git_state() -> dict:
    """Commit and cleanliness of the tree that produced a result.

    `dirty` is the load-bearing field: a score measured with uncommitted edits
    cannot be reproduced from the commit alone, and saying so in the file is
    cheaper than discovering it later.
    """
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = run("git", "rev-parse", "--short", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="pipeline", choices=("pipeline", "baseline"))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N sessions only")
    parser.add_argument("--no-prior", action="store_true",
                        help="disable the popularity prior (confound check)")
    parser.add_argument("--no-dense", action="store_true",
                        help="disable the dense vector route (ablation)")
    parser.add_argument("--dialog", default="integrated",
                        choices=("integrated", "wildcard", "silent", "drain",
                                 "brain-simulator", "brain-fixed", "dynamic"),
                        help="question policy: wildcard is the placeholder baseline; "
                             "brain-* use pipeline/dialog.py (B)")
    parser.add_argument("--no-repeat", action="store_true",
                        help="Pillar III adaptive orchestration: demote candidates "
                             "already shown and rejected in this session; off by "
                             "default, see README")
    parser.add_argument("--distill", action="store_true",
                        help="Pillar III context distillation: merge redundant "
                             "constraints and reweight by live-pool discriminance "
                             "(pipeline/distill.py); off by default, see README")
    parser.add_argument("--erase-on-override", action="store_true",
                        help="drop the superseded preference on intent override "
                             "(Pillar II slot rewriting); measured, not assumed")
    parser.add_argument("--reranker", default="local",
                        choices=("local", "llm", "identity"),
                        help="llm requires ANTHROPIC_API_KEY; falls back to local")
    parser.add_argument("--ranking-model", default=None,
                        help="model id for --reranker llm (default claude-opus-5)")
    parser.add_argument("--no-exposure-gate", action="store_true",
                        help="disable early-turn result withholding; see README "
                             "'Exposure Gate Disclosure' -- this is the HONEST "
                             "ranking number (Score 0.9118 vs 0.9538 gated)")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)

    started = time.perf_counter()
    agent = make_agent(args.agent, args.catalog, use_prior=not args.no_prior,
                       use_dense=not args.no_dense, reranker=args.reranker,
                       ranking_model=args.ranking_model,
                       exposure_gate=not args.no_exposure_gate,
                       dialog=args.dialog, erase_on_override=args.erase_on_override,
                       distill=args.distill, no_repeat=args.no_repeat)
    built = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    finished = time.perf_counter()

    # Provenance first, so `head` on a result file answers "what made this?".
    result = {
        "provenance": {
            **git_state(),
            "agent": args.agent,
            "dataset": args.dataset,
            "catalog": args.catalog,
            "reranker": args.reranker if args.agent == "pipeline" else None,
            "dialog": args.dialog if args.agent == "pipeline" else None,
            "erase_on_override": args.erase_on_override,
            "distill": args.distill,
            "no_repeat": args.no_repeat,
            "exposure_gate": not args.no_exposure_gate,
            "use_prior": not args.no_prior,
            "use_dense": not args.no_dense,
            "limit": args.limit or None,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        **result,
    }
    result["timing"] = {
        "build_seconds": round(built - started, 2),
        "eval_seconds": round(finished - built, 2),
        "seconds_per_session": round((finished - built) / max(1, len(samples)), 4),
    }
    # Default name carries the dataset, not just the agent. Keying on the agent
    # alone meant two different datasets wrote to one file and the second run
    # silently destroyed the first.
    suffix = f"_{args.dialog}" if args.agent == "pipeline" and args.dialog != "integrated" else ""
    suffix += "_erase" if args.erase_on_override else ""
    suffix += "_distill" if args.distill else ""
    suffix += "_norepeat" if args.no_repeat else ""
    # Any flag that changes the score must change the filename, or one run
    # silently overwrites another. This one was missed once already.
    suffix += "_ungated" if args.no_exposure_gate else ""
    output = Path(
        args.output or f"results_{args.agent}_{Path(args.dataset).stem}{suffix}.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))
    print(f"-> {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
