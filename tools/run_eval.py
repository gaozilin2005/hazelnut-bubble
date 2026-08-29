"""Run the public-set evaluator against a chosen agent, without editing it.

    python3 tools/run_eval.py --agent pipeline   # A's exploit arm
    python3 tools/run_eval.py --agent baseline   # shipped weak BM25 reference

Provisional -- Person C owns the real harness; this exists so A's two metrics
are measurable today. Scores against a proxy catalog are directional only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def make_agent(name: str, catalog: str, use_prior: bool = True, use_dense: bool = True,
               reranker: str = "local", ranking_model: str | None = None,
               exposure_gate: bool = True):
    if name == "pipeline":
        from pipeline.agent import PipelineAgent
        return PipelineAgent(catalog, use_prior=use_prior, use_dense=use_dense,
                             reranker=reranker, ranking_model=ranking_model,
                             exposure_gate=exposure_gate)
    from starter.agent import Agent
    return Agent(catalog)


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
                       exposure_gate=not args.no_exposure_gate)
    built = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    finished = time.perf_counter()

    result["timing"] = {
        "build_seconds": round(built - started, 2),
        "eval_seconds": round(finished - built, 2),
        "seconds_per_session": round((finished - built) / max(1, len(samples)), 4),
    }
    output = Path(args.output or f"results_{args.agent}.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
