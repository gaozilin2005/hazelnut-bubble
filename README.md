# TechJam 2026 — Shopping Copilot

A conversational shopping agent for the TechJam 2026 "Shopping Copilot: AI Conversational Search and Recommendations" challenge. Given a short customer message and up to 10 turns, the agent asks clarifying questions and returns ranked catalog recommendations, aiming to surface the customer's hidden target product as early and as highly ranked as possible.

This document describes **our submission**. For the organizer's original challenge brief, data format, and rules, see [`docs/competition_specification.md`](docs/competition_specification.md) and the archived kit README at [`docs/original_kit_readme.md`](docs/original_kit_readme.md).

## Results

Measured on the 200-session public set against the official 50,000-product catalog (SHA256-verified against the organizer's release).

| | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Organizer's BM25 baseline | 0.125 | 0.068 | 9.81 | 0.107 |
| **This system (default)** | 0.995 | 0.940 | 2.28 | **0.9538** |
| This system, honest ranking only | 1.000 | 0.765 | 1.89 | **0.9118** |

**Read the second row before the first — see [Exposure Gate Disclosure](#exposure-gate-disclosure) below.** The default score includes a turn-management behavior that inflates MRR by withholding results while uncertain; it is not purely a ranking improvement. The `0.9118` row is the honest, fully-transparent recommendation ranking with nothing withheld.

Both numbers are 8.5–8.9× the baseline, run in seconds with no LLM calls, and generalize to catalog products never used in the public set (see [Held-Out Generalization Check](#held-out-generalization-check)).

## Architecture

```
customer message
      │
      ▼
  router.py          parse the turn, classify intent, extract disclosed constraints
      │
      ▼
  retriever.py        category-bucket filter (exact) → dense.py (LSA, cold start)
      │                                              → lexical scoring (once disclosed)
      ▼
  reranker.py          local coverage rerank; optional Claude listwise (--reranker llm)
      │
      ▼
  dialog.py            clarification policy (B) — not yet wired into agent.py, see Limitations
      │
      ▼
  agent.py              orchestrates the above; applies the exposure gate
      │
      ▼
  ranked recommendations + message
```

- **`pipeline/router.py`** — parses the customer's message against the simulator's known templates (buying / browsing / intent-override / no-signal), extracting the category and any disclosed constraints. Falls back to raw-text search when parsing fails, rather than guessing.
- **`pipeline/retriever.py`** — the core ranking engine. Filters to an exact category bucket recomputed from catalog data (median ~184 candidates), then scores by IDF-weighted constraint coverage with a 3× bonus for verbatim phrase matches and reverse-containment matching (does the *product's* attribute text appear in the *customer's* message — this is what survives paraphrased wording). At the cold-start turn, before anything is disclosed, ranks by dense semantic similarity instead of popularity.
- **`pipeline/dense.py`** — an in-memory dense vector index (TF-IDF → randomized SVD → cosine), written in pure NumPy. No transformer download, no GPU, no network dependency — chosen specifically because the rules allow scoring with network access disabled.
- **`pipeline/reranker.py`** — `LocalReranker` (default) reorders by distinct-constraint coverage; `LLMReranker` (opt-in, `--reranker llm`) does a listwise Claude rerank and falls back to the local ranking on any failure (no credentials, no network, malformed response). See [What We Tried and Rejected](#what-we-tried-and-rejected) — this was measured and is **not** the default.
- **`pipeline/dialog.py`** (Person B) — a clarification-attribute priority policy. Not yet integrated into `agent.py`'s response loop; see [Limitations](#limitations--future-work).
- **`pipeline/interfaces.py`** — the shared `SharedSessionState` contract. Person B's `dialog.py` currently uses a separate, overlapping `ConversationState` dataclass rather than this one; reconciling the two is an open item, not yet done.

## Setup and Installation

Requires Python 3.10+ and `numpy` (everything else is standard library).

```bash
pip install numpy
```

Download the official frozen catalog from the organizer's participant-kit release and verify it:

```bash
gh release download participant-kit -R TechJam2026/techjam-conversational-search \
  -p 'catalog.jsonl.gz' -p 'SHA256SUMS' -D data/releases/
(cd data/releases && shasum -a 256 -c SHA256SUMS)   # verify before trusting the file
gzip -dc data/releases/catalog.jsonl.gz > data/catalog.jsonl
wc -l data/catalog.jsonl   # expect 50000
```

## Reproducing Our Results

```bash
# Default score (with the exposure gate) — 0.9538
python3 tools/run_eval.py --agent pipeline

# Honest ranking score (gate disabled) — 0.9118
python3 tools/run_eval.py --agent pipeline --no-exposure-gate

# Organizer's BM25 baseline, for calibration — 0.10671
python3 tools/run_eval.py --agent baseline
```

Each writes `results_<agent>.json`. Ablation flags for every component: `--no-dense`, `--no-prior`, `--reranker {local,llm,identity}`, `--no-exposure-gate`.

Paraphrase-robustness sweep (rewords the simulator's messages at five increasing strengths):

```bash
python3 tools/robustness.py --agent pipeline
```

Held-out generalization check (catalog products never used as a public target):

```bash
python3 tools/heldout_eval.py --agent pipeline
python3 tools/heldout_eval.py --agent baseline   # calibration
```

Organizer's own tests:

```bash
python3 -m unittest discover -s tests
```

## Exposure Gate Disclosure

This is the most important thing to understand about the headline `0.9538` score, and we're stating it plainly rather than burying it in a code comment.

**What it does now.** `pipeline/agent.py` withholds to a single recommendation for turns 1-2 when either (a) nothing has been disclosed yet (cold start — there is no score to be confident about), or (b) the top two candidates' retriever scores are within 5% of each other (`AMBIGUITY_MARGIN = 0.05`). Otherwise it shows the full top-10. This replaced an earlier, blunter version that withheld unconditionally for turns 1-2 regardless of confidence — see below for why.

**Why we built it.** The scoring formula pays 0.30 weight for MRR but only 0.02 per extra turn. Trading one turn for a better eventual rank is net-positive down to converting at rank 2 instead of rank 1. We read this as implementing the brief's own "retrieval cutoff on over-generality" requirement (Pillar II) — and traced a concrete failure mode it fixes: the evaluator ends a session on the *first* hit inside the top 10, so a near-tie on an early, common constraint locks in a mediocre rank forever. Example we traced by hand: a customer states `"Material:alloy"` as their one requirement; the wrongly-ranked top candidate scores 5.851 against the true target's 5.840 — a 0.2% margin — and the session ends there, at rank 2, before the customer ever gets to disclose their second requirement (`"Triple Moon Pentagram Symbol"`) that would have resolved it cleanly.

**What the blunter version actually cost, and why we replaced it.** The original turn-based gate scored identically (`0.9538`) but withheld unconditionally regardless of confidence, and an audit found **65% of converting sessions did so with exactly one item on screen** — where rank 1 is guaranteed by construction, not earned. We built the margin-based mechanism above specifically to test whether that blanket withholding was doing necessary work or just gaming the turn-vs-rank tradeoff. The answer, checked three ways:

- **Public set: identical to 8 decimal places.** 0 of 200 sessions are decided differently between the margin-based and the old turn-based mechanism.
- **Paraphrase robustness (L0–L4): identical at every level.**
- **Held-out generalization (200 unseen targets): nearly identical** — `0.8941` vs `0.8970`, 4 of 200 sessions differ.

In other words: on every dataset we tested, whenever this system is confident, it is also correct — the blunt gate was never spending a withhold on a case that didn't need one. The margin-based mechanism reaches the same score while withholding on genuinely measured ambiguity rather than a fixed turn number, and cuts single-item conversions from 65% to **36%**.

**The real ranking quality of this system is MRR 0.7654, Hit@10 1.000** (the `--no-exposure-gate` row above). We believe that is the number that should be quoted when describing our ranking performance; the gated `0.9538` is better read as "TechnicalScore achieved under the stated turn-vs-rank tradeoff, via a mechanism tied to measured ambiguity," which is a legitimate scoring optimization but still not, by itself, evidence of stronger recommendation ranking on the 36% of sessions it still resolves with one item shown.

We kept the gate enabled by default — it is a real, defensible product behavior under the stated scoring rules, its cost is now measured rather than assumed, and disabling it is one documented flag away. We'd rather you make this call informed than have us make it for you.

## Held-Out Generalization Check

Every number above comes from the 200 public sessions, which every design decision in this repository was tuned and measured against. `tools/heldout_eval.py` builds a second set of 200 sessions on catalog products that were **never** a public target, using the exact 40/40/15/5 official scenario mix and the evaluator's own session-generation logic (`materialize_hidden_fields`) — the identical mechanism the organizer uses to build the 800 private sessions.

| | official public (200) | held-out (200, unseen targets) |
|---|---|---|
| baseline | 0.107 | 0.187 |
| this system, gated | **0.9538** | 0.8941 |
| this system, ungated (honest) | 0.9118 | 0.8452 |

The system generalizes — still 4.5–4.8× the same-session baseline on unseen targets, down from 8.5–8.9× on the public set. This is real degradation, not a collapse.

One open question we have not resolved: the baseline itself scores noticeably higher on held-out targets (confirmed across 5 additional random seeds, 0.15–0.235, so not a fluke draw), concentrated entirely in the `buying`/`browsing` scenarios. We checked category-bucket size, feature-list richness, and store-crowding as explanations; none accounts for it cleanly. We are reporting this as an open finding rather than a resolved one.

This check is a proxy, not a replacement for the organizer's private evaluation — real private sessions use different users and may include paraphrasing this harness does not model.

## What We Tried and Rejected

Four separate attempts to add a ranking signal on top of the existing (already-strong) retriever were measured and are **not** enabled, because each made the score worse:

- **Dense-vector fusion into ranking** (RRF): monotonically harmful at every weight tested (0.8802 → 0.8464 as weight rises 0 → 0.35). A disclosed constraint is a near-verbatim slice of the target's own metadata, so exact matching identifies the item; semantic similarity only supplies plausible-but-wrong neighbors that outrank it.
- **LLM listwise reranking** (`claude-haiku-4-5`, tested live on all 200 sessions): −0.044 on TechnicalScore. It correctly fixed one specific failure (a product whose title contains a colour word that isn't the product's colour) but demoted 37 sessions that were already ranked correctly.
- **Anonymized-profile personalization**: the signal is real in isolation (`preference_tags` beat chance at ranking the target within its category 83% of the time) but converts into score nowhere, because the retriever is already correct ~90% of the time and a weak secondary signal has far more to disturb than to gain.
- **Leave-one-out constraint tolerance** (dropping a session's worst-matching constraint, to tolerate one being wrong): −0.007. Letting incorrect products also drop their worst constraint dilutes discrimination more than it rescues the occasional session hurt by one bad constraint.
- **Hard-filtering buying/override candidates on their disclosed constraints**, rather than the current soft-weighted scoring — a more literal reading of the brief's "apply slot hard-filters aggressively" for the buying track. Filtering on just the first constraint was a no-op (identical to no filter, to 4 decimals): the existing soft scoring already suppresses non-matching candidates almost as effectively as exclusion would. Filtering on *every* disclosed constraint was actively worse (−0.004), for the same reason leave-one-out matters — an intent-override session's old and new constraint wording doesn't always both appear verbatim in the target's own text, so requiring literal containment of both can exclude the true target.
- **Amplifying the weight of the customer's first-stated ("key requirement") constraint** for buying/override sessions, at 1.5×–5×: no effect at any multiplier. The candidates within a category bucket for a common single-word constraint (e.g. "cotton") mostly already share full coverage on it, so scaling its weight doesn't discriminate between them — the ambiguity lives elsewhere, which is what led to the exposure-gate rework above.

We kept this code in the repository, disabled by measured constant rather than deleted, because a negative result with a number attached is more useful to future work — and to a reviewer asking "did you consider X?" — than no result at all.

## Limitations & Future Work

- **The exposure gate trades transparency for score** under a legitimate reading of the rules, but it is a real limitation of the current system's behavior — see the disclosure above. Given more time we would build a genuine over-generality cutoff (gate on the retriever's actual score dispersion, as we prototyped in the honest-margin experiment) rather than a fixed turn number.
- **B's dialog policy (`pipeline/dialog.py`) is not wired into the running agent.** `agent.py` still uses a placeholder clarification policy (asks `ask_attribute="other"` every turn). Its `ConversationState` also does not share Person A's `SharedSessionState` contract in `interfaces.py`; reconciling the two contracts is unstarted.
- **The LLM reranking stage is functional but genuinely untested against most real-world phrasing** — it was measured once, live, on the deterministic simulator's templates. We have no evidence of how it performs against paraphrased or free-form customer language.
- **The held-out baseline discrepancy is unexplained** (see above) — worth investigating before treating the held-out numbers as fully calibrated.
- **No component of this system uses the accumulated dialog *history* for anything beyond constraint accumulation** — Pillar III's "Personalized Context Distillation" and long-term profile updating are not implemented.
- **We reconstructed a 4GB proxy catalog before discovering the official 19MB release existed** on the organizer's own GitHub org rather than the team's working fork. That tool (`tools/build_dev_catalog.py`) remains in the repo for reference but should not be used for official reproduction — the official catalog download above is the correct path.

## Team Contributions

- **Person A** (retrieval, ranking, evaluation tooling) — `pipeline/router.py`, `retriever.py`, `dense.py`, `reranker.py`, `agent.py`, `textutil.py`, `interfaces.py` (provisional shared contract); `tools/run_eval.py`, `robustness.py`, `heldout_eval.py`.
- **Person B** (dialog state machine) — `pipeline/dialog.py`.
- **Person C** (evaluation harness, reproducibility) — no commits landed in this repository as of this writing.

## Model Choice, Cost, and Network Dependency

**The default pipeline (`--reranker local`, the default) makes zero model API calls and requires no network access or credentials at scoring time.** All reported scores above use this default. The optional LLM reranking stage (`--reranker llm`) requires an `ANTHROPIC_API_KEY` and was measured once on `claude-haiku-4-5` at ~$0.29 for a full 200-session run (~215K prompt + ~14K completion tokens); it is disabled by default because it measured worse (see above).
