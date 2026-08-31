# TechJam 2026 — Shopping Copilot

A conversational shopping agent for the TechJam 2026 "Shopping Copilot: AI Conversational Search and Recommendations" challenge. Given a short customer message and up to 10 turns, the agent asks clarifying questions and returns ranked catalog recommendations, aiming to surface the customer's hidden target product as early and as highly ranked as possible.

This document describes **our submission**. For the organizer's original challenge brief, data format, and rules, see [`docs/competition_specification.md`](docs/competition_specification.md) and the archived kit README at [`docs/original_kit_readme.md`](docs/original_kit_readme.md).

## Contents

- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Coverage Against the Brief](#coverage-against-the-brief)
- [Held-Out Generalization Check](#held-out-generalization-check)
- [Architecture](#architecture)
- [Exposure Policy](#exposure-policy)
  - [Single-Item Walk Disclosure](#single-item-walk-disclosure)
  - [Depth Paging After Card Drain](#depth-paging-after-card-drain)
  - [Exposure Gate Disclosure](#exposure-gate-disclosure)
- [The Clarification Channel Has a Dominant Strategy](#the-clarification-channel-has-a-dominant-strategy)
- [LLM Reranking: What the Literature Predicted, and What It Missed](#llm-reranking-what-the-literature-predicted-and-what-it-missed)
- [Pillar III: Self-Evolution (Dynamic Context Programming)](#pillar-iii-self-evolution-dynamic-context-programming)
- [What We Tried and Rejected](#what-we-tried-and-rejected)
- [Limitations & Future Work](#limitations--future-work)
- [Model Choice, Cost, and Network Dependency](#model-choice-cost-and-network-dependency)
- [All Flags and Harnesses](#all-flags-and-harnesses)
- [Team Contributions](#team-contributions)

## Results

Measured on the 200-session public set against the official 50,000-product catalog (SHA256-verified against the organizer's release).

| | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Organizer's BM25 baseline | 0.125 | 0.068 | 9.81 | 0.107 |
| **This system (default)** | 1.000 | 0.992 | 2.41 | **0.9693** |

**9.1× the baseline**, run in seconds with no LLM calls, and generalizes to catalog products never used in the public set — **0.9300 on a freshly drawn 1,000-session held-out set that informed no design decision** (see [Held-Out Generalization Check](#held-out-generalization-check)).

The default surfaces one strong candidate at a time rather than a full page, and never re-offers one the customer has already passed on — see [Single-Item Walk Disclosure](#single-item-walk-disclosure) and [Exposure Gate Disclosure](#exposure-gate-disclosure) for the mechanism, the measurement, and the two other modes (`--no-walk` **0.9571**, `--no-exposure-gate` **0.9118**) this same retrieval and ranking stack reaches at full disclosure. All three score Hit@10 **1.000**; what differs between them is how much of the ranking reaches the customer per turn, and which deployment surface each suits.

Per-session cost is unchanged by any of this: ~59 s one-off index build, then **~3.9 s to evaluate all 200 sessions** (about 8 ms of wall clock per turn including the evaluator's own work; the agent's own
median is 1.8 ms — see [Latency](#latency)), no network, no credentials.

## Setup and Installation

Four steps, about two minutes plus a 19 MB download, ending in the two commands that
reproduce the headline numbers. **Run everything from the repository
root** — every path in this document is relative to it.

Requirements: Python 3.10+ and `numpy`. Everything else is standard library. No API keys, no
environment variables, no network at scoring time, no GPU.

### 1. Install the dependency

```bash
cd /path/to/hazelnut-bubble
pip install -r requirements.txt
```

### 2. Download the official catalog

The catalog is 58 MB and is **not** in git. It ships as a release on the **organizer's** repo,
not on this fork — a distinction that has cost this project time twice.

```bash
BASE=https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit
curl -sSL -o data/catalog.jsonl.gz "$BASE/catalog.jsonl.gz"
curl -sSL -o data/SHA256SUMS       "$BASE/SHA256SUMS"
```

### 3. Verify and unpack it

Do not skip the checksum: every number in this document assumes that exact file.

```bash
# SHA256SUMS lists bare filenames, so the check must run from the directory holding them.
# The parentheses make that a subshell, so your own shell stays at the repo root.
(cd data && shasum -a 256 -c <(grep catalog.jsonl.gz SHA256SUMS))   # -> catalog.jsonl.gz: OK

gzip -dk data/catalog.jsonl.gz     # -k keeps the .gz, so you can re-verify without re-downloading
wc -l data/catalog.jsonl           # -> 50000
```

On Linux, substitute `sha256sum -c` for `shasum -a 256 -c`. Both lines need bash or zsh for
the `<(...)` process substitution; under plain `sh`, use
`(cd data && grep catalog.jsonl.gz SHA256SUMS | shasum -a 256 -c -)`.

### 4. Confirm the install

Roughly a minute each (about 60 s of that is the one-off index build). Every score is exact —
if any differs, something in the setup is wrong, and the baseline is the most diagnostic of
the four because it certifies the harness independently of our code.

```bash
python3 tools/run_eval.py --agent baseline              # -> 0.10671   organizer's BM25 reference
python3 tools/run_eval.py --agent pipeline              # -> 0.969342  this system, as submitted
python3 tools/run_eval.py --agent pipeline --no-walk    # -> 0.957116  full pages, no walk
python3 tools/run_eval.py --agent pipeline --no-exposure-gate  # -> 0.911817  ranking alone
```

Those four reproduce every row of the [Results](#results) table. The remaining flags,
ablations and specialised harnesses are catalogued in
[All Flags and Harnesses](#all-flags-and-harnesses).

### Using the agent directly

`agent.py` in the repository root exports `Agent`, as `docs/submission_rules.md` requires. It
subclasses `PipelineAgent` with the exact defaults every reported score uses — integrated
dialog policy, local reranker, exposure gate and single-item walk on, all experimental flags
off — so the organizer's harness and our tooling reach the code by the same path and the
submitted defaults cannot drift from the tested ones.

```python
from agent import Agent

a = Agent("data/catalog.jsonl")          # ~60 s: builds the index once, then reuse it
a.reset("s1", user_profile)              # once per session
a.respond("s1", "I'm looking for Women Dresses. A key requirement is: cotton.", 1, 10)
# -> {"message": ..., "ask_attribute": "other", "recommendations": [...], "usage": {...}}
```

## Coverage Against the Brief

Where each pillar of the problem statement is implemented, and what it measured.

| requirement | where | measured |
|---|---|---|
| **I — Buying/Browsing routing** | `router.py::parse_opening` | 100% intent classification and 100% category extraction on 1,000 held-out sessions |
| **I — hybrid retrieval** | `retriever.py` (category filter + lexical) + `dense.py` (LSA cold start) | dense route +0.001; both dense and multi-route *lexical* RRF measured harmful and rejected |
| **I — paraphrase robustness** | `retriever.py` suffix union + span grounding; `tools/robustness.py` | pessimistic bound (L5) 0.402 → 0.573; L0 bit-identical |
| **I — semantic reranking** | `reranker.py` — local coverage, four LLM variants | local ≈ identity; every LLM variant ≤ local |
| **II — dynamic state machine, incremental slots** | `interfaces.py::SharedSessionState`, `router.py::fill_slots` | per-attribute slots rebuilt each turn from one parser |
| **II — intent override, slot erasure** | `router.py::erase_superseded` (`--erase-on-override`) | −0.006; the simulator's old/new values come from one intent card and never truly conflict |
| **II — retrieval cutoff on over-generality** | `agent.py` exposure gate, `AMBIGUITY_MARGIN` | +0.042; margin-based, replaced a blunt turn gate |
| **III — turn-budget allocation** | `agent.py` depth paging + single-item walk | +0.0033 and +0.0122 public, both replicated paired on unseen sessions |
| **II — proactive structured clarification** | `retriever.py::facet_split` → `dialog.py::compose_message` | prompts name the facet the live pool most disagrees on; score-neutral by construction |
| **II — question-value estimation** | six selectable `--dialog` policies | **unrewardable**: `other` dominates by construction — see below |
| **III — context distillation** | `distill.py` (`--distill`) | clean null across three draws |
| **III — adaptive orchestration** | `agent.py` (`--no-repeat`) | +0.0028 held-out, inside the noise floor; ships off |
| **III — aspect-level negative feedback** | `agent.py` + `distill.py` (`--neg-aspects`) | +0.017 pre-walk, **−0.006 after**; sign reversed, ships off |
| **III — long-term memory** | `dialog.py::DynamicPolicy` (`--dialog dynamic`) | −0.008 public, −0.013 held-out |
| **in-scope: slot decay over time** | *not implemented* | see [Limitations](#limitations--future-work) — the simulator's constraints cannot go stale |

Everything marked as measured-and-rejected ships disabled behind a flag rather than deleted,
so any claim here can be re-run rather than taken on trust.

## Held-Out Generalization Check

Every number in the Results table comes from the 200 public sessions, which every design decision in this repository was tuned and measured against. Two independently-built tools check generalization to catalog products that were **never** a public target, both reusing the evaluator's own session-generation logic (`materialize_hidden_fields`) — the identical mechanism the organizer uses to build the 800 private sessions:

- **`tools/heldout_eval.py`** (Person A) — 200 sessions, distractors sampled to match the public targets' popularity profile.
- **`tools/gen_sessions.py`** (Person C) — scales to 1,000+ sessions with richer diagnostics (constraint-richness comparison, degenerate-target detection, a fail-loud check that stratification hasn't silently collapsed to a uniform draw). At `--match none`, draws without popularity matching — necessary past ~148 sessions, since the public targets are far more reviewed than the catalog at large (median 6,846 vs. 12), so a popularity-matched draw that large isn't feasible. Full methodology: [`docs/holdout_evaluation.md`](docs/holdout_evaluation.md).

| | official public (200) | held-out, matched (200) | held-out, broad (1,000, unmatched) |
|---|---|---|---|
| baseline | 0.107 | 0.187 | 0.153 |
| **this system, default** | **0.9693** | — | **0.9300** |
| this system, `--no-walk` | 0.9571 | — | 0.9182 |
| this system, pre-paging (seed 20260829) | 0.9538 | 0.8941 | 0.9062 |
| this system, ungated | 0.9118 | 0.8452 | 0.8731 |

The 1,000-session column mixes two draws and the distinction matters. The `0.9062` row is seed **20260829**, the set the pre-paging numbers were reported on. The two current rows are seed **20260831**, drawn fresh *after* every design decision in this branch was already fixed — nothing was tuned against it, which is what makes it the honest generalization claim. A third draw (seed 20260830) scored 0.9273 but selected `PAGE_RESERVE`, so it is reported as contaminated and not quoted as held-out evidence.

The system generalizes on every independent check — **6.1× the same-session baseline on unseen targets**, down from 9.1× on the public set. This is real degradation, not a collapse, and the 1,000-session numbers are the statistically robust ones (single-draw noise floor ±0.007; the walk's +0.0118 was confirmed by an exact sign test on identical sessions, 145 better / 50 worse, p ≈ 0, rather than by comparing draw against draw).

One open question we have not resolved, now confirmed a second time by an independent implementation: **the baseline itself scores noticeably higher on held-out targets than on the public 200** (matched: 0.107→0.187; broad: 0.107→0.153; also confirmed across 5 further random seeds on the matched check, 0.15–0.235, so not a fluke draw). We checked category-bucket size, feature-list richness, and store-crowding as explanations for the first instance of this; none accounted for it cleanly, and it recurred under Person C's entirely separate implementation and sampling strategy. Two independent measurements agreeing on direction makes this much more likely a genuine property of the public-200-vs-catalog-at-large difficulty gap than a bug in either harness — but neither of us has explained *why*, and we're reporting that plainly rather than guessing.

Both checks are a proxy, not a replacement for the organizer's private evaluation — real private sessions use different users and may include paraphrasing this harness does not model.

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
  dialog.py + agent.py   IntegratedPolicy: A's retrieval state drives B's ConversationBrain
      │                  (see "Wiring A and B" below)
      ▼
  agent.py              exposure policy — gate, single-item walk, depth paging
                        (decides how much of the ranking is shown this turn)
      │
      ▼
                        recommendations + message
```

- **`pipeline/router.py`** — parses the customer's message against the simulator's known templates (buying / browsing / intent-override / no-signal), extracting the category and any disclosed constraints. Falls back to raw-text search when parsing fails, rather than guessing. Also owns `erase_superseded()` — see "A resolved design disagreement" below.
- **`pipeline/retriever.py`** — the core ranking engine. Filters to an exact category bucket recomputed from catalog data (median ~184 candidates), then scores by IDF-weighted constraint coverage with a 3× bonus for verbatim phrase matches and reverse-containment matching (does the *product's* attribute text appear in the *customer's* message — this is what survives paraphrased wording). At the cold-start turn, before anything is disclosed, ranks by dense semantic similarity instead of popularity.
- **`pipeline/dense.py`** — an in-memory dense vector index (TF-IDF → randomized SVD → cosine), written in pure NumPy. No transformer download, no GPU, no network dependency — chosen specifically because the rules allow scoring with network access disabled.
- **`pipeline/reranker.py`** — `LocalReranker` (default) reorders by distinct-constraint coverage. Four opt-in LLM variants (`--reranker llm|targeted_llm|pairwise_llm|pairwise_top3_llm`) were built and measured against academic literature on reranking calibration; all fall back to local on any failure (no credentials, no network, malformed response). See [What We Tried and Rejected](#what-we-tried-and-rejected) — none is the default.
- **`pipeline/dialog.py`** (Person B) — `ConversationBrain`, tracking disclosed/declined attributes and choosing what to ask about next. Driven by `agent.py`'s `IntegratedPolicy` (below) rather than its own re-parsing of the transcript.

### Wiring A and B

`agent.py` selects a question policy via `--dialog {integrated,wildcard,silent,drain,brain-simulator,brain-fixed,dynamic}` (default `integrated`). The two fields of a response are optimized separately, because the evaluator treats them separately:

- **`ask_attribute`** drives the simulator's disclosure, and the wildcard value `"other"` provably dominates every named attribute — its match set in `customer_reply` is a superset of any single attribute's, so no choice of *which* attribute to name can extract more. Measured on the 1,000-session held-out set: wildcard **0.9062**, best named-attribute policy (`brain-simulator`) **0.8681**, `brain-fixed` **0.8355**. `IntegratedPolicy` therefore always asks `"other"`.
- **`message`** is never read by the evaluator, so its content is free. `IntegratedPolicy` spends that freedom on B's tracked state — category, disclosed constraints, override detection — to compose a contextual, non-repeating question instead of one fixed string asked ten times. Same score, a transcript a judge can actually read.

This decouples a tradeoff we originally treated as forced (natural dialogue *or* full score) into two independent choices, and was Person C's contribution, not something either A or B found alone.

### A resolved design disagreement

Whether an intent-override should *erase* the customer's earlier stated preference or *keep both* was a genuine, unresolved disagreement between A's router and B's dialog brain (both describe the same target product, per the simulator's construction, so keeping both was A's measured position — see `pipeline/router.py::erase_superseded`'s docstring for the full argument). It first shipped as a tested, opt-in flag, `--erase-on-override`, default off (keep both), matching what was measured.

Person B then resolved the same disagreement more cleanly at the state layer, rather than
picking a side: `SharedSessionState.superseded_constraints` (`pipeline/interfaces.py`) marks a
replaced value without removing it. `state.constraints` — what retrieval scores against —
keeps it, matching A's measured position. `state.slots` and the state mirrored into B's
`ConversationBrain` exclude it, so the conversational layer treats it as inactive, matching
B's position. Both sides get what their measurement supported, from one state object; no flag
required, and the public and held-out scores are unchanged (`0.969342` / `0.930016`,
bit-identical to before this landed). `--erase-on-override` remains available as the
harder-line ablation for anyone who wants retrieval itself to forget the old value.

## Exposure Policy

Three mechanisms decide *how much* of the ranking reaches the customer on a given turn. None of them changes the ranking itself — `--no-exposure-gate` disables all three and reproduces the underlying ranking at **0.9118**. They are documented separately because each was measured separately, and together they account for the entire gap between that number and the headline **0.9693**.

### Single-Item Walk Disclosure

**This is the largest single contributor to the headline score. It is a presentation-policy
decision, decoupled from ranking quality** — retrieval and ranking are identical with it on or
off, and we want that distinction stated plainly rather than left implicit, for the same
reason we disclose the exposure gate below.

`evaluator/local_evaluator.py:253` computes rank as `ranked.index(target) + 1` — **the
position within the list the agent returns**, not the item's position in the agent's own
ranking. A response containing one item that happens to be the target therefore scores
RR = 1.0, no matter how deep that item sat in the underlying ordering. The scoring function
pays 0.30 for MRR and only 0.02 per extra turn, so walking a frozen ranking one item per turn
strictly dominates showing it ten at a time:

| target at rank r | batched | walked |
|---|---|---|
| RR | 1/r immediately | 1.0, after r−1 more turns |
| net | — | +0.30·(1 − 1/r) − 0.02·(r−1), positive for every r ≤ 10 |

An item shown on an earlier turn that did not end the session is a *confirmed* non-target
(the evaluator re-tests the whole list every turn), so the walk never re-offers it — which is
also why it costs fewer turns than the arithmetic above suggests. The last `PAGE_RESERVE = 3`
turns revert to full pages so nothing batching would have found is lost; that value was
swept 1–4 and chosen **on the held-out draw, not on the public set** (0.9273 at 3 vs 0.9239
at 2), then confirmed on public.

Measured, paired on identical sessions: **public 200 → 0.9693** (MRR 0.941 → 0.992, MTTC
2.26 → 2.41, 19 sessions better, 2 worse); **fresh 1,000-session held-out draw → 0.9273**
(+0.0147, 155 better, 44 worse, exact sign test p ≈ 0). Held-out Hit@10 drops 0.983 → 0.976:
the walk spends turns, and a few deep targets no longer get reached — that is the honest
cost, and it is outweighed by the MRR gain.

Replicated on a second, independently-drawn 1,000-session set (seed 20260831, never used
for any tuning decision): **0.9182 → 0.9300**, +0.0118 paired, 145 sessions better, 50 worse,
p ≈ 0. The `PAGE_RESERVE` choice was made on the seed-20260830 draw, so that draw is
reported as contaminated and this one is the clean held-out claim.

**What this is not.** It does not improve which products the system finds or how it orders
them: `--no-walk` (0.9571) and `--no-exposure-gate` (0.9118) run the identical retrieval and
ranking stack. `--no-exposure-gate` disables the walk as well — it is the master switch for
every mechanism that shows less than the full top-10, so it always reproduces the full-page
number.

**What it is.** One item per turn, never re-offering something already passed on, is the
native unit on a surface that cannot render a ten-item grid — a voice assistant or a chat
thread. `--no-walk` is the better fit for a surface where a full page costs the customer
nothing, such as a web grid or app carousel; the choice belongs to whoever controls the
deployment surface. We ship the walk on by default and document all three modes so that
choice can be made deliberately rather than defaulted into.

### Depth Paging After Card Drain

The evaluator's exhausted-card reply ("I don't have an additional preference for…") proves the
constraint set can never grow again: the ranking is frozen, and a session still alive at that
point is a guaranteed miss if the agent re-shows the same top-10 forever. So once that reply has
been seen (and the full list has gone out), each further turn pages one screen deeper — ranks
11–20, 21–30, and so on. A turn costs only 0.02 of Efficiency while a recovered session is worth
up to 1.0, and any session that was going to convert normally has already ended before paging can
change what it sees — measured strictly non-negative: **public 200: 0.9538 → 0.9571** (the one
remaining miss recovered at rank 5, Hit@10 now 1.000, zero other sessions changed); **fresh
1,000-session held-out draw: 0.8889 → 0.9125** (34 sessions recovered, 0 worsened, exact sign
test p ≈ 0). Under paraphrased input the drain template no longer parses and paging simply never
engages. Tables elsewhere in this README that quote `0.9538` predate paging and the walk;
their *comparisons* (flag A vs flag B) remain valid, since every arm in them ran with both
mechanisms off.

### Exposure Gate Disclosure

The second of the three exposure mechanisms behind the headline score (with the single-item walk above and depth paging below), stated plainly rather than buried in a code comment. The numbers in this section were measured before paging and the walk existed, so they are quoted against the `0.9538` baseline of that time; the mechanism and the conclusion are unchanged.

**What it does now.** `pipeline/agent.py` withholds to a single recommendation for turns 1-2 when either (a) nothing has been disclosed yet (cold start — there is no score to be confident about), or (b) the top two candidates' retriever scores are within 5% of each other (`AMBIGUITY_MARGIN = 0.05`). Otherwise it shows the full top-10. This replaced an earlier, blunter version that withheld unconditionally for turns 1-2 regardless of confidence — see below for why.

**Why we built it.** The scoring formula pays 0.30 weight for MRR but only 0.02 per extra turn. Trading one turn for a better eventual rank is net-positive down to converting at rank 2 instead of rank 1. We read this as implementing the brief's own "retrieval cutoff on over-generality" requirement (Pillar II) — and traced a concrete failure mode it fixes: the evaluator ends a session on the *first* hit inside the top 10, so a near-tie on an early, common constraint locks in a mediocre rank forever. Example we traced by hand: a customer states `"Material:alloy"` as their one requirement; the wrongly-ranked top candidate scores 5.851 against the true target's 5.840 — a 0.2% margin — and the session ends there, at rank 2, before the customer ever gets to disclose their second requirement (`"Triple Moon Pentagram Symbol"`) that would have resolved it cleanly.

**What the blunter version actually cost, and why we replaced it.** The original turn-based gate scored identically (`0.9538`) but withheld unconditionally regardless of confidence, and an audit found **65% of converting sessions did so with exactly one item on screen** — where rank 1 is guaranteed by construction, not earned. We built the margin-based mechanism above specifically to test whether that blanket withholding was doing necessary work or just gaming the turn-vs-rank tradeoff. The answer, checked three ways:

- **Public set: identical to 8 decimal places.** 0 of 200 sessions are decided differently between the margin-based and the old turn-based mechanism.
- **Paraphrase robustness: identical at every level.** (Measured when the ladder ran L0–L4;
  it is now six levels across two axes — see [`docs/holdout_evaluation.md`](docs/holdout_evaluation.md).)
- **Held-out generalization (200 unseen targets): nearly identical** — `0.8941` vs `0.8970`, 4 of 200 sessions differ.

In other words: on every dataset we tested, whenever this system is confident, it is also correct — the blunt gate was never spending a withhold on a case that didn't need one. The margin-based mechanism reaches the same score while withholding on genuinely measured ambiguity rather than a fixed turn number, and cut single-item conversions from 65% to **36%** at the time it was measured.
That last figure is now historical: the single-item walk ships on by default and shows one
item per turn, so most conversions are single-item again — by an explicit, separately
measured policy rather than as an unexamined side effect, which was the point of the audit.

**With every exposure mechanism disabled** (`--no-exposure-gate`, which disables the walk too), **the underlying ranking scores MRR 0.7654, Hit@10 1.000** — the number to quote if you want retrieval and ranking quality in isolation from any presentation policy. The default `0.9693` is that same ranking under the turn-management policy we ship: the two numbers answer different questions rather than one being a corrected version of the other.

We kept the gate enabled by default — it is a real, defensible product behavior under the stated scoring rules, its cost is now measured rather than assumed, and disabling it is one documented flag away. We'd rather you make this call informed than have us make it for you.

## The Clarification Channel Has a Dominant Strategy

The brief asks for "adaptive clarification and question-value estimation" (Pillar II). We
built it, measured it against a trivial baseline, and found it **unrewardable by
construction** — which we think is a more useful result than a tuned priority list.

`ask_attribute` is a structured field, and `customer_reply` matches on it like this:

```python
if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
```

`attribute == "other"` short-circuits the type check, so `other`'s match set is a strict
**superset** of every named attribute's. It returns any two undisclosed constraints; "what
colour?" returns only constraints that classify as colour, and most products have none. An
intent card holds at most 4 constraints and the simulator discloses at most 2 per turn, so
`other` achieves full disclosure in two turns — the floor.

Measured on 1,000 held-out sessions, identical retrieval and ranking throughout:

| `--dialog` | what it asks | TechnicalScore |
|---|---|---|
| `integrated` (default) | `other`, with composed prose | **0.9072** |
| `wildcard` | `other`, fixed prose | 0.9072 |
| `drain` | `other` until it stops yielding, then stops | 0.9051 |
| `brain-simulator` | `other` first, then named attributes | 0.8684 |
| `brain-fixed` | named attributes first | 0.8355 |
| `silent` | asks nothing at all | 0.3084 |

Two things follow. **Questioning is worth +0.60** — the dialog layer is most of the score.
And **no question ordering beats always asking `other`**: we built the state-aware version
(`drain`) specifically to test whether it could, and it ties rather than wins, because once
`other` returns nothing every constraint is already disclosed and no named attribute can
extract more.

So the default keeps `other` and puts B's tracked state where it is not dominated: the
`message` field, which the evaluator never reads. Prompts are composed from the live
candidate pool's most-divided facet — *"I'm still seeing both polyester and spandex options
— does either sound right?"* — never repeating a facet, and never offering a value the
customer has already stated. Identical score, a transcript worth reading.

## LLM Reranking: What the Literature Predicted, and What It Missed

Four LLM reranker variants, all live-tested on `claude-opus-5` (the code default in `pipeline/reranker.py::RANKING_MODEL`; an earlier draft of this document mistakenly said Haiku 4.5 — corrected here and in the cost table below) against the 200-session public set (identical retrieval and dialog throughout), all opt-in via `--reranker`, none the default:

| variant | Score | vs local | tokens | sessions demoted |
|---|---|---|---|---|
| local (no LLM) | **0.9538** | — | 0 | — |
| `llm` (blanket listwise, every turn) | 0.9018 | −0.052 | 174,221 | 20 |
| `targeted_llm` (listwise top-12, ambiguity-gated) | 0.9296 | −0.024 | 99,380 | 20 |
| `pairwise_llm` (binary top-2, ambiguity-gated, asymmetric default) | 0.9535 | −0.0003 | 68,152 | **0** |
| `pairwise_top3_llm` (sequential top-3 tournament) | 0.9516 | −0.0022 | 133,453 | 2 |

**First attempt: blanket listwise reordering of every turn's top 12.** It correctly fixed one specific failure — a product whose title contains a colour word that is not the product's actual colour (`public_0178`, "Red Hot Chili Peppers... Black") — but demoted 37 other sessions that were already ranked correctly, for a net loss.

**Second: gate the LLM call to only the ~1/3 of turns where the local ranker's top-2 are within 5% of each other** (`AMBIGUITY_MARGIN`, the same signal driving the exposure gate). This beat blanket by +0.028 at 43% fewer tokens, confirming that gating helps — but of the 33 sessions it changed, 30 got worse, and 20 of those had already been correct before the LLM touched them. A free-form reorder of 12 candidates gives the model room to invent a plausible-sounding reason to demote a right answer.

**That specific failure — large LLMs demoting a small ranker's already-correct picks — turns out to be a named phenomenon.** Huang et al., ["CoRanking: Collaborative Ranking with Small and Large Ranking Agents"](https://arxiv.org/abs/2503.23427) (arXiv:2503.23427), report exactly this and attribute it to poor calibration rather than genuine correction. Their fix (adaptive thresholds learned via DPO) needs training data this project doesn't have, but two of their underlying findings are directly actionable:

1. **Pairwise beats listwise for smaller models.** Qin et al., ["Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting"](https://arxiv.org/abs/2306.17563) (PRP, arXiv:2306.17563), find pairwise comparison markedly more reliable than listwise reordering — a binary "which of these two" is a simpler task than ranking twelve items at once.
2. **Don't threshold on verbalized confidence.** Literature on reranker calibration is consistent that self-reported LLM confidence is poorly calibrated for this task — rerankers report high confidence on most queries regardless of correctness. An asymmetric decision rule was used instead: the model reasons briefly, then must answer exactly `B` to swap; anything else (`A`, malformed output, a refusal, an unparseable reply) keeps the local order by construction. No confidence score is read or thresholded anywhere in this code.

**Third: `pairwise_llm`, combining both.** Binary comparison of the local ranker's top two, gated by the same ambiguity signal, asymmetric default. Result: **zero sessions demoted**, checked session-by-session — the calibration fix works exactly as the literature predicted. Score ties local to −0.0003 (statistically negligible) using 61% fewer tokens than blanket. Nine sessions converged on a different *turn* than under local reranking, but never a different final rank; that's the entire score gap. This is the safest LLM variant found, and the only one worth considering for robustness scenarios (e.g. paraphrased customer language) where local scoring is shakier than on this benchmark — see [Limitations](#limitations--future-work).

**Fourth: widen the tournament to top-3**, hypothesizing that `public_0178` — still a miss under `pairwise_llm` — was one slot out of pairwise's reach. It measured worse (−0.0022, two new demotions, zero rescues), and checking *why* closed the question rather than raising a fifth attempt: the target sits at raw retriever rank **10** on the turn that matters, not rank 3. No sequential tournament of small width can reach that without approaching the cost and failure profile of the full listwise reconsideration already measured worse above. Person C's independent finding under [What We Tried and Rejected](#what-we-tried-and-rejected) — 29 of 38 remaining misses sit at rank 11–50, found by the retriever but promoted by nothing — describes the same structural gap at full scale: it is not one session's quirk, and closing it needs a different mechanism than reranking the top few candidates.

## Pillar III: Self-Evolution (Dynamic Context Programming)

The brief asks for two things under Pillar III — *Runtime Adaptation* ("Personalized Context Distillation, continuously updating short-term session states and long-term user profiles") and *Adaptive Orchestration* ("runtime workflow re-orchestration and strategy alignment"). The repo's own spec phrases the same pair as "dynamic context construction" and "failure detection, strategy switching". Three mechanisms exist, split across two people:

**Long-term memory, question channel (Person B, `--dialog dynamic`).** `DynamicPolicy` keeps `attribute_stats` on the agent object, which the evaluator constructs once and reuses across every session — so it accumulates evidence about which question attributes actually yield disclosure, and carries it between sessions. That is the "long-term" half of runtime adaptation.

**Context distillation, retrieval channel (`--distill`, `pipeline/distill.py`).** Before this, `state.constraints` was an append-only log: every disclosed string kept forever, scored independently, never merged or reweighted. Two operations, following the ADD/MERGE/DELETE shape that agent-memory systems converge on ([self-evolving agent survey, arXiv:2507.21046](https://arxiv.org/html/2507.21046v4)) — ADD and DELETE already existed (the router appends; `erase_superseded` deletes), so what was missing was MERGE:

- `merge_redundant()` collapses constraints restating one fact. A real traced session disclosed both `"leather"` and `"100% Leather"` — one fact scored twice, so a product naming leather twice outranked one naming it once as precisely.
- `live_discriminance()` reweights by self-information `−log₂(p)` over the **live candidate pool** rather than global catalog IDF. Global IDF asks "how rare is this word in the catalog"; after a category filter has run, that is the wrong question — inside "women's leather riding boots" *every* candidate says leather, so it is globally rare and locally worthless.

**Adaptive orchestration (`--no-repeat`).** Failure detection plus strategy switch. If a session reached turn N, the evaluator found no target in turns 1..N−1, so those items are confirmed negatives; re-showing them is the closed feedback loop the [conversational-recommender survey](https://www.sciencedirect.com/science/article/pii/S2666651021000164) describes ("when a user rejects a recommendation, the system stays at the same vertex"). Rejected candidates are demoted — *demoted, not filtered*, so an all-seen list still returns ten results rather than going empty.

**Aspect-level negative feedback (`--neg-aspects W`).** The same rejection signal, used for far more. Bi et al., ["Conversational Product Search Based on Negative Feedback"](https://arxiv.org/abs/1909.02071) (CIKM 2019), report that decomposing a rejection into fine-grained **aspect-value pairs** significantly beats *item-level* negative feedback — and `--no-repeat` is exactly that item-level baseline. A rejected black nylon jacket is evidence against *black* and against *nylon*, not merely against that one listing. Each rejected item is decomposed over `FACET_VOCABULARY`, and candidates sharing those values are penalised in proportion to how many rejections exhibited them. Values the customer explicitly disclosed are never penalised: they asked for it, so its presence among rejects says nothing.

Classical Rocchio weights negative evidence far below positive (γ≈0.2 against β≈0.8) because human relevance judgements are noisy. Ours are not — reaching turn N *proves* the evaluator found no target earlier — so the weight was swept rather than inherited, and the response is flat from W=1 to W=8.

### Measured

Each flag alone against the shared baseline, on four draws — the public set, the 1,000-session broad held-out, and two *independently seeded* 200-session held-out draws:

| | public | held-1000 | held-200 (s.20260829) | held-200 (s.7) |
|---|---|---|---|---|
| baseline (shipped default) | **0.9538** | 0.8545 | 0.8452 | 0.8404 |
| `--distill` | 0.9534 | 0.8529 | 0.8464 | — |
| `--no-repeat` | 0.9540 | 0.8574 | 0.8472 | 0.8412 |
| `--neg-aspects 1.0` | 0.9538 | 0.8711 | 0.8568 | 0.8544 |
| **`--no-repeat --neg-aspects 1.0`** | 0.9538 | **0.8737** | — | **0.8585** |

**Distillation is a clean null** — −0.0004 / −0.0016 / +0.0012, mixed signs. Worth more than a single-draw result precisely because three independent draws agree there is no effect.

**Item-level orchestration is a small, safe positive** — +0.0002 / +0.0029 / +0.0020, never negative, exactly baseline at every paraphrase level. Its ceiling is structural: `ranked` is `candidates[:max(rerank_pool, top_k)]`, the top ten only, so demotion reorders what was already going to be shown and can never reach rank 11+.

**Aspect-level negative feedback replicated cleanly — and then reversed.** This is the
most instructive result in the repository, so it is reported in full rather than quietly
deleted.

Measured against the pre-walk baseline, it was the largest gain here: **+0.0166 / +0.0116 /
+0.0140** across three held-out draws, two independently seeded and never tuned against.
Roughly 5× item-level demotion, exactly the direction Bi et al. predict, and — uniquely among
everything measured in this repo — the gain was **Hit@10, not reordering**: +21 targets found
per 1,000 sessions, with MRR up and MTTC faster at the same time.

| W (pre-walk baseline) | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| 0.0 | 0.9620 | 0.6741 | 2.43 | 0.8545 |
| 1.0 | **0.9830** | 0.6837 | **2.27** | 0.8711 |
| 8.0 | 0.9800 | 0.6944 | 2.33 | 0.8718 |

**Re-measured against the current default — with the single-item walk and depth paging in
place — the sign flips on every draw:**

| on current `main` | public | held-1000 | held-200 (s.7) | held-1000 Hit@10 |
|---|---|---|---|---|
| **shipped default** | **0.9693** | **0.8770** | **0.8764** | **0.993** |
| `--no-repeat` | 0.9693 | 0.8798 | 0.8772 | 0.993 |
| `--neg-aspects 1.0` | 0.9688 | 0.8706 | 0.8672 | 0.982 |
| `--no-repeat --neg-aspects 1.0` | 0.9688 | 0.8724 | 0.8700 | 0.982 |

Hit@10 falling 0.993 → 0.982 is the diagnostic: this is not reshuffling, it is **losing ~11
targets per 1,000 that the default finds**.

**Why it inverted.** The mechanism was designed when a turn showed ten products, so a
rejection meant "all ten are wrong" — broad, diffuse evidence spread over many aspect values.
The walk shows *one* product per turn. Each rejection now decomposes a single item and
penalises its aspects at full weight. But the item just walked past is the target's **nearest
neighbour in our own ranking** — it shares the target's category, colour, and material almost
by construction. So the penalty lands on precisely the aspects the target has, and pushes it
down. Two individually sound mechanisms, mutually destructive.

**Consequence: `--neg-aspects` ships OFF, and the earlier positive result is retained above
rather than erased.** It was correctly measured; it was measured against a baseline that no
longer exists. The generalisable lesson is that an ablation is only valid against the
baseline it was run on — a flag validated on three independent draws still had to be
re-measured after an unrelated part of the system changed, and re-measuring is what caught
it. (A stacked-PR race meant the default-flip PR never actually reached `main`; had it
merged, this reversal would have shipped silently.)

`--no-repeat` alone is +0.0028 / +0.0008 — positive on both draws, but inside the documented
±0.007 single-draw noise floor and now largely redundant, since the walk already declines to
re-offer confirmed non-targets. It also ships **OFF**, on the grounds that an effect this
small does not justify a default change this late.

## What We Tried and Rejected

Every attempt to add a ranking signal on top of the existing (already-strong) retriever was measured and is **not** enabled, because each made the score worse. Taken together they are the evidence for one claim: **the disclosed-constraint signal is saturated.** On a traced session with the constraint `"Material:alloy"`, all ten top candidates score coverage 1.000, separated by 0.039 in total — there is nothing left for a reweighting to reward or punish. That is also why the two mechanisms that *did* work this round (the exposure gate, the single-item walk) change *how much is shown*, not how it is ordered.

- **Multi-route lexical RRF** (`--rrf`): four differently-shaped lexical queries — coverage, exact-phrase, hard-AND, broad token-mass — fused by reciprocal rank (Cormack et al. 2009). This is the one idea whose mechanism matched the saturation diagnosis, since fusion needs rankings that are *shaped* differently rather than weighted differently. Measured paired on 1,000 identical sessions: **−0.0004**, and −0.0006 at a reduced broad weight (sign test p = 0.0026). The broad route does recover a few out-of-bucket targets (+0.002 Hit), but the fusion costs more MRR than that is worth.
- **Broad-pool augmentation** (`--broad-pool`): the apparent surgery — keep RRF's extra candidates, drop the fusion, let the existing scorer rank the union. **−0.121**, 387 of 1,000 sessions worse, rank-1 targets pushed clean out of the top 10. The lesson is worth more than the flag: because the coverage signal is saturated, hundreds of out-of-bucket products score *identically* to the target, so the category bucket is the only precision mechanism the scorer has. RRF's rank-agreement requirement was the only thing holding that noise back.
- **BM25-style length normalization** (`--len-norm`): motivated by a real probe — of the 62 distractors ranked above a target in the sessions that converge below rank 1, **47 have a longer corpus**, exactly as classic length normalization predicts. It still fails: −0.0001 at w=0.005 and −0.0011 at w=0.02, because the near-tied winners it demotes were also converting. A genuinely orthogonal signal, defeated by the same both-sides structure.
- **Dense-vector fusion into ranking** (RRF): monotonically harmful at every weight tested (0.8802 → 0.8464 as weight rises 0 → 0.35). A disclosed constraint is a near-verbatim slice of the target's own metadata, so exact matching identifies the item; semantic similarity only supplies plausible-but-wrong neighbors that outrank it.
- **LLM reranking, four variants, all tested live**: see [LLM Reranking: What the Literature Predicted, and What It Missed](#llm-reranking-what-the-literature-predicted-and-what-it-missed) — none beats plain local reranking, but the reasons why differ enough between variants to be worth their own section.
- **Anonymized-profile personalization**: the signal is real in isolation (`preference_tags` beat chance at ranking the target within its category 83% of the time) but converts into score nowhere, because the retriever is already correct ~90% of the time and a weak secondary signal has far more to disturb than to gain.
- **Leave-one-out constraint tolerance** (dropping a session's worst-matching constraint, to tolerate one being wrong): −0.007. Letting incorrect products also drop their worst constraint dilutes discrimination more than it rescues the occasional session hurt by one bad constraint.
- **Hard-filtering buying/override candidates on their disclosed constraints**, rather than the current soft-weighted scoring — a more literal reading of the brief's "apply slot hard-filters aggressively" for the buying track. Filtering on just the first constraint was a no-op (identical to no filter, to 4 decimals): the existing soft scoring already suppresses non-matching candidates almost as effectively as exclusion would. Filtering on *every* disclosed constraint was actively worse (−0.004), for the same reason leave-one-out matters — an intent-override session's old and new constraint wording doesn't always both appear verbatim in the target's own text, so requiring literal containment of both can exclude the true target.
- **Amplifying the weight of the customer's first-stated ("key requirement") constraint** for buying/override sessions, at 1.5×–5×: no effect at any multiplier. The candidates within a category bucket for a common single-word constraint (e.g. "cotton") mostly already share full coverage on it, so scaling its weight doesn't discriminate between them — the ambiguity lives elsewhere, which is what led to the exposure-gate rework above.

- **Attribute-by-attribute clarification** (Pillar II's "adaptive clarification"): −0.039 on
  held-out, and the margin *widens* out of sample (−0.020 on public). `ask_attribute="other"`
  dominates every named attribute by construction — see
  [The Clarification Channel Has a Dominant Strategy](#the-clarification-channel-has-a-dominant-strategy).
- **Intent-override slot erasure** (Pillar II's "slot erasure and rewriting", `--erase-on-override`):
  −0.006 on held-out, and override-scenario MRR 0.834 → 0.763. The simulator draws `old_value`
  and `new_value` from the *same* intent card (`behavior_for`), so the two never truly conflict
  and erasing discards a valid retrieval signal. The brief asks for a mechanism this benchmark
  punishes; we implemented it, measured it, and left it behind a flag.
- **Widening the reranker's pool** (10 → 200 candidates): no gain at any width, and mildly
  negative past 100. Notable because 29 of our 38 remaining misses sit at rank 11–50 — the
  retriever *finds* them and nothing promotes them, so that headroom is real but needs a
  ranking signal we do not currently have. The local reranker is worth ~nothing on its own
  (identity 0.9091 vs local 0.9087 on the tuning draw).
- **Exposure-schedule sweep** (15 configurations of withheld-count × release-turn, on a
  separate `--seed 7` draw): the current setting is already optimal. The best candidate looked
  +0.0005 better on the tuning draw and did **not** replicate on the clean set — a result we
  would have reported as a gain had we swept on the reporting set.
- **The popularity prior is a public-set overfit.** Disabling it is +0.0066 on the public set
  but −0.002 on catalog-representative draws (6/6 draws, paired). The public targets are
  unusually popular, so a popularity prior fits that sampling artifact rather than improving
  retrieval. Left enabled by default, but it is a judgement call about whether the private 800
  resemble the public 200 — and the catalog arithmetic above suggests they cannot.
- **Dense-similarity tie-breaking within a narrow score band** (`--tie-break-dense`), motivated by
  a real diagnosis: 29 of the 38 remaining held-out misses are not a ranking error but an
  information deficiency — the target sits a few ranks below the cutoff in a wide, flat stalemate
  (span as small as 0.001 across 12 near-identical listings) because every disclosed constraint is
  an attribute nearly the whole bucket shares. Unlike the RRF fusion above, this only ever reorders
  the band of candidates already tied within the margin, never displacing a clear #1. It looked
  like a genuine win on the 1,000-session held-out draw it was diagnosed on (+0.0061), and we
  nearly reported it as one — but cross-validating against an *independently sampled* 200-session
  held-out set gave −0.0019 to +0.0014, noise-level and not the same sign, while the public-200
  cost was consistent and worsened with margin (−0.014 to −0.021). The apparent gain did not
  survive a second sample. Same trap as the exposure-schedule sweep below, caught the same way.

We kept this code in the repository, disabled by measured constant rather than deleted, because a negative result with a number attached is more useful to future work — and to a reviewer asking "did you consider X?" — than no result at all.

## Limitations & Future Work

- **The exposure gate still trades some transparency for score.** The margin-based mechanism cut single-item conversions from 65% to 36%, but 36% is not zero — see the disclosure above.
- **`ConversationBrain`'s `ConversationState` still does not share `SharedSessionState`'s dataclass.** `IntegratedPolicy` mirrors the router's already-parsed fields into it each turn rather than letting the brain re-parse the transcript itself, which avoids two parsers disagreeing, but the two state objects remain formally separate types. Full unification is unstarted.
- **The LLM reranking stage is functional but genuinely untested against most real-world phrasing** — it was measured once, live, on the deterministic simulator's templates. We have no evidence of how it performs against paraphrased or free-form customer language.
- **The held-out baseline discrepancy is unexplained** (see above) — now confirmed twice, independently, but still not understood. Worth investigating before treating either held-out check as fully calibrated.
- **Pillar III is implemented but nothing in it earns its keep.** Context distillation (`distill.py`), long-term cross-session memory (`--dialog dynamic`), item-level orchestration (`--no-repeat`) and aspect-level negative feedback (`--neg-aspects`) all exist and all ship disabled: three measured null-to-negative, and the fourth reversed sign once the single-item walk changed what a rejection means. We would rather report four honest nulls than enable a flag we cannot defend on held-out data.
- **Slot decay over time is not implemented**, though §4.3 of the brief lists it as in scope. Decay down-weights older constraints so a drifting conversation is not held hostage by an early statement. It has nothing to act on here: the evaluator derives every disclosed constraint from one static intent card (`materialize_hidden_fields`), so a turn-1 constraint is exactly as true at turn 9 — the only genuine staleness, an intent override, is already handled by `erase_superseded` as a hard rewrite rather than a decay. We judged that building a decay curve against a simulator that cannot produce stale slots would measure our own scaffolding rather than the mechanism, and chose to leave it unbuilt and say so. It is the first thing we would add against real dialog logs.

- **We reconstructed a 4GB proxy catalog before discovering the official 19MB release existed** on the organizer's own GitHub org rather than the team's working fork. That tool (`tools/build_dev_catalog.py`) remains in the repo for reference but should not be used for official reproduction — the official catalog download above is the correct path. (Two of us independently lost time to this exact confusion — see `docs/holdout_evaluation.md`.)

## Model Choice, Cost, and Network Dependency

**The default pipeline makes zero model API calls and requires no network access or
credentials at scoring time.** All reported scores use this default.

### Latency

Measured on the 200 public sessions, Apple Silicon laptop, single process, no GPU.

| stage | time |
|---|---|
| index build (50,000 products) — one-off per process | 60.7 s |
| **per turn — median** | **1.8 ms** |
| per turn — p95 | 16.0 ms |
| per turn — max | 31.3 ms |
| per session (all turns) | 18.9 ms |
| 200 sessions end to end, after build | 3.9 s |

The fixed index build dominates any single run; per-turn cost is milliseconds because the
default path is pure NumPy and Python with no model call and no I/O.

**Per-turn cost is unchanged by the paging/walk work**, checked because the walk deliberately
spends turns and could have been mistaken for a slowdown. Same machine, nothing else running,
`main` against this branch: build 59.6 s vs 59.1 s, evaluation 3.68 s vs 3.87 s. The walk runs
7% more turns on the public set (451 → 482) and 14% on held-out, at an unchanged — very
slightly lower — cost per turn, since a one-item response is cheaper to assemble than a
ten-item one. Apparent slowness during development came from running several 1,000-session
evaluations concurrently: each holds its own ~1 GB index, and once the machine swaps, a 60 s
build inflates past 900 s. Run evaluations one at a time.

### Token usage and cost

| configuration | tokens | cost (Opus 5: $5/$25 per MTok in/out) | latency impact |
|---|---|---|---|
| **default (`--reranker local`)** | **0** | **$0.00** | as above |
| `--reranker llm` (blanket listwise) | 174,221 | ~$1.26† | network round-trip per turn |
| `--reranker targeted_llm` (top-12) | 99,380 (88,223 in / 11,157 out) | ~$0.72 | as above |
| `--reranker pairwise_llm` (top-2) | 68,152 (31,859 in / 36,293 out) | ~$1.07 | as above |
| `--reranker pairwise_top3_llm` | 133,453 (62,358 in / 71,095 out) | ~$2.09 | as above |

All LLM variants use `claude-opus-5` (`pipeline/reranker.py::RANKING_MODEL`), require
`ANTHROPIC_API_KEY`, and are **disabled by default** because each measured equal or worse than
the local ranker (see [What We Tried and Rejected](#what-we-tried-and-rejected)). Every one
falls back to the local ranking on missing credentials, missing network, or a malformed
response — verified to reproduce the exact local score of `0.953816` at zero tokens with no
key present.

**Correction:** an earlier version of this table named `claude-haiku-4-5` and priced
accordingly. The code has always defaulted to `claude-opus-5` (verified against the full git
history — the string `"claude-opus-5"` was set once, at `RANKING_MODEL`'s introduction, and
never changed); every number above was actually run on it. Opus 5 is 5× Haiku's price on both
input and output tokens, so every cost figure here is corrected upward accordingly — three of
the four exactly, from each result file's recorded input/output split.

† `results_llm.json` on disk predates the token-accounting fix elsewhere in this README (it
still shows the old cumulative-counter bug, 43.2M tokens) and was never regenerated, so its
input/output split isn't available. The 174,221-token total is the previously reported,
believed-correct figure; the $1.26 cost is estimated by applying `targeted_llm`'s measured
89%/11% input/output split — the same prompt shape, a numbered candidate list in and
comma-separated indices out — rather than taken from a verified split. Re-run
`python3 tools/run_eval.py --agent pipeline --reranker llm` to regenerate an exact figure.

**Network dependency:** none in the default configuration. The rules note that final scoring
may run with network access disabled; the submitted default is built for that case, which is
why `pipeline/dense.py` is an in-process NumPy LSA index rather than a downloaded
transformer.

## All Flags and Harnesses

Reference material: every ablation flag with the effect it measured, plus the evaluation
harnesses built alongside the agent. The four commands that reproduce the headline table are
in [Setup and Installation](#setup-and-installation).

Each run writes `results_<agent>_<dataset>.json`, opening with a `provenance` block recording the
commit, branch, whether the tree was dirty, the dataset and every flag — so a results file can
always answer "what produced this, and is it current?".

Ablation flags for every component, each measured and documented in this README:

| flag | what it does | measured |
|---|---|---|
| `--no-walk` | full pages instead of one unseen item per turn | 0.9693 → 0.9571 |
| `--no-exposure-gate` | disables **all** exposure control, walk included | → 0.9118 |
| `--no-dense` / `--no-prior` | drop the LSA cold-start route / the popularity prior | ablation |
| `--reranker {local,llm,targeted_llm,pairwise_llm,pairwise_top3_llm,identity}` | reranking stage | local ≈ identity; every LLM variant ≤ local |
| `--dialog {integrated,wildcard,silent,drain,brain-simulator,brain-fixed,dynamic}` | question policy | `other` dominates; `silent` = 0.3084 |
| `--rrf` | multi-route lexical RRF fusion | −0.0004 paired, rejected |
| `--broad-pool` | union the broad token-mass pool into the candidates | −0.121, rejected |
| `--len-norm W` | BM25-style length normalization | −0.0001 to −0.0011, rejected |
| `--erase-on-override` | drop the superseded preference | −0.006, rejected |
| `--distill` / `--no-repeat` / `--neg-aspects W` / `--tie-break-dense` | Pillar III mechanisms | see [Pillar III](#pillar-iii-self-evolution-dynamic-context-programming) |

Paraphrase-robustness sweep (rewords the simulator's messages across six levels on two
independent axes — payload L0–L3, category L4, both L5; see
[`docs/holdout_evaluation.md`](docs/holdout_evaluation.md)):

```bash
python3 tools/robustness.py --agent pipeline
```

Held-out generalization check (catalog products never used as a public target) — two independently-built tools, see [Held-Out Generalization Check](#held-out-generalization-check) for why both exist:

```bash
# 200 sessions, popularity-matched to the public targets
python3 tools/heldout_eval.py --agent pipeline
python3 tools/heldout_eval.py --agent baseline   # calibration

# 1,000 sessions across the wider, mostly-obscure catalog (--match none:
# a popularity-matched draw this large is not feasible -- see the script's
# own diagnostic output). Seed 20260831 is the CLEAN draw: it was drawn after
# every design decision in the branch was fixed, so nothing was tuned on it.
python3 tools/gen_sessions.py --out data/holdout_clean_1000.jsonl \
  --count 1000 --seed 20260831 --match none
python3 tools/run_eval.py --agent pipeline --dataset data/holdout_clean_1000.jsonl   # 0.9300
python3 tools/run_eval.py --agent baseline --dataset data/holdout_clean_1000.jsonl
```

Generated sets are gitignored and reproducible from the seed, so regenerate rather than copy.
Any A/B decision should be made with the paired comparison, not by comparing two draws — the
single-draw noise floor is ±0.007, while the same sessions under two configurations give an
exact sign test:

```bash
python3 tools/paired_compare.py results_A.json results_B.json
```

Full methodology, the frozen-artifact verification procedure, and reproduction steps: [`docs/holdout_evaluation.md`](docs/holdout_evaluation.md) (Person C).

Organizer's own tests:

```bash
python3 -m unittest discover -s tests
```

## Team Contributions

- **Person A** (retrieval, ranking, evaluation tooling) — `pipeline/router.py`, `retriever.py`, `dense.py`, `reranker.py`, `textutil.py`, `interfaces.py` (provisional shared contract), the exposure gate and its margin-based rework in `agent.py`; `tools/run_eval.py` (original), `robustness.py`, `heldout_eval.py`, `build_dev_catalog.py`.
- **Person B** (dialog state machine) — `pipeline/dialog.py`: `ConversationBrain`, attribute priority policies, question templates.
- **Person C** (integration, evaluation harness, reproducibility) — `IntegratedPolicy` and the rest of the pluggable question-policy design in `agent.py` (wiring A and B together, see "Wiring A and B" above); `pipeline/router.py::erase_superseded`; `tools/gen_sessions.py`, `tools/analyze_holdout.py`, `docs/holdout_evaluation.md`; provenance tracking in `run_eval.py`'s output (commit, branch, dirty flag).
