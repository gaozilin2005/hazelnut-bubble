# Hazelnut — Shopping Copilot

Hazelnut is a conversational shopping agent for the TechJam 2026 "Shopping Copilot: AI Conversational Search and Recommendations" challenge. Given a short customer message and up to 10 turns, the agent asks clarifying questions and returns ranked catalog recommendations, aiming to surface the customer's hidden target product as early and as highly ranked as possible.

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
- [Signature Tiebreak Disclosure](#signature-tiebreak-disclosure)
- [LLM Reranking: What the Literature Predicted, and What It Missed](#llm-reranking-what-the-literature-predicted-and-what-it-missed)
- [Pillar III: Self-Evolution (Dynamic Context Programming)](#pillar-iii-self-evolution-dynamic-context-programming)
- [What We Tried and Rejected](#what-we-tried-and-rejected)
- [Limitations & Future Work](#limitations--future-work)
- [Model Choice, Cost, and Network Dependency](#model-choice-cost-and-network-dependency)
- [Reproducing Our Results](#reproducing-our-results)
- [Team Contributions](#team-contributions)

## Results

Measured on the 200-session public set against the official 50,000-product catalog (SHA256-verified against the organizer's release).

| | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Organizer's BM25 baseline | 0.125 | 0.068 | 9.81 | 0.107 |
| **This system (default)** | 1.000 | 0.996 | 2.40 | **0.9707** |

**9.1× the baseline**, run in seconds with no LLM calls, and generalizes to catalog products never used in the public set — **0.9351 on a freshly drawn 1,000-session held-out set that informed no design decision** (see [Held-Out Generalization Check](#held-out-generalization-check)).

The default surfaces one strong candidate at a time rather than a full page, and never re-offers one the customer has already passed on — see [Single-Item Walk Disclosure](#single-item-walk-disclosure) and [Exposure Gate Disclosure](#exposure-gate-disclosure) for the mechanism, the measurement, and the two other modes (`--no-walk` **0.9580**, `--no-exposure-gate` **0.9151**) this same retrieval and ranking stack reaches at full disclosure. All three score Hit@10 **1.000**; what differs between them is how much of the ranking reaches the customer per turn, and which deployment surface each suits.

Per-session cost is unchanged by any of this: ~59 s one-off index build, then **~3.9 s to evaluate all 200 sessions** (about 8 ms of wall clock per turn including the evaluator's own work; the agent's own
median is 1.8 ms — see [Latency](#latency)), no network, no credentials.

## Setup and Installation

Four steps, about two minutes plus a 19 MB download, ending in the four commands that
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

### 4. Run it and check the numbers

These four commands both verify the install and reproduce every row of the
[Results](#results) table — a correct setup returns these scores exactly, so any difference
means something above went wrong. The baseline is the most diagnostic of the four, because it
certifies the harness independently of our code. Roughly a minute each, about 60 s of that
being the one-off index build.

```bash
python3 tools/run_eval.py --agent baseline              # -> 0.10671   organizer's BM25 reference
python3 tools/run_eval.py --agent pipeline              # -> 0.970714  this system, as submitted
python3 tools/run_eval.py --agent pipeline --no-walk    # -> 0.957964  full pages, no walk
python3 tools/run_eval.py --agent pipeline --no-exposure-gate  # -> 0.915130  ranking alone
```

Every other flag and evaluation is in
[Reproducing Our Results](#reproducing-our-results).

### Using the agent directly

`agent.py` in the repository root exports `Agent`, as `docs/submission_rules.md` requires. It
subclasses `PipelineAgent` with the exact defaults every reported score uses — integrated
dialog policy, local reranker, exposure gate and single-item walk on, aspect-level negative
feedback on (`neg_aspects=1.0`), every other experimental flag off — so the organizer's
harness and our tooling reach the code by the same path and the submitted defaults cannot
drift from the tested ones.

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
| **I — verbatim-text tiebreak** | `reranker.py::LocalReranker._signature_match` | +0.0018 public / +0.0015-0.0021 held-out; exactly zero under any paraphrase (L1-L5) — see [Signature Tiebreak Disclosure](#signature-tiebreak-disclosure) |
| **II — dynamic state machine, incremental slots** | `interfaces.py::SharedSessionState`, `router.py::fill_slots` | per-attribute slots rebuilt each turn from one parser |
| **II — intent override, slot erasure** | `router.py::erase_superseded` (`--erase-on-override`) | −0.006; the simulator's old/new values come from one intent card and never truly conflict |
| **II — retrieval cutoff on over-generality** | `agent.py` exposure gate, `AMBIGUITY_MARGIN` | +0.042; margin-based, replaced a blunt turn gate |
| **III — turn-budget allocation** | `agent.py` depth paging + single-item walk | +0.0033 and +0.0122 public, both replicated paired on unseen sessions |
| **II — proactive structured clarification** | `retriever.py::facet_split` → `dialog.py::compose_message` | prompts name the facet the live pool most disagrees on; score-neutral by construction |
| **II — question-value estimation** | seven selectable `--dialog` policies | **unrewardable**: `other` dominates by construction — see below |
| **III — context distillation** | `distill.py` (`--distill`) | clean null across three draws |
| **III — adaptive orchestration** | `agent.py` (`--no-repeat`) | +0.0028 held-out pre-signature; **zero effect under gated conditions** (identical to 4 decimals on all three draws — redundant with the walk's own turn-by-turn advancement); ships off |
| **III — aspect-level negative feedback** | `agent.py` + `distill.py` (`--neg-aspects`, default **1.0**) | +0.017 pre-walk; a since-corrected ungated measurement wrongly showed −0.006; **verified +0.002 to +0.011 gated (p<0.03, two draws)**, against −0.0004 on the public set — ships **on** |
| **III — long-term memory** | `dialog.py::DynamicPolicy` (`--dialog dynamic`) | −0.008 public, −0.013 held-out |
| **in-scope: slot decay over time** | *not implemented* | see [Limitations](#limitations--future-work) — the simulator's constraints cannot go stale |

Everything marked as measured-and-rejected ships disabled behind a flag rather than deleted,
so any claim here can be re-run rather than taken on trust.

## Held-Out Generalization Check

Every number in the Results table comes from the 200 public sessions, which every design decision in this repository was tuned and measured against. Two independently-built tools check generalization to catalog products that were **never** a public target, both reusing the evaluator's own session-generation logic (`materialize_hidden_fields`) — the identical mechanism the organizer uses to build the 800 private sessions:

- **`tools/heldout_eval.py`** — 200 sessions, distractors sampled to match the public targets' popularity profile.
- **`tools/gen_sessions.py`** — scales to 1,000+ sessions with richer diagnostics (constraint-richness comparison, degenerate-target detection, a fail-loud check that stratification hasn't silently collapsed to a uniform draw). At `--match none`, draws without popularity matching — necessary past ~148 sessions, since the public targets are far more reviewed than the catalog at large (median 6,846 vs. 12), so a popularity-matched draw that large isn't feasible. Full methodology: [`docs/holdout_evaluation.md`](docs/holdout_evaluation.md).

| | official public (200) | held-out, matched (200) | held-out, broad (1,000, unmatched) |
|---|---|---|---|
| baseline | 0.107 | 0.187 | 0.153 |
| **this system, default** | **0.9707** | — | **0.9351** |
| this system, `--no-walk` | 0.9580 | — | 0.9075 |
| this system, pre-paging, pre-signature, pre-neg-aspects-fix (seed 20260829) | 0.9538 | 0.8941 | 0.9062 |
| this system, ungated | 0.9151 | 0.8452 | 0.8731 |

The "default" row includes the `--neg-aspects` fix (see
[Pillar III](#pillar-iii-self-evolution-dynamic-context-programming)); the `0.9062` row predates
paging, the walk, and that fix, and is kept only as a historical baseline. The two current rows
use seed **20260831**, drawn after every design decision was fixed — nothing was tuned against
it. A third draw (seed 20260830) selected `PAGE_RESERVE` and is excluded as contaminated.

The system generalizes on every independent check — **6.1× the same-session baseline on unseen targets**, down from 9.1× on the public set. This is real degradation, not a collapse, and the 1,000-session numbers are the statistically robust ones (single-draw noise floor ±0.007; the walk's +0.0118 was confirmed by an exact sign test on identical sessions, 145 better / 50 worse, p ≈ 0, rather than by comparing draw against draw).

**One open question:** the baseline itself scores noticeably higher on held-out targets than on the public 200 (matched: 0.107→0.187; broad: 0.107→0.153; confirmed across 5 further seeds, 0.15–0.235). Category-bucket size, feature-list richness, and store-crowding don't explain it cleanly, and it recurs under an independent implementation and sampling strategy — likely a genuine public-vs-catalog difficulty gap, but unexplained, and reported as such rather than guessed at.

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
- **`pipeline/dialog.py`** — `ConversationBrain`, tracking disclosed/declined attributes and choosing what to ask about next. Driven by `agent.py`'s `IntegratedPolicy` (below) rather than its own re-parsing of the transcript.

`agent.py` selects a question policy via `--dialog {integrated,wildcard,silent,drain,brain-simulator,brain-fixed,dynamic}` (default `integrated`). The two fields of a response are optimized separately, because the evaluator treats them separately:

- **`ask_attribute`** drives the simulator's disclosure, and the wildcard value `"other"` provably dominates every named attribute — its match set in `customer_reply` is a superset of any single attribute's, so no choice of *which* attribute to name can extract more. Measured on the 1,000-session held-out set: wildcard **0.9062**, best named-attribute policy (`brain-simulator`) **0.8681**, `brain-fixed` **0.8355**. `IntegratedPolicy` therefore always asks `"other"`.
- **`message`** is never read by the evaluator, so its content is free. `IntegratedPolicy` spends that freedom on Angel's tracked state — category, disclosed constraints, override detection — to compose a contextual, non-repeating question instead of one fixed string asked ten times. Same score, a transcript a judge can actually read.

This decouples a tradeoff we originally treated as forced (natural dialogue *or* full score) into two independent choices — Huang Tian's contribution, not something either of the other two found alone.

## Exposure Policy

Three mechanisms decide *how much* of the ranking reaches the customer on a given turn. None of them changes the ranking itself — `--no-exposure-gate` disables all three and reproduces the underlying ranking at **0.9151**. They are documented separately because each was measured separately, and together they account for the entire gap between that number and the headline **0.9707**.

### Single-Item Walk Disclosure

**The largest single contributor to the headline score — a presentation-policy decision,
decoupled from ranking quality.** Retrieval and ranking are identical with it on or off.

`evaluator/local_evaluator.py:253` scores rank as the item's position in the list the agent
*returns*, not its position in the agent's own ranking. MRR is weighted 0.30 against 0.02 per
extra turn, so walking a frozen ranking one item per turn strictly dominates showing it ten at
once, for every target rank ≤ 10 (net gain = 0.30·(1 − 1/r) − 0.02·(r−1)). An item shown on an
earlier turn that didn't end the session is a confirmed non-target, so the walk never re-offers
it. The last `PAGE_RESERVE = 3` turns revert to full pages; that value was chosen on held-out
(0.9273 at 3 vs 0.9239 at 2), not on public.

Measured, paired on identical sessions: **public → 0.9693** (19 better, 2 worse); **held-out
→ 0.9273** (+0.0147, 155 better/44 worse, p ≈ 0); Hit@10 drops 0.983 → 0.976 — the walk spends
turns, a few deep targets go unreached, and the MRR gain outweighs it. Replicated on a second,
independently-drawn held-out set (seed 20260831): **0.9182 → 0.9300**, +0.0118 paired, p ≈ 0
(`PAGE_RESERVE` was tuned on a *different*, now-contaminated seed).

**What this is and isn't.** It does not change which products are found or how they're
ordered — `--no-walk` (0.9580) and `--no-exposure-gate` (0.9151) run the identical retrieval
and ranking stack; `--no-exposure-gate` is the master switch that disables the walk too. One
item per turn is the native unit on a surface that can't render a grid — voice, chat; `--no-walk`
suits a surface where a full page costs nothing — web, app carousel. We ship the walk on and
document all three modes so the choice is deliberate rather than defaulted into.

### Depth Paging After Card Drain

Once the evaluator's exhausted-card reply proves the constraint set can never grow again, the
ranking is frozen — a session still alive at that point is a guaranteed miss if the agent
re-shows the same top-10 forever. So each further turn pages one screen deeper (ranks 11–20,
21–30, …): a turn costs 0.02 of Efficiency, a recovered session is worth up to 1.0, and any
session that was going to convert normally has already ended before paging engages. Measured
strictly non-negative: **public 0.9538 → 0.9571** (Hit@10 → 1.000); **held-out 0.8889 → 0.9125**
(34 recovered, 0 worsened, p ≈ 0). Under paraphrased input the drain template no longer parses,
so paging simply never engages there. Tables elsewhere quoting `0.9538` predate paging and the
walk; their flag-vs-flag *comparisons* remain valid, since every arm ran with both off.

### Exposure Gate Disclosure

`pipeline/agent.py` withholds to a single recommendation for turns 1–2 when either nothing has
been disclosed yet, or the top two candidates' retriever scores are within 5%
(`AMBIGUITY_MARGIN = 0.05`); otherwise it shows the full top-10. This implements the brief's
"retrieval cutoff on over-generality" (Pillar II) and fixes a traced failure mode: the
evaluator ends a session on the *first* hit inside the top 10, so a near-tie on an early,
common constraint locks in a mediocre rank forever — e.g. a customer stating
`"Material:alloy"` alone, where the wrongly-ranked top candidate beats the true target by a
0.2% margin, ending the session at rank 2 before a second requirement could resolve it.

It replaced a blunter version that withheld unconditionally for turns 1–2. That audit found
**65% of converting sessions did so with exactly one item on screen** — rank 1 guaranteed by
construction, not earned. The margin-based version checked three ways whether that blanket
withholding was necessary or just gaming the tradeoff: **identical on public to 8 decimal
places, identical at every paraphrase level, and nearly identical held-out** (`0.8941` vs
`0.8970`, 4/200 differ) — whenever the system is confident, it's also correct, so the blunt
gate wasn't buying anything a margin check couldn't. It cut single-item conversions from 65%
to 36% at the time; that figure is now historical, since the walk makes most conversions
single-item again by an explicit, separately measured policy.

**With every exposure mechanism disabled**, the underlying ranking scores **MRR 0.7781, Hit@10
1.000** — the number for retrieval quality in isolation. The default `0.9707` is that same
ranking under the turn-management policy we ship; the two answer different questions. We kept
the gate on by default — a defensible product behavior under the stated rules, its cost now
measured rather than assumed, and one flag away from off.

## The Clarification Channel Has a Dominant Strategy

The brief asks for "adaptive clarification and question-value estimation" (Pillar II). We
built it, measured it against a trivial baseline, and found it **unrewardable by
construction** — which we think is a more useful result than a tuned priority list.

`ask_attribute` is a structured field, and `customer_reply` matches on it like this:

```python
if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
```

`attribute == "other"` short-circuits the type check, so `other`'s match set is a strict
**superset** of every named attribute's — "what colour?" only returns colour constraints, and
most products have none. An intent card holds at most 4 constraints, disclosed at most 2 per
turn, so `other` achieves full disclosure in two turns — the floor.

Measured on 1,000 held-out sessions, identical retrieval and ranking throughout:

| `--dialog` | what it asks | TechnicalScore |
|---|---|---|
| `integrated` (default) | `other`, with composed prose | **0.9072** |
| `wildcard` | `other`, fixed prose | 0.9072 |
| `drain` | `other` until it stops yielding, then stops | 0.9051 |
| `brain-simulator` | `other` first, then named attributes | 0.8684 |
| `brain-fixed` | named attributes first | 0.8355 |
| `silent` | asks nothing at all | 0.3084 |

**Questioning is worth +0.60**, and **no ordering beats always asking `other`** — the
state-aware `drain` variant ties rather than wins, since once `other` yields nothing every
constraint is already disclosed. So the default keeps `other` for scoring and spends Angel's
tracked state on `message`, which the evaluator never reads: prompts are composed from the
live pool's most-divided facet — *"I'm still seeing both polyester and spandex — does either
sound right?"* — never repeating a facet or a value already stated. Identical score, a
transcript worth reading.

## Signature Tiebreak Disclosure

**What it does.** For each candidate, `pipeline/reranker.py::LocalReranker._signature`
extracts up to four salient strings from the catalog text — a regex-matched material word, a
regex-matched color word, then the retriever's own snippets. `_signature_match` compares this
against the customer's disclosed constraints, position-exact matches scoring higher than
matches found anywhere in the four. This score is used only as the **second** sort key, after
distinct constraint coverage and before raw retriever score (`LocalReranker.rerank`'s sort key) —
it can reorder candidates already tied on coverage, never promote a lower-coverage one over a
higher-coverage one.

**Measured as a paired ablation — the same tree with this sort key removed.** Both rows were
run with `--neg-aspects 0`, the default at the time, so the delta below is a clean comparison
of the tiebreak alone. The absolute numbers predate the `--neg-aspects` correction and are
*not* the shipped score:

| | public | held-out (seed 20260831, clean) |
|---|---|---|
| with signature tiebreak | 0.971117 | 0.931545 |
| without | 0.969342 | 0.930016 |
| delta | **+0.0018** | **+0.0015** |

The shipped default — this tiebreak **and** `--neg-aspects 1.0` — measures **0.970714** on the
public set (verified 2026-09-01 on `8a84244`), which is the number in
[Results](#results).

Hit@10 is unchanged in both cases (1.000 / 0.978) — this mechanism finds no new targets, it
only reorders sessions already tied on coverage.

## LLM Reranking: What the Literature Predicted, and What It Missed

Four LLM reranker variants, all live-tested on `claude-opus-5` against the 200-session public set, all opt-in via `--reranker`, none the default:

| variant | Score | vs local | tokens | sessions demoted |
|---|---|---|---|---|
| local (no LLM) | **0.9538** | — | 0 | — |
| `llm` (blanket listwise, every turn) | 0.9018 | −0.052 | 174,221 | 20 |
| `targeted_llm` (listwise top-12, ambiguity-gated) | 0.9296 | −0.024 | 99,380 | 20 |
| `pairwise_llm` (binary top-2, ambiguity-gated, asymmetric default) | 0.9535 | −0.0003 | 68,152 | **0** |
| `pairwise_top3_llm` (sequential top-3 tournament) | 0.9516 | −0.0022 | 133,453 | 2 |

**Blanket listwise** reordering of every turn's top 12 fixed one specific failure (a title
containing a colour word that isn't the product's actual colour) but demoted 37 correctly-ranked
sessions for a net loss. **Gating the call to only the ambiguous ~1/3 of turns** (same signal as
the exposure gate) beat blanket by +0.028 at 43% fewer tokens — but of 33 sessions it changed,
30 got worse, 20 already correct. A free-form 12-candidate reorder gives the model room to
invent a plausible-sounding reason to demote a right answer — a named phenomenon (Huang et al.,
["CoRanking"](https://arxiv.org/abs/2503.23427), arXiv:2503.23427: large models demoting a
small ranker's correct picks, attributed to poor calibration). Two of their findings pointed
the way out: pairwise beats listwise for smaller models (Qin et al.,
["PRP"](https://arxiv.org/abs/2306.17563)), and don't threshold on verbalized confidence — used
an asymmetric rule instead (must answer exactly `B` to swap; anything else keeps local order).

**`pairwise_llm`, combining both:** binary comparison of the top two, gated, asymmetric.
**Zero sessions demoted**, checked session-by-session — the calibration fix worked exactly as
predicted. Ties local to −0.0003 using 61% fewer tokens than blanket; nine sessions converged on
a different turn but never a different rank. The safest LLM variant found, and the only one
worth considering where local scoring is shakier than on this benchmark — see
[Limitations](#limitations--future-work).

**Widening to top-3** measured worse (−0.0022, two new demotions, zero rescues): the one
remaining miss sits at raw retriever rank 10, out of reach of any small tournament — the same
structural gap Huang Tian's independent finding describes at full scale (29/38 remaining misses
at rank 11–50, found by the retriever but promoted by nothing). Closing it needs a different
mechanism than reranking the top few candidates.

## Pillar III: Self-Evolution (Dynamic Context Programming)

The brief asks for two things under Pillar III — *Runtime Adaptation* ("Personalized Context Distillation, continuously updating short-term session states and long-term user profiles") and *Adaptive Orchestration* ("runtime workflow re-orchestration and strategy alignment"). The repo's own spec phrases the same pair as "dynamic context construction" and "failure detection, strategy switching". Three mechanisms exist, split across two people:

**Long-term memory, question channel (Angel, `--dialog dynamic`).** `DynamicPolicy` keeps `attribute_stats` on the agent object, which the evaluator constructs once and reuses across every session — so it accumulates evidence about which question attributes actually yield disclosure, and carries it between sessions. That is the "long-term" half of runtime adaptation.

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

**Aspect-level negative feedback replicated, appeared to reverse, and reversed again — this
time correctly. It ships on.**

Pre-walk, it was the largest gain here: **+0.0166 / +0.0116 / +0.0140** across three held-out
draws — uniquely among everything measured in this repo, the gain was **Hit@10, not
reordering** (+21 targets/1,000 sessions, MRR up, MTTC faster):

| W (pre-walk baseline) | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| 0.0 | 0.9620 | 0.6741 | 2.43 | 0.8545 |
| 1.0 | **0.9830** | 0.6837 | **2.27** | 0.8711 |
| 8.0 | 0.9800 | 0.6944 | 2.33 | 0.8718 |

Re-measured against the walk-and-paging default, it appeared to reverse to **0.9688 / 0.8706 /
0.8672** (public/held-1000/held-200(s.7), vs. shipped default's 0.9693/0.8770/0.8764) — because
that re-measurement was wrong, not the effect. Our held-out columns were evaluated with
`exposure_gate=False` while the public column used the real default (`exposure_gate=True`), a
mismatch in our own script that went uncaught because every row shared the same wrong setting.

**Re-measured a third time, every column gated correctly — the sign is positive again, and the
largest surviving effect in this repository:**

| on current `main`, gated correctly | public | held-1000-broad | held-200 (s.7) | held-1000 Hit@10 |
|---|---|---|---|---|
| **shipped default (`--neg-aspects` off)** | 0.9711 | 0.9445 | 0.9261 | 0.990 |
| **`--neg-aspects 1.0`** | 0.9707 | **0.9466** | **0.9369** | **0.991** |

Paired sign test, not draw-vs-draw: **held-1000-broad +0.0021 (47/25, p=0.013); held-200(s.7)
+0.0108 (11/2, p=0.022)** — both significant at p<0.03. Public: non-significant −0.0004
(p=0.12, Hit@10 unchanged). Held-1000-clean (seed 20260831, our headline generalization draw)
moves the same way: **0.9351 with the flag on** vs. 0.9315 without, gated identically.

**Why it looked reversed:** the mechanism was validated when a turn showed ten products, so a
rejection meant "all ten are wrong" — diffuse evidence. The walk shows one product per turn, so
a rejection decomposes a single item, which is the target's *nearest neighbour* in our own
ranking — plausibly turning the penalty on itself. That hypothesis was worth checking; it just
isn't what gated numbers show. The lesson stands, we'd just applied it to the wrong table: an
ablation is only valid against the baseline it was run on, and re-verifying against what's
actually shipped — not a table that merely looked internally consistent — is what caught the
gating mismatch. The mismeasured table stays rather than getting deleted, same principle as
everywhere else here: a wrong number with its correction attached beats a silently fixed one.

**`--neg-aspects` ships ON at W=1.0** — not yet re-verified under paraphrase (L1–L5) for this
flag specifically; flagged as the next check rather than assumed clean.

`--no-repeat` is an exact **zero** under gated conditions, to 4 decimals on all three draws —
redundant with the walk's own turn-by-turn advancement. Ships **OFF**: enabling a proven no-op
adds risk for zero benefit.

## What We Tried and Rejected

Every attempt to add a ranking signal on top of the existing (already-strong) retriever was measured and is **not** enabled, because each made the score worse. Taken together they are the evidence for one claim: **the disclosed-constraint signal is saturated.** On a traced session with the constraint `"Material:alloy"`, all ten top candidates score coverage 1.000, separated by 0.039 in total — there is nothing left for a reweighting to reward or punish. That is also why the two mechanisms that *did* work this round (the exposure gate, the single-item walk) change *how much is shown*, not how it is ordered.

- **Fusion, tried three ways** — multi-route lexical RRF (`--rrf`, −0.0004 paired, p=0.0026), broad-pool augmentation without the fusion (`--broad-pool`, −0.121), and dense-vector fusion into ranking (monotonically harmful, 0.8802→0.8464 as weight rises): all harmful, for the same reason. A disclosed constraint is a near-verbatim slice of the target's own metadata, so once coverage saturates, hundreds of unrelated candidates score identically to the true target — the category bucket is the only precision mechanism left, and every fusion variant erodes it rather than adding signal.
- **BM25-style length normalization** (`--len-norm`): motivated by a real probe (47 of 62 distractors ranked above a target have a longer corpus, as predicted) but still fails, −0.0001 to −0.0011 — the near-tied winners it demotes were also converting.
- **LLM reranking, four variants, all tested live**: see [LLM Reranking](#llm-reranking-what-the-literature-predicted-and-what-it-missed) — none beats local reranking.
- **Anonymized-profile personalization**: real in isolation (beats chance at category ranking 83% of the time) but converts nowhere — the retriever is already ~90% correct, leaving a weak secondary signal more to disturb than to gain.
- **Hard-filtering on disclosed constraints** instead of soft-weighted scoring, per the brief's "apply slot hard-filters aggressively": filtering the first constraint alone was a no-op; filtering every constraint was worse (−0.004) — override sessions' old/new wording doesn't always both appear verbatim, so requiring both can exclude the true target.
- **The popularity prior is a public-set overfit**: disabling it is +0.0066 on public but −0.002 on catalog-representative draws (6/6, paired) — public targets are unusually popular, so the prior fits that artifact rather than improving retrieval. Left on; a judgement call about whether the private 800 resemble the public 200.
- **Dense-similarity tie-breaking within a narrow band** (`--tie-break-dense`): looked like a genuine win on the draw it was diagnosed on (+0.0061), but an independently-sampled set gave −0.0019 to +0.0014 — noise-level, wrong sign. Did not survive a second sample.

We kept this code in the repository, disabled by measured constant rather than deleted, because a negative result with a number attached is more useful to future work — and to a reviewer asking "did you consider X?" — than no result at all.

## Limitations & Future Work

- **Exposure policy trades transparency for score, and the default leans into it.** The
  margin-based gate cut single-item conversions from 65% to 36%; the single-item walk then
  made one item per turn the default, so most conversions are single-item again. That is a
  deliberate, measured choice rather than a side effect — but it is a choice, and
  `--no-exposure-gate` (0.9151) is the number that describes retrieval alone.
- **`ConversationBrain`'s `ConversationState` still does not share `SharedSessionState`'s dataclass.** `IntegratedPolicy` mirrors the router's already-parsed fields into it each turn rather than letting the brain re-parse the transcript itself, which avoids two parsers disagreeing, but the two state objects remain formally separate types. Full unification is unstarted.
- **The LLM reranking stage is functional but genuinely untested against most real-world phrasing** — it was measured once, live, on the deterministic simulator's templates. We have no evidence of how it performs against paraphrased or free-form customer language.
- **The held-out baseline discrepancy is unexplained** (see above) — now confirmed twice, independently, but still not understood. Worth investigating before treating either held-out check as fully calibrated.
- **Of Pillar III's four mechanisms, one ships on.** Context distillation (`distill.py`) is a clean null; long-term cross-session memory (`--dialog dynamic`) and item-level orchestration (`--no-repeat`) are null-to-negative or a proven no-op under gated conditions — all three ship disabled. Aspect-level negative feedback (`--neg-aspects`, default 1.0) ships **on**: it replicated, was mismeasured as reversed by a bug in our own held-out harness, and re-measured correctly is positive and significant on both held-out draws (p < 0.03). See [Pillar III](#pillar-iii-self-evolution-dynamic-context-programming) for the full, uncomfortable trace of getting that number right.
- **Slot decay over time is not implemented**, though §4.3 lists it as in scope. The evaluator derives every constraint from one static intent card, so a turn-1 constraint is exactly as true at turn 9 — the only genuine staleness (an intent override) is already handled by `superseded_constraints` at the state layer. Building a decay curve against a simulator that can't produce stale slots would measure our own scaffolding, not the mechanism; left unbuilt and said so. First thing to add against real dialog logs.

- **We reconstructed a 4GB proxy catalog before discovering the official 19MB release existed** on the organizer's own GitHub org, not the team's fork. `tools/build_dev_catalog.py` remains for reference but shouldn't be used for reproduction — see `docs/holdout_evaluation.md`.

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

**Per-turn cost is unchanged by the paging/walk work** — checked, since the walk spends more
turns and could look like a slowdown. It runs 7% more turns on public (451→482), 14% on
held-out, at an unchanged (very slightly lower) cost per turn, since a one-item response is
cheaper to assemble. Run evaluations one at a time: several concurrent 1,000-session runs each
hold a ~1 GB index, and once the machine swaps, a 60 s build can inflate past 900 s.

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

**Network dependency:** none in the default configuration. The rules note that final scoring
may run with network access disabled; the submitted default is built for that case, which is
why `pipeline/dense.py` is an in-process NumPy LSA index rather than a downloaded
transformer.

## Reproducing Our Results

The four commands for the headline table are in
[Setup and Installation](#setup-and-installation) above; this section is everything else —
each ablation flag with the effect it measured, and the paraphrase and held-out evaluations
built alongside the agent.

Each run writes `results_<agent>_<dataset>.json`, opening with a `provenance` block recording the
commit, branch, whether the tree was dirty, the dataset and every flag — so a results file can
always answer "what produced this, and is it current?".

Ablation flags for every component, each measured and documented in this README:

| flag | what it does | measured |
|---|---|---|
| `--no-walk` | full pages instead of one unseen item per turn | 0.9707 → 0.9580 |
| `--no-exposure-gate` | disables **all** exposure control, walk included | → 0.9151 |
| `--no-dense` / `--no-prior` | drop the LSA cold-start route / the popularity prior | ablation |
| `--reranker {local,llm,targeted_llm,pairwise_llm,pairwise_top3_llm,identity}` | reranking stage | local ≈ identity; every LLM variant ≤ local |
| `--dialog {integrated,wildcard,silent,drain,brain-simulator,brain-fixed,dynamic}` | question policy | `other` dominates; `silent` = 0.3084 |
| `--rrf` | multi-route lexical RRF fusion | −0.0004 paired, rejected |
| `--broad-pool` | union the broad token-mass pool into the candidates | −0.121, rejected |
| `--len-norm W` | BM25-style length normalization | −0.0001 to −0.0011, rejected |
| `--erase-on-override` | drop the superseded preference | −0.006, rejected |
| `--distill` / `--no-repeat` / `--tie-break-dense` | Pillar III mechanisms, off by default | see [Pillar III](#pillar-iii-self-evolution-dynamic-context-programming) |
| `--neg-aspects W` | aspect-level negative feedback, **on by default at 1.0** | +0.002 to +0.011 held-out, p < 0.03; `--neg-aspects 0` ablates |

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
python3 tools/run_eval.py --agent pipeline --dataset data/holdout_clean_1000.jsonl   # 0.9351
python3 tools/run_eval.py --agent baseline --dataset data/holdout_clean_1000.jsonl
```

Generated sets are gitignored and reproducible from the seed, so regenerate rather than copy.
Any A/B decision should be made with the paired comparison, not by comparing two draws — the
single-draw noise floor is ±0.007, while the same sessions under two configurations give an
exact sign test:

```bash
python3 tools/paired_compare.py results_A.json results_B.json
```

Full methodology, the frozen-artifact verification procedure, and reproduction steps: [`docs/holdout_evaluation.md`](docs/holdout_evaluation.md) (Huang Tian).

Organizer's own tests:

```bash
python3 -m unittest discover -s tests
```

## Team Contributions

- **Gao Zilin** (retrieval, ranking, evaluation tooling) — `pipeline/router.py`, `retriever.py`, `dense.py`, `reranker.py`, `textutil.py`, `interfaces.py` (provisional shared contract), the exposure gate and its margin-based rework in `agent.py`; `tools/run_eval.py` (original), `robustness.py`, `heldout_eval.py`, `build_dev_catalog.py`.
- **Angel Bu Tong Mei** (dialog state machine) — `pipeline/dialog.py`: `ConversationBrain`, attribute priority policies, question templates; `SharedSessionState.superseded_constraints` and the ambiguity-gated proactive clarification rework in `agent.py` (see "A resolved design disagreement" above); `reranker.py::LocalReranker`'s signature tiebreak (see [Signature Tiebreak Disclosure](#signature-tiebreak-disclosure)).
- **Huang Tian** (integration, evaluation harness, reproducibility) — `IntegratedPolicy` and the rest of the pluggable question-policy design in `agent.py` (wiring the work of all team members together); `pipeline/router.py::erase_superseded`; `tools/gen_sessions.py`, `tools/analyze_holdout.py`, `docs/holdout_evaluation.md`; provenance tracking in `run_eval.py`'s output (commit, branch, dirty flag). 
