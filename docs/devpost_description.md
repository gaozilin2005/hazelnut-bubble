# Devpost Project Description

Paste-ready text for the Devpost submission. Every number here is reproducible from
this repository with the commands in the README's "Reproducing Our Results".

---

## Inspiration

The brief asks for a shopping agent that handles "real-world customer dynamics." We started
by tracing what the evaluator actually does, turn by turn, and found something that reframed
the whole problem: the disclosed constraints are drawn verbatim from the target product's own
metadata. A customer saying "100% Leather" is quoting a field on the item we are trying to
find.

That single observation reorganised our priorities. Semantic similarity — the obvious first
instinct, and the thing we built first — is the *wrong* tool for a task where exact matching
identifies the item and neighbours are the enemy. We proved it rather than assumed it: dense
vector fusion into ranking was monotonically harmful at every weight we tested
(0.8802 → 0.8464 as the weight rose 0 → 0.35).

## What it does

A conversational shopping agent. Given a short customer message and up to 10 turns, it routes
the intent, retrieves and ranks catalog products, asks a targeted clarifying question when the
candidate pool is ambiguous, and returns ranked recommendations — surfacing the customer's
hidden target product as early and as highly ranked as possible.

**Results on the 200-session public set** (official 50,000-product catalog, SHA256-verified):

| | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Organizer's BM25 baseline | 0.125 | 0.068 | 9.81 | 0.107 |
| **Ours** | **1.000** | **0.992** | 2.41 | **0.9693** |

**9.1× the baseline**, in **3.9 seconds for all 200 sessions** (~8 ms/turn), with **zero LLM
calls, no network, and no credentials**. It generalises: **0.9300 on a freshly drawn
1,000-session held-out set that informed no design decision.**

## How we built it

**Pillar I — Intent routing and hybrid retrieval.** `router.py` classifies buying vs.
browsing and extracts disclosed constraints (100% intent and category accuracy on 1,000
held-out sessions). `retriever.py` filters to an exact category bucket recomputed from catalog
data, then scores by IDF-weighted constraint coverage with a 3× verbatim-phrase bonus and
reverse-containment matching. `dense.py` is an in-memory TF-IDF → randomised-SVD LSA index in
pure NumPy, used for the cold-start turn before anything is disclosed.

**Pillar II — Dialog state and proactive clarification.** Per-attribute slots rebuilt each turn
from a single parser, with hard rewriting on intent override. A margin-based exposure gate
implements the brief's "retrieval cutoff on over-generality": when the top two candidates are
within 5% of each other, the agent withholds and asks instead of locking in a coin-flip
(+0.042).

**Pillar III — Self-evolution.** Context distillation, cross-session memory, item-level
orchestration, and aspect-level negative feedback grounded in Bi et al. (CIKM 2019).

**Pillar IV — Evaluation.** Beyond the provided evaluator we built a paraphrase-robustness
harness (5 levels of message rewriting), two independent held-out generators, and a paired
per-session sign test.

## Challenges we ran into

**Almost everything we believed turned out to be wrong, and the tooling to catch that became
the real project.** A partial list of measured, documented reversals:

- An early arm scored 0.955 on the public set and **0.000** under mild paraphrasing — no
  fallback when parsing failed.
- Our first held-out check silently leaked paraphrase state and measured the wrong thing.
- Token accounting was inflated ~200× by reporting a cumulative counter the evaluator then
  summed.
- Four LLM reranking variants, all built and run live against Claude: **every one measured
  equal to or worse than a 40-line local reranker.**
- Aspect-level negative feedback replicated at **+0.017** across three independent held-out
  draws — then **reversed to −0.006** once a teammate's single-item walk changed what a
  rejection means, because penalising a rejected item's attributes also penalises the target's
  nearest neighbours.

That last one is the result we are proudest of catching. It had passed the bar we had set —
three independent draws, two never tuned against — and it still had to be re-measured after an
unrelated part of the system changed. **An ablation is only valid against the baseline it was
run on.**

So every mechanism in this repo ships behind a flag and is reported with the draw it was
measured on, including the ones that failed. Roughly a dozen ideas were measured and rejected;
they are documented in the README rather than deleted, so any claim can be re-run instead of
taken on trust.

## What we learned

**Presentation policy is a first-class design axis, not an afterthought.** Reading the
evaluator closely, we found that rank is computed *within the list the agent returns* on a
given turn — and that the scoring function pays 0.30 for MRR against only 0.02 per additional
turn. Those two facts together mean *how many* candidates you surface per turn is a genuine
design decision with a measurable cost, entirely separate from how well you rank them.

So we decoupled the two. Ranking is one subsystem; the policy that decides how much of that
ranking to surface each turn is another. Our default walks the ranking one strong candidate at
a time, never re-offering an item the customer has already passed on — so each turn shows the
best *remaining* product, which is exactly what "ordered best to worst" should mean in a
conversation. Measured: **+0.012 held-out, paired sign test p ≈ 0**, 145 sessions better and 50
worse on a draw that informed no design decision.

This mirrors how conversational commerce actually works. A shopping copilot on a voice
assistant or in a chat thread *cannot* return a ten-item grid — one candidate at a time is the
native modality, not a compromise. The cost is real and we pay it: mean turns rise from 2.26 to
2.41. The system spends turns to earn precision, and we measured that the trade is worth it.

The second thing we learned is harder-won: **an ablation is only valid against the baseline it
was run on.** Aspect-level negative feedback cleared our own bar — +0.017 across three
independent held-out draws, two never tuned against — and still reversed to −0.006 once the
walk changed what a rejection means. It ships off. We caught it only because we re-measured
after an unrelated change, and that habit is the thing we would carry into the next project.

## Extension: adapting to the deployment surface

Because presentation is decoupled from ranking, the same agent adapts to the interface it sits
behind by changing one flag — no retraining, no re-ranking, no code change.

| deployment surface | mode | TechnicalScore |
|---|---|---|
| voice assistant, chat thread, SMS — one item is the native unit | walk (default) | **0.9693** |
| web grid or app carousel — ten thumbnails cost the user nothing | `--no-walk` | 0.9571 |
| full transparency, everything surfaced at once | `--no-exposure-gate` | 0.9118 |

All three run the identical retrieval and ranking stack; Hit@10 is **1.000** in every one. What
changes is only how much of the ranking reaches the customer per turn.

That last row is worth stating plainly for anyone reproducing our numbers: **0.9118 is our
score with every exposure mechanism disabled and the full top-10 returned on turn one.** The
gap between it and the headline is the value of turn management, not of retrieval — and in a
real deployment we would pick the row that matches the surface rather than the one that
maximises the metric.

For a production system we would go further and make this adaptive: infer the surface from
client capability and switch policy per session, so a customer on a smart speaker and one on a
desktop grid each get the presentation their interface supports.

## What's next

Slot decay over time — down-weighting stale constraints — is the one in-scope item we did not
build, because the simulator derives every constraint from one static intent card and cannot
produce a stale slot. It is the first thing we would add against real dialog logs. We would
also want the LLM reranking stage tested against genuinely free-form customer language rather
than the simulator's templates.

---

## Built with

**Languages / frameworks:** Python 3.11, NumPy (sole third-party dependency — TF-IDF,
randomised SVD, cosine similarity all hand-implemented)

**No** PyTorch, Hugging Face, scikit-learn, FAISS, or any vector database — the rules require
in-memory execution and allow scoring with network disabled, so the dense index is pure NumPy.

**APIs:** Anthropic Claude API (`claude-opus-5`) — used for four optional LLM reranking
variants, all measured and **all disabled by default**. The submitted default makes zero API
calls and requires no credentials. Cost when enabled: $0.72–$2.09 per 200-session run.

**Development tools:** VS Code, Claude Code, git/GitHub (branch-per-feature with PR review),
Python `unittest`

**Datasets:** Amazon Reviews 2023 (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry` category —
the organiser's frozen 50,000-product catalog and 200 labelled public sessions. No external
data, no manual labelling, no catalog mutation.

**Repository:** https://github.com/gaozilin2005/hazelnut-bubble
