# Devpost Project Description

Paste-ready text for the Devpost submission. Every number here is reproducible from
this repository with the commands in the README's "Setup and Installation"
and "Reproducing Our Results".

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
| **Ours** | **1.000** | **0.996** | 2.40 | **0.9707** |

**9.1× the baseline**, in **3.9 seconds for all 200 sessions** (~8 ms of wall clock per turn including the
evaluator; the agent's own median is 1.8 ms), with **zero LLM
calls, no network, and no credentials**. It generalises: **0.9351 on a freshly drawn
1,000-session held-out set that informed no design decision.**

## How we built it

**Pillar I — Intent routing and hybrid retrieval.** `router.py` classifies buying vs.
browsing and extracts disclosed constraints (100% intent and category accuracy on 1,000
held-out sessions). `retriever.py` filters to an exact category bucket recomputed from catalog
data, then scores by IDF-weighted constraint coverage with a 3× verbatim-phrase bonus and
reverse-containment matching. `dense.py` is an in-memory TF-IDF → randomised-SVD LSA index in
pure NumPy, used for the cold-start turn before anything is disclosed.

**Pillar II — Dialog state and proactive clarification.** Per-attribute slots rebuilt each turn
from a single parser. On intent override the superseded value is marked inactive for the
dialog but kept for retrieval (`superseded_constraints`); erasing it outright is a flag,
measured −0.006 and shipped off. A margin-based exposure gate
implements the brief's "retrieval cutoff on over-generality": when the top two candidates are
within 5% of each other, the agent withholds and asks instead of locking in a coin-flip
(+0.042).

**Pillar III — Self-evolution.** Context distillation, cross-session memory, item-level
orchestration, and aspect-level negative feedback grounded in Bi et al. (CIKM 2019).

**Beyond the brief — evaluation.** We built a paraphrase-robustness harness (six levels
across two independent axes — payload and category), two independent held-out generators, and
a paired per-session sign test.

## Challenges we ran into

**Almost everything we believed turned out to be wrong, and the tooling to catch that became
the real project.** A partial list of measured, documented reversals:

- An early arm scored 0.955 on the public set and **0.000** under mild paraphrasing — no
  fallback when parsing failed.
- Our first held-out check silently leaked paraphrase state and measured the wrong thing.
- Token accounting was inflated ~270× (43.2M tokens reported against ~160K actual) by
  reporting a cumulative counter the evaluator then summed again each turn.
- Four LLM reranking variants, all built and run live against Claude: **every one measured
  equal to or worse than a 40-line local reranker.**
- A teammate's late addition to the reranker described itself, in its own code comment, as reconstructing our evaluator's hidden answer-key logic. It doesn't — it pattern-matches catalog text, and measured effect is small and vanishes under any paraphrasing — but the framing alone was worth catching and rewriting before submission; a claim like that shouldn't survive on the strength of nobody reading the docstring closely.
- Aspect-level negative feedback replicated at **+0.017** across three independent held-out
  draws, then appeared to reverse to −0.006 once a teammate's single-item walk changed what a
  rejection means — and that reversal turned out to be wrong too: our own held-out harness was
  evaluating it with the exposure gate off while the public number used it on, a mismatch that
  looked internally consistent because every row in the table used the same wrong setting.
  Measured correctly, gated throughout, it is positive again and statistically significant
  (p < 0.03 on two held-out draws) — and it ships on.

That is the result we are proudest of catching, twice over. It had passed our own bar — three
independent draws, two never tuned against — and still needed re-verification after an
unrelated part of the system changed. The second time, it needed re-verification against our
*own measurement code*, which is a harder thing to doubt than a teammate's mechanism. **An
ablation is only valid against the baseline it was actually run on** — including when the
baseline itself is the thing that's wrong.

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

The second thing we learned is harder-won, and it took two rounds to actually learn it. Our
rule was **an ablation is only valid against the baseline it was run on** — so when aspect-level
negative feedback (+0.017 across three independent held-out draws) appeared to reverse to
−0.006 once the walk changed what a rejection means, we treated that as the rule working:
caught by re-measuring after an unrelated change, exactly as intended. It shipped off on that
basis. It turned out the second measurement was itself wrong — evaluated with the exposure
gate off against a baseline that used it on — and correcting that reversed the reversal:
+0.002 to +0.011 on held-out, significant at p < 0.03. The rule was right; we just hadn't
applied it to our own harness as rigorously as we'd applied it to the mechanism. It ships on
now, and the habit we're carrying forward is re-verifying the measurement, not just the claim.

## Extension: adapting to the deployment surface

Because presentation is decoupled from ranking, the same agent adapts to the interface it sits
behind by changing one flag — no retraining, no re-ranking, no code change.

| deployment surface | mode | TechnicalScore |
|---|---|---|
| voice assistant, chat thread, SMS — one item is the native unit | walk (default) | **0.9707** |
| web grid or app carousel — ten thumbnails cost the user nothing | `--no-walk` | 0.9580 |
| full transparency, everything surfaced at once | `--no-exposure-gate` | 0.9151 |

All three run the identical retrieval and ranking stack; Hit@10 is **1.000** in every one. What
changes is only how much of the ranking reaches the customer per turn.

That last row is worth stating plainly for anyone reproducing our numbers: **0.9151 is our
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

**Languages / frameworks:** Python 3.10+, NumPy (sole third-party dependency — TF-IDF,
randomised SVD, cosine similarity all hand-implemented)

**No** PyTorch, Hugging Face, scikit-learn, FAISS, or any vector database — the rules require
in-memory execution and allow scoring with network disabled, so the dense index is pure NumPy.

**APIs: none on the scored path.** Every number above was produced with **zero API calls, no
credentials, and no network**. That is deliberate: the rules allow scoring with network
disabled, so the system has no key to leak, no rate limit to hit, and no per-query cost.

The repository does contain Claude integration, and we would rather point at it than have you
find it: `pipeline/reranker.py` holds four LLM reranking variants configured for
`claude-opus-5`, which we built and measured live. Every one scored equal to or worse than the
40-line local reranker, so all four ship disabled — and because `requirements.txt` is one line
(`numpy`), the `anthropic` SDK is not even installed by default, so those paths fall back to
the local ranker rather than calling out. They are documented as negative results, not as part
of the system; the README has the per-variant numbers and what their failure modes taught us
about reranker calibration.

**Development tools:** VS Code, Claude Code, git/GitHub (branch-per-feature with PR review),
Python `unittest`

**Datasets:** Amazon Reviews 2023 (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry` category —
the organiser's frozen 50,000-product catalog and 200 labelled public sessions. No external
data, no manual labelling, no catalog mutation.

**Repository:** https://github.com/gaozilin2005/hazelnut-bubble
