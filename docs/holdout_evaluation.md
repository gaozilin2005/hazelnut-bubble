# Held-Out Evaluation

How we measure whether the pipeline generalises, rather than whether it memorised the
200 released sessions.

Measured 2026-08-29 on the real catalog, at commit `8e88776` (confidence-gated exposure).
Person C owns this document.

## Why this exists

Every number the team quoted before this point was measured on `data/public_set.jsonl` —
the same 200 sessions the pipeline was tuned against. Technical Execution is 35% of
judging; the other 65% asks whether the approach generalises. A score on the tuning set
cannot answer that question, and the organizer keeps 800 sessions private, so we have to
build the held-out set ourselves.

## The pieces

A **session** is a customer plus one hidden target product. The target is a `parent_asin`
pointing into the catalog. The customer's dialogue is not stored — the evaluator generates
it at run time from the target.

| file | contents | tracked |
|---|---|---|
| `data/catalog.jsonl` | 50,000 products (title, features, details, price, `rating_number`) | no — download |
| `data/public_set.jsonl` | 200 released sessions, used for tuning | yes |
| `data/holdout_matched_148.jsonl` | 148 generated sessions, popularity-matched | no — regenerate |
| `data/holdout_broad_1000.jsonl` | 1,000 generated sessions, catalog-representative | no — regenerate |

Generated sets are gitignored. They are fully reproducible from `--seed`, so the seed is
the artifact worth recording, not the file.

## The frozen artifacts

Three things are the organizer's and must stay byte-identical to what they published:
`evaluator/local_evaluator.py`, `data/public_set.jsonl`, and `data/catalog.jsonl`. The rules
disallow *"code that modifies evaluator files"*; `tools/` is our own instrumentation and
imports the evaluator without touching it. Verify at any time:

```bash
git fetch -q https://github.com/TechJam2026/techjam-conversational-search main
git diff --stat FETCH_HEAD..HEAD -- evaluator/ data/public_set.jsonl   # must be empty
```

The catalog itself is not in git — 58 MB, and it ships with that **upstream organizer repo**,
not with our fork. Our fork has no releases of its own; two sessions have lost time to this.
Expected digest:

```
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8  catalog.jsonl.gz
```

Harness certification is [step 1](#step-1--certify-the-harness): the BM25 starter must
return `hit_rate_at_10 0.125`, `mrr 0.068034`, `mttc 9.81`, `recommended_technical_score
0.10671` — matching `docs/baseline_results.json` to five decimals. It does. Five-decimal
agreement on a non-trivial agent certifies the harness far better than an oracle or null
agent, both of which pass trivially.

## How sessions are generated

`tools/gen_sessions.py` simulates no customer behaviour. This is the whole basis of its
validity, and it rests on one observation:

> The released public sessions carry only `sample_id`, `scenario_type`, `user_profile`,
> `ground_truth`, `category_bucket`, `difficulty_bucket`. They contain **no** `intent_card`
> and **no** `behavior`.

The evaluator derives those two fields itself in `materialize_hidden_fields()`
(`evaluator/local_evaluator.py:204`), seeding its RNG from `sample_id` + `scenario_type`.
Public and generated sessions therefore traverse byte-identical customer-simulation code.
A generated session is four required fields; everything the customer says is the frozen
evaluator's work, not ours.

What the generator chooses:

- **Targets** — sampled from the 49,800 catalog products the public set does not use.
- **Scenario mix** — fixed 40/40/15/5, verified against the public set's 80/80/30/10.
- **Profiles** — whole `user_profile` objects resampled from the public 200. Copying an
  entire profile preserves tag co-occurrence and the `average_prior_rating` /
  `rating_style` / `summary` agreement that per-field sampling breaks. Nothing currently
  reads `user_profile` (see [Dead inputs](#dead-inputs)), so this is insurance, not a fix.

```bash
python3 tools/gen_sessions.py --out data/holdout_broad_1000.jsonl \
    --count 1000 --seed 20260829 --match none
python3 tools/run_eval.py --agent pipeline --dataset data/holdout_broad_1000.jsonl
```

Each run also writes `<out>.meta.json` comparing the generated set against the public set
on every axis it attempted to match. Read it. It is what caught the bug below.

Full command sequence: [Running this yourself](#running-this-yourself).

## The popularity ceiling

The organizer did not choose their 200 targets uniformly. Their median target has **6,846**
reviews; the median catalog product has **12**. The 5-core leave-last-out sampling is
heavily popularity-biased.

That matters because if we draw targets uniformly, our sessions are about obscure products,
and a score drop becomes ambiguous: unseen targets, or just harder products? So
`--match popularity` (the default) bins targets into five strata at the public set's own
quintile boundaries and reproduces its stratum shares.

The catalog cannot support that at arbitrary size:

| stratum | review range | public | share | available | supports N= |
|---|---|---|---|---|---|
| S0 | 0 – 648 | 41 | 0.205 | 47,602 | 232,204 |
| S1 | 648 – 3,751 | 40 | 0.200 | 1,870 | 9,350 |
| S2 | 3,751 – 9,263 | 40 | 0.200 | 214 | 1,070 |
| S3 | 9,263 – 24,583 | 40 | 0.200 | 85 | 425 |
| S4 | 24,583+ | 39 | 0.195 | **29** | **148** |

Only 68 products in the entire catalog exceed 24,583 reviews, and the public set already
claims 39 of them. Twenty-nine products can fill a 19.5% share of at most **148 sessions**.

**This is a ceiling on popularity matching, not on the generator.** Drop the matching and
any size is available.

### The failure it hides

Requesting 1,000 popularity-matched sessions does not error. Scarce strata empty, and the
nearest-non-empty fallback spills their draws into denser, more obscure strata. The result
claims to be popularity-matched with a **median of 1,328 against the public 6,846** —
quietly a different benchmark. The first run did exactly this and reported
`exhausted_strata_draws: 0`, because the fallback always succeeded.

The generator now computes the ceiling up front and warns. Note which check caught this:
not the assertion written to catch it, but the public-vs-generated median comparison in the
metadata. **Report the comparison, do not assert the mechanism.**

## Results

Real catalog, `PipelineAgent`, local reranker, seed 20260829.

| set | n | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|---|
| BM25 baseline (public) | 200 | 0.125 | 0.068 | 9.81 | 0.10671 |
| **public — tuned on** | 200 | 0.995 | 0.940 | 2.28 | **0.9538** |
| held-out, popularity-matched | 148 | 0.980 | 0.905 | 2.47 | 0.9319 |
| **held-out, catalog-representative** | 1,000 | 0.962 | 0.874 | 2.80 | **0.9072** |

`TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × clip((11 − MTTC)/10, 0, 1)`

The 1,000-session number is the headline: it is the largest sample and the closest thing we
have to the private set. The 148-session number exists only to close off the objection that
the drop came from testing on a different *kind* of product rather than on unseen ones.

### Confidence-gated exposure held up out of sample

Commit `488176c` withholds all but the top candidate until turn 3. It was tuned on the
public set, so the held-out sets are the test of whether it was tuned *to* the public set:

| set | before | after | gain |
|---|---|---|---|
| public — tuned on | 0.9118 | 0.9538 | +0.042 |
| held-out, matched (148) | 0.8937 | 0.9319 | +0.038 |
| held-out, broad (1,000) | 0.8545 | 0.9072 | **+0.053** |

**The gain is larger out of sample than in it.** A change overfitted to the tuning set shows
the opposite. It also closed the generalisation gap, public → broad, from 0.057 to 0.047, and
cost no held-out hits at all — the 1.000 → 0.995 Hit@10 loss is one public session
(`public_0178`) and does not recur on either held-out set.

Broad set by scenario:

| scenario | n | Hit@10 | MRR | Score |
|---|---|---|---|---|
| buying | 400 | 0.963 | 0.895 | 0.9223 |
| browsing | 400 | 0.958 | 0.880 | 0.9059 |
| intent_override | 150 | 0.980 | 0.834 | 0.8872 |
| boundary | 50 | 0.940 | 0.778 | 0.8559 |

Boundary is the thinnest scenario everywhere — n=10 in the public set, n=7 under popularity
matching, n=50 only in the broad draw. Any boundary claim should cite the broad set.

## Running this yourself

Everything below runs from the repo root on Python 3.10+, standard library only. No API
keys. No network after step 1. The whole sequence takes about four minutes, most of it
index building.

### One-time setup

The catalog is not in git — it is 58 MB and ships with the **upstream organizer repo**, not
with our fork. Our fork has no releases; two sessions have now lost time to this.

```bash
BASE=https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit
curl -sSL -o data/catalog.jsonl.gz "$BASE/catalog.jsonl.gz"
curl -sSL -o data/SHA256SUMS       "$BASE/SHA256SUMS"

# SHA256SUMS lists bare filenames, so -c must run from the directory holding them.
(cd data && shasum -a 256 -c <(grep catalog.jsonl.gz SHA256SUMS))   # -> catalog.jsonl.gz: OK

gzip -dk data/catalog.jsonl.gz    # -> data/catalog.jsonl, 50,000 lines
```

Do not skip the checksum. Every number in this document assumes that exact file.

### Step 1 — certify the harness

Before trusting any score, reproduce the organizer's published baseline:

```bash
python3 tools/run_eval.py --agent baseline --dataset data/public_set.jsonl
```

Must print `recommended_technical_score: 0.10671`, matching `docs/baseline_results.json`
to five decimals. If it does not, stop — something is wrong with the catalog or the
environment, and nothing measured after this point is trustworthy.

### Step 2 — generate the held-out sets

```bash
python3 tools/gen_sessions.py --out data/holdout_matched_148.jsonl  --count 148  --seed 20260829
python3 tools/gen_sessions.py --out data/holdout_broad_1000.jsonl   --count 1000 --seed 20260829 --match none
```

These are gitignored and reproducible from the seed, so regenerate rather than copying files
between machines — verified byte-identical on re-run. A different seed gives a genuinely
different set: seeds 20260829 and 99999 share 16 of 1,000 targets, which is about what
independent draws from 49,800 candidates should collide on.

`--match none` on the second is deliberate: a popularity-matched draw cannot exceed 148
sessions. See [The popularity ceiling](#the-popularity-ceiling).

### Step 3 — score the agent

```bash
python3 tools/run_eval.py --agent pipeline --dataset data/public_set.jsonl
python3 tools/run_eval.py --agent pipeline --dataset data/holdout_matched_148.jsonl
python3 tools/run_eval.py --agent pipeline --dataset data/holdout_broad_1000.jsonl
```

Output filenames are derived from the agent and dataset — `results_pipeline_public_set.json`
and so on — so runs on different datasets cannot overwrite each other. Pass `--output` only
if you want a different name.

Useful flags: `--reranker {local,llm,identity}`, `--no-prior`, `--no-dense` for ablations,
and `--limit N` for a quick smoke test.

### Step 4 — slice the results

```bash
python3 tools/analyze_holdout.py \
    --dataset data/holdout_broad_1000.jsonl \
    --results results_pipeline_holdout_broad_1000.json
```

`run_eval.py` reports overall and by-scenario metrics only. The popularity and difficulty
breakdowns come from here — it joins the result file to the session file on `sample_id`,
using the per-row `rating_number` and `constraint_count` that `gen_sessions.py` writes. It
will refuse to run against `data/public_set.jsonl`, which carries neither.

### Expected output

| command | file written | expected score |
|---|---|---|
| step 1 | `results_baseline_public_set.json` | **0.10671** exactly |
| step 3a | `results_pipeline_public_set.json` | 0.9538 |
| step 3b | `results_pipeline_holdout_matched_148.json` | 0.9319 |
| step 3c | `results_pipeline_holdout_broad_1000.json` | 0.9072 |

Only step 1 must match exactly — it is a fixed reference. The other three move whenever the
pipeline changes, which is the point.

### Checking whether a result is current

Every result file opens with a `provenance` block:

```json
"provenance": {
  "commit": "8e88776",
  "branch": "person-c/evaluation",
  "dirty": false,
  "agent": "pipeline",
  "dataset": "data/holdout_broad_1000.jsonl",
  "catalog": "data/catalog.jsonl",
  "reranker": "local",
  "use_prior": true,
  "use_dense": true,
  "generated_at": "2026-08-29T05:47:12+00:00"
}
```

Compare `commit` against `git log --oneline -1`. If they differ, the file predates the code
and its scores are historical. **`dirty: true` means the tree had uncommitted changes**, so
the run cannot be reproduced from that commit alone — fine while iterating, not fine for a
number you intend to quote or put in a report.

This block exists because a results file reading `0.854544` once sat on disk looking current
after the agent beneath it had been replaced.

### Timings

Measured on the 2026-08-29 run; 200 sessions unless noted.

| stage | time |
|---|---|
| BM25 index build | 2.1s |
| BM25 eval | 26.5s |
| pipeline index build (50k products) | ~60s |
| pipeline eval, 200 sessions | 2.3s |
| pipeline eval, 1,000 sessions | 22.4s |
| session generation, 1,000 sessions | 5.8s |

The fixed ~60s index build dominates any single run; per-session cost is 12–22ms and zero
tokens. Batch datasets rather than re-running to amortise it.

## Findings

**The old 0.854 was proxy-catalog pessimism.** The pipeline scored **0.9118** on the public
set with the real catalog at commit `556afbf`, against 0.854 on the reconstructed one — off by 0.058, six times
the ±0.009 calibration band. That band was validated with BM25 and does not transfer: the
pipeline's exploit is far more sensitive to catalog fidelity than BM25 is. *Retire the
reconstructed catalog for scoring.* (Coincidence worth flagging: the broad held-out score is
also ≈0.854. The two numbers are unrelated; do not let them merge.)

**It generalises.** Only 0.047 separates the tuning set from a 1,000-session held-out draw.
This is the expected result once the mechanism is named: the exploit lives in the
evaluator's `intent_card`, which derives constraints from *any* target's own fields. It was
never specific to the 200 released targets, so unseen targets do not threaten it.
Overfitting is not the risk here — rewording is.

**Popularity does not drive the score.** Broad set, by popularity quintile:

| quintile | median reviews | Hit@10 | MRR | Score |
|---|---|---|---|---|
| Q1 | 1 | 0.950 | 0.879 | 0.9015 |
| Q2 | 4 | 0.970 | 0.870 | 0.9111 |
| Q3 | 12 | 0.950 | 0.858 | 0.8924 |
| Q4 | 44 | 0.950 | 0.871 | 0.8983 |
| Q5 | 268 | 0.990 | 0.893 | 0.9325 |

Flat across a 268× range. The popularity prior is not what makes retrieval work, and the
matched/broad distinction matters less than it appeared when the ceiling was found.

**The private 800 are probably more obscure than the public 200.** The popular products
needed to match the public profile at n=800 do not exist in the catalog — the arithmetic
above forbids it. So the private set most likely skews toward ordinary products, which makes
the catalog-representative draw the better proxy for final judging. *This is an inference
from catalog arithmetic, not an organizer statement.*

## Dead inputs

Confirmed by inspection, worth knowing before optimising against them:

- **`user_profile` reaches nothing that scores.** The customer simulator (`initial_message`,
  `customer_reply`) uses only `scenario_type`, `intent_card`, `behavior` and
  `coarse_category`. `PipelineAgent.reset` stores the profile into `SharedSessionState` and
  never reads it. This changes the moment B conditions the question policy on it.
- **`category_bucket` is constant** (`"clothing"` for all 200) and **`difficulty_bucket` is
  never read** by `evaluator/`, `pipeline/`, `tools/` or `starter/`.
- **Constraint count saturates.** `intent_card` emits at most 2 hard + 2 soft constraints, and
  96.5% of targets sit at the cap of 4. It is therefore a poor difficulty proxy, and the
  degenerate-target concern is empirically moot: **0 of 200** public targets and 0.2% of the
  catalog fail a ≥2-constraint test. `--min-constraints` defaults to 0 for this reason —
  filtering held-out targets while the public set went unfiltered would make held-out
  systematically easier.

## Contamination rule

The generator writes a data file, not a score. It is reusable indefinitely — **but the moment
you tune against a draw, that draw is training data.**

- Draw a fresh `--seed` for every held-out measurement.
- Record which seed produced which number, next to the number.
- Use `--exclude <prior file>` for draws that must be disjoint.
- Never report a held-out score from a seed that informed a design decision.

## Open

- **`pipeline/dialog.py` is not wired in.** Commit `4ceb202` adds a 300-line
  `ConversationBrain`, but nothing imports it — `grep -rn dialog pipeline/ tools/ starter/`
  returns only comments. `PipelineAgent._ask` is still the placeholder that returns
  `"other"` every turn. **Every number in this document was produced by the placeholder**,
  so B's policy is currently unmeasured, not underperforming.
- **L3/L4 inversion reproduced on the real catalog.** Commit `488176c`'s own ladder shows it
  in both arms: always-10 L3 0.6516 / L4 0.7073, withholding L3 0.6675 / L4 0.6981. It is
  therefore not an artifact of the reconstructed catalog and not specific to one agent —
  which points at the ladder's level definitions rather than at the pipeline. The robustness ladder reports L4 (0.654) above L3 (0.616), which
  compounding degradations cannot do; Correction 03 quotes fixed-L3 as 0.656. Unresolved.
  The fix is a level-independence assertion in the harness, and the ladder should be rebuilt
  on the real catalog — a ladder measured on the reconstructed catalog inherits the error
  documented above at every level.
- All results here are **L0 verbatim**. The robustness ladder, not the held-out set, is where
  this approach is actually at risk.
