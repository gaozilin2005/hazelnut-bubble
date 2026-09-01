# Demo Video Script

Target: **4:27**, screen recording with voiceover. No front-end exists, so this is the
walkthrough format the brief explicitly permits for backend/NLP tracks ("API usage, inference
examples, or result analysis" — §4.5).

Narration is **614 words**; the timestamps below are computed from that at 150 wpm plus command
runtime, so they are what it will actually take. **Don't ad-lib past the script** — an
earlier draft ran to six minutes of speech, which is how these overrun.

## Deliverables compliance (§4.5, item 3)

| requirement | how this satisfies it |
|---|---|
| "Demonstrates your solution working end-to-end" | Three live runs, real command output rather than slides: routing across every scenario type (0:30), a full multi-turn session including an intent override (0:50), and the official evaluator over all 200 sessions (1:40). |
| "Short video" | 4:27 |
| Uploaded to YouTube, **public** visibility | Unlisted or private fails the requirement — set it to Public. |
| Linked in the Devpost description | Paste the URL into `docs/devpost_description.md` before submitting. |
| No third-party trademarks or copyrighted content | No music, no logos, no stock footage. Product titles from the organizer's own catalog appear in output; that is inherent to demonstrating the solution on the provided dataset. |

**Before recording:** pre-run each command once so the ~60 s index build is warm, then re-run on
camera. Two terminals at 16pt+, repo open in VS Code, and **confirm no `ANTHROPIC_API_KEY` is
visible on screen**.

---

## 0:00 — 0:36 · The problem, and the insight

> "Shopping search breaks when the customer can't name what they want — so the agent has to ask,
> and every wasted question loses them.
>
> This is Hazelnut, our agent for Track 4. Reading the evaluator line by line reframed the whole
> project: the constraints a customer discloses are drawn **verbatim from the target product's
> own metadata**. When they say '100% Leather', they're quoting a field on the item we're looking
> for. So exact matching *identifies* the target, and semantic neighbours are the enemy —
> plausible, wrong, and outranking the truth."

**Screen:** `evaluator/local_evaluator.py` on `materialize_hidden_fields`, then cut to terminal.

## 0:36 — 1:03 · Live: dual-track routing (Pillar I)

```bash
python3 tools/demo_session.py --routing
```

> "Pillar I wants dual-track routing. Intent lives in the *shape* of the opener, read before any
> retrieval: a 'still exploring' tail is browsing, a 'key requirement' marker is buying, a bare
> category-then-value opener is an override. Buying locks hard constraints; browsing opens the
> dense route. A hundred percent accurate on intent and category across a thousand held-out
> sessions."

## 1:03 — 2:07 · Live: a full conversation (Pillar II)

```bash
python3 tools/demo_session.py --sample public_0002
```

> "A real scored session, turn by turn — the customer side is the evaluator's own simulator, not
> a mock.
>
> Turn one: routed, filtered to an exact category bucket, ranked by IDF-weighted constraint
> coverage. Watch the question — it isn't scripted. The surviving candidates split between black
> and brown, so it asks about *that*: Pillar II's proactive clarification, driven by whichever
> facet actually divides the live pool.
>
> Turn two, slots accumulate and the target hits the top — but the evaluator won't count it yet,
> because the override hasn't fired.
>
> Turn three, the customer changes their mind — the case that breaks naive state machines. The
> agent acknowledges the switch and marks the old value inactive for the dialog, but keeps it as
> retrieval evidence: erasing it outright measured six thousandths worse.
>
> Turn four: rank one."

## 2:07 — 2:38 · The numbers (Pillar IV)

```bash
python3 tools/run_eval.py --agent pipeline
```

> "The official evaluator, unmodified, all two hundred sessions. Hit Rate at 10: one hundred
> percent. MRR 0.996. Mean turns to conversion 2.4. TechnicalScore **0.9707** against a BM25
> baseline of 0.107 — nine times the baseline, and Pillar IV's three metrics all at once.
>
> It generalises: **0.9351** on a thousand-session held-out draw that informed no design decision
> we made."

## 2:38 — 3:26 · Presentation policy is a design axis

> "The most interesting decision isn't in the ranker. Rank is scored *within the list you return
> on a turn*, and MRR is weighted 0.30 against 0.02 per extra turn — so how much you surface is a
> design decision, separate from how well you rank. So we decoupled them: the default walks the
> ranking one candidate at a time, never re-offering something already passed on. Plus 0.012
> held-out, sign test p near zero.
>
> And that's not just a scoring trick — a copilot on a voice assistant *cannot* return a ten-item
> grid. One flag adapts the agent to its surface: 0.9707 walking, 0.9580 batched for a web grid,
> 0.9151 with everything at once — identical retrieval in all three."

**Screen:** README "Exposure Policy", three-mode comparison.

## 3:26 — 4:04 · Why you can trust the numbers (Pillar III)

> "Most of what we believed turned out to be wrong. Four LLM reranking variants, all run live
> against Claude — every one equal to or worse than a forty-line local reranker.
>
> And our best Pillar III mechanism replicated at plus 0.017, then appeared to reverse, and we
> nearly shipped it off. On the third check the reversal was *our own measurement bug*. Corrected,
> it's positive and significant on two draws, and it ships on. An ablation is only valid against
> the baseline it was run on — including when your own harness is what's wrong."

## 4:04 — 4:27 · Close

> "Every mechanism sits behind a flag with the draw it was measured on, including a dozen that
> failed — all of it re-runs from the README.
>
> One dependency. No GPU, no network, no credentials, eight milliseconds a turn. It runs on the
> laptop it was built on, and it would run behind a real storefront tomorrow. Thanks for
> watching."

**Screen:** end on README "Coverage Against the Brief".

---

## Recording notes

- **The three live runs are the requirement.** §4.5 asks the video to demonstrate the solution
  *working end-to-end*; a README walkthrough alone doesn't satisfy it. Everything else supports.
- **Don't cut the 3:26 block.** Catching your own measurement bug is the strongest credibility signal
  in the submission, and it costs 30 seconds.
- Keep all three mode numbers on screen through the 2:38 block — a judge who spots that gap unprompted
  weighs it far worse than one walked through it.
- If you need it shorter, cut the `--routing` demo (0:36) — the conversation demo re-shows
  routing implicitly, and it is the one block whose point survives being dropped.
- Say "Claude Opus 5" rather than showing a key. The default path makes zero API calls.
- `demo_session.py --scenario buying` converts in two turns if the override session runs long on
  camera.
