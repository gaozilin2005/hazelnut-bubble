# Demo Video Script

Target: **4–5 minutes**, screen recording with voiceover. No front-end exists, so this is the
walkthrough format the brief explicitly permits for backend/NLP tracks ("API usage, inference
examples, or result analysis").

**Requirements checklist:** upload to YouTube · set visibility **Public** · link in the Devpost
description · no third-party trademarks or copyrighted music.

**Before recording:** run `python3 tools/run_eval.py --agent pipeline` once so the index is
warm and the 60 s build does not eat your runtime. Have a terminal with a readable font size
(16pt+) and the repo open in VS Code.

---

## 0:00 — 0:30 · The problem, and the one insight

> "TechJam Track 4 asks for a conversational shopping agent: a customer describes what they
> want, and the agent has ten turns to surface the exact product they had in mind out of a
> fifty-thousand-item Amazon catalog.
>
> We started by reading the evaluator line by line — and found the thing that reframed the
> whole project. The constraints a customer discloses are drawn *verbatim from the target
> product's own metadata*. When the simulated customer says '100% Leather', they are quoting a
> field on the item we are trying to find."

**Screen:** `evaluator/local_evaluator.py`, scrolled to `materialize_hidden_fields`.

## 0:30 — 1:15 · Why that killed our first design

> "That inverts the obvious approach. Our first instinct — and our first implementation — was
> dense semantic retrieval. But if the customer is quoting the target's metadata, then exact
> matching *identifies* the item, and semantic neighbours are actively the enemy: they are
> plausible, they are wrong, and they outrank the truth.
>
> We measured it instead of arguing about it. Fusing dense vectors into the ranking was
> monotonically harmful at every weight we tried."

**Screen:** README "What We Tried and Rejected", the 0.8802 → 0.8464 line.

> "So dense retrieval stayed — but only for the cold-start turn, before anything is disclosed.
> That is the one place it beats popularity, and it is worth about a thousandth of a point.
> Everything else is exact matching over an IDF-weighted constraint coverage score."

## 1:15 — 2:15 · Live end-to-end run

**Screen:** terminal. Type and run:

```bash
python3 tools/run_eval.py --agent pipeline
```

> "This is the official evaluator, unmodified, on all two hundred public sessions. No LLM
> calls, no network, no API key. One NumPy dependency."

Let it finish; point at the output.

> "Hit Rate at 10: one hundred percent. MRR: 0.992. Mean turns to conversion: 2.4.
> TechnicalScore 0.9693 — against the organiser's BM25 baseline at 0.107. About nine times the
> baseline, and the whole run takes under four seconds after the index build."

Then show a single session concretely:

```bash
python3 -c "
from agent import Agent
a = Agent('data/catalog.jsonl')
a.reset('demo', {})
r = a.respond('demo', \"I'm looking for women's leather riding boots.\", turn=1, top_k=10)
print(r['message']); print(r['ask_attribute']); print(r['recommendations'][:3])
"
```

> "One turn, in isolation: it routes the intent, filters to the category bucket, ranks, and
> asks about the attribute the surviving candidates most disagree on — that is the brief's
> proactive clarification, driven by which facet actually splits the live pool."

## 2:15 — 3:15 · Presentation policy as a design axis

> "Now the design decision we think is the most interesting one in the system.
>
> Reading the evaluator closely, we found that rank is computed *within the list you return on
> a given turn* — and that the scoring function pays 0.30 for MRR against only 0.02 for an
> extra turn. Which means *how many* candidates you surface per turn is a real design decision
> with a measurable cost, completely separate from how well you rank them.
>
> So we decoupled the two. Ranking is one subsystem. The policy deciding how much of that
> ranking to surface each turn is another."

**Screen:** README "Single-Item Walk Disclosure", showing the payoff table.

> "Our default walks the ranking one strong candidate at a time — and never re-offers something
> the customer has already passed on. So every turn shows the best *remaining* product. That is
> what 'ordered best to worst' should mean inside a conversation.
>
> And it mirrors how conversational commerce actually works. A shopping copilot on a voice
> assistant, or in a chat thread, *cannot* hand back a ten-item grid. One candidate at a time
> is the native unit there, not a compromise.
>
> It costs us turns — mean turns to conversion rise from 2.26 to 2.41 — and we measured that
> the trade is worth it: plus 0.012 on a held-out draw that informed no design decision, paired
> sign test, p approximately zero."

Now show the modes, running one live:

```bash
python3 tools/run_eval.py --agent pipeline --no-walk
```

> "Because presentation is decoupled from ranking, one flag adapts the agent to whatever
> interface it sits behind. Walk mode for voice and chat: 0.9693. Batched mode for a web grid,
> where ten thumbnails cost the user nothing: 0.9571. And with every exposure mechanism off,
> full top-ten on turn one: 0.9118.
>
> Identical retrieval and ranking in all three — Hit@10 is one-point-zero in every one. The
> only thing that changes is how much of the ranking reaches the customer each turn. In a real
> deployment you would pick the row that matches your surface."

## 3:15 — 4:15 · Measurement discipline

> "The reason we trust any of these numbers is that most of what we believed turned out to be
> wrong.
>
> Four LLM reranking variants — listwise, targeted, pairwise, pairwise-top-3 — all built, all
> run live against Claude. Every single one measured equal to or worse than a forty-line local
> reranker. When the customer is quoting the target's metadata, there is no semantic judgment
> left to add.
>
> And the sharpest lesson came last week."

**Screen:** README Pillar III, the two tables side by side.

> "Aspect-level negative feedback — grounded in Bi et al., CIKM 2019 — replicated at plus
> 0.017 across three independent held-out draws, two of which we had never tuned against. By
> our own standards it had passed.
>
> Then a teammate landed the single-item walk, and we re-measured. It reversed to minus 0.006.
> Hit@10 fell from 0.993 to 0.982 — it was losing eleven real targets per thousand sessions.
>
> The reason is subtle: penalising a rejected item's attributes also penalises the target's
> nearest neighbours, and with a one-item-per-turn walk, every rejection *is* a near neighbour.
> Two sound mechanisms, mutually destructive.
>
> An ablation is only valid against the baseline it was run on. So that flag ships off, and the
> reversal is documented in the README rather than deleted."

## 4:15 — 4:45 · Close

> "Every mechanism we built sits behind a flag with the draw it was measured on — including
> about a dozen that failed. Nothing here has to be taken on trust; all of it can be re-run.
>
> Three point nine seconds for two hundred sessions. One dependency. No network, no
> credentials, no GPU. And it generalises: 0.93 on a thousand-session held-out draw that
> informed no design decision we made.
>
> Thanks for watching."

**Screen:** end on the README's "Coverage Against the Brief" table.

---

## Recording notes

- **Do not skip the 3:15–4:15 block.** The reversal story is the strongest evidence of
  engineering judgment in the whole submission and it is what separates this from a
  leaderboard entry.
- In 2:15–3:15, keep all three mode numbers on screen. The argument lands *because* it is
  concrete about the trade-off and about what the system scores in each mode — a judge who
  spots the difference unprompted will weigh it far worse than one who is walked through it.
- If you run long, cut the 0:30–1:15 dense-retrieval segment first; it is the least
  load-bearing.
- Say "Claude Haiku 4.5" rather than showing any API key on screen. Check the terminal for a
  visible `ANTHROPIC_API_KEY` before recording.
