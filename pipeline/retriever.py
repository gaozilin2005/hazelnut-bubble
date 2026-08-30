"""Hard-filter + lexical retrieval ("exploit arm" of the A/B).

Two observations about the harness drive this design:

1. The opening message's category is `coarse_category(target["categories"])` --
   a pure function of catalog data (evaluator/local_evaluator.py::coarse_category).
   Recomputing it over all 50k products yields an exact bucket that is
   guaranteed to contain the target. It is a lookup key, not a fuzzy signal.

2. Every disclosed constraint is a verbatim slice of the target's own
   features/details, altered only by whitespace collapse, punctuation strip, and
   a 180-char truncation. After normalization, containment is an exact test.

Both degrade: an unparsed category skips the filter, and a constraint that fails
containment still scores by IDF-weighted token overlap.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.dense import DenseIndex
from pipeline.distill import (
    live_discriminance,
    merge_redundant,
    rejected_aspect_penalty,
)
from pipeline.interfaces import (
    INTENT_BROWSING,
    INTENT_UNKNOWN,
    SharedSessionState,
)
from pipeline.textutil import (
    FACET_VOCABULARY,
    STOPWORDS,
    normalize,
    product_corpus,
    terms,
)

BUDGET_RE = re.compile(r"budget around \$?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
EXCLUDED_CATEGORIES = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
# Full-phrase containment is the exploit; token overlap is the fallback. The
# multiplier is what separates "quotes the product verbatim" from "shares words".
PHRASE_BONUS = 3.0
PRICE_BONUS = 2.0
UNFILTERED_CANDIDATES = 400
# Safe personalization from the anonymized profile. Measured on the public set,
# a product's overlap with the customer's preference_tags ranks the target at the
# 25th percentile of its category bucket against 50th for chance, beating chance
# in 83% of sessions. It is a weak, broad signal -- it cannot identify an item,
# only tilt ties -- so it is weighted like the popularity prior, not like a
# constraint. purchase_frequency is constant across every session and carries no
# information; rating_style merely restates average_prior_rating.
PROFILE_WEIGHT = 0.0
PROFILE_TAGS = (
    "fit", "material", "comfort", "style", "durability",
    "performance", "warmth", "weather",
)
# Dual-track routing, settled by measurement rather than assumption.
#
# Once ANY constraint has been disclosed, the lexical track owns ranking. A
# disclosed constraint is a verbatim slice of the target's own metadata, so
# exact matching identifies the item outright; fusing dense similarity in is
# monotonically harmful (0.8802 lexical-only, 0.8783 at RRF weight 0.05, down
# to 0.8464 at 0.35) because it supplies plausible neighbours that outrank the
# true target.
#
# Before any constraint is disclosed -- the open-ended browsing cold start --
# lexical has nothing to match and falls back to popularity. There the dense
# track is decisively better: browsing MRR 0.465 -> 0.761, overall 0.8802 ->
# 0.9119. Dense also supplies candidates when the category filter finds no
# bucket at all.
DENSE_CANDIDATES = {
    "buying":   150,   # hard requirements stated: filter track leads, narrow net
    "override": 150,
    "browsing": 400,   # opens vague: cast wide for cross-category coverage
    "unknown":  400,
}
# Browsing is diversity-first: cap any one brand so the shortlist spans the
# category instead of ten variants of one product.
MAX_PER_STORE_BROWSING = 3
# Opt-in (--tie-break-dense), off by default -- MEASURED AND NOT RECOMMENDED,
# kept for the record. Motivated by a real diagnosis: 29 of 38 misses on the
# 1,000-session held-out set are not a ranking error but an information
# deficiency -- the target sits a few ranks below the cutoff in a wide, FLAT
# stalemate (e.g. scores 5.409-5.428 across 12 near-identical "women's leather
# riding boot" listings, gap to the cutoff as small as 0.001), because every
# disclosed constraint is a generic attribute nearly the whole bucket shares.
# The mechanism is narrower than the RRF fusion already measured harmful
# (README) -- dense only ever reorders the band of candidates already within
# TIE_BREAK_MARGIN of the cutoff, never displacing a #1 with a clear lead --
# but it does not hold up under cross-validation:
#
#   dataset                              margin=0   margin=0.01  margin=0.05
#   held-out, 1,000 broad (unmatched)     0.8545     0.8606       0.8606
#   held-out, 200 (popularity-matched)    0.8452     0.8433       0.8466
#   public 200, gated (the tuning set)    0.9538     0.9398       0.9331
#
# The apparent +0.0061 gain on the 1,000-session draw does not replicate on an
# independently-sampled 200-session draw (-0.0019 to +0.0014, noise-level),
# while the public-200 cost is consistent and *worsens* with margin. This is
# the same single-draw-does-not-replicate trap documented for the exposure-
# schedule sweep in README -- reported as a negative result, not shipped as a
# judgement call, because the "positive" signal itself did not survive a
# second independent sample. Inert under paraphrase (L1-L4 unchanged): the
# guard skips tie-breaking whenever constraints fail to parse, matching the
# exposure gate's own cold-start guard.
TIE_BREAK_MARGIN = 0.01
# Reverse containment: a product's own attribute string found inside the user's
# text. Recovers the verbatim-quote signal no matter how the message is worded.
SNIPPET_BONUS = 2.5
MIN_SNIPPET_CHARS = 18
# Length normalization (--len-norm W, off at 0.0): subtract W * log1p(corpus
# tokens) from every score. Motivated by a probe of the 19 public sessions
# converting at rank>1: 47 of the 62 distractors ranked ABOVE those targets
# have a longer corpus than the target. The mechanism is classic BM25: every
# disclosed constraint is a verbatim slice of the TARGET's metadata, so the
# target matches with maximal precision, while a distractor matches the same
# text incidentally inside a longer document. The prior term is ~0.05-0.12 and
# saturated ties are separated by ~0.001-0.04, so a weight of ~0.01-0.03
# decides ties without disturbing real score gaps.
LEN_NORM_DEFAULT = 0.0
MAX_CATEGORY_TOKENS = 8
# Buckets are small (median ~5 on a 50k catalog), so when the category is named
# loosely we can union several near-matches and still hand the scorer a tight
# candidate set. Far safer than committing to one bucket on a weak match.
#
# --- Paraphrase hardening (fires ONLY when the exact template paths fail) ---
#
# Both mechanisms below exist for the private-set paraphrase risk measured by
# tools/robustness.py: L5 (payload rewritten AND category reworded) scored 0.390
# against L0 0.954. They are gated behind exact-path failure -- a parsed
# category, or parsed constraints -- so on template input (L0) they never run
# and the score is bit-identical by construction.
#
# 1. Category suffix union. A person names the taxonomy LEAF ("hoodies"), not
#    the full path the catalog key carries ("novelty more active hoodies"), so
#    a reworded category is characteristically a suffix of the true bucket key.
#    When no full key occurs in the message, the longest message n-gram that is
#    a word-suffix of one or more bucket keys selects the union of those
#    buckets. The union is guaranteed to contain the target's bucket whenever
#    the customer said any suffix of its key, and stays small (1,114 keys; the
#    worst suffix is shared by 35). Distinct from the fuzzy-bucket matching
#    measured harmful above: suffix equality is exact, not similarity-based.
SUFFIX_UNION_MAX = 1500
# 2. Catalog-grounded constraint extraction. When no template parsed, the query
#    used to be the raw customer messages, which dilutes IDF coverage with
#    scaffolding prose ("here's the thing that matters --"). Instead, the
#    longest spans of the message that occur VERBATIM in some product corpus
#    are lifted out as constraints -- recovering the phrase-containment signal
#    the scorer was built around -- and the leftover informative tokens are
#    kept as one bag per message so a genuinely rewritten payload still scores
#    by token overlap.
SPAN_MIN_TOKENS = 3   # 2 admitted shuffled fragments at L3 (-0.050); see below
SPAN_MAX_TOKENS = 12
SPAN_VERIFY_DOCS = 400      # substring checks per candidate span, capped
SPANS_PER_MESSAGE = 3       # a reply discloses at most 2 constraints; 3 is slack
# --- Multi-route lexical RRF (--rrf, opt-in, measured before any default flip).
#
# Motivated by the saturated-signal analysis: every mechanism that reweights
# disclosed-constraint coverage is exhausted (all ten top candidates cover
# 1.000), so the remaining headroom needs rankings SHAPED differently, not
# weighted differently. Four lexical routes over the same disclosures, fused by
# reciprocal rank (Cormack et al. 2009): candidates that several
# differently-shaped queries independently rank well rise above ones a single
# scorer happens to favour.
#
#   coverage  the existing score() ordering (the champion)
#   phrase    verbatim full-phrase containment only, weighted by phrase length
#   broad     IDF token-mass over the WHOLE catalog -- can reach outside the
#             category bucket, which nothing else here can
#   hard      pool members satisfying EVERY constraint, popularity-ordered
#
# Inert under paraphrase by the same guard the dense tie-break uses: it engages
# only when constraints actually parsed.
RRF_K = 60
RRF_WEIGHTS = {"coverage": 1.0, "phrase": 0.9, "hard": 0.7, "broad": 0.15}
RRF_BROAD_DEPTH = 350
# --- Broad-pool augmentation (--broad-pool): MEASURED AND REJECTED, kept for
# the record alongside --rrf. The paired decision-draw measurement (seed
# 20260830, n=1000) split RRF into its two effects: the broad token-mass route
# recovers outright misses whose target the category bucket never contained
# (+0.002 Hit), while the rank fusion degrades MRR (-0.005 to -0.006 at broad
# weight 0.4 and 0.15 alike; sign test p=0.0026 at 0.15). The obvious surgery
# -- keep the candidates, drop the fusion, let score() rank the union -- is
# CATASTROPHIC: -0.121, 387 of 1000 sessions worse, rank-1 targets pushed
# clean out of the top 10. Mechanism: the coverage signal is saturated, so
# hundreds of out-of-bucket products score IDENTICALLY to the target; the
# category bucket is the only precision mechanism the scorer has, and the
# union removes it. RRF's rank-agreement requirement was the only thing
# keeping the broad route's noise out; with it, the net is still negative.
# Conclusion: multi-route retrieval cannot beat a saturated-but-correct
# in-bucket ranking on this benchmark. Both flags stay off.


def coarse_category(values: list[str]) -> str:
    """Reimplementation of the evaluator's function.

    Deliberately not imported: a submission must not depend on organizer files,
    which may not ship with the scoring harness.
    """
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in EXCLUDED_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


class HybridRetriever:
    """Retriever protocol. Owns Hit@10."""

    def __init__(
        self, use_prior: bool = True, use_dense: bool = True,
        tie_break_dense: bool = False, tie_break_margin: float = TIE_BREAK_MARGIN,
        distill: bool = False, multi_route: bool = False,
        broad_pool: bool = False, len_norm: float = LEN_NORM_DEFAULT,
    ) -> None:
        # Length-normalization weight (--len-norm). See LEN_NORM_DEFAULT above.
        self.len_norm = len_norm
        self.log_lengths: list[float] = []
        # Multi-route lexical RRF (--rrf). See RRF_WEIGHTS above.
        self.multi_route = multi_route
        # Broad-pool augmentation (--broad-pool). See the note above.
        self.broad_pool = broad_pool
        # Pillar III context distillation (--distill): merge redundant
        # constraints, then weight each by how much it narrows the LIVE
        # candidate pool rather than by global catalog IDF. See
        # pipeline/distill.py. Off by default; measured, see README.
        self.distill = distill
        self._weights: dict[str, float] = {}
        # Aspect-level negative feedback (--neg-aspects), set per turn by the
        # agent from the items this session has already had rejected.
        self.neg_weight: float = 0.0
        self._neg: dict[str, float] = {}
        self.use_dense = use_dense
        self.tie_break_dense = tie_break_dense
        self.tie_break_margin = tie_break_margin
        self.dense = DenseIndex()
        self.stores: list[str] = []
        # The popularity prior is a legitimate tie-breaker, but on a dev catalog
        # whose distractors are less-reviewed than the targets it behaves like an
        # oracle. Toggle it off to measure how much score is real signal.
        self.use_prior = use_prior
        self.asins: list[str] = []
        self.position: dict[str, int] = {}
        self.corpora: list[str] = []
        self.prices: list[float | None] = []
        self.priors: list[float] = []
        self.snippets: list[list[str]] = []
        self.tagsets: list[frozenset[str]] = []
        self.buckets: dict[str, list[int]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[int]] = defaultdict(list)
        # word-suffix of a bucket key -> the keys it terminates ("hoodies" ->
        # every key ending in "hoodies"). Built once in build().
        self.key_suffixes: dict[str, list[str]] = {}
        # message -> (verbatim spans, leftover tokens); messages repeat across
        # turns within a session, so grounding each one once is enough.
        self._span_cache: dict[str, tuple[list[str], list[str]]] = {}
        # IDF a span's best token must clear to count as distinctive (set in
        # build): roughly "rarer than 1 in 20 products".
        self._span_min_idf: float = 0.0
        # Per-retrieve(): True when no category bucket was found, which lets
        # span-free grounding stand in for raw messages (see _query_terms).
        self._grounded_ok: bool = False

    def build(self, catalog_path: str) -> None:
        document_frequency: dict[str, int] = defaultdict(int)
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                index = len(self.asins)
                self.asins.append(str(product["parent_asin"]))
                self.position[str(product["parent_asin"])] = index
                corpus = normalize(product_corpus(product))
                self.corpora.append(corpus)
                self.log_lengths.append(math.log1p(corpus.count(" ") + 1))

                price = product.get("price")
                self.prices.append(float(price) if isinstance(price, (int, float)) else None)
                # Weak tie-break only. Single-word constraints ("polyester")
                # leave large ties; popularity beats catalog order there.
                rating_count = product.get("rating_number") or 0
                rating = product.get("average_rating") or 0.0
                self.priors.append(
                    (math.log1p(rating_count) * 0.01 + float(rating) * 0.001)
                    if self.use_prior else 0.0
                )

                # Individual attribute strings, kept whole. The simulator draws
                # its constraints from exactly these fields.
                snippets = []
                for value in (product.get("features") or []):
                    snippets.append(normalize(str(value)))
                details = product.get("details")
                if isinstance(details, dict):
                    snippets.extend(normalize(str(value)) for value in details.values())
                self.snippets.append(
                    [text for text in dict.fromkeys(snippets) if len(text) >= MIN_SNIPPET_CHARS]
                )

                self.tagsets.append(frozenset(t for t in PROFILE_TAGS if t in corpus))
                self.stores.append(normalize(str(product.get("store") or "")))
                categories = [str(value) for value in product.get("categories") or []]
                self.buckets[normalize(coarse_category(categories))].append(index)

                for token in set(terms(corpus)):
                    document_frequency[token] += 1
                    self.postings[token].append(index)

        total = max(1, len(self.asins))
        self.idf = {
            token: math.log(1.0 + total / (1.0 + count))
            for token, count in document_frequency.items()
        }
        self._span_min_idf = math.log(1.0 + total / (1.0 + total / 20))
        suffixes: dict[str, list[str]] = defaultdict(list)
        for key in self.buckets:
            words = key.split()
            for start in range(len(words)):
                suffixes[" ".join(words[start:])].append(key)
        self.key_suffixes = dict(suffixes)
        if self.use_dense:
            self.dense.build(self.corpora)

    # -- candidate selection -------------------------------------------------

    def _scan_category(self, text: str) -> str | None:
        """Find the longest known bucket key occurring in the message.

        Independent of sentence structure, so it survives rewording that a
        positional template parse cannot. Longest match wins: "Active Hoodies"
        beats "Active".
        """
        tokens = terms(normalize(text), keep_stopwords=True)
        best: str | None = None
        for width in range(min(MAX_CATEGORY_TOKENS, len(tokens)), 0, -1):
            for start in range(len(tokens) - width + 1):
                key = " ".join(tokens[start : start + width])
                if key in self.buckets:
                    if best is None or len(key) > len(best):
                        best = key
            if best:
                return best
        return best

    def _bucket(self, state: SharedSessionState) -> list[int] | None:
        if state.category and state.category_confident:
            key = normalize(state.category)
            if key in self.buckets:
                return self.buckets[key]
        opening = state.messages[0] if state.messages else ""
        # Template parse failed or named an unknown category: scan the opening
        # message for a category the catalog actually has.
        scanned = self._scan_category(opening)
        if scanned:
            # The scanned phrase may be a full key of the WRONG bucket when the
            # customer shortened a longer taxonomy path to its tail ("active
            # hoodies" is both a key and the suffix of others). Take every key
            # the phrase terminates -- always including the exact match -- so
            # the target's bucket cannot be excluded by the collision. Measured
            # both ways on the paraphrase ladder: the union is worth +0.023 at
            # L4 and +0.007 at L5, and exactly 0.000 at L3 where the full true
            # key is present (it is the longest match, and its supersets add
            # only small sibling buckets the scorer keeps below the target).
            return self._union_keys(self.key_suffixes.get(scanned, [scanned]))
        # No full bucket key occurs anywhere in the message. Before giving up
        # on the filter entirely, try the suffix route: a shortened category
        # name is the tail of its key ("hoodies" for "... active hoodies").
        union = self._suffix_union(opening)
        if union is not None:
            return union
        # No suffix match either. Measured on the paraphrase harness: matching
        # buckets fuzzily and either (a) filtering to them or (b) adding them to
        # the global pool BOTH score worse than doing nothing -- (a) drops the
        # target when its bucket misses the cut, (b) dilutes precision. Falling
        # through to global scoring is the best of the three. Do not "improve"
        # this without re-running tools/robustness.py.
        return None

    def _union_keys(self, keys: list[str]) -> list[int]:
        union: set[int] = set()
        for key in keys:
            union.update(self.buckets.get(key, ()))
        return sorted(union)

    def _suffix_union(self, text: str) -> list[int] | None:
        """Buckets whose key ENDS WITH the longest matching message n-gram.

        Exact word-suffix equality, not fuzzy similarity -- the mechanism the
        harmful-fuzzy-matching measurement above does not cover. Returns None
        when nothing matches or the union grows past SUFFIX_UNION_MAX (a
        one-word suffix like "sets" can legitimately be too generic to filter
        on; global scoring is then the safer arm).
        """
        tokens = terms(normalize(text), keep_stopwords=True)
        for width in range(min(MAX_CATEGORY_TOKENS, len(tokens)), 0, -1):
            matched: list[str] = []
            for start in range(len(tokens) - width + 1):
                phrase = " ".join(tokens[start : start + width])
                matched.extend(self.key_suffixes.get(phrase, ()))
            if matched:
                union = self._union_keys(list(dict.fromkeys(matched)))
                if len(union) > SUFFIX_UNION_MAX:
                    return None
                return union
        return None

    def _by_token_mass(self, state: SharedSessionState, limit: int) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for constraint in self._query_terms(state):
            for token in set(terms(normalize(constraint))):
                weight = self.idf.get(token)
                if weight is None:
                    continue
                for index in self.postings[token]:
                    scores[index] += weight
        if not scores:
            return list(range(min(limit, len(self.asins))))
        ranked = sorted(scores, key=lambda index: -scores[index])
        return ranked[:limit]

    # -- scoring -------------------------------------------------------------

    def _query_terms(self, state: SharedSessionState) -> list[str]:
        """Parsed constraints when available, grounded spans or raw text otherwise.

        Without this the retriever has no query at all when template parsing
        fails, and returns catalog order.

        The grounded branch is gated on INTENT_UNKNOWN -- i.e. the opening
        template did not parse -- because on template input (L0) an empty
        constraint list is a fact about the SESSION (browsing, boundary), not a
        parse failure, and the raw-message fallback there is measured behavior
        that must not drift.

        Within that branch, grounding replaces the raw messages only when it
        EARNED it: a verbatim span was actually found, or there is no category
        bucket (self._grounded_ok, set per retrieve()). Ablated on the paired
        paraphrase harness: with spans found (L4, payload verbatim in reworded
        prose) grounding is worth +0.10; with a bucket present but the payload
        rewritten so no spans exist (L3), replacing whole messages with token
        bags reshapes the per-constraint mass weighting and costs -0.056, while
        the raw messages score fine. So: spans, or no filter to fall back on --
        otherwise raw.
        """
        if state.constraints:
            if self.distill:
                return merge_redundant(state.constraints)
            return state.constraints
        messages = [message for message in state.messages if message.strip()]
        if state.intent == INTENT_UNKNOWN and messages:
            grounded: list[str] = []
            spans_found = False
            for message in messages:
                spans, leftover = self._ground_message(message)
                grounded.extend(spans)
                # A span that is just the category name re-found (the one part
                # of these templates that stays verbatim at every paraphrase
                # level) discriminates nothing within the bucket the filter
                # already selected with it -- it must not vouch for grounding,
                # or the L3 case (payload rewritten, category intact) gets
                # grounded on the strength of information already spent.
                spans_found = spans_found or any(
                    not self._category_like(span) for span in spans
                )
                if leftover:
                    grounded.append(" ".join(leftover))
            if grounded and (spans_found or self._grounded_ok):
                return list(dict.fromkeys(grounded))
        return messages

    def _category_like(self, span: str) -> bool:
        """Is this span (part of) a bucket key, i.e. category rather than payload?"""
        return span in self.key_suffixes or any(
            span in key or key in span for key in self.buckets
        )

    def _ground_message(self, message: str) -> tuple[list[str], list[str]]:
        """-> (verbatim catalog spans, leftover informative tokens) for one message.

        Greedy left-to-right, longest span first. A span is accepted when it
        occurs verbatim in at least one product corpus AND carries at least one
        distinctive token (idf above _span_min_idf) -- the second condition
        keeps scaffolding prose like "the thing that matters" from becoming a
        constraint just because some product description also says it.

        Leftover tokens (informative but not part of any accepted span) come
        back as a bag so a genuinely rewritten payload still scores by
        IDF-weighted token overlap, exactly as the raw-message fallback did --
        minus the scaffolding dilution.
        """
        cached = self._span_cache.get(message)
        if cached is not None:
            return cached
        tokens = normalize(message).split()
        n = len(tokens)
        found: list[tuple[float, str]] = []
        covered = [False] * n
        i = 0
        while i < n:
            # Extend right while some product could still contain the whole
            # span, snapshotting the candidate-doc set at each length so the
            # back-off below verifies against the right pool.
            snapshots: list[set[int] | None] = []
            docs: set[int] | None = None
            j = i
            while j < min(n, i + SPAN_MAX_TOKENS):
                token = tokens[j]
                posted = self.postings.get(token)
                if posted is not None:
                    docs = set(posted) if docs is None else docs.intersection(posted)
                    if not docs:
                        break
                elif token not in STOPWORDS and len(token) > 1:
                    # A content token no product contains: no verbatim span
                    # can cross it.
                    break
                j += 1
                snapshots.append(docs)
            end = None
            for stop in range(j, i + SPAN_MIN_TOKENS - 1, -1):
                pool = snapshots[stop - i - 1]
                if pool is None:
                    continue
                span_tokens = tokens[i:stop]
                if max((self.idf.get(t, 0.0) for t in span_tokens), default=0.0) < self._span_min_idf:
                    continue
                span = " ".join(span_tokens)
                if self._span_occurs(span, pool):
                    found.append((sum(self.idf.get(t, 0.0) for t in span_tokens), span))
                    end = stop
                    break
            if end is None:
                i += 1
            else:
                for x in range(i, end):
                    covered[x] = True
                i = end
        found.sort(key=lambda item: -item[0])
        spans = [span for _, span in found[:SPANS_PER_MESSAGE]]
        leftover = [t for x, t in enumerate(tokens) if not covered[x] and t in self.idf]
        result = (spans, leftover)
        self._span_cache[message] = result
        return result

    def _span_occurs(self, span: str, pool: set[int]) -> bool:
        for checked, index in enumerate(pool):
            if checked >= SPAN_VERIFY_DOCS:
                return False
            if span in self.corpora[index]:
                return True
        return False

    def _satisfies(self, index: int, constraint: str) -> bool:
        """Same containment test the scorer uses, as a hard yes/no."""
        corpus = self.corpora[index]
        text = normalize(constraint)
        if not text:
            return False
        tokens = [t for t in terms(text) if t in self.idf]
        if not tokens:
            return False
        return text in corpus or all(t in corpus for t in tokens)

    def _distil_weights(
        self, state: SharedSessionState, candidates: list[int]
    ) -> dict[str, float]:
        """Per-constraint weights from the live pool, cached for this ranking pass.

        Computed once per retrieve() rather than per candidate: it is O(pool x
        constraints) and the scorer is called once per candidate, so folding it
        into score() would make ranking quadratic.
        """
        constraints = self._query_terms(state)
        if not constraints or not candidates:
            return {}
        counts = {
            constraint: sum(1 for i in candidates if self._satisfies(i, constraint))
            for constraint in constraints
        }
        return live_discriminance(constraints, counts, len(candidates))

    def _blob(self, state: SharedSessionState) -> str:
        return normalize(" ".join(state.messages))

    def _profile_bonus(self, index: int, state: SharedSessionState) -> float:
        if PROFILE_WEIGHT <= 0:
            return 0.0
        tags = [str(t).lower() for t in (state.user_profile.get("preference_tags") or [])]
        tags = [t for t in tags if t in PROFILE_TAGS]
        if not tags:
            return 0.0
        present = self.tagsets[index]
        return PROFILE_WEIGHT * (sum(1 for t in tags if t in present) / len(tags))

    def set_rejected(self, rejected_asins: list[str], disclosed: str) -> None:
        """Decompose this session's rejections into aspect-value evidence."""
        corpora = [
            self.corpora[self.position[a]] for a in rejected_asins if a in self.position
        ]
        self._neg = rejected_aspect_penalty(corpora, disclosed)

    def score(self, index: int, state: SharedSessionState, blob: str = "") -> float:
        corpus = self.corpora[index]
        total = self.priors[index] + self._profile_bonus(index, state)
        if self.len_norm:
            total -= self.len_norm * self.log_lengths[index]
        if self.neg_weight and self._neg:
            # Aspect-level negative feedback: subtract evidence shared with
            # rejected items. Item-level demotion (--no-repeat) only moves the
            # exact rejected listing; this generalises to its attributes.
            total -= self.neg_weight * sum(
                share for value, share in self._neg.items() if value in corpus
            )
        # Reverse containment. At L0 this agrees with constraint matching; when
        # the wrapper text is reworded it is the signal that still fires,
        # because the quoted attribute itself is usually left intact.
        if blob:
            for snippet in self.snippets[index]:
                if snippet in blob:
                    total += SNIPPET_BONUS * math.log1p(len(snippet))
                    break
        for constraint in self._query_terms(state):
            budget = BUDGET_RE.search(constraint)
            if budget:
                price = self.prices[index]
                if price is not None and abs(price - float(budget.group(1))) < 0.005:
                    total += PRICE_BONUS
                continue
            tokens = [token for token in terms(normalize(constraint)) if token in self.idf]
            if not tokens:
                continue
            mass = sum(self.idf[token] for token in tokens)
            matched = sum(self.idf[token] for token in set(tokens) if token in corpus)
            if mass <= 0:
                continue
            # Coverage is IDF-weighted, so a common one-word constraint scores
            # far below a long distinctive feature bullet. That is intended.
            coverage = matched / mass
            if len(constraint) > 12 and normalize(constraint) in corpus:
                coverage *= PHRASE_BONUS
            # Distillation reweights by live-pool discriminance; absent it the
            # weight is 1.0 and this is the original scorer exactly.
            total += coverage * math.log1p(mass) * self._weights.get(constraint, 1.0)
        return total

    def _query_text(self, state: SharedSessionState) -> str:
        """What the dense route searches for: the disclosed constraints, plus the
        category, which is the only signal a browsing session opens with."""
        parts = list(state.constraints)
        if state.category:
            parts.append(state.category)
        return " ".join(parts) if parts else " ".join(state.messages)

    def _diversify(self, ordered: list[int], limit: int) -> list[int]:
        kept: list[int] = []
        seen: dict[str, int] = {}
        overflow: list[int] = []
        for index in ordered:
            store = self.stores[index]
            if store and seen.get(store, 0) >= MAX_PER_STORE_BROWSING:
                overflow.append(index)
                continue
            seen[store] = seen.get(store, 0) + 1
            kept.append(index)
            if len(kept) >= limit:
                return kept
        return (kept + overflow)[:limit]

    def facet_split(
        self, candidate_ids: list[str], pool: int = 60, minimum_share: float = 0.12,
        skip_attributes: set[str] | None = None, disclosed: str = "",
    ) -> tuple[str, list[tuple[str, int]]] | None:
        """-> the facet that best divides the live candidate pool, and its top values.

        This is the signal behind proactive clarification: rather than asking
        "anything else?", ask about the attribute on which the surviving
        candidates actually disagree. A facet where 92% of candidates say nothing
        splits nothing; one at 60/34 splits well.

        Returns None when no facet divides the pool -- then there is nothing
        concrete to offer and the caller should fall back to an open question.
        """
        window = candidate_ids[:pool]
        if len(window) < 4:
            return None
        corpora = [
            self.corpora[self.position[asin]] for asin in window
            if asin in self.position
        ]
        if not corpora:
            return None

        skip_attributes = skip_attributes or set()
        best: tuple[float, str, list[tuple[str, int]]] | None = None
        for attribute, vocabulary in FACET_VOCABULARY.items():
            # Never re-offer a facet already put to the customer, and never offer
            # a value they have already stated -- both read as not listening.
            if attribute in skip_attributes:
                continue
            counts = [
                (value, sum(1 for text in corpora if value in text))
                for value in vocabulary if value not in disclosed
            ]
            present = [
                (value, count) for value, count in counts
                if count / len(corpora) >= minimum_share
            ]
            if len(present) < 2:
                continue
            present.sort(key=lambda item: item[1], reverse=True)
            # Balance, not volume: two values at 50/50 divide the pool better
            # than one at 90% and another at 15%.
            top, second = present[0][1], present[1][1]
            balance = second / top
            if best is None or balance > best[0]:
                best = (balance, attribute, present[:2])
        return (best[1], best[2]) if best else None

    def _rrf_fuse(self, state: SharedSessionState, ordered: list[int]) -> list[int]:
        """Fuse four lexical routes by reciprocal rank. See RRF_WEIGHTS above.

        `ordered` is the champion (coverage) ordering and doubles as the
        deterministic tie-break, so with all other routes weighted zero this is
        the identity. The broad route may introduce candidates from outside the
        category bucket; they enter the fused list only on the strength of
        whole-catalog token mass, ranked after anything the champion also saw
        unless multiple routes agree.
        """
        constraints = state.constraints
        if not constraints or len(ordered) < 2:
            return ordered
        champion_rank = {index: rank for rank, index in enumerate(ordered)}

        phrase_mass: dict[int, float] = {}
        for index in ordered:
            corpus = self.corpora[index]
            mass = 0.0
            for constraint in constraints:
                text = normalize(constraint)
                if len(text) > 12 and text in corpus:
                    mass += math.log1p(len(text))
            if mass > 0:
                phrase_mass[index] = mass
        phrase = sorted(phrase_mass, key=lambda d: (-phrase_mass[d], champion_rank[d]))

        hard = [
            index for index in ordered
            if all(self._satisfies(index, c) for c in constraints)
        ]
        hard.sort(key=lambda d: -self.priors[d])  # stable: ties keep champion order

        broad = self._by_token_mass(state, RRF_BROAD_DEPTH)

        fused: dict[int, float] = defaultdict(float)
        for name, ranking in (
            ("coverage", ordered), ("phrase", phrase), ("hard", hard), ("broad", broad),
        ):
            weight = RRF_WEIGHTS[name]
            if weight <= 0:
                continue
            for rank, index in enumerate(ranking):
                fused[index] += weight / (RRF_K + rank)
        outside = len(ordered)  # broad-only candidates tie-break after the pool
        return sorted(
            fused, key=lambda d: (-fused[d], champion_rank.get(d, outside), d)
        )

    def _break_ties(
        self, ordered: list[int], state: SharedSessionState, blob: str, k: int
    ) -> list[int]:
        """Reorder ONLY the band of candidates within tie_break_margin of the
        rank-k cutoff, by dense similarity. Everything outside the band --
        including a clear #1 with no real competition -- keeps its lexical
        position untouched.
        """
        if len(ordered) <= k or not state.constraints:
            return ordered
        scores = [self.score(index, state, blob) for index in ordered]
        cutoff = scores[k - 1]
        tolerance = self.tie_break_margin * max(abs(cutoff), 1e-6)

        lo = k - 1
        while lo > 0 and (scores[lo - 1] - cutoff) <= tolerance:
            lo -= 1
        hi = k - 1
        while hi + 1 < len(ordered) and (cutoff - scores[hi + 1]) <= tolerance:
            hi += 1
        if lo == hi:
            return ordered

        band = ordered[lo : hi + 1]
        similarity = self.dense.similarity(self._query_text(state), np.asarray(band))
        band_ranked = [band[i] for i in np.argsort(-similarity)]
        return ordered[:lo] + band_ranked + ordered[hi + 1 :]

    def retrieve(
        self, state: SharedSessionState, k: int, cutoff: int | None = None
    ) -> list[str]:
        """Route -> gather candidates -> rank.

        `k` is how many candidates to return -- callers ask for more than the
        scored top-10 so a reranker downstream has room to reorder. `cutoff`
        is where tie-breaking should anchor: the position that actually gets
        scored (agent.py passes the evaluator's real top_k here). They are NOT
        the same number in production -- agent.py calls retrieve(state, 200)
        for reranker headroom while cutoff stays 10. Reusing `k` as the
        tie-break anchor was a real bug caught before any live measurement:
        the tie-break band formed around position 200, not 10, and did
        nothing useful.

        Filter track (buying/override): the exact category bucket, precision-first.
        Dense track: engaged when the filter track yields nothing -- a reworded or
        unrecognised category -- where it recovers recall the lexical route cannot.
        """
        candidates = self._bucket(state)
        # Must be set before _by_token_mass and score() run: both call
        # _query_terms, whose grounding gate reads it.
        self._grounded_ok = candidates is None
        if candidates is not None and self.broad_pool and state.constraints:
            candidates = sorted(
                set(candidates) | set(self._by_token_mass(state, RRF_BROAD_DEPTH))
            )
        if candidates is None:
            widened = set(self._by_token_mass(state, UNFILTERED_CANDIDATES))
            if self.use_dense:
                width = DENSE_CANDIDATES.get(state.intent, DENSE_CANDIDATES["unknown"])
                similarity = self.dense.similarity(self._query_text(state))
                widened.update(int(i) for i in np.argsort(-similarity)[:width])
            candidates = sorted(widened)

        if self.use_dense and state.turn <= 1 and not state.constraints:
            # Dense track, opening turn only. The customer has said nothing but a
            # category, so there is nothing to match literally and popularity is
            # the only lexical signal left.
            #
            # The guard is on the TURN, not merely on constraints being empty:
            # if message parsing fails, constraints stay empty for the whole
            # session, and keying off that alone strands every later turn in the
            # dense track while the lexical route could have matched the payload
            # still sitting verbatim in the raw message. Measured: that mistake
            # cost 0.800 -> 0.543 at paraphrase level L1.
            array = np.asarray(candidates)
            similarity = self.dense.similarity(self._query_text(state), array)
            ordered = [int(array[i]) for i in np.argsort(-similarity)]
        else:
            # Filter track: rank by literal constraint matching.
            blob = self._blob(state)
            # Set for the whole turn, not just this sort: the reranker and the
            # exposure gate both call score() afterwards, and they must see the
            # same weighting that produced the ranking they are reasoning about.
            # Recomputed on every retrieve(), so it never goes stale across turns.
            self._weights = (
                self._distil_weights(state, candidates) if self.distill else {}
            )
            ordered = sorted(
                candidates, key=lambda index: (-self.score(index, state, blob), index)
            )
            if self.multi_route:
                ordered = self._rrf_fuse(state, ordered)
            if self.use_dense and self.tie_break_dense:
                ordered = self._break_ties(ordered, state, blob, cutoff or k)
        if state.intent == INTENT_BROWSING:
            ordered = self._diversify(ordered, max(k, 1))
        return [self.asins[index] for index in ordered[:k]]
