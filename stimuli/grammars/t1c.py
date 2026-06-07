"""
stimuli/grammars/t1c.py — Stimulus grammar for Thread T1c (Lewis vs Stalnaker).

─── CONCEPT: What T1c tests ──────────────────────────────────────────────────
T1b established the coarse mechanism (worlds-based vs Pearl). T1c asks the finer
question *within* the worlds-based camp: does the model's counterfactual geometry
match Lewis (similarity-set selection) or Stalnaker (single closest-world)?

Lewis: "If A were true, C would be true" holds iff C holds in ALL closest
  A-worlds. At a symmetric tie (no unique closest world) the truth value is
  INDETERMINATE — the conditional ranges over a *set* of equally-close worlds.

Stalnaker: holds iff C holds in THE closest A-world (single selection + Limit
  Assumption — there is always a unique closest world). Ties are still
  DETERMINATE — one world is picked.

─── CONCEPT: The discriminator is DISPERSION, not separability ───────────────
A probe that separates "clear" from "tie" sentences proves only that the two
sentence-types differ — true under BOTH theories. It cannot distinguish Lewis
from Stalnaker.

The theories diverge on the GEOMETRY of the tie representations:

  Lewis      → tie activations are DIFFUSE. No unique world is selected, so the
               representation spreads over the set of equally-close worlds.
               dispersion(tie) > dispersion(clear).

  Stalnaker  → tie activations COLLAPSE to a centroid. One world is picked, so
               ties are as tightly clustered as determinate clear cases.
               dispersion(tie) ≈ dispersion(clear).

The experiment therefore measures a dispersion ratio, not a probe accuracy.

─── CONCEPT: Why minimal pairs (matched diversity) ───────────────────────────
A dispersion ratio is only interpretable if clear and tie have the SAME
underlying topical diversity. If tie sentences span coin/die/path while clear
sentences span match/rain/drug, tie variance is inflated by topic spread — you
would misread topical diversity as Lewisian indeterminacy.

So clear and tie are MINIMAL PAIRS over the same domains: identical template,
identical noun, identical action — differing only in an adjective that flips the
antecedent between determinate (asymmetric) and indeterminate (symmetric):

  clear_case (determinate):   "If the loaded coin had been flipped, the outcome would have"
  tie_case   (indeterminate): "If the fair coin had been flipped, the outcome would have"

Matched length, matched topic, matched vocabulary. Any residual dispersion gap
is the indeterminacy signal, not a confound.

─── CONCEPT: Lexically varied bias cue ───────────────────────────────────────
If every determinate sentence said "loaded" and every tie said "fair", the probe
could read those two words instead of the structural property. Each domain
therefore draws the cue from a LIST of synonyms — {loaded, weighted, rigged,
biased, crooked} vs {fair, balanced, unbiased, even, ordinary} — so the
distinction the geometry must encode is the *semantic category* (asymmetric vs
symmetric), not a single lexical item. The cue diversity is matched across the
two classes (5 adjectives each), so it cancels in the dispersion ratio.

─── CONCEPT: near_miss (secondary) ───────────────────────────────────────────
near_miss probes fine-grained similarity ordering: two worlds just below / just
above a threshold, both close to actual. Under Lewis the proximity metric orders
them; under Stalnaker each resolves to its own closest world. Secondary to the
clear-vs-tie dispersion test.
"""

from __future__ import annotations

import itertools
import random
from typing import Any


# ── Tie domains: minimal-pair cores ───────────────────────────────────────────
# Each domain crosses a determinate (asymmetric) adjective against an
# indeterminate (symmetric) adjective over a shared clause core. The pair
# differs in exactly one word — the adjective — so clear_case and tie_case are
# matched on topic, length, and structure.
#
# asymmetric_adjs → clear_case  (unique closest world: a biased coin has a
#                                determinate most-likely outcome)
# symmetric_adjs  → tie_case    (no unique closest world: a fair coin is
#                                symmetric between outcomes — Lewis-indeterminate)

_TIE_DOMAINS: list[dict[str, Any]] = [
    {
        "domain_id": "coin",
        "noun": "coin",
        "action_past": "been flipped",
        "tail": "the outcome",
        "asymmetric_adjs": ["loaded", "weighted", "rigged", "biased", "crooked"],
        "symmetric_adjs":  ["fair", "balanced", "unbiased", "even", "ordinary"],
    },
    {
        "domain_id": "die",
        "noun": "die",
        "action_past": "been rolled",
        "tail": "the result",
        "asymmetric_adjs": ["loaded", "weighted", "rigged", "shaved", "biased"],
        "symmetric_adjs":  ["fair", "balanced", "unbiased", "even", "standard"],
    },
    {
        "domain_id": "route",
        "noun": "route",
        "action_past": "been taken",
        "tail": "the traveller",
        "asymmetric_adjs": ["shorter", "faster", "quicker", "steeper", "longer"],
        "symmetric_adjs":  ["equal", "identical", "parallel", "matching", "twin"],
    },
    {
        "domain_id": "match",
        "noun": "match",
        "action_past": "been replayed",
        "tail": "the winner",
        "asymmetric_adjs": ["lopsided", "uneven", "mismatched", "rigged", "fixed"],
        "symmetric_adjs":  ["even", "tied", "balanced", "close", "level"],
    },
    {
        "domain_id": "twins",
        "noun": "twin",
        "action_past": "been chosen",
        "tail": "the result",
        "asymmetric_adjs": ["taller", "older", "stronger", "smarter", "heavier"],
        "symmetric_adjs":  ["identical", "matching", "indistinct", "similar", "alike"],
    },
]

# Shared frames — structurally identical to the T1a/T1b counterfactual templates,
# adjective slot added. {adj} {noun} carries the entire manipulation.
_TIE_TEMPLATES: list[str] = [
    "If the {adj} {noun} had {action_past}, {tail} would have",
    "Had the {adj} {noun} {action_past}, {tail} would have",
    "Suppose the {adj} {noun} had {action_past}; {tail} would have",
    "In a scenario where the {adj} {noun} had {action_past}, {tail} would have",
    "If the {adj} {noun} were to have {action_past}, {tail} would have",
]


# ── near_miss domains: just-below vs just-above a threshold ────────────────────
# Both sides are one step from the actual threshold — symmetric near-misses.
# Secondary condition: tests fine-grained similarity ordering, not the primary
# Lewis/Stalnaker dispersion test.

_NEAR_MISS_DOMAINS: list[dict[str, str]] = [
    {"domain_id": "boiling_point", "below": "one degree below boiling",  "above": "one degree above boiling",  "subject": "the temperature", "tail": "the water"},
    {"domain_id": "speed_limit",   "below": "one unit below the limit",  "above": "one unit above the limit",  "subject": "the speed",       "tail": "the driver"},
    {"domain_id": "passing_mark",  "below": "one point below the mark",  "above": "one point above the mark",  "subject": "the score",       "tail": "the student"},
    {"domain_id": "seal_pressure", "below": "one unit below the rating", "above": "one unit above the rating", "subject": "the pressure",    "tail": "the seal"},
    {"domain_id": "dose_threshold","below": "one milligram below the threshold", "above": "one milligram above the threshold", "subject": "the dose", "tail": "the effect"},
]

_NEAR_MISS_TEMPLATES: list[str] = [
    "If {subject} had been {delta}, {tail} would have",
    "Had {subject} been {delta}, {tail} would have",
    "Suppose {subject} had been {delta}; {tail} would have",
    "In a scenario where {subject} had been {delta}, {tail} would have",
    "If {subject} were to have been {delta}, {tail} would have",
]


# clear/tie minimal pairs: 5 domains × 5 templates × 5 adjective-pairs = 125
# near_miss pairs:         5 domains × 5 templates                     =  25
_MAX_PAIRS: int = 150


def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n T1c stimulus pairs.

    clear_case vs tie_case — minimal pairs over shared domains, differing only in
      a determinate (asymmetric) vs indeterminate (symmetric) adjective. This is
      the primary dispersion test: under Lewis, tie activations are more diffuse
      than clear; under Stalnaker they collapse to a centroid like clear.

    near_miss — just-below vs just-above a threshold (secondary similarity-ordering
      condition).

    Pair label conventions:
      ("clear_case", "tie_case")   — sentence_a determinate, sentence_b symmetric
      ("near_miss",  "near_miss")  — sentence_a just-below, sentence_b just-above

    clear_case and tie_case are produced in equal numbers (125 each) so the
    dispersion comparison is over matched-size, matched-diversity sets.

    Args:
        n:    Number of pairs to return. Must be <= 150.
        seed: Instance-level RNG seed — no global state side effects.

    Returns:
        List of n StimulusPair dicts conforming to stimulus.schema.json.

    Raises:
        ValueError: if n > 150.
    """
    if n > _MAX_PAIRS:
        raise ValueError(f"n={n} exceeds maximum of {_MAX_PAIRS} pairs.")

    all_pairs: list[dict[str, Any]] = []

    def make_pair(sentence_a: str, sentence_b: str, label_a: str, label_b: str, notes: str = "") -> dict[str, Any]:
        pair: dict[str, Any] = {
            "thread_id": "t1c",
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
            "label_a": label_a,
            "label_b": label_b,
            "theoretical_distinction": "Lewis similarity-set vs Stalnaker single-selection",
            "frequency_matched": False,
            "generation_grammar": "stimuli/grammars/t1c.py",
        }
        if notes:
            pair["notes"] = notes
        return pair

    # clear_case vs tie_case — minimal pairs (adjective is the only difference)
    for domain in _TIE_DOMAINS:
        adjective_pairs = list(zip(domain["asymmetric_adjs"], domain["symmetric_adjs"]))
        for template, (asymmetric_adj, symmetric_adj) in itertools.product(_TIE_TEMPLATES, adjective_pairs):
            clear_sentence = template.format(
                adj=asymmetric_adj, noun=domain["noun"],
                action_past=domain["action_past"], tail=domain["tail"],
            )
            tie_sentence = template.format(
                adj=symmetric_adj, noun=domain["noun"],
                action_past=domain["action_past"], tail=domain["tail"],
            )
            all_pairs.append(make_pair(
                clear_sentence, tie_sentence, "clear_case", "tie_case",
                notes=f"minimal pair ({domain['domain_id']}): '{asymmetric_adj}' determinate vs '{symmetric_adj}' symmetric",
            ))

    # near_miss vs near_miss — symmetric just-below / just-above a threshold
    for domain in _NEAR_MISS_DOMAINS:
        for template in _NEAR_MISS_TEMPLATES:
            below_sentence = template.format(subject=domain["subject"], delta=domain["below"], tail=domain["tail"])
            above_sentence = template.format(subject=domain["subject"], delta=domain["above"], tail=domain["tail"])
            all_pairs.append(make_pair(
                below_sentence, above_sentence, "near_miss", "near_miss",
                notes=f"near-miss ({domain['domain_id']}): just-below vs just-above threshold",
            ))

    random.Random(seed).shuffle(all_pairs)
    selected_pairs = all_pairs[:n]

    for index, pair in enumerate(selected_pairs):
        pair["pair_id"] = f"t1c_{index + 1:04d}"

    return selected_pairs


def generate_behavioral_items() -> list[dict[str, Any]]:
    """
    Return behavioral gate items for T1c.

    Tests that the model handles the DETERMINATE (asymmetric) cases correctly —
    competence on the clear pole of the minimal pairs. The symmetric ties are
    deliberately not gated: their answer is indeterminate, so there is no correct
    forced choice to score.

    Returns:
        List of 4 forced-choice dicts: question, choice_a, choice_b, correct.
    """
    return [
        {
            "question": "A coin is loaded to favor heads. If it had been flipped, it would most likely have landed",
            "choice_a": "heads",
            "choice_b": "tails",
            "correct": "a",
        },
        {
            "question": "One route is clearly shorter than the other. If the shorter route had been taken, the traveller would have arrived",
            "choice_a": "sooner",
            "choice_b": "later",
            "correct": "a",
        },
        {
            "question": "A die is weighted toward six. If it had been rolled, the result would most likely have been",
            "choice_a": "a six",
            "choice_b": "a one",
            "correct": "a",
        },
        {
            "question": "In a lopsided match the stronger team dominates. If the lopsided match had been replayed, the winner would most likely have been",
            "choice_a": "the stronger team",
            "choice_b": "the weaker team",
            "correct": "a",
        },
    ]
