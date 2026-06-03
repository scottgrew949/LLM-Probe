"""
stimuli/grammars/t1c.py — Stimulus grammar for Thread T1c (Lewis vs Stalnaker).

─── CONCEPT: What T1c tests ──────────────────────────────────────────────────
T1b established GPT-2 uses Pearl's do-calculus mechanism. T1c asks a finer
question within the worlds-based camp: does the model's counterfactual geometry
match Lewis (similarity ordering) or Stalnaker (single closest world)?

Lewis and Stalnaker both evaluate counterfactuals via possible worlds — they
disagree on the structure of the selection function.

Lewis: "If A were true, C would be true" holds iff C is true in ALL closest
  A-worlds. Borderline cases (worlds equidistant from actual) produce
  indeterminate truth values. Lewis's logic is a variadic conditional — it
  ranges over a set of closest worlds.

Stalnaker: "If A were true, C would be true" holds iff C is true in THE
  closest A-world (single selection). Stalnaker adds the Limit Assumption —
  there is always a unique closest world. Borderline cases have determinate
  truth values (one world wins).

─── CONCEPT: Three conditions ────────────────────────────────────────────────
clear_case — both agree (positive control):
  "If the match had been struck, the fire would have ignited."
  Both Lewis and Stalnaker agree: the closest world where match was struck
  is one where fire ignites. No indeterminacy.

tie_case — Lewis predicts indeterminate, Stalnaker predicts determinate:
  "If the coin had landed heads, it would have been a fair toss."
  Lewis: no unique closest heads-world (fair coin = symmetry). Indeterminate.
  Stalnaker: one world is selected. Determinate truth value.

near_miss — worlds that are nearly identical but diverge on one variable:
  "If the temperature had been 99°C instead of 100°C, the water would not
   have boiled." Tests whether model tracks fine-grained similarity ordering.
  Lewis: depends on full similarity metric. Stalnaker: single closest world.

─── CONCEPT: Geometric predictions ───────────────────────────────────────────
If model implements Lewis:
  tie_case representations are diffuse — no single centroid. The model
  treats the tie as genuinely indeterminate. Geometric variance on tie cases
  is higher than on clear cases.

If model implements Stalnaker:
  tie_case representations converge to a single centroid — the model picks
  one world. Geometric variance on tie cases ≈ clear cases.

─── CONCEPT: T1b prerequisite ────────────────────────────────────────────────
T1c runs only after T1b establishes which coarse-grained mechanism (Lewis vs
Pearl). T1c is within-Lewis — it only matters if T1b finds a worlds-based
mechanism, but we run it regardless for completeness.
"""

from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path
from typing import Any

_T1A_PATH = Path(__file__).parent / "t1a.py"
_t1a_spec = importlib.util.spec_from_file_location("t1a_grammar", _T1A_PATH)
_t1a_module = importlib.util.module_from_spec(_t1a_spec)
_t1a_spec.loader.exec_module(_t1a_module)
DOMAINS: list[dict[str, Any]] = _t1a_module.DOMAINS


_EFFECT_PAST_STATE: dict[str, str] = {
    "match_fire":       "ignited",
    "rain_ground":      "been wet",
    "drug_recovery":    "recovered",
    "fertilizer_crop":  "grown",
    "switch_light":     "turned on",
    "watering_plant":   "grown",
    "study_exam":       "passed the exam",
    "exercise_fitness": "improved",
    "heater_temp":      "warmed",
    "smoking_cancer":   "developed",
    "barometer_storm":  "arrived",
    "dam_valley":       "flooded",
}

_CLEAR_DOMAINS: list[dict[str, Any]] = [
    {**domain, "effect_past_state": _EFFECT_PAST_STATE[domain["domain_id"]]}
    for domain in DOMAINS
    if domain["domain_id"] not in {"barometer_storm"}
]


# ── Templates ─────────────────────────────────────────────────────────────────

CLEAR_CASE_TEMPLATES: list[str] = [
    "If {cause_noun} had {cause_verb_past}, {effect_noun} would have",
    "Had {cause_noun} {cause_verb_past}, {effect_noun} would have",
    "Suppose {cause_noun} had {cause_verb_past}; {effect_noun} would have",
    "In a scenario where {cause_noun} had {cause_verb_past}, {effect_noun} would have",
    "If {cause_noun} were to have {cause_verb_past}, {effect_noun} would have",
]

# Tie cases use symmetric/indeterminate framings — both outcome-A and outcome-B
# sentences equally plausible. Tests whether model represents indeterminacy.
# sentence_a: one determinate completion; sentence_b: equally valid alternative.
TIE_CASE_ITEMS: list[dict[str, str]] = [
    {
        "sentence_a": "If the coin had landed heads, the outcome would have",
        "sentence_b": "If the coin had landed tails, the outcome would have",
        "domain": "fair_coin",
    },
    {
        "sentence_a": "If the die had landed on an even number, the result would have",
        "sentence_b": "If the die had landed on an odd number, the result would have",
        "domain": "fair_die",
    },
    {
        "sentence_a": "If the left path had been chosen, the traveller would have",
        "sentence_b": "If the right path had been chosen, the traveller would have",
        "domain": "symmetric_paths",
    },
    {
        "sentence_a": "If player A had won the symmetric tournament, the prize would have",
        "sentence_b": "If player B had won the symmetric tournament, the prize would have",
        "domain": "symmetric_tournament",
    },
    {
        "sentence_a": "If the first identical twin had been chosen, the result would have",
        "sentence_b": "If the second identical twin had been chosen, the result would have",
        "domain": "identical_twins",
    },
]

# Near-miss cases: worlds almost identical to actual, one variable differs.
# Tests fine-grained similarity ordering — Lewis needs a full metric.
NEAR_MISS_ITEMS: list[dict[str, str]] = [
    {
        "sentence_a": "If the temperature had been 99 degrees instead of 100, the water would have",
        "sentence_b": "If the temperature had been 101 degrees instead of 100, the water would have",
        "domain": "boiling_point",
    },
    {
        "sentence_a": "If the speed had been one unit below the threshold, the car would have",
        "sentence_b": "If the speed had been one unit above the threshold, the car would have",
        "domain": "speed_threshold",
    },
    {
        "sentence_a": "If the score had been one point less than the passing mark, the student would have",
        "sentence_b": "If the score had been one point more than the passing mark, the student would have",
        "domain": "passing_mark",
    },
    {
        "sentence_a": "If the pressure had dropped one unit less, the seal would have",
        "sentence_b": "If the pressure had dropped one unit more, the seal would have",
        "domain": "pressure_seal",
    },
    {
        "sentence_a": "If the dose had been one milligram below the threshold, the effect would have",
        "sentence_b": "If the dose had been one milligram above the threshold, the effect would have",
        "domain": "dose_threshold",
    },
]

_MAX_PAIRS: int = 200


def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n T1c stimulus pairs across three conditions.

    clear_case: unambiguous counterfactuals from T1a domains.
    tie_case: symmetric worlds — Lewis predicts indeterminacy, Stalnaker does not.
    near_miss: worlds differing by one variable — tests fine-grained similarity.

    Args:
        n: number of pairs to generate. Must be <= 200.
        seed: random seed for reproducibility.

    Returns:
        List of n StimulusPair dicts conforming to stimulus.schema.json.
    """
    if n > _MAX_PAIRS:
        raise ValueError(f"n={n} exceeds maximum of {_MAX_PAIRS} pairs.")

    rng = random.Random(seed)
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

    # Clear cases: pair two clear-case sentences from different domains
    for domain_a, domain_b, tmpl_a, tmpl_b in itertools.product(
        _CLEAR_DOMAINS[:5], _CLEAR_DOMAINS[5:], CLEAR_CASE_TEMPLATES[:3], CLEAR_CASE_TEMPLATES[2:]
    ):
        sentence_a = tmpl_a.format(
            cause_noun=domain_a["cause_noun"],
            cause_verb_past=domain_a["cause_verb_past"],
            effect_noun=domain_a["effect_noun"],
        )
        sentence_b = tmpl_b.format(
            cause_noun=domain_b["cause_noun"],
            cause_verb_past=domain_b["cause_verb_past"],
            effect_noun=domain_b["effect_noun"],
        )
        all_pairs.append(make_pair(sentence_a, sentence_b, "clear_case", "clear_case"))

    # Tie cases: pair sentence_a with sentence_b from tie items
    for item in TIE_CASE_ITEMS:
        for tmpl in CLEAR_CASE_TEMPLATES:
            domain = rng.choice(_CLEAR_DOMAINS)
            clear_sentence = tmpl.format(
                cause_noun=domain["cause_noun"],
                cause_verb_past=domain["cause_verb_past"],
                effect_noun=domain["effect_noun"],
            )
            all_pairs.append(make_pair(
                item["sentence_a"], item["sentence_b"], "tie_case", "tie_case",
                notes=f"symmetric domain: {item['domain']}"
            ))
            all_pairs.append(make_pair(
                clear_sentence, item["sentence_a"], "clear_case", "tie_case",
                notes=f"clear vs tie — Lewis/Stalnaker discriminator"
            ))

    # Near-miss cases
    for item in NEAR_MISS_ITEMS:
        for tmpl in CLEAR_CASE_TEMPLATES:
            domain = rng.choice(_CLEAR_DOMAINS)
            clear_sentence = tmpl.format(
                cause_noun=domain["cause_noun"],
                cause_verb_past=domain["cause_verb_past"],
                effect_noun=domain["effect_noun"],
            )
            all_pairs.append(make_pair(
                item["sentence_a"], item["sentence_b"], "near_miss", "near_miss",
                notes=f"near-miss domain: {item['domain']}"
            ))
            all_pairs.append(make_pair(
                clear_sentence, item["sentence_a"], "clear_case", "near_miss",
                notes="clear vs near-miss — similarity ordering test"
            ))

    rng.shuffle(all_pairs)
    selected = all_pairs[:n]

    for idx, pair in enumerate(selected):
        pair["pair_id"] = f"t1c_{idx + 1:04d}"

    return selected


def generate_behavioral_items() -> list[dict[str, Any]]:
    """Return behavioral gate items for T1c."""
    return [
        {
            "question": "If the water temperature had been 99 degrees instead of 100, the water would have",
            "choice_a": "not boiled",
            "choice_b": "boiled",
            "correct": "a",
        },
        {
            "question": "If the student's score had been just above the passing mark, the student would have",
            "choice_a": "passed",
            "choice_b": "failed",
            "correct": "a",
        },
        {
            "question": "If the match had been struck, the fire would have",
            "choice_a": "ignited",
            "choice_b": "gone out",
            "correct": "a",
        },
        {
            "question": "If the drug had been administered, the patient would have",
            "choice_a": "recovered",
            "choice_b": "worsened",
            "correct": "a",
        },
    ]
