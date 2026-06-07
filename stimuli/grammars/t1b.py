"""
stimuli/grammars/t1b.py — Stimulus grammar for Thread T1b (Lewis vs Pearl mechanism test).

─── CONCEPT: What T1b tests ──────────────────────────────────────────────────
T1a confirmed Level 3 exists. T1b asks: which mechanism implements it?

Lewis/Stalnaker evaluate counterfactuals by similarity ordering over possible
worlds. Pearl evaluates them by do-calculus on a structural causal model.
These are not the same dispute — they disagree on specific cases where the
frameworks diverge.

─── CONCEPT: Three conditions ────────────────────────────────────────────────
forward_causal — both frameworks agree (positive control):
  "If the match had been struck, the fire would have ignited."
  Both Lewis (closest world where match struck = world where fire ignites)
  and Pearl (do(match_struck=true) → fire ignites) say TRUE.

backtracking — intended Lewis vs Pearl separator (SEE CONSTRUCT WARNING below):
  "If the fire had ignited, the match must have been struck."
  Lewis: bans the backtracking reading via asymmetry-of-similarity weighting.
  Pearl: distinguishes the *interventional* do(fire) (which leaves the match) from
         the *counterfactual* (whose abduction step infers the match WAS struck).

common_cause — Pearl's canonical confound (barometer/storm domain):
  "If the barometer had not fallen, the storm would not have arrived."
  Pearl: do(barometer=steady) does not change atmospheric pressure → storm still
         arrives → the counterfactual is FALSE. Canonical confound, correct.
  Lewis: under standard miracle-weighting Lewis AGREES — hold the past fixed,
         small miracle on the barometer, pressure and storm unchanged → also FALSE.

─── CONSTRUCT WARNING (added after expert review) ─────────────────────────────
Two theoretical defects mean the conditions below may NOT cleanly separate Lewis
from Pearl, and the experiment must be read as provisional:
  1. "Pearl makes backtracking FALSE" is a do()-vs-counterfactual category error.
     Pearl's abduction step also makes the backtracking counterfactual TRUE, so a
     probe separating forward from backtracking may be reading thematic role
     (cause-noun vs effect-noun in the antecedent), not the framework.
  2. Standard Lewis (miracle-weighting) agrees with Pearl on the common_cause case
     (storm still arrives), so that contrast is likely a non-divergence.
A faithful separator probably needs an explicit interventional (do) framing, not
the indicative/subjunctive conditionals here. Reframe before trusting a verdict.

─── CONCEPT: Geometric predictions ───────────────────────────────────────────
If model implements Lewis:
  forward_causal and backtracking cluster together — same worlds-based
  mechanism, only directional framing differs. Common_cause also clusters
  with forward (same similarity reasoning). Linear probe cannot separate
  backtracking from forward_causal.

If model implements Pearl:
  forward_causal is geometrically distinct from backtracking — causal
  graph direction is encoded in the representation. Common_cause is also
  distinct from forward_causal (spurious correlation vs genuine causation).
  Linear probe separates all three conditions.

─── CONCEPT: T1a prerequisite ────────────────────────────────────────────────
T1b ONLY RUNS if T1a summary.json has level3_confirmed=True. This is
enforced by prerequisite_experiment_id in ExperimentConfig, checked by
check_phase_gate() at runtime. The grammar itself does not enforce this.

─── Causal graphs ────────────────────────────────────────────────────────────
Imported from t1a.py — see that module for full SCM documentation.
Critical barometer domain: P → B (pressure → barometer), P → S (pressure →
storm), B ⊥ S | P. Barometer does NOT cause storm. Pearl's canonical confound.
"""

from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path
from typing import Any

# ── Import DOMAINS from t1a (canonical domain vocabulary lives there) ─────────
_T1A_PATH = Path(__file__).parent / "t1a.py"
_t1a_spec = importlib.util.spec_from_file_location("t1a_grammar", _T1A_PATH)
_t1a_module = importlib.util.module_from_spec(_t1a_spec)
_t1a_spec.loader.exec_module(_t1a_module)
DOMAINS: list[dict[str, Any]] = _t1a_module.DOMAINS


# ── Domain augmentation ───────────────────────────────────────────────────────

# effect_past_state: the effect noun in past participial form for backtracking
# templates — "If {effect_noun} had {effect_past_state}, {cause_noun} must have"
_EFFECT_PAST_STATE: dict[str, str] = {
    "match_fire":      "ignited",
    "rain_ground":     "flooded",       # "been wet" (2w) → "flooded" (1w): matches cause_verb_past "fallen" (1w)
    "drug_recovery":   "recovered",
    "fertilizer_crop": "grown",
    "switch_light":    "shone",         # "turned on" (2w) → "shone" (1w): matches cause_verb_past "flipped" (1w); intransitive-grammatical, unlike "lit"
    "watering_plant":  "grown",
    "study_exam":      "passed",        # "passed the exam" (3w) → "passed" (1w): matches cause_verb_past "done" (1w)
    "exercise_fitness":"improved",
    "heater_temp":     "heated up",    # "warmed" (1w) → "heated up" (2w): matches cause_verb_past "turned on" (2w)
    "smoking_cancer":  "developed",
}

# dam_valley excluded: protective direction (dam prevents flooding) makes
# backtracking doubly marked (reversed + negated), confounding the signal.
# barometer_storm excluded from Type A: used separately for common_cause pairs.
_BACKTRACKING_EXCLUDED: frozenset[str] = frozenset({"barometer_storm", "dam_valley"})

_FORWARD_BACKTRACKING_DOMAINS: list[dict[str, Any]] = [
    {**domain, "effect_past_state": _EFFECT_PAST_STATE[domain["domain_id"]]}
    for domain in DOMAINS
    if domain["domain_id"] not in _BACKTRACKING_EXCLUDED
]

_BAROMETER_DOMAIN: dict[str, Any] = {
    **next(d for d in DOMAINS if d["domain_id"] == "barometer_storm"),
    "effect_past_state": "arrived",
}


# ── Templates ─────────────────────────────────────────────────────────────────

FORWARD_TEMPLATES: list[str] = [
    "If {cause_noun} had {cause_verb_past}, {effect_noun} would have",
    "Had {cause_noun} {cause_verb_past}, {effect_noun} would have",
    "Suppose {cause_noun} had {cause_verb_past}; {effect_noun} would have",
    "In a scenario where {cause_noun} had {cause_verb_past}, {effect_noun} would have",
    "If {cause_noun} were to have {cause_verb_past}, {effect_noun} would have",
]

# Structurally identical to FORWARD_TEMPLATES — same sentence frames, cause/effect
# slots swapped. The only difference is causal direction, not surface vocabulary.
# Removing "must have", "Since", "Given that" — all discriminable at layer 0.
BACKTRACKING_TEMPLATES: list[str] = [
    "If {effect_noun} had {effect_past_state}, {cause_noun} would have",
    "Had {effect_noun} {effect_past_state}, {cause_noun} would have",
    "Suppose {effect_noun} had {effect_past_state}; {cause_noun} would have",
    "In a scenario where {effect_noun} had {effect_past_state}, {cause_noun} would have",
    "If {effect_noun} were to have {effect_past_state}, {cause_noun} would have",
]

# Hand-authored complete strings — barometer domain requires specific negation
# syntax that doesn't generalize to a clean format pattern.
COMMON_CAUSE_TEMPLATES: list[str] = [
    "If the barometer had not fallen, the storm would not have",
    "Had the barometer remained steady, the storm would not have",
    "If the barometer had stayed high, the storm would not have",
    "In a world where the barometer had not dropped, the storm would not have",
    "Had someone kept the barometer from falling, the storm would not have",
]

# Type A: 10 domains × 5 × 5 = 250
# Type B:  1 domain  × 5 × 5 =  25
# Type C:  1 domain  × 5 × 5 =  25
# Total = 300
# Note: common_cause condition has 50 sentences vs ~275 each for forward/backtracking.
# Use stratified sampling when training the linear probe.
_MAX_PAIRS: int = 300

_FREQUENCY_NOTES: str = (
    "Backtracking sentences share domain nouns with their forward counterparts — "
    "only the cause/effect verb differs. Barometer pairs (Types B and C) have "
    "near-identical content word sets. V7 expected to pass across all 300 pairs."
)


# ── Behavioral items ──────────────────────────────────────────────────────────
# Three classes — grounded in Pearl's causal canon from "Causality" (2000)
# and "The Book of Why" (2018).
# Class 1: causal direction knowledge (prerequisite)
# Class 2: backtracking conditional reasoning (core T1b test)
# Class 3: common-cause confound awareness (Pearl's canonical separator)

_BEHAVIORAL_ITEMS: list[dict[str, Any]] = [
    # ── Class 1: Causal direction ─────────────────────────────────────────────
    # Tests model knows which entity is cause, which is effect.
    # GPT-2 medium handles direct-cause domains reliably.
    # Item 4 (barometer/storm) excluded — GPT-2 treats spurious correlation
    # as direct causation, making it unreliable. Common-cause confound
    # awareness is tested analytically in the experiment, not the gate.
    {
        "question": "A match is struck and a fire ignites. What caused the fire?",
        "choice_a": "the match being struck",
        "choice_b": "the fire igniting",
        "correct": "a",
    },
    {
        "question": "Rain falls and the ground becomes wet. The wetness of the ground is caused by",
        "choice_a": "the rain",
        "choice_b": "the dry ground",
        "correct": "a",
    },
    {
        "question": "Atmospheric pressure drops and a storm arrives. The storm is caused by",
        "choice_a": "the drop in atmospheric pressure",
        "choice_b": "the barometer falling",
        "correct": "a",
    },
    {
        "question": "When a drug is administered and a patient recovers, what caused the recovery?",
        "choice_a": "the drug being administered",
        "choice_b": "the patient recovering",
        "correct": "a",
    },

    # ── Class 2: Backtracking conditional reasoning ───────────────────────────
    # Tests model tracks what the past state of an effect implies about its cause.
    # Item 6 original ("match must have been ___") failed due to surface pattern
    # "extinguished" — replaced with a completion framing that avoids this.
    {
        "question": "If the ground had been wet, we could infer that it had probably",
        "choice_a": "rained",
        "choice_b": "dried out",
        "correct": "a",
    },
    {
        "question": "If the fire had ignited, we know something must have caused it to",
        "choice_a": "start",
        "choice_b": "go out",
        "correct": "a",
    },
    {
        "question": "If the patient had recovered, the drug was likely",
        "choice_a": "administered",
        "choice_b": "withheld",
        "correct": "a",
    },
    {
        "question": "If the light had turned on, the switch must have been",
        "choice_a": "flipped",
        "choice_b": "broken",
        "correct": "a",
    },
]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n T1b stimulus pairs across three conditions.

    Type A (forward_causal vs backtracking): 10 domains × 5 × 5 = 250 pairs.
    Type B (forward_causal vs common_cause):  1 domain  × 5 × 5 =  25 pairs.
    Type C (backtracking vs common_cause):    1 domain  × 5 × 5 =  25 pairs.

    Common-cause condition has 50 sentences vs ~275 each for the other two
    conditions — use stratified sampling when training the linear probe.

    Args:
        n:    Number of pairs to return. Must be ≤ 300.
        seed: Instance-level RNG seed — no global state side effects.

    Returns:
        List of n StimulusPair dicts conforming to stimulus.schema.json.

    Raises:
        ValueError: if n > 300.
    """
    if n > _MAX_PAIRS:
        raise ValueError(
            f"n={n} exceeds maximum of {_MAX_PAIRS} pairs "
            f"(250 forward/backtracking + 25 forward/common_cause + 25 backtracking/common_cause)."
        )

    all_pairs: list[dict[str, Any]] = []

    def make_pair(sentence_a: str, sentence_b: str, label_a: str, label_b: str, notes: str = "") -> dict[str, Any]:
        pair: dict[str, Any] = {
            "thread_id": "t1b",
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
            "label_a": label_a,
            "label_b": label_b,
            "theoretical_distinction": "Lewis/Stalnaker possible-worlds vs Pearl do-calculus",
            "frequency_matched": False,
            "generation_grammar": "stimuli/grammars/t1b.py",
        }
        if notes:
            pair["notes"] = notes
        return pair

    # Type A: forward_causal vs backtracking
    for domain, fwd_tmpl, bck_tmpl in itertools.product(
        _FORWARD_BACKTRACKING_DOMAINS, FORWARD_TEMPLATES, BACKTRACKING_TEMPLATES
    ):
        sentence_a = fwd_tmpl.format(
            cause_noun=domain["cause_noun"],
            cause_verb_past=domain["cause_verb_past"],
            effect_noun=domain["effect_noun"],
        )
        sentence_b = bck_tmpl.format(
            effect_noun=domain["effect_noun"],
            effect_past_state=domain["effect_past_state"],
            cause_noun=domain["cause_noun"],
        )
        all_pairs.append(make_pair(sentence_a, sentence_b, "forward_causal", "backtracking"))

    # Type B: forward_causal vs common_cause (barometer domain)
    for fwd_tmpl, cc_sentence in itertools.product(FORWARD_TEMPLATES, COMMON_CAUSE_TEMPLATES):
        sentence_a = fwd_tmpl.format(
            cause_noun=_BAROMETER_DOMAIN["cause_noun"],
            cause_verb_past=_BAROMETER_DOMAIN["cause_verb_past"],
            effect_noun=_BAROMETER_DOMAIN["effect_noun"],
        )
        all_pairs.append(make_pair(
            sentence_a, cc_sentence, "forward_causal", "common_cause",
            notes="common_cause condition — use stratified sampling in probe training",
        ))

    # Type C: backtracking vs common_cause (barometer domain)
    for bck_tmpl, cc_sentence in itertools.product(BACKTRACKING_TEMPLATES, COMMON_CAUSE_TEMPLATES):
        sentence_a = bck_tmpl.format(
            effect_noun=_BAROMETER_DOMAIN["effect_noun"],
            effect_past_state=_BAROMETER_DOMAIN["effect_past_state"],
            cause_noun=_BAROMETER_DOMAIN["cause_noun"],
        )
        all_pairs.append(make_pair(
            sentence_a, cc_sentence, "backtracking", "common_cause",
            notes="common_cause condition — use stratified sampling in probe training",
        ))

    random.Random(seed).shuffle(all_pairs)
    selected_pairs = all_pairs[:n]

    for index, pair in enumerate(selected_pairs):
        pair["pair_id"] = f"t1b_{index + 1:04d}"

    return selected_pairs


def generate_behavioral_items() -> list[dict[str, Any]]:
    """
    Return behavioral gate items for T1b.

    Tests causal direction knowledge, backtracking inference, and common-cause
    confound awareness. All grounded in Pearl's canonical causal examples.

    Returns:
        List of 12 forced-choice dicts: question, choice_a, choice_b, correct.
    """
    return list(_BEHAVIORAL_ITEMS)
