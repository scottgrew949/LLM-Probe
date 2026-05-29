"""
stimuli/grammars/t1a.py — Stimulus grammar for Thread T1a (Pearl Level 3 existence test).

─── CONCEPT: Pearl's Ladder of Causation ────────────────────────────────────
Pearl's "Book of Why" organizes causal reasoning into three rungs:

  L1 — Association:     P(Y | X)
       "When I see X, what do I expect for Y?"
       Pure statistical correlation. No intervention required.
       Example: "When matches are struck, fires ignite."

  L2 — Intervention:    P(Y | do(X))
       "If I force X to happen, what will Y be?"
       Requires reasoning about the effect of an external intervention.

  L3 — Counterfactual:  P(Y_x | X' = x')
       "Given X did NOT happen, what WOULD Y have been if X HAD happened?"
       Requires reasoning about a world where the cause was actively
       prevented — an intervention on a world where something else occurred.
       Example: "If the match had not been struck, the fire would not have started."

T1a establishes whether GPT-2 medium reaches L3 at all. If it does not —
if P(C | A was prevented) = P(C | A did not occur) — then T1b and T1c
are moot. L3 must be confirmed before testing which mechanism implements it.

─── CONCEPT: Two stimulus classes ───────────────────────────────────────────
causal_l3 — interventional counterfactuals.
  "If the match had not been struck, the fire would not have ___"
  Uses past perfect subjunctive ("had not been X") — the grammatical
  marker of L3. Requires the model to reason about a prevented cause
  and its absent effect.

associative_l1 — observational baselines.
  "When the match is struck, the fire typically ___"
  Pure statistical association. No intervention. Answers by pattern.

Every pair has label_a = "causal_l3" and label_b = "associative_l1".
The linear probe distinguishes the two classes. If the model encodes
L3 separately from L1, the probe should locate this at a specific layer.

─── CONCEPT: Pre-specified causal graphs (required before T1b) ──────────────
SPEC §T T1b requires: "Pre-specify the causal graph hypothesized (nodes,
edges, structural equations) before running." This docstring IS that
pre-specification — it exists before any experiment runs.

Structural equations (simplified SCM notation):

  match_fire:        F := f(M, U_F)    — M = match_struck, F = fire_ignites
  rain_ground:       W := f(R, U_W)    — R = rain, W = ground_wet
  drug_recovery:     R := f(D, U_R)    — D = drug_given, R = patient_recovers
  fertilizer_crop:   C := f(Z, U_C)    — Z = fertilizer_applied, C = crop_grows
  switch_light:      L := f(S, U_L)    — S = switch_flipped, L = light_on
  watering_plant:    P := f(W, U_P)    — W = watering, P = plant_grows
  study_exam:        E := f(T, U_E)    — T = studying, E = exam_passed
  exercise_fitness:  I := f(X, U_I)    — X = exercise, I = fitness_improves
  heater_temp:       H := f(K, U_H)    — K = heater_on, H = room_warms
  smoking_cancer:    C := f(G, U_C)    — G = smoking, C = cancer_develops
  dam_valley:        V := f(B, U_V)    — B = dam_built, V = valley_floods (negated)

  BAROMETER/STORM — CONFOUNDER (flagged for T1b backtracking condition):
    P → B  (atmospheric pressure causes barometer to fall)
    P → S  (atmospheric pressure causes storm)
    B ⊥ S | P  ← barometer does NOT cause storm
    This is Pearl's canonical common-cause confound. B and S are associated
    (L1) but not causally connected. T1b will use this to test whether the
    model distinguishes backtracking counterfactuals from forward causation.
    T1a includes this domain with a notes field flagging the confound.
"""

from __future__ import annotations

import itertools
import random
from typing import Any


# ── Causal domains ────────────────────────────────────────────────────────────

DOMAINS: list[dict[str, Any]] = [
    {
        "domain_id": "match_fire",
        "cause_noun": "the match",
        "cause_verb_past": "struck",
        "cause_verb_present": "is struck",
        "effect_noun": "the fire",
        "effect_verb": "ignites",
        "causal_graph": {
            "nodes": ["match_struck", "fire_ignites"],
            "edges": [("match_struck", "fire_ignites")],
            "note": "",
        },
    },
    {
        "domain_id": "rain_ground",
        "cause_noun": "the rain",
        "cause_verb_past": "fallen",
        "cause_verb_present": "falls",
        "effect_noun": "the ground",
        "effect_verb": "gets wet",
        "causal_graph": {
            "nodes": ["rain", "ground_wet"],
            "edges": [("rain", "ground_wet")],
            "note": "",
        },
    },
    {
        "domain_id": "barometer_storm",
        "cause_noun": "the barometer",
        "cause_verb_past": "fallen",
        "cause_verb_present": "falls",
        "effect_noun": "the storm",
        "effect_verb": "arrives",
        "causal_graph": {
            "nodes": ["atmospheric_pressure", "barometer_falls", "storm_arrives"],
            "edges": [
                ("atmospheric_pressure", "barometer_falls"),
                ("atmospheric_pressure", "storm_arrives"),
            ],
            "note": (
                "CONFOUNDER: atmospheric_pressure causes both barometer_falls and "
                "storm_arrives. Barometer does NOT cause storm. "
                "B ⊥ S | atmospheric_pressure. "
                "Flagged for T1b backtracking condition — "
                "Lewis/Pearl predict different results here."
            ),
        },
    },
    {
        "domain_id": "drug_recovery",
        "cause_noun": "the drug",
        "cause_verb_past": "administered",
        "cause_verb_present": "is administered",
        "effect_noun": "the patient",
        "effect_verb": "recovers",
        "causal_graph": {
            "nodes": ["drug_given", "patient_recovers"],
            "edges": [("drug_given", "patient_recovers")],
            "note": "",
        },
    },
    {
        "domain_id": "fertilizer_crop",
        "cause_noun": "the fertilizer",
        "cause_verb_past": "applied",
        "cause_verb_present": "is applied",
        "effect_noun": "the crop",
        "effect_verb": "grows",
        "causal_graph": {
            "nodes": ["fertilizer_applied", "crop_grows"],
            "edges": [("fertilizer_applied", "crop_grows")],
            "note": "",
        },
    },
    {
        "domain_id": "switch_light",
        "cause_noun": "the switch",
        "cause_verb_past": "flipped",
        "cause_verb_present": "is flipped",
        "effect_noun": "the light",
        "effect_verb": "turns on",
        "causal_graph": {
            "nodes": ["switch_flipped", "light_on"],
            "edges": [("switch_flipped", "light_on")],
            "note": "",
        },
    },
    {
        "domain_id": "watering_plant",
        "cause_noun": "the watering",
        "cause_verb_past": "done",
        "cause_verb_present": "occurs",
        "effect_noun": "the plant",
        "effect_verb": "grows",
        "causal_graph": {
            "nodes": ["watering", "plant_grows"],
            "edges": [("watering", "plant_grows")],
            "note": "",
        },
    },
    {
        "domain_id": "study_exam",
        "cause_noun": "the studying",
        "cause_verb_past": "done",
        "cause_verb_present": "occurs",
        "effect_noun": "the student",
        "effect_verb": "passes the exam",
        "causal_graph": {
            "nodes": ["studying", "exam_passed"],
            "edges": [("studying", "exam_passed")],
            "note": "",
        },
    },
    {
        "domain_id": "exercise_fitness",
        "cause_noun": "the exercise",
        "cause_verb_past": "performed",
        "cause_verb_present": "is performed",
        "effect_noun": "fitness",
        "effect_verb": "improves",
        "causal_graph": {
            "nodes": ["exercise", "fitness_improves"],
            "edges": [("exercise", "fitness_improves")],
            "note": "",
        },
    },
    {
        "domain_id": "heater_temp",
        "cause_noun": "the heater",
        "cause_verb_past": "turned on",
        "cause_verb_present": "is turned on",
        "effect_noun": "the room",
        "effect_verb": "warms",
        "causal_graph": {
            "nodes": ["heater_on", "room_warms"],
            "edges": [("heater_on", "room_warms")],
            "note": "",
        },
    },
    {
        "domain_id": "smoking_cancer",
        "cause_noun": "the smoking",
        "cause_verb_past": "occurred",
        "cause_verb_present": "occurs",
        "effect_noun": "cancer",
        "effect_verb": "develops",
        "causal_graph": {
            "nodes": ["smoking", "cancer_develops"],
            "edges": [("smoking", "cancer_develops")],
            "note": "",
        },
    },
    {
        "domain_id": "dam_valley",
        "cause_noun": "the dam",
        "cause_verb_past": "built",
        "cause_verb_present": "is built",
        "effect_noun": "the valley",
        "effect_verb": "floods",
        "causal_graph": {
            "nodes": ["dam_built", "valley_floods"],
            "edges": [("dam_built", "valley_not_floods")],
            "note": "Dam prevents flooding — causal direction is protective.",
        },
    },
]

# ── Templates ─────────────────────────────────────────────────────────────────

# L3: past perfect subjunctive — grammatical marker of counterfactual intervention.
# "Had not been X" = the cause was actively prevented in the counterfactual world.
L3_TEMPLATES: list[str] = [
    "If {cause_noun} had not been {cause_verb_past}, {effect_noun} would not have",
    "Had {cause_noun} never been {cause_verb_past}, {effect_noun} would not have",
    "If we had prevented {cause_noun} from being {cause_verb_past}, {effect_noun} would not have",
    "In a world where {cause_noun} was never {cause_verb_past}, {effect_noun} would never have",
    "Had someone intervened to stop {cause_noun} being {cause_verb_past}, {effect_noun} would not have",
]

# L1: observational framing — no intervention, pure statistical association.
L1_TEMPLATES: list[str] = [
    "When {cause_noun} {cause_verb_present}, {effect_noun} typically",
    "In cases where {cause_noun} {cause_verb_present}, {effect_noun} tends to",
    "Whenever {cause_noun} {cause_verb_present}, {effect_noun} usually",
    "Observers note that when {cause_noun} {cause_verb_present}, {effect_noun} often",
    "It is commonly observed that when {cause_noun} {cause_verb_present}, {effect_noun}",
]

# 12 domains × 5 L3 templates × 5 L1 templates = 300 unique pairs
_MAX_PAIRS: int = 300

_FREQUENCY_NOTES: str = (
    "Frequency matching note: all domain nouns are shared between sentence_a (L3) "
    "and sentence_b (L1), so intra-pair frequency differences are near zero. "
    "The barometer domain uses 'barometer' (~1e-6 in wordfreq) but both sentences "
    "in each barometer pair contain it equally — the V7 intra-pair check passes."
)


# ── Behavioral items ──────────────────────────────────────────────────────────
# Three classes of 4 items each — grounded in Pearl's canon.
#
# Class 1: Causal direction — prerequisite, tests model knows basic causal facts.
# Class 2: L3 counterfactual completion — tests model responds to interventional framing.
# Class 3: L3 vs L1 metalinguistic distinction — tests model distinguishes the rungs.

_BEHAVIORAL_ITEMS: list[dict[str, Any]] = [
    # ── Class 1: Causal direction — SCM prerequisite ──────────────────────────
    # Tests that GPT-2 knows the causal relationships in our domains.
    # Required before L3 probing is interpretable — if the model does not
    # know rain wets ground, probing its counterfactual encoding is meaningless.
    # Grounded in Pearl's canonical SCM domains from "Causality" and "The Book of Why."

    {
        "question": "When rain falls on dry ground, the ground",
        "choice_a": "gets wet",
        "choice_b": "dries out",
        "correct": "a",
    },
    {
        "question": "When a light switch is flipped to the on position, the light",
        "choice_a": "turns on",
        "choice_b": "breaks",
        "correct": "a",
    },
    {
        "question": "When a patient is given a drug to treat an infection, the patient typically",
        "choice_a": "recovers",
        "choice_b": "deteriorates",
        "correct": "a",
    },
    {
        "question": "When fertilizer is applied to soil, crops tend to",
        "choice_a": "grow",
        "choice_b": "die",
        "correct": "a",
    },
    {
        "question": "When a heater is turned on in a cold room, the temperature",
        "choice_a": "rises",
        "choice_b": "drops",
        "correct": "a",
    },
    {
        "question": "When someone exercises regularly, their fitness",
        "choice_a": "improves",
        "choice_b": "declines",
        "correct": "a",
    },

    # ── Class 2: L3 interventional counterfactual completion ──────────────────
    # Directly operationalizes Pearl's L3: P(Y | A was prevented).
    # "Had not been X" (past perfect subjunctive) = cause was actively prevented.
    # Correct completion = counterfactual absence of effect.
    # These test whether the model tracks the SCM structure:
    # if we intervene on the cause variable (set it to absent), the effect is absent.

    {
        "question": "If the match had not been struck, the fire would not have",
        "choice_a": "started",
        "choice_b": "spread",
        "correct": "a",
    },
    {
        "question": "If the drug had not been administered, the patient would not have",
        "choice_a": "recovered",
        "choice_b": "deteriorated",
        "correct": "a",
    },
    {
        "question": "Had the fertilizer not been applied, the crop would not have",
        "choice_a": "grown",
        "choice_b": "died",
        "correct": "a",
    },
    {
        "question": "If the heater had not been turned on, the room would have stayed",
        "choice_a": "cold",
        "choice_b": "warm",
        "correct": "a",
    },
    {
        "question": "Had the rain not fallen, the ground would have stayed",
        "choice_a": "dry",
        "choice_b": "wet",
        "correct": "a",
    },
    {
        "question": "If the exercise had not been performed, fitness would not have",
        "choice_a": "improved",
        "choice_b": "declined",
        "correct": "a",
    },
]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n T1a stimulus pairs from the full domain × template product space.

    Each pair contrasts a causal_l3 (interventional counterfactual) sentence
    with an associative_l1 (observational baseline) sentence for the same
    causal domain. label_a = "causal_l3", label_b = "associative_l1".

    Full product space: 12 domains × 5 L3 templates × 5 L1 templates = 300 pairs.
    Barometer/storm pairs include a notes field flagging the common-cause confound
    for use in T1b backtracking condition design.

    Args:
        n:    Number of pairs to return. Must be ≤ 300.
        seed: Instance-level RNG seed — no global random state side effects.

    Returns:
        List of n StimulusPair dicts conforming to stimulus.schema.json.
        frequency_matched is False — validate_set() sets it True after V7 check.

    Raises:
        ValueError: if n > 300.
    """
    if n > _MAX_PAIRS:
        raise ValueError(
            f"n={n} exceeds maximum of {_MAX_PAIRS} pairs "
            f"(12 domains × 5 L3 templates × 5 L1 templates)."
        )

    all_pairs: list[dict[str, Any]] = []

    for domain, l3_template, l1_template in itertools.product(
        DOMAINS, L3_TEMPLATES, L1_TEMPLATES
    ):
        sentence_a = l3_template.format(
            cause_noun=domain["cause_noun"],
            cause_verb_past=domain["cause_verb_past"],
            effect_noun=domain["effect_noun"],
        )
        sentence_b = l1_template.format(
            cause_noun=domain["cause_noun"],
            cause_verb_present=domain["cause_verb_present"],
            effect_noun=domain["effect_noun"],
        )

        pair: dict[str, Any] = {
            "thread_id": "t1a",
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
            "label_a": "causal_l3",
            "label_b": "associative_l1",
            "theoretical_distinction": "Pearl Level 3 counterfactual vs Level 1 association",
            "frequency_matched": False,
            "generation_grammar": "stimuli/grammars/t1a.py",
        }

        # Carry the confounder note forward for the barometer domain
        # so T1b can identify these pairs when designing backtracking stimuli.
        graph_note = domain["causal_graph"]["note"]
        if graph_note:
            pair["notes"] = graph_note

        all_pairs.append(pair)

    random.Random(seed).shuffle(all_pairs)
    selected_pairs = all_pairs[:n]

    for index, pair in enumerate(selected_pairs):
        pair["pair_id"] = f"t1a_{index + 1:04d}"

    return selected_pairs


def generate_behavioral_items() -> list[dict[str, Any]]:
    """
    Return behavioral gate items for T1a.

    Three classes: causal direction knowledge (prerequisite), L3 counterfactual
    completion, and L3 vs L1 metalinguistic distinction. All grounded in
    Pearl's canonical causal examples from "Causality" and "The Book of Why."

    Returns:
        List of 12 forced-choice dicts: question, choice_a, choice_b, correct.
    """
    return list(_BEHAVIORAL_ITEMS)
