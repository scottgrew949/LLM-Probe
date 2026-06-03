"""
stimuli/grammars/t1d.py — Stimulus grammar for Thread T1d (causal identification).

─── CONCEPT: What T1d tests ──────────────────────────────────────────────────
T1b established which counterfactual mechanism GPT-2 uses (Lewis or Pearl).
T1d tests a deeper question: does the model's representation respect do-calculus
identification conditions?

Pearl's do-calculus provides formal criteria for when a causal effect can be
identified (estimated) from observational data despite confounding. Two key criteria:

  Back-door criterion: A set Z of observed variables satisfies back-door for
    (X, Y) if Z blocks all back-door paths from X to Y and no Z is a descendant
    of X. If satisfied, P(Y|do(X)) = sum_z P(Y|X,z)P(z).

  Front-door criterion: A set M satisfies front-door for (X, Y) if M blocks
    all directed paths from X to Y, there are no unblocked back-door paths from
    X to M, and all back-door paths from M to Y are blocked by X.

─── CONCEPT: Four conditions ─────────────────────────────────────────────────
back_door_adjustable: confounded, observed covariate Z blocks the back-door path.
  Back-door adjustment valid: P(Y|do(X)) = sum_z P(Y|X,z)P(z).

front_door_adjustable: hidden confounder U affects both X and Y, but mediator M
  (X -> M -> Y) is observed. Front-door adjustment valid via M.

confounded_not_adjustable: confounded, but no valid adjustment set is available.
  The causal effect is not identified from observational data.

unconfounded_control: simple direct causation X -> Y. No confounding.
  Trivially identified. Positive control.

─── CONCEPT: Geometric prediction ────────────────────────────────────────────
If model encodes identifiability:
  adjustable (back_door + front_door) clusters away from not_adjustable.
  run_identification_probe in probes/probes.py tests this binary split.

If model implements Pearl's full do-calculus:
  All four conditions are linearly separable — each causal graph structure
  has a geometrically distinct internal representation.
"""

from __future__ import annotations

import random
from typing import Any


# ── Back-door adjustable templates ───────────────────────────────────────────
# Structure: X -> Y, Z -> {X, Y}. Z observed. Back-door criterion satisfied.

BACK_DOOR_TEMPLATES: list[tuple[str, str]] = [
    (
        "{agent} {treatment_verb} because of {confounder}. "
        "{confounder_capitalized} also directly affects {outcome_noun}. "
        "Adjusting for {confounder}, {agent}'s {treatment_noun} changes {outcome_noun}.",
        "{agent} does not {treatment_verb_base}. "
        "{confounder_capitalized} still affects {outcome_noun} directly. "
        "Without {treatment_noun}, {outcome_noun} differs by {confounder} alone.",
    ),
    (
        "Researchers found that {treatment_noun} increases {outcome_noun}. "
        "Both are influenced by {confounder}. "
        "Controlling for {confounder} isolates the direct effect of {treatment_noun}.",
        "If {confounder} is held constant, {treatment_noun} still predicts {outcome_noun}. "
        "The back-door path through {confounder} is blocked by conditioning. "
        "The adjusted estimate is valid.",
    ),
]

# ── Front-door adjustable templates ──────────────────────────────────────────
# Structure: X -> M -> Y, hidden U -> {X, Y}. M observed. Front-door satisfied.

FRONT_DOOR_TEMPLATES: list[tuple[str, str]] = [
    (
        "{treatment_noun} causes {mediator_noun}. "
        "{mediator_noun} then causes {outcome_noun}. "
        "A hidden factor affects both {treatment_noun} and {outcome_noun} independently.",
        "Even with the hidden factor, {mediator_noun} fully mediates the path. "
        "Front-door adjustment through {mediator_noun} identifies the causal effect. "
        "The effect of {treatment_noun} on {outcome_noun} is estimable.",
    ),
    (
        "{agent} was exposed to {treatment_noun}, which caused {mediator_noun}. "
        "{mediator_noun} led to {outcome_noun}. "
        "Genetic factors independently influenced both exposure and outcome.",
        "The complete pathway runs through {mediator_noun}. "
        "Adjusting for {mediator_noun} at each step recovers the total effect. "
        "No direct {treatment_noun} to {outcome_noun} path bypasses {mediator_noun}.",
    ),
]

# ── Confounded not adjustable templates ──────────────────────────────────────
# Structure: X -> Y, U -> {X, Y}. U unobserved. No valid adjustment set.

CONFOUNDED_NOT_ADJUSTABLE_TEMPLATES: list[tuple[str, str]] = [
    (
        "{treatment_noun} is associated with {outcome_noun}. "
        "An unmeasured factor drives both. "
        "No observed variable can block the confounding path.",
        "The association between {treatment_noun} and {outcome_noun} is not causal. "
        "Without observing the hidden factor, the effect cannot be identified. "
        "The causal effect remains unestimable from this data.",
    ),
    (
        "Studies show {treatment_noun} correlates with {outcome_noun}. "
        "A latent variable affects both {treatment_noun} and {outcome_noun}. "
        "No measured covariate satisfies the back-door criterion.",
        "{agent}'s {treatment_noun} and {outcome_noun} share a hidden common cause. "
        "Conditioning on observed variables does not remove the confounding. "
        "The causal question is not answerable from this observational data.",
    ),
]

# ── Unconfounded control templates ────────────────────────────────────────────
# Structure: X -> Y. No confounding. Trivially identified.

UNCONFOUNDED_TEMPLATES: list[tuple[str, str]] = [
    (
        "{treatment_noun} directly causes {outcome_noun}. "
        "There are no common causes of both. "
        "The causal effect is identifiable without adjustment.",
        "Increasing {treatment_noun} increases {outcome_noun}. "
        "No confounders affect this relationship. "
        "The observational and interventional distributions are identical.",
    ),
    (
        "{agent} applied {treatment_noun}, which produced {outcome_noun}. "
        "No other factor influenced both. "
        "The direct causal path is the only path between them.",
        "Removing {treatment_noun} eliminates {outcome_noun}. "
        "The relationship is purely causal with no confounding structure. "
        "P(Y|X) equals P(Y|do(X)) in this case.",
    ),
]

# ── Domain slots ──────────────────────────────────────────────────────────────

DOMAIN_SLOTS: list[dict[str, str]] = [
    {
        "agent": "John",
        "treatment_verb": "smokes",
        "treatment_verb_base": "smoke",
        "treatment_noun": "smoking",
        "outcome_noun": "lung cancer risk",
        "confounder": "genetic predisposition",
        "confounder_capitalized": "Genetic predisposition",
        "mediator_noun": "tar deposits",
    },
    {
        "agent": "The patient",
        "treatment_verb": "exercises",
        "treatment_verb_base": "exercise",
        "treatment_noun": "exercise",
        "outcome_noun": "cardiovascular health",
        "confounder": "socioeconomic status",
        "confounder_capitalized": "Socioeconomic status",
        "mediator_noun": "reduced inflammation",
    },
    {
        "agent": "The factory",
        "treatment_verb": "emits pollutants",
        "treatment_verb_base": "emit pollutants",
        "treatment_noun": "pollution exposure",
        "outcome_noun": "respiratory disease rates",
        "confounder": "urban density",
        "confounder_capitalized": "Urban density",
        "mediator_noun": "particulate accumulation",
    },
    {
        "agent": "The student",
        "treatment_verb": "attends tutoring",
        "treatment_verb_base": "attend tutoring",
        "treatment_noun": "tutoring",
        "outcome_noun": "exam scores",
        "confounder": "parental income",
        "confounder_capitalized": "Parental income",
        "mediator_noun": "improved study habits",
    },
]


def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n stimulus pairs for T1d — causal identification conditions.

    Pairs are balanced across four conditions: back_door_adjustable,
    front_door_adjustable, confounded_not_adjustable, unconfounded_control.

    Each pair contrasts two sentences with different identification structures.
    sentence_a is the primary condition; sentence_b is a contrast from a
    different condition, so each pair directly tests the distinction.

    Args:
        n: number of pairs to generate.
        seed: random seed for reproducibility.

    Returns:
        List of dicts with keys: sentence_a, sentence_b, label_a, label_b.
    """
    rng = random.Random(seed)

    condition_template_map: dict[str, list[tuple[str, str]]] = {
        "back_door_adjustable": BACK_DOOR_TEMPLATES,
        "front_door_adjustable": FRONT_DOOR_TEMPLATES,
        "confounded_not_adjustable": CONFOUNDED_NOT_ADJUSTABLE_TEMPLATES,
        "unconfounded_control": UNCONFOUNDED_TEMPLATES,
    }
    condition_order = list(condition_template_map.keys())

    pairs: list[dict[str, Any]] = []
    items_per_condition = max(1, n // len(condition_order))

    for condition_label, condition_templates in condition_template_map.items():
        contrast_conditions = [c for c in condition_order if c != condition_label]
        for _ in range(items_per_condition):
            domain_slots = rng.choice(DOMAIN_SLOTS)
            template_a, _ = rng.choice(condition_templates)
            contrast_label = rng.choice(contrast_conditions)
            _, template_b = rng.choice(condition_template_map[contrast_label])

            sentence_a = template_a.format(**domain_slots)
            sentence_b = template_b.format(**domain_slots)

            pairs.append({
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "label_a": condition_label,
                "label_b": contrast_label,
            })

    rng.shuffle(pairs)
    return pairs[:n]
