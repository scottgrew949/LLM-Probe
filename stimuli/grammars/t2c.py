"""
stimuli/grammars/t2c.py — Stimulus grammar for Thread T2c (two-dimensional semantics).

─── CONCEPT: Two-dimensional semantics (Chalmers) ────────────────────────────
Standard possible-worlds semantics assigns each expression one intension: a
function from possible worlds to extensions. Chalmers' two-dimensional framework
splits this into two distinct intensions:

Primary intension: how a term fixes reference given the actual world as the
  epistemic base. "Water" picks out the watery stuff in the actual world.
  In a Twin Earth world where XYZ is the watery stuff, "water" would pick
  out XYZ under the primary intension. Epistemic / descriptive dimension.

Secondary intension: what a term picks out across possible worlds once reference
  is fixed by the actual world. "Water" rigidly picks out H2O in all possible
  worlds (including worlds where H2O is rare). Metaphysical / rigid dimension.

The key dissociation: "Water is H2O" is secondarily necessary (true in all
  metaphysically possible worlds) but primarily contingent (in a world where
  watery stuff is XYZ, the primary intension of "water is H2O" would be false).
  This is Chalmers' central move against Kripke's two-dimensionalism.

─── CONCEPT: Three conditions ────────────────────────────────────────────────
primary_sensitive: sentences invoking primary intension — how reference is
  fixed. Context: epistemic possibility, "what stuff turns out to be".
  "Water is whatever the watery stuff turns out to be in this world."

secondary_necessary: sentences invoking secondary intension — rigid reference.
  Context: metaphysical necessity, "in all possible worlds".
  "Water is necessarily H2O in every possible world."

primary_secondary_dissociation: secondarily necessary but primarily contingent.
  The Chalmers signature case — a posteriori necessity.
  "If this substance had turned out not to be H2O, water would not be H2O."

─── Relationship to T2b ──────────────────────────────────────────────────────
T2b established hyperintensionality — distinctions finer than possible worlds.
T2c tests whether those distinctions are organized along the primary/secondary
dimension specifically. T2b is a prerequisite (V16).

─── MODEL REQUIREMENT ────────────────────────────────────────────────────────
Requires reliable knowledge of scientific identity claims (water = H2O,
heat = molecular motion) and modal reasoning. GPT-2 and Pythia fail this.
T2c runs only on Llama 3.2 3B in Phase 7, gated by T2b passing (V16).
"""

from __future__ import annotations

import random
from typing import Any


# ── Primary intension templates ───────────────────────────────────────────────

PRIMARY_TEMPLATES: list[tuple[str, str]] = [
    (
        "{term_a} is whatever the {description} turns out to be in this world.",
        "{term_b} is whatever the {description} turns out to be in this world.",
    ),
    (
        "When we say '{term_a}', we mean the {description} we encounter here.",
        "The term '{term_b}' refers to the {description} we actually find.",
    ),
    (
        "In a world where {description} is {alternate_substance}, '{term_a}' would pick out {alternate_substance}.",
        "If {description} turned out to be {alternate_substance}, '{term_b}' would refer to {alternate_substance}.",
    ),
]

# ── Secondary intension templates ─────────────────────────────────────────────

SECONDARY_TEMPLATES: list[tuple[str, str]] = [
    (
        "{term_a} is necessarily {scientific_identity} in every possible world.",
        "In all possible worlds where {term_a} exists, it is {scientific_identity}.",
    ),
    (
        "It is metaphysically necessary that {term_a} is {scientific_identity}.",
        "No possible world contains {term_a} that is not {scientific_identity}.",
    ),
    (
        "{term_a} could not have been anything other than {scientific_identity}.",
        "Even in counterfactual scenarios, {term_a} remains {scientific_identity}.",
    ),
]

# ── Primary-secondary dissociation templates ──────────────────────────────────

DISSOCIATION_TEMPLATES: list[tuple[str, str]] = [
    (
        "If {term_a} had turned out not to be {scientific_identity}, {term_a} would not be {scientific_identity}.",
        "Had the {description} in this world been {alternate_substance}, {term_a} would have been {alternate_substance}.",
    ),
    (
        "{term_a} is {scientific_identity} is a posteriori necessary — contingent on how things turned out.",
        "We could have discovered that {term_a} is {alternate_substance} if the {description} were different.",
    ),
    (
        "The necessity of '{term_a} is {scientific_identity}' depends on which world fixes reference.",
        "Primary intension of '{term_a} is {scientific_identity}' is false in worlds where {description} is {alternate_substance}.",
    ),
]

# ── Domain slots ──────────────────────────────────────────────────────────────

DOMAIN_SLOTS: list[dict[str, str]] = [
    {
        "term_a": "water",
        "term_b": "H2O",
        "description": "watery stuff",
        "scientific_identity": "H2O",
        "alternate_substance": "XYZ",
    },
    {
        "term_a": "heat",
        "term_b": "molecular motion",
        "description": "heat phenomenon",
        "scientific_identity": "mean molecular kinetic energy",
        "alternate_substance": "caloric fluid",
    },
    {
        "term_a": "gold",
        "term_b": "the element with atomic number 79",
        "description": "yellowish metal",
        "scientific_identity": "the element with atomic number 79",
        "alternate_substance": "fool's gold",
    },
    {
        "term_a": "Hesperus",
        "term_b": "Phosphorus",
        "description": "bright evening star",
        "scientific_identity": "Venus",
        "alternate_substance": "a different planet",
    },
]


def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n stimulus pairs for T2c — two-dimensional semantics conditions.

    Three conditions: primary_sensitive, secondary_necessary,
    primary_secondary_dissociation. Each pair contrasts two sentences invoking
    different intension dimensions from the same domain.

    Args:
        n: number of pairs to generate.
        seed: random seed for reproducibility.

    Returns:
        List of dicts with keys: sentence_a, sentence_b, label_a, label_b.
    """
    rng = random.Random(seed)

    condition_template_map: dict[str, list[tuple[str, str]]] = {
        "primary_sensitive": PRIMARY_TEMPLATES,
        "secondary_necessary": SECONDARY_TEMPLATES,
        "primary_secondary_dissociation": DISSOCIATION_TEMPLATES,
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
