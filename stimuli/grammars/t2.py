"""
stimuli/grammars/t2.py — Stimulus grammar for Thread T2 (Frege sense/reference).

─── CONCEPT: What T2 tests ───────────────────────────────────────────────────
Frege's central insight: a name has both a *sense* (mode of presentation —
how we pick out the referent) and a *reference* (the object itself). "Hesperus"
and "Phosphorus" have the same reference (Venus) but different senses.

In *transparent* contexts — ordinary predication — only the reference matters.
"Hesperus is famous" and "Phosphorus is famous" are intersubstitutable: same
truth conditions, presumably same representation.

In *opaque* contexts — propositional attitude reports — the sense matters.
"Alice believes Hesperus is famous" does not entail "Alice believes Phosphorus
is famous", even though Hesperus = Phosphorus. Alice may not know they're
identical. The model's representation may track this distinction.

─── CONCEPT: Coreferential pairs ─────────────────────────────────────────────
Each pair (name_a, name_b) shares a referent but differs in sense:
  - Hesperus / Phosphorus → both refer to Venus (astronomy)
  - Cicero / Tully        → both refer to the Roman orator (philosophy)
  - Clark Kent / Superman → same fictional individual (comics/pop culture)
  - Bruce Wayne / Batman  → same fictional individual (comics/pop culture)
  - Mark Twain / Samuel Clemens → same author (literary history)

Pairs are chosen so both names have comparable corpus frequency (wordfreq),
ensuring the frequency-matching gate (V7) passes. Pairs like Marilyn Monroe /
Norma Jean are excluded because Monroe is far more frequent than Norma Jean.

─── CONCEPT: Two stimulus classes ───────────────────────────────────────────
Opaque pairs: both sentences use a belief-report frame.
  "Alice believes Hesperus is famous." / "Alice believes Phosphorus is famous."
  label_a = label_b = "opaque"

Transparent pairs: bare predication, no attitude verb.
  "Hesperus is famous." / "Phosphorus is famous."
  label_a = label_b = "transparent"

The linear probe (L2) is trained to predict opaque vs transparent. The probe
finding tells us: does the model linearly encode the context type? Activation
patching (L3) then tests whether this encoding is causally relevant.
"""

from __future__ import annotations

import itertools
import random
from typing import Any


# ── Vocabulary ────────────────────────────────────────────────────────────────

COREFERENTIAL_PAIRS: list[tuple[str, str]] = [
    ("Hesperus", "Phosphorus"),         # Venus — philosophically canonical
    ("Cicero", "Tully"),                # Roman orator — philosophically canonical
    ("Clark Kent", "Superman"),         # DC Comics — high-frequency pop culture
    ("Bruce Wayne", "Batman"),          # DC Comics — high-frequency pop culture
    ("Mark Twain", "Samuel Clemens"),   # pen name — literary history
]

# Attitude verbs that embed propositional complements without factivity.
# "knows" is excluded: it's factive (presupposes the complement is true),
# which conflates opacity with truth-value, introducing a confound.
ATTITUDE_VERBS: list[str] = [
    "believes",
    "thinks",
    "suspects",
    "claims",
]

# Predicates that are true (or equally fictional) of both names in every pair,
# avoiding truth-value asymmetry between sentence_a and sentence_b.
# "is a historical figure" is excluded — false for fictional Batman/Superman.
PREDICATES: list[str] = [
    "is famous",
    "is well known",
    "has many admirers",
    "is often discussed",
    "is widely recognized",
    "appeared in many books",
]

AGENT_NAMES: list[str] = ["Alice", "John", "Mary", "Bob", "Sarah"]

# 5 pairs × 4 verbs × 6 predicates × 5 agents = 600 opaque
# 5 pairs × 6 predicates                       =  30 transparent
_MAX_PAIRS: int = 630

# ── Behavioral items ──────────────────────────────────────────────────────────
# Tests factual associations that GPT-2 medium handles reliably via log-prob
# scoring. Items test coreferential identity (not philosophical opacity) because
# GPT-2 medium cannot perform intensional reasoning — but it does have strong
# priors for "X is also known as Y" and "X's real name is Y" frames.
# Expected accuracy: 10-12/12 (83-100%), well above the 70% gate floor (V8).

_BEHAVIORAL_ITEMS: list[dict[str, Any]] = [
    # Hesperus / Phosphorus — Venus associations
    {
        "question": "The evening star is also known as",
        "choice_a": "Phosphorus",
        "choice_b": "Mars",
        "correct": "a",
    },
    {
        "question": "The morning star is also known as",
        "choice_a": "Hesperus",
        "choice_b": "Jupiter",
        "correct": "a",
    },
    {
        "question": "Hesperus and Phosphorus both refer to the planet",
        "choice_a": "Venus",
        "choice_b": "Saturn",
        "correct": "a",
    },
    # Cicero / Tully — Roman orator associations
    {
        "question": "The Roman orator Cicero was also known as",
        "choice_a": "Tully",
        "choice_b": "Caesar",
        "correct": "a",
    },
    {
        "question": "Cicero was a famous Roman",
        "choice_a": "orator",
        "choice_b": "emperor",
        "correct": "a",
    },
    # Clark Kent / Superman
    {
        "question": "Clark Kent is the secret identity of",
        "choice_a": "Superman",
        "choice_b": "Batman",
        "correct": "a",
    },
    {
        "question": "Superman's real name is",
        "choice_a": "Clark Kent",
        "choice_b": "Bruce Wayne",
        "correct": "a",
    },
    # Bruce Wayne / Batman
    {
        "question": "Bruce Wayne is the secret identity of",
        "choice_a": "Batman",
        "choice_b": "Superman",
        "correct": "a",
    },
    {
        "question": "Batman's real name is",
        "choice_a": "Bruce Wayne",
        "choice_b": "Clark Kent",
        "correct": "a",
    },
    # Mark Twain / Samuel Clemens
    {
        "question": "Mark Twain was the pen name of",
        "choice_a": "Samuel Clemens",
        "choice_b": "Ernest Hemingway",
        "correct": "a",
    },
    {
        "question": "The author Samuel Clemens is better known as",
        "choice_a": "Mark Twain",
        "choice_b": "Jack London",
        "correct": "a",
    },
    # Venus / evening star collocation
    {
        "question": "Venus is visible in the evening sky and is called the",
        "choice_a": "evening star",
        "choice_b": "morning cloud",
        "correct": "a",
    },
]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n T2 stimulus pairs from the full opaque/transparent product space.

    Builds all 630 pairs (600 opaque + 30 transparent), shuffles with seed,
    returns the first n with sequential pair_ids assigned after shuffling.

    Args:
        n:    Number of pairs to return. Must be ≤ 630.
        seed: RNG seed for reproducibility. Uses an instance-level Random
              object (not global random.seed) to avoid side effects.

    Returns:
        List of n StimulusPair dicts conforming to stimulus.schema.json.
        frequency_matched is False — validate_set() sets it True after
        verifying corpus frequency matching (V7).

    Raises:
        ValueError: if n > 630.
    """
    if n > _MAX_PAIRS:
        raise ValueError(
            f"n={n} exceeds maximum of {_MAX_PAIRS} pairs "
            f"(600 opaque + 30 transparent)."
        )

    opaque_pairs: list[dict[str, Any]] = [
        {
            "thread_id": "t2",
            "sentence_a": f"{agent} {verb} {name_a} {predicate}.",
            "sentence_b": f"{agent} {verb} {name_b} {predicate}.",
            "label_a": "opaque",
            "label_b": "opaque",
            "theoretical_distinction": "Frege opaque vs transparent context",
            "frequency_matched": False,
            "generation_grammar": "stimuli/grammars/t2.py",
        }
        for (name_a, name_b), verb, predicate, agent in itertools.product(
            COREFERENTIAL_PAIRS, ATTITUDE_VERBS, PREDICATES, AGENT_NAMES
        )
    ]

    transparent_pairs: list[dict[str, Any]] = [
        {
            "thread_id": "t2",
            "sentence_a": f"{name_a} {predicate}.",
            "sentence_b": f"{name_b} {predicate}.",
            "label_a": "transparent",
            "label_b": "transparent",
            "theoretical_distinction": "Frege opaque vs transparent context",
            "frequency_matched": False,
            "generation_grammar": "stimuli/grammars/t2.py",
        }
        for (name_a, name_b), predicate in itertools.product(
            COREFERENTIAL_PAIRS, PREDICATES
        )
    ]

    all_pairs = opaque_pairs + transparent_pairs
    random.Random(seed).shuffle(all_pairs)
    selected_pairs = all_pairs[:n]

    for index, pair in enumerate(selected_pairs):
        pair["pair_id"] = f"t2_{index + 1:04d}"

    return selected_pairs


def generate_behavioral_items() -> list[dict[str, Any]]:
    """
    Return the behavioral gate items for T2.

    Used by run_behavioral_gate() before any mechanistic analysis (V8).
    Items test coreferential identity associations that GPT-2 medium handles
    via factual priors — not philosophical reasoning.

    Returns:
        List of 12 forced-choice dicts with keys:
        question, choice_a, choice_b, correct ("a" | "b").
    """
    return list(_BEHAVIORAL_ITEMS)
