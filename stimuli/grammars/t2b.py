"""
stimuli/grammars/t2b.py — Stimulus grammar for Thread T2b (Hyperintensionality).

─── CONCEPT: What T2b tests ──────────────────────────────────────────────────
Hyperintensionality is the property of making distinctions *finer* than
possible worlds. Lewis/Stalnaker semantics says: if two sentences have the
same truth conditions across all possible worlds, they have the same meaning.
Hyperintensional semantics denies this — "7 is prime" and "7 is not composite"
are necessarily co-extensive (same truth conditions in every world) but may
differ in cognitive content.

If GPT-2's representations track this difference, the model is
hyperintensional — it encodes something beyond what possible-worlds semantics
can explain.

─── CONCEPT: Three required stimulus classes ─────────────────────────────────
Class 1 — logically_equivalent
  Syntactically distinct but derivable from each other by pure logic:
  conjunction commutativity, double negation elimination, definitional
  identity. Representations should CONVERGE. This class establishes the
  baseline: the probe can detect convergence for genuinely equivalent pairs.

Class 2 — intensionally_equivalent  ← the critical hyperintensionality test
  Same truth conditions in every possible world, but different cognitive
  content. Lewis/Stalnaker predict convergence. Hyperintensional semantics
  predicts divergence. Finding: if class 2 cosine distance exceeds the 95th
  percentile of the permutation null → model is hyperintensional.
  Canonical examples: "7 is prime" / "7 is not composite".

Class 3 — intensionally_distinct     ← positive control
  Different truth conditions — sentence_a and sentence_b disagree about facts.
  Representations must DIVERGE. If class 3 pairs do not diverge, the probe
  is not working and results are uninterpretable.

─── CONCEPT: Why mathematical domain ────────────────────────────────────────
Mathematical truths are necessary — they hold in all possible worlds without
exception. This means worlds-based semantics has no discriminating power here:
all true mathematical sentences are true in all worlds and are therefore
trivially co-extensional under Lewis/Stalnaker. Any representational
divergence within mathematical truths is evidence for hyperintensionality
specifically, not just intensionality.

─── CONCEPT: Hand-authored vs templated ─────────────────────────────────────
Pairs are hand-authored, not templated. Mathematical equivalences require
specific logical relationships that cannot be verified programmatically
without semantic checking. There are ~10 well-motivated content domains;
hand-authoring 30 pairs produces higher-quality stimuli than template
instantiation would.
"""

from __future__ import annotations

import random
from typing import Any


# ── Vocabulary ────────────────────────────────────────────────────────────────

# Class 1: logically equivalent pairs.
# Derivable from each other by conjunction commutativity, double negation
# elimination, or definitional identity of predicates. Representations
# should converge under any reasonable semantics.
_CLASS1_PAIRS: list[tuple[str, str]] = [
    ("Seven is prime and odd.", "Seven is odd and prime."),
    ("Five is prime and odd.", "Five is odd and prime."),
    ("Three is odd and prime.", "Three is prime and odd."),
    ("Two is even and prime.", "Two is prime and even."),
    ("Four is even and positive.", "Four is positive and even."),
    ("Six is even and composite.", "Six is composite and even."),
    ("Nine is odd and composite.", "Nine is composite and odd."),
    ("It is not the case that seven is not prime.", "Seven is prime."),
    ("Nine is not even.", "Nine is odd."),                    # diff=0.78 — passes V7
    ("Six is not prime.", "Six is composite."),               # diff=0.44 — passes V7
]

# Class 2: intensionally equivalent pairs. ← THE CRITICAL TEST
# Same truth conditions in all possible worlds, different cognitive content.
# Lewis/Stalnaker predict convergence. Hyperintensional semantics predicts
# divergence. Pairs 1-3 (prime / not composite) are the canonical test cases.
# Note: pair 7 ("even" / "divisible by two") has frequency diff=0.95 —
# borderline but passes V7 gate.
_CLASS2_PAIRS: list[tuple[str, str]] = [
    ("Seven is prime.", "Seven is not composite."),           # canonical — prime = not composite
    ("Five is prime.", "Five is not composite."),
    ("Two is prime.", "Two is not composite."),
    ("The square root of four is two.", "Two squared is four."),  # inverse operations
    ("Bachelors are unmarried.", "Unmarried men are bachelors."),  # definitional biconditional
    ("Nine is a perfect square.", "Three times three equals nine."),
    ("Six is even.", "Six is divisible by two."),             # diff=0.95, borderline
    ("Eight is a cube.", "Eight equals two cubed."),
    ("A bachelor is an unmarried man.", "An unmarried man is a bachelor."),
    ("Three is odd.", "Three is not even."),
]

# Class 3: intensionally distinct pairs. ← POSITIVE CONTROL
# Different truth conditions — sentence_a and sentence_b disagree about facts.
# Representations must diverge. If they do not, the probe is not working.
_CLASS3_PAIRS: list[tuple[str, str]] = [
    ("Seven is prime.", "Seven is even."),
    ("Five is prime.", "Five is even."),
    ("Three is odd.", "Three is even."),
    ("Nine is a perfect square.", "Nine is a prime number."),
    ("Two is prime.", "Two is odd."),
    ("Six is composite.", "Six is prime."),
    ("All bachelors are unmarried.", "Some bachelors are married."),
    ("The square root of four is two.", "The square root of four is three."),
    ("Four is even.", "Four is odd."),
    ("Eight is composite.", "Eight is prime."),
]

_MAX_PAIRS: int = 30  # 10 per class


# ── Behavioral items ──────────────────────────────────────────────────────────
# Tests whether GPT-2 medium knows which pairs are logically equivalent (class 1)
# and which are intensionally distinct (class 3). Uses mathematical and logical
# knowledge GPT-2 reliably has from training — not philosophical reasoning.
# Expected accuracy: ≥10/12, well above the 0.70 gate floor (V8).

_BEHAVIORAL_ITEMS: list[dict[str, Any]] = [
    # Class 1 knowledge: conjunction commutativity
    {
        "question": "Seven is prime and odd. This is the same as saying:",
        "choice_a": "Seven is odd and prime",
        "choice_b": "Seven is odd and even",
        "correct": "a",
    },
    # Class 1 knowledge: double negation elimination
    {
        "question": "It is not the case that seven is not prime. Therefore seven is:",
        "choice_a": "prime",
        "choice_b": "even",
        "correct": "a",
    },
    # Class 1 knowledge: not even = odd
    {
        "question": "True or false: nine is not even.",
        "choice_a": "True",
        "choice_b": "False",
        "correct": "a",
    },
    # Class 2 knowledge: prime = not composite (canonical hyperintensionality pair)
    {
        "question": "Seven is prime. Is seven composite?",
        "choice_a": "No, seven is not composite",
        "choice_b": "Yes, seven is composite",
        "correct": "a",
    },
    # Class 2 knowledge: not composite = prime
    {
        "question": "Seven is not composite. Therefore seven is:",
        "choice_a": "prime",
        "choice_b": "even",
        "correct": "a",
    },
    # Class 3 knowledge: prime ≠ even
    {
        "question": "Seven is prime. Is seven also even?",
        "choice_a": "No, seven is odd",
        "choice_b": "Yes, seven is even",
        "correct": "a",
    },
    # Class 3 knowledge: all unmarried ≠ some married
    {
        "question": "All bachelors are unmarried. Does this mean some bachelors are married?",
        "choice_a": "No",
        "choice_b": "Yes",
        "correct": "a",
    },
    # Class 3 knowledge: sqrt(4) = 2, not 3
    {
        "question": "The square root of four is:",
        "choice_a": "two",
        "choice_b": "three",
        "correct": "a",
    },
    # Class 2 knowledge: inverse operations
    {
        "question": "Two squared is four. The square root of four is:",
        "choice_a": "two",
        "choice_b": "eight",
        "correct": "a",
    },
    # Class 1 knowledge: commutativity
    {
        "question": "Three is odd and prime. Three is prime and:",
        "choice_a": "odd",
        "choice_b": "even",
        "correct": "a",
    },
    # Class 2 knowledge: even = divisible by 2
    {
        "question": "Six is even. Is six divisible by two?",
        "choice_a": "Yes",
        "choice_b": "No",
        "correct": "a",
    },
    # Class 2 knowledge: prime combined with commutativity
    {
        "question": "Five is prime and odd. Five is also:",
        "choice_a": "not composite",
        "choice_b": "even",
        "correct": "a",
    },
]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n T2b stimulus pairs across all three hyperintensionality classes.

    All 30 pairs are hand-authored (10 per class) — see module docstring for
    why templates are inappropriate for mathematical equivalence pairs.
    Shuffles with seed, returns first n, assigns sequential pair_ids after
    shuffling.

    Args:
        n:    Number of pairs to return. Must be ≤ 30.
        seed: RNG seed. Uses instance-level Random to avoid global side effects.

    Returns:
        List of n StimulusPair dicts conforming to stimulus.schema.json.
        frequency_matched is False — validate_set() sets it True after V7 check.

    Raises:
        ValueError: if n > 30.
    """
    if n > _MAX_PAIRS:
        raise ValueError(
            f"n={n} exceeds maximum of {_MAX_PAIRS} pairs "
            f"(10 logically_equivalent + 10 intensionally_equivalent + 10 intensionally_distinct)."
        )

    def build_pairs(raw_pairs: list[tuple[str, str]], label: str) -> list[dict[str, Any]]:
        return [
            {
                "thread_id": "t2b",
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "label_a": label,
                "label_b": label,
                "theoretical_distinction": "Hyperintensionality — distinctions finer than possible worlds",
                "frequency_matched": False,
                "generation_grammar": "stimuli/grammars/t2b.py",
            }
            for sentence_a, sentence_b in raw_pairs
        ]

    all_pairs = (
        build_pairs(_CLASS1_PAIRS, "logically_equivalent")
        + build_pairs(_CLASS2_PAIRS, "intensionally_equivalent")
        + build_pairs(_CLASS3_PAIRS, "intensionally_distinct")
    )

    random.Random(seed).shuffle(all_pairs)
    selected_pairs = all_pairs[:n]

    for index, pair in enumerate(selected_pairs):
        pair["pair_id"] = f"t2b_{index + 1:04d}"

    return selected_pairs


def generate_behavioral_items() -> list[dict[str, Any]]:
    """
    Return behavioral gate items for T2b.

    Tests GPT-2 medium's mathematical and logical knowledge — the prerequisite
    for hyperintensionality probing. Items cover logical equivalence (class 1),
    prime/not-composite identity (class 2), and intensional distinctness (class 3).

    Returns:
        List of 12 forced-choice dicts: question, choice_a, choice_b, correct.
    """
    return list(_BEHAVIORAL_ITEMS)
