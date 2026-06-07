"""
stimuli/grammars/t1b.py — Stimulus grammar for Thread T1b (mechanism-geometry test).

─── CONCEPT: what T1b tests (rewritten 2026-06-07) ───────────────────────────
Lewis's closest-world counterfactual and Pearl's do() coincide on the simple
recursive SCMs a small LM can handle (Briggs 2012; Halpern 2013) — so a
truth-condition probe cannot separate them. But that convergence is about
ANSWERS, not about the GEOMETRY of the computation. T1b therefore asks a
mechanism question: is the counterfactual representation organized like a causal
GRAPH (cluster by causal topology) or like a holistic SIMILARITY metric (cluster
by topic)? See docs/superpowers/specs/2026-06-07-t1b-mechanism-geometry-redesign.md.

─── CONCEPT: factorial domain x structure (the decorrelation) ────────────────
We cross 4 causal STRUCTURES x 5 DOMAINS. Crossing makes the two theoretical
matrices (M_graph = same structure, M_sim = same domain) decorrelated, so partial
RSA can attribute the model's geometry to one theory net of the other (V23).
Every sentence's label is "{structure}|{domain}".

  chain:     A -> B -> C   "If A had happened, C would have"      (mediated)
  fork:      A <- C -> B   "If A had happened, B would have"      (common cause)
  direct:    A -> B        "If A had happened, B would have"      (single edge)
  collider:  A -> C <- B   "If A had happened, C would have"      (effect of two)

─── CONCEPT: do/see behavioral gate ──────────────────────────────────────────
The V8 gate is interventionist competence: 'observed X' licenses back-inference
to X's cause; 'forced/made X' severs it. If the model cannot tell do from see, it
has no causal structure for the geometry test to read.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

# ── Factors ───────────────────────────────────────────────────────────────────

STRUCTURES: list[str] = ["chain", "fork", "direct", "collider"]

# Each domain supplies three content nouns A, B, C with matched register so the
# only thing that varies across structures within a domain is the causal frame.
DOMAINS: list[dict[str, str]] = [
    {"domain": "weather",    "a": "the storm",    "b": "the flooding",   "c": "the pressure drop"},
    {"domain": "medicine",   "a": "the infection","b": "the fever",      "c": "the exposure"},
    {"domain": "mechanical", "a": "the engine",   "b": "the vibration",  "c": "the fuel surge"},
    {"domain": "plant",      "a": "the drought",  "b": "the wilting",    "c": "the heat wave"},
    {"domain": "finance",    "a": "the layoffs",  "b": "the downturn",   "c": "the rate hike"},
]

# Structure templates. {a},{b},{c} are filled from the domain. Each ends mid-clause
# so the model must complete — and each is length-matched across structures.
STRUCTURE_TEMPLATES: dict[str, str] = {
    "chain":    "If {a} had occurred, then through {b}, {c} would have",
    "fork":     "If {a} had occurred, then alongside it, {b} would have",
    "direct":   "If {a} had occurred, then directly, {b} would have",
    "collider": "If {a} had occurred, then together with {b}, {c} would have",
}

# ── Direct-asymmetry probe sentences (for the L3 step) ────────────────────────
# For each domain, a cause-final and an effect-final one-clause sentence whose
# LAST token is the cause noun head / effect noun head respectively. The runner
# extracts the last-token activation and patches cause<->effect.
ASYM_TEMPLATES: dict[str, str] = {
    "cause_final":  "The effect that followed was driven entirely by {a}",
    "effect_final": "The cause that came first then produced {b}",
}

_MAX_PAIRS: int = 200  # 4 structures x 5 domains => 20 cells; 10 sentences/cell across pairs


# ── Behavioral items: do vs see ───────────────────────────────────────────────
_BEHAVIORAL_ITEMS: list[dict[str, Any]] = [
    {"question": "Maria saw that the sprinkler was on. Had it most likely rained earlier?",
     "choice_a": "yes, rain likely turned it on", "choice_b": "no, that cannot be inferred", "correct": "a"},
    {"question": "Maria turned the sprinkler on herself. Had it most likely rained earlier?",
     "choice_a": "yes, it likely rained", "choice_b": "no, that cannot be inferred", "correct": "b"},
    {"question": "We observed the alarm ringing. Was there most likely smoke?",
     "choice_a": "yes, smoke likely set it off", "choice_b": "no, nothing follows", "correct": "a"},
    {"question": "An engineer forced the alarm to ring for a test. Was there most likely smoke?",
     "choice_a": "yes, there was smoke", "choice_b": "no, that cannot be inferred", "correct": "b"},
    {"question": "The doctor saw the patient's fever was gone. Had the drug most likely worked?",
     "choice_a": "yes, the drug likely worked", "choice_b": "no, nothing follows", "correct": "a"},
    {"question": "A nurse made the thermometer read normal by cooling it. Had the fever most likely broken?",
     "choice_a": "yes, the fever broke", "choice_b": "no, that cannot be inferred", "correct": "b"},
    {"question": "We saw the barometer reading low. Was a storm most likely coming?",
     "choice_a": "yes, a storm was likely coming", "choice_b": "no, nothing follows", "correct": "a"},
    {"question": "A technician set the barometer to read low by hand. Was a storm most likely coming?",
     "choice_a": "yes, a storm was coming", "choice_b": "no, that cannot be inferred", "correct": "b"},
]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate up to n T1b stimulus pairs across the factorial domain x structure grid,
    plus the direct-asymmetry probe sentences.

    Sentences are paired so neither factor dominates the pair channel: half the
    pairs hold domain fixed and vary structure; half hold structure fixed and vary
    domain. Each sentence carries label "{structure}|{domain}".

    Args:
        n:    number of pairs to return (<= _MAX_PAIRS for the core grid; asym
              probe pairs are appended and not counted against n).
        seed: instance RNG seed (no global state).

    Returns:
        List of StimulusPair dicts (schema-conforming) with factorial labels.

    Raises:
        ValueError: if n > _MAX_PAIRS.
    """
    if n > _MAX_PAIRS:
        raise ValueError(f"n={n} exceeds maximum of {_MAX_PAIRS} core pairs.")

    rng = random.Random(seed)

    def render(structure: str, domain_row: dict[str, str]) -> str:
        return STRUCTURE_TEMPLATES[structure].format(
            a=domain_row["a"], b=domain_row["b"], c=domain_row["c"])

    # Build the full cell list: one sentence per (structure, domain).
    cells: list[tuple[str, str, str]] = []  # (sentence, structure, domain)
    for structure, domain_row in itertools.product(STRUCTURES, DOMAINS):
        cells.append((render(structure, domain_row), structure, domain_row["domain"]))

    def make_pair(sa, la, sb, lb, notes="") -> dict[str, Any]:
        pair = {
            "thread_id": "t1b",
            "sentence_a": sa, "sentence_b": sb,
            "label_a": la, "label_b": lb,
            "theoretical_distinction": "graph-structured (Pearl) vs similarity-structured (Lewis)",
            "frequency_matched": False,
            "generation_grammar": "stimuli/grammars/t1b.py",
        }
        if notes:
            pair["notes"] = notes
        return pair

    pairs: list[dict[str, Any]] = []
    # Within-domain across-structure pairs
    for domain_row in DOMAINS:
        domain_cells = [c for c in cells if c[2] == domain_row["domain"]]
        for (sa, structure_a, dom), (sb, structure_b, _) in itertools.combinations(domain_cells, 2):
            pairs.append(make_pair(sa, f"{structure_a}|{dom}", sb, f"{structure_b}|{dom}"))
    # Within-structure across-domain pairs
    for structure in STRUCTURES:
        structure_cells = [c for c in cells if c[1] == structure]
        for (sa, st, dom_a), (sb, _, dom_b) in itertools.combinations(structure_cells, 2):
            pairs.append(make_pair(sa, f"{st}|{dom_a}", sb, f"{st}|{dom_b}"))

    rng.shuffle(pairs)
    selected = pairs[:n]

    # Append direct-asymmetry probe pairs (cause_final vs effect_final per domain).
    for domain_row in DOMAINS:
        cause_sentence = ASYM_TEMPLATES["cause_final"].format(a=domain_row["a"])
        effect_sentence = ASYM_TEMPLATES["effect_final"].format(b=domain_row["b"])
        selected.append(make_pair(
            cause_sentence, f"direct_asym|{domain_row['domain']}",
            effect_sentence, f"direct_asym|{domain_row['domain']}",
            notes=f"asym_probe cause_token={domain_row['a']} effect_token={domain_row['b']}",
        ))

    for index, pair in enumerate(selected):
        pair["pair_id"] = f"t1b_{index + 1:04d}"
    return selected


def generate_behavioral_items() -> list[dict[str, Any]]:
    """Return do/see forced-choice items for the V8 gate (interventionist competence)."""
    return list(_BEHAVIORAL_ITEMS)
