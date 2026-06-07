"""
stimuli/grammars/t1d.py — Stimulus grammar for Thread T1d (causal identification).

─── CONCEPT: What T1d tests ──────────────────────────────────────────────────
T1b established which counterfactual mechanism GPT-2 uses. T1d tests a deeper
question: does the representation respect Pearl's do-calculus *identification*
conditions — when a causal effect can be recovered from observational data?

─── CONCEPT: The discriminator is an IMPLICIT structural property ────────────
Identifiability of P(Y | do(X)) in a confounded graph X←Z→Y, X→Y turns on ONE
thing: whether the confounder Z is **observed**. If Z is measured, the back-door
adjustment identifies the effect; if Z is hidden, no adjustment set exists and
the effect is not identified. Nothing else about the graph changes.

The earlier version of this grammar STATED the verdict in the stimulus ("the
adjusted estimate is valid", "the effect cannot be identified", "P(Y|X) equals
P(Y|do(X))"). A probe then only had to detect those phrases — a keyword
classifier, not a representation of identifiability. This version states NO
verdict. The back_door and confounded conditions are **minimal pairs** that
differ in exactly one word: whether the confounder is described as observed or
hidden. The probe must encode that structural property — which *is* the
identifiability-relevant feature — with no lexical tell of the answer.

─── CONCEPT: Lexically varied observed/hidden cue ────────────────────────────
If every identified case said "measured" and every non-identified case said
"hidden", the probe could read those two words. Each draws from a matched list of
synonyms — {recorded, measured, tracked, logged, observed} vs {unrecorded,
unmeasured, untracked, unlogged, hidden} — so the encoded distinction is the
*category* (observed vs unobserved), not a single lexical item.

─── CONCEPT: Four conditions ─────────────────────────────────────────────────
back_door_adjustable      X→Y, Z→{X,Y}, Z OBSERVED. Identified by adjustment.
confounded_not_adjustable X→Y, Z→{X,Y}, Z HIDDEN.   Not identified.
  ↑ these two are the primary minimal-pair contrast (the only difference is the
    observed/hidden cue, which is exactly what flips identifiability).
front_door_adjustable     X→M→Y, hidden U→{X,Y}, mediator M OBSERVED. Identified
                          via the front door. A second, distinct identification route.
unconfounded_control      X→Y, no common cause. Trivially identified. Positive control.

─── CONCEPT: Primary criterion ───────────────────────────────────────────────
The headline test is the binary back_door_adjustable vs confounded_not_adjustable
contrast (graph-isomorphic, observed vs hidden confounder) — this isolates
adjustability. front_door and unconfounded are additional identified conditions,
reported separately; lumping all three identified types together inflates a
binary probe via the easy unconfounded split rather than testing adjustability.
"""

from __future__ import annotations

import itertools
import random
from typing import Any


# ── Observed / hidden cue (the identifiability flip), lexically varied ─────────
_OBSERVED_CUES: list[str] = ["recorded", "measured", "tracked", "logged", "observed"]
_HIDDEN_CUES:   list[str] = ["unrecorded", "unmeasured", "untracked", "unlogged", "hidden"]


# ── Domain slots — relations only, no verdict vocabulary ──────────────────────
DOMAIN_SLOTS: list[dict[str, str]] = [
    {"treatment": "smoking",          "outcome": "lung cancer risk",        "confounder": "genetic predisposition", "mediator": "tar buildup"},
    {"treatment": "regular exercise", "outcome": "cardiovascular health",   "confounder": "socioeconomic status",   "mediator": "lower inflammation"},
    {"treatment": "factory emissions","outcome": "respiratory disease",     "confounder": "urban density",          "mediator": "airborne particulates"},
    {"treatment": "tutoring",         "outcome": "exam scores",             "confounder": "family income",          "mediator": "study habits"},
]


# ── Templates ─────────────────────────────────────────────────────────────────
# back_door and confounded share these frames verbatim — the cue ({cue}) is the
# only slot that flips the condition, so the two conditions are minimal pairs.
_CONFOUNDED_TEMPLATES: list[str] = [
    "{treatment} affects {outcome}, and {confounder}, which is {cue}, affects both {treatment} and {outcome}.",
    "{confounder}, which is {cue}, affects both {treatment} and {outcome}, and {treatment} affects {outcome}.",
    "Here {treatment} affects {outcome} while {confounder}, which is {cue}, affects both of them.",
]

# front_door: X→M→Y with a hidden common cause; the mediator is observed.
_FRONT_DOOR_TEMPLATES: list[str] = [
    "{treatment} affects {outcome} only through {mediator}, which is {observed_cue}, while a {hidden_cue} factor affects both {treatment} and {outcome}.",
    "{treatment} influences {mediator}, which is {observed_cue}, and {mediator} affects {outcome}, while a {hidden_cue} factor affects both ends.",
    "The path from {treatment} to {outcome} runs through {mediator}, which is {observed_cue}, and a {hidden_cue} factor affects both {treatment} and {outcome}.",
]

# unconfounded: X→Y with no common cause.
_UNCONFOUNDED_TEMPLATES: list[str] = [
    "{treatment} affects {outcome}, and nothing else affects both {treatment} and {outcome}.",
    "{treatment} affects {outcome}, with no other factor affecting both {treatment} and {outcome}.",
    "Here {treatment} affects {outcome} and no common cause affects both of them.",
]


# back_door vs confounded: 4 domains × 3 templates × 5 cue-pairs = 60 minimal pairs
# front_door vs unconfounded: 4 domains × 3 templates × 5 cue-pairs = 60 pairs
_MAX_PAIRS: int = 120


def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n T1d stimulus pairs across four identification conditions.

    Primary minimal-pair contrast (sentence_a vs sentence_b share a frame, differ
    only in the observed/hidden cue, which is exactly what flips identifiability):
      ("back_door_adjustable", "confounded_not_adjustable")

    Secondary contrast (additional identified route vs positive control):
      ("front_door_adjustable", "unconfounded_control")

    back_door and confounded are produced in equal numbers (60 each) so the binary
    identifiability probe is over matched-size, matched-frame sets.

    Args:
        n:    Number of pairs to return. Must be <= 120.
        seed: Instance-level RNG seed.

    Returns:
        List of n StimulusPair dicts conforming to stimulus.schema.json.

    Raises:
        ValueError: if n > 120.
    """
    if n > _MAX_PAIRS:
        raise ValueError(f"n={n} exceeds maximum of {_MAX_PAIRS} pairs.")

    cue_pairs = list(zip(_OBSERVED_CUES, _HIDDEN_CUES))
    all_pairs: list[dict[str, Any]] = []

    def make_pair(sentence_a: str, sentence_b: str, label_a: str, label_b: str, notes: str) -> dict[str, Any]:
        return {
            "thread_id": "t1d",
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
            "label_a": label_a,
            "label_b": label_b,
            "theoretical_distinction": "Pearl do-calculus identifiability (observed vs hidden confounder)",
            "frequency_matched": False,
            "generation_grammar": "stimuli/grammars/t1d.py",
            "notes": notes,
        }

    # Primary: back_door (observed confounder) vs confounded (hidden confounder) —
    # minimal pairs, cue is the only difference.
    for domain, template, (observed_cue, hidden_cue) in itertools.product(
        DOMAIN_SLOTS, _CONFOUNDED_TEMPLATES, cue_pairs
    ):
        back_door_sentence = template.format(cue=observed_cue, **domain)
        confounded_sentence = template.format(cue=hidden_cue, **domain)
        all_pairs.append(make_pair(
            back_door_sentence, confounded_sentence,
            "back_door_adjustable", "confounded_not_adjustable",
            notes=f"minimal pair: confounder '{observed_cue}' (identified) vs '{hidden_cue}' (not identified)",
        ))

    # Secondary: front_door (identified via observed mediator) vs unconfounded
    # control. Each front-door frame is paired with one unconfounded frame chosen
    # by index, so this yields 4 × 3 × 5 = 60 pairs (matched to the primary count).
    for template_index, (domain, front_template, (observed_cue, hidden_cue)) in enumerate(
        itertools.product(DOMAIN_SLOTS, _FRONT_DOOR_TEMPLATES, cue_pairs)
    ):
        unconf_template = _UNCONFOUNDED_TEMPLATES[template_index % len(_UNCONFOUNDED_TEMPLATES)]
        front_sentence = front_template.format(observed_cue=observed_cue, hidden_cue=hidden_cue, **domain)
        unconf_sentence = unconf_template.format(**domain)
        all_pairs.append(make_pair(
            front_sentence, unconf_sentence,
            "front_door_adjustable", "unconfounded_control",
            notes="front-door identified (observed mediator) vs unconfounded positive control",
        ))

    random.Random(seed).shuffle(all_pairs)
    selected_pairs = all_pairs[:n]

    for index, pair in enumerate(selected_pairs):
        pair["pair_id"] = f"t1d_{index + 1:04d}"

    return selected_pairs


def generate_behavioral_items() -> list[dict[str, Any]]:
    """
    Return behavioral gate items for T1d.

    Competence check: the model should recognize that an association driven by a
    common cause is not evidence of a direct causal effect, and that adjusting for
    an observed common cause is what licenses a causal read. Forced-choice; the
    structural cue (observed vs hidden) is named so the gate tests reasoning, not
    the minimal-pair inference the stimuli isolate.

    Returns:
        List of 4 forced-choice dicts: question, choice_a, choice_b, correct.
    """
    return [
        {
            "question": "Smoking and lung cancer are both driven by a hidden common cause. From this data alone, the effect of smoking on cancer is",
            "choice_a": "not identifiable",
            "choice_b": "identifiable",
            "correct": "a",
        },
        {
            "question": "A confounder of treatment and outcome is fully measured. Adjusting for it, the causal effect of treatment is",
            "choice_a": "identifiable",
            "choice_b": "not identifiable",
            "correct": "a",
        },
        {
            "question": "Two variables are correlated only because an unmeasured factor drives both. Their correlation is",
            "choice_a": "not causal",
            "choice_b": "causal",
            "correct": "a",
        },
        {
            "question": "Treatment affects outcome only through an observed mediator, with a hidden common cause. Via the mediator, the effect is",
            "choice_a": "identifiable",
            "choice_b": "not identifiable",
            "correct": "a",
        },
    ]
