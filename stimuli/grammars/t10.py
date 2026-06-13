"""
stimuli/grammars/t10.py — Stimulus grammar for T10 (compositionality / systematicity).

─── CONCEPT: attribute binding under controlled combination frequency ────────
T10 tests whether the model binds an attribute to an object via a structure-
sensitive mechanism (Fodor and Pylyshyn, Hypothesis A) or by interpolation over
seen combinations (Hypothesis B). The discriminating variable is combination
frequency: a structured mechanism binds equally well regardless of how often the
color-object pair occurred; an interpolator degrades on rare pairs.

The fatal confound is conflating combination frequency with word frequency. We
avoid it by holding the word pool fixed (common colors, common objects) and
varying only whether the pairing is TYPICAL (high combination frequency, "red
apple") or ATYPICAL (low, "blue apple"). Same words in both bins; only the
pairing's frequency differs.

Atypical bindings are also a behavioral binding test: a model answering from
priors says the typical color; a model that actually bound says the bound color.

Each scene has two objects so binding is non-trivial (the model must keep them
apart). The retrieval question probes one object's color.
"""

from __future__ import annotations

import random
from typing import Any

from wordfreq import word_frequency

# Common color words and the objects with their canonical (typical) colors.
COLORS: list[str] = [
    "red", "green", "blue", "yellow", "purple", "orange", "brown", "white", "black", "pink",
]

OBJECT_TYPICAL_COLORS: dict[str, list[str]] = {
    "apple": ["red", "green"],
    "banana": ["yellow"],
    "grass": ["green"],
    "sky": ["blue"],
    "snow": ["white"],
    "carrot": ["orange"],
    "cherry": ["red"],
    "lemon": ["yellow"],
    "leaf": ["green"],
    "coal": ["black"],
    "ocean": ["blue"],
    "flamingo": ["pink"],
}


def _freq_log10(word: str) -> float:
    import math
    f = word_frequency(word, "en")
    return math.log10(f) if f > 0 else -8.0


def generate(seed: int = 42) -> list[dict[str, Any]]:
    """
    One typical and one atypical scene per object. Deterministic.

    Each record is a two-object scene plus a retrieval question about the target
    object's color. combo_bin is 'typical' or 'atypical'. Word pool is identical
    across bins, so only combination frequency varies.
    """
    rng = random.Random(seed)
    objects = list(OBJECT_TYPICAL_COLORS)
    records: list[dict[str, Any]] = []

    for target in objects:
        typ = OBJECT_TYPICAL_COLORS[target]
        atyp = [c for c in COLORS if c not in typ]

        for combo_bin, pool in (("typical", typ), ("atypical", atyp)):
            target_color = rng.choice(pool)

            distractor = rng.choice([o for o in objects if o != target])
            d_typ = OBJECT_TYPICAL_COLORS[distractor]
            distractor_color = rng.choice(
                [c for c in d_typ if c != target_color]
                or [c for c in COLORS if c != target_color]
            )

            target_np = (target_color, target)
            distractor_np = (distractor_color, distractor)
            first, second = target_np, distractor_np
            if rng.random() < 0.5:
                first, second = second, first

            scene = f"the {first[0]} {first[1]} and the {second[0]} {second[1]}"
            records.append({
                "thread_id": "t10",
                "scene": scene,
                "sentence": f"There is {scene}.",
                "question": f"What color is the {target}?",
                "answer": target_color,
                "probed_object": target,
                "distractor_object": distractor,
                "combo_bin": combo_bin,
                "typical_color": typ[0],          # the prior-based (often wrong) answer
                "target_word_freq_log10": _freq_log10(target),
                "color_word_freq_log10": _freq_log10(target_color),
            })

    for i, r in enumerate(records):
        r["item_id"] = f"t10_{i + 1:04d}"
    return records


def generate_behavioral_items() -> list[dict[str, Any]]:
    """
    Forced-choice gate items from the ATYPICAL scenes, where binding and prior
    disagree. choice_a is the correctly bound color; choice_b is the object's
    typical color. A model that binds picks a; one using priors picks b.
    """
    items: list[dict[str, Any]] = []
    for r in generate():
        if r["combo_bin"] == "atypical" and r["answer"] != r["typical_color"]:
            items.append({
                "question": f"{r['sentence']} {r['question']}",
                "choice_a": r["answer"],
                "choice_b": r["typical_color"],
                "correct": "a",
            })
    return items
