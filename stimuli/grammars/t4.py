"""
stimuli/grammars/t4.py — Entity stimulus grammar for Thread T4 (Quinean ontology).

─── CONCEPT: an internally-linked entity web for RSA ─────────────────────────
T4 measures the model's representational GEOMETRY over entities (RSA against the
four theory matrices in stimuli/theoretical_matrices/t4_matrices.py). Unlike the
other threads, T4 stimuli are individual ENTITIES, not minimal pairs — RSA needs
a set, not contrasts. Each entity is presented in a neutral, matched carrier
sentence; the entity's token span is located at extraction time (model-specific),
not here.

The entity set is a coherent web, not a flat list. Canonical philosophical
exemplars are hand-curated per stratum (Socrates, the number seven, wisdom, the
fall of Rome, Sherlock…); tropes are then GENERATED from an instantiation
relation (property → bearers), which guarantees the trope/property/bearer links
are internally consistent. Those links are exactly what the nominalism and trope
matrices consume — a dangling link silently corrupts them (see the referential-
integrity test).

DUL category mapping per docs/methods/t4-ontology-matrices.md §2. Strata:
concrete | abstract | property | event | trope | fictional.
"""

from __future__ import annotations

from typing import Any

from wordfreq import word_frequency

from stimuli.theoretical_matrices.t4_matrices import Entity

# ── Canonical exemplars per stratum: (name, display, dul_category) ──────────────
# `name` is the link id (lowercase, article-free); `display` is the surface form.

_CONCRETE: list[tuple[str, str]] = [
    ("socrates", "Socrates"),
    ("plato", "Plato"),
    ("aristotle", "Aristotle"),
    ("everest", "Mount Everest"),
    ("moon", "the Moon"),
    ("ruby", "the ruby"),
    ("rose", "the rose"),
    ("diamond", "the diamond"),
    ("sugar", "the sugar"),
    ("honey", "the honey"),
    ("snow", "the snow"),
    ("swan", "the swan"),
    ("ball", "the ball"),
]
# More abstracta than any other stratum: abstract↔abstract is the strongest
# nominalism/trope differentiator (nom=0, trope=1), and those pairs grow ~n^2.
_ABSTRACT: list[tuple[str, str, str]] = [
    ("seven", "the number seven", "dul:Abstract"),
    ("three", "the number three", "dul:Abstract"),
    ("two", "the number two", "dul:Abstract"),
    ("five", "the number five", "dul:Abstract"),
    ("eleven", "the number eleven", "dul:Abstract"),
    ("twelve", "the number twelve", "dul:Abstract"),
    ("justice", "justice", "dul:SocialObject"),
    ("beauty", "beauty", "dul:SocialObject"),
    ("equality", "equality", "dul:SocialObject"),
    ("truth", "truth", "dul:SocialObject"),
]
_PROPERTY: list[tuple[str, str]] = [
    ("wisdom", "wisdom"),
    ("courage", "courage"),
    ("redness", "redness"),
    ("roundness", "roundness"),
    ("hardness", "hardness"),
    ("sweetness", "sweetness"),
    ("whiteness", "whiteness"),
    ("being_prime", "being prime"),
]
_EVENT: list[tuple[str, str]] = [
    ("fall_of_rome", "the fall of Rome"),
    ("french_revolution", "the French Revolution"),
    ("big_bang", "the Big Bang"),
    ("renaissance", "the Renaissance"),
]
_FICTIONAL: list[tuple[str, str]] = [
    ("sherlock", "Sherlock Holmes"),
    ("hamlet", "Hamlet"),
    ("pegasus", "Pegasus"),
    ("gandalf", "Gandalf"),
]

# property name → bearer names that instantiate it. Drives property.instances AND
# trope generation. Bearers must be names present above. `being_prime` is
# instantiated by abstract numbers (instances need not be concrete); it generates
# no trope (a "primeness trope" is not a natural linguistic object).
_INSTANTIATION: dict[str, list[str]] = {
    "wisdom": ["socrates", "plato", "aristotle"],
    "courage": ["socrates"],
    "redness": ["ruby", "rose"],
    "roundness": ["moon", "ball"],
    "hardness": ["diamond", "ruby"],
    "sweetness": ["sugar", "honey"],
    "whiteness": ["snow", "swan"],
    "being_prime": ["seven", "three", "five", "eleven"],
}
# properties that form natural tropes (a subset of _INSTANTIATION keys).
# being_prime excluded — a "primeness trope" is not a natural linguistic object.
_TROPE_FORMING: frozenset[str] = frozenset({
    "wisdom", "courage", "redness", "roundness", "hardness", "sweetness", "whiteness",
})

_CARRIER = "Consider {display}."

# stopwords stripped before computing a representative word frequency
_STOPWORDS: frozenset[str] = frozenset({"the", "a", "an", "of", "number", "being"})


def _display_by_name() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, display in _CONCRETE:
        out[name] = display
    for name, display, _ in _ABSTRACT:
        out[name] = display
    for name, display in _PROPERTY:
        out[name] = display
    return out


def _possessive(display: str) -> str:
    """'Socrates' → "Socrates'"; 'the Moon' → "the Moon's"; 'the ruby' → "the ruby's"."""
    return display + "'" if display.endswith("s") else display + "'s"


def _representative_frequency_log10(display: str) -> float:
    """Mean log10 corpus frequency over content words in the display (V7 / faithfulness)."""
    words = [w.strip(".,'") for w in display.lower().split()]
    content = [w for w in words if w and w not in _STOPWORDS]
    if not content:
        content = words
    import math
    freqs = [word_frequency(w, "en") for w in content]
    freqs = [f for f in freqs if f > 0] or [1e-8]
    return float(sum(math.log10(f) for f in freqs) / len(freqs))


def _record(name: str, display: str, stratum: str, dul_category: str,
            universal: str | None = None, bearer: str | None = None,
            instances: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "thread_id": "t4",
        "name": name,
        "display": display,
        "stratum": stratum,
        "dul_category": dul_category,
        "universal": universal,
        "bearer": bearer,
        "instances": list(instances),
        "carrier": _CARRIER.format(display=display),
        "frequency_log10": _representative_frequency_log10(display),
    }


def generate(n: int | None = None) -> list[dict[str, Any]]:
    """
    The full T4 entity set as records (see module docstring). Deterministic.

    Order: concrete, abstract, property (with resolved instances), event,
    fictional, then generated tropes. `n`, if given, caps the result (the
    canonical core is the minimum useful set; grow the pools to scale).
    """
    records: list[dict[str, Any]] = []

    for name, display in _CONCRETE:
        records.append(_record(name, display, "concrete", "dul:PhysicalObject"))
    for name, display, cat in _ABSTRACT:
        records.append(_record(name, display, "abstract", cat))
    for name, display in _PROPERTY:
        instances = tuple(_INSTANTIATION.get(name, ()))
        records.append(_record(name, display, "property", "dul:Concept", instances=instances))
    for name, display in _EVENT:
        records.append(_record(name, display, "event", "dul:Event"))
    for name, display in _FICTIONAL:
        records.append(_record(name, display, "fictional", "IntentionalObject"))

    # generated tropes — links are consistent by construction
    displays = _display_by_name()
    for prop_name in _TROPE_FORMING:
        prop_display = displays[prop_name]
        for bearer in _INSTANTIATION[prop_name]:
            bearer_display = displays[bearer]
            trope_display = f"{_possessive(bearer_display)} {prop_display}"
            records.append(_record(
                name=f"{bearer}_{prop_name}",
                display=trope_display,
                stratum="trope",
                dul_category="dul:Quality",
                universal=prop_name,
                bearer=bearer,
            ))

    for index, r in enumerate(records):
        r["entity_id"] = f"t4_{index + 1:04d}"

    return records if n is None else records[:n]


def to_entities(records: list[dict[str, Any]]) -> list[Entity]:
    """Convert grammar records to t4_matrices.Entity for matrix construction."""
    return [
        Entity(
            name=r["name"],
            stratum=r["stratum"],
            dul_category=r["dul_category"],
            universal=r["universal"],
            bearer=r["bearer"],
            instances=tuple(r["instances"]),
        )
        for r in records
    ]
