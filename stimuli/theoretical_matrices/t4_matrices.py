"""
stimuli/theoretical_matrices/t4_matrices.py — Theoretical similarity matrices for T4.

─── CONCEPT: four metaphysical theories, four predicted geometries ───────────
T4 asks which formal ontology best describes how a model arranges entity
representations. We encode each rival theory as a pairwise SIMILARITY matrix
over the entity stimuli (1 = the theory says these are alike, 0 = unlike),
matching probes.run_rsa's convention (higher = closer). RSA then asks which
theory's predicted geometry the model's actual geometry correlates with.

Entity categories come from DOLCE+DnS Ultralite (DUL) — a published, citable
upper ontology — NOT from intuition. See docs/methods/t4-ontology-matrices.md
for the scaffold, the entity→category mapping, and the per-theory rules this
module implements. Rules confirmed 2026-06-13.

The four theories (distinctive signature in brackets):
  Platonism      — abstracta are a real, unified realm   [abstract realm clusters;
                   tropes instantiate the ONE universal]
  Nominalism     — no abstracta/universals               [no abstract cluster;
                   properties sit near their instances]
  Trope theory   — properties are particular tropes       [resemblance classes:
                   same-kind tropes cluster; no universal entity: property↔trope = 0]
  4-dimensionalism — objects are spacetime worms         [object/event split
                   collapses: PhysicalObject ~ Event]

These distinctive cells are what make the matrices discriminable. Because all
four range over the same entities they share some structure, so we do NOT assert
a hard pairwise decorrelation (that is right for T1b's two crossed matrices, wrong
for four overlapping ones). Instead pairwise_theory_correlations() reports the
full inter-matrix correlation as an artifact, and the T4 verdict (in the runner)
requires the winning theory's RSA to beat the runner-up beyond the Mantel null —
a model comparison, not merely winner > null.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── DUL category scaffold (V13). Allowed categories + stratum mapping. ──────────
# "IntentionalObject" is a documented sentinel for fictional/intentional objects,
# which DUL has no native category for (see docs §2) — kept out of the realm rules.
DUL_CATEGORIES: frozenset[str] = frozenset({
    "dul:PhysicalObject",
    "dul:Event",
    "dul:Abstract",
    "dul:Concept",
    "dul:Quality",
    "dul:SocialObject",
    "IntentionalObject",
})

STRATUM_TO_DUL: dict[str, frozenset[str]] = {
    "concrete": frozenset({"dul:PhysicalObject"}),
    "abstract": frozenset({"dul:Abstract", "dul:SocialObject"}),
    "property": frozenset({"dul:Concept"}),
    "event": frozenset({"dul:Event"}),
    "trope": frozenset({"dul:Quality"}),
    "fictional": frozenset({"IntentionalObject"}),
}

_ABSTRACT_REALM_STRATA: frozenset[str] = frozenset({"abstract", "property"})
_CONCRETE_KIND_STRATA: frozenset[str] = frozenset({"concrete", "event"})


@dataclass(frozen=True)
class Entity:
    """
    One entity stimulus, tagged with its DUL category and the links the theory
    rules need. `universal`/`bearer`/`instances` carry the metadata the stimulus
    set (#16) must supply; without them nominalism and trope rules cannot compute.

    name:       stable identifier, also the referent used by bearer/instances links.
    stratum:    concrete | abstract | property | event | trope | fictional.
    dul_category: a DUL category IRI-localname (validated against DUL_CATEGORIES).
    universal:  for a trope, the name of the property it falls under (e.g. 'wisdom').
    bearer:     for a trope, the object it inheres in (e.g. 'socrates').
    instances:  for a property, the concrete bearers exhibiting it.
    """
    name: str
    stratum: str
    dul_category: str
    universal: str | None = None
    bearer: str | None = None
    instances: tuple[str, ...] = field(default_factory=tuple)


# ── Validation (fail-loud — the deferred DUL-category check) ────────────────────
def _validate(entities: list[Entity]) -> None:
    for e in entities:
        if e.dul_category not in DUL_CATEGORIES:
            raise ValueError(
                f"Unknown DUL category {e.dul_category!r} for entity {e.name!r}. "
                f"Allowed: {sorted(DUL_CATEGORIES)}."
            )
        allowed = STRATUM_TO_DUL.get(e.stratum)
        if allowed is None:
            raise ValueError(
                f"Unknown stratum {e.stratum!r} for entity {e.name!r}. "
                f"Allowed: {sorted(STRATUM_TO_DUL)}."
            )
        if e.dul_category not in allowed:
            raise ValueError(
                f"stratum/category mismatch for {e.name!r}: stratum {e.stratum!r} "
                f"requires DUL category in {sorted(allowed)}, got {e.dul_category!r}."
            )


def _build(entities: list[Entity], alike) -> np.ndarray:
    """Build a symmetric 0/1 similarity matrix from a pairwise `alike(a, b)` predicate."""
    _validate(entities)
    n = len(entities)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        m[i, i] = 1.0
        for j in range(i + 1, n):
            v = 1.0 if (alike(entities[i], entities[j]) or alike(entities[j], entities[i])) else 0.0
            m[i, j] = v
            m[j, i] = v
    return m


# ── The four theory rules (see module docstring + docs §3) ──────────────────────
def _platonism_alike(a: Entity, b: Entity) -> bool:
    # abstract realm (numbers, universals, social abstracta) clusters
    if a.stratum in _ABSTRACT_REALM_STRATA and b.stratum in _ABSTRACT_REALM_STRATA:
        return True
    # a trope instantiates the ONE universal it falls under
    if a.stratum == "trope" and b.stratum == "property" and a.universal == b.name:
        return True
    # two tropes of the same universal are related via that single universal
    if a.stratum == "trope" and b.stratum == "trope" and a.universal == b.universal and a.universal is not None:
        return True
    # concretes / events group by kind
    if a.stratum == b.stratum and a.stratum in _CONCRETE_KIND_STRATA:
        return True
    return False


def _nominalism_alike(a: Entity, b: Entity) -> bool:
    # NO abstract realm: two abstract-realm entities do not cluster for being abstract
    # concretes group by kind
    if a.stratum == b.stratum and a.stratum in _CONCRETE_KIND_STRATA:
        return True
    # a property sits near its concrete instances
    if a.stratum == "property" and b.name in a.instances:
        return True
    # a trope sits near its bearer (its concrete grounding)
    if a.stratum == "trope" and a.bearer == b.name:
        return True
    return False


def _trope_alike(a: Entity, b: Entity) -> bool:
    # Resemblance-class trope theory (Williams, Campbell): properties are
    # particular tropes; exactly-resembling tropes form a resemblance CLASS that
    # does the work universals would → same-kind tropes cluster (=1). No universal
    # ENTITY exists, so property↔trope = 0 and property↔property = 0.
    # concretes / events group by kind
    if a.stratum == b.stratum and a.stratum in _CONCRETE_KIND_STRATA:
        return True
    # abstract objects (numbers) grouped by kind
    if a.stratum == b.stratum and a.stratum == "abstract":
        return True
    # resemblance class: tropes of the same universal-kind resemble → cluster
    if a.stratum == "trope" and b.stratum == "trope" and a.universal == b.universal and a.universal is not None:
        return True
    # a trope is bound to its bearer
    if a.stratum == "trope" and a.bearer == b.name:
        return True
    return False


def _fourdim_alike(a: Entity, b: Entity) -> bool:
    # the distinctive move: object/event split collapses — both are 4D worms
    if a.stratum in _CONCRETE_KIND_STRATA and b.stratum in _CONCRETE_KIND_STRATA:
        return True
    # otherwise default grouping by kind
    if a.stratum == b.stratum:
        return True
    return False


def build_platonism_matrix(entities: list[Entity]) -> np.ndarray:
    return _build(entities, _platonism_alike)


def build_nominalism_matrix(entities: list[Entity]) -> np.ndarray:
    return _build(entities, _nominalism_alike)


def build_trope_matrix(entities: list[Entity]) -> np.ndarray:
    return _build(entities, _trope_alike)


def build_fourdim_matrix(entities: list[Entity]) -> np.ndarray:
    return _build(entities, _fourdim_alike)


THEORY_BUILDERS = {
    "platonism": build_platonism_matrix,
    "nominalism": build_nominalism_matrix,
    "trope": build_trope_matrix,
    "fourdim": build_fourdim_matrix,
}


def build_all_theory_matrices(entities: list[Entity]) -> dict[str, np.ndarray]:
    """All four theory similarity matrices, keyed by theory name."""
    return {name: builder(entities) for name, builder in THEORY_BUILDERS.items()}


# ── Discriminability + faithfulness ─────────────────────────────────────────────
def corr_between_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Pearson correlation of the two matrices' off-diagonal upper triangles."""
    n = matrix_a.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    a_flat = matrix_a[rows, cols]
    b_flat = matrix_b[rows, cols]
    if a_flat.std() == 0 or b_flat.std() == 0:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def pairwise_theory_correlations(matrices: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """
    Inter-matrix correlation for every theory pair — the discriminability artifact.
    Written before any model run so collinear theories are visible up front; the
    verdict (in the runner) requires winner > runner-up beyond the Mantel null.
    """
    names = list(matrices)
    out: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[(names[i], names[j])] = corr_between_matrices(matrices[names[i]], matrices[names[j]])
    return out


def frequency_similarity_matrix(freqs: list[float], max_log10_diff: float = 1.0) -> np.ndarray:
    """
    Faithfulness control. M[i,j] = 1.0 iff entities i, j have corpus frequencies
    within `max_log10_diff` orders of magnitude. Correlating a theory matrix with
    this exposes a 'match' that is really word-frequency structure (V7 / surface
    null), not ontology.
    """
    log_freqs = np.log10(np.asarray(freqs, dtype=float))
    n = len(freqs)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        m[i, i] = 1.0
        for j in range(i + 1, n):
            v = 1.0 if abs(log_freqs[i] - log_freqs[j]) <= max_log10_diff else 0.0
            m[i, j] = v
            m[j, i] = v
    return m
