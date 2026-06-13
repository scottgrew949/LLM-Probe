"""
tests/test_t4_matrices.py — T4 theoretical similarity matrices.

Encodes the rule spec from docs/methods/t4-ontology-matrices.md §3 (confirmed
2026-06-13). Each test pins a DISTINCTIVE cell — the cell where one theory
disagrees with the others — because those are what make the four matrices
discriminable. Pure logic, no model: runnable as `.venv/bin/python -m pytest`.
"""

from __future__ import annotations

import numpy as np
import pytest

from stimuli.theoretical_matrices.t4_matrices import (
    DUL_CATEGORIES,
    Entity,
    build_all_theory_matrices,
    build_fourdim_matrix,
    build_nominalism_matrix,
    build_platonism_matrix,
    build_trope_matrix,
    corr_between_matrices,
    frequency_similarity_matrix,
    pairwise_theory_correlations,
)

# ── A small entity set spanning every stratum + the linked property/trope cases ──
ENTITIES: list[Entity] = [
    Entity("socrates", "concrete", "dul:PhysicalObject"),
    Entity("everest", "concrete", "dul:PhysicalObject"),
    Entity("seven", "abstract", "dul:Abstract"),
    Entity("justice", "abstract", "dul:SocialObject"),
    Entity("wisdom", "property", "dul:Concept", instances=("socrates", "plato")),
    Entity("being_prime", "property", "dul:Concept", instances=("seven",)),
    Entity("fall_of_rome", "event", "dul:Event"),
    Entity("socrates_wisdom", "trope", "dul:Quality", universal="wisdom", bearer="socrates"),
    Entity("plato_wisdom", "trope", "dul:Quality", universal="wisdom", bearer="plato"),
    Entity("sherlock", "fictional", "IntentionalObject"),
]


def _idx(name: str) -> int:
    for i, e in enumerate(ENTITIES):
        if e.name == name:
            return i
    raise KeyError(name)


# ── Structural properties (all four matrices) ───────────────────────────────────
@pytest.mark.parametrize("builder", [
    build_platonism_matrix, build_nominalism_matrix, build_trope_matrix, build_fourdim_matrix,
])
class TestMatrixShape:
    def test_symmetric(self, builder):
        m = builder(ENTITIES)
        assert np.allclose(m, m.T)

    def test_unit_diagonal(self, builder):
        m = builder(ENTITIES)
        assert np.allclose(np.diag(m), 1.0)

    def test_binary_similarity(self, builder):
        m = builder(ENTITIES)
        assert set(np.unique(m)).issubset({0.0, 1.0})

    def test_square_n(self, builder):
        m = builder(ENTITIES)
        assert m.shape == (len(ENTITIES), len(ENTITIES))


# ── Platonism: abstract realm clusters; universals unified ──────────────────────
class TestPlatonism:
    def setup_method(self):
        self.m = build_platonism_matrix(ENTITIES)

    def test_two_abstracts_cluster(self):
        # number 7 and justice both in the abstract realm → close
        assert self.m[_idx("seven"), _idx("justice")] == 1.0

    def test_abstract_concrete_cleavage(self):
        assert self.m[_idx("seven"), _idx("socrates")] == 0.0

    def test_trope_instantiates_its_universal(self):
        # Socrates' wisdom → the one universal wisdom
        assert self.m[_idx("socrates_wisdom"), _idx("wisdom")] == 1.0

    def test_two_tropes_share_the_universal(self):
        # both wisdom-tropes point to the same universal → related under Platonism
        assert self.m[_idx("socrates_wisdom"), _idx("plato_wisdom")] == 1.0


# ── Nominalism: no abstract realm; properties sit near instances ────────────────
class TestNominalism:
    def setup_method(self):
        self.m = build_nominalism_matrix(ENTITIES)

    def test_no_abstract_cluster(self):
        # the Platonism-distinctive cell flips to 0
        assert self.m[_idx("seven"), _idx("justice")] == 0.0

    def test_property_near_instance(self):
        assert self.m[_idx("wisdom"), _idx("socrates")] == 1.0

    def test_trope_near_bearer(self):
        assert self.m[_idx("socrates_wisdom"), _idx("socrates")] == 1.0


# ── Trope theory: properties particular; no universals ──────────────────────────
class TestTrope:
    def setup_method(self):
        self.m = build_trope_matrix(ENTITIES)

    def test_property_trope_no_shared_universal(self):
        # the confirmed key cell: 0 (no universal to share)
        assert self.m[_idx("socrates_wisdom"), _idx("wisdom")] == 0.0

    def test_two_tropes_resemble(self):
        # resemblance-class trope theory: same-kind tropes form a resemblance
        # class → 1. Still distinct from Platonism, which clusters via the ONE
        # universal entity (property↔trope = 1); trope theory keeps property↔trope = 0.
        assert self.m[_idx("socrates_wisdom"), _idx("plato_wisdom")] == 1.0

    def test_trope_near_bearer(self):
        assert self.m[_idx("socrates_wisdom"), _idx("socrates")] == 1.0


# ── 4-dimensionalism: object/event split collapses ──────────────────────────────
class TestFourDim:
    def setup_method(self):
        self.m = build_fourdim_matrix(ENTITIES)

    def test_object_event_merge(self):
        assert self.m[_idx("socrates"), _idx("fall_of_rome")] == 1.0

    def test_object_event_distinct_under_others(self):
        # the 4D-distinctive cell is 0 under all three 3D theories
        for builder in (build_platonism_matrix, build_nominalism_matrix, build_trope_matrix):
            m = builder(ENTITIES)
            assert m[_idx("socrates"), _idx("fall_of_rome")] == 0.0


# ── Discriminability + faithfulness machinery ───────────────────────────────────
class TestMachinery:
    def test_all_builders_present(self):
        mats = build_all_theory_matrices(ENTITIES)
        assert set(mats) == {"platonism", "nominalism", "trope", "fourdim"}

    def test_pairwise_correlations_reported(self):
        mats = build_all_theory_matrices(ENTITIES)
        corrs = pairwise_theory_correlations(mats)
        # 4 choose 2 = 6 pairs
        assert len(corrs) == 6
        for v in corrs.values():
            assert -1.0 <= v <= 1.0

    def test_matrices_are_not_identical(self):
        # discriminability sanity: no two theory matrices are the same
        mats = build_all_theory_matrices(ENTITIES)
        names = list(mats)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                assert not np.array_equal(mats[names[a]], mats[names[b]])

    def test_frequency_similarity_matrix(self):
        freqs = [1e-3, 1e-3, 1e-6]  # first two within 1 order, third far
        fm = frequency_similarity_matrix(freqs, max_log10_diff=1.0)
        assert fm[0, 1] == 1.0
        assert fm[0, 2] == 0.0
        assert np.allclose(np.diag(fm), 1.0)

    def test_corr_between_identical_is_one(self):
        m = build_platonism_matrix(ENTITIES)
        assert corr_between_matrices(m, m) == pytest.approx(1.0)


# ── Fail-loud validation (the deferred DUL-category check) ──────────────────────
class TestValidation:
    def test_unknown_dul_category_raises(self):
        bad = [Entity("x", "concrete", "dul:NotARealCategory")]
        with pytest.raises(ValueError, match="DUL category"):
            build_platonism_matrix(bad)

    def test_stratum_category_mismatch_raises(self):
        # concrete must map to PhysicalObject, not Event
        bad = [Entity("x", "concrete", "dul:Event")]
        with pytest.raises(ValueError, match="stratum"):
            build_platonism_matrix(bad)

    def test_known_categories_listed(self):
        assert "dul:PhysicalObject" in DUL_CATEGORIES
        assert "dul:Quality" in DUL_CATEGORIES
