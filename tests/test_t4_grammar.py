"""
tests/test_t4_grammar.py — T4 entity stimulus grammar.

The grammar must emit a coherent, internally-linked entity web (the trope /
property / bearer links that t4_matrices consumes). The make-or-break test is
referential integrity: a dangling link silently corrupts the nominalism and
trope matrices. Pure logic — `.venv/bin/python -m pytest`.
"""

from __future__ import annotations

import pytest

from stimuli.grammars.t4 import generate, to_entities
from stimuli.theoretical_matrices.t4_matrices import (
    STRATUM_TO_DUL,
    build_all_theory_matrices,
    pairwise_theory_correlations,
)

RECORDS = generate()
NAMES = {r["name"] for r in RECORDS}


class TestBasics:
    def test_nonempty_and_deterministic(self):
        assert len(RECORDS) > 0
        assert [r["name"] for r in generate()] == [r["name"] for r in RECORDS]

    def test_unique_names(self):
        names = [r["name"] for r in RECORDS]
        assert len(names) == len(set(names))

    def test_all_strata_present(self):
        strata = {r["stratum"] for r in RECORDS}
        assert strata == set(STRATUM_TO_DUL)  # every stratum represented

    def test_carrier_contains_display(self):
        for r in RECORDS:
            assert r["display"] in r["carrier"]

    def test_frequency_recorded(self):
        for r in RECORDS:
            assert isinstance(r["frequency_log10"], float)


class TestCategoryConsistency:
    def test_stratum_category_consistent(self):
        for r in RECORDS:
            assert r["dul_category"] in STRATUM_TO_DUL[r["stratum"]]


class TestReferentialIntegrity:
    """Every link must resolve to an entity in the set — the make-or-break check."""

    def test_property_instances_resolve(self):
        for r in RECORDS:
            if r["stratum"] == "property":
                assert len(r["instances"]) > 0, f"{r['name']} has no instances"
                for inst in r["instances"]:
                    assert inst in NAMES, f"{r['name']} instance {inst!r} dangling"

    def test_trope_links_resolve(self):
        prop_names = {r["name"] for r in RECORDS if r["stratum"] == "property"}
        for r in RECORDS:
            if r["stratum"] == "trope":
                assert r["universal"] in prop_names, f"{r['name']} universal dangling"
                assert r["bearer"] in NAMES, f"{r['name']} bearer dangling"

    def test_trope_display_is_possessive(self):
        for r in RECORDS:
            if r["stratum"] == "trope":
                assert "'" in r["display"]  # "Socrates' wisdom" / "the ruby's redness"

    def test_nonproperty_nontrope_links_empty(self):
        for r in RECORDS:
            if r["stratum"] not in ("property", "trope"):
                assert not r["instances"]
                assert r["universal"] is None
                assert r["bearer"] is None


class TestFeedsMatrices:
    def test_to_entities_validates(self):
        # to_entities must yield records the matrix builders accept (no raise)
        ents = to_entities(RECORDS)
        build_all_theory_matrices(ents)

    def test_matrices_discriminable_on_real_set(self):
        ents = to_entities(RECORDS)
        mats = build_all_theory_matrices(ents)
        corrs = pairwise_theory_correlations(mats)
        # on the real entity web, no two theory matrices are perfectly collinear
        for pair, c in corrs.items():
            assert abs(c) < 0.999, f"{pair} collinear (corr={c:.3f})"

    def test_at_least_one_trope_and_property_linked(self):
        # a trope whose universal is an actual property, sharing a bearer that is
        # an instance of that property — the linked triangle the rules hinge on
        ents = to_entities(RECORDS)
        props = {e.name: e for e in ents if e.stratum == "property"}
        found = False
        for e in ents:
            if e.stratum == "trope" and e.universal in props:
                if e.bearer in props[e.universal].instances:
                    found = True
                    break
        assert found, "no fully-linked trope↔property↔bearer triangle"
