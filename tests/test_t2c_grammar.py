"""
tests/test_t2c_grammar.py — Tests for T2c two-dimensional semantics grammar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from stimuli.grammars.t2c import generate


class TestT2cGrammar:
    def test_generates_requested_count(self):
        pairs = generate(n=30)
        assert len(pairs) == 30

    def test_all_three_conditions_present(self):
        pairs = generate(n=60)
        label_set = set()
        for pair in pairs:
            label_set.add(pair["label_a"])
            label_set.add(pair["label_b"])
        required_conditions = {
            "primary_sensitive",
            "secondary_necessary",
            "primary_secondary_dissociation",
        }
        assert required_conditions.issubset(label_set)

    def test_pairs_have_required_fields(self):
        pairs = generate(n=10)
        for pair in pairs:
            assert "sentence_a" in pair
            assert "sentence_b" in pair
            assert "label_a" in pair
            assert "label_b" in pair
            assert isinstance(pair["sentence_a"], str)
            assert len(pair["sentence_a"]) > 10

    def test_no_empty_sentences(self):
        pairs = generate(n=20)
        for pair in pairs:
            assert pair["sentence_a"].strip() != ""
            assert pair["sentence_b"].strip() != ""

    def test_reproducible_with_same_seed(self):
        pairs_first = generate(n=20, seed=7)
        pairs_second = generate(n=20, seed=7)
        assert [p["sentence_a"] for p in pairs_first] == [p["sentence_a"] for p in pairs_second]
