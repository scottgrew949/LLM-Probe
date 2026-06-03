"""
tests/test_t1d_grammar.py — Tests for T1d causal identification grammar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from stimuli.grammars.t1d import generate


class TestT1dGrammar:
    def test_generates_requested_count(self):
        pairs = generate(n=40)
        assert len(pairs) == 40

    def test_all_four_conditions_present(self):
        pairs = generate(n=80)
        label_set = set()
        for pair in pairs:
            label_set.add(pair["label_a"])
            label_set.add(pair["label_b"])
        required_conditions = {
            "back_door_adjustable",
            "front_door_adjustable",
            "confounded_not_adjustable",
            "unconfounded_control",
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

    def test_conditions_balanced(self):
        pairs = generate(n=80)
        condition_counts: dict[str, int] = {}
        for pair in pairs:
            for label in [pair["label_a"], pair["label_b"]]:
                condition_counts[label] = condition_counts.get(label, 0) + 1
        counts = list(condition_counts.values())
        assert max(counts) <= min(counts) * 2
