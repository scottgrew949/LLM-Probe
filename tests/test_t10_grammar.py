"""tests/test_t10_grammar.py — T10 attribute-binding stimulus grammar."""

from __future__ import annotations

from stimuli.grammars.t10 import (
    COLORS,
    OBJECT_TYPICAL_COLORS,
    generate,
    generate_behavioral_items,
)

RECORDS = generate()


class TestBasics:
    def test_deterministic(self):
        assert [r["item_id"] for r in generate()] == [r["item_id"] for r in RECORDS]

    def test_both_bins_present(self):
        bins = {r["combo_bin"] for r in RECORDS}
        assert bins == {"typical", "atypical"}

    def test_balanced_bins(self):
        typ = sum(r["combo_bin"] == "typical" for r in RECORDS)
        atyp = sum(r["combo_bin"] == "atypical" for r in RECORDS)
        assert typ == atyp  # one of each per object


class TestSceneIntegrity:
    def test_answer_is_target_color_and_in_scene(self):
        for r in RECORDS:
            assert f"{r['answer']} {r['probed_object']}" in r["scene"]

    def test_two_distinct_objects(self):
        for r in RECORDS:
            assert r["probed_object"] != r["distractor_object"]
            assert r["probed_object"] in r["scene"]
            assert r["distractor_object"] in r["scene"]

    def test_question_targets_probed_object(self):
        for r in RECORDS:
            assert r["probed_object"] in r["question"]


class TestFrequencyControl:
    def test_typical_binding_uses_a_canonical_color(self):
        for r in RECORDS:
            if r["combo_bin"] == "typical":
                assert r["answer"] in OBJECT_TYPICAL_COLORS[r["probed_object"]]

    def test_atypical_binding_is_noncanonical(self):
        for r in RECORDS:
            if r["combo_bin"] == "atypical":
                assert r["answer"] not in OBJECT_TYPICAL_COLORS[r["probed_object"]]

    def test_words_common_in_both_bins(self):
        # the control: individual word frequencies are not lower in the atypical bin
        for bin_name in ("typical", "atypical"):
            for r in RECORDS:
                if r["combo_bin"] == bin_name:
                    assert r["color_word_freq_log10"] > -6.0
                    assert r["target_word_freq_log10"] > -7.0


class TestBehavioralItems:
    def test_items_discriminate_binding_from_prior(self):
        items = generate_behavioral_items()
        assert len(items) > 0
        for it in items:
            assert it["choice_a"] != it["choice_b"]   # bound color vs typical prior
            assert it["correct"] == "a"
            assert it["choice_a"] in COLORS and it["choice_b"] in COLORS
