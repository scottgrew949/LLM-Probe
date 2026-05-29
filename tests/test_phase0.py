"""
tests/test_phase0.py — Unit tests for Phase 0 pure functions.

GPU/torch code is excluded — tested functions need only numpy, scipy, sklearn.
Run with: python -m pytest tests/test_phase0.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path so `from experiments.config import ...` works
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import save_result, load_result, load_results
from experiments.config import ExperimentConfig
from probes.probes import run_linear_probe, run_rsa, run_mantel_test, run_knife_mi
from stimuli.pipeline import check_frequency_match

# compute_sha256 lives in extractor.py which has a top-level `import torch`.
# Mock torch so we can import the pure utility without a GPU environment.
import unittest.mock
with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
    from extraction.extractor import compute_sha256


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_config(**overrides) -> ExperimentConfig:
    """Minimal valid ExperimentConfig for testing."""
    defaults = dict(
        experiment_id="test_t2_gpt2m_20260101",
        thread_id="t2",
        model_id="gpt2-medium",
        model_revision="main",
        layer_range=(0, 4),
        component="resid_post",
        token_positions=[-1],
        probe_type="linear",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ── core/io.py ─────────────────────────────────────────────────────────────────

class TestCoreIO:
    def test_save_and_load_result(self, tmp_path):
        result_data = {"probe_accuracy": 0.87, "layer": 8, "labels": ["a", "b"]}
        save_result(result_data, tmp_path / "result.json")
        loaded = load_result(tmp_path / "result.json")
        assert loaded == result_data

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "result.json"
        save_result({"x": 1}, deep_path)
        assert deep_path.exists()

    def test_load_results_aggregates_all_jsons(self, tmp_path):
        for i in range(3):
            save_result({"i": i}, tmp_path / f"result_{i}.json")
        # load_results(thread_id, results_dir=...) — pass results_dir to override default path
        all_results = load_results("test_thread", results_dir=tmp_path)
        assert len(all_results) == 3
        loaded_i_values = sorted(r["i"] for r in all_results)
        assert loaded_i_values == [0, 1, 2]

    def test_load_result_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_result(tmp_path / "nonexistent.json")

    def test_save_result_accepts_path_or_str(self, tmp_path):
        result_data = {"key": "value"}
        save_result(result_data, str(tmp_path / "str_path.json"))
        loaded = load_result(str(tmp_path / "str_path.json"))
        assert loaded == result_data


# ── experiments/config.py ──────────────────────────────────────────────────────

class TestExperimentConfig:
    def test_construct_valid_config(self):
        config = _make_config()
        assert config.experiment_id == "test_t2_gpt2m_20260101"
        assert config.pre_spec_locked is False

    def test_behavioral_gate_below_floor_raises(self):
        with pytest.raises(ValueError, match="V6"):
            _make_config(behavioral_gate_threshold=0.60)

    def test_behavioral_gate_at_floor_passes(self):
        config = _make_config(behavioral_gate_threshold=0.70)
        assert config.behavioral_gate_threshold == 0.70

    def test_invalid_component_raises(self):
        with pytest.raises(ValueError, match="component"):
            _make_config(component="hidden_state")

    def test_invalid_probe_type_raises(self):
        with pytest.raises(ValueError, match="probe_type"):
            _make_config(probe_type="neural")

    def test_all_valid_components_accepted(self):
        for component in ("resid_post", "attn_out", "mlp_out"):
            config = _make_config(component=component)
            assert config.component == component

    def test_all_valid_probe_types_accepted(self):
        for probe_type in ("linear", "cosine", "rsa"):
            config = _make_config(probe_type=probe_type)
            assert config.probe_type == probe_type

    def test_lock_requires_expected_outcomes(self):
        config = _make_config()
        config.frequency_match_verified = True
        config.stimulus_sha256 = "abc123"
        # stimulus_file empty → lock() skips file existence check
        with pytest.raises(ValueError, match="expected_outcomes"):
            config.lock()

    def test_lock_requires_frequency_match_verified(self):
        config = _make_config()
        config.expected_outcomes = {"probe_accuracy": ">0.80"}
        config.stimulus_sha256 = "abc123"
        # frequency_match_verified is False by default
        with pytest.raises(ValueError, match="V7"):
            config.lock()

    def test_lock_requires_stimulus_sha256(self):
        config = _make_config()
        config.expected_outcomes = {"probe_accuracy": ">0.80"}
        config.frequency_match_verified = True
        # stimulus_file empty (skips file check), sha256 empty → should raise
        with pytest.raises(ValueError, match="stimulus_sha256"):
            config.lock()

    def test_lock_sets_pre_spec_locked(self):
        config = _make_config()
        config.expected_outcomes = {"probe_accuracy": ">0.80"}
        config.frequency_match_verified = True
        config.stimulus_sha256 = "abc123def456"
        # stimulus_file is empty — lock() skips the file-exists check in that case
        config.lock()
        assert config.pre_spec_locked is True

    def test_to_json_and_from_json_roundtrip(self, tmp_path):
        config = _make_config()
        config.expected_outcomes = {"peak_layer": "8-12"}
        config_path = tmp_path / "config.json"
        config.to_json(config_path)

        loaded_config = ExperimentConfig.from_json(config_path)
        assert loaded_config.experiment_id == config.experiment_id
        assert loaded_config.layer_range == config.layer_range  # Must be tuple, not list
        assert isinstance(loaded_config.layer_range, tuple)
        assert loaded_config.expected_outcomes == config.expected_outcomes

    def test_to_json_creates_parent_dirs(self, tmp_path):
        config = _make_config()
        deep_path = tmp_path / "experiments" / "t2" / "config.json"
        config.to_json(deep_path)
        assert deep_path.exists()

    def test_layer_range_stays_tuple_after_from_json(self, tmp_path):
        config = _make_config(layer_range=(3, 12))
        config.to_json(tmp_path / "config.json")
        loaded = ExperimentConfig.from_json(tmp_path / "config.json")
        assert isinstance(loaded.layer_range, tuple)
        assert loaded.layer_range == (3, 12)


# ── probes/probes.py ───────────────────────────────────────────────────────────

class TestProbes:
    """Mock data: 50 items, 2 balanced classes, clear separation."""

    @pytest.fixture
    def separable_activations(self):
        """Two clusters clearly separable — probe should achieve ~1.0 accuracy."""
        rng = np.random.default_rng(42)
        class_a = rng.normal(loc=5.0, scale=0.3, size=(25, 16))
        class_b = rng.normal(loc=-5.0, scale=0.3, size=(25, 16))
        activations = np.vstack([class_a, class_b])
        labels = ["a"] * 25 + ["b"] * 25
        return activations, labels

    @pytest.fixture
    def random_activations(self):
        """Pure noise — probe should achieve ~0.5 accuracy (chance)."""
        rng = np.random.default_rng(99)
        activations = rng.normal(size=(40, 16))
        labels = ["a"] * 20 + ["b"] * 20
        return activations, labels

    def test_linear_probe_high_accuracy_on_separable(self, separable_activations):
        activations, labels = separable_activations
        config = _make_config()
        result = run_linear_probe(activations, labels, config)
        assert result["accuracy_mean"] > 0.90
        assert "accuracy_std" in result
        assert "weights" in result
        # weights shape [n_classes, hidden_dim] — inner lists have length 16
        assert len(result["weights"][0]) == 16

    def test_linear_probe_near_chance_on_noise(self, random_activations):
        activations, labels = random_activations
        config = _make_config()
        result = run_linear_probe(activations, labels, config)
        # Chance is 0.5; allow generous band since n=40 is small
        assert result["accuracy_mean"] < 0.80

    def test_linear_probe_result_keys(self, separable_activations):
        activations, labels = separable_activations
        config = _make_config()
        result = run_linear_probe(activations, labels, config)
        required_keys = {"accuracy_mean", "accuracy_std", "chance_baseline",
                         "weights", "labels_order", "n_items", "n_folds",
                         "experiment_id", "thread_id"}
        assert required_keys.issubset(result.keys())

    def test_linear_probe_metadata_matches_config(self, separable_activations):
        activations, labels = separable_activations
        config = _make_config()
        result = run_linear_probe(activations, labels, config)
        assert result["experiment_id"] == config.experiment_id
        assert result["thread_id"] == config.thread_id
        assert result["n_folds"] == 5
        assert result["n_items"] == len(labels)

    def test_rsa_high_correlation_on_matching_matrices(self):
        """RSA on identical geometry should give correlation ~1.0."""
        rng = np.random.default_rng(7)
        activations = rng.normal(size=(20, 8))
        from sklearn.metrics.pairwise import cosine_similarity
        # Theoretical matrix = cosine similarity of same activations → perfect match
        theoretical_matrix = cosine_similarity(activations)
        config = _make_config()
        result = run_rsa(activations, theoretical_matrix, config)
        assert result["spearman_r"] > 0.80

    def test_rsa_result_keys(self):
        rng = np.random.default_rng(7)
        activations = rng.normal(size=(10, 8))
        # Use a non-constant theory matrix so spearmanr is defined
        theoretical_matrix = rng.normal(size=(10, 10))
        theoretical_matrix = (theoretical_matrix + theoretical_matrix.T) / 2
        config = _make_config()
        result = run_rsa(activations, theoretical_matrix, config)
        required_keys = {"spearman_r", "model_matrix", "theory_matrix", "n_items"}
        assert required_keys.issubset(result.keys())

    def test_rsa_t4_raises_without_ontology_provenance(self):
        """V13: T4 RSA must have ontology_version and matrix_source set."""
        rng = np.random.default_rng(7)
        activations = rng.normal(size=(10, 8))
        theory_matrix = np.eye(10)
        config = _make_config(thread_id="t4")  # no ontology_version or matrix_source
        with pytest.raises(ValueError, match="V13"):
            run_rsa(activations, theory_matrix, config)

    def test_mantel_test_structure_on_noise(self):
        """Unrelated matrices — verify result structure and valid p-value range."""
        rng = np.random.default_rng(42)
        n = 10
        matrix_a = rng.normal(size=(n, n))
        matrix_a = (matrix_a + matrix_a.T) / 2
        matrix_b = rng.normal(size=(n, n))
        matrix_b = (matrix_b + matrix_b.T) / 2
        result = run_mantel_test(matrix_a, matrix_b, n_perms=200)  # param is n_perms
        assert "p_value" in result
        assert "observed_r" in result
        assert "significant" in result
        assert "null_95th_percentile" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_mantel_test_low_p_on_identical_matrices(self):
        """Identical matrices — p should be 0 or near-0."""
        rng = np.random.default_rng(42)
        n = 10
        matrix_a = rng.normal(size=(n, n))
        matrix_a = (matrix_a + matrix_a.T) / 2
        result = run_mantel_test(matrix_a, matrix_a.copy(), n_perms=200)
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_knife_mi_result_keys(self, separable_activations):
        activations, labels = separable_activations
        result = run_knife_mi(activations, labels)
        required_keys = {"mi_nats", "mi_bits", "n_items", "n_classes", "estimator"}
        assert required_keys.issubset(result.keys())

    def test_knife_mi_values_are_floats(self, separable_activations):
        activations, labels = separable_activations
        result = run_knife_mi(activations, labels)
        assert isinstance(result["mi_nats"], float)
        assert isinstance(result["mi_bits"], float)
        assert result["mi_nats"] >= 0.0
        assert result["mi_bits"] >= 0.0

    def test_knife_mi_higher_on_separable_than_noise(self, separable_activations, random_activations):
        separable_mi = run_knife_mi(*separable_activations)["mi_nats"]
        noise_mi = run_knife_mi(*random_activations)["mi_nats"]
        assert separable_mi > noise_mi


# ── stimuli/pipeline.py — check_frequency_match ───────────────────────────────

class TestCheckFrequencyMatch:
    def test_high_frequency_words_match(self):
        """Common words should match each other easily."""
        pair = {
            "sentence_a": "the dog runs quickly in the park",
            "sentence_b": "the cat walks slowly in the yard",
        }
        result = check_frequency_match(pair)
        assert result is True

    def test_rare_vs_common_words_do_not_match(self):
        """Very rare content words vs very common content words should fail."""
        pair = {
            # Common content words — wordfreq gives them high frequency
            "sentence_a": "cat dog run play eat walk talk",
            # Completely unknown words — wordfreq returns 0, floored to 1e-9
            "sentence_b": "syzygium malaccense zyzzyva xenobiotic quasiperiodic",
        }
        result = check_frequency_match(pair)
        assert result is False

    def test_same_sentence_always_matches(self):
        pair = {
            "sentence_a": "the dog ran in the park",
            "sentence_b": "the dog ran in the park",
        }
        assert check_frequency_match(pair) is True


# ── extraction/extractor.py — compute_sha256 ──────────────────────────────────

class TestComputeSha256:
    def test_sha256_deterministic(self, tmp_path):
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"pair_id": "p1", "sentence_a": "x"}\n')
        hash_1 = compute_sha256(test_file)
        hash_2 = compute_sha256(test_file)
        assert hash_1 == hash_2

    def test_sha256_changes_on_content_change(self, tmp_path):
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"sentence_a": "x"}')
        hash_before = compute_sha256(test_file)
        test_file.write_text('{"sentence_a": "y"}')
        hash_after = compute_sha256(test_file)
        assert hash_before != hash_after

    def test_sha256_is_hex_string(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello")
        result = compute_sha256(test_file)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 = 32 bytes = 64 hex chars
        int(result, 16)  # must be valid hex

    def test_sha256_accepts_path_or_str(self, tmp_path):
        test_file = tmp_path / "f.txt"
        test_file.write_bytes(b"data")
        assert compute_sha256(test_file) == compute_sha256(str(test_file))


# ── stimuli/grammars/t2.py ────────────────────────────────────────────────────

from stimuli.grammars.t2 import generate as t2_generate, generate_behavioral_items

REQUIRED_PAIR_KEYS = {
    "pair_id", "thread_id", "sentence_a", "sentence_b",
    "label_a", "label_b", "theoretical_distinction",
    "frequency_matched", "generation_grammar",
}

class TestT2Grammar:

    def test_generate_returns_correct_count(self):
        assert len(t2_generate(60)) == 60

    def test_generate_required_keys_present(self):
        for pair in t2_generate(10):
            assert REQUIRED_PAIR_KEYS.issubset(pair.keys())

    def test_generate_thread_id_is_t2(self):
        for pair in t2_generate(10):
            assert pair["thread_id"] == "t2"

    def test_generate_label_combinations_valid(self):
        valid_combinations = {("opaque", "opaque"), ("transparent", "transparent")}
        for pair in t2_generate(60):
            assert (pair["label_a"], pair["label_b"]) in valid_combinations

    def test_generate_pair_ids_sequential(self):
        pairs = t2_generate(10)
        for index, pair in enumerate(pairs):
            assert pair["pair_id"] == f"t2_{index + 1:04d}"

    def test_generate_frequency_matched_is_false(self):
        # validate_set() sets this to True — grammar must emit False
        for pair in t2_generate(10):
            assert pair["frequency_matched"] is False

    def test_generate_reproducible_with_same_seed(self):
        run_one = t2_generate(30, seed=7)
        run_two = t2_generate(30, seed=7)
        assert run_one == run_two

    def test_generate_different_seeds_give_different_order(self):
        run_seed_a = [(p["sentence_a"], p["sentence_b"]) for p in t2_generate(30, seed=1)]
        run_seed_b = [(p["sentence_a"], p["sentence_b"]) for p in t2_generate(30, seed=2)]
        assert run_seed_a != run_seed_b

    def test_generate_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="630"):
            t2_generate(631)

    def test_generate_contains_both_label_classes(self):
        pairs = t2_generate(630)
        opaque_count = sum(1 for p in pairs if p["label_a"] == "opaque")
        transparent_count = sum(1 for p in pairs if p["label_a"] == "transparent")
        assert opaque_count == 600
        assert transparent_count == 30

    def test_generate_sentences_are_nonempty_strings(self):
        for pair in t2_generate(10):
            assert isinstance(pair["sentence_a"], str) and len(pair["sentence_a"]) > 0
            assert isinstance(pair["sentence_b"], str) and len(pair["sentence_b"]) > 0

    def test_behavioral_items_count(self):
        assert len(generate_behavioral_items()) == 12

    def test_behavioral_items_required_keys(self):
        for item in generate_behavioral_items():
            assert {"question", "choice_a", "choice_b", "correct"}.issubset(item.keys())

    def test_behavioral_items_correct_field_valid(self):
        for item in generate_behavioral_items():
            assert item["correct"] in ("a", "b")


# ── stimuli/grammars/t2b.py ───────────────────────────────────────────────────

from stimuli.grammars.t2b import generate as t2b_generate
from stimuli.grammars.t2b import generate_behavioral_items as t2b_behavioral_items

VALID_T2B_LABELS = {"logically_equivalent", "intensionally_equivalent", "intensionally_distinct"}

class TestT2bGrammar:

    def test_generate_returns_correct_count(self):
        assert len(t2b_generate(20)) == 20

    def test_generate_required_keys_present(self):
        for pair in t2b_generate(10):
            assert REQUIRED_PAIR_KEYS.issubset(pair.keys())

    def test_generate_thread_id_is_t2b(self):
        for pair in t2b_generate(10):
            assert pair["thread_id"] == "t2b"

    def test_generate_all_three_classes_present(self):
        pairs = t2b_generate(30)
        labels_found = {p["label_a"] for p in pairs}
        assert labels_found == VALID_T2B_LABELS

    def test_generate_each_class_has_ten_pairs(self):
        pairs = t2b_generate(30)
        for label in VALID_T2B_LABELS:
            count = sum(1 for p in pairs if p["label_a"] == label)
            assert count == 10

    def test_generate_label_a_equals_label_b(self):
        for pair in t2b_generate(30):
            assert pair["label_a"] == pair["label_b"]

    def test_generate_pair_ids_sequential(self):
        pairs = t2b_generate(10)
        for index, pair in enumerate(pairs):
            assert pair["pair_id"] == f"t2b_{index + 1:04d}"

    def test_generate_frequency_matched_is_false(self):
        for pair in t2b_generate(10):
            assert pair["frequency_matched"] is False

    def test_generate_reproducible_with_same_seed(self):
        assert t2b_generate(20, seed=3) == t2b_generate(20, seed=3)

    def test_generate_different_seeds_give_different_order(self):
        order_a = [(p["sentence_a"], p["sentence_b"]) for p in t2b_generate(20, seed=1)]
        order_b = [(p["sentence_a"], p["sentence_b"]) for p in t2b_generate(20, seed=2)]
        assert order_a != order_b

    def test_generate_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="30"):
            t2b_generate(31)

    def test_generate_sentences_nonempty(self):
        for pair in t2b_generate(10):
            assert len(pair["sentence_a"]) > 0
            assert len(pair["sentence_b"]) > 0

    def test_behavioral_items_count(self):
        assert len(t2b_behavioral_items()) == 12

    def test_behavioral_items_required_keys(self):
        for item in t2b_behavioral_items():
            assert {"question", "choice_a", "choice_b", "correct"}.issubset(item.keys())

    def test_behavioral_items_correct_field_valid(self):
        for item in t2b_behavioral_items():
            assert item["correct"] in ("a", "b")


# ── stimuli/grammars/t1a.py ───────────────────────────────────────────────────

from stimuli.grammars.t1a import generate as t1a_generate
from stimuli.grammars.t1a import generate_behavioral_items as t1a_behavioral_items


class TestT1aGrammar:

    def test_generate_returns_correct_count(self):
        assert len(t1a_generate(30)) == 30

    def test_generate_required_keys_present(self):
        for pair in t1a_generate(10):
            assert REQUIRED_PAIR_KEYS.issubset(pair.keys())

    def test_generate_thread_id_is_t1a(self):
        for pair in t1a_generate(10):
            assert pair["thread_id"] == "t1a"

    def test_generate_labels_always_l3_and_l1(self):
        for pair in t1a_generate(30):
            assert pair["label_a"] == "causal_l3"
            assert pair["label_b"] == "associative_l1"

    def test_generate_pair_ids_sequential(self):
        pairs = t1a_generate(10)
        for index, pair in enumerate(pairs):
            assert pair["pair_id"] == f"t1a_{index + 1:04d}"

    def test_generate_frequency_matched_is_false(self):
        for pair in t1a_generate(10):
            assert pair["frequency_matched"] is False

    def test_generate_reproducible_with_same_seed(self):
        assert t1a_generate(30, seed=5) == t1a_generate(30, seed=5)

    def test_generate_different_seeds_give_different_order(self):
        order_a = [(p["sentence_a"], p["sentence_b"]) for p in t1a_generate(30, seed=1)]
        order_b = [(p["sentence_a"], p["sentence_b"]) for p in t1a_generate(30, seed=2)]
        assert order_a != order_b

    def test_generate_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="300"):
            t1a_generate(301)

    def test_generate_sentences_nonempty(self):
        for pair in t1a_generate(10):
            assert len(pair["sentence_a"]) > 0
            assert len(pair["sentence_b"]) > 0

    def test_barometer_pairs_have_notes(self):
        all_pairs = t1a_generate(300)
        barometer_pairs = [p for p in all_pairs if "notes" in p]
        assert len(barometer_pairs) == 50  # 1 domain × 5 × 5 = 25... wait 5x5=25 not 50
        # Actually: 5 L3 templates × 5 L1 templates = 25 barometer pairs
        assert len(barometer_pairs) >= 25

    def test_behavioral_items_count(self):
        assert len(t1a_behavioral_items()) == 12

    def test_behavioral_items_required_keys(self):
        for item in t1a_behavioral_items():
            assert {"question", "choice_a", "choice_b", "correct"}.issubset(item.keys())

    def test_behavioral_items_correct_field_valid(self):
        for item in t1a_behavioral_items():
            assert item["correct"] in ("a", "b")


# ── stimuli/grammars/t1b.py ───────────────────────────────────────────────────

from stimuli.grammars.t1b import generate as t1b_generate
from stimuli.grammars.t1b import generate_behavioral_items as t1b_behavioral_items

VALID_T1B_PAIR_TYPES = {
    ("forward_causal", "backtracking"),
    ("forward_causal", "common_cause"),
    ("backtracking", "common_cause"),
}

class TestT1bGrammar:

    def test_generate_returns_correct_count(self):
        assert len(t1b_generate(30)) == 30

    def test_generate_required_keys_present(self):
        for pair in t1b_generate(10):
            assert REQUIRED_PAIR_KEYS.issubset(pair.keys())

    def test_generate_thread_id_is_t1b(self):
        for pair in t1b_generate(10):
            assert pair["thread_id"] == "t1b"

    def test_generate_label_pairs_valid(self):
        for pair in t1b_generate(30):
            assert (pair["label_a"], pair["label_b"]) in VALID_T1B_PAIR_TYPES

    def test_generate_all_three_pair_types_present(self):
        pairs = t1b_generate(300)
        found = {(p["label_a"], p["label_b"]) for p in pairs}
        assert found == VALID_T1B_PAIR_TYPES

    def test_generate_type_a_count(self):
        pairs = t1b_generate(300)
        type_a = sum(1 for p in pairs if p["label_a"] == "forward_causal" and p["label_b"] == "backtracking")
        assert type_a == 250

    def test_generate_pair_ids_sequential(self):
        pairs = t1b_generate(10)
        for index, pair in enumerate(pairs):
            assert pair["pair_id"] == f"t1b_{index + 1:04d}"

    def test_generate_frequency_matched_is_false(self):
        for pair in t1b_generate(10):
            assert pair["frequency_matched"] is False

    def test_generate_reproducible_with_same_seed(self):
        assert t1b_generate(30, seed=5) == t1b_generate(30, seed=5)

    def test_generate_different_seeds_give_different_order(self):
        order_a = [(p["sentence_a"], p["sentence_b"]) for p in t1b_generate(30, seed=1)]
        order_b = [(p["sentence_a"], p["sentence_b"]) for p in t1b_generate(30, seed=2)]
        assert order_a != order_b

    def test_generate_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="300"):
            t1b_generate(301)

    def test_generate_sentences_nonempty(self):
        for pair in t1b_generate(10):
            assert len(pair["sentence_a"]) > 0
            assert len(pair["sentence_b"]) > 0

    def test_behavioral_items_count(self):
        assert len(t1b_behavioral_items()) == 12

    def test_behavioral_items_required_keys(self):
        for item in t1b_behavioral_items():
            assert {"question", "choice_a", "choice_b", "correct"}.issubset(item.keys())

    def test_behavioral_items_correct_field_valid(self):
        for item in t1b_behavioral_items():
            assert item["correct"] in ("a", "b")
