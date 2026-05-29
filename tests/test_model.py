"""
tests/test_model.py — Integration tests requiring a real GPU and GPT-2 medium.

─── CONCEPT: Why these tests exist ───────────────────────────────────────────
test_phase0.py covers all pure functions (no model needed). This file covers
everything that requires a real HookedTransformer: activation extraction,
behavioral gating, patching, layer sweeps. These tests only run on Colab T4 —
the pytestmark at the top skips them everywhere else.

─── RUN ON COLAB ─────────────────────────────────────────────────────────────
    !git clone https://github.com/scottgrew949/LLM-Probe
    %cd LLM-Probe
    !pip install transformer_lens wordfreq jsonschema
    !python -m pytest tests/test_model.py -v

─── DESIGN DECISIONS ─────────────────────────────────────────────────────────
- Session-scoped model: GPT-2 medium loads once (~30s), shared across all tests.
- Session-scoped extraction: extract_activations runs once (~60s for 4 layers),
  all tests that need activations share the result.
- Validated/ path: temp stimulus file written under a path containing "validated"
  as a directory component, satisfying extractor._assert_validated_path (V12).
- Surface null redirect: monkeypatch PROJECT_ROOT in run.py so surface_null.json
  lands in tmp_path, not the real project tree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# ── GPU guard ─────────────────────────────────────────────────────────────────
# All tests in this file skip when CUDA is unavailable (Intel Mac, no GPU).
# On Colab T4: torch.cuda.is_available() == True → tests run.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA — run on Colab T4, not locally",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer
from experiments.config import ExperimentConfig
from extraction.extractor import extract_activations, compute_sha256
from stimuli.pipeline import run_behavioral_gate
from interventions.interventions import (
    patch_activation,
    mean_ablate,
    run_layer_sweep,
    assert_specificity_valid,
)
from experiments.run import run_surface_null


# ── Constants ─────────────────────────────────────────────────────────────────

GPT2_MEDIUM_HIDDEN_DIM = 1024
GPT2_MEDIUM_VOCAB_SIZE = 50257

# 8 pairs: 4 opaque (belief reports), 4 transparent.
# Balanced 8 opaque + 8 transparent sentences (16 total) for StratifiedKFold(n_splits=5).
# Hesperus/Phosphorus and Cicero/Tully are the canonical T2 coreferential pairs.
STIMULUS_PAIRS: list[dict[str, Any]] = [
    {
        "pair_id": "t2_0001", "thread_id": "t2",
        "sentence_a": "Alice believes Hesperus is visible tonight.",
        "sentence_b": "Alice believes Phosphorus is visible tonight.",
        "label_a": "opaque", "label_b": "opaque",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
    {
        "pair_id": "t2_0002", "thread_id": "t2",
        "sentence_a": "John thinks Hesperus is a planet.",
        "sentence_b": "John thinks Phosphorus is a planet.",
        "label_a": "opaque", "label_b": "opaque",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
    {
        "pair_id": "t2_0003", "thread_id": "t2",
        "sentence_a": "Hesperus is visible at dusk.",
        "sentence_b": "Phosphorus is visible at dusk.",
        "label_a": "transparent", "label_b": "transparent",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
    {
        "pair_id": "t2_0004", "thread_id": "t2",
        "sentence_a": "Alice believes Cicero was an orator.",
        "sentence_b": "Alice believes Tully was an orator.",
        "label_a": "opaque", "label_b": "opaque",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
    {
        "pair_id": "t2_0005", "thread_id": "t2",
        "sentence_a": "Cicero was an orator.",
        "sentence_b": "Tully was an orator.",
        "label_a": "transparent", "label_b": "transparent",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
    {
        "pair_id": "t2_0006", "thread_id": "t2",
        "sentence_a": "Mary knows Hesperus rises at dawn.",
        "sentence_b": "Mary knows Phosphorus rises at dawn.",
        "label_a": "opaque", "label_b": "opaque",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
    {
        "pair_id": "t2_0007", "thread_id": "t2",
        "sentence_a": "Hesperus orbits the sun.",
        "sentence_b": "Phosphorus orbits the sun.",
        "label_a": "transparent", "label_b": "transparent",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
    {
        "pair_id": "t2_0008", "thread_id": "t2",
        "sentence_a": "Hesperus was named by ancient Greeks.",
        "sentence_b": "Phosphorus was named by ancient Greeks.",
        "label_a": "transparent", "label_b": "transparent",
        "theoretical_distinction": "Frege opaque vs transparent context",
        "frequency_matched": True,
    },
]

# High-confidence factual items. GPT-2 medium reliably prefers the correct choice
# for factual geography and common collocations — safe to assert gate passes.
BEHAVIORAL_ITEMS: list[dict[str, Any]] = [
    {
        "question": "The capital of France is",
        "choice_a": "Paris",
        "choice_b": "Berlin",
        "correct": "a",
    },
    {
        "question": "Water is composed of hydrogen and",
        "choice_a": "oxygen",
        "choice_b": "nitrogen",
        "correct": "a",
    },
    {
        "question": "The sun rises in the",
        "choice_a": "east",
        "choice_b": "ceiling",
        "correct": "a",
    },
    {
        "question": "Dogs are commonly known as",
        "choice_a": "pets",
        "choice_b": "furniture",
        "correct": "a",
    },
]


# ── Session-scoped fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def gpt2_medium_model() -> HookedTransformer:
    """Load GPT-2 medium once for the full session. ~30s on T4."""
    model = HookedTransformer.from_pretrained("gpt2-medium")
    model.eval()
    return model


@pytest.fixture(scope="session")
def validated_stimulus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Write STIMULUS_PAIRS to a JSONL file whose path contains 'validated' as a
    directory component. extractor._assert_validated_path checks for this.
    tmp_path_factory (not tmp_path) is required for session-scoped fixtures.
    """
    base_directory = tmp_path_factory.mktemp("root")
    validated_directory = base_directory / "validated" / "t2"
    validated_directory.mkdir(parents=True)
    stimulus_file_path = validated_directory / "pairs.validated.jsonl"
    with stimulus_file_path.open("w") as file_handle:
        for pair in STIMULUS_PAIRS:
            file_handle.write(json.dumps(pair) + "\n")
    return stimulus_file_path


@pytest.fixture(scope="session")
def stimulus_sha256_hex(validated_stimulus_path: Path) -> str:
    """SHA256 of the validated stimulus file. Must match config.stimulus_sha256."""
    return compute_sha256(validated_stimulus_path)


@pytest.fixture(scope="session")
def locked_config(
    validated_stimulus_path: Path,
    stimulus_sha256_hex: str,
) -> ExperimentConfig:
    """
    Construct and lock the canonical test config.
    layer_range=(0, 3) keeps extraction to 4 layers instead of 24.
    """
    config = ExperimentConfig(
        experiment_id="test_t2_gpt2m_model",
        thread_id="t2",
        model_id="gpt2-medium",
        model_revision="main",
        layer_range=(0, 3),
        component="resid_post",
        token_positions=[-1],
        probe_type="linear",
        stimulus_file=str(validated_stimulus_path),
        stimulus_sha256=stimulus_sha256_hex,
        frequency_match_verified=True,
        expected_outcomes={"probe_accuracy": ">0.5"},
    )
    config.lock()
    return config


@pytest.fixture(scope="session")
def activation_results(
    locked_config: ExperimentConfig,
    gpt2_medium_model: HookedTransformer,
) -> list[dict[str, Any]]:
    """
    Run extract_activations once for the session. ~60s on T4 for 4 layers × 16 sentences.
    All tests that need activations share this result — do not call extract_activations again.
    """
    return extract_activations(locked_config, gpt2_medium_model)


@pytest.fixture(scope="session")
def baseline_logits_fixture(gpt2_medium_model: HookedTransformer) -> list[float]:
    """Single forward pass to get unpatched baseline logits for patching tests."""
    target_sentence = "Alice believes Phosphorus is visible."
    with torch.no_grad():
        logits = gpt2_medium_model(target_sentence)
    return logits[0, -1, :].tolist()


# ── TestExtractActivations ────────────────────────────────────────────────────

class TestExtractActivations:

    def test_returns_list_of_dicts(self, activation_results):
        assert isinstance(activation_results, list)
        assert all(isinstance(d, dict) for d in activation_results)

    def test_result_count_matches_layer_range(self, activation_results, locked_config):
        start, end = locked_config.layer_range
        expected_count = (end - start + 1) * len(locked_config.token_positions)
        assert len(activation_results) == expected_count

    def test_required_keys_present(self, activation_results):
        required_keys = {
            "model_id", "layer", "component",
            "token_position", "activations", "labels", "pair_ids",
        }
        for activation_dict in activation_results:
            assert required_keys.issubset(activation_dict.keys())

    def test_activations_shape(self, activation_results):
        # 8 pairs × 2 sentences each = 16 items; GPT-2 medium hidden_dim = 1024
        expected_n_items = len(STIMULUS_PAIRS) * 2
        for activation_dict in activation_results:
            assert len(activation_dict["activations"]) == expected_n_items
            assert len(activation_dict["activations"][0]) == GPT2_MEDIUM_HIDDEN_DIM

    def test_labels_and_pair_ids_length(self, activation_results):
        expected_n_items = len(STIMULUS_PAIRS) * 2
        for activation_dict in activation_results:
            assert len(activation_dict["labels"]) == expected_n_items
            assert len(activation_dict["pair_ids"]) == expected_n_items

    def test_layer_indices_cover_full_range(self, activation_results, locked_config):
        extracted_layers = {d["layer"] for d in activation_results}
        start, end = locked_config.layer_range
        assert extracted_layers == set(range(start, end + 1))

    def test_pair_ids_have_a_and_b_suffixes(self, activation_results):
        first_pair_ids = activation_results[0]["pair_ids"]
        assert first_pair_ids[0].endswith("_a")
        assert first_pair_ids[1].endswith("_b")

    def test_activations_are_floats(self, activation_results):
        assert isinstance(activation_results[0]["activations"][0][0], float)


# ── TestExtractActivationsV12 ─────────────────────────────────────────────────

class TestExtractActivationsV12:

    def test_raises_if_stimulus_not_under_validated(
        self, tmp_path, gpt2_medium_model
    ):
        """V12: extractor must reject any stimulus file not under a 'validated/' directory."""
        non_validated_directory = tmp_path / "generated" / "t2"
        non_validated_directory.mkdir(parents=True)
        non_validated_file = non_validated_directory / "pairs.jsonl"
        with non_validated_file.open("w") as file_handle:
            for pair in STIMULUS_PAIRS[:2]:
                file_handle.write(json.dumps(pair) + "\n")

        file_sha256 = compute_sha256(non_validated_file)
        config = ExperimentConfig(
            experiment_id="test_v12_path",
            thread_id="t2",
            model_id="gpt2-medium",
            model_revision="main",
            layer_range=(0, 0),
            component="resid_post",
            token_positions=[-1],
            probe_type="linear",
            stimulus_file=str(non_validated_file),
            stimulus_sha256=file_sha256,
            frequency_match_verified=True,
            expected_outcomes={"probe_accuracy": ">0.5"},
        )
        config.lock()

        with pytest.raises(ValueError, match="validated"):
            extract_activations(config, gpt2_medium_model)

    def test_raises_if_sha256_mismatch(
        self, validated_stimulus_path, gpt2_medium_model
    ):
        """V12: extractor must reject if SHA256 doesn't match the locked value."""
        wrong_sha256 = "deadbeef" * 8  # 64 hex chars, wrong hash

        config = ExperimentConfig(
            experiment_id="test_v12_sha256",
            thread_id="t2",
            model_id="gpt2-medium",
            model_revision="main",
            layer_range=(0, 0),
            component="resid_post",
            token_positions=[-1],
            probe_type="linear",
            stimulus_file=str(validated_stimulus_path),
            stimulus_sha256=wrong_sha256,
            frequency_match_verified=True,
            expected_outcomes={"probe_accuracy": ">0.5"},
        )
        # lock() only checks sha256 is non-empty, not that it's correct —
        # the mismatch is caught by the extractor at runtime.
        config.lock()

        with pytest.raises(ValueError, match="SHA256"):
            extract_activations(config, gpt2_medium_model)


# ── TestRunBehavioralGate ─────────────────────────────────────────────────────

class TestRunBehavioralGate:

    @pytest.fixture(scope="class")
    def gate_result(self, gpt2_medium_model):
        return run_behavioral_gate(BEHAVIORAL_ITEMS, gpt2_medium_model, threshold=0.70)

    def test_returns_expected_keys(self, gate_result):
        assert {"passed", "accuracy", "n_items", "details"}.issubset(gate_result.keys())

    def test_accuracy_is_float_in_range(self, gate_result):
        assert isinstance(gate_result["accuracy"], float)
        assert 0.0 <= gate_result["accuracy"] <= 1.0

    def test_n_items_matches_input_count(self, gate_result):
        assert gate_result["n_items"] == len(BEHAVIORAL_ITEMS)

    def test_details_has_correct_structure(self, gate_result):
        required_detail_keys = {
            "question", "model_choice", "correct_choice",
            "is_correct", "log_prob_choice_a", "log_prob_choice_b",
        }
        for detail in gate_result["details"]:
            assert required_detail_keys.issubset(detail.keys())
            assert detail["model_choice"] in ("a", "b")

    def test_passes_high_confidence_items(self, gate_result):
        # GPT-2 medium reliably gets factual geography and common collocations correct.
        assert gate_result["passed"] is True

    def test_threshold_below_floor_raises(self, gpt2_medium_model):
        with pytest.raises(ValueError, match="V8"):
            run_behavioral_gate(BEHAVIORAL_ITEMS, gpt2_medium_model, threshold=0.60)

    def test_log_probs_are_non_positive(self, gate_result):
        # Log-probabilities are always ≤ 0
        for detail in gate_result["details"]:
            assert detail["log_prob_choice_a"] <= 0.0
            assert detail["log_prob_choice_b"] <= 0.0


# ── TestPatchActivation ───────────────────────────────────────────────────────

class TestPatchActivation:

    @pytest.fixture(scope="class")
    def patch_result_with_baseline(
        self, activation_results, gpt2_medium_model, baseline_logits_fixture
    ):
        source_activation = np.array(activation_results[0]["activations"][0])
        return patch_activation(
            source_activation,
            {"stimulus": "Alice believes Phosphorus is visible."},
            layer=0,
            component="resid_post",
            token_position=-1,
            model=gpt2_medium_model,
            baseline_logits=baseline_logits_fixture,
        )

    def test_returns_expected_keys(self, patch_result_with_baseline):
        required_keys = {
            "logits", "top_tokens", "top_probs",
            "patched_layer", "patched_component", "patched_token_position",
            "kl_from_baseline",
        }
        assert required_keys.issubset(patch_result_with_baseline.keys())

    def test_logits_has_correct_size(self, patch_result_with_baseline):
        assert len(patch_result_with_baseline["logits"]) == GPT2_MEDIUM_VOCAB_SIZE
        assert isinstance(patch_result_with_baseline["logits"][0], float)

    def test_top_tokens_is_five_strings(self, patch_result_with_baseline):
        assert len(patch_result_with_baseline["top_tokens"]) == 5
        assert all(isinstance(t, str) for t in patch_result_with_baseline["top_tokens"])

    def test_top_probs_are_valid(self, patch_result_with_baseline):
        assert len(patch_result_with_baseline["top_probs"]) == 5
        assert all(p >= 0.0 for p in patch_result_with_baseline["top_probs"])
        # Sum of top-5 probs is ≤ 1.0 (rest of vocab takes the remainder)
        assert sum(patch_result_with_baseline["top_probs"]) <= 1.0 + 1e-6

    def test_kl_from_baseline_is_float_when_provided(self, patch_result_with_baseline):
        assert isinstance(patch_result_with_baseline["kl_from_baseline"], float)

    def test_kl_is_none_without_baseline(self, activation_results, gpt2_medium_model):
        source_activation = np.array(activation_results[0]["activations"][0])
        result = patch_activation(
            source_activation,
            {"stimulus": "Alice believes Phosphorus is visible."},
            layer=0,
            component="resid_post",
            token_position=-1,
            model=gpt2_medium_model,
        )
        assert result["kl_from_baseline"] is None

    def test_patched_layer_and_component_echoed(self, patch_result_with_baseline):
        assert patch_result_with_baseline["patched_layer"] == 0
        assert patch_result_with_baseline["patched_component"] == "resid_post"


# ── TestMeanAblate ────────────────────────────────────────────────────────────

class TestMeanAblate:

    @pytest.fixture(scope="class")
    def mean_ablation_result(
        self, activation_results, gpt2_medium_model, baseline_logits_fixture
    ):
        all_activations = np.array(activation_results[0]["activations"])
        return mean_ablate(
            all_activations,
            {"stimulus": "Alice believes Phosphorus is visible."},
            layer=0,
            component="resid_post",
            token_position=-1,
            model=gpt2_medium_model,
            baseline_logits=baseline_logits_fixture,
        )

    def test_returns_expected_keys(self, mean_ablation_result):
        required_keys = {
            "logits", "top_tokens", "top_probs",
            "patched_layer", "patched_component", "patched_token_position",
            "kl_from_baseline",
        }
        assert required_keys.issubset(mean_ablation_result.keys())

    def test_kl_from_baseline_is_float(self, mean_ablation_result):
        assert isinstance(mean_ablation_result["kl_from_baseline"], float)

    def test_result_differs_from_specific_patch(
        self, mean_ablation_result, activation_results, gpt2_medium_model, baseline_logits_fixture
    ):
        # Mean activation (information-free) and the first item's activation should
        # produce different outputs — they should not be identical vectors.
        source_activation = np.array(activation_results[0]["activations"][0])
        specific_patch_result = patch_activation(
            source_activation,
            {"stimulus": "Alice believes Phosphorus is visible."},
            layer=0,
            component="resid_post",
            token_position=-1,
            model=gpt2_medium_model,
            baseline_logits=baseline_logits_fixture,
        )
        assert (
            mean_ablation_result["top_tokens"] != specific_patch_result["top_tokens"]
            or mean_ablation_result["top_probs"] != specific_patch_result["top_probs"]
        )


# ── TestRunLayerSweep ─────────────────────────────────────────────────────────

class TestRunLayerSweep:

    @pytest.fixture(scope="class")
    def sweep_result(self, activation_results, locked_config, gpt2_medium_model):
        source_activation_by_layer = {
            activation_dict["layer"]: np.array(activation_dict["activations"][0])
            for activation_dict in activation_results
        }
        return run_layer_sweep(
            source_activation_by_layer,
            {"stimulus": "Phosphorus is visible at dusk."},
            layer_range=locked_config.layer_range,
            component="resid_post",
            token_position=-1,
            model=gpt2_medium_model,
        )

    def test_returns_expected_keys(self, sweep_result):
        assert {"layer_effects", "peak_layer", "component", "token_position"}.issubset(
            sweep_result.keys()
        )

    def test_layer_effects_covers_full_range(self, sweep_result, locked_config):
        start, end = locked_config.layer_range
        assert set(sweep_result["layer_effects"].keys()) == set(range(start, end + 1))

    def test_layer_effects_are_floats(self, sweep_result):
        assert all(isinstance(v, float) for v in sweep_result["layer_effects"].values())

    def test_peak_layer_is_in_range(self, sweep_result, locked_config):
        start, end = locked_config.layer_range
        assert start <= sweep_result["peak_layer"] <= end

    def test_peak_layer_has_max_effect(self, sweep_result):
        peak = sweep_result["peak_layer"]
        assert sweep_result["layer_effects"][peak] == max(sweep_result["layer_effects"].values())

    def test_component_echoed_in_result(self, sweep_result):
        assert sweep_result["component"] == "resid_post"


# ── TestAssertSpecificityValid ────────────────────────────────────────────────

class TestAssertSpecificityValid:
    """Pure numerical logic — no model. Included here since it lives in interventions.py."""

    def test_passes_when_ratio_exceeds_threshold(self):
        assert_specificity_valid(3.0, 1.0, layer=5)

    def test_raises_when_ratio_below_threshold(self):
        with pytest.raises(ValueError, match="V2"):
            assert_specificity_valid(1.2, 1.0, layer=5)

    def test_passes_when_mean_ablation_near_zero(self):
        # mean_ablation_kl < 1e-8 → pass through, ratio not computed
        assert_specificity_valid(0.5, 1e-10, layer=5)

    def test_raises_at_boundary(self):
        with pytest.raises(ValueError, match="V2"):
            assert_specificity_valid(1.49, 1.0, layer=3)

    def test_custom_min_ratio_passes(self):
        assert_specificity_valid(2.0, 1.0, layer=0, min_ratio=1.8)

    def test_custom_min_ratio_raises(self):
        with pytest.raises(ValueError, match="V2"):
            assert_specificity_valid(1.7, 1.0, layer=0, min_ratio=1.8)


# ── TestRunSurfaceNull ────────────────────────────────────────────────────────

class TestRunSurfaceNull:
    """
    run_surface_null does not use a model but writes to PROJECT_ROOT / experiments/.
    Monkeypatching PROJECT_ROOT in run.py redirects the write to tmp_path.
    """

    @pytest.fixture
    def surface_null_config(self, validated_stimulus_path, stimulus_sha256_hex):
        # thread_id="t_test_model" avoids collision with real T2 result directories.
        config = ExperimentConfig(
            experiment_id="test_surface_null",
            thread_id="t_test_model",
            model_id="gpt2-medium",
            model_revision="main",
            layer_range=(0, 3),
            component="resid_post",
            token_positions=[-1],
            probe_type="linear",
            stimulus_file=str(validated_stimulus_path),
            stimulus_sha256=stimulus_sha256_hex,
            frequency_match_verified=True,
            expected_outcomes={"probe_accuracy": ">0.5"},
        )
        config.lock()
        return config

    @pytest.fixture
    def surface_null_outdir(self, tmp_path, monkeypatch):
        """Redirect run.PROJECT_ROOT so the output lands in tmp_path."""
        import experiments.run as run_module
        monkeypatch.setattr(run_module, "PROJECT_ROOT", tmp_path)
        return tmp_path

    def test_returns_expected_keys(self, surface_null_config, surface_null_outdir):
        result = run_surface_null(surface_null_config)
        required_keys = {
            "surface_classifier_accuracy", "mean_freq_diff",
            "mean_length_diff_tokens", "mean_vocab_overlap", "n_pairs",
        }
        assert required_keys.issubset(result.keys())

    def test_n_pairs_matches_stimulus_set(self, surface_null_config, surface_null_outdir):
        result = run_surface_null(surface_null_config)
        assert result["n_pairs"] == len(STIMULUS_PAIRS)

    def test_accuracy_is_float_in_range(self, surface_null_config, surface_null_outdir):
        result = run_surface_null(surface_null_config)
        assert isinstance(result["surface_classifier_accuracy"], float)
        assert 0.0 <= result["surface_classifier_accuracy"] <= 1.0

    def test_writes_surface_null_json(self, surface_null_config, surface_null_outdir):
        run_surface_null(surface_null_config)
        expected_path = (
            surface_null_outdir
            / "experiments"
            / surface_null_config.thread_id
            / "results"
            / "surface_null.json"
        )
        assert expected_path.exists()

    def test_written_json_matches_returned_dict(self, surface_null_config, surface_null_outdir):
        result = run_surface_null(surface_null_config)
        written_path = (
            surface_null_outdir
            / "experiments"
            / surface_null_config.thread_id
            / "results"
            / "surface_null.json"
        )
        with written_path.open("r") as file_handle:
            written_data = json.load(file_handle)
        assert written_data == result
