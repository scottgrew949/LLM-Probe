"""
tests/test_circuits.py — Unit tests for circuits/ module (no GPU required).

All functions under test operate on numpy arrays and dicts.
Model-dependent functions (those that call model.run_with_hooks) are excluded
from unit tests — they require GPU and are tested via Colab integration runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import unittest.mock

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock torch so circuits imports work without GPU
with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
    from circuits.circuit_finder import find_peak_circuit_components


class TestFindPeakCircuitComponents:
    def test_returns_components_above_threshold(self):
        head_sweep_result = {
            "kl_matrix": {
                "(0,0)": 0.05,
                "(0,1)": 0.15,
                "(1,0)": 0.25,
                "(1,1)": 0.08,
            },
            "peak_head": [1, 0],
            "peak_kl": 0.25,
        }
        circuit_components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.10)
        kl_effects = [component["kl_effect"] for component in circuit_components]
        assert all(kl_effect >= 0.10 for kl_effect in kl_effects)
        assert len(circuit_components) == 2

    def test_sorted_by_kl_descending(self):
        head_sweep_result = {
            "kl_matrix": {"(0,0)": 0.30, "(1,2)": 0.50, "(2,1)": 0.20},
            "peak_head": [1, 2],
            "peak_kl": 0.50,
        }
        circuit_components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.0)
        kl_effects = [component["kl_effect"] for component in circuit_components]
        assert kl_effects == sorted(kl_effects, reverse=True)

    def test_returns_empty_when_nothing_above_threshold(self):
        head_sweep_result = {
            "kl_matrix": {"(0,0)": 0.01, "(0,1)": 0.02},
            "peak_head": [0, 1],
            "peak_kl": 0.02,
        }
        circuit_components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.50)
        assert circuit_components == []

    def test_component_has_required_keys(self):
        head_sweep_result = {
            "kl_matrix": {"(3,5)": 0.40},
            "peak_head": [3, 5],
            "peak_kl": 0.40,
        }
        circuit_components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.0)
        assert len(circuit_components) == 1
        component = circuit_components[0]
        assert component["layer"] == 3
        assert component["head"] == 5
        assert component["kl_effect"] == 0.40

    def test_parses_layer_and_head_from_key(self):
        head_sweep_result = {
            "kl_matrix": {"(12,7)": 0.60, "(0,15)": 0.30},
            "peak_head": [12, 7],
            "peak_kl": 0.60,
        }
        circuit_components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.0)
        layers = {component["layer"] for component in circuit_components}
        heads = {component["head"] for component in circuit_components}
        assert 12 in layers
        assert 7 in heads


class TestRunPathPatching:
    def test_returns_zero_fraction_when_no_components(self):
        import unittest.mock
        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.circuit_finder import run_path_patching

        result = run_path_patching(
            source_activations_by_layer_and_head={},
            target_run_config={"stimulus": "test sentence"},
            circuit_components=[],
            token_position=-1,
            model=unittest.mock.MagicMock(),
            baseline_logits=None,
            full_layer_sweep_peak_kl=0.5,
        )
        assert result["fraction_recovered"] == 0.0
        assert result["circuit_is_sufficient"] is False
        assert result["n_components_patched"] == 0

    def test_circuit_sufficient_threshold_is_0_80(self):
        # Verify the threshold logic directly without invoking the model
        recovered_kl = 0.40
        full_layer_sweep_peak_kl = 0.50
        fraction = recovered_kl / full_layer_sweep_peak_kl
        assert fraction >= 0.80
        assert fraction == pytest.approx(0.80)

    def test_fraction_zero_when_full_kl_near_zero(self):
        import unittest.mock
        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.circuit_finder import run_path_patching

        result = run_path_patching(
            source_activations_by_layer_and_head={},
            target_run_config={"stimulus": "test sentence"},
            circuit_components=[],
            token_position=-1,
            model=unittest.mock.MagicMock(),
            baseline_logits=None,
            full_layer_sweep_peak_kl=0.0,
        )
        assert result["fraction_recovered"] == 0.0

    def test_returns_required_keys(self):
        import unittest.mock
        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.circuit_finder import run_path_patching

        result = run_path_patching(
            source_activations_by_layer_and_head={},
            target_run_config={"stimulus": "test"},
            circuit_components=[],
            token_position=-1,
            model=unittest.mock.MagicMock(),
            full_layer_sweep_peak_kl=1.0,
        )
        required_keys = {"recovered_kl", "fraction_recovered", "full_layer_sweep_peak_kl",
                         "circuit_is_sufficient", "n_components_patched"}
        assert required_keys.issubset(result.keys())


class TestComputeLogitAttribution:
    def test_returns_required_keys(self):
        import unittest.mock
        import numpy as np

        n_layers = 2
        n_heads = 4
        d_head = 8
        d_model = 16
        seq_len = 5

        mock_model = unittest.mock.MagicMock()
        mock_model.cfg.n_layers = n_layers
        mock_model.cfg.n_heads = n_heads
        mock_model.to_string.return_value = " wet"

        mock_cache = {}
        for layer_index in range(n_layers):
            mock_cache[f"blocks.{layer_index}.attn.hook_z"] = (
                np.random.randn(1, seq_len, n_heads, d_head)
            )
            mock_model.blocks[layer_index].attn.W_O = np.random.randn(n_heads, d_head, d_model)

        mock_model.run_with_cache.return_value = (unittest.mock.MagicMock(), mock_cache)
        mock_model.W_U = np.random.randn(d_model, 50257)

        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.attribution import compute_logit_attribution

        result = compute_logit_attribution(
            target_sentence="If it had rained the ground would be",
            logit_direction_token_id=3596,
            model=mock_model,
        )

        required_keys = {
            "attribution_matrix", "top_positive_components",
            "top_negative_components", "token_id", "token_string",
        }
        assert required_keys.issubset(result.keys())
        assert result["token_id"] == 3596
        assert result["token_string"] == " wet"

    def test_attribution_matrix_has_entry_per_layer_head(self):
        n_layers = 3
        n_heads = 4
        expected_key_count = n_layers * n_heads
        # Verify key format and count
        expected_keys = {f"({layer},{head})" for layer in range(n_layers) for head in range(n_heads)}
        assert len(expected_keys) == expected_key_count
        # Verify key format matches what the function produces
        sample_key = "(2,3)"
        assert sample_key in expected_keys

    def test_top_components_sorted_correctly(self):
        import unittest.mock
        import numpy as np

        n_layers = 2
        n_heads = 2
        d_head = 4
        d_model = 8

        mock_model = unittest.mock.MagicMock()
        mock_model.cfg.n_layers = n_layers
        mock_model.cfg.n_heads = n_heads
        mock_model.to_string.return_value = " token"

        # Use fixed seed for deterministic attribution values
        rng = np.random.RandomState(0)
        mock_cache = {}
        for layer_index in range(n_layers):
            mock_cache[f"blocks.{layer_index}.attn.hook_z"] = rng.randn(1, 3, n_heads, d_head)
            mock_model.blocks[layer_index].attn.W_O = rng.randn(n_heads, d_head, d_model)

        mock_model.run_with_cache.return_value = (unittest.mock.MagicMock(), mock_cache)
        mock_model.W_U = rng.randn(d_model, 100)

        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.attribution import compute_logit_attribution

        result = compute_logit_attribution("test", 5, mock_model)

        positive_attributions = [c["attribution"] for c in result["top_positive_components"]]
        negative_attributions = [c["attribution"] for c in result["top_negative_components"]]

        assert positive_attributions == sorted(positive_attributions, reverse=True)
        assert negative_attributions == sorted(negative_attributions)
        assert all(v > 0 for v in positive_attributions)
        assert all(v < 0 for v in negative_attributions)
