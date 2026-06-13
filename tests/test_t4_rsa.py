"""
tests/test_t4_rsa.py — synthetic smoke for the T4 RSA runner.

Exercises the real code path (extract → RDM → run_rsa → run_mantel → per-layer
output) on a FAKE TransformerLens-shaped model, so the wiring is V2-verified
without downloading a model. Does not assert result quality — only that the
pipeline runs and returns the right structure.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.t4_rsa import extract_entity_activations, run_t4_rsa
from stimuli.grammars.t4 import generate

D_MODEL = 32
N_LAYERS = 4


class _FakeModel:
    """Minimal HookedTransformer stand-in: deterministic per-text activations."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL)

    def to_tokens(self, text: str):
        # deterministic token count; +1 mimics the BOS transformer_lens prepends
        n = max(2, len(text.split()) + 1)
        return np.arange(n).reshape(1, n)

    def run_with_cache(self, tokens, names_filter=None):
        seq = tokens.shape[1]
        # activations depend on text length via `seq` so different entities differ,
        # but are deterministic (seeded by seq) so the test is reproducible
        cache = {}
        for layer in range(self.cfg.n_layers):
            name = f"blocks.{layer}.hook_resid_post"
            if names_filter is None or names_filter(name):
                rng = np.random.default_rng(seed=1000 * layer + seq)
                cache[name] = rng.standard_normal((1, seq, self.cfg.d_model))
        return None, cache


def _config():
    return SimpleNamespace(
        thread_id="t4",
        model_id="fake",
        ontology_version="DUL 3.27 (snapshot 2021-02-22)",
        matrix_source="stimuli/theoretical_matrices/t4_matrices.py",
    )


RECORDS = generate()


def test_extract_shapes():
    model = _FakeModel()
    layers = [1, 2, 3]
    acts = extract_entity_activations(model, RECORDS, layers)
    assert set(acts) == set(layers)
    for layer in layers:
        assert acts[layer].shape == (len(RECORDS), D_MODEL)


def test_run_t4_rsa_structure():
    model = _FakeModel()
    result = run_t4_rsa(RECORDS, model, _config(), layers=[1, 2], n_perms=50, seed=0)
    assert result["n_items"] == len(RECORDS)
    assert result["layers"] == [1, 2]
    # every layer has all four theories with the expected keys
    for layer in (1, 2):
        for theory in ("platonism", "nominalism", "trope", "fourdim"):
            entry = result["per_layer"][layer][theory]
            assert {"spearman_r", "observed_r", "p_value", "significant",
                    "null_95th_percentile", "exceeds_null_floor"} <= set(entry)
    # discriminability artifact carried through
    assert "nominalism|trope" in result["inter_matrix_corr"]


def test_default_layers_exclude_layer_zero():
    model = _FakeModel()
    result = run_t4_rsa(RECORDS, model, _config(), n_perms=20)
    assert 0 not in result["layers"]
    assert result["layers"] == [1, 2, 3]


def test_v13_enforced():
    model = _FakeModel()
    bad = _config()
    bad.ontology_version = None
    with pytest.raises(ValueError):
        run_t4_rsa(RECORDS, model, bad, layers=[1], n_perms=10)
