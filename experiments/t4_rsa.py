"""
experiments/t4_rsa.py — T4 RSA runner (Quinean ontology).

Wires the T4 pieces into one experiment:
  entity carriers → per-layer activation at the entity token → model RDM
  → RSA vs each theory matrix (run_rsa) → Mantel significance (run_mantel_test)
  → per-layer results + the inter-matrix discriminability artifact.

Model-agnostic: takes any TransformerLens HookedTransformer (Pythia pilot, Llama
headline). The entity's representation is the LAST-token activation of the carrier
with its trailing period stripped — i.e. the entity head sits at the final
position (carriers are "Consider {display}."). resid_post is read per layer.

V13: the caller's config must carry ontology_version + matrix_source (run_rsa
enforces it for t4). Verdict logic (winner > runner-up beyond null; nominalism/
trope reported jointly when collinear) lives in the summary step, not here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from probes.probes import run_mantel_test, run_rsa
from stimuli.grammars.t4 import to_entities
from stimuli.theoretical_matrices.t4_matrices import (
    build_all_theory_matrices,
    pairwise_theory_correlations,
)

_RESID_POST = "blocks.{layer}.hook_resid_post"


def _to_numpy(vec: Any) -> np.ndarray:
    """Accept a torch tensor or array-like; return a 1-D float numpy vector."""
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=float).reshape(-1)


def _carrier_for_extraction(carrier: str) -> str:
    """Drop the trailing period so the entity head is the final token."""
    return carrier.rstrip().rstrip(".").rstrip()


def extract_entity_activations(
    model: Any, records: list[dict[str, Any]], layers: list[int]
) -> dict[int, np.ndarray]:
    """
    For each layer, an (n_items, d_model) matrix of the entity-token activation.

    One forward pass per entity (run_with_cache over all requested layers), taking
    the last-token resid_post. Caller loads the model once and passes it in.
    """
    names = [_RESID_POST.format(layer=layer) for layer in layers]
    names_set = set(names)
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    for record in records:
        text = _carrier_for_extraction(record["carrier"])
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n in names_set
        )
        for layer in layers:
            vec = cache[_RESID_POST.format(layer=layer)][0, -1]
            per_layer[layer].append(_to_numpy(vec))

    return {layer: np.vstack(vectors) for layer, vectors in per_layer.items()}


def run_t4_rsa(
    records: list[dict[str, Any]],
    model: Any,
    config: Any,
    layers: list[int] | None = None,
    n_perms: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Full T4 RSA across layers and theories.

    Args:
        records: T4 entity records (stimuli.grammars.t4.generate()).
        model:   HookedTransformer (loaded by caller).
        config:  must have ontology_version + matrix_source (V13).
        layers:  layers to test; defaults to 1..n_layers-1 (layer 0 excluded —
                 raw embeddings, not semantic geometry).

    Returns dict:
        per_layer[layer][theory] = {spearman_r, observed_r, p_value, significant,
                                    null_95th_percentile, exceeds_null_floor}
        inter_matrix_corr: discriminability artifact (theory-pair → corr)
        layers, n_items
    """
    if layers is None:
        layers = list(range(1, model.cfg.n_layers))

    entities = to_entities(records)
    theory_matrices = build_all_theory_matrices(entities)
    acts_by_layer = extract_entity_activations(model, records, layers)

    per_layer: dict[int, dict[str, Any]] = {}
    for layer in layers:
        acts = acts_by_layer[layer]
        layer_result: dict[str, Any] = {}
        for theory, theory_matrix in theory_matrices.items():
            rsa = run_rsa(acts, theory_matrix, config)
            model_matrix = np.asarray(rsa["model_matrix"], dtype=float)
            mantel = run_mantel_test(model_matrix, theory_matrix, n_perms=n_perms, seed=seed)
            layer_result[theory] = {
                "spearman_r": rsa["spearman_r"],
                "observed_r": mantel["observed_r"],
                "p_value": mantel["p_value"],
                "significant": mantel["significant"],
                "null_95th_percentile": mantel["null_95th_percentile"],
                "exceeds_null_floor": mantel["exceeds_null_floor"],
            }
        per_layer[layer] = layer_result

    inter = pairwise_theory_correlations(theory_matrices)
    return {
        "thread_id": getattr(config, "thread_id", "t4"),
        "model_id": getattr(config, "model_id", None),
        "n_items": len(records),
        "layers": list(layers),
        "per_layer": per_layer,
        "inter_matrix_corr": {f"{a}|{b}": v for (a, b), v in inter.items()},
    }
