"""
circuits/attribution.py — Direct logit attribution per attention head.

─── CONCEPT: Direct logit attribution ────────────────────────────────────────
The residual stream at the final token position is the sum of contributions
from every attention head and MLP layer. The unembedding matrix W_U projects
this sum into logit space to produce next-token probabilities.

Direct logit attribution decomposes the logit for a specific token into
additive per-head contributions:

    attr(layer, head, token) = head_output(layer, head) · W_U[:, token]

where head_output(layer, head) = z[layer, head] @ W_O[layer, head]

z[layer, head]: per-head value output from hook_z, shape (d_head,)
W_O[layer, head]: output projection matrix, shape (d_head, d_model)

Positive attribution: head pushes probability toward the target token.
Negative attribution: head pushes probability away from the target token.

Note: this is an approximation — it omits layer norm scaling before W_U.
For publication-quality results, apply final layer norm via
TransformerLens cache.apply_ln_to_stack before projecting.

─── MODEL LOADING CONVENTION ─────────────────────────────────────────────────
Accept model as a required parameter. Load once, pass through.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch


def _to_numpy(tensor_or_array: Any) -> np.ndarray:
    """Convert a torch.Tensor or numpy array to a numpy array."""
    if hasattr(tensor_or_array, "detach"):
        return tensor_or_array.detach().cpu().numpy()
    return np.array(tensor_or_array)


def compute_logit_attribution(
    target_sentence: str,
    logit_direction_token_id: int,
    model: Any,
    token_position: int = -1,
) -> dict[str, Any]:
    """
    Decompose the logit for logit_direction_token_id into per-head contributions.

    For each (layer, head), computes how much that head pushes model probability
    toward or away from logit_direction_token_id at token_position.

    Args:
        target_sentence: input string to run through the model.
        logit_direction_token_id: vocab index of the token whose logit to decompose.
            For T1b: use the token id of the forward-causal completion word.
        model: HookedTransformer instance.
        token_position: sequence position to analyze. Default -1 (last token).

    Returns:
        {
          "attribution_matrix": dict[str, float] — key "(layer,head)" -> attribution,
          "top_positive_components": list[{layer, head, attribution}] — top 5 positive,
          "top_negative_components": list[{layer, head, attribution}] — top 5 negative,
          "token_id": int,
          "token_string": str — string representation of the target token,
        }
    """
    _, activation_cache = model.run_with_cache(target_sentence)

    unembedding_matrix = model.W_U  # (d_model, vocab_size)

    logit_direction_vector = _to_numpy(unembedding_matrix[:, logit_direction_token_id])

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attribution_matrix: dict[str, float] = {}

    for layer_index in range(n_layers):
        hook_z_key = f"blocks.{layer_index}.attn.hook_z"
        per_head_value_outputs = activation_cache[hook_z_key]  # (batch, seq, n_heads, d_head)

        output_projection = model.blocks[layer_index].attn.W_O  # (n_heads, d_head, d_model)

        for head_index in range(n_heads):
            head_z = _to_numpy(per_head_value_outputs[0, token_position, head_index, :])
            head_output_projection = _to_numpy(output_projection[head_index])

            # head_output_vector: (d_model,) = (d_head,) @ (d_head, d_model)
            head_output_vector = head_z @ head_output_projection

            attribution_value = float(np.dot(head_output_vector, logit_direction_vector))

            matrix_key = f"({layer_index},{head_index})"
            attribution_matrix[matrix_key] = attribution_value

    all_components = []
    for key_string, attribution_value in attribution_matrix.items():
        inner_content = key_string.strip("()")
        layer_string, head_string = inner_content.split(",")
        all_components.append({
            "layer": int(layer_string),
            "head": int(head_string),
            "attribution": attribution_value,
        })

    top_positive_components = sorted(
        [component for component in all_components if component["attribution"] > 0],
        key=lambda component: component["attribution"],
        reverse=True,
    )[:5]

    top_negative_components = sorted(
        [component for component in all_components if component["attribution"] < 0],
        key=lambda component: component["attribution"],
    )[:5]

    return {
        "attribution_matrix": attribution_matrix,
        "top_positive_components": top_positive_components,
        "top_negative_components": top_negative_components,
        "token_id": logit_direction_token_id,
        "token_string": model.to_string(logit_direction_token_id),
    }
