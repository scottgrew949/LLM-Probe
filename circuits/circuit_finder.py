"""
circuits/circuit_finder.py — Head-level and path-level causal analysis.

─── CONCEPT: From layers to circuits ─────────────────────────────────────────
Layer-resolved patching (L3 in interventions/interventions.py) identifies
WHICH LAYER causally encodes a distinction. Circuit analysis goes one level
deeper: within that layer, WHICH ATTENTION HEADS drive the effect?

The key tool is hook_z — the per-head value output in TransformerLens.
Shape: (batch, seq, n_heads, d_head). Patching a specific head means replacing
[:, token_position, head_index, :] with the source activation for that head.

run_head_sweep patches every (layer, head) independently, building a KL matrix.
find_peak_circuit_components filters that matrix to the meaningful components.

─── MODEL LOADING CONVENTION ─────────────────────────────────────────────────
All functions accept `model` as a required parameter. Load once, pass through.
Never load the model inside these functions.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch


def run_head_sweep(
    source_activations_by_layer_and_head: dict[tuple[int, int], np.ndarray],
    target_run_config: dict[str, Any],
    layer_range: tuple[int, int],
    n_heads: int,
    token_position: int,
    model: Any,
    baseline_logits: list[float] | None = None,
) -> dict[str, Any]:
    """
    Patch each attention head independently via hook_z. Returns KL matrix.

    For each (layer, head) in layer_range x range(n_heads), replaces the
    hook_z activation at token_position for that head with the corresponding
    source activation, then measures KL(patched || baseline).

    Args:
        source_activations_by_layer_and_head: dict keyed by (layer, head_index)
            -> np.ndarray of shape (d_head,). Typically mean over source condition.
            Only (layer, head) pairs present in this dict are swept.
        target_run_config: dict with 'stimulus' key (target sentence string).
        layer_range: (start_layer, end_layer) inclusive.
        n_heads: number of attention heads in the model (e.g. 16 for GPT-2 medium).
        token_position: which token position to patch (e.g. -1 for last token).
        model: HookedTransformer instance. Caller loads once and passes in.
        baseline_logits: if provided, KL(patched || baseline) is computed per head.

    Returns:
        {
          "kl_matrix": dict[str, float] — key "(layer,head)" -> KL value,
          "peak_head": [int, int] — [layer, head] with highest KL,
          "peak_kl": float,
          "n_layers_swept": int,
          "n_heads": int,
        }
    """
    model_device = next(model.parameters()).device
    target_sentence = target_run_config["stimulus"]

    kl_matrix: dict[str, float] = {}
    peak_kl = 0.0
    peak_head = [0, 0]

    start_layer, end_layer = layer_range

    for layer_index in range(start_layer, end_layer + 1):
        for head_index in range(n_heads):
            if (layer_index, head_index) not in source_activations_by_layer_and_head:
                continue

            source_head_activation = source_activations_by_layer_and_head[(layer_index, head_index)]
            source_tensor = torch.tensor(
                source_head_activation, dtype=torch.float32
            ).to(model_device)

            hook_name = f"blocks.{layer_index}.attn.hook_z"

            def make_head_replacement_hook(replacement_tensor, target_head_index, target_token_position):
                def replace_head_at_position(activation_value, hook):
                    activation_value[:, target_token_position, target_head_index, :] = replacement_tensor
                    return activation_value
                return replace_head_at_position

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    target_sentence,
                    fwd_hooks=[(hook_name, make_head_replacement_hook(
                        source_tensor, head_index, token_position
                    ))],
                )

            final_token_logits = patched_logits[0, -1, :]

            kl_value = 0.0
            if baseline_logits is not None:
                baseline_log_probs = torch.log_softmax(
                    torch.tensor(baseline_logits, dtype=torch.float32).to(model_device),
                    dim=-1,
                )
                patched_log_probs = torch.log_softmax(final_token_logits, dim=-1)
                kl_value = torch.nn.functional.kl_div(
                    baseline_log_probs,
                    patched_log_probs,
                    reduction="sum",
                    log_target=True,
                ).item()

            matrix_key = f"({layer_index},{head_index})"
            kl_matrix[matrix_key] = kl_value

            if kl_value > peak_kl:
                peak_kl = kl_value
                peak_head = [layer_index, head_index]

    return {
        "kl_matrix": kl_matrix,
        "peak_head": peak_head,
        "peak_kl": peak_kl,
        "n_layers_swept": end_layer - start_layer + 1,
        "n_heads": n_heads,
    }


def find_peak_circuit_components(
    head_sweep_result: dict[str, Any],
    kl_threshold: float,
) -> list[dict[str, Any]]:
    """
    Filter head sweep KL matrix to components above kl_threshold.

    Args:
        head_sweep_result: return value of run_head_sweep.
        kl_threshold: minimum KL to count as a circuit component.
            Use config.circuit_kl_threshold (default 0.1).

    Returns:
        List of dicts: [{"layer": int, "head": int, "kl_effect": float}, ...]
        sorted by kl_effect descending. Empty list if none exceed threshold.
    """
    circuit_components = []
    for key_string, kl_value in head_sweep_result["kl_matrix"].items():
        if kl_value >= kl_threshold:
            inner_content = key_string.strip("()")
            layer_string, head_string = inner_content.split(",")
            circuit_components.append({
                "layer": int(layer_string),
                "head": int(head_string),
                "kl_effect": kl_value,
            })
    return sorted(circuit_components, key=lambda component: component["kl_effect"], reverse=True)


def run_path_patching(
    source_activations_by_layer_and_head: dict[tuple[int, int], np.ndarray],
    target_run_config: dict[str, Any],
    circuit_components: list[dict[str, Any]],
    token_position: int,
    model: Any,
    baseline_logits: list[float] | None = None,
    full_layer_sweep_peak_kl: float = 0.0,
) -> dict[str, Any]:
    """
    Patch all peak circuit components simultaneously, measure recovered KL.

    Builds one hook per circuit component and runs the target stimulus with
    all components patched at once. fraction_recovered measures how much of
    the full layer sweep effect is explained by just the circuit components.

    circuit_is_sufficient = fraction_recovered >= 0.80.

    Args:
        source_activations_by_layer_and_head: same as run_head_sweep.
        target_run_config: dict with 'stimulus' key.
        circuit_components: output of find_peak_circuit_components.
            Each item has keys: 'layer', 'head', 'kl_effect'.
        token_position: which token to patch.
        model: HookedTransformer instance.
        baseline_logits: unpatched model output logits for KL computation.
        full_layer_sweep_peak_kl: peak KL from the layer sweep (L3). Used to
            compute fraction_recovered.

    Returns:
        {
          "recovered_kl": float,
          "fraction_recovered": float,
          "full_layer_sweep_peak_kl": float,
          "circuit_is_sufficient": bool,
          "n_components_patched": int,
        }
    """
    if not circuit_components:
        return {
            "recovered_kl": 0.0,
            "fraction_recovered": 0.0,
            "full_layer_sweep_peak_kl": full_layer_sweep_peak_kl,
            "circuit_is_sufficient": False,
            "n_components_patched": 0,
        }

    model_device = next(model.parameters()).device
    target_sentence = target_run_config["stimulus"]

    forward_hooks = []
    n_components_patched = 0

    for circuit_component in circuit_components:
        layer_index = circuit_component["layer"]
        head_index = circuit_component["head"]
        if (layer_index, head_index) not in source_activations_by_layer_and_head:
            continue

        source_head_activation = source_activations_by_layer_and_head[(layer_index, head_index)]
        source_tensor = torch.tensor(
            source_head_activation, dtype=torch.float32
        ).to(model_device)

        hook_name = f"blocks.{layer_index}.attn.hook_z"

        def make_multi_head_hook(replacement_tensor, target_head_index, target_token_position):
            def replace_head_at_position(activation_value, hook):
                activation_value[:, target_token_position, target_head_index, :] = replacement_tensor
                return activation_value
            return replace_head_at_position

        forward_hooks.append((
            hook_name,
            make_multi_head_hook(source_tensor, head_index, token_position),
        ))
        n_components_patched += 1

    with torch.no_grad():
        patched_logits = model.run_with_hooks(target_sentence, fwd_hooks=forward_hooks)

    final_token_logits = patched_logits[0, -1, :]

    recovered_kl = 0.0
    if baseline_logits is not None:
        baseline_log_probs = torch.log_softmax(
            torch.tensor(baseline_logits, dtype=torch.float32).to(model_device),
            dim=-1,
        )
        patched_log_probs = torch.log_softmax(final_token_logits, dim=-1)
        recovered_kl = torch.nn.functional.kl_div(
            baseline_log_probs,
            patched_log_probs,
            reduction="sum",
            log_target=True,
        ).item()

    fraction_recovered = (
        recovered_kl / full_layer_sweep_peak_kl
        if full_layer_sweep_peak_kl > 1e-8
        else 0.0
    )

    return {
        "recovered_kl": recovered_kl,
        "fraction_recovered": fraction_recovered,
        "full_layer_sweep_peak_kl": full_layer_sweep_peak_kl,
        "circuit_is_sufficient": fraction_recovered >= 0.80,
        "n_components_patched": n_components_patched,
    }
