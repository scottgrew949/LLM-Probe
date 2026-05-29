"""
interventions/interventions.py — Activation patching and ablation (L3 analysis).

─── CONCEPT: Activation patching as causal intervention ──────────────────────
Activation patching is the sharpest tool in mechanistic interpretability.
A linear probe (L2) can tell you where information is *stored* — which layer
encodes the distinction. But correlation is not causation. Maybe the distinction
is stored at layer 8, but the model actually uses a different layer to make its
decision.

Activation patching answers the causal question directly. The procedure:

  1. Run a "source" stimulus (e.g., opaque context) → capture activation at layer L.
  2. Run a "target" stimulus (e.g., transparent context) → at layer L, *replace*
     its activation with the one from the source.
  3. Check what the model outputs. If the output shifts toward what the model
     would say for the source stimulus, the activation at layer L *causally*
     controls that output.

This is a surgical counterfactual intervention in the Lewis sense: holding
everything else fixed, what would have happened if the internal state at this
moment had been different?

─── CONCEPT: Mean ablation as specificity control ────────────────────────────
Patching source → target proves causal role. But is the effect *specific* to
the philosophical distinction, or would any large change to the activation
at that layer shift the output?

Mean ablation answers this: instead of patching with a real source stimulus,
patch with the *mean activation* over all stimuli. This is the "average"
internal state — it carries no specific information. If mean ablation shifts
the output as much as specific patching, the effect is not specific to the
distinction you're testing.

assert_specificity_valid enforces the requirement that patch effect > ablation
effect. [INVARIANT V2]

─── CONCEPT: Layer-resolved patching ─────────────────────────────────────────
Patching one layer at a time across all layers produces a "causal map" — a
profile showing which layers have causal influence over the output. Combined
with the linear probe accuracy profile (which layers *store* the information),
you get two complementary pictures:
  - Where is it stored? (L2 probe)
  - Where is it used? (L3 patching)

These can differ — early layers may store a distinction that only affects output
when processed through later layers.

─── MODEL LOADING CONVENTION ─────────────────────────────────────────────────
All functions in this module accept `model` as a required parameter. The model
is loaded once by the caller (typically run_experiment in experiments/run.py)
and passed through. Never load the model inside these functions — that would
cause O(N) model loads per layer sweep, wasting minutes of GPU time.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch


# ── Core patching ─────────────────────────────────────────────────────────────

def patch_activation(
    source_activation: np.ndarray,
    target_run_config: dict[str, Any],
    layer: int,
    component: str,
    token_position: int,
    model: Any,  # HookedTransformer — caller loads once and passes in
    baseline_logits: list[float] | None = None,
) -> dict[str, Any]:
    """
    Run target stimulus, replacing its activation at (layer, component, token_position)
    with source_activation, and return the resulting output.

    This is the atomic patching operation. All higher-level patching functions
    (layer-resolved sweep, T5 asymmetry patch, T6 cross-context patch) build
    on this.

    Args:
        source_activation:  np.ndarray of shape (hidden_dim,).
                            The activation to inject from the source run.
        target_run_config:  Dict with 'stimulus' (the target sentence string) and
                            other run parameters.
        layer:              Which layer to patch.
        component:          'resid_post' | 'attn_out' | 'mlp_out'
        token_position:     Which token position to patch.
        model:              HookedTransformer instance. Caller is responsible for
                            loading the model and setting model.eval().
        baseline_logits:    Optional. If provided, KL(patched || baseline) is computed
                            and included in the result as "kl_from_baseline". Pass this
                            when you need an effect-size measure, e.g. for specificity
                            checks or layer sweeps.

    Returns:
        Dict with:
          "logits": list[float]           — output logits for next-token prediction
          "top_tokens": list[str]         — top-5 predicted tokens
          "top_probs": list[float]        — corresponding probabilities
          "patched_layer": int
          "patched_component": str
          "patched_token_position": int
          "kl_from_baseline": float | None — KL(patched || baseline), or None if
                                             baseline_logits was not provided
    """
    component_to_hook_suffix = {
        "resid_post": "hook_resid_post",
        "attn_out": "hook_attn_out",
        "mlp_out": "hook_mlp_out",
    }
    hook_name = f"blocks.{layer}.{component_to_hook_suffix[component]}"

    # Move source activation to the same device as the model
    model_device = next(model.parameters()).device
    source_activation_tensor = torch.tensor(source_activation, dtype=torch.float32).to(model_device)

    def replace_activation_at_position(activation_value, hook):
        activation_value[:, token_position, :] = source_activation_tensor
        return activation_value

    target_sentence = target_run_config["stimulus"]

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            target_sentence,
            fwd_hooks=[(hook_name, replace_activation_at_position)],
        )

    # logits shape: (1, seq_len, vocab_size) — take last token position
    final_token_logits = patched_logits[0, -1, :]
    top_token_probs = torch.softmax(final_token_logits, dim=-1)
    top_5_values, top_5_indices = torch.topk(top_token_probs, k=5)
    top_5_tokens = [model.to_string(idx.item()) for idx in top_5_indices]

    kl_from_baseline = None
    if baseline_logits is not None:
        baseline_log_probs = torch.log_softmax(
            torch.tensor(baseline_logits, dtype=torch.float32).to(model_device), dim=-1
        )
        patched_log_probs = torch.log_softmax(final_token_logits, dim=-1)
        kl_from_baseline = torch.nn.functional.kl_div(
            baseline_log_probs,
            patched_log_probs,
            reduction="sum",
            log_target=True,
        ).item()

    return {
        "logits": final_token_logits.tolist(),
        "top_tokens": top_5_tokens,
        "top_probs": top_5_values.tolist(),
        "patched_layer": layer,
        "patched_component": component,
        "patched_token_position": token_position,
        "kl_from_baseline": kl_from_baseline,
    }


def mean_ablate(
    activations: np.ndarray,
    target_run_config: dict[str, Any],
    layer: int,
    component: str,
    token_position: int,
    model: Any,
    baseline_logits: list[float] | None = None,
) -> dict[str, Any]:
    """
    Replace activation at (layer, component, token_position) with the mean
    over all activations in the stimulus set, then return model output.

    The mean activation is the "information-free" baseline. It is the centroid
    of the activation distribution — the point that carries no specific stimulus
    information.

    Args:
        activations:        np.ndarray of shape (n_items, hidden_dim).
                            The full activation set for this layer/component/position.
                            Mean is computed over axis=0.
        target_run_config:  Dict with target stimulus and run parameters.
        layer, component, token_position, model, baseline_logits: same as patch_activation.

    Returns:
        Same structure as patch_activation returns.
    """
    mean_activation = activations.mean(axis=0)
    return patch_activation(
        mean_activation, target_run_config, layer, component, token_position, model, baseline_logits
    )


def run_layer_sweep(
    source_activation_by_layer: dict[int, np.ndarray],
    target_run_config: dict[str, Any],
    layer_range: tuple[int, int],
    component: str,
    token_position: int,
    model: Any,
) -> dict[str, Any]:
    """
    Patch source activation into target run at every layer in layer_range,
    one layer at a time. Returns the output effect at each layer.

    This produces the "causal map" — how much does patching at each layer
    shift the output toward the source? Combined with probe accuracy per layer,
    this reveals whether storage and use co-locate or diverge.

    Args:
        source_activation_by_layer: Dict mapping layer index → activation vector.
        target_run_config:          Dict with target stimulus.
        layer_range:                (start, end) inclusive.
        component, token_position:  same as patch_activation.
        model:                      HookedTransformer instance.

    Returns:
        Dict with:
          "layer_effects": dict[int, float]  — KL(patched || baseline) per layer
          "peak_layer": int                  — layer with largest effect
          "component": str
          "token_position": int

    Note:
        Effect size is KL divergence between the patched and unpatched output
        distributions. High KL = patching at this layer strongly shifts what
        the model predicts — i.e., this layer causally controls the output.
    """
    target_sentence = target_run_config["stimulus"]

    # Run the target stimulus unpatched to get the baseline output distribution
    with torch.no_grad():
        baseline_logits = model(target_sentence)[0, -1, :].tolist()

    layer_effects: dict[int, float] = {}
    for layer_index in range(layer_range[0], layer_range[1] + 1):
        patch_result = patch_activation(
            source_activation_by_layer[layer_index],
            target_run_config,
            layer_index,
            component,
            token_position,
            model,
            baseline_logits=baseline_logits,
        )
        layer_effects[layer_index] = patch_result["kl_from_baseline"]

    peak_layer = max(layer_effects, key=lambda layer_idx: layer_effects[layer_idx])

    return {
        "layer_effects": layer_effects,
        "peak_layer": peak_layer,
        "component": component,
        "token_position": token_position,
    }


# ── Specificity enforcement ───────────────────────────────────────────────────

def assert_specificity_valid(
    specific_patch_kl: float,
    mean_ablation_kl: float,
    layer: int,
    min_ratio: float = 1.5,
) -> None:
    """
    [INVARIANT V2] Assert that the specific patch effect is meaningfully larger
    than the mean-ablation effect at the same layer.

    Both arguments are KL divergences from the clean (unpatched) baseline output
    distribution. KL(specific_patch || baseline) measures how much the targeted
    intervention shifts the output. KL(mean_ablation || baseline) measures how
    much simply replacing the activation with the dataset mean shifts the output.

    If the two are similar, the effect is not specific to the philosophical
    distinction — any change to that layer's activation would shift the output
    by roughly the same amount. The result would not support a mechanistic claim.

    This is called in the write path for every patch result. If it fails,
    the result is not written and a ValueError is raised.

    Args:
        specific_patch_kl:  KL(specific_patch || baseline). From patch_activation
                            with baseline_logits provided.
        mean_ablation_kl:   KL(mean_ablation || baseline). From mean_ablate
                            with baseline_logits provided.
        layer:              Which layer was patched. Used in the error message.
        min_ratio:          specific_patch_kl / mean_ablation_kl must exceed this.
                            Default 1.5 — specific patch must shift output 50% more
                            than mean ablation to claim specificity.

    Raises:
        ValueError: if the specificity ratio < min_ratio.
    """
    if mean_ablation_kl < 1e-8:
        # Mean ablation had essentially no effect — the activation at this layer
        # carries no information about the output. Specific patching may still be
        # meaningful, but we can't compute a ratio. Pass through.
        return

    specificity_ratio = specific_patch_kl / mean_ablation_kl

    if specificity_ratio < min_ratio:
        raise ValueError(
            f"Specificity check failed at layer {layer} (V2). "
            f"KL(specific_patch || baseline)={specific_patch_kl:.4f}, "
            f"KL(mean_ablation || baseline)={mean_ablation_kl:.4f}, "
            f"ratio={specificity_ratio:.3f} < required {min_ratio}. "
            f"The patching effect is not specific to the philosophical distinction — "
            f"any activation change at this layer shifts output by roughly the same amount."
        )


# ── Thread-specific patching ──────────────────────────────────────────────────

def patch_t5_asymmetry(
    source_grounded: np.ndarray,
    source_ungrounded: np.ndarray,
    grounded_run_config: dict[str, Any],
    ungrounded_run_config: dict[str, Any],
    config: Any,
    model: Any,
) -> dict[str, Any]:
    """
    T5-specific patch: test Fine's grounding asymmetry by patching in both directions.

    Fine's asymmetry condition: if A grounds B, then B does not ground A.
    Forward patch: inject grounded (A-type) activation into the ungrounded (B-type)
    target — measures how much knowing A's representation shifts B's output.
    Backward patch: inject ungrounded (B-type) activation into the grounded (A-type)
    target — should produce a *smaller* shift if grounding is asymmetric.

    [INVARIANT V4] Raises if config.t5_asymmetry_thresholds is None.

    Args:
        source_grounded:        Activation from a grounded (A-type, e.g. number) stimulus.
        source_ungrounded:      Activation from an ungrounded (B-type, e.g. property) stimulus.
        grounded_run_config:    Run config for the grounded target (used in backward patch).
        ungrounded_run_config:  Run config for the ungrounded target (used in forward patch).
        config:                 ExperimentConfig with t5_asymmetry_thresholds set.
        model:                  HookedTransformer instance.

    Returns:
        Dict with forward and backward KL effects, and whether asymmetry threshold is met.
    """
    if config.t5_asymmetry_thresholds is None:
        raise ValueError(
            "config.t5_asymmetry_thresholds is None (V4). "
            "Set thresholds in ExperimentConfig before running T5 patching."
        )

    patch_layer = config.layer_range[0]
    patch_token_position = config.token_positions[0]

    # Forward: inject grounded activation into ungrounded target
    forward_result = patch_activation(
        source_grounded,
        ungrounded_run_config,
        patch_layer,
        config.component,
        patch_token_position,
        model,
    )

    # Backward: inject ungrounded activation into grounded target
    # If grounding is asymmetric, this backward effect should be smaller
    backward_result = patch_activation(
        source_ungrounded,
        grounded_run_config,
        patch_layer,
        config.component,
        patch_token_position,
        model,
    )

    forward_top_prob = forward_result["top_probs"][0]
    backward_top_prob = backward_result["top_probs"][0]

    asymmetry_threshold = config.t5_asymmetry_thresholds.get("math_vs_unrelated_floor", 0.0)
    asymmetry_is_met = (forward_top_prob - backward_top_prob) > asymmetry_threshold

    return {
        "forward_patch_top_prob": forward_top_prob,
        "backward_patch_top_prob": backward_top_prob,
        "asymmetry_delta": forward_top_prob - backward_top_prob,
        "asymmetry_threshold": asymmetry_threshold,
        "asymmetry_met": asymmetry_is_met,
    }


def patch_t6_cross_context(
    source_rigid_activation: np.ndarray,
    target_description_run_config: dict[str, Any],
    config: Any,
    model: Any,
) -> dict[str, Any]:
    """
    T6-specific patch: inject activation from a rigid designator (proper name)
    context into a definite description context across modal frames.

    Kripke's rigid designation thesis: proper names refer to the same individual
    in all possible worlds; definite descriptions do not (they pick out whoever
    satisfies the description in each world). If the model encodes this distinction,
    patching a proper name's representation into a definite description context
    should shift the output in a predictable direction across modal operators
    ('necessarily', 'possibly', 'in world w').

    Args:
        source_rigid_activation:         Activation from a proper name (rigid) stimulus.
        target_description_run_config:   Run config for a definite description target.
                                         May contain sub-keys "necessity", "possibility",
                                         "counterfactual" with context-specific stimuli.
        config:                          ExperimentConfig for T6.
        model:                           HookedTransformer instance.

    Returns:
        Dict with patching effect (top_prob) per modal operator type.
    """
    modal_context_types = ["necessity", "possibility", "counterfactual"]
    effect_per_context: dict[str, float] = {}

    for context_type in modal_context_types:
        context_specific_config = target_description_run_config.get(
            context_type, target_description_run_config
        )
        patch_result = patch_activation(
            source_rigid_activation,
            context_specific_config,
            config.layer_range[0],
            config.component,
            config.token_positions[0],
            model,
        )
        effect_per_context[context_type] = patch_result["top_probs"][0]

    return {"effect_per_context_type": effect_per_context}
