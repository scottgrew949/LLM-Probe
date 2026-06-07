"""
extraction/extractor.py — Activation extraction via TransformerLens.

─── CONCEPT: Activations as internal representations ─────────────────────────
A transformer processes text by passing a vector through a sequence of layers.
At each layer, every token has a "representation" — a high-dimensional vector
encoding something about that token in context. These vectors are the activations.

The philosophical bet of mechanistic interpretability is that these activations
are not just computational noise — they encode structured information that we
can read and manipulate. extract_activations reaches inside the forward pass
and pulls these vectors out.

─── CONCEPT: Hooks ───────────────────────────────────────────────────────────
TransformerLens works by attaching "hooks" to specific points in the model's
computation graph. A hook is a function that runs at a designated moment during
the forward pass and can read (or write) the activation at that point. This is
non-destructive by default: the model's computation continues normally, but
you've captured the intermediate state.

This is the computational analogue of measuring brain activity with fMRI without
changing what the brain does.

─── CONCEPT: Component choice ────────────────────────────────────────────────
Three main components to hook:
  resid_post  — The full residual stream after each transformer block.
                This is the "total state" of the model at that layer.
                Best starting point for most threads.
  attn_out    — The output of attention heads only (before MLP).
                Use to ask: is this distinction encoded in attention or MLP?
  mlp_out     — The output of the MLP only (before next residual add).
                Attention handles "routing information between tokens";
                MLP is often described as "factual recall" storage.

─── INVARIANT V12: validated/ path and SHA256 ────────────────────────────────
This function refuses to run on stimulus files not under stimuli/validated/.
It also computes the SHA256 of the stimulus file at runtime and compares it
to config.stimulus_sha256. If they don't match, it raises — the stimulus set
has changed since the config was locked, and results would not be reproducible.

─── OUTPUT FORMAT ────────────────────────────────────────────────────────────
Returns a dict:
    {
      "model_id": str,
      "layer": int,
      "component": str,
      "token_position": int,
      "activations": list[list[float]],  # shape: [n_pairs, hidden_dim]
      "labels": list[str],               # theoretical label per item
      "pair_ids": list[str],             # per-sentence id, "_a"/"_b" suffixed — traceability
      "pair_group_ids": list[str],       # base pair_id, SHARED by both sentences of a pair
    }

pair_group_ids is the leakage-safe grouping key (S4): both sentences of a minimal
pair carry the same base id, so passing it to run_linear_probe(pair_ids=...) keeps
them on the same side of every cross-validation fold. pair_ids stays suffixed so a
single row remains individually traceable.

One dict per (layer, component, token_position) combination.
"""

from __future__ import annotations

import hashlib
import json
import torch
from pathlib import Path
from typing import Any

# ExperimentConfig imported at function level to avoid circular imports
# (config.py doesn't import extractor.py, so it's safe, but keeping explicit)


def extract_activations(config: Any, model: Any) -> list[dict[str, Any]]:
    """
    Extract activations from a model at all layers in config.layer_range.

    Runs the model on each stimulus pair in config.stimulus_file, hooking
    into the specified component at each layer and capturing the activation
    at each token position in config.token_positions.

    Args:
        config: ExperimentConfig. Must have:
                - pre_spec_locked == True (V1)
                - stimulus_file under stimuli/validated/ (V12)
                - stimulus_sha256 matching the actual file (V12)
                - frequency_match_verified == True (V7)
        model:  HookedTransformer instance. Caller loads the model once and
                passes it here. This avoids reloading per-layer.

    Returns:
        List of activation dicts, one per (layer, token_position) combination.
        Each dict has shape described in module docstring.

    Raises:
        ValueError: if stimulus_file is not under stimuli/validated/ (V12)
        ValueError: if SHA256 of stimulus_file doesn't match config.stimulus_sha256 (V12)
        ValueError: if config.pre_spec_locked is False (V1)
        ValueError: if config.frequency_match_verified is False (V7)
        ValueError: if a token_position is out of range for a given stimulus
    """
    # These checks are mandatory — do not remove them.
    _assert_validated_path(config.stimulus_file)
    _assert_sha256_match(config.stimulus_file, config.stimulus_sha256)
    _assert_pre_spec_locked(config)
    _assert_frequency_matched(config)

    stimulus_file_path = Path(config.stimulus_file)
    stimulus_pairs = []
    with stimulus_file_path.open("r") as stimulus_file_handle:
        for raw_line in stimulus_file_handle:
            stripped_line = raw_line.strip()
            if stripped_line:
                stimulus_pairs.append(json.loads(stripped_line))

    layer_activation_sets: list[dict] = []

    start_layer, end_layer = config.layer_range

    component_to_hook_suffix = {
        "resid_post": "hook_resid_post",
        "attn_out": "hook_attn_out",
        "mlp_out": "hook_mlp_out",
    }
    hook_suffix = component_to_hook_suffix[config.component]

    for layer_index in range(start_layer, end_layer + 1):
        for token_position in config.token_positions:
            hook_name = f"blocks.{layer_index}.{hook_suffix}"

            captured_activations: list[list[float]] = []
            captured_labels: list[str] = []
            captured_pair_ids: list[str] = []
            captured_pair_group_ids: list[str] = []

            for stimulus_pair in stimulus_pairs:
                for sentence_key, label_key, id_suffix in [
                    ("sentence_a", "label_a", "_a"),
                    ("sentence_b", "label_b", "_b"),
                ]:
                    sentence = stimulus_pair[sentence_key]
                    label = stimulus_pair.get(label_key, sentence_key[-1])
                    pair_id = stimulus_pair.get("pair_id", "")

                    captured_activation: dict[str, Any] = {}

                    def capture_hook(activation_value, hook, _pos=token_position, _store=captured_activation):
                        sequence_length = activation_value.shape[1]
                        # Resolve negative indices (e.g., -1 = last token)
                        resolved_position = _pos if _pos >= 0 else sequence_length + _pos
                        if resolved_position < 0 or resolved_position >= sequence_length:
                            raise ValueError(
                                f"token_position={_pos} is out of range for sentence "
                                f"'{sentence}' which has {sequence_length} tokens. "
                                f"Use -1 for the last token, or check config.token_positions."
                            )
                        _store["vector"] = activation_value[0, resolved_position, :].detach().cpu()
                        return activation_value

                    with torch.no_grad():
                        model.run_with_hooks(sentence, fwd_hooks=[(hook_name, capture_hook)])

                    captured_activations.append(captured_activation["vector"].tolist())
                    captured_labels.append(label)
                    captured_pair_ids.append(f"{pair_id}{id_suffix}")
                    captured_pair_group_ids.append(pair_id)

            layer_activation_sets.append({
                "model_id": config.model_id,
                "layer": layer_index,
                "component": config.component,
                "token_position": token_position,
                "activations": captured_activations,
                "labels": captured_labels,
                "pair_ids": captured_pair_ids,
                "pair_group_ids": captured_pair_group_ids,
            })

    return layer_activation_sets


# ── Enforcement helpers ───────────────────────────────────────────────────────

def _assert_validated_path(stimulus_file: str) -> None:
    """
    [V12] Raise if stimulus_file is not under stimuli/validated/.

    Raw generated files (stimuli/generated/) have not been frequency-matched
    or schema-validated. Only validated/ files are acceptable inputs.
    """
    p = Path(stimulus_file)
    # Walk up the path and check for 'validated' as a parent directory component
    if "validated" not in p.parts:
        raise ValueError(
            f"stimulus_file must be under stimuli/validated/. "
            f"Got: {stimulus_file} (V12). "
            f"Run validate_set() first to move stimuli to validated/."
        )


def _assert_sha256_match(stimulus_file: str, expected_sha256: str) -> None:
    """
    [V12] Compute SHA256 of stimulus_file and raise if it doesn't match expected.

    This ensures that even if the file has been modified after config was locked,
    the mismatch is caught before extraction runs — not discovered later when
    results are inconsistent.
    """
    actual_sha256 = compute_sha256(stimulus_file)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for stimulus file '{stimulus_file}' (V12). "
            f"Expected: {expected_sha256}. "
            f"Actual:   {actual_sha256}. "
            f"The stimulus file has changed since the config was locked. "
            f"Re-lock the config with the current file's SHA256, or restore the original file."
        )


def _assert_pre_spec_locked(config: Any) -> None:
    """[V1] Raise if config.pre_spec_locked is False."""
    if not config.pre_spec_locked:
        raise ValueError(
            "config.pre_spec_locked is False (V1). "
            "Call config.lock() after writing expected_outcomes before extracting."
        )


def _assert_frequency_matched(config: Any) -> None:
    """[V7] Raise if config.frequency_match_verified is False."""
    if not config.frequency_match_verified:
        raise ValueError(
            "config.frequency_match_verified is False (V7). "
            "Run validate_set() on stimulus pairs before extracting."
        )


def compute_sha256(file_path: str | Path) -> str:
    """
    Compute the SHA256 hash of a file and return it as a hex string.

    Use this when constructing a config — compute the hash of your validated
    stimulus file and store it in config.stimulus_sha256 before calling lock().

    Args:
        file_path: Path to any file.

    Returns:
        Lowercase hex SHA256 string, e.g. "a3f2c1..."
    """
    file_path_resolved = Path(file_path)
    with file_path_resolved.open("rb") as binary_file_handle:
        file_content = binary_file_handle.read()
    return hashlib.sha256(file_content).hexdigest()
