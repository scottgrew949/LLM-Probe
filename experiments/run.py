"""
experiments/run.py — Experiment execution: surface null, phase gate, full run.

─── CONCEPT: The three-layer methodology ─────────────────────────────────────
Every thread runs the same three analyses, in this order:

  1. Surface-statistics null (run_surface_null)
     Measure how much of the observed signal can be explained by surface features:
     word frequency and sentence length. This is the *boring* explanation. If
     surface stats explain 90% of the probe accuracy, the probe is not measuring
     semantics. Written to surface_null.json before anything else.

  2. Linear probe (L2) — see probes/probes.py
     Locate which layer encodes the philosophical distinction. Runs on activations
     from extraction/extractor.py.

  3. Activation patching (L3) — see interventions/interventions.py
     Prove causal role. The layer identified by L2 should be the layer with
     the largest L3 patching effect. If they agree, that's a strong mechanistic
     finding.

─── CONCEPT: Why surface null runs first (V11) ───────────────────────────────
Writing surface_null.json before any other result is a methodological commitment:
you have measured the boring explanation before you look at the interesting one.
This prevents a subtle form of motivated reasoning — if you ran probes first and
got high accuracy, you might be less motivated to run the surface null carefully.
Checking surface stats first keeps you honest.

run_experiment() checks that surface_null.json exists before writing any other
result. [INVARIANT V11]

─── CONCEPT: Phase gate ──────────────────────────────────────────────────────
check_phase_gate() enforces the sequencing requirements before a run starts:
  - Config is locked (V1)
  - Frequency matching verified (V7)
  - Behavioral gate was run and passed (V8)
  - Surface null written (V11)
  - For T1b/T1c: prerequisite experiment has a passing result (V10)
  - For T5: asymmetry thresholds are set (V4)
  - For T4: ontology provenance is documented (V13)

This is called at the top of run_experiment(). Failing a gate fails loudly with
a specific error message — not silently or with garbage output.

─── MODEL LOADING CONVENTION ─────────────────────────────────────────────────
run_experiment loads the model once and passes it to every function that needs
it. This means one model load per experiment, not one per layer. All functions
downstream (extract_activations, run_layer_sweep, patch_activation) accept the
model as a required parameter.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.config import ExperimentConfig
from core.io import save_result, load_result

# Project root: one level up from experiments/
PROJECT_ROOT = Path(__file__).parent.parent


# ── Surface-statistics null ───────────────────────────────────────────────────

def run_surface_null(config: ExperimentConfig) -> dict[str, Any]:
    """
    Compute surface-statistics baseline for the stimulus set.

    Trains a logistic regression classifier on surface features of individual
    sentences to predict their theoretical label (e.g. "opaque" vs "transparent").
    The accuracy of this surface classifier is the ceiling for the "boring
    explanation" — if the L2 probe significantly exceeds it, the probe is
    detecting real semantic content, not surface artifacts.

    Surface features per sentence:
      - Mean log10 corpus frequency of content words
      - Token count

    Jaccard vocabulary overlap is a pair-level property (same value for both
    sentences in a pair) and cannot discriminate between the two theoretical
    labels within a pair, so it is excluded from classifier features. It is
    reported as a summary statistic only.

    [INVARIANT V11] Writes result to experiments/{thread_id}/results/surface_null.json.
    This file must exist before any other result is written.

    Args:
        config: ExperimentConfig. Reads stimulus_file.

    Returns:
        Dict with:
          "surface_classifier_accuracy": float — how well surface predicts labels
          "mean_freq_diff": float              — mean |freq_a - freq_b| across pairs
          "mean_length_diff_tokens": float     — mean |len_a - len_b| across pairs
          "mean_vocab_overlap": float          — mean Jaccard across pairs (summary only)
          "n_pairs": int

    Side effects:
        Writes surface_null.json to experiments/{config.thread_id}/results/.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from wordfreq import word_frequency
    import re

    stimulus_file_path = Path(config.stimulus_file)
    stimulus_pairs = []
    with stimulus_file_path.open("r") as stimulus_file_handle:
        for raw_line in stimulus_file_handle:
            stripped_line = raw_line.strip()
            if stripped_line:
                stimulus_pairs.append(json.loads(stripped_line))

    function_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "and", "or", "but", "not",
        "that", "this", "it", "its",
    }

    def mean_log10_frequency(sentence: str) -> float:
        words = re.findall(r"[a-z]+", sentence.lower())
        content_words = [w for w in words if w not in function_words]
        if not content_words:
            return 0.0
        log_freqs = [math.log10(max(word_frequency(w, "en"), 1e-9)) for w in content_words]
        return sum(log_freqs) / len(log_freqs)

    # One feature row per sentence: [mean_log10_freq, token_count]
    # Label = theoretical label of that sentence (e.g. "opaque", "transparent")
    surface_feature_rows = []
    sentence_labels = []

    # Pair-level summary statistics (not used as classifier features)
    freq_diffs = []
    length_diffs = []
    jaccard_overlaps = []

    for stimulus_pair in stimulus_pairs:
        sentence_a = stimulus_pair.get("sentence_a", "")
        sentence_b = stimulus_pair.get("sentence_b", "")
        label_a = stimulus_pair.get("label_a", "a")
        label_b = stimulus_pair.get("label_b", "b")

        freq_a = mean_log10_frequency(sentence_a)
        freq_b = mean_log10_frequency(sentence_b)
        tokens_a = sentence_a.lower().split()
        tokens_b = sentence_b.lower().split()

        surface_feature_rows.append([freq_a, len(tokens_a)])
        sentence_labels.append(label_a)

        surface_feature_rows.append([freq_b, len(tokens_b)])
        sentence_labels.append(label_b)

        freq_diffs.append(abs(freq_a - freq_b))
        length_diffs.append(abs(len(tokens_a) - len(tokens_b)))

        set_a = set(tokens_a)
        set_b = set(tokens_b)
        jaccard = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 1.0
        jaccard_overlaps.append(jaccard)

    surface_feature_matrix = np.array(surface_feature_rows)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(sentence_labels)

    surface_classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=config.seed)
    cross_val_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    surface_cross_val_scores = cross_val_score(
        surface_classifier, surface_feature_matrix, encoded_labels, cv=cross_val_splitter
    )

    surface_null_result = {
        "surface_classifier_accuracy": float(surface_cross_val_scores.mean()),
        "mean_freq_diff": float(np.mean(freq_diffs)),
        "mean_length_diff_tokens": float(np.mean(length_diffs)),
        "mean_vocab_overlap": float(np.mean(jaccard_overlaps)),
        "n_pairs": len(stimulus_pairs),
        "thread_id": config.thread_id,
        "experiment_id": config.experiment_id,
        "note": (
            "surface_classifier_accuracy uses per-sentence features [log10_freq, length]. "
            "mean_vocab_overlap (Jaccard) is a summary statistic only, not a classifier feature."
        ),
    }

    # [INVARIANT V11] Write surface_null.json FIRST — before any other result
    results_directory = PROJECT_ROOT / "experiments" / config.thread_id / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    save_result(surface_null_result, results_directory / "surface_null.json")

    return surface_null_result


# ── Phase gate ────────────────────────────────────────────────────────────────

def check_phase_gate(config: ExperimentConfig) -> None:
    """
    Check all pre-conditions for running an experiment. Raises if any fail.

    Checks performed:
      - V1: config.pre_spec_locked is True
      - V7: config.frequency_match_verified is True
      - V8: behavioral_gate.json exists for this thread and shows passed=True
      - V11: surface_null.json exists for this thread
      - V10: for T1b/T1c, prerequisite experiment result exists and passed
      - V4: for T5, t5_asymmetry_thresholds is not None
      - V13: for T4, ontology_version and matrix_source are not None

    Args:
        config: ExperimentConfig to check.

    Raises:
        ValueError: for any failed gate, with specific V-number and fix instructions.
    """
    results_directory = PROJECT_ROOT / "experiments" / config.thread_id / "results"

    # V1: config must be locked (pre-registration complete)
    if not config.pre_spec_locked:
        raise ValueError("V1: config.pre_spec_locked is False. Call config.lock() first.")

    # V7: frequency matching must be verified before extraction
    if not config.frequency_match_verified:
        raise ValueError("V7: frequency_match_verified is False. Run validate_set() first.")

    # V8: behavioral gate must have been run and passed
    behavioral_gate_path = results_directory / "behavioral_gate.json"
    if not behavioral_gate_path.exists():
        raise ValueError(
            f"V8: behavioral_gate.json not found at {behavioral_gate_path}. "
            f"Run run_behavioral_gate() and save the result before any mechanistic analysis."
        )
    behavioral_gate_result = load_result(behavioral_gate_path)
    if not behavioral_gate_result.get("passed", False):
        accuracy = behavioral_gate_result.get("accuracy", 0.0)
        raise ValueError(
            f"V8: Behavioral gate did not pass (accuracy={accuracy:.2f} < 0.70). "
            f"The model does not exhibit the philosophical distinction behaviorally. "
            f"There is nothing to explain mechanistically."
        )

    # V11: surface null must have been written before any other result
    surface_null_path = results_directory / "surface_null.json"
    if not surface_null_path.exists():
        raise ValueError(
            f"V11: surface_null.json not found at {surface_null_path}. "
            f"Run run_surface_null(config) before any other analysis."
        )

    # V10: T1b and T1c require T1a to have confirmed Level 3 encoding
    if config.thread_id in ("t1b", "t1c"):
        if config.prerequisite_experiment_id is None:
            raise ValueError(
                "V10: T1b/T1c require prerequisite_experiment_id set in config. "
                "Set it to the experiment_id of the passing T1a run."
            )
        prerequisite_summary_path = (
            PROJECT_ROOT / "experiments" / config.prerequisite_experiment_id / "results" / "summary.json"
        )
        if not prerequisite_summary_path.exists():
            raise ValueError(
                f"V10: Prerequisite experiment '{config.prerequisite_experiment_id}' "
                f"has no summary.json at {prerequisite_summary_path}. "
                f"Run T1a first and ensure it produces a passing summary."
            )
        prerequisite_summary = load_result(prerequisite_summary_path)
        if not prerequisite_summary.get("level3_confirmed", False):
            raise ValueError(
                f"V10: Prerequisite experiment '{config.prerequisite_experiment_id}' "
                f"did not confirm Level 3 (level3_confirmed=False in summary.json). "
                f"T1b/T1c require T1a to confirm causal hierarchy before running."
            )

    # V4: T5 requires asymmetry thresholds pre-specified from null distributions
    if config.thread_id == "t5" and config.t5_asymmetry_thresholds is None:
        raise ValueError("V4: T5 requires t5_asymmetry_thresholds set in config.")

    # V13: T4 requires ontology provenance documented before RSA
    if config.thread_id == "t4":
        if config.ontology_version is None or config.matrix_source is None:
            raise ValueError(
                "V13: T4 requires ontology_version and matrix_source set in config."
            )


# ── Full experiment runner ────────────────────────────────────────────────────

def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """
    Run a full experiment for one thread on one model.

    Loads the model once, then runs the full pipeline:
      1. check_phase_gate(config)           — validate all pre-conditions
      2. extract_activations(config, model) — hook model, capture internals
      3. run_linear_probe(activations, ...) — L2: where is the distinction?
      4. run_rsa(...) if T4               — does geometry match ontology?
      5. run_layer_sweep(...)              — L3: where is it used?
      6. assert_specificity_valid(...)     — V2: was effect specific?
      7. save all results
      8. write summary.json

    Args:
        config: ExperimentConfig with pre_spec_locked == True.

    Returns:
        Summary dict written to experiments/{thread_id}/results/summary.json.

    Raises:
        ValueError: if check_phase_gate fails.
        ValueError: if config.pre_spec_locked is False (V1).
    """
    from transformer_lens import HookedTransformer
    from extraction.extractor import extract_activations
    from probes.probes import run_linear_probe
    from interventions.interventions import run_layer_sweep, assert_specificity_valid, mean_ablate

    # V1: double-check before any work begins
    if not config.pre_spec_locked:
        raise ValueError("V1: config.pre_spec_locked is False. Cannot run experiment.")

    check_phase_gate(config)

    config.run_timestamp = datetime.datetime.utcnow().isoformat()

    # Load model once — pass to every downstream function
    model = HookedTransformer.from_pretrained(config.model_id)
    model.eval()

    results_directory = PROJECT_ROOT / "experiments" / config.thread_id / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract activations — one dict per (layer, token_position)
    layer_activation_sets = extract_activations(config, model)

    # Step 2: Linear probe at each layer (L2 — where is the distinction stored?)
    probe_results_by_layer: dict[int, dict] = {}
    for activation_set in layer_activation_sets:
        layer_index = activation_set["layer"]
        activations_array = np.array(activation_set["activations"])
        labels = activation_set["labels"]

        probe_result = run_linear_probe(activations_array, labels, config)
        probe_result["layer"] = layer_index
        probe_result["token_position"] = activation_set["token_position"]

        save_result(probe_result, results_directory / f"probe_layer_{layer_index}.json")
        probe_results_by_layer[layer_index] = probe_result

    # Step 3: Layer sweep (L3 — where is the distinction causally used?)
    # Source: mean activation of class A across all pairs at each layer.
    # Target: a class B sentence. KL divergence at each layer = causal effect size.
    stimulus_file_path = Path(config.stimulus_file)
    stimulus_pairs = []
    with stimulus_file_path.open("r") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                stimulus_pairs.append(json.loads(stripped))

    sweep_result = None
    if stimulus_pairs:
        # Build mean source activation (class A = even-indexed items: a0, a1, a2, ...)
        # Activations are interleaved: a0, b0, a1, b1, ... so even indices are all A-class
        mean_source_activations_by_layer: dict[int, np.ndarray] = {}
        for activation_set in layer_activation_sets:
            all_activations = np.array(activation_set["activations"])
            class_a_activations = all_activations[::2]  # every other row starting at 0
            mean_source_activations_by_layer[activation_set["layer"]] = class_a_activations.mean(axis=0)

        # Use last pair's sentence_b as the target — avoids overlap with source computation
        target_run_config = {"stimulus": stimulus_pairs[-1]["sentence_b"]}

        sweep_result = run_layer_sweep(
            mean_source_activations_by_layer,
            target_run_config,
            config.layer_range,
            config.component,
            config.token_positions[0],
            model,
        )
        save_result(sweep_result, results_directory / "layer_sweep.json")

        # V2: specificity check at the peak layer
        # Mean-ablate the target at the peak layer and compare KL effects
        peak_layer = sweep_result["peak_layer"]
        peak_activations = np.array(layer_activation_sets[peak_layer]["activations"])

        # Get baseline logits for the target stimulus (unpatched)
        import torch
        with torch.no_grad():
            baseline_logits = model(target_run_config["stimulus"])[0, -1, :].tolist()

        mean_ablation_result = mean_ablate(
            peak_activations,
            target_run_config,
            peak_layer,
            config.component,
            config.token_positions[0],
            model,
            baseline_logits=baseline_logits,
        )

        specific_patch_kl = sweep_result["layer_effects"][peak_layer]
        mean_ablation_kl = mean_ablation_result["kl_from_baseline"] or 0.0

        assert_specificity_valid(specific_patch_kl, mean_ablation_kl, peak_layer)

    # Step 4: Find peak layers and compare L2 vs L3
    probe_accuracy_by_layer = {
        layer_idx: probe_results_by_layer[layer_idx]["accuracy_mean"]
        for layer_idx in probe_results_by_layer
    }
    peak_probe_layer = max(probe_accuracy_by_layer, key=lambda l: probe_accuracy_by_layer[l])
    peak_patch_layer = sweep_result["peak_layer"] if sweep_result else None

    summary = {
        "experiment_id": config.experiment_id,
        "thread_id": config.thread_id,
        "model_id": config.model_id,
        "run_timestamp": config.run_timestamp,
        "peak_probe_layer": peak_probe_layer,
        "peak_probe_accuracy": float(probe_accuracy_by_layer[peak_probe_layer]),
        "peak_patch_layer": peak_patch_layer,
        "layers_agree": peak_probe_layer == peak_patch_layer,
        "expected_outcomes": config.expected_outcomes,
        "n_layers_probed": len(probe_results_by_layer),
        "level3_confirmed": None,  # T1a-specific; set by T1a post-processing
    }

    save_result(summary, results_directory / "summary.json")
    return summary
