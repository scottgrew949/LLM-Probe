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
    from sklearn.preprocessing import LabelEncoder
    from core.text_utils import mean_log10_frequency
    from probes.probes import _adaptive_grouped_cv_accuracy

    stimulus_file_path = Path(config.stimulus_file)
    stimulus_pairs = []
    with stimulus_file_path.open("r") as stimulus_file_handle:
        for raw_line in stimulus_file_handle:
            stripped_line = raw_line.strip()
            if stripped_line:
                stimulus_pairs.append(json.loads(stripped_line))

    # One feature row per sentence: [mean_log10_freq, token_count]
    # Label = theoretical label of that sentence (e.g. "opaque", "transparent")
    surface_feature_rows = []
    sentence_labels = []
    # Grouping key per row: both sentences of a pair share the base pair_id, so
    # the surface classifier is held to the same leakage-safe folds as the L2
    # probe (S4) — a pair's near-duplicate surface features cannot straddle folds.
    surface_pair_group_ids = []

    # Pair-level summary statistics (not used as classifier features)
    freq_diffs = []
    length_diffs = []
    jaccard_overlaps = []

    for pair_index, stimulus_pair in enumerate(stimulus_pairs):
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

        pair_group_id = stimulus_pair.get("pair_id", str(pair_index))
        surface_pair_group_ids.append(pair_group_id)  # sentence_a row
        surface_pair_group_ids.append(pair_group_id)  # sentence_b row

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
    surface_cross_validation = _adaptive_grouped_cv_accuracy(
        surface_classifier, surface_feature_matrix, encoded_labels, config.seed,
        groups=surface_pair_group_ids,
    )
    if surface_cross_validation["scores"] is None:
        surface_classifier_accuracy = float("nan")
    else:
        surface_classifier_accuracy = float(surface_cross_validation["scores"].mean())

    surface_null_result = {
        "surface_classifier_accuracy": surface_classifier_accuracy,
        "mean_freq_diff": float(np.mean(freq_diffs)),
        "mean_length_diff_tokens": float(np.mean(length_diffs)),
        "mean_vocab_overlap": float(np.mean(jaccard_overlaps)),
        "n_pairs": len(stimulus_pairs),
        "n_folds": surface_cross_validation["n_splits"],
        "thread_id": config.thread_id,
        "experiment_id": config.experiment_id,
        "note": (
            "surface_classifier_accuracy uses per-sentence features [log10_freq, length], "
            "cross-validated with pair-grouped folds (same leakage-safe split as the L2 probe). "
            "mean_vocab_overlap (Jaccard) is a summary statistic only, not a classifier feature."
        ),
    }
    if surface_cross_validation["note"] is not None:
        surface_null_result["cv_note"] = surface_cross_validation["note"]

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
    # results_directory uses the full (possibly suffixed) thread_id so a
    # replication variant's outputs stay isolated. The thread-specific gates below
    # key on config.base_thread instead, so a "_pythia" variant cannot bypass
    # V5/V10/V14/V15 — see ExperimentConfig.base_thread.
    results_directory = PROJECT_ROOT / "experiments" / config.thread_id / "results"
    base_thread_id = config.base_thread

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
    if base_thread_id in ("t1b", "t1c"):
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

    # For t1d: only require T1b summary exists (not outcome-gated)
    # T1d interpretation depends on pearl_confirmed but does not gate on it
    if base_thread_id == "t1d" and config.prerequisite_experiment_id:
        t1b_summary_path = (
            PROJECT_ROOT / "experiments" / config.prerequisite_experiment_id / "results" / "summary.json"
        )
        if not t1b_summary_path.exists():
            raise FileNotFoundError(
                f"V10: T1d requires T1b summary at {t1b_summary_path}. "
                f"Run T1b first."
            )

    # V14: T1d requires identification_criterion
    if base_thread_id == "t1d" and not config.identification_criterion:
        raise ValueError(
            "V14: identification_criterion is None for T1d. "
            "Set to 'back_door' or 'front_door' before running."
        )

    # V15: T1d requires confounder_structure
    if base_thread_id == "t1d" and not config.confounder_structure:
        raise ValueError(
            "V15: confounder_structure is None for T1d. "
            "Define the formal causal graph before running."
        )

    # V16: T2c requires T2b gate passing
    if base_thread_id == "t2c" and config.prerequisite_experiment_id:
        t2b_summary_path = (
            PROJECT_ROOT / "experiments" / config.prerequisite_experiment_id / "results" / "summary.json"
        )
        if not t2b_summary_path.exists():
            raise FileNotFoundError(
                f"V16: T2c requires T2b summary at {t2b_summary_path}. "
                f"Run T2b on Llama 3.2 3B first."
            )

    # V17: T2c requires intension_type
    if base_thread_id == "t2c" and not config.intension_type:
        raise ValueError(
            "V17: intension_type is None for T2c. "
            "Set to 'primary', 'secondary', or 'dissociation' before running."
        )

    # V18: circuit analysis requires layer sweep results
    if config.circuit_analysis_enabled:
        layer_sweep_results_path = (
            PROJECT_ROOT / "experiments" / config.thread_id / "results" / "layer_sweep.json"
        )
        if not layer_sweep_results_path.exists():
            raise FileNotFoundError(
                f"V18: circuit_analysis_enabled=True but layer_sweep.json not found "
                f"at {layer_sweep_results_path}. Run layer sweep first."
            )

    # V4: T5 requires asymmetry thresholds pre-specified from null distributions
    if base_thread_id == "t5" and config.t5_asymmetry_thresholds is None:
        raise ValueError("V4: T5 requires t5_asymmetry_thresholds set in config.")

    # V13: T4 requires ontology provenance documented before RSA
    if base_thread_id == "t4":
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
    from interventions.interventions import (
        run_layer_sweep_multi_target, assert_specificity_valid, norm_matched_control_kl,
    )

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

        # pair_group_ids groups both sentences of a minimal pair into one fold (S4).
        probe_result = run_linear_probe(
            activations_array, labels, config, pair_ids=activation_set["pair_group_ids"]
        )
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

        # Targets: every class-B sentence. Multi-target sweep → mean KL + bootstrap
        # CI over targets, not a single-sentence n=1 effect.
        target_sentences = [pair["sentence_b"] for pair in stimulus_pairs]

        sweep_result = run_layer_sweep_multi_target(
            mean_source_activations_by_layer,
            target_sentences,
            config.layer_range,
            config.component,
            config.token_positions[0],
            model,
            seed=config.seed,
        )
        save_result(sweep_result, results_directory / "layer_sweep.json")

        # V2: specificity at the peak layer vs norm-matched random directions
        # (calibrated control, not the grand-mean ablation).
        peak_layer = sweep_result["peak_layer"]
        control_result = norm_matched_control_kl(
            mean_source_activations_by_layer[peak_layer], target_sentences,
            peak_layer, config.component, config.token_positions[0], model,
            seed=config.seed,
        )
        specific_patch_kl = sweep_result["mean_kl_by_layer"][peak_layer]
        assert_specificity_valid(specific_patch_kl, control_result["control_kl_p95"], peak_layer)

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
