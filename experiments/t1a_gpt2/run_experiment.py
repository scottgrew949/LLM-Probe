"""
experiments/t1a/run_experiment.py — T1a full mechanistic experiment.

Runs the complete L2 + L3 pipeline for Thread T1a:
  - Locks ExperimentConfig with pre-specified outcomes (pre-registration)
  - Runs surface-statistics null (V11)
  - Extracts activations at all 24 layers
  - Trains linear probe at each layer (L2 — where is Level 3 stored?)
  - Runs activation patching sweep (L3 — where does it causally control output?)
  - Determines level3_confirmed from pre-specified criterion
  - Unlocks or blocks T1b and T1c

Prerequisite: experiments/t1a/run_validation.py must have run and passed.

Usage (Colab):
    !python experiments/t1a/run_experiment.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

THREAD_ID  = "t1a_gpt2"
MODEL_ID   = "gpt2-medium"

# GPT-2 medium: 24 layers (0-23). Layer 8 is the boundary between
# "early" (syntactic/surface) and "middle/late" (semantic/causal).
GPT2_MEDIUM_N_LAYERS   = 24
MIDDLE_LATE_LAYER_FLOOR = 8

VALIDATED_PATH    = PROJECT_ROOT / "stimuli" / "validated" / "t1a" / "pairs.validated.jsonl"
RESULTS_DIR       = PROJECT_ROOT / "experiments" / THREAD_ID / "results"
CONFIG_PATH       = PROJECT_ROOT / "experiments" / THREAD_ID / "config.json"
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH      = RESULTS_DIR / "summary.json"


# ── Guard: validated file must exist ─────────────────────────────────────────

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Expected: " + str(VALIDATED_PATH))
    print()
    print("This file is written by run_validation.py.")
    print("Run experiments/t1a/run_validation.py first, then re-run this script.")
    sys.exit(1)


# ── Imports (after guard so missing-file error is immediate) ──────────────────

from extraction.extractor import compute_sha256
from experiments.config import ExperimentConfig
from experiments.run import run_surface_null, run_experiment
from stimuli.pipeline import verify_stimulus_file_frequency_matched
from core.io import load_result, save_result


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T1a Mechanistic Experiment")
print("Pearl Level 3 Counterfactual Existence Test")
print("=" * 60)
print()

# ── Step 1: Build config, lock, save ─────────────────────────────────────────

print("[Step 1] Building and locking experiment config...")

# Pre-registration: interpretation key for all possible outcomes.
# These are NOT predictions — they define what each result would mean.
# Written before data collection. This is the computational enforcement
# of Frege's rule against HARKing.
expected_outcomes = {
    "level3_confirmed_criterion": (
        "probe accuracy at peak layer > 0.70 "
        "AND probe accuracy > surface_null_accuracy + 0.10"
    ),
    "outcome_if_separable_middle_late_layers": (
        "causal_l3 and associative_l1 are linearly separable at layers 8-23 "
        "— model encodes Pearl Level 3 structure semantically, not just syntactically. "
        "T1b and T1c proceed."
    ),
    "outcome_if_separable_early_layers_only": (
        "Separable at layers 0-7 only — distinction is syntactic/surface. "
        "Model detects the past-perfect-subjunctive grammatical marker ('had not been'), "
        "not the causal structure. Level 3 confirmed but mechanism is surface, not semantic."
    ),
    "outcome_if_not_separable": (
        "causal_l3 and associative_l1 not linearly separable at any layer. "
        "Model treats interventional and observational framing identically internally. "
        "Level 3 absent. T1b and T1c moot."
    ),
    "outcome_if_layers_agree": (
        "Peak L2 probe layer == peak L3 patching layer — "
        "storage and causal use co-locate. Strong mechanistic finding: "
        "the layer that stores the L3 distinction is the same layer that uses it."
    ),
    "outcome_if_layers_disagree": (
        "Peak L2 layer != peak L3 layer — "
        "distinction is stored at one depth and used at another. "
        "Information flow finding: encoding precedes use."
    ),
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t1a_gpt2m_" + date_stamp,
    thread_id=THREAD_ID,
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, GPT2_MEDIUM_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=verify_stimulus_file_frequency_matched(VALIDATED_PATH),
    expected_outcomes=expected_outcomes,
)

config.lock()
config.to_json(CONFIG_PATH)

print("  Experiment ID : " + config.experiment_id)
print("  Stimulus file : " + str(VALIDATED_PATH))
print("  Layers        : 0 to " + str(GPT2_MEDIUM_N_LAYERS - 1) + " (all " + str(GPT2_MEDIUM_N_LAYERS) + " layers)")
print("  Component     : resid_post (full residual stream)")
print("  Token position: -1 (last token — full sentence context)")
print("  Config locked : " + str(config.pre_spec_locked))
print("  Config saved  : " + str(CONFIG_PATH))
print()


# ── Step 2: Surface-statistics null ──────────────────────────────────────────
# [INVARIANT V11] surface_null.json must exist before check_phase_gate runs.
# run_experiment() calls check_phase_gate() internally — so run_surface_null()
# must be called first.

print("[Step 2] Running surface-statistics null...")
print("  Measuring how much probe signal is explained by word frequency and length alone.")

surface_null_result = run_surface_null(config)
surface_null_accuracy = surface_null_result["surface_classifier_accuracy"]

print("  Surface classifier accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print("  (A probe must exceed this to claim semantic/causal encoding.)")
print("  Written to: " + str(SURFACE_NULL_PATH))
print()


# ── Step 3: Full experiment ───────────────────────────────────────────────────
# run_experiment() handles internally:
#   - check_phase_gate(config) — validates V1, V7, V8, V11
#   - HookedTransformer.from_pretrained(model_id) — model loads here, not above
#   - extract_activations at all 24 layers
#   - run_linear_probe at each layer (L2)
#   - run_layer_sweep activation patching (L3)
#   - assert_specificity_valid at peak layer (V2)
#   - saves probe_layer_{n}.json, layer_sweep.json, summary.json

print("[Step 3] Running full mechanistic experiment...")
print("  Loading " + MODEL_ID + " and running L2 + L3 pipeline.")
print("  This takes ~10-15 minutes on Colab T4.")
print()

experiment_summary = run_experiment(config)

print()
print("  Experiment complete.")
print()


# ── Step 4: Post-process level3_confirmed ────────────────────────────────────
# run_experiment() writes summary.json with level3_confirmed=None.
# Apply the pre-specified criterion and update the summary.

print("[Step 4] Evaluating level3_confirmed criterion...")

peak_probe_accuracy = experiment_summary["peak_probe_accuracy"]
peak_probe_layer    = experiment_summary["peak_probe_layer"]
peak_patch_layer    = experiment_summary["peak_patch_layer"]
layers_agree        = experiment_summary["layers_agree"]

# Re-read surface null from disk for explicit traceability
surface_null_on_disk = load_result(SURFACE_NULL_PATH)
surface_null_accuracy = surface_null_on_disk["surface_classifier_accuracy"]

# Pre-specified criterion (from expected_outcomes above)
probe_exceeds_threshold = peak_probe_accuracy > 0.70
probe_exceeds_surface   = peak_probe_accuracy > surface_null_accuracy + 0.10
level3_confirmed        = probe_exceeds_threshold and probe_exceeds_surface

# Select interpretation
if level3_confirmed:
    if peak_probe_layer >= MIDDLE_LATE_LAYER_FLOOR:
        interpretation = expected_outcomes["outcome_if_separable_middle_late_layers"]
    else:
        interpretation = expected_outcomes["outcome_if_separable_early_layers_only"]
else:
    interpretation = expected_outcomes["outcome_if_not_separable"]

layer_agreement_note = (
    expected_outcomes["outcome_if_layers_agree"]
    if layers_agree
    else expected_outcomes["outcome_if_layers_disagree"]
)

# Update and persist summary — overwrites the None version from run_experiment()
experiment_summary["level3_confirmed"]              = level3_confirmed
experiment_summary["interpretation"]                = interpretation
experiment_summary["layer_agreement_interpretation"] = layer_agreement_note
experiment_summary["surface_null_accuracy"]         = surface_null_accuracy
experiment_summary["level3_confirmed_criterion"]    = expected_outcomes["level3_confirmed_criterion"]

save_result(experiment_summary, SUMMARY_PATH)
print("  Updated summary saved: " + str(SUMMARY_PATH))
print()


# ── Step 5: Layer-by-layer probe accuracy table ───────────────────────────────

print("[Step 5] Layer-by-layer probe accuracy (L2 — where is the distinction stored?)")
print()
print("  Layer   Accuracy    Std    Chance")
print("  -----   --------   -----   ------")

for layer_index in range(GPT2_MEDIUM_N_LAYERS):
    probe_result_path           = RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json")
    probe_data                  = load_result(probe_result_path)
    probe_layer_accuracy        = probe_data["accuracy_mean"]
    probe_layer_std             = probe_data["accuracy_std"]
    probe_layer_chance_baseline = probe_data["chance_baseline"]
    peak_marker = "  <-- PEAK (L2)" if layer_index == peak_probe_layer else ""
    print(
        "  " + str(layer_index).rjust(5) +
        "   " + str(round(probe_layer_accuracy * 100, 1)).rjust(6) + "%" +
        "   " + str(round(probe_layer_std * 100, 1)).rjust(4) + "%" +
        "   " + str(round(probe_layer_chance_baseline * 100, 1)).rjust(5) + "%" +
        peak_marker
    )

print()

# Patching layer effects
layer_sweep = load_result(RESULTS_DIR / "layer_sweep.json")
# JSON round-trip converts int dict keys to strings
peak_patch_kl = layer_sweep["layer_effects"][str(peak_patch_layer)]

print("  Peak patching layer (L3 — where is it causally used?)")
print("  Layer " + str(peak_patch_layer) + "   KL effect from baseline = " + str(round(peak_patch_kl, 4)))
print()


# ── Step 6: Final result ──────────────────────────────────────────────────────

print("=" * 60)
print("T1a Final Result")
print("=" * 60)
print()
print("Peak probe layer    : Layer " + str(peak_probe_layer) + "  (accuracy=" + str(round(peak_probe_accuracy * 100, 1)) + "%)")
print("Peak patching layer : Layer " + str(peak_patch_layer) + "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
print("L2 / L3 agreement   : " + ("YES" if layers_agree else "NO"))
print()
print("Surface null accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%  (baseline from surface features)")
print("Probe accuracy        : " + str(round(peak_probe_accuracy * 100, 1)) + "%  (L2 peak)")
print("Excess over surface   : " + ("+") + str(round((peak_probe_accuracy - surface_null_accuracy) * 100, 1)) + "%  (must be >10% for L3 confirmed)")
print()
print("Probe > 0.70          : " + str(probe_exceeds_threshold))
print("Probe > surface + 0.10: " + str(probe_exceeds_surface))
print("level3_confirmed      : " + str(level3_confirmed))
print()
print("Interpretation:")
print("  " + interpretation)
print()
print("Layer agreement:")
print("  " + layer_agreement_note)
print()
print("=" * 60)
if level3_confirmed:
    print("T1b and T1c: UNLOCKED")
    print()
    print("  T1a confirms GPT-2 medium encodes Pearl Level 3 causal structure.")
    print("  The model distinguishes interventional counterfactuals from")
    print("  observational statements in its internal representations.")
    print()
    print("  Next: design T1b stimuli (forward, backtracking, common-cause)")
    print("  to test Lewis/Stalnaker vs Pearl mechanism.")
    sys.exit(0)
else:
    print("T1b and T1c: BLOCKED")
    print()
    print("  T1a did not confirm Pearl Level 3. GPT-2 medium treats interventional")
    print("  and observational framing identically at the representational level.")
    print("  Cross-architecture replication on Pythia 1.4B recommended before")
    print("  concluding Level 3 is absent.")
    sys.exit(1)
