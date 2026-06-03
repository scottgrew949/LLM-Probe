"""
experiments/t1d/run_experiment.py — T1d mechanistic experiment: causal identification.

Tests whether GPT-2 medium representations respect do-calculus identification
conditions — specifically, whether the model internally distinguishes confounded
structures with a valid adjustment set from those without one.

T1d is informative regardless of T1b outcome:
  If T1b = Pearl: tests whether Pearl-consistent representations respect
    identification constraints (strong claim — full do-calculus).
  If T1b = Lewis: tests whether Lewis-consistent representations fail
    identification tests (consistent with worlds-ordering having no
    notion of adjustability).

Prerequisite: T1b must be complete (not outcome-gated).

Usage (Colab):
    !python experiments/t1d/run_experiment.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

THREAD_ID = "t1d"
MODEL_ID = "gpt2-medium"
GPT2_MEDIUM_N_LAYERS = 24

VALIDATED_PATH = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / THREAD_ID / "results"
CONFIG_PATH = PROJECT_ROOT / "experiments" / THREAD_ID / "config.json"
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

T1B_PREREQUISITE_ID = "t1b"


# ── Guards ────────────────────────────────────────────────────────────────────

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Expected: " + str(VALIDATED_PATH))
    print("  Run scripts/run_validation.py --thread t1d first.")
    sys.exit(1)

t1b_summary_path = PROJECT_ROOT / "experiments" / "t1b" / "results" / "summary.json"
if not t1b_summary_path.exists():
    print("ERROR: T1b summary not found.")
    print("  Expected: " + str(t1b_summary_path))
    print("  Run experiments/t1b/run_experiment.py first.")
    sys.exit(1)


# ── Imports ───────────────────────────────────────────────────────────────────

from extraction.extractor import compute_sha256, extract_activations
from experiments.config import ExperimentConfig
from experiments.run import run_surface_null, check_phase_gate
from probes.probes import run_linear_probe, run_identification_probe
from interventions.interventions import run_layer_sweep, assert_specificity_valid, mean_ablate
from core.io import load_result, save_result
import torch


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T1d Mechanistic Experiment")
print("Causal Identification: Back-Door vs Front-Door vs Unidentifiable")
print("=" * 60)
print()

# Load T1b outcome — T1d interpretation is conditioned on it
t1b_summary = load_result(t1b_summary_path)
pearl_confirmed_in_t1b = t1b_summary.get("pearl_confirmed", False)
print("[Context] T1b pearl_confirmed = " + str(pearl_confirmed_in_t1b))
if pearl_confirmed_in_t1b:
    print("  T1d tests whether Pearl-consistent representations respect identification.")
else:
    print("  T1d tests whether Lewis-consistent representations fail identification.")
print()

# ── Step 1: Build config ──────────────────────────────────────────────────────

print("[Step 1] Building and locking experiment config...")

confounder_structure_description: dict = {
    "conditions": {
        "back_door_adjustable": {
            "nodes": ["treatment", "outcome", "confounder"],
            "edges": [["treatment", "outcome"], ["confounder", "treatment"], ["confounder", "outcome"]],
            "criterion": "back_door",
            "adjustment_set": ["confounder"],
        },
        "front_door_adjustable": {
            "nodes": ["treatment", "mediator", "outcome", "hidden_confounder"],
            "edges": [["treatment", "mediator"], ["mediator", "outcome"],
                      ["hidden_confounder", "treatment"], ["hidden_confounder", "outcome"]],
            "criterion": "front_door",
            "adjustment_set": ["mediator"],
        },
        "confounded_not_adjustable": {
            "nodes": ["treatment", "outcome", "hidden_confounder"],
            "edges": [["treatment", "outcome"],
                      ["hidden_confounder", "treatment"], ["hidden_confounder", "outcome"]],
            "criterion": "none",
            "adjustment_set": [],
        },
        "unconfounded_control": {
            "nodes": ["treatment", "outcome"],
            "edges": [["treatment", "outcome"]],
            "criterion": "trivial",
            "adjustment_set": [],
        },
    }
}

expected_outcomes_description: dict = {
    "identification_criterion": (
        "Binary probe accuracy (adjustable vs not_adjustable) at peak layer. "
        "> 0.70 -> model encodes adjustability. <= 0.70 -> no identifiability representation."
    ),
    "outcome_if_pearl_and_encodes_identification": (
        "back_door_adjustable and front_door_adjustable cluster together, "
        "separated from confounded_not_adjustable. Pearl representations respect "
        "do-calculus identification conditions — the full causal hierarchy."
    ),
    "outcome_if_lewis_and_fails_identification": (
        "No separation between adjustable and not_adjustable conditions. "
        "Lewis similarity ordering has no notion of identifiability — "
        "the model geometry is flat across identification conditions."
    ),
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t1d_gpt2m_" + date_stamp,
    thread_id=THREAD_ID,
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, GPT2_MEDIUM_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=True,
    expected_outcomes=expected_outcomes_description,
    prerequisite_experiment_id=T1B_PREREQUISITE_ID,
    identification_criterion="back_door",
    confounder_structure=confounder_structure_description,
)

config.lock()
config.to_json(CONFIG_PATH)

print("  Experiment ID         : " + config.experiment_id)
print("  Identification crit.  : " + str(config.identification_criterion))
print("  Prerequisite (T1b)    : " + T1B_PREREQUISITE_ID)
print("  Config locked         : " + str(config.pre_spec_locked))
print()

# ── Step 2: Surface null ──────────────────────────────────────────────────────

print("[Step 2] Running surface-statistics null...")
surface_null_result = run_surface_null(config)
surface_null_accuracy = surface_null_result["surface_classifier_accuracy"]
print("  Surface classifier accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print()

# ── Step 3: Phase gate ────────────────────────────────────────────────────────

print("[Step 3] Checking phase gate...")
check_phase_gate(config)
print("  All gates passed.")
print()

# ── Step 4: Load model and extract activations ────────────────────────────────

print("[Step 4] Loading model and extracting activations...")
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_ID)
model.eval()

layer_activation_sets = extract_activations(config, model)
print("  Extraction complete.")
print()

# ── Step 5: Four-class probe at each layer ────────────────────────────────────

print("[Step 5] Running four-class probe at each layer...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

probe_results_by_layer: dict[int, dict] = {}
for activation_set in layer_activation_sets:
    layer_index = activation_set["layer"]
    layer_activations = np.array(activation_set["activations"])
    layer_labels = activation_set["labels"]

    probe_result = run_linear_probe(layer_activations, layer_labels, config)
    probe_result["layer"] = layer_index
    save_result(probe_result, RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json"))
    probe_results_by_layer[layer_index] = probe_result

peak_probe_layer = max(
    probe_results_by_layer,
    key=lambda layer_index: probe_results_by_layer[layer_index]["accuracy_mean"]
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
print("  Four-class probe complete. Peak layer: " + str(peak_probe_layer))
print()

# ── Step 6: Binary identification probe at peak layer ────────────────────────

print("[Step 6] Running binary identification probe at peak layer " + str(peak_probe_layer) + "...")
peak_activation_set = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
peak_activations = np.array(peak_activation_set["activations"])
peak_labels = peak_activation_set["labels"]

identification_probe_result = run_identification_probe(peak_activations, peak_labels, config)
identification_probe_result["layer"] = peak_probe_layer
save_result(identification_probe_result, RESULTS_DIR / "identification_probe.json")

identification_accuracy = identification_probe_result["accuracy_mean"]
identification_criterion_met = identification_accuracy > 0.70
print("  Identification probe accuracy : " + str(round(identification_accuracy * 100, 1)) + "%")
print("  Criterion met (> 70%)         : " + str(identification_criterion_met))
print()

# ── Step 7: Layer sweep (L3) ──────────────────────────────────────────────────

print("[Step 7] Running layer sweep (adjustable -> not_adjustable direction)...")

adjustable_label_set = {"back_door_adjustable", "front_door_adjustable"}
adjustable_indices = [
    i for i, label in enumerate(layer_activation_sets[0]["labels"])
    if label in adjustable_label_set
]

mean_adjustable_activation_by_layer: dict[int, np.ndarray] = {
    activation_set["layer"]: np.array(activation_set["activations"])[adjustable_indices].mean(axis=0)
    for activation_set in layer_activation_sets
}

not_adjustable_sentences: list[str] = []
with VALIDATED_PATH.open("r") as validated_file:
    for raw_line in validated_file:
        stripped = raw_line.strip()
        if stripped:
            pair = json.loads(stripped)
            if pair.get("label_b") == "confounded_not_adjustable":
                not_adjustable_sentences.append(pair["sentence_b"])

target_run_config = {"stimulus": not_adjustable_sentences[-1]}

sweep_result = run_layer_sweep(
    mean_adjustable_activation_by_layer,
    target_run_config,
    config.layer_range,
    config.component,
    config.token_positions[0],
    model,
)
save_result(sweep_result, RESULTS_DIR / "layer_sweep.json")

peak_patch_layer = sweep_result["peak_layer"]
peak_patch_kl = sweep_result["layer_effects"][peak_patch_layer]
layers_agree = peak_probe_layer == peak_patch_layer

with torch.no_grad():
    baseline_logits = model(target_run_config["stimulus"])[0, -1, :].tolist()

mean_ablation_result = mean_ablate(
    peak_activations, target_run_config, peak_patch_layer,
    config.component, config.token_positions[0], model,
    baseline_logits=baseline_logits,
)
assert_specificity_valid(
    peak_patch_kl, mean_ablation_result["kl_from_baseline"] or 0.0, peak_patch_layer
)

print("  Peak patching layer : " + str(peak_patch_layer) + "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
print("  L2 / L3 agreement   : " + ("YES" if layers_agree else "NO"))
print()

# ── Step 8: Summary ───────────────────────────────────────────────────────────

summary: dict = {
    "experiment_id": config.experiment_id,
    "thread_id": config.thread_id,
    "model_id": config.model_id,
    "run_timestamp": datetime.datetime.utcnow().isoformat(),
    "t1b_pearl_confirmed": pearl_confirmed_in_t1b,
    "peak_probe_layer": peak_probe_layer,
    "peak_probe_accuracy_4class": float(peak_probe_accuracy),
    "identification_probe_accuracy": float(identification_accuracy),
    "identification_criterion_met": identification_criterion_met,
    "surface_null_accuracy": float(surface_null_accuracy),
    "peak_patch_layer": peak_patch_layer,
    "peak_patch_kl": float(peak_patch_kl),
    "layers_agree": layers_agree,
    "expected_outcomes": config.expected_outcomes,
}

save_result(summary, SUMMARY_PATH)

print("=" * 60)
print("T1d Results — Causal Identification")
print("=" * 60)
print()
print("Identification probe (adjustable vs not): " +
      str(round(identification_accuracy * 100, 1)) + "%  " +
      ("ENCODES IDENTIFIABILITY" if identification_criterion_met else "NO IDENTIFIABILITY REPRESENTATION"))
print("T1b context: " + ("Pearl mechanism" if pearl_confirmed_in_t1b else "Lewis mechanism"))
print()
if pearl_confirmed_in_t1b and identification_criterion_met:
    print("Finding: Pearl-consistent model WITH identifiability encoding. Full do-calculus.")
elif pearl_confirmed_in_t1b and not identification_criterion_met:
    print("Finding: Pearl-consistent model WITHOUT identifiability. Partial do-calculus only.")
elif not pearl_confirmed_in_t1b and not identification_criterion_met:
    print("Finding: Lewis-consistent model fails identification test. Consistent with worlds-ordering.")
else:
    print("Finding: Lewis-consistent model passes identification test. Unexpected — investigate.")
