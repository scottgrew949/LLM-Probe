"""
experiments/t2c/run_experiment.py — T2c mechanistic experiment: two-dimensional semantics.

Tests whether Llama 3.2 3B encodes the primary/secondary intension distinction
from Chalmers' two-dimensional framework. Runs in Phase 7 after T2b confirms
hyperintensionality on Llama 3.2 3B.

Prerequisite: T2b must have passed behavioral gate on Llama 3.2 3B (V16).

Usage (Colab — Phase 7 only):
    !python experiments/t2c/run_experiment.py --intension_type primary
    !python experiments/t2c/run_experiment.py --intension_type secondary
    !python experiments/t2c/run_experiment.py --intension_type dissociation
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description="T2c two-dimensional semantics experiment.")
parser.add_argument(
    "--intension_type",
    required=True,
    choices=["primary", "secondary", "dissociation"],
    help="Which intension condition to run.",
)
args = parser.parse_args()

THREAD_ID = "t2c"
MODEL_ID = "meta-llama/Llama-3.2-3B"
LLAMA_3B_N_LAYERS = 28

VALIDATED_PATH = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / THREAD_ID / "results" / args.intension_type
CONFIG_PATH = PROJECT_ROOT / "experiments" / THREAD_ID / ("config_" + args.intension_type + ".json")
SUMMARY_PATH = RESULTS_DIR / "summary.json"

T2B_PREREQUISITE_ID = "t2b_llama"

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Run scripts/run_validation.py --thread t2c first.")
    sys.exit(1)

t2b_summary_path = PROJECT_ROOT / "experiments" / T2B_PREREQUISITE_ID / "results" / "summary.json"
if not t2b_summary_path.exists():
    print("ERROR: T2b Llama summary not found.")
    print("  Run T2b on Llama 3.2 3B first.")
    sys.exit(1)

from extraction.extractor import compute_sha256, extract_activations
from experiments.config import ExperimentConfig
from experiments.run import run_surface_null, check_phase_gate
from stimuli.pipeline import verify_stimulus_file_frequency_matched
from probes.probes import run_linear_probe
from interventions.interventions import run_layer_sweep, assert_specificity_valid, mean_ablate
from core.io import load_result, save_result
import torch

print("=" * 60)
print("PoL-Probe — T2c Experiment: Two-Dimensional Semantics")
print("Intension type: " + args.intension_type)
print("=" * 60)
print()

t2b_summary = load_result(t2b_summary_path)
t2b_hyperintensional = t2b_summary.get("hyperintensional_finding", False)
print("[Context] T2b hyperintensional = " + str(t2b_hyperintensional))
print()

expected_outcomes_by_intension_type: dict[str, dict] = {
    "primary": {
        "prediction": (
            "Representations of primary_sensitive pairs diverge from secondary_necessary. "
            "Model encodes how reference is fixed, not just what it picks out. "
            "Primary intension is a distinct geometric region."
        ),
    },
    "secondary": {
        "prediction": (
            "Representations of secondary_necessary pairs converge — model encodes "
            "rigid reference across worlds. Metaphysical necessity produces compact cluster."
        ),
    },
    "dissociation": {
        "prediction": (
            "Representations of primary_secondary_dissociation pairs form a third "
            "distinct cluster — neither purely primary nor secondary. "
            "The two-dimensional split is geometrically encoded."
        ),
    },
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t2c_llama3b_" + args.intension_type + "_" + date_stamp,
    thread_id=THREAD_ID,
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, LLAMA_3B_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=verify_stimulus_file_frequency_matched(VALIDATED_PATH),
    expected_outcomes=expected_outcomes_by_intension_type[args.intension_type],
    prerequisite_experiment_id=T2B_PREREQUISITE_ID,
    intension_type=args.intension_type,
)

config.lock()
config.to_json(CONFIG_PATH)
print("  Config locked for intension_type=" + args.intension_type)
print()

surface_null_result = run_surface_null(config)
surface_null_accuracy = surface_null_result["surface_classifier_accuracy"]
print("[Surface null] accuracy=" + str(round(surface_null_accuracy * 100, 1)) + "%")
print()

check_phase_gate(config)
print("[Phase gate] Passed.")
print()

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_ID)
model.eval()

layer_activation_sets = extract_activations(config, model)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
probe_results_by_layer: dict[int, dict] = {}
for activation_set in layer_activation_sets:
    layer_index = activation_set["layer"]
    layer_activations = np.array(activation_set["activations"])
    layer_labels = activation_set["labels"]
    # pair_group_ids keeps both sentences of a minimal pair in one fold (S4).
    probe_result = run_linear_probe(
        layer_activations, layer_labels, config, pair_ids=activation_set["pair_group_ids"]
    )
    probe_result["layer"] = layer_index
    save_result(probe_result, RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json"))
    probe_results_by_layer[layer_index] = probe_result

peak_probe_layer = max(
    probe_results_by_layer,
    key=lambda layer_index: probe_results_by_layer[layer_index]["accuracy_mean"]
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
print("[L2 probe] Peak layer=" + str(peak_probe_layer) +
      "  accuracy=" + str(round(peak_probe_accuracy * 100, 1)) + "%")

summary: dict = {
    "experiment_id": config.experiment_id,
    "thread_id": config.thread_id,
    "intension_type": args.intension_type,
    "model_id": config.model_id,
    "run_timestamp": datetime.datetime.utcnow().isoformat(),
    "t2b_hyperintensional": t2b_hyperintensional,
    "peak_probe_layer": peak_probe_layer,
    "peak_probe_accuracy": float(peak_probe_accuracy),
    "surface_null_accuracy": float(surface_null_accuracy),
    "expected_outcomes": config.expected_outcomes,
}
save_result(summary, SUMMARY_PATH)

print()
print("=" * 60)
print("T2c complete — intension_type=" + args.intension_type)
print("Peak probe accuracy: " + str(round(peak_probe_accuracy * 100, 1)) + "%")
print("=" * 60)
