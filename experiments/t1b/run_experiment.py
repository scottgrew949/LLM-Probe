"""
experiments/t1b/run_experiment.py — T1b mechanistic experiment: Lewis vs Pearl.

Tests which mechanism GPT-2 medium uses to evaluate counterfactuals:
  - Lewis/Stalnaker: possible-worlds similarity ordering
  - Pearl: do-calculus on a structural causal model

The separating test: can the model geometrically distinguish forward_causal
from backtracking counterfactuals? Same surface grammar — only causal direction
differs. If separable → Pearl. If not → Lewis.

Prerequisite: experiments/t1b/run_validation.py AND experiments/t1a/run_experiment.py
must have run and passed (T1a level3_confirmed=True required).

Usage (Colab):
    !python experiments/t1b/run_experiment.py
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

THREAD_ID  = "t1b"
MODEL_ID   = "gpt2-medium"
GPT2_MEDIUM_N_LAYERS = 24

VALIDATED_PATH    = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
RESULTS_DIR       = PROJECT_ROOT / "experiments" / THREAD_ID / "results"
CONFIG_PATH       = PROJECT_ROOT / "experiments" / THREAD_ID / "config.json"
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH      = RESULTS_DIR / "summary.json"

# T1b requires T1a to have confirmed Level 3 existence.
# "t1a" resolves to experiments/t1a/results/summary.json in check_phase_gate().
T1A_PREREQUISITE_ID = "t1a"


# ── Guards ────────────────────────────────────────────────────────────────────

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Expected: " + str(VALIDATED_PATH))
    print("  Run scripts/run_validation.py --thread t1b first.")
    sys.exit(1)

t1a_summary_path = PROJECT_ROOT / "experiments" / "t1a" / "results" / "summary.json"
if not t1a_summary_path.exists():
    print("ERROR: T1a summary not found.")
    print("  Expected: " + str(t1a_summary_path))
    print("  Run experiments/t1a/run_experiment.py first.")
    sys.exit(1)


# ── Imports ───────────────────────────────────────────────────────────────────

from extraction.extractor import compute_sha256
from experiments.config import ExperimentConfig
from experiments.run import run_surface_null, check_phase_gate
from extraction.extractor import extract_activations
from probes.probes import run_linear_probe
from interventions.interventions import run_layer_sweep, assert_specificity_valid, mean_ablate
from core.io import load_result, save_result
import torch


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T1b Mechanistic Experiment")
print("Lewis/Stalnaker Possible-Worlds vs Pearl Do-Calculus")
print("=" * 60)
print()

# ── Step 1: Build config ──────────────────────────────────────────────────────

print("[Step 1] Building and locking experiment config...")

expected_outcomes = {
    "lewis_vs_pearl_criterion": (
        "Pairwise forward_causal vs backtracking probe accuracy at peak layer. "
        "> 0.70 → Pearl (causal direction encoded — do-calculus). "
        "<= 0.70 → Lewis (direction not encoded — similarity ordering)."
    ),
    "outcome_if_pearl": (
        "forward_causal and backtracking are linearly separable. "
        "Model encodes causal direction in its representations — "
        "consistent with Pearl's do-calculus: intervening on cause vs effect "
        "produces geometrically distinct internal states."
    ),
    "outcome_if_lewis": (
        "forward_causal and backtracking are NOT linearly separable. "
        "Model treats both as structurally equivalent counterfactuals — "
        "consistent with Lewis/Stalnaker: similarity ordering over possible "
        "worlds does not encode causal direction."
    ),
    "common_cause_prediction_lewis": (
        "Under Lewis: common_cause clusters with forward_causal — "
        "both are evaluated by similarity ordering, confound is invisible."
    ),
    "common_cause_prediction_pearl": (
        "Under Pearl: common_cause is distinct from forward_causal — "
        "do-calculus correctly separates genuine causation from confounded correlation."
    ),
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t1b_gpt2m_" + date_stamp,
    thread_id="t1b",
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, GPT2_MEDIUM_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=True,
    expected_outcomes=expected_outcomes,
    prerequisite_experiment_id=T1A_PREREQUISITE_ID,
)

config.lock()
config.to_json(CONFIG_PATH)

print("  Experiment ID        : " + config.experiment_id)
print("  Prerequisite (T1a)   : " + T1A_PREREQUISITE_ID)
print("  Three conditions     : forward_causal, backtracking, common_cause")
print("  Config locked        : " + str(config.pre_spec_locked))
print()

# ── Step 2: Surface null ──────────────────────────────────────────────────────

print("[Step 2] Running surface-statistics null...")

surface_null_result = run_surface_null(config)
surface_null_accuracy = surface_null_result["surface_classifier_accuracy"]

print("  Surface classifier accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print("  Written to: " + str(SURFACE_NULL_PATH))
print()

# ── Step 3: Phase gate ────────────────────────────────────────────────────────
# Checks: V1, V7, V8, V11, V10 (T1a level3_confirmed=True)

print("[Step 3] Checking phase gate...")
check_phase_gate(config)
print("  All gates passed (T1a level3_confirmed verified).")
print()

# ── Step 4: Load model and extract activations ────────────────────────────────

print("[Step 4] Loading model and extracting activations at all 24 layers...")
print("  This takes ~10-15 minutes on Colab T4.")
print()

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_ID)
model.eval()

layer_activation_sets = extract_activations(config, model)
print("  Extraction complete.")
print()

# ── Step 5: Binary probe at each layer (forward_causal vs backtracking) ──────
# Primary Lewis vs Pearl test — 2-class only.
# common_cause items exist in the validated file but are filtered here:
# they serve as an auxiliary analysis point, not the primary probe.

print("[Step 5] Running binary probe at each layer (forward_causal vs backtracking)...")
print()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

probe_results_by_layer: dict[int, dict] = {}

for activation_set in layer_activation_sets:
    layer_index = activation_set["layer"]
    all_activations = np.array(activation_set["activations"])
    all_labels = activation_set["labels"]

    # Filter to forward_causal and backtracking only
    binary_mask = [label in ("forward_causal", "backtracking") for label in all_labels]
    binary_activations = all_activations[binary_mask]
    binary_labels = [label for label, keep in zip(all_labels, binary_mask) if keep]

    probe_result = run_linear_probe(binary_activations, binary_labels, config)
    probe_result["layer"] = layer_index
    probe_result["token_position"] = activation_set["token_position"]

    save_result(probe_result, RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json"))
    probe_results_by_layer[layer_index] = probe_result

peak_probe_layer = max(
    probe_results_by_layer,
    key=lambda layer_index: probe_results_by_layer[layer_index]["accuracy_mean"]
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
chance_baseline = probe_results_by_layer[peak_probe_layer]["chance_baseline"]

print("  Binary probe complete.")
print()

# ── Step 6: Pairwise probes at peak layer ─────────────────────────────────────
# The Lewis vs Pearl test: can model distinguish forward from backtracking?

print("[Step 6] Running pairwise probes at peak layer " + str(peak_probe_layer) + "...")

peak_activation_set = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
all_activations = np.array(peak_activation_set["activations"])
all_labels = peak_activation_set["labels"]

def pairwise_probe_accuracy(label_a: str, label_b: str) -> float:
    """Train binary probe on the two specified conditions, return accuracy."""
    mask = [l in (label_a, label_b) for l in all_labels]
    filtered_activations = all_activations[mask]
    filtered_labels = [l for l in all_labels if l in (label_a, label_b)]
    if len(set(filtered_labels)) < 2:
        return 0.5
    result = run_linear_probe(filtered_activations, filtered_labels, config)
    return result["accuracy_mean"]

forward_vs_backtracking = pairwise_probe_accuracy("forward_causal", "backtracking")
forward_vs_common_cause = pairwise_probe_accuracy("forward_causal", "common_cause")
backtracking_vs_common_cause = pairwise_probe_accuracy("backtracking", "common_cause")

pairwise_results = {
    "forward_vs_backtracking": forward_vs_backtracking,
    "forward_vs_common_cause": forward_vs_common_cause,
    "backtracking_vs_common_cause": backtracking_vs_common_cause,
    "peak_layer": peak_probe_layer,
}
save_result(pairwise_results, RESULTS_DIR / "pairwise_probe_results.json")

print("  forward_causal vs backtracking : " + str(round(forward_vs_backtracking * 100, 1)) + "%  (THE Lewis vs Pearl test)")
print("  forward_causal vs common_cause : " + str(round(forward_vs_common_cause * 100, 1)) + "%")
print("  backtracking vs common_cause   : " + str(round(backtracking_vs_common_cause * 100, 1)) + "%")
print()

# ── Step 7: Layer sweep (L3 patching) ─────────────────────────────────────────
# Source: mean forward_causal activations. Target: a backtracking sentence.

print("[Step 7] Running layer sweep (L3 patching — forward_causal into backtracking)...")

forward_indices = [i for i, l in enumerate(layer_activation_sets[0]["labels"]) if l == "forward_causal"]
backtracking_indices = [i for i, l in enumerate(layer_activation_sets[0]["labels"]) if l == "backtracking"]

mean_forward_by_layer: dict[int, np.ndarray] = {
    act_set["layer"]: np.array(act_set["activations"])[forward_indices].mean(axis=0)
    for act_set in layer_activation_sets
}

# Use last backtracking sentence as target
backtracking_sentences = []
import json
with VALIDATED_PATH.open("r") as f:
    for raw_line in f:
        stripped = raw_line.strip()
        if stripped:
            pair = json.loads(stripped)
            if pair.get("label_b") == "backtracking":
                backtracking_sentences.append(pair["sentence_b"])

target_run_config = {"stimulus": backtracking_sentences[-1]}

sweep_result = run_layer_sweep(
    mean_forward_by_layer,
    target_run_config,
    config.layer_range,
    config.component,
    config.token_positions[0],
    model,
)
save_result(sweep_result, RESULTS_DIR / "layer_sweep.json")

peak_patch_layer = sweep_result["peak_layer"]
peak_patch_kl = sweep_result["layer_effects"][peak_patch_layer]

# Specificity check
peak_act_set = next(s for s in layer_activation_sets if s["layer"] == peak_patch_layer)
peak_activations = np.array(peak_act_set["activations"])

with torch.no_grad():
    baseline_logits = model(target_run_config["stimulus"])[0, -1, :].tolist()

mean_ablation_result = mean_ablate(
    peak_activations, target_run_config, peak_patch_layer,
    config.component, config.token_positions[0], model,
    baseline_logits=baseline_logits,
)
assert_specificity_valid(peak_patch_kl, mean_ablation_result["kl_from_baseline"] or 0.0, peak_patch_layer)

print("  Peak patching layer : " + str(peak_patch_layer) + "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
print()

# ── Step 8: Determine Lewis vs Pearl ─────────────────────────────────────────

# Pre-specified criterion
pearl_confirmed = forward_vs_backtracking > 0.70
layers_agree    = peak_probe_layer == peak_patch_layer

if pearl_confirmed:
    mechanism_interpretation = expected_outcomes["outcome_if_pearl"]
else:
    mechanism_interpretation = expected_outcomes["outcome_if_lewis"]

if forward_vs_common_cause < backtracking_vs_common_cause:
    common_cause_note = expected_outcomes["common_cause_prediction_lewis"]
else:
    common_cause_note = expected_outcomes["common_cause_prediction_pearl"]

summary = {
    "experiment_id": config.experiment_id,
    "thread_id": config.thread_id,
    "model_id": config.model_id,
    "run_timestamp": datetime.datetime.utcnow().isoformat(),
    "peak_probe_layer": peak_probe_layer,
    "peak_probe_accuracy_3class": float(peak_probe_accuracy),
    "chance_baseline": float(chance_baseline),
    "surface_null_accuracy": float(surface_null_accuracy),
    "pairwise_forward_vs_backtracking": float(forward_vs_backtracking),
    "pairwise_forward_vs_common_cause": float(forward_vs_common_cause),
    "pairwise_backtracking_vs_common_cause": float(backtracking_vs_common_cause),
    "peak_patch_layer": peak_patch_layer,
    "peak_patch_kl": float(peak_patch_kl),
    "layers_agree": layers_agree,
    "pearl_confirmed": pearl_confirmed,
    "mechanism_interpretation": mechanism_interpretation,
    "common_cause_interpretation": common_cause_note,
    "expected_outcomes": config.expected_outcomes,
}

save_result(summary, SUMMARY_PATH)

# ── Step 9: Print results ─────────────────────────────────────────────────────

print("=" * 60)
print("T1b Results — Lewis vs Pearl")
print("=" * 60)
print()
print("3-class probe accuracy by layer:")
print()
print("  Layer   Accuracy    Chance")
print("  -----   --------   ------")
for layer_index in range(GPT2_MEDIUM_N_LAYERS):
    probe = probe_results_by_layer[layer_index]
    acc    = probe["accuracy_mean"]
    chance = probe["chance_baseline"]
    marker = "  <-- PEAK" if layer_index == peak_probe_layer else ""
    print(
        "  " + str(layer_index).rjust(5) +
        "   " + str(round(acc * 100, 1)).rjust(6) + "%" +
        "   " + str(round(chance * 100, 1)).rjust(5) + "%" +
        marker
    )

print()
print("Pairwise probe at peak layer " + str(peak_probe_layer) + ":")
print("  forward_causal vs backtracking : " + str(round(forward_vs_backtracking * 100, 1)) + "%  (criterion > 70% for Pearl)")
print("  forward_causal vs common_cause : " + str(round(forward_vs_common_cause * 100, 1)) + "%")
print("  backtracking vs common_cause   : " + str(round(backtracking_vs_common_cause * 100, 1)) + "%")
print()
print("Peak patching layer : Layer " + str(peak_patch_layer) + "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
print("L2 / L3 agreement   : " + ("YES" if layers_agree else "NO"))
print()
print("Surface null accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print()
print("=" * 60)
print("MECHANISM: " + ("PEARL (do-calculus)" if pearl_confirmed else "LEWIS (possible-worlds)"))
print("=" * 60)
print()
print("Interpretation:")
print("  " + mechanism_interpretation)
print()
print("Common-cause position:")
print("  " + common_cause_note)
print()
if pearl_confirmed:
    print("T1c: UNLOCKED — test Lewis vs Stalnaker within worlds-based semantics")
else:
    print("T1c: RELEVANT — Lewis confirmed; T1c can probe similarity structure further")
