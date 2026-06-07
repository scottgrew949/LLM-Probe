"""
experiments/t1c/run_experiment.py — T1c mechanistic experiment: Lewis vs Stalnaker.

Within the worlds-based camp established by T1b, T1c asks the finer question:
does GPT-2's counterfactual geometry match Lewis (similarity-set selection) or
Stalnaker (single closest-world selection)?

Three conditions:
  clear_case  — canonical counterfactual; Lewis and Stalnaker agree (positive control)
  tie_case    — symmetric nearest worlds; Lewis: indeterminate; Stalnaker: determinate
  near_miss   — worlds differing by one variable; tests fine-grained similarity ordering

Discriminating criterion:
  clear_case vs tie_case pairwise accuracy > 0.70
    → model geometrically separates them → Lewis-consistent (indeterminacy encoded)
  <= 0.70
    → model treats them equivalently → Stalnaker-consistent (all determinate)

T1c is most interpretable when T1b confirmed worlds-based (not Pearl), but runs
regardless and records the T1b context in its summary.

Prerequisite: experiments/t1b/run_experiment.py must have run first.

Usage (Colab):
    !python experiments/t1c/run_experiment.py
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

THREAD_ID            = "t1c"
MODEL_ID             = "gpt2-medium"
GPT2_MEDIUM_N_LAYERS = 24

VALIDATED_PATH    = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
RESULTS_DIR       = PROJECT_ROOT / "experiments" / THREAD_ID / "results"
CONFIG_PATH       = PROJECT_ROOT / "experiments" / THREAD_ID / "config.json"
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH      = RESULTS_DIR / "summary.json"

# T1c requires T1b to have run — its interpretation is conditioned on T1b outcome.
T1B_PREREQUISITE_ID = "t1b"


# ── Guards ────────────────────────────────────────────────────────────────────

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Expected: " + str(VALIDATED_PATH))
    print("  Run scripts/run_validation.py --thread t1c first.")
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
from stimuli.pipeline import verify_stimulus_file_frequency_matched
from probes.probes import run_linear_probe
from interventions.interventions import run_layer_sweep, assert_specificity_valid, mean_ablate
from core.io import load_result, save_result
import torch


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T1c Mechanistic Experiment")
print("Lewis Similarity-Set vs Stalnaker Single-Selection")
print("=" * 60)
print()

# Load T1b context — T1c interpretation is conditioned on it
t1b_summary = load_result(t1b_summary_path)
pearl_confirmed_in_t1b = t1b_summary.get("pearl_confirmed", False)
print("[Context] T1b pearl_confirmed = " + str(pearl_confirmed_in_t1b))
if pearl_confirmed_in_t1b:
    print("  T1b found Pearl mechanism. T1c within-worlds test is less central,")
    print("  but still probes whether worlds-structure is partially encoded.")
else:
    print("  T1b found Lewis/Stalnaker mechanism. T1c discriminates between them.")
print()


# ── Step 1: Build config ──────────────────────────────────────────────────────

print("[Step 1] Building and locking experiment config...")

expected_outcomes: dict = {
    "lewis_vs_stalnaker_criterion": (
        "Pairwise clear_case vs tie_case probe accuracy at peak layer. "
        "> 0.70 → Lewis (model encodes indeterminacy — tie_case is geometrically distinct). "
        "<= 0.70 → Stalnaker (model treats tie_case as determinate — same cluster as clear_case)."
    ),
    "outcome_if_lewis": (
        "clear_case and tie_case are linearly separable. "
        "Model represents the symmetric-worlds indeterminacy Lewis predicts — "
        "tie_case activations occupy a distinct geometric region, reflecting "
        "that no unique closest world is selected."
    ),
    "outcome_if_stalnaker": (
        "clear_case and tie_case are NOT linearly separable. "
        "Model treats both as determinate counterfactuals — "
        "consistent with Stalnaker's Limit Assumption: every counterfactual "
        "has a unique closest world, so tie cases resolve to one centroid."
    ),
    "near_miss_prediction": (
        "near_miss cases test fine-grained similarity ordering. "
        "Under Lewis, near_miss representations track proximity metric — "
        "they should cluster between clear_case and tie_case. "
        "Under Stalnaker, near_miss collapses to clear_case (determinate, unique world)."
    ),
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t1c_gpt2m_" + date_stamp,
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
    prerequisite_experiment_id=T1B_PREREQUISITE_ID,
)

config.lock()
config.to_json(CONFIG_PATH)

print("  Experiment ID        : " + config.experiment_id)
print("  Prerequisite (T1b)   : " + T1B_PREREQUISITE_ID)
print("  Three conditions     : clear_case, tie_case, near_miss")
print("  Config locked        : " + str(config.pre_spec_locked))
print()


# ── Step 2: Surface null ──────────────────────────────────────────────────────

print("[Step 2] Running surface-statistics null...")

surface_null_result  = run_surface_null(config)
surface_null_accuracy = surface_null_result["surface_classifier_accuracy"]

print("  Surface classifier accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print("  Written to: " + str(SURFACE_NULL_PATH))
print()


# ── Step 3: Phase gate ────────────────────────────────────────────────────────

print("[Step 3] Checking phase gate...")
check_phase_gate(config)
print("  All gates passed (T1b prerequisite verified).")
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


# ── Step 5: 3-class probe at each layer ──────────────────────────────────────
# All three conditions run together — peak layer localises where the
# Lewis/Stalnaker geometry is most distinct.

print("[Step 5] Running 3-class probe at each layer (clear_case, tie_case, near_miss)...")
print()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

probe_results_by_layer: dict[int, dict] = {}

for activation_set in layer_activation_sets:
    layer_index    = activation_set["layer"]
    layer_activations = np.array(activation_set["activations"])
    layer_labels   = activation_set["labels"]

    # pair_group_ids keeps both sentences of a minimal pair in one fold (S4).
    probe_result = run_linear_probe(
        layer_activations, layer_labels, config,
        pair_ids=activation_set["pair_group_ids"],
    )
    probe_result["layer"] = layer_index
    probe_result["token_position"] = activation_set["token_position"]

    save_result(probe_result, RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json"))
    probe_results_by_layer[layer_index] = probe_result

peak_probe_layer   = max(
    probe_results_by_layer,
    key=lambda layer_index: probe_results_by_layer[layer_index]["accuracy_mean"],
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
chance_baseline     = probe_results_by_layer[peak_probe_layer]["chance_baseline"]

print("  3-class probe complete. Peak layer: " + str(peak_probe_layer))
print()


# ── Step 6: Pairwise probes at peak layer ─────────────────────────────────────
# clear_case vs tie_case is the primary Lewis/Stalnaker discriminator.

print("[Step 6] Running pairwise probes at peak layer " + str(peak_probe_layer) + "...")

peak_activation_set  = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
peak_all_activations = np.array(peak_activation_set["activations"])
peak_all_labels      = peak_activation_set["labels"]
peak_all_pair_ids    = peak_activation_set["pair_group_ids"]


def pairwise_probe_accuracy(label_a: str, label_b: str) -> float:
    """Train binary probe on the two specified conditions, return mean CV accuracy."""
    condition_membership_mask = [
        condition_label in (label_a, label_b) for condition_label in peak_all_labels
    ]
    filtered_activations = peak_all_activations[condition_membership_mask]
    filtered_labels = [
        condition_label for condition_label in peak_all_labels
        if condition_label in (label_a, label_b)
    ]
    # Group ids filtered in lockstep so a pair stays in one fold (S4).
    filtered_pair_group_ids = [
        pair_group_id for pair_group_id, keep in zip(peak_all_pair_ids, condition_membership_mask)
        if keep
    ]
    if len(set(filtered_labels)) < 2:
        return 0.5
    probe_result = run_linear_probe(
        filtered_activations, filtered_labels, config,
        pair_ids=filtered_pair_group_ids,
    )
    return probe_result["accuracy_mean"]


clear_vs_tie        = pairwise_probe_accuracy("clear_case", "tie_case")
clear_vs_near_miss  = pairwise_probe_accuracy("clear_case", "near_miss")
tie_vs_near_miss    = pairwise_probe_accuracy("tie_case", "near_miss")

pairwise_results = {
    "clear_vs_tie":       clear_vs_tie,
    "clear_vs_near_miss": clear_vs_near_miss,
    "tie_vs_near_miss":   tie_vs_near_miss,
    "peak_layer":         peak_probe_layer,
}
save_result(pairwise_results, RESULTS_DIR / "pairwise_probe_results.json")

print("  clear_case vs tie_case   : " + str(round(clear_vs_tie * 100, 1)) + "%  (THE Lewis/Stalnaker test)")
print("  clear_case vs near_miss  : " + str(round(clear_vs_near_miss * 100, 1)) + "%")
print("  tie_case vs near_miss    : " + str(round(tie_vs_near_miss * 100, 1)) + "%")
print()


# ── Step 7: Layer sweep (L3 patching) ─────────────────────────────────────────
# Source: mean clear_case activations. Target: a tie_case sentence.
# Patching clear_case → tie_case tests whether clear-case representations
# causally drive the output — L3 causal verification.

print("[Step 7] Running layer sweep (L3 patching — clear_case into tie_case)...")

clear_case_indices = [
    i for i, condition_label in enumerate(layer_activation_sets[0]["labels"])
    if condition_label == "clear_case"
]

mean_clear_case_by_layer: dict[int, np.ndarray] = {
    layer_activation_bundle["layer"]: np.array(layer_activation_bundle["activations"])[clear_case_indices].mean(axis=0)
    for layer_activation_bundle in layer_activation_sets
}

# Use last tie_case sentence as patching target
tie_case_sentences: list[str] = []
with VALIDATED_PATH.open("r") as validated_jsonl_file:
    for raw_line in validated_jsonl_file:
        stripped_line = raw_line.strip()
        if stripped_line:
            stimulus_pair = json.loads(stripped_line)
            if stimulus_pair.get("label_a") == "tie_case":
                tie_case_sentences.append(stimulus_pair["sentence_a"])

target_run_config = {"stimulus": tie_case_sentences[-1]}

sweep_result = run_layer_sweep(
    mean_clear_case_by_layer,
    target_run_config,
    config.layer_range,
    config.component,
    config.token_positions[0],
    model,
)
save_result(sweep_result, RESULTS_DIR / "layer_sweep.json")

peak_patch_layer = sweep_result["peak_layer"]
peak_patch_kl    = sweep_result["layer_effects"][peak_patch_layer]

# Specificity check (V2)
peak_act_set     = next(s for s in layer_activation_sets if s["layer"] == peak_patch_layer)
peak_activations = np.array(peak_act_set["activations"])

with torch.no_grad():
    baseline_logits = model(target_run_config["stimulus"])[0, -1, :].tolist()

mean_ablation_result = mean_ablate(
    peak_activations, target_run_config, peak_patch_layer,
    config.component, config.token_positions[0], model,
    baseline_logits=baseline_logits,
)
assert_specificity_valid(
    peak_patch_kl, mean_ablation_result["kl_from_baseline"] or 0.0, peak_patch_layer,
)

print("  Peak patching layer : " + str(peak_patch_layer) + "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
print()


# ── Step 8: Determine Lewis vs Stalnaker ──────────────────────────────────────

# Pre-specified criterion (see expected_outcomes above)
stalnaker_confirmed = clear_vs_tie <= 0.70
layers_agree        = peak_probe_layer == peak_patch_layer

if stalnaker_confirmed:
    mechanism_interpretation = expected_outcomes["outcome_if_stalnaker"]
else:
    mechanism_interpretation = expected_outcomes["outcome_if_lewis"]

summary: dict = {
    "experiment_id":               config.experiment_id,
    "thread_id":                   config.thread_id,
    "model_id":                    config.model_id,
    "run_timestamp":               datetime.datetime.utcnow().isoformat(),
    "t1b_pearl_confirmed":         pearl_confirmed_in_t1b,
    "peak_probe_layer":            peak_probe_layer,
    "peak_probe_accuracy_3class":  float(peak_probe_accuracy),
    "chance_baseline":             float(chance_baseline),
    "surface_null_accuracy":       float(surface_null_accuracy),
    "pairwise_clear_vs_tie":       float(clear_vs_tie),
    "pairwise_clear_vs_near_miss": float(clear_vs_near_miss),
    "pairwise_tie_vs_near_miss":   float(tie_vs_near_miss),
    "peak_patch_layer":            peak_patch_layer,
    "peak_patch_kl":               float(peak_patch_kl),
    "layers_agree":                layers_agree,
    "stalnaker_confirmed":         stalnaker_confirmed,
    "mechanism_interpretation":    mechanism_interpretation,
    "expected_outcomes":           config.expected_outcomes,
}

save_result(summary, SUMMARY_PATH)


# ── Step 9: Print results ─────────────────────────────────────────────────────

print("=" * 60)
print("T1c Results — Lewis vs Stalnaker")
print("=" * 60)
print()
print("3-class probe accuracy by layer:")
print()
print("  Layer   Accuracy    Chance")
print("  -----   --------   ------")
for layer_index in range(GPT2_MEDIUM_N_LAYERS):
    layer_probe_result    = probe_results_by_layer[layer_index]
    layer_probe_accuracy  = layer_probe_result["accuracy_mean"]
    layer_chance_baseline = layer_probe_result["chance_baseline"]
    marker = "  <-- PEAK" if layer_index == peak_probe_layer else ""
    print(
        "  " + str(layer_index).rjust(5) +
        "   " + str(round(layer_probe_accuracy * 100, 1)).rjust(6) + "%" +
        "   " + str(round(layer_chance_baseline * 100, 1)).rjust(5) + "%" +
        marker
    )

print()
print("Pairwise probe at peak layer " + str(peak_probe_layer) + ":")
print("  clear_case vs tie_case   : " + str(round(clear_vs_tie * 100, 1)) + "%  (criterion > 70% for Lewis)")
print("  clear_case vs near_miss  : " + str(round(clear_vs_near_miss * 100, 1)) + "%")
print("  tie_case vs near_miss    : " + str(round(tie_vs_near_miss * 100, 1)) + "%")
print()
print("Peak patching layer : Layer " + str(peak_patch_layer) + "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
print("L2 / L3 agreement   : " + ("YES" if layers_agree else "NO"))
print()
print("Surface null accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print()
print("T1b context : " + ("Pearl — T1c probes partial worlds-structure" if pearl_confirmed_in_t1b
                          else "Lewis/Stalnaker — T1c discriminates the selection function"))
print()
print("=" * 60)
print("MECHANISM: " + ("STALNAKER (single-selection, Limit Assumption)"
                       if stalnaker_confirmed else "LEWIS (similarity-set, indeterminacy at ties)"))
print("=" * 60)
print()
print("Interpretation:")
print("  " + mechanism_interpretation)
