"""
experiments/t1c/run_experiment.py — T1c mechanistic experiment: Lewis vs Stalnaker.

Within the worlds-based camp established by T1b, T1c asks the finer question:
does GPT-2's counterfactual geometry match Lewis (similarity-set selection) or
Stalnaker (single closest-world selection)?

Three conditions:
  clear_case  — canonical counterfactual; Lewis and Stalnaker agree (positive control)
  tie_case    — symmetric nearest worlds; Lewis: indeterminate; Stalnaker: determinate
  near_miss   — worlds differing by one variable; tests fine-grained similarity ordering

Discriminating criterion (DISPERSION, not separability — a probe separates clear
from tie under BOTH theories):
  participation-ratio of tie_case / clear_case activations, bootstrap 95% CI vs
  the null ratio = 1:
    CI entirely above 1 (and above the layer-0 lexical baseline) → Lewis
      (tie cloud diffuse — indeterminacy encoded)
    CI brackets 1 → Stalnaker (cannot reject equal dispersion — tie collapses to
      a clear-like centroid)

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

THREAD_ID            = "t1c_gpt2"
MODEL_ID             = "gpt2-medium"
GPT2_MEDIUM_N_LAYERS = 24

VALIDATED_PATH    = PROJECT_ROOT / "stimuli" / "validated" / "t1c" / "pairs.validated.jsonl"
RESULTS_DIR       = PROJECT_ROOT / "experiments" / THREAD_ID / "results"
CONFIG_PATH       = PROJECT_ROOT / "experiments" / THREAD_ID / "config.json"
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH      = RESULTS_DIR / "summary.json"

# T1c requires T1b to have run — its interpretation is conditioned on T1b outcome.
T1B_PREREQUISITE_ID = "t1b_gpt2"


# ── Guards ────────────────────────────────────────────────────────────────────

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Expected: " + str(VALIDATED_PATH))
    print("  Run scripts/run_validation.py --thread t1c_gpt2 first.")
    sys.exit(1)

t1b_summary_path = PROJECT_ROOT / "experiments" / "t1b_gpt2" / "results" / "summary.json"
if not t1b_summary_path.exists():
    print("ERROR: T1b summary not found.")
    print("  Expected: " + str(t1b_summary_path))
    print("  Run experiments/t1b_gpt2/run_experiment.py first.")
    sys.exit(1)


# ── Imports ───────────────────────────────────────────────────────────────────

from extraction.extractor import compute_sha256, extract_activations
from experiments.config import ExperimentConfig
from experiments.run import run_surface_null, check_phase_gate
from stimuli.pipeline import verify_stimulus_file_frequency_matched
from probes.probes import run_linear_probe, run_dispersion_analysis
from interventions.interventions import (
    run_layer_sweep_multi_target, assert_specificity_valid, norm_matched_control_kl,
)
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
        "DISPERSION ratio (not separability). participation-ratio of tie_case "
        "activations / clear_case at the peak layer, bootstrap 95% CI vs null ratio=1. "
        "CI entirely above 1.0 (and above the layer-0 lexical baseline) → Lewis (tie "
        "cloud diffuse — indeterminacy encoded). CI brackets 1.0 → Stalnaker (cannot "
        "reject equal dispersion — tie collapses to a clear-like centroid). No hand-set "
        "threshold. A probe separating clear from tie is NOT the test — it succeeds "
        "under both theories."
    ),
    "outcome_if_lewis": (
        "tie_case activations are MORE dispersed than clear_case (participation-ratio "
        "ratio > 1, CI excludes 1). The model represents the symmetric-worlds "
        "indeterminacy Lewis predicts — no unique closest world is selected, so the "
        "tie representation spreads over the equally-close set."
    ),
    "outcome_if_stalnaker": (
        "tie_case activations are NO more dispersed than clear_case (ratio ≈ 1). "
        "Consistent with Stalnaker's Limit Assumption: a unique closest world is "
        "always selected, so ties collapse to a single centroid like determinate cases."
    ),
    "near_miss_prediction": (
        "near_miss is a secondary similarity-ordering condition. Under Lewis its "
        "dispersion sits between clear and tie; under Stalnaker it collapses toward "
        "clear. Reported, not part of the primary criterion."
    ),
    "probe_role": (
        "The clear-vs-tie linear probe is a SANITY CHECK only: it confirms the two "
        "conditions are distinguishable at all. It does not discriminate Lewis from "
        "Stalnaker — both predict distinguishable sentence-types."
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


# ── Step 5: 3-class probe at each layer (SANITY CHECK + peak-layer locator) ───
# The probe is NOT the Lewis/Stalnaker discriminator — separability holds under
# both theories. It serves two secondary purposes: (a) confirm the conditions
# are distinguishable at all, and (b) locate the peak layer where the distinction
# is most represented, used as the reference layer for the dispersion test.

print("[Step 5] Running 3-class probe at each layer (sanity check + peak-layer locator)...")
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

# Exclude layer 0 (raw token + positional embeddings) from peak selection so the
# reference layer reflects computed structure, not a lexical/positional surface
# confound. Select on balanced accuracy (near_miss is a smaller class).
candidate_layers = [layer_index for layer_index in probe_results_by_layer if layer_index != 0]
peak_probe_layer   = max(
    candidate_layers,
    key=lambda layer_index: probe_results_by_layer[layer_index]["balanced_accuracy_mean"],
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
chance_baseline     = probe_results_by_layer[peak_probe_layer]["chance_baseline"]

print("  3-class probe complete. Peak layer: " + str(peak_probe_layer))
print()


# ── Step 6: Dispersion analysis at peak layer (THE Lewis/Stalnaker test) ──────
# Lewis vs Stalnaker is a DISPERSION question, not a separability one. We compare
# the spread of the tie_case cloud to the clear_case cloud. Participation ratio
# (effective dimensionality) is the discriminating measure; total variance and
# median pairwise distance are reported alongside for robustness.

print("[Step 6] Running dispersion analysis at peak layer " + str(peak_probe_layer) + " (Lewis/Stalnaker test)...")

peak_activation_set  = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
peak_all_activations = np.array(peak_activation_set["activations"])
peak_all_labels      = peak_activation_set["labels"]
peak_all_pair_ids    = peak_activation_set["pair_group_ids"]

dispersion_result = run_dispersion_analysis(
    peak_all_activations, peak_all_labels,
    label_clear="clear_case", label_tie="tie_case",
    seed=config.seed,
)
save_result(dispersion_result, RESULTS_DIR / "dispersion_analysis.json")

participation_ratio_value = dispersion_result["dispersion_ratios"]["participation_ratio"]
participation_ci          = dispersion_result["dispersion_ratio_ci_95"]["participation_ratio"]
print("  Dispersion ratio tie/clear (participation ratio) : " +
      str(round(participation_ratio_value, 3)) +
      "  95% CI [" + str(round(participation_ci[0], 3)) + ", " + str(round(participation_ci[1], 3)) + "]")
print("    total_variance ratio       : " + str(round(dispersion_result["dispersion_ratios"]["total_variance"], 3)))
print("    median_pairwise_dist ratio : " + str(round(dispersion_result["dispersion_ratios"]["median_pairwise_dist"], 3)))
print("    matched n per class        : " + str(dispersion_result["n_per_class"]))
print()

# Layer-0 lexical baseline: layer 0 is raw token + positional embeddings, so any
# tie/clear dispersion gap THERE is purely lexical (the symmetric vs asymmetric
# adjectives differ as words), not a worlds computation. A genuine Lewis effect
# must EXCEED this baseline — the peak-layer CI lower bound has to clear the
# layer-0 ratio, else the dispersion is just the adjective embeddings.
layer_0_activation_set = next(s for s in layer_activation_sets if s["layer"] == 0)
layer_0_dispersion = run_dispersion_analysis(
    np.array(layer_0_activation_set["activations"]), layer_0_activation_set["labels"],
    label_clear="clear_case", label_tie="tie_case", seed=config.seed,
)
layer_0_participation_ratio = layer_0_dispersion["dispersion_ratios"]["participation_ratio"]
save_result(layer_0_dispersion, RESULTS_DIR / "dispersion_layer0_baseline.json")
dispersion_exceeds_lexical_baseline = bool(participation_ci[0] > layer_0_participation_ratio)
print("  Layer-0 lexical baseline ratio : " + str(round(layer_0_participation_ratio, 3)) +
      "   (peak CI lower must exceed this: " + str(dispersion_exceeds_lexical_baseline) + ")")
print()


# Secondary sanity check: are clear and tie even distinguishable? (Both theories
# predict YES — this is not the discriminator, just a floor check.)
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
    "role":               "sanity_check_only — not the Lewis/Stalnaker discriminator",
}
save_result(pairwise_results, RESULTS_DIR / "pairwise_probe_results.json")

print("  Sanity-check probe (distinguishable at all?):")
print("    clear_case vs tie_case   : " + str(round(clear_vs_tie * 100, 1)) + "%")
print("    clear_case vs near_miss  : " + str(round(clear_vs_near_miss * 100, 1)) + "%")
print("    tie_case vs near_miss    : " + str(round(tie_vs_near_miss * 100, 1)) + "%")
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

# Targets: every tie_case sentence (always the sentence_b member of a
# (clear_case, tie_case) pair). Multi-target sweep with a bootstrap CI over targets.
tie_case_sentences: list[str] = []
with VALIDATED_PATH.open("r") as validated_jsonl_file:
    for raw_line in validated_jsonl_file:
        stripped_line = raw_line.strip()
        if stripped_line:
            stimulus_pair = json.loads(stripped_line)
            if stimulus_pair.get("label_b") == "tie_case":
                tie_case_sentences.append(stimulus_pair["sentence_b"])

sweep_result = run_layer_sweep_multi_target(
    mean_clear_case_by_layer,
    tie_case_sentences,
    config.layer_range,
    config.component,
    config.token_positions[0],
    model,
    seed=config.seed,
)
save_result(sweep_result, RESULTS_DIR / "layer_sweep.json")

peak_patch_layer = sweep_result["peak_layer"]
peak_patch_kl    = sweep_result["mean_kl_by_layer"][peak_patch_layer]
peak_patch_kl_ci = sweep_result["kl_ci_95_by_layer"][peak_patch_layer]

# Specificity (V2): clear-mean patch vs norm-matched random directions at peak layer.
control_result = norm_matched_control_kl(
    mean_clear_case_by_layer[peak_patch_layer], tie_case_sentences,
    peak_patch_layer, config.component, config.token_positions[0], model,
    seed=config.seed,
)
assert_specificity_valid(peak_patch_kl, control_result["control_kl_p95"], peak_patch_layer)

print("  Peak patching layer : " + str(peak_patch_layer)
      + "  (mean KL=" + str(round(peak_patch_kl, 4))
      + ", 95% CI [" + str(round(peak_patch_kl_ci[0], 4)) + ", " + str(round(peak_patch_kl_ci[1], 4)) + "]"
      + ", n_targets=" + str(sweep_result["n_targets"]) + ")")
print("  Norm-matched control KL : " + str(round(control_result["mean_control_kl"], 4)))
print()


# ── Step 8: Determine Lewis vs Stalnaker (from DISPERSION, not the probe) ──────

# Threshold-free verdict from run_dispersion_analysis: CI entirely above 1 → Lewis;
# CI brackets 1 → Stalnaker; CI entirely below 1 → inconclusive. Lewis ADDITIONALLY
# requires the CI lower to exceed the layer-0 lexical baseline — otherwise a ratio
# above 1 is just the symmetric/asymmetric adjective embeddings, not a worlds effect.
lewis_confirmed     = dispersion_result["lewis_confirmed"] and dispersion_exceeds_lexical_baseline
stalnaker_confirmed = dispersion_result["stalnaker_confirmed"]
layers_agree        = peak_probe_layer == peak_patch_layer

if lewis_confirmed:
    mechanism_label          = "LEWIS (similarity-set, indeterminacy at ties)"
    mechanism_interpretation = expected_outcomes["outcome_if_lewis"]
elif stalnaker_confirmed:
    mechanism_label          = "STALNAKER (single-selection, Limit Assumption)"
    mechanism_interpretation = expected_outcomes["outcome_if_stalnaker"]
elif dispersion_result["lewis_confirmed"] and not dispersion_exceeds_lexical_baseline:
    mechanism_label          = "INCONCLUSIVE (dispersion ratio > 1 but not above the layer-0 lexical baseline)"
    mechanism_interpretation = (
        "The tie cloud is more dispersed than clear, but not by more than the layer-0 "
        "lexical baseline — the gap is attributable to the adjective embeddings, not a "
        "worlds computation. Not a valid Lewis signal."
    )
else:
    mechanism_label          = "INCONCLUSIVE (dispersion ratio CI lies below 1)"
    mechanism_interpretation = (
        "The tie cloud is LESS dispersed than clear (CI below 1) — neither the Lewis "
        "diffusion nor the Stalnaker equal-dispersion prediction. The geometry does not "
        "pre-specify a verdict."
    )

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
    "dispersion_participation_ratio":     float(participation_ratio_value),
    "dispersion_participation_ratio_ci":  [float(participation_ci[0]), float(participation_ci[1])],
    "dispersion_total_variance_ratio":    float(dispersion_result["dispersion_ratios"]["total_variance"]),
    "dispersion_median_dist_ratio":       float(dispersion_result["dispersion_ratios"]["median_pairwise_dist"]),
    "dispersion_n_per_class":             int(dispersion_result["n_per_class"]),
    "dispersion_layer0_baseline_ratio":   float(layer_0_participation_ratio),
    "dispersion_exceeds_lexical_baseline": dispersion_exceeds_lexical_baseline,
    "sanity_clear_vs_tie_probe":   float(clear_vs_tie),
    "sanity_clear_vs_near_miss":   float(clear_vs_near_miss),
    "sanity_tie_vs_near_miss":     float(tie_vs_near_miss),
    "peak_patch_layer":            peak_patch_layer,
    "peak_patch_kl_mean":          float(peak_patch_kl),
    "peak_patch_kl_ci_95":         [float(peak_patch_kl_ci[0]), float(peak_patch_kl_ci[1])],
    "peak_patch_n_targets":        int(sweep_result["n_targets"]),
    "norm_matched_control_kl":     float(control_result["mean_control_kl"]),
    "layers_agree":                layers_agree,
    "lewis_confirmed":             lewis_confirmed,
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
print("DISPERSION test at peak layer " + str(peak_probe_layer) + " (tie/clear ratio):")
print("  participation ratio  : " + str(round(participation_ratio_value, 3)) +
      "  95% CI [" + str(round(participation_ci[0], 3)) + ", " + str(round(participation_ci[1], 3)) + "]" +
      "   (CI entirely > 1 & > layer-0 baseline = Lewis; CI brackets 1 = Stalnaker)")
print("  total variance       : " + str(round(dispersion_result["dispersion_ratios"]["total_variance"], 3)))
print("  median pairwise dist : " + str(round(dispersion_result["dispersion_ratios"]["median_pairwise_dist"], 3)))
print("  matched n per class  : " + str(dispersion_result["n_per_class"]))
print()
print("Sanity-check probe (NOT the discriminator):")
print("  clear_case vs tie_case   : " + str(round(clear_vs_tie * 100, 1)) + "%  (expect distinguishable under both theories)")
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
print("MECHANISM: " + mechanism_label)
print("=" * 60)
print()
print("Interpretation:")
print("  " + mechanism_interpretation)
