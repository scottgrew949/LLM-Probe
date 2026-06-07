"""
experiments/t1b_pythia/run_experiment.py — T1b mechanistic experiment: Lewis vs Pearl on Pythia 1.4B.

Tests which mechanism Pythia 1.4B uses to evaluate counterfactuals:
  - Lewis/Stalnaker: possible-worlds similarity ordering
  - Pearl: do-calculus on a structural causal model

NOTE: Construct under review (2026-06-06). forward_causal vs backtracking may
separate thematic role, not Lewis vs Pearl — Pearl's abduction step also makes
backtracking TRUE, so the old "Pearl=FALSE backtracking" was a do()-vs-counterfactual
category error. Standard Lewis with miracle-weighting also agrees with Pearl on
the common_cause case. A faithful separator requires an explicit interventional (do)
framing. Verdicts from this run are provisional; the summary records the caveat.

Prerequisite: experiments/t1b/run_validation.py AND experiments/t1a_pythia/run_experiment.py
must have run and passed (T1a level3_confirmed=True required).

Usage (Colab):
    !python experiments/t1b_pythia/run_experiment.py
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

THREAD_ID  = "t1b_pythia"
MODEL_ID   = "EleutherAI/pythia-1.4b"
PYTHIA_1_4B_N_LAYERS = 24

VALIDATED_PATH    = PROJECT_ROOT / "stimuli" / "validated" / "t1b" / "pairs.validated.jsonl"
RESULTS_DIR       = PROJECT_ROOT / "experiments" / THREAD_ID / "results"
CONFIG_PATH       = PROJECT_ROOT / "experiments" / THREAD_ID / "config.json"
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH      = RESULTS_DIR / "summary.json"

# T1b requires T1a to have confirmed Level 3 existence.
# "t1a_pythia" resolves to experiments/t1a_pythia/results/summary.json in check_phase_gate().
T1A_PREREQUISITE_ID = "t1a_pythia"


# ── Guards ────────────────────────────────────────────────────────────────────

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Expected: " + str(VALIDATED_PATH))
    print("  Run scripts/run_validation.py --thread t1b first.")
    sys.exit(1)

t1a_summary_path = PROJECT_ROOT / "experiments" / "t1a_pythia" / "results" / "summary.json"
if not t1a_summary_path.exists():
    print("ERROR: T1a summary not found.")
    print("  Expected: " + str(t1a_summary_path))
    print("  Run experiments/t1a_pythia/run_experiment.py first.")
    sys.exit(1)


# ── Imports ───────────────────────────────────────────────────────────────────

from extraction.extractor import compute_sha256
from experiments.config import ExperimentConfig
from experiments.run import run_surface_null, check_phase_gate
from extraction.extractor import extract_activations
from stimuli.pipeline import verify_stimulus_file_frequency_matched
from probes.probes import run_linear_probe, probe_beats_null
from interventions.interventions import (
    run_layer_sweep_multi_target, assert_specificity_valid, norm_matched_control_kl, patch_activation,
)
from core.io import load_result, save_result
import torch


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T1b Mechanistic Experiment (Pythia 1.4B)")
print("Lewis/Stalnaker Possible-Worlds vs Pearl Do-Calculus")
print("=" * 60)
print()

# ── Step 1: Build config ──────────────────────────────────────────────────────

print("[Step 1] Building and locking experiment config...")

expected_outcomes = {
    "lewis_vs_pearl_criterion": (
        "Balanced pairwise accuracy at the peak layer, calibrated against a "
        "shuffled-label null (no fixed cutoff). PEARL needs a POSITIVE signature: "
        "forward_causal vs backtracking BEATS its null (balanced acc > the null's "
        "95th percentile). LEWIS needs a POSITIVE collapse signature: forward vs "
        "backtracking AND forward vs common_cause both FAIL to beat their nulls "
        "(model treats them as one class). Neither met → inconclusive. 'Not separable' "
        "alone does NOT imply Lewis (affirming the null)."
    ),
    "outcome_if_pearl": (
        "forward_causal and backtracking are linearly separable. "
        "Model encodes causal direction in its representations — "
        "consistent with Pearl's do-calculus separating intervention on cause vs effect."
    ),
    "outcome_if_lewis": (
        "forward_causal vs backtracking AND forward_causal vs common_cause are both "
        "at chance — the model treats the conditions as one equivalence class, "
        "consistent with a single similarity-ordering mechanism that does not encode "
        "causal direction or distinguish genuine causation from a common-cause confound."
    ),
    "construct_caveat": (
        "CONSTRUCT UNDER REVIEW. (1) 'Pearl makes backtracking FALSE' was a do()-vs-"
        "counterfactual category error: Pearl's abduction step also makes backtracking "
        "TRUE, so forward-vs-backtracking may separate thematic role, not Lewis vs Pearl. "
        "(2) Standard Lewis with miracle-weighting AGREES with Pearl that the barometer "
        "common_cause counterfactual is false (storm still arrives) — the stated "
        "Lewis/Pearl split on common_cause may be a non-divergence. A faithful separator "
        "likely needs an explicit interventional (do) framing. Treat any mechanism verdict "
        "from this thread as provisional until the stimuli are reframed."
    ),
    "common_cause_note": (
        "common_cause is reported descriptively only. Under canonical Lewis (miracle-"
        "weighting) and Pearl alike, intervening on the barometer leaves the storm — so "
        "this contrast does not cleanly separate the two theories and is not used as a criterion."
    ),
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t1b_pythia14b_" + date_stamp,
    thread_id="t1b_pythia",
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, PYTHIA_1_4B_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=verify_stimulus_file_frequency_matched(VALIDATED_PATH),
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

# Surface diagnostic: collect every sentence per class, then report BPE token
# length per class. The probe extracts the FINAL token's activation, and the
# final-token positional embedding is determined by the token count — not the
# word count. Word-count parity (enforced in the grammar) does NOT guarantee
# BPE token-count parity, so we check the real thing here with the model's own
# tokenizer. A forward/backtracking token-length gap is a positional confound
# the layer-0 probe would read as "causal structure".
import json as _json
sentences_by_class: dict[str, list[str]] = {"forward_causal": [], "backtracking": [], "common_cause": []}
with VALIDATED_PATH.open("r") as _diag_file:
    for _raw_line in _diag_file:
        _stripped = _raw_line.strip()
        if not _stripped:
            continue
        _pair = _json.loads(_stripped)
        for _key, _lbl in (("sentence_a", _pair["label_a"]), ("sentence_b", _pair["label_b"])):
            if _lbl in sentences_by_class:
                sentences_by_class[_lbl].append(_pair[_key])

token_lengths_by_class: dict[str, list[int]] = {}
for _label, _sentences in sentences_by_class.items():
    token_lengths_by_class[_label] = [model.to_tokens(_s).shape[1] for _s in _sentences]

print("  Sample sentences + BPE token-length stats by class:")
for _label in ("forward_causal", "backtracking", "common_cause"):
    _lengths = token_lengths_by_class[_label]
    if not _lengths:
        continue
    _mean_len = sum(_lengths) / len(_lengths)
    print("  [" + _label + "]  tok mean=" + str(round(_mean_len, 2)) +
          " min=" + str(min(_lengths)) + " max=" + str(max(_lengths)) + " n=" + str(len(_lengths)))
    for _sent in sentences_by_class[_label][:3]:
        print("    (" + str(model.to_tokens(_sent).shape[1]) + "tok) " + _sent)

forward_mean_tokens = sum(token_lengths_by_class["forward_causal"]) / len(token_lengths_by_class["forward_causal"])
backtracking_mean_tokens = sum(token_lengths_by_class["backtracking"]) / len(token_lengths_by_class["backtracking"])
token_parity_gap = abs(forward_mean_tokens - backtracking_mean_tokens)
# Stimuli were BPE-token-parity verified on GPT-2's tokenizer. Pythia uses the
# GPT-NeoX tokenizer (different merges/vocab), so word-parity does NOT guarantee
# token-parity here. A systematic final-token gap is a positional-embedding
# confound the probe can exploit. Record it as a hard flag in the summary so a
# tainted verdict is self-identifying — not just a console warning that scrolls
# away in a Colab log.
tokenizer_parity_confound = bool(token_parity_gap > 0.5)
if tokenizer_parity_confound:
    print()
    print("  WARNING: forward_causal vs backtracking mean BPE token-length gap = " +
          str(round(token_parity_gap, 2)) + " tokens (Pythia tokenizer).")
    print("  The final-token position differs systematically between classes — a")
    print("  positional-embedding confound the probe can exploit. This run's verdict")
    print("  will be flagged tokenizer_parity_confound=True in summary.json.")
print()


RESULTS_DIR.mkdir(parents=True, exist_ok=True)

probe_results_by_layer: dict[int, dict] = {}

for activation_set in layer_activation_sets:
    layer_index = activation_set["layer"]
    all_activations = np.array(activation_set["activations"])
    all_labels = activation_set["labels"]

    # Filter to forward_causal and backtracking only. Filter the group ids in
    # lockstep so both sentences of a surviving pair keep their shared id (S4).
    all_pair_group_ids = activation_set["pair_group_ids"]
    binary_mask = [label in ("forward_causal", "backtracking") for label in all_labels]
    binary_activations = all_activations[binary_mask]
    binary_labels = [label for label, keep in zip(all_labels, binary_mask) if keep]
    binary_pair_group_ids = [pair_group_id for pair_group_id, keep in zip(all_pair_group_ids, binary_mask) if keep]

    probe_result = run_linear_probe(
        binary_activations, binary_labels, config, pair_ids=binary_pair_group_ids
    )
    probe_result["layer"] = layer_index
    probe_result["token_position"] = activation_set["token_position"]

    save_result(probe_result, RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json"))
    probe_results_by_layer[layer_index] = probe_result

# Exclude layer 0 from peak selection: it is raw token + positional embeddings
# (no attention, no context), so a peak there reflects a lexical/positional
# surface confound, not computed causal structure. Select on balanced accuracy.
candidate_layers = [layer_index for layer_index in probe_results_by_layer if layer_index != 0]
peak_probe_layer = max(
    candidate_layers,
    key=lambda layer_index: probe_results_by_layer[layer_index]["balanced_accuracy_mean"]
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
peak_probe_balanced_accuracy = probe_results_by_layer[peak_probe_layer]["balanced_accuracy_mean"]
chance_baseline = probe_results_by_layer[peak_probe_layer]["chance_baseline"]

layer_0_accuracy = probe_results_by_layer[0]["accuracy_mean"]
layer0_surface_confound = bool(layer_0_accuracy > 0.70)
if layer0_surface_confound:
    print()
    print("  WARNING: Layer 0 probe accuracy = " + str(round(layer_0_accuracy * 100, 1)) + "%")
    print("  Layer 0 is raw token + positional embeddings — no attention, no context.")
    print("  High accuracy here = surface confound (sentence length or template keyword).")
    print("  This run's verdict will be flagged layer0_surface_confound=True in summary.json.")
    print()

print("  Binary probe complete.")
print()

# ── Step 6: Pairwise probes at peak layer ─────────────────────────────────────
# The Lewis vs Pearl test: can model distinguish forward from backtracking?

print("[Step 6] Running pairwise probes at peak layer " + str(peak_probe_layer) + "...")

peak_activation_set = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
all_activations = np.array(peak_activation_set["activations"])
all_labels = peak_activation_set["labels"]
all_pair_group_ids = peak_activation_set["pair_group_ids"]

def pairwise_probe_accuracy(label_a: str, label_b: str) -> float:
    """Train binary probe on the two specified conditions, return accuracy."""
    condition_membership_mask = [condition_label in (label_a, label_b) for condition_label in all_labels]
    filtered_activations = all_activations[condition_membership_mask]
    filtered_labels = [condition_label for condition_label in all_labels if condition_label in (label_a, label_b)]
    # Group ids filtered in lockstep with the rows, so a pair stays in one fold (S4).
    filtered_pair_group_ids = [
        pair_group_id for pair_group_id, keep in zip(all_pair_group_ids, condition_membership_mask) if keep
    ]
    if len(set(filtered_labels)) < 2:
        return 0.5
    result = run_linear_probe(
        filtered_activations, filtered_labels, config, pair_ids=filtered_pair_group_ids
    )
    # Balanced accuracy: common_cause has ~50 items vs ~275, so raw accuracy is
    # floored by the majority class. Balanced accuracy keeps chance at 0.5.
    return result["balanced_accuracy_mean"]

def pairwise_beats_null(label_a: str, label_b: str) -> dict:
    """Calibrated decision for one contrast: does balanced accuracy beat a shuffled-label null?"""
    condition_membership_mask = [condition_label in (label_a, label_b) for condition_label in all_labels]
    filtered_activations = all_activations[condition_membership_mask]
    filtered_labels = [condition_label for condition_label in all_labels if condition_label in (label_a, label_b)]
    filtered_pair_group_ids = [
        pair_group_id for pair_group_id, keep in zip(all_pair_group_ids, condition_membership_mask) if keep
    ]
    if len(set(filtered_labels)) < 2:
        return {"beats_null": False, "null_balanced_p95": float("nan")}
    return probe_beats_null(filtered_activations, filtered_labels, config, pair_ids=filtered_pair_group_ids, seed=config.seed)

forward_vs_backtracking = pairwise_probe_accuracy("forward_causal", "backtracking")
forward_vs_common_cause = pairwise_probe_accuracy("forward_causal", "common_cause")
backtracking_vs_common_cause = pairwise_probe_accuracy("backtracking", "common_cause")

# Calibrated decisions (replace the magic 0.70 / 0.60 thresholds).
forward_vs_backtracking_null = pairwise_beats_null("forward_causal", "backtracking")
forward_vs_common_cause_null = pairwise_beats_null("forward_causal", "common_cause")

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

forward_indices = [i for i, condition_label in enumerate(layer_activation_sets[0]["labels"]) if condition_label == "forward_causal"]
backtracking_indices = [i for i, condition_label in enumerate(layer_activation_sets[0]["labels"]) if condition_label == "backtracking"]

mean_forward_by_layer: dict[int, np.ndarray] = {
    layer_activation_bundle["layer"]: np.array(layer_activation_bundle["activations"])[forward_indices].mean(axis=0)
    for layer_activation_bundle in layer_activation_sets
}

# Targets: every backtracking sentence. Multi-target sweep gives a per-layer mean
# KL with a bootstrap CI over targets, not a single-sentence n=1 effect.
backtracking_sentences = []
import json
with VALIDATED_PATH.open("r") as validated_jsonl_file:
    for raw_line in validated_jsonl_file:
        stripped = raw_line.strip()
        if stripped:
            stimulus_pair = json.loads(stripped)
            if stimulus_pair.get("label_b") == "backtracking":
                backtracking_sentences.append(stimulus_pair["sentence_b"])

sweep_result = run_layer_sweep_multi_target(
    mean_forward_by_layer,
    backtracking_sentences,
    config.layer_range,
    config.component,
    config.token_positions[0],
    model,
    seed=config.seed,
)
save_result(sweep_result, RESULTS_DIR / "layer_sweep.json")

peak_patch_layer = sweep_result["peak_layer"]
peak_patch_kl = sweep_result["mean_kl_by_layer"][peak_patch_layer]
peak_patch_kl_ci = sweep_result["kl_ci_95_by_layer"][peak_patch_layer]

# Specificity: forward-mean patch vs norm-matched random directions at peak layer,
# averaged over targets. Isolates direction from perturbation magnitude.
control_result = norm_matched_control_kl(
    mean_forward_by_layer[peak_patch_layer], backtracking_sentences,
    peak_patch_layer, config.component, config.token_positions[0], model,
    seed=config.seed,
)
assert_specificity_valid(peak_patch_kl, control_result["control_kl_p95"], peak_patch_layer)

print("  Peak patching layer : " + str(peak_patch_layer)
      + "  (mean KL=" + str(round(peak_patch_kl, 4))
      + ", 95% CI [" + str(round(peak_patch_kl_ci[0], 4)) + ", " + str(round(peak_patch_kl_ci[1], 4)) + "]"
      + ", n_targets=" + str(sweep_result["n_targets"]) + ")")
print("  Norm-matched control KL : " + str(round(control_result["mean_control_kl"], 4)))

# Illustrative direction check on one target (qualitative only — the quantitative
# evidence is the multi-target sweep + control above; this just shows WHAT a patch
# does to one completion).
illustrative_target = {"stimulus": backtracking_sentences[-1]}
with torch.no_grad():
    baseline_target_probs = torch.softmax(model(illustrative_target["stimulus"])[0, -1, :], dim=-1)
    baseline_top = baseline_target_probs.topk(5)
baseline_completions = [
    (model.to_string(token_idx), token_prob)
    for token_prob, token_idx in zip(baseline_top.values.tolist(), baseline_top.indices.tolist())
]
patched_direction_result = patch_activation(
    mean_forward_by_layer[peak_patch_layer], illustrative_target,
    peak_patch_layer, config.component, config.token_positions[0], model,
)
patched_completions = list(zip(patched_direction_result["top_tokens"], patched_direction_result["top_probs"]))
print("  Illustrative completions (one target) before | after patch:")
for (before_token, before_prob), (after_token, after_prob) in zip(baseline_completions, patched_completions):
    left  = (repr(before_token) + " " + str(round(before_prob * 100, 1)) + "%").ljust(20)
    print("    " + left + "|  " + repr(after_token) + " " + str(round(after_prob * 100, 1)) + "%")
print()

# ── Step 8: Determine Lewis vs Pearl (three-way, calibrated signatures) ───────
# No magic thresholds. Each contrast's verdict is "does balanced accuracy beat the
# 95th percentile of a shuffled-label null for this geometry?" (probe_beats_null).
#   PEARL: forward vs backtracking BEATS its null (direction is positively encoded).
#   LEWIS: forward vs backtracking AND forward vs common_cause both FAIL to beat
#          their nulls (model collapses them to one equivalence class — a positive
#          collapse signature, not mere absence at an arbitrary ceiling).
#   else inconclusive.
layers_agree = peak_probe_layer == peak_patch_layer

pearl_confirmed = forward_vs_backtracking_null["beats_null"]
lewis_confirmed = (
    not forward_vs_backtracking_null["beats_null"]
    and not forward_vs_common_cause_null["beats_null"]
)

if pearl_confirmed and not lewis_confirmed:
    mechanism_label = "PEARL (do-calculus)"
    mechanism_interpretation = expected_outcomes["outcome_if_pearl"]
elif lewis_confirmed and not pearl_confirmed:
    mechanism_label = "LEWIS (possible-worlds)"
    mechanism_interpretation = expected_outcomes["outcome_if_lewis"]
else:
    mechanism_label = "INCONCLUSIVE"
    mechanism_interpretation = (
        "Neither a positive Pearl signature (forward vs backtracking beats its "
        "shuffled-label null) nor a positive Lewis collapse signature (both contrasts "
        "fail to beat their nulls) was met. The geometry does not pre-specify a verdict. "
        "NOTE: T1b's stimulus construct is itself under review — forward vs backtracking "
        "as built may separate thematic role rather than Lewis vs Pearl, and standard "
        "Lewis (miracle-weighting) agrees with Pearl on the common_cause case. Do not "
        "read a mechanism claim from this run until the construct is reframed."
    )

common_cause_note = expected_outcomes["common_cause_note"]

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
    "peak_patch_kl_mean": float(peak_patch_kl),
    "peak_patch_kl_ci_95": [float(peak_patch_kl_ci[0]), float(peak_patch_kl_ci[1])],
    "peak_patch_n_targets": int(sweep_result["n_targets"]),
    "norm_matched_control_kl": float(control_result["mean_control_kl"]),
    "layers_agree": layers_agree,
    "forward_vs_backtracking_beats_null": bool(forward_vs_backtracking_null["beats_null"]),
    "forward_vs_backtracking_null_p95": float(forward_vs_backtracking_null["null_balanced_p95"]),
    "forward_vs_common_cause_beats_null": bool(forward_vs_common_cause_null["beats_null"]),
    "pearl_confirmed": pearl_confirmed,
    "lewis_confirmed": lewis_confirmed,
    "mechanism_label": mechanism_label,
    "mechanism_interpretation": mechanism_interpretation,
    "construct_caveat": expected_outcomes["construct_caveat"],
    "common_cause_interpretation": common_cause_note,
    # Confound flags — stimuli were token-parity verified on GPT-2, not Pythia.
    # verdict_confounded=True means a positional/lexical surface confound is present
    # and the mechanism verdict above must NOT be read as causal-structure evidence.
    "tokenizer_parity_confound": tokenizer_parity_confound,
    "tokenizer_parity_gap_tokens": float(token_parity_gap),
    "layer0_surface_confound": layer0_surface_confound,
    "layer0_probe_accuracy": float(layer_0_accuracy),
    "verdict_confounded": bool(tokenizer_parity_confound or layer0_surface_confound),
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
for layer_index in range(PYTHIA_1_4B_N_LAYERS):
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
print("Pairwise balanced accuracy at peak layer " + str(peak_probe_layer) + " (chance 50%, calibrated vs shuffled-label null):")
print("  forward_causal vs backtracking : " + str(round(forward_vs_backtracking * 100, 1)) + "%  "
      + "(null p95 " + str(round(forward_vs_backtracking_null["null_balanced_p95"] * 100, 1)) + "%, "
      + "beats null: " + str(forward_vs_backtracking_null["beats_null"]) + " → Pearl signature)")
print("  forward_causal vs common_cause : " + str(round(forward_vs_common_cause * 100, 1)) + "%  "
      + "(beats null: " + str(forward_vs_common_cause_null["beats_null"]) + ")")
print("  backtracking vs common_cause   : " + str(round(backtracking_vs_common_cause * 100, 1)) + "%")
print()
print("Peak patching layer : Layer " + str(peak_patch_layer) + "  (mean KL=" + str(round(peak_patch_kl, 4))
      + ", 95% CI [" + str(round(peak_patch_kl_ci[0], 4)) + ", " + str(round(peak_patch_kl_ci[1], 4)) + "])")
print("L2 / L3 agreement   : " + ("YES" if layers_agree else "NO"))
print()
print("Surface null accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print()
print("=" * 60)
print("MECHANISM: " + mechanism_label)
if summary["verdict_confounded"]:
    print("  *** VERDICT CONFOUNDED — DO NOT INTERPRET ***")
    if tokenizer_parity_confound:
        print("  - tokenizer_parity_confound: BPE token-length gap = "
              + str(round(token_parity_gap, 2)) + " tokens (Pythia tokenizer breaks GPT-2 parity)")
    if layer0_surface_confound:
        print("  - layer0_surface_confound: layer-0 probe = "
              + str(round(layer_0_accuracy * 100, 1)) + "% (surface/positional signal)")
print("=" * 60)
print()
print("Interpretation:")
print("  " + mechanism_interpretation)
print()
print("Construct caveat:")
print("  " + expected_outcomes["construct_caveat"])
print()
print("Common-cause position:")
print("  " + common_cause_note)
print()
print("T1c runs regardless of this verdict (records the T1b context); it tests "
      "Lewis vs Stalnaker via dispersion within the worlds-based camp.")
