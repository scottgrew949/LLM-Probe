"""
experiments/t1a/run_validation.py — T1a stimulus validation + behavioral gate.

Run once on Colab T4 before any mechanistic analysis. Produces three files
that downstream pipeline steps depend on:

  stimuli/generated/t1a/pairs.jsonl
  stimuli/validated/t1a/pairs.validated.jsonl
  experiments/t1a/results/behavioral_gate.json

Usage (Colab):
    !python experiments/t1a/run_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stimuli.pipeline import generate_pairs, validate_set, run_behavioral_gate
from stimuli.grammars.t1a import generate_behavioral_items
from core.io import save_result


# ── Constants ─────────────────────────────────────────────────────────────────

THREAD_ID = "t1a"
N_PAIRS = 300        # Full space: 12 domains × 5 L3 templates × 5 L1 templates
MODEL_NAME = "gpt2-medium"
GATE_THRESHOLD = 0.70

GRAMMAR_FILE = PROJECT_ROOT / "stimuli" / "grammars" / "t1a.py"
GENERATED_PATH = PROJECT_ROOT / "stimuli" / "generated" / THREAD_ID / "pairs.jsonl"
VALIDATED_PATH = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
GATE_RESULT_PATH = PROJECT_ROOT / "experiments" / THREAD_ID / "results" / "behavioral_gate.json"


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T1a Validation Pipeline")
print("=" * 60)
print()

# Step 0: Load model ───────────────────────────────────────────────────────────
print("[Step 0] Loading model...")
print("  " + MODEL_NAME)

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_NAME)
model.eval()

print("  Loaded. n_layers=" + str(model.cfg.n_layers) + ", d_model=" + str(model.cfg.d_model))
print()

# Step 1: Generate pairs ───────────────────────────────────────────────────────
print("[Step 1] Generating stimulus pairs...")
print("  Grammar: " + str(GRAMMAR_FILE))
print("  n=" + str(N_PAIRS) + " (12 domains x 5 L3 templates x 5 L1 templates)")

generated_pairs = generate_pairs(
    grammar_file=GRAMMAR_FILE,
    n=N_PAIRS,
    thread_id=THREAD_ID,
)

print("  Generated: " + str(len(generated_pairs)) + " pairs")
print("  Written to: " + str(GENERATED_PATH))
print()

# Step 2: Validate ─────────────────────────────────────────────────────────────
print("[Step 2] Validating pairs (schema + frequency matching)...")

try:
    validated_pairs = validate_set(generated_pairs, thread_id=THREAD_ID)
except ValueError as validation_error:
    print()
    print("ERROR: validate_set() failed — all pairs rejected.")
    print("  " + str(validation_error))
    print()
    print("PIPELINE FAILED. Check grammar output against stimulus.schema.json.")
    sys.exit(1)

n_rejected = len(generated_pairs) - len(validated_pairs)
n_l3 = sum(1 for p in validated_pairs if p.get("label_a") == "causal_l3")
n_l1 = sum(1 for p in validated_pairs if p.get("label_b") == "associative_l1")

print("  Passed:   " + str(len(validated_pairs)) + " pairs")
print("  Rejected: " + str(n_rejected) + " pairs")
print("  causal_l3 sentences:      " + str(n_l3))
print("  associative_l1 sentences: " + str(n_l1))
print("  Written to: " + str(VALIDATED_PATH))
print()

# Step 3: Behavioral gate ──────────────────────────────────────────────────────
print("[Step 3] Running behavioral gate...")
print("  Threshold: " + str(int(GATE_THRESHOLD * 100)) + "%")

behavioral_items = generate_behavioral_items()
print("  Scoring " + str(len(behavioral_items)) + " forced-choice items on " + MODEL_NAME + "...")

gate_result = run_behavioral_gate(
    behavioral_items=behavioral_items,
    model=model,
    threshold=GATE_THRESHOLD,
)

n_correct = round(gate_result["accuracy"] * gate_result["n_items"])
print("  Accuracy: " + str(round(gate_result["accuracy"] * 100, 1)) + "%  (" + str(n_correct) + "/" + str(gate_result["n_items"]) + " correct)")
print("  Passed:   " + str(gate_result["passed"]))
print()

# Step 4: Save gate result ─────────────────────────────────────────────────────
print("[Step 4] Saving gate result...")
save_result(gate_result, GATE_RESULT_PATH)
print("  Written to: " + str(GATE_RESULT_PATH))
print()

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 60)
if gate_result["passed"]:
    print("RESULT: PASS")
    print()
    print("Output files:")
    print("  " + str(GENERATED_PATH))
    print("  " + str(VALIDATED_PATH))
    print("  " + str(GATE_RESULT_PATH))
    print()
    print("Next: Task #36 — pre-specify T1a outcomes and lock config.")
    sys.exit(0)
else:
    print("RESULT: FAIL")
    print()
    print("  Behavioral gate did not pass.")
    print("  Accuracy " + str(round(gate_result["accuracy"] * 100, 1)) + "% is below threshold " + str(int(GATE_THRESHOLD * 100)) + "%.")
    print()
    print("  Run inspect_gate.py to see which items failed.")
    print("  behavioral_gate.json saved with passed=False.")
    sys.exit(1)
