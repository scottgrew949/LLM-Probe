"""
experiments/t2/run_validation.py — T2 stimulus validation + behavioral gate.

Run once on Colab T4 before any mechanistic analysis. Produces three files
that downstream pipeline steps (check_phase_gate, run_experiment) depend on:

  stimuli/generated/t2/pairs.jsonl
  stimuli/validated/t2/pairs.validated.jsonl
  experiments/t2/results/behavioral_gate.json

Usage (Colab):
    !python experiments/t2/run_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve project root: experiments/t2/run_validation.py → three .parent calls
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Guard against duplicate insertion on repeated Colab cell execution
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stimuli.pipeline import generate_pairs, validate_set, run_behavioral_gate
from stimuli.grammars.t2 import generate_behavioral_items
from core.io import save_result


# ── Constants ─────────────────────────────────────────────────────────────────

THREAD_ID = "t2"
N_PAIRS = 630        # Full space: 600 opaque + 30 transparent
MODEL_NAME = "gpt2-medium"
GATE_THRESHOLD = 0.70

GRAMMAR_FILE = PROJECT_ROOT / "stimuli" / "grammars" / "t2.py"
GENERATED_PATH = PROJECT_ROOT / "stimuli" / "generated" / THREAD_ID / "pairs.jsonl"
VALIDATED_PATH = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
GATE_RESULT_PATH = PROJECT_ROOT / "experiments" / THREAD_ID / "results" / "behavioral_gate.json"


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T2 Validation Pipeline")
print("=" * 60)
print()

# Step 0: Load model ───────────────────────────────────────────────────────────
print("[Step 0] Loading model...")
print(f"  {MODEL_NAME}")

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_NAME)
model.eval()

print(f"  Loaded. n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
print()

# Step 1: Generate pairs ───────────────────────────────────────────────────────
print("[Step 1] Generating stimulus pairs...")
print(f"  Grammar: {GRAMMAR_FILE}")
print(f"  n={N_PAIRS} (600 opaque + 30 transparent)")

generated_pairs = generate_pairs(
    grammar_file=GRAMMAR_FILE,
    n=N_PAIRS,
    thread_id=THREAD_ID,
)

print(f"  Generated: {len(generated_pairs)} pairs")
print(f"  Written to: {GENERATED_PATH}")
print()

# Step 2: Validate ─────────────────────────────────────────────────────────────
print("[Step 2] Validating pairs (schema + frequency matching)...")

try:
    validated_pairs = validate_set(generated_pairs, thread_id=THREAD_ID)
except ValueError as validation_error:
    print()
    print("ERROR: validate_set() failed — all pairs rejected.")
    print(f"  {validation_error}")
    print()
    print("PIPELINE FAILED. Check grammar output against stimulus.schema.json.")
    sys.exit(1)

n_rejected = len(generated_pairs) - len(validated_pairs)
n_opaque = sum(1 for pair in validated_pairs if pair.get("label_a") == "opaque")
n_transparent = sum(1 for pair in validated_pairs if pair.get("label_a") == "transparent")

print(f"  Passed:   {len(validated_pairs)} pairs")
print(f"  Rejected: {n_rejected} pairs")
print(f"  Breakdown: {n_opaque} opaque, {n_transparent} transparent")
print(f"  Written to: {VALIDATED_PATH}")
print()

# Step 3: Behavioral gate ──────────────────────────────────────────────────────
print("[Step 3] Running behavioral gate...")
print(f"  Threshold: {GATE_THRESHOLD:.0%}")

behavioral_items = generate_behavioral_items()
print(f"  Scoring {len(behavioral_items)} forced-choice items on {MODEL_NAME}...")

gate_result = run_behavioral_gate(
    behavioral_items=behavioral_items,
    model=model,
    threshold=GATE_THRESHOLD,
)

n_correct = round(gate_result["accuracy"] * gate_result["n_items"])
print(f"  Accuracy: {gate_result['accuracy']:.1%}  ({n_correct}/{gate_result['n_items']} correct)")
print(f"  Passed:   {gate_result['passed']}")
print()

# Step 4: Save gate result ─────────────────────────────────────────────────────
print("[Step 4] Saving gate result...")
save_result(gate_result, GATE_RESULT_PATH)
print(f"  Written to: {GATE_RESULT_PATH}")
print()

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 60)
if gate_result["passed"]:
    print("RESULT: PASS")
    print()
    print("Output files:")
    print(f"  {GENERATED_PATH}")
    print(f"  {VALIDATED_PATH}")
    print(f"  {GATE_RESULT_PATH}")
    print()
    print("Next: Task #19 — write T2 expected outcomes and lock config.")
    sys.exit(0)
else:
    print("RESULT: FAIL")
    print()
    print(f"  Behavioral gate did not pass.")
    print(f"  Accuracy {gate_result['accuracy']:.1%} < threshold {GATE_THRESHOLD:.0%}.")
    print()
    print("  GPT-2 medium does not exhibit the Frege sense/reference")
    print("  distinction behaviorally. Nothing to explain mechanistically.")
    print()
    print("  gate_result['details'] has per-item model_choice vs correct_choice.")
    print("  behavioral_gate.json saved with passed=False.")
    print("  Do NOT proceed to Task #19 until gate passes.")
    sys.exit(1)
