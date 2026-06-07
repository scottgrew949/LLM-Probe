"""
scripts/run_validation.py — Stimulus validation + behavioral gate for any thread.

Generates pairs, validates schema + frequency matching, runs behavioral gate,
saves results. Must pass before mechanistic experiment can run.

Usage (Colab):
    !python scripts/run_validation.py --thread t1a
    !python scripts/run_validation.py --thread t2
    !python scripts/run_validation.py --thread t2b
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--thread", required=True, help="Thread ID: t1a, t2, t2b, t1b, t1c, t3, t4, t5, t6")
args = parser.parse_args()

THREAD_ID = args.thread

# Max pairs per thread — full product space for each grammar
N_PAIRS_BY_THREAD: dict[str, int] = {
    "t1a": 300,
    "t1b": 300,
    "t1c": 200,
    "t2":  630,
    "t2b": 30,
    "t3":  300,
    "t4":  300,
    "t5":  300,
    "t6":  300,
}

if THREAD_ID not in N_PAIRS_BY_THREAD:
    print("ERROR: Unknown thread '" + THREAD_ID + "'.")
    print("Known threads: " + ", ".join(sorted(N_PAIRS_BY_THREAD)))
    sys.exit(1)

N_PAIRS = N_PAIRS_BY_THREAD[THREAD_ID]

# ── Paths ─────────────────────────────────────────────────────────────────────

GRAMMAR_FILE    = PROJECT_ROOT / "stimuli" / "grammars" / (THREAD_ID + ".py")
GENERATED_PATH  = PROJECT_ROOT / "stimuli" / "generated" / THREAD_ID / "pairs.jsonl"
VALIDATED_PATH  = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
GATE_RESULT_PATH = PROJECT_ROOT / "experiments" / THREAD_ID / "results" / "behavioral_gate.json"

if not GRAMMAR_FILE.exists():
    print("ERROR: Grammar file not found: " + str(GRAMMAR_FILE))
    sys.exit(1)

# ── Imports ───────────────────────────────────────────────────────────────────

from stimuli.pipeline import generate_pairs, validate_set, run_behavioral_gate
from core.io import save_result

# Load generate_behavioral_items from the thread's grammar module dynamically
grammar_spec = importlib.util.spec_from_file_location("grammar_module", GRAMMAR_FILE)
grammar_module = importlib.util.module_from_spec(grammar_spec)
grammar_spec.loader.exec_module(grammar_module)

# ── Pipeline ──────────────────────────────────────────────────────────────────

MODEL_NAME     = "gpt2-medium"
GATE_THRESHOLD = 0.70

print("=" * 60)
print("PoL-Probe — Validation Pipeline  [Thread: " + THREAD_ID + "]")
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

# Step 1: Generate ─────────────────────────────────────────────────────────────
print("[Step 1] Generating stimulus pairs...")
print("  Grammar: " + str(GRAMMAR_FILE))
print("  n=" + str(N_PAIRS))

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
    sys.exit(1)

n_rejected = len(generated_pairs) - len(validated_pairs)

# Count individual sentence labels (both sentence_a and sentence_b).
# Pair-level label_a counts alone omit conditions that appear only as label_b.
sentence_label_counts: dict[str, int] = {}
for pair in validated_pairs:
    for label_key in ("label_a", "label_b"):
        label = pair.get(label_key, "unknown")
        sentence_label_counts[label] = sentence_label_counts.get(label, 0) + 1

print("  Passed:   " + str(len(validated_pairs)) + " pairs")
print("  Rejected: " + str(n_rejected) + " pairs")
print("  Individual sentence counts by label:")
for label, count in sorted(sentence_label_counts.items()):
    print("    " + label + ": " + str(count))
print("  Written to: " + str(VALIDATED_PATH))
print()

# Step 3: Behavioral gate ──────────────────────────────────────────────────────
print("[Step 3] Running behavioral gate...")
print("  Threshold: " + str(int(GATE_THRESHOLD * 100)) + "%")

behavioral_items = grammar_module.generate_behavioral_items()
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

# Step 4: Save ─────────────────────────────────────────────────────────────────
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
    print("Next: run the experiment script for thread " + THREAD_ID + ".")
    sys.exit(0)
else:
    print("RESULT: FAIL")
    print()
    print("  Behavioral gate did not pass.")
    print("  Accuracy " + str(round(gate_result["accuracy"] * 100, 1)) + "% is below threshold " + str(int(GATE_THRESHOLD * 100)) + "%.")
    print()
    print("  Run scripts/inspect_gate.py --thread " + THREAD_ID + " to see which items failed.")
    sys.exit(1)
