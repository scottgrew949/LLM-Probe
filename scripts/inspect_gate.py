"""
scripts/inspect_gate.py — Human-readable behavioral gate result inspector.

Usage (Colab):
    !python scripts/inspect_gate.py --thread t1a
    !python scripts/inspect_gate.py --thread t2
    !python scripts/inspect_gate.py --thread t2b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--thread", required=True, help="Thread ID, e.g. t1a, t2, t2b")
args = parser.parse_args()

GATE_RESULT_PATH = PROJECT_ROOT / "experiments" / args.thread / "results" / "behavioral_gate.json"

if not GATE_RESULT_PATH.exists():
    print("ERROR: Gate result not found at " + str(GATE_RESULT_PATH))
    print("Run scripts/run_validation.py --thread " + args.thread + " first.")
    sys.exit(1)

with GATE_RESULT_PATH.open("r") as result_file:
    gate_result = json.load(result_file)

n_correct = round(gate_result["accuracy"] * gate_result["n_items"])
accuracy_percent = round(gate_result["accuracy"] * 100, 1)
gate_status = "PASS" if gate_result["passed"] else "FAIL"

print("=" * 60)
print("Behavioral Gate Results — Thread " + args.thread.upper())
print("=" * 60)
print()
print("Overall accuracy : " + str(accuracy_percent) + "%")
print("Items correct    : " + str(n_correct) + " out of " + str(gate_result["n_items"]))
print("Gate status      : " + gate_status + " (threshold 70%)")
print()
print("-" * 60)
print()

for item_number, item_result in enumerate(gate_result["details"], start=1):
    model_answer = (
        item_result["choice_a"]
        if item_result["model_choice"] == "a"
        else item_result["choice_b"]
    )
    correct_answer = (
        item_result["choice_a"]
        if item_result["correct_choice"] == "a"
        else item_result["choice_b"]
    )
    item_status = "PASS" if item_result["is_correct"] else "FAIL"

    print("Item " + str(item_number) + " [" + item_status + "]")
    print("  Question       : " + item_result["question"])
    print("  Model chose    : " + model_answer)
    print("  Correct answer : " + correct_answer)
    print()
