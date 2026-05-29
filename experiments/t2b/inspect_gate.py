"""
experiments/t2b/inspect_gate.py — Human-readable T2b behavioral gate inspector.

Usage (Colab):
    !python experiments/t2b/inspect_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GATE_RESULT_PATH = PROJECT_ROOT / "experiments" / "t2b" / "results" / "behavioral_gate.json"

with GATE_RESULT_PATH.open("r") as result_file:
    gate_result = json.load(result_file)

n_correct = round(gate_result["accuracy"] * gate_result["n_items"])
n_total = gate_result["n_items"]
accuracy_percent = round(gate_result["accuracy"] * 100, 1)
gate_status = "PASS" if gate_result["passed"] else "FAIL"

print("=" * 60)
print("Behavioral Gate Results — T2b Hyperintensionality")
print("=" * 60)
print()
print("Overall accuracy : " + str(accuracy_percent) + "%")
print("Items correct    : " + str(n_correct) + " out of " + str(n_total))
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
