"""
core/io.py — Shared I/O for all PoL-Probe experiments.

─── CONCEPT: Convention over configuration ───────────────────────────────────
Every experiment writes results through save_result and reads them back through
load_result / load_results. Centralizing I/O here means:
  - All results land in the same place in the same format.
  - Format changes happen once, not in every experiment script.
  - load_results(thread_id) lets you aggregate across a whole thread easily
    for summary statistics and comparison tables.

─── CONCEPT: Immutable result files ──────────────────────────────────────────
In mechanistic interpretability, a result you can't reload is a result you
can't verify. A result that was silently overwritten is a result you can't
trust. These functions are the single write path — if you want audit logging
or overwrite protection, add it here and it applies everywhere.

─── FILE LAYOUT ──────────────────────────────────────────────────────────────
Results live under:
    experiments/{thread_id}/results/
        surface_null.json       ← always written first (V11)
        probe_results.jsonl     ← one JSON object per line, one per probe run
        mantel_result.json      ← RSA Mantel test result
        summary.json            ← human-readable summary of the experiment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_result(result: dict[str, Any], path: str | Path) -> None:
    """
    Write a result dict to a JSON file at path.

    Creates parent directories if they don't exist. Writes with indent=2
    for human readability. Always use this instead of writing json directly —
    it ensures consistent formatting and directory creation everywhere.

    Args:
        result: Any dict. Must be JSON-serializable (no numpy arrays, no sets).
                Convert numpy values to float/int before calling.
        path:   Destination. Should be under experiments/{thread_id}/results/.

    Raises:
        TypeError: if result contains non-JSON-serializable values.

    Example:
        save_result({"probe_accuracy": 0.83, "layer": 9}, "experiments/t2/results/probe_results.json")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(path: str | Path) -> dict[str, Any]:
    """
    Load a single result dict from a JSON file.

    Args:
        path: Path to the JSON result file.

    Returns:
        Parsed dict.

    Raises:
        FileNotFoundError: if path does not exist.
        json.JSONDecodeError: if file is malformed.
    """
    with open(Path(path), "r") as f:
        return json.load(f)


def load_results(thread_id: str, results_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load all result JSON files for a given thread.

    Scans experiments/{thread_id}/results/ for all *.json files and returns
    them as a list of dicts, sorted by filename. Useful for:
      - Aggregating probe results across layers
      - Checking which experiments have already run
      - Building summary tables for a thread

    Args:
        thread_id:   e.g. "t2", "t1a", "t5"
        results_dir: Override default path. Default is
                     experiments/{thread_id}/results/ relative to project root.
                     Pass an explicit path in tests to avoid filesystem coupling.

    Returns:
        List of result dicts, sorted by filename alphabetically.
        Empty list if the results directory does not exist.

    Example:
        results = load_results("t2")
        accuracies = [r["probe_accuracy"] for r in results if "probe_accuracy" in r]
    """
    project_root = Path(__file__).parent.parent
    results_directory = Path(results_dir) if results_dir is not None else project_root / "experiments" / thread_id / "results"

    if not results_directory.exists():
        return []

    all_json_files = sorted(results_directory.glob("*.json"), key=lambda p: p.name)
    loaded_results = [load_result(result_file_path) for result_file_path in all_json_files]
    return loaded_results
