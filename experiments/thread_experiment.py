"""
experiments/thread_experiment.py — Abstract base for all thread experiment runners.

Template Method pattern: run() owns the invariant skeleton; subclasses implement:
  - build_locked_config(n_layers) → ExperimentConfig   (pre-registration)
  - analyze(config, model, n_layers) → dict             (thread-specific pipeline)
  - print_results(summary) → None                       (final formatted output)

Adding a new model: add its suffix + model_id to MODEL_SPECS in experiments/config.py.
Adding a new thread: subclass ThreadExperiment and create a 4-line shim at
experiments/{thread_id}/run_experiment.py.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ThreadExperiment(ABC):
    """
    Abstract base for PoL-Probe thread runners.

    run() is the template method. All path/identity state is set in __init__
    so subclasses and unit tests can inspect it without running the full pipeline.
    """

    def __init__(self, thread_id: str) -> None:
        from experiments.config import base_thread_of, model_id_for_thread
        self.thread_id = thread_id
        self.base_thread = base_thread_of(thread_id)
        self.model_id = model_id_for_thread(thread_id)
        self.validated_path = (
            PROJECT_ROOT / "stimuli" / "validated" / self.base_thread / "pairs.validated.jsonl"
        )
        self.results_dir = PROJECT_ROOT / "experiments" / thread_id / "results"
        self.config_path = PROJECT_ROOT / "experiments" / thread_id / "config.json"

    # ── Template method ───────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the complete experiment for this thread+model.

        Skeleton (invariant across all threads):
          1. Guard checks (validated file, prerequisites)
          2. Load model (determines n_layers for config)
          3. Build and lock ExperimentConfig (pre-registration)
          4. Surface-statistics null (V11)
          5. Pre-phase-gate hook (T1b: matrix build + decorrelation artifact)
          6. Phase gate (V1, V7, V8, V11, V10, V23, …)
          7. Thread-specific analysis
          8. Save summary.json

        Returns the summary dict (also saved to results_dir/summary.json).
        """
        self._check_guards()

        print("Loading " + self.model_id + "...")
        model, n_layers = self._load_model()
        print("  n_layers=" + str(n_layers) + ", d_model=" + str(model.cfg.d_model))
        print()

        config = self.build_locked_config(n_layers)
        config.to_json(self.config_path)
        print("Config locked: " + config.experiment_id)
        print()

        from experiments.run import run_surface_null, check_phase_gate
        surface_null = run_surface_null(config)
        print("Surface null: " + str(round(surface_null["surface_classifier_accuracy"] * 100, 1)) + "%")
        print()

        self._pre_phase_gate_hook(config, model, n_layers)

        check_phase_gate(config)
        print("Phase gate: passed")
        print()

        summary = self.analyze(config, model, n_layers)

        from core.io import save_result
        self.results_dir.mkdir(parents=True, exist_ok=True)
        save_result(summary, self.results_dir / "summary.json")

        self.print_results(summary)
        return summary

    # ── Overridable hooks ─────────────────────────────────────────────────────

    def _check_guards(self) -> None:
        """
        Verify prerequisites exist before any heavy work starts.
        Override to add thread-specific checks (e.g. T1c checking T1b summary).
        Call super()._check_guards() first in overrides.
        """
        if not self.validated_path.exists():
            print("ERROR: Validated stimulus file not found.")
            print("  Expected: " + str(self.validated_path))
            print("  Run: scripts/run_validation.py --thread " + self.thread_id)
            sys.exit(1)

    def _load_model(self):
        """Load HookedTransformer from self.model_id. Returns (model, n_layers)."""
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(self.model_id)
        model.eval()
        return model, model.cfg.n_layers

    def _pre_phase_gate_hook(self, config, model, n_layers: int) -> None:
        """
        Called after surface null, before phase gate.
        Default: no-op. T1b overrides to build theoretical matrices and
        write matrix_decorrelation.json (required by V23 gate).
        """
        pass

    # ── Abstract methods ──────────────────────────────────────────────────────

    @abstractmethod
    def build_locked_config(self, n_layers: int):
        """
        Build, lock, and return ExperimentConfig.

        Called after model is loaded so n_layers (from model.cfg.n_layers) is
        available for setting layer_range. Must call config.lock() before returning.
        """
        ...

    @abstractmethod
    def analyze(self, config, model, n_layers: int) -> dict:
        """
        Run the thread-specific analysis pipeline.

        Called after phase gate passes. Receives the already-loaded model so
        extraction and patching share the same model instance. Returns the
        summary dict — the caller (run()) saves it to disk.
        """
        ...

    @abstractmethod
    def print_results(self, summary: dict) -> None:
        """Print the final formatted result table and verdict."""
        ...
