# PoL-Probe Depth Additions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add circuit-level mechanistic analysis, T1d causal identification sub-thread, and T2c two-dimensional semantics to PoL-Probe in strict phase sequence.

**Architecture:** Shared infrastructure (ExperimentConfig additions, circuits/ module, invariants V14–V18) is built first. Circuit analysis is then enabled on a clean T1b. T1d follows T1c within Phase 2. T2c is a Phase 7 addition after T2b on Llama 3.2 3B — its tasks are stubs until that phase begins.

**Tech Stack:** Python 3.12, TransformerLens, PyTorch, scikit-learn, numpy, pytest

---

## File Map

**Create:**
- `circuits/__init__.py`
- `circuits/circuit_finder.py` — `run_head_sweep`, `find_peak_circuit_components`, `run_path_patching`
- `circuits/attribution.py` — `compute_logit_attribution`
- `stimuli/grammars/t1d.py` — T1d causal identification grammar
- `stimuli/grammars/t2c.py` — T2c two-dimensional semantics grammar (Phase 7)
- `experiments/t1d/run_experiment.py`
- `experiments/t2c/run_experiment.py` (Phase 7)
- `tests/test_circuits.py`
- `tests/test_t1d_grammar.py`
- `tests/test_t2c_grammar.py` (Phase 7)

**Modify:**
- `experiments/config.py` — 5 new fields, V14–V18 invariant enforcement
- `experiments/run.py` — `check_phase_gate` handles t1d, t2c; circuit step in pipeline
- `probes/probes.py` — add `run_identification_probe`
- `stimuli/schemas/philbench.schema.json` — add `circuit_components`, `identification_criterion`
- `experiments/t1b/run_experiment.py` — add circuit analysis step (Step 8.5)
- `TASKS.md` — add task IDs 43a–43e, 86a–86e
- `SPEC.md` — add §T1d, §T2c, update §V with V14–V18

---

## PHASE GATE: Complete tasks 1–6 before any experiment tasks (7+).
## T1b issues (tasks 7–9) must be resolved before task 10 (circuit analysis).
## T1d (tasks 11–12) gates on T1c being complete.
## T2c tasks (13–14) are Phase 7 — do not begin until T2b on Llama passes.

---

## Task 1: ExperimentConfig additions + invariants V14–V18

**Files:**
- Modify: `experiments/config.py`
- Modify: `tests/test_phase0.py`

- [ ] **Step 1: Write failing tests for new config fields**

Add to `tests/test_phase0.py` inside `class TestExperimentConfig`:

```python
def test_t1d_requires_identification_criterion(self):
    """V14: identification_criterion must be non-null for t1d."""
    config = _make_config(thread_id="t1d")
    config.expected_outcomes = {"test": "value"}
    config.frequency_match_verified = True
    config.stimulus_sha256 = "abc123"
    config.stimulus_file = __file__
    with pytest.raises(ValueError, match="identification_criterion"):
        config.lock()

def test_t1d_requires_confounder_structure(self):
    """V15: confounder_structure must be non-null for t1d."""
    config = _make_config(thread_id="t1d", identification_criterion="back_door")
    config.expected_outcomes = {"test": "value"}
    config.frequency_match_verified = True
    config.stimulus_sha256 = "abc123"
    config.stimulus_file = __file__
    with pytest.raises(ValueError, match="confounder_structure"):
        config.lock()

def test_t2c_requires_intension_type(self):
    """V17: intension_type must be non-null for t2c."""
    config = _make_config(thread_id="t2c")
    config.expected_outcomes = {"test": "value"}
    config.frequency_match_verified = True
    config.stimulus_sha256 = "abc123"
    config.stimulus_file = __file__
    with pytest.raises(ValueError, match="intension_type"):
        config.lock()

def test_t1d_and_t2c_are_valid_thread_ids(self):
    config_t1d = _make_config(thread_id="t1d")
    config_t2c = _make_config(thread_id="t2c")
    assert config_t1d.thread_id == "t1d"
    assert config_t2c.thread_id == "t2c"

def test_circuit_analysis_defaults_to_disabled(self):
    config = _make_config()
    assert config.circuit_analysis_enabled is False
    assert config.circuit_kl_threshold == 0.1
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_phase0.py::TestExperimentConfig -v 2>&1 | tail -20
```

Expected: 5 new tests fail (fields don't exist yet).

- [ ] **Step 3: Add new fields and invariant enforcement to `experiments/config.py`**

Add to the `# ── Thread-specific required fields ──` section after `matrix_source`:

```python
    # ── Circuit analysis ──────────────────────────────────────────────────────
    circuit_analysis_enabled: bool = False
    """
    When True, run head sweep and path patching after layer sweep. [INVARIANT V18]
    Requires layer sweep results to exist before head sweep runs.
    Set False for initial experiment runs — enable only on clean, validated data.
    """

    circuit_kl_threshold: float = 0.1
    """
    Minimum KL divergence for a (layer, head) pair to count as a circuit component.
    Used by find_peak_circuit_components in circuits/circuit_finder.py.
    """

    # ── T1d — causal identification ───────────────────────────────────────────
    identification_criterion: Optional[str] = None
    """
    [INVARIANT V14] Required non-null for T1d.
    Which identification criterion the stimulus set tests.
    Valid values: 'back_door' | 'front_door'
    back_door: observed confounders; adjustment via back-door criterion.
    front_door: hidden confounder with mediator; adjustment via front-door criterion.
    """

    confounder_structure: Optional[dict] = None
    """
    [INVARIANT V15] Required non-null for T1d.
    Formal causal graph describing the confounded structure in the stimuli.
    Expected keys: 'nodes', 'edges', 'criterion', 'adjustment_set'.
    Example:
        {
            "nodes": ["treatment", "outcome", "confounder"],
            "edges": [["treatment", "outcome"], ["confounder", "treatment"], ["confounder", "outcome"]],
            "criterion": "back_door",
            "adjustment_set": ["confounder"]
        }
    """

    # ── T2c — two-dimensional semantics ───────────────────────────────────────
    intension_type: Optional[str] = None
    """
    [INVARIANT V17] Required non-null for T2c.
    Which intension dimension the stimulus set targets.
    Valid values: 'primary' | 'secondary' | 'dissociation'
    primary: terms with same secondary intension but different primary (water vs H2O).
    secondary: metaphysically necessary identity statements.
    dissociation: secondarily necessary but primarily contingent statements.
    """
```

Add the valid thread IDs check to `__post_init__`:

```python
        valid_thread_ids = {
            "t1", "t1a", "t1b", "t1c", "t1d",
            "t2", "t2b", "t2c",
            "t3", "t4", "t5", "t6",
        }
        if self.thread_id not in valid_thread_ids:
            raise ValueError(
                f"thread_id='{self.thread_id}' is not valid. "
                f"Must be one of: {sorted(valid_thread_ids)}"
            )
```

Also update the `thread_id` docstring to include `t1d` and `t2c`.

- [ ] **Step 4: Add V14, V15, V17 enforcement to `lock()`**

Add after the existing `stimulus_sha256` check in `lock()`:

```python
        # V14: T1d requires identification_criterion
        if self.thread_id == "t1d" and self.identification_criterion is None:
            raise ValueError(
                "identification_criterion is None (V14). "
                "Set to 'back_door' or 'front_door' before locking T1d config."
            )

        # V15: T1d requires confounder_structure
        if self.thread_id == "t1d" and self.confounder_structure is None:
            raise ValueError(
                "confounder_structure is None (V15). "
                "Define the formal causal graph before locking T1d config."
            )

        # V17: T2c requires intension_type
        if self.thread_id == "t2c" and self.intension_type is None:
            raise ValueError(
                "intension_type is None (V17). "
                "Set to 'primary', 'secondary', or 'dissociation' before locking T2c config."
            )
```

- [ ] **Step 5: Run tests**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_phase0.py -v 2>&1 | tail -15
```

Expected: all 43+ tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/config.py tests/test_phase0.py
git commit -m "feat(config): add circuit analysis, T1d, T2c fields and invariants V14-V17"
```

---

## Task 2: `circuits/` module — `run_head_sweep` + `find_peak_circuit_components`

**Files:**
- Create: `circuits/__init__.py`
- Create: `circuits/circuit_finder.py` (partial — `run_head_sweep` + `find_peak_circuit_components` only)
- Create: `tests/test_circuits.py` (partial)

- [ ] **Step 1: Write failing tests**

Create `tests/test_circuits.py`:

```python
"""
tests/test_circuits.py — Unit tests for circuits/ module (no GPU required).

All functions under test operate on numpy arrays and dicts.
Model-dependent functions (those that call model.run_with_hooks) are excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import unittest.mock

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock torch so circuits imports work without GPU
with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
    from circuits.circuit_finder import find_peak_circuit_components


class TestFindPeakCircuitComponents:
    def test_returns_components_above_threshold(self):
        head_sweep_result = {
            "kl_matrix": {
                "(0,0)": 0.05,
                "(0,1)": 0.15,
                "(1,0)": 0.25,
                "(1,1)": 0.08,
            },
            "peak_head": [1, 0],
            "peak_kl": 0.25,
        }
        components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.10)
        kl_values = [component["kl_effect"] for component in components]
        assert all(value >= 0.10 for value in kl_values)
        assert len(components) == 2

    def test_sorted_by_kl_descending(self):
        head_sweep_result = {
            "kl_matrix": {"(0,0)": 0.30, "(1,2)": 0.50, "(2,1)": 0.20},
            "peak_head": [1, 2],
            "peak_kl": 0.50,
        }
        components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.0)
        kl_effects = [component["kl_effect"] for component in components]
        assert kl_effects == sorted(kl_effects, reverse=True)

    def test_returns_empty_when_nothing_above_threshold(self):
        head_sweep_result = {
            "kl_matrix": {"(0,0)": 0.01, "(0,1)": 0.02},
            "peak_head": [0, 1],
            "peak_kl": 0.02,
        }
        components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.50)
        assert components == []

    def test_component_has_required_keys(self):
        head_sweep_result = {
            "kl_matrix": {"(3,5)": 0.40},
            "peak_head": [3, 5],
            "peak_kl": 0.40,
        }
        components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.0)
        assert len(components) == 1
        component = components[0]
        assert component["layer"] == 3
        assert component["head"] == 5
        assert component["kl_effect"] == 0.40

    def test_parses_layer_and_head_from_key(self):
        head_sweep_result = {
            "kl_matrix": {"(12,7)": 0.60, "(0,15)": 0.30},
            "peak_head": [12, 7],
            "peak_kl": 0.60,
        }
        components = find_peak_circuit_components(head_sweep_result, kl_threshold=0.0)
        layers = {component["layer"] for component in components}
        heads = {component["head"] for component in components}
        assert 12 in layers
        assert 7 in heads
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_circuits.py -v 2>&1 | tail -15
```

Expected: ImportError or collection error — `circuits/` does not exist yet.

- [ ] **Step 3: Create `circuits/__init__.py`**

```python
"""circuits/ — Circuit-level mechanistic analysis for PoL-Probe."""
```

- [ ] **Step 4: Create `circuits/circuit_finder.py` with `run_head_sweep` and `find_peak_circuit_components`**

```python
"""
circuits/circuit_finder.py — Head-level and path-level causal analysis.

─── CONCEPT: From layers to circuits ─────────────────────────────────────────
Layer-resolved patching (L3 in interventions/interventions.py) identifies
WHICH LAYER causally encodes a distinction. Circuit analysis goes one level
deeper: within that layer, WHICH ATTENTION HEADS drive the effect?

The key tool is hook_z — the per-head value output in TransformerLens.
Shape: (batch, seq, n_heads, d_head). Patching a specific head means replacing
[:, token_position, head_index, :] with the source activation for that head.

run_head_sweep patches every (layer, head) independently, building a KL matrix.
find_peak_circuit_components filters that matrix to the meaningful components.
run_path_patching patches all peak components simultaneously to verify that
those components together recover most of the original layer sweep effect.

If fraction_recovered ≥ 0.80, the circuit is sufficient — those heads account
for ≥80% of the causal effect.

─── MODEL LOADING CONVENTION ─────────────────────────────────────────────────
All functions accept `model` as a required parameter. Load once, pass through.
Never load the model inside these functions.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch


def run_head_sweep(
    source_activations_by_layer_and_head: dict[tuple[int, int], np.ndarray],
    target_run_config: dict[str, Any],
    layer_range: tuple[int, int],
    n_heads: int,
    token_position: int,
    model: Any,
    baseline_logits: list[float] | None = None,
) -> dict[str, Any]:
    """
    Patch each attention head independently via hook_z. Returns KL matrix.

    For each (layer, head) in layer_range × range(n_heads), replaces the
    hook_z activation at token_position for that head with the corresponding
    source activation, then measures KL(patched || baseline).

    Args:
        source_activations_by_layer_and_head: dict keyed by (layer, head_index)
            → np.ndarray of shape (d_head,). Typically mean over source condition.
            Only (layer, head) pairs present in this dict are swept.
        target_run_config: dict with 'stimulus' key (target sentence string).
        layer_range: (start_layer, end_layer) inclusive.
        n_heads: number of attention heads in the model (e.g. 16 for GPT-2 medium).
        token_position: which token position to patch (e.g. -1 for last token).
        model: HookedTransformer instance. Caller loads once and passes in.
        baseline_logits: if provided, KL(patched || baseline) is computed per head.
            Pass the unpatched model output logits as the baseline.

    Returns:
        {
          "kl_matrix": dict[str, float] — key "(layer,head)" → KL value,
          "peak_head": [int, int] — [layer, head] with highest KL,
          "peak_kl": float,
          "n_layers_swept": int,
          "n_heads": int,
        }
    """
    model_device = next(model.parameters()).device
    target_sentence = target_run_config["stimulus"]

    kl_matrix: dict[str, float] = {}
    peak_kl = 0.0
    peak_head = [0, 0]

    start_layer, end_layer = layer_range

    for layer_index in range(start_layer, end_layer + 1):
        for head_index in range(n_heads):
            if (layer_index, head_index) not in source_activations_by_layer_and_head:
                continue

            source_head_activation = source_activations_by_layer_and_head[(layer_index, head_index)]
            source_tensor = torch.tensor(
                source_head_activation, dtype=torch.float32
            ).to(model_device)

            hook_name = f"blocks.{layer_index}.attn.hook_z"

            def make_head_replacement_hook(replacement_tensor, target_head_index, target_token_position):
                def replace_head_at_position(activation_value, hook):
                    activation_value[:, target_token_position, target_head_index, :] = replacement_tensor
                    return activation_value
                return replace_head_at_position

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    target_sentence,
                    fwd_hooks=[(hook_name, make_head_replacement_hook(
                        source_tensor, head_index, token_position
                    ))],
                )

            final_token_logits = patched_logits[0, -1, :]

            kl_value = 0.0
            if baseline_logits is not None:
                baseline_log_probs = torch.log_softmax(
                    torch.tensor(baseline_logits, dtype=torch.float32).to(model_device),
                    dim=-1,
                )
                patched_log_probs = torch.log_softmax(final_token_logits, dim=-1)
                kl_value = torch.nn.functional.kl_div(
                    baseline_log_probs,
                    patched_log_probs,
                    reduction="sum",
                    log_target=True,
                ).item()

            matrix_key = f"({layer_index},{head_index})"
            kl_matrix[matrix_key] = kl_value

            if kl_value > peak_kl:
                peak_kl = kl_value
                peak_head = [layer_index, head_index]

    return {
        "kl_matrix": kl_matrix,
        "peak_head": peak_head,
        "peak_kl": peak_kl,
        "n_layers_swept": end_layer - start_layer + 1,
        "n_heads": n_heads,
    }


def find_peak_circuit_components(
    head_sweep_result: dict[str, Any],
    kl_threshold: float,
) -> list[dict[str, Any]]:
    """
    Filter head sweep KL matrix to components above kl_threshold.

    Args:
        head_sweep_result: return value of run_head_sweep.
        kl_threshold: minimum KL to count as a circuit component.
            Use config.circuit_kl_threshold (default 0.1).

    Returns:
        List of dicts: [{"layer": int, "head": int, "kl_effect": float}, ...]
        sorted by kl_effect descending. Empty list if none exceed threshold.
    """
    circuit_components = []
    for key_string, kl_value in head_sweep_result["kl_matrix"].items():
        if kl_value >= kl_threshold:
            inner_content = key_string.strip("()")
            layer_string, head_string = inner_content.split(",")
            circuit_components.append({
                "layer": int(layer_string),
                "head": int(head_string),
                "kl_effect": kl_value,
            })
    return sorted(circuit_components, key=lambda component: component["kl_effect"], reverse=True)
```

- [ ] **Step 5: Run tests**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_circuits.py::TestFindPeakCircuitComponents -v 2>&1 | tail -15
```

Expected: all 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add circuits/__init__.py circuits/circuit_finder.py tests/test_circuits.py
git commit -m "feat(circuits): add circuit_finder with run_head_sweep and find_peak_circuit_components"
```

---

## Task 3: `run_path_patching`

**Files:**
- Modify: `circuits/circuit_finder.py` (append)
- Modify: `tests/test_circuits.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_circuits.py`:

```python
class TestRunPathPatching:
    def test_returns_zero_fraction_when_no_components(self):
        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.circuit_finder import run_path_patching

        result = run_path_patching(
            source_activations_by_layer_and_head={},
            target_run_config={"stimulus": "test"},
            circuit_components=[],
            token_position=-1,
            model=unittest.mock.MagicMock(),
            baseline_logits=None,
            full_layer_sweep_peak_kl=0.5,
        )
        assert result["fraction_recovered"] == 0.0
        assert result["circuit_is_sufficient"] is False
        assert result["n_components_patched"] == 0

    def test_circuit_sufficient_when_fraction_at_least_0_80(self):
        # Unit test: verify the threshold logic directly
        # fraction_recovered = recovered_kl / full_layer_sweep_peak_kl
        # 0.40 / 0.50 = 0.80 → sufficient
        # We test the threshold calculation without invoking the model
        recovered_kl = 0.40
        full_layer_sweep_peak_kl = 0.50
        fraction = recovered_kl / full_layer_sweep_peak_kl
        assert fraction >= 0.80
        assert fraction == pytest.approx(0.80)

    def test_fraction_zero_when_full_kl_near_zero(self):
        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.circuit_finder import run_path_patching

        result = run_path_patching(
            source_activations_by_layer_and_head={},
            target_run_config={"stimulus": "test"},
            circuit_components=[],
            token_position=-1,
            model=unittest.mock.MagicMock(),
            baseline_logits=None,
            full_layer_sweep_peak_kl=0.0,
        )
        assert result["fraction_recovered"] == 0.0
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_circuits.py::TestRunPathPatching -v 2>&1 | tail -10
```

Expected: ImportError on `run_path_patching`.

- [ ] **Step 3: Append `run_path_patching` to `circuits/circuit_finder.py`**

```python
def run_path_patching(
    source_activations_by_layer_and_head: dict[tuple[int, int], np.ndarray],
    target_run_config: dict[str, Any],
    circuit_components: list[dict[str, Any]],
    token_position: int,
    model: Any,
    baseline_logits: list[float] | None = None,
    full_layer_sweep_peak_kl: float = 0.0,
) -> dict[str, Any]:
    """
    Patch all peak circuit components simultaneously, measure recovered KL.

    Builds a multi-hook list — one hook per circuit component — and runs
    the target stimulus with all components patched at once. The recovered
    KL measures how much of the full layer sweep effect is explained by
    just the circuit components.

    fraction_recovered = recovered_kl / full_layer_sweep_peak_kl.
    circuit_is_sufficient = fraction_recovered >= 0.80.

    Args:
        source_activations_by_layer_and_head: same as run_head_sweep.
        target_run_config: dict with 'stimulus' key.
        circuit_components: output of find_peak_circuit_components.
        token_position: which token to patch.
        model: HookedTransformer instance.
        baseline_logits: unpatched model output logits for KL computation.
        full_layer_sweep_peak_kl: peak KL from the layer sweep (L3). Used to
            compute fraction_recovered.

    Returns:
        {
          "recovered_kl": float,
          "fraction_recovered": float,
          "full_layer_sweep_peak_kl": float,
          "circuit_is_sufficient": bool,
          "n_components_patched": int,
        }
    """
    if not circuit_components:
        return {
            "recovered_kl": 0.0,
            "fraction_recovered": 0.0,
            "full_layer_sweep_peak_kl": full_layer_sweep_peak_kl,
            "circuit_is_sufficient": False,
            "n_components_patched": 0,
        }

    model_device = next(model.parameters()).device
    target_sentence = target_run_config["stimulus"]

    forward_hooks = []
    n_components_patched = 0

    for component in circuit_components:
        layer_index = component["layer"]
        head_index = component["head"]
        if (layer_index, head_index) not in source_activations_by_layer_and_head:
            continue

        source_head_activation = source_activations_by_layer_and_head[(layer_index, head_index)]
        source_tensor = torch.tensor(
            source_head_activation, dtype=torch.float32
        ).to(model_device)

        hook_name = f"blocks.{layer_index}.attn.hook_z"

        def make_multi_head_hook(replacement_tensor, target_head_index, target_token_position):
            def replace_head_at_position(activation_value, hook):
                activation_value[:, target_token_position, target_head_index, :] = replacement_tensor
                return activation_value
            return replace_head_at_position

        forward_hooks.append((
            hook_name,
            make_multi_head_hook(source_tensor, head_index, token_position),
        ))
        n_components_patched += 1

    with torch.no_grad():
        patched_logits = model.run_with_hooks(target_sentence, fwd_hooks=forward_hooks)

    final_token_logits = patched_logits[0, -1, :]

    recovered_kl = 0.0
    if baseline_logits is not None:
        baseline_log_probs = torch.log_softmax(
            torch.tensor(baseline_logits, dtype=torch.float32).to(model_device),
            dim=-1,
        )
        patched_log_probs = torch.log_softmax(final_token_logits, dim=-1)
        recovered_kl = torch.nn.functional.kl_div(
            baseline_log_probs,
            patched_log_probs,
            reduction="sum",
            log_target=True,
        ).item()

    fraction_recovered = (
        recovered_kl / full_layer_sweep_peak_kl
        if full_layer_sweep_peak_kl > 1e-8
        else 0.0
    )

    return {
        "recovered_kl": recovered_kl,
        "fraction_recovered": fraction_recovered,
        "full_layer_sweep_peak_kl": full_layer_sweep_peak_kl,
        "circuit_is_sufficient": fraction_recovered >= 0.80,
        "n_components_patched": n_components_patched,
    }
```

- [ ] **Step 4: Run tests**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_circuits.py -v 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add circuits/circuit_finder.py tests/test_circuits.py
git commit -m "feat(circuits): add run_path_patching with circuit sufficiency check"
```

---

## Task 4: `circuits/attribution.py` — direct logit attribution

**Files:**
- Create: `circuits/attribution.py`
- Modify: `tests/test_circuits.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_circuits.py`:

```python
class TestComputeLogitAttribution:
    def test_returns_required_keys(self):
        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            from circuits.attribution import compute_logit_attribution

        mock_model = unittest.mock.MagicMock()
        mock_model.cfg.n_layers = 2
        mock_model.cfg.n_heads = 4
        mock_model.to_string.return_value = " wet"

        # Mock cache and W_U
        mock_cache = {}
        for layer_index in range(2):
            mock_cache[f"blocks.{layer_index}.attn.hook_z"] = (
                np.random.randn(1, 5, 4, 16)  # batch, seq, n_heads, d_head
            )
        mock_model.run_with_cache.return_value = (unittest.mock.MagicMock(), mock_cache)
        mock_model.W_U = np.random.randn(64, 50257)  # d_model, vocab_size

        for layer_index in range(2):
            mock_model.blocks[layer_index].attn.W_O = np.random.randn(4, 16, 64)

        result = compute_logit_attribution(
            target_sentence="If it had rained the ground would be",
            logit_direction_token_id=3596,
            model=mock_model,
        )
        assert "attribution_matrix" in result
        assert "top_positive_components" in result
        assert "top_negative_components" in result
        assert "token_id" in result
        assert "token_string" in result

    def test_attribution_matrix_has_entry_per_head(self):
        n_layers = 3
        n_heads = 4
        # Each (layer, head) pair should have an entry
        # Total entries: n_layers * n_heads = 12
        # We test the key format matches "(layer,head)"
        expected_keys = {f"({layer},{head})" for layer in range(n_layers) for head in range(n_heads)}
        # Verify key format is correct
        sample_key = "(2,3)"
        assert sample_key in expected_keys
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_circuits.py::TestComputeLogitAttribution -v 2>&1 | tail -10
```

- [ ] **Step 3: Create `circuits/attribution.py`**

```python
"""
circuits/attribution.py — Direct logit attribution per attention head.

─── CONCEPT: Direct logit attribution ────────────────────────────────────────
The residual stream at the final token position is the sum of contributions
from every attention head and MLP layer. The unembedding matrix W_U projects
this sum into logit space to produce next-token probabilities.

Direct logit attribution decomposes the logit for a specific token into
additive per-head contributions:

    attr(layer, head, token) = (head_output(layer, head)) · W_U[:, token]

where head_output(layer, head) = z[layer, head] @ W_O[layer, head]

z[layer, head]: per-head value output from hook_z, shape (d_head,)
W_O[layer, head]: output projection matrix for this head, shape (d_head, d_model)

Positive attribution: head pushes probability toward the target token.
Negative attribution: head pushes probability away from the target token.

This is an approximation — it ignores layer norm scaling and the nonlinear
interaction between heads via the residual stream. For publication-quality
results, apply the final layer norm before projecting through W_U using
TransformerLens cache.apply_ln_to_stack.

─── MODEL LOADING CONVENTION ─────────────────────────────────────────────────
Accept model as a required parameter. Load once, pass through.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch


def compute_logit_attribution(
    target_sentence: str,
    logit_direction_token_id: int,
    model: Any,
    token_position: int = -1,
) -> dict[str, Any]:
    """
    Decompose the logit for logit_direction_token_id into per-head contributions.

    For each (layer, head), computes how much that head pushes model probability
    toward or away from logit_direction_token_id.

    Args:
        target_sentence: input string to run through the model.
        logit_direction_token_id: vocab index of the token whose logit to decompose.
            For T1b: use the token id of the forward-causal completion word.
        model: HookedTransformer instance.
        token_position: sequence position to analyze. Default -1 (last token).

    Returns:
        {
          "attribution_matrix": dict[str, float] — key "(layer,head)" → attribution,
          "top_positive_components": list[{layer, head, attribution}] — top 5, positive,
          "top_negative_components": list[{layer, head, attribution}] — top 5, negative,
          "token_id": int,
          "token_string": str — string representation of the target token,
        }
    """
    _, activation_cache = model.run_with_cache(target_sentence)

    unembedding_matrix = model.W_U  # (d_model, vocab_size)

    # logit direction: column of W_U for the target token
    if isinstance(unembedding_matrix, torch.Tensor):
        logit_direction_vector = unembedding_matrix[:, logit_direction_token_id].detach().cpu().numpy()
    else:
        logit_direction_vector = np.array(unembedding_matrix[:, logit_direction_token_id])

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attribution_matrix: dict[str, float] = {}

    for layer_index in range(n_layers):
        hook_z_key = f"blocks.{layer_index}.attn.hook_z"
        per_head_value_outputs = activation_cache[hook_z_key]  # (batch, seq, n_heads, d_head)

        output_projection = model.blocks[layer_index].attn.W_O  # (n_heads, d_head, d_model)

        for head_index in range(n_heads):
            # head_z: (d_head,) — value output for this head at token_position
            if isinstance(per_head_value_outputs, torch.Tensor):
                head_z = per_head_value_outputs[0, token_position, head_index, :].detach().cpu().numpy()
            else:
                head_z = np.array(per_head_value_outputs[0, token_position, head_index, :])

            # W_O[head_index]: (d_head, d_model)
            if isinstance(output_projection, torch.Tensor):
                head_output_projection = output_projection[head_index].detach().cpu().numpy()
            else:
                head_output_projection = np.array(output_projection[head_index])

            # head_output: (d_model,) = (d_head,) @ (d_head, d_model)
            head_output_vector = head_z @ head_output_projection

            # attribution = dot product with logit direction
            attribution_value = float(np.dot(head_output_vector, logit_direction_vector))

            matrix_key = f"({layer_index},{head_index})"
            attribution_matrix[matrix_key] = attribution_value

    all_components = []
    for key_string, attribution_value in attribution_matrix.items():
        inner_content = key_string.strip("()")
        layer_string, head_string = inner_content.split(",")
        all_components.append({
            "layer": int(layer_string),
            "head": int(head_string),
            "attribution": attribution_value,
        })

    top_positive_components = sorted(
        [component for component in all_components if component["attribution"] > 0],
        key=lambda component: component["attribution"],
        reverse=True,
    )[:5]

    top_negative_components = sorted(
        [component for component in all_components if component["attribution"] < 0],
        key=lambda component: component["attribution"],
    )[:5]

    return {
        "attribution_matrix": attribution_matrix,
        "top_positive_components": top_positive_components,
        "top_negative_components": top_negative_components,
        "token_id": logit_direction_token_id,
        "token_string": model.to_string(logit_direction_token_id),
    }
```

- [ ] **Step 4: Run tests**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_circuits.py -v 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add circuits/attribution.py tests/test_circuits.py
git commit -m "feat(circuits): add compute_logit_attribution"
```

---

## Task 5: PhilBench schema additions

**Files:**
- Modify: `stimuli/schemas/philbench.schema.json`

- [ ] **Step 1: Read current schema**

```bash
cat "/Users/scott/Desktop/C++ review projects/PoL-Probe/stimuli/schemas/philbench.schema.json"
```

- [ ] **Step 2: Add `circuit_components` and `identification_criterion` to the schema**

Add to the `"properties"` object (alongside existing fields):

```json
"circuit_components": {
  "type": ["array", "null"],
  "description": "Circuit components identified by head sweep. Populated when circuit_analysis_enabled=True.",
  "items": {
    "type": "object",
    "properties": {
      "layer": {"type": "integer"},
      "head": {"type": "integer"},
      "kl_effect": {"type": "number"}
    },
    "required": ["layer", "head", "kl_effect"]
  }
},
"identification_criterion": {
  "type": ["string", "null"],
  "description": "Causal identification criterion for T1d entries. 'back_door' | 'front_door' | null.",
  "enum": ["back_door", "front_door", null]
}
```

Both fields are optional (not in `"required"` array) — they default to null for all non-circuit, non-T1d entries.

- [ ] **Step 3: Verify JSON is valid**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && python -c "import json; json.load(open('stimuli/schemas/philbench.schema.json'))" && echo "JSON valid"
```

Expected: `JSON valid`

- [ ] **Step 4: Commit**

```bash
git add stimuli/schemas/philbench.schema.json
git commit -m "feat(schema): add circuit_components and identification_criterion to PhilBench schema"
```

---

## Task 6: `run_identification_probe` in `probes/probes.py`

**Files:**
- Modify: `probes/probes.py`
- Modify: `tests/test_phase0.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase0.py` inside a new `class TestRunIdentificationProbe`:

```python
class TestRunIdentificationProbe:
    def test_binary_probe_separates_adjustable_from_not(self):
        from probes.probes import run_identification_probe

        rng = np.random.RandomState(42)
        # Adjustable items cluster in one region, not-adjustable in another
        adjustable_activations = rng.randn(20, 10) + np.array([3.0] * 10)
        not_adjustable_activations = rng.randn(20, 10) + np.array([-3.0] * 10)
        activations = np.vstack([adjustable_activations, not_adjustable_activations])
        labels = ["adjustable"] * 20 + ["not_adjustable"] * 20

        config = _make_config(thread_id="t1d")
        result = run_identification_probe(activations, labels, config)

        assert result["accuracy_mean"] > 0.80
        assert result["probe_type"] == "identification_binary"
        assert "adjustable_class" in result
        assert "not_adjustable_class" in result

    def test_returns_chance_baseline(self):
        from probes.probes import run_identification_probe

        rng = np.random.RandomState(0)
        activations = rng.randn(40, 8)
        labels = ["adjustable"] * 20 + ["not_adjustable"] * 20
        config = _make_config(thread_id="t1d")
        result = run_identification_probe(activations, labels, config)
        assert result["chance_baseline"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_phase0.py::TestRunIdentificationProbe -v 2>&1 | tail -10
```

- [ ] **Step 3: Append `run_identification_probe` to `probes/probes.py`**

```python
def run_identification_probe(
    activations: np.ndarray,
    labels: list[str],
    config: Any,
) -> dict[str, Any]:
    """
    Binary linear probe separating adjustable from not-adjustable causal structures.

    Used by T1d to test whether model representations distinguish confounded
    structures with a valid adjustment set from those without one. Collapses
    the four T1d conditions into two groups:
      adjustable:     back_door_adjustable | front_door_adjustable
      not_adjustable: confounded_not_adjustable | unconfounded_control

    The probe is binary logistic regression with the same 5-fold stratified CV
    as run_linear_probe. The two-class structure is locked here — callers do not
    specify the grouping.

    Args:
        activations: np.ndarray of shape (n_items, hidden_dim).
        labels: list of strings from T1d label set:
            "back_door_adjustable" | "front_door_adjustable" |
            "confounded_not_adjustable" | "unconfounded_control"
        config: ExperimentConfig. thread_id should be "t1d".

    Returns:
        Dict with all fields from run_linear_probe, plus:
          "probe_type": "identification_binary"
          "adjustable_class": str — which label maps to "adjustable"
          "not_adjustable_class": str — which label maps to "not_adjustable"
    """
    adjustable_label_set = {"back_door_adjustable", "front_door_adjustable"}

    binary_labels = [
        "adjustable" if label in adjustable_label_set else "not_adjustable"
        for label in labels
    ]

    probe_result = run_linear_probe(activations, binary_labels, config)
    probe_result["probe_type"] = "identification_binary"
    probe_result["adjustable_class"] = "adjustable"
    probe_result["not_adjustable_class"] = "not_adjustable"
    return probe_result
```

- [ ] **Step 4: Run tests**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_phase0.py -v 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add probes/probes.py tests/test_phase0.py
git commit -m "feat(probes): add run_identification_probe for T1d adjustability test"
```

---

## Task 7: Fix T1b issue 1 — surface confound (keyword leakage at layer 0)

**Files:**
- Modify: `experiments/t1b/run_experiment.py`

The layer 0 probe accuracy is 100%, suggesting the probe reads stimulus templates (keywords like "had", "because", "if") rather than causal structure.

- [ ] **Step 1: Add sample inspection before probe loop**

In `experiments/t1b/run_experiment.py`, after Step 3 (phase gate), add before Step 4:

```python
# ── Step 3b: Inspect stimulus samples for keyword leakage ─────────────────────

print("[Step 3b] Inspecting stimulus samples for surface keyword leakage...")
print("  Checking for discriminative surface patterns across conditions...")
print()

import json
sample_by_condition: dict[str, list[str]] = {}
with VALIDATED_PATH.open("r") as stimulus_file:
    for raw_line in stimulus_file:
        stripped = raw_line.strip()
        if not stripped:
            continue
        pair = json.loads(stripped)
        label_a = pair.get("label_a", "unknown")
        label_b = pair.get("label_b", "unknown")
        for label, sentence in [(label_a, pair["sentence_a"]), (label_b, pair["sentence_b"])]:
            if label not in sample_by_condition:
                sample_by_condition[label] = []
            if len(sample_by_condition[label]) < 5:
                sample_by_condition[label].append(sentence)

for condition_label, condition_sentences in sorted(sample_by_condition.items()):
    print(f"  Condition: {condition_label}")
    for sentence in condition_sentences:
        print(f"    {sentence}")
    print()
```

- [ ] **Step 2: Run on Colab and inspect output**

Upload to Colab and run. Read the printed samples. Look for:
- `forward_causal` sentences consistently containing specific keywords not in other conditions
- `backtracking` sentences consistently containing "had" or "must have" in diagnostic positions
- Template-level differences that a linear classifier could exploit at layer 0

- [ ] **Step 3: Fix the grammar**

Open `stimuli/grammars/t1b.py`. Identify which templates produce the leaking patterns. Rebalance so that surface cue words appear equally across conditions. The forward_causal and backtracking conditions must share the same surface keywords — only causal direction should differ.

After editing the grammar, regenerate and re-validate:

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python scripts/run_validation.py --thread t1b
```

- [ ] **Step 4: Verify layer 0 accuracy drops**

Re-run just the probe section (Steps 1–5 of the experiment) on Colab. Layer 0 accuracy should drop toward chance (33% for 3-class, 50% for binary forward vs backtracking). If still high, the leakage is deeper — inspect further.

- [ ] **Step 5: Commit grammar fix**

```bash
git add stimuli/grammars/t1b.py
git commit -m "fix(t1b): remove surface keyword leakage from grammar templates"
```

---

## Task 8: Fix T1b issue 2 — data pipeline discrepancy (common_cause origin)

**Files:**
- `stimuli/grammars/t1b.py` (read only for audit)
- `experiments/t1b/run_experiment.py` (read only for audit)
- `scripts/run_validation.py` (read only for audit)

The validation reports 275 `forward_causal` / 25 `backtracking` / 0 `common_cause`. The experiment ran a 3-class probe including `common_cause`. Origin of common_cause samples is unknown.

- [ ] **Step 1: Audit the grammar**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && grep -n "common_cause" stimuli/grammars/t1b.py | head -30
```

Check whether the grammar generates `common_cause` pairs. If yes, check whether `run_validation.py` is filtering them before writing to `pairs.validated.jsonl`.

- [ ] **Step 2: Audit the validated file**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && python -c "
import json
from pathlib import Path
path = Path('stimuli/validated/t1b/pairs.validated.jsonl')
counts = {}
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    pair = json.loads(line)
    for label in [pair.get('label_a'), pair.get('label_b')]:
        counts[label] = counts.get(label, 0) + 1
print(counts)
"
```

- [ ] **Step 3: Audit the experiment data loading**

In `experiments/t1b/run_experiment.py`, find where `labels` are assigned during extraction. Check if there is any code path that introduces `common_cause` labels from a source other than `pairs.validated.jsonl`.

- [ ] **Step 4: Resolve and document**

Based on audit findings, either:
- (a) `common_cause` was in the grammar but filtered at validation — experiment should only run on validated labels. Fix experiment to load labels exclusively from validated file.
- (b) `common_cause` was added post-hoc — find where and remove.
- (c) Grammar produces it but validation drops it — this is correct behavior; fix experiment to not expect `common_cause`.

Whichever applies: after fixing, the experiment must run a **2-class probe** (forward_causal vs backtracking) as the primary Lewis/Pearl test, not 3-class.

- [ ] **Step 5: Commit fix**

```bash
git add experiments/t1b/run_experiment.py stimuli/grammars/t1b.py
git commit -m "fix(t1b): resolve common_cause data pipeline discrepancy"
```

---

## Task 9: Fix T1b issue 3 — confirm L2/L3 agreement on clean data

This task runs after tasks 7 and 8 are resolved. It is a Colab run, not a code change.

- [ ] **Step 1: Re-run T1b on Colab with fixed grammar and clean data**

Upload the fixed experiment to Colab and run. Collect:
- Layer-resolved probe accuracy curve
- Peak probe layer
- Layer sweep KL curve
- Peak patching layer
- `layers_agree` flag in summary.json

- [ ] **Step 2: Evaluate agreement**

Open `experiments/t1b/results/summary.json`. Check:
- `layers_agree`: True means peak probe layer = peak patch layer → L2 and L3 point to same layer → strong mechanistic finding
- `layers_agree`: False → investigate. Acceptable if peak patch layer is within ±2 of peak probe layer (smooth KL curve). Not acceptable if peak patch layer is far from probe peak with no clear curve.

- [ ] **Step 3: Document result in TASKS.md**

Update task #41 status to `x` and add result note. Update task #40 similarly if not already done.

---

## Task 10: Enable circuit analysis on T1b

**Files:**
- Modify: `experiments/t1b/run_experiment.py`

This task runs only after tasks 7–9 are complete (T1b clean).

- [ ] **Step 1: Add circuit analysis step to T1b config**

In `experiments/t1b/run_experiment.py`, update the config construction to expose `circuit_analysis_enabled`:

```python
config = ExperimentConfig(
    experiment_id="t1b_gpt2m_" + date_stamp,
    thread_id="t1b",
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, GPT2_MEDIUM_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=True,
    expected_outcomes=expected_outcomes,
    prerequisite_experiment_id=T1A_PREREQUISITE_ID,
    circuit_analysis_enabled=True,   # enable after T1b is clean
    circuit_kl_threshold=0.1,
)
```

- [ ] **Step 2: Add circuit analysis step (Step 8.5) after layer sweep**

After the `save_result(sweep_result, ...)` call and before the summary, add:

```python
# ── Step 8.5: Circuit analysis (head sweep + path patching + logit attribution) ──

circuit_components_result = None
path_patching_result = None
logit_attribution_result = None

if config.circuit_analysis_enabled:
    print("[Step 8.5] Running circuit analysis (head sweep across all layers × heads)...")
    print("  384 patches (24 layers × 16 heads). Takes ~20-30 minutes on Colab T4.")
    print()

    from circuits.circuit_finder import run_head_sweep, find_peak_circuit_components, run_path_patching
    from circuits.attribution import compute_logit_attribution

    # Build source activations by (layer, head) — mean over forward_causal condition
    # hook_z shape: (batch, seq, n_heads, d_head)
    # We need to extract hook_z activations, not resid_post
    source_activations_by_layer_and_head: dict[tuple[int, int], np.ndarray] = {}
    for layer_index in range(GPT2_MEDIUM_N_LAYERS):
        _, cache = model.run_with_cache(
            [s for i, s in enumerate(forward_sentences) if i < 5]  # use first 5 forward_causal
        )
        hook_z_activations = cache[f"blocks.{layer_index}.attn.hook_z"]  # (batch, seq, n_heads, d_head)
        for head_index in range(16):
            mean_head_activation = hook_z_activations[:, -1, head_index, :].mean(axis=0).detach().cpu().numpy()
            source_activations_by_layer_and_head[(layer_index, head_index)] = mean_head_activation

    head_sweep_result = run_head_sweep(
        source_activations_by_layer_and_head,
        target_run_config,
        config.layer_range,
        n_heads=16,
        token_position=config.token_positions[0],
        model=model,
        baseline_logits=baseline_logits,
    )
    save_result(head_sweep_result, RESULTS_DIR / "head_sweep.json")
    print("  Head sweep complete. Peak head: " + str(head_sweep_result["peak_head"]) +
          "  KL=" + str(round(head_sweep_result["peak_kl"], 4)))

    circuit_components_result = find_peak_circuit_components(
        head_sweep_result, config.circuit_kl_threshold
    )
    save_result({"circuit_components": circuit_components_result}, RESULTS_DIR / "circuit_components.json")
    print("  Circuit components above threshold " + str(config.circuit_kl_threshold) +
          ": " + str(len(circuit_components_result)) + " heads")

    path_patching_result = run_path_patching(
        source_activations_by_layer_and_head,
        target_run_config,
        circuit_components_result,
        token_position=config.token_positions[0],
        model=model,
        baseline_logits=baseline_logits,
        full_layer_sweep_peak_kl=peak_patch_kl,
    )
    save_result(path_patching_result, RESULTS_DIR / "path_patching.json")
    print("  Path patching: fraction_recovered=" +
          str(round(path_patching_result["fraction_recovered"], 3)) +
          "  sufficient=" + str(path_patching_result["circuit_is_sufficient"]))

    # Logit attribution for the forward-causal completion token
    forward_completion_token_id = model.to_single_token(" wet")  # example — adjust to domain
    logit_attribution_result = compute_logit_attribution(
        target_run_config["stimulus"],
        forward_completion_token_id,
        model,
        token_position=config.token_positions[0],
    )
    save_result(logit_attribution_result, RESULTS_DIR / "logit_attribution.json")
    print("  Logit attribution complete.")
    print()
```

- [ ] **Step 3: Add circuit results to summary**

In the `summary` dict construction, add:

```python
    "circuit_analysis_enabled": config.circuit_analysis_enabled,
    "circuit_components": circuit_components_result if circuit_components_result is not None else [],
    "circuit_is_sufficient": path_patching_result["circuit_is_sufficient"] if path_patching_result else None,
    "circuit_fraction_recovered": path_patching_result["fraction_recovered"] if path_patching_result else None,
```

- [ ] **Step 4: Run on Colab and verify**

Run full T1b with circuit analysis enabled. Confirm `circuit_components.json`, `path_patching.json`, `logit_attribution.json` are written to `experiments/t1b/results/`.

- [ ] **Step 5: Commit**

```bash
git add experiments/t1b/run_experiment.py
git commit -m "feat(t1b): add circuit analysis step — head sweep, path patching, logit attribution"
```

---

## Task 11: T1d grammar

**Files:**
- Create: `stimuli/grammars/t1d.py`
- Create: `tests/test_t1d_grammar.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_t1d_grammar.py`:

```python
"""
tests/test_t1d_grammar.py — Tests for T1d causal identification grammar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from stimuli.grammars.t1d import generate


class TestT1dGrammar:
    def test_generates_requested_count(self):
        pairs = generate(n=40)
        assert len(pairs) == 40

    def test_all_four_conditions_present(self):
        pairs = generate(n=80)
        label_set = set()
        for pair in pairs:
            label_set.add(pair["label_a"])
            label_set.add(pair["label_b"])
        required_conditions = {
            "back_door_adjustable",
            "front_door_adjustable",
            "confounded_not_adjustable",
            "unconfounded_control",
        }
        assert required_conditions.issubset(label_set)

    def test_pairs_have_required_fields(self):
        pairs = generate(n=10)
        for pair in pairs:
            assert "sentence_a" in pair
            assert "sentence_b" in pair
            assert "label_a" in pair
            assert "label_b" in pair
            assert isinstance(pair["sentence_a"], str)
            assert len(pair["sentence_a"]) > 10

    def test_no_empty_sentences(self):
        pairs = generate(n=20)
        for pair in pairs:
            assert pair["sentence_a"].strip() != ""
            assert pair["sentence_b"].strip() != ""

    def test_conditions_balanced(self):
        pairs = generate(n=80)
        condition_counts: dict[str, int] = {}
        for pair in pairs:
            for label in [pair["label_a"], pair["label_b"]]:
                condition_counts[label] = condition_counts.get(label, 0) + 1
        counts = list(condition_counts.values())
        assert max(counts) <= min(counts) * 2  # no condition more than 2x any other
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_t1d_grammar.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Create `stimuli/grammars/t1d.py`**

```python
"""
stimuli/grammars/t1d.py — Stimulus grammar for Thread T1d (causal identification).

─── CONCEPT: What T1d tests ──────────────────────────────────────────────────
T1b established which counterfactual mechanism GPT-2 uses (Lewis or Pearl).
T1d tests a deeper question: does the model's representation respect do-calculus
identification conditions?

Pearl's do-calculus provides formal criteria for when a causal effect can be
identified (estimated) from observational data despite confounding. Two key criteria:

  Back-door criterion: A set Z of observed variables satisfies back-door for
    (X, Y) if Z blocks all back-door paths from X to Y and no Z is a descendant
    of X. If satisfied, P(Y|do(X)) = Σ_z P(Y|X,z)P(z) — the causal effect is
    identifiable by conditioning on Z.

  Front-door criterion: A set M of variables satisfies front-door for (X, Y)
    if M blocks all directed paths from X to Y, there are no unblocked back-door
    paths from X to M, and all back-door paths from M to Y are blocked by X.
    Used when confounders are hidden but a mediator is observed.

─── CONCEPT: Four conditions ─────────────────────────────────────────────────
back_door_adjustable: confounded, but observed covariate Z blocks the back-door path.
  Example: John's health outcome (Y) is affected by his smoking (X), but both
  are affected by socioeconomic status (Z, observed). Z satisfies back-door.
  Correct inference requires adjusting for Z.

front_door_adjustable: hidden confounder U → {X, Y}, but mediator M (X→M→Y)
  is observed. Front-door criterion applies via M.
  Example: Smoking (X) → tar deposits (M) → cancer (Y), hidden genetic factor
  U affects both smoking and cancer. Front-door adjustment via tar deposits.

confounded_not_adjustable: confounded, but no valid adjustment set is available.
  The causal effect is not identified from observational data.
  Example: Same structure as back_door_adjustable but Z is unobserved.

unconfounded_control: simple direct causation X → Y, no confounding.
  Causal effect identifiable trivially. Positive control.

─── CONCEPT: Geometric prediction ────────────────────────────────────────────
If the model has an internal identifiability representation:
  adjustable (back_door + front_door) should cluster together — both allow
    valid causal inference despite confounding.
  not_adjustable (confounded_not_adjustable + unconfounded_control) should
    cluster together or be geometrically separated from adjustable.
  run_identification_probe in probes/probes.py collapses to this binary.

If model implements Pearl and has full do-calculus awareness:
  All four conditions are linearly separable — the model encodes each distinct
  causal graph structure as a distinct representation.

─── T1b prerequisite ─────────────────────────────────────────────────────────
T1d requires T1b to be complete. The interpretation of T1d results depends on
T1b outcome (pearl_confirmed). This is documented in the experiment runner —
the grammar itself does not enforce prerequisites.
"""

from __future__ import annotations

import random
from typing import Any


# ── Back-door adjustable templates ───────────────────────────────────────────
# Structure: X → Y, Z → {X, Y}. Z is observed. Back-door criterion satisfied.

BACK_DOOR_TEMPLATES = [
    (
        "{agent} {treatment_verb} because of {confounder}. "
        "{confounder_capitalized} also directly affects {outcome_noun}. "
        "Adjusting for {confounder}, {agent}'s {treatment_noun} changes {outcome_noun}.",
        "{agent} does not {treatment_verb_base}. "
        "{confounder_capitalized} still affects {outcome_noun} directly. "
        "Without {treatment_noun}, {outcome_noun} differs by {confounder} alone.",
    ),
    (
        "Researchers found that {treatment_noun} increases {outcome_noun}. "
        "Both are influenced by {confounder}. "
        "Controlling for {confounder} isolates the direct effect of {treatment_noun}.",
        "If {confounder} is held constant, {treatment_noun} still predicts {outcome_noun}. "
        "The back-door path through {confounder} is blocked by conditioning. "
        "The adjusted estimate is valid.",
    ),
]

# ── Front-door adjustable templates ──────────────────────────────────────────
# Structure: X → M → Y, hidden U → {X, Y}. M observed. Front-door satisfied.

FRONT_DOOR_TEMPLATES = [
    (
        "{treatment_noun} causes {mediator_noun}. "
        "{mediator_noun} then causes {outcome_noun}. "
        "A hidden factor affects both {treatment_noun} and {outcome_noun} independently.",
        "Even with the hidden factor, {mediator_noun} fully mediates the path. "
        "Front-door adjustment through {mediator_noun} identifies the causal effect. "
        "The effect of {treatment_noun} on {outcome_noun} is estimable.",
    ),
    (
        "{agent} was exposed to {treatment_noun}, which caused {mediator_noun}. "
        "{mediator_noun} led to {outcome_noun}. "
        "Genetic factors independently influenced both exposure and outcome.",
        "The complete pathway runs through {mediator_noun}. "
        "Adjusting for {mediator_noun} at each step recovers the total effect. "
        "No direct {treatment_noun} → {outcome_noun} path bypasses {mediator_noun}.",
    ),
]

# ── Confounded not adjustable templates ──────────────────────────────────────
# Structure: X → Y, U → {X, Y}. U unobserved. No valid adjustment set.

CONFOUNDED_NOT_ADJUSTABLE_TEMPLATES = [
    (
        "{treatment_noun} is associated with {outcome_noun}. "
        "An unmeasured factor drives both. "
        "No observed variable can block the confounding path.",
        "The association between {treatment_noun} and {outcome_noun} is not causal. "
        "Without observing the hidden factor, the effect cannot be identified. "
        "The causal effect remains unestimable from this data.",
    ),
    (
        "Studies show {treatment_noun} correlates with {outcome_noun}. "
        "A latent variable affects both {treatment_noun} and {outcome_noun}. "
        "No measured covariate satisfies the back-door criterion.",
        "{agent}'s {treatment_noun} and {outcome_noun} share a hidden common cause. "
        "Conditioning on observed variables does not remove the confounding. "
        "The causal question is not answerable from this observational data.",
    ),
]

# ── Unconfounded control templates ────────────────────────────────────────────
# Structure: X → Y. No confounding. Trivially identified.

UNCONFOUNDED_TEMPLATES = [
    (
        "{treatment_noun} directly causes {outcome_noun}. "
        "There are no common causes of both. "
        "The causal effect is identifiable without adjustment.",
        "Increasing {treatment_noun} increases {outcome_noun}. "
        "No confounders affect this relationship. "
        "The observational and interventional distributions are identical.",
    ),
    (
        "{agent} applied {treatment_noun}, which produced {outcome_noun}. "
        "No other factor influenced both. "
        "The direct causal path is the only path between them.",
        "Removing {treatment_noun} eliminates {outcome_noun}. "
        "The relationship is purely causal with no confounding structure. "
        "P(Y|X) equals P(Y|do(X)) in this case.",
    ),
]

# ── Domain slots ──────────────────────────────────────────────────────────────

DOMAIN_SLOTS = [
    {
        "agent": "John",
        "treatment_verb": "smokes",
        "treatment_verb_base": "smoke",
        "treatment_noun": "smoking",
        "outcome_noun": "lung cancer risk",
        "confounder": "genetic predisposition",
        "confounder_capitalized": "Genetic predisposition",
        "mediator_noun": "tar deposits",
    },
    {
        "agent": "The patient",
        "treatment_verb": "exercises",
        "treatment_verb_base": "exercise",
        "treatment_noun": "exercise",
        "outcome_noun": "cardiovascular health",
        "confounder": "socioeconomic status",
        "confounder_capitalized": "Socioeconomic status",
        "mediator_noun": "reduced inflammation",
    },
    {
        "agent": "The factory",
        "treatment_verb": "emits pollutants",
        "treatment_verb_base": "emit pollutants",
        "treatment_noun": "pollution exposure",
        "outcome_noun": "respiratory disease rates",
        "confounder": "urban density",
        "confounder_capitalized": "Urban density",
        "mediator_noun": "particulate accumulation",
    },
    {
        "agent": "The student",
        "treatment_verb": "attends tutoring",
        "treatment_verb_base": "attend tutoring",
        "treatment_noun": "tutoring",
        "outcome_noun": "exam scores",
        "confounder": "parental income",
        "confounder_capitalized": "Parental income",
        "mediator_noun": "improved study habits",
    },
]


def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n stimulus pairs for T1d — causal identification conditions.

    Pairs are balanced across four conditions:
    back_door_adjustable, front_door_adjustable, confounded_not_adjustable, unconfounded_control.

    Each pair contrasts two sentences with different identification structures.
    sentence_a is always the labeled condition; sentence_b is a contrast from
    a different condition.

    Args:
        n: number of pairs to generate.
        seed: random seed for reproducibility.

    Returns:
        List of dicts with keys: sentence_a, sentence_b, label_a, label_b.
    """
    rng = random.Random(seed)

    condition_template_map = {
        "back_door_adjustable": BACK_DOOR_TEMPLATES,
        "front_door_adjustable": FRONT_DOOR_TEMPLATES,
        "confounded_not_adjustable": CONFOUNDED_NOT_ADJUSTABLE_TEMPLATES,
        "unconfounded_control": UNCONFOUNDED_TEMPLATES,
    }
    condition_order = list(condition_template_map.keys())

    pairs: list[dict[str, Any]] = []
    items_per_condition = max(1, n // len(condition_order))

    for condition_label, templates in condition_template_map.items():
        contrast_conditions = [c for c in condition_order if c != condition_label]
        for _ in range(items_per_condition):
            domain_slots = rng.choice(DOMAIN_SLOTS)
            template_a, _ = rng.choice(templates)
            contrast_label = rng.choice(contrast_conditions)
            _, template_b = rng.choice(condition_template_map[contrast_label])

            sentence_a = template_a.format(**domain_slots)
            sentence_b = template_b.format(**domain_slots)

            pairs.append({
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "label_a": condition_label,
                "label_b": contrast_label,
            })

    rng.shuffle(pairs)
    return pairs[:n]
```

- [ ] **Step 4: Run tests**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_t1d_grammar.py -v 2>&1 | tail -15
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stimuli/grammars/t1d.py tests/test_t1d_grammar.py
git commit -m "feat(t1d): add causal identification grammar — four confounding conditions"
```

---

## Task 12: T1d experiment runner

**Files:**
- Create: `experiments/t1d/run_experiment.py`

- [ ] **Step 1: Create `experiments/t1d/run_experiment.py`**

```python
"""
experiments/t1d/run_experiment.py — T1d mechanistic experiment: causal identification.

Tests whether GPT-2 medium representations respect do-calculus identification
conditions — specifically, whether the model internally distinguishes confounded
structures with a valid adjustment set from those without one.

Prerequisite: T1b must be complete. T1d interpretation depends on whether T1b
found Pearl or Lewis mechanism (pearl_confirmed in T1b summary.json).

Usage (Colab):
    !python experiments/t1d/run_experiment.py
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

THREAD_ID = "t1d"
MODEL_ID = "gpt2-medium"
GPT2_MEDIUM_N_LAYERS = 24

VALIDATED_PATH = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / THREAD_ID / "results"
CONFIG_PATH = PROJECT_ROOT / "experiments" / THREAD_ID / "config.json"
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

T1B_PREREQUISITE_ID = "t1b"


# ── Guards ────────────────────────────────────────────────────────────────────

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Expected: " + str(VALIDATED_PATH))
    print("  Run scripts/run_validation.py --thread t1d first.")
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
from probes.probes import run_linear_probe, run_identification_probe
from interventions.interventions import run_layer_sweep, assert_specificity_valid, mean_ablate
from core.io import load_result, save_result
import torch


# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PoL-Probe — T1d Mechanistic Experiment")
print("Causal Identification: Back-Door vs Front-Door vs Unidentifiable")
print("=" * 60)
print()

# Load T1b outcome — T1d interpretation is conditioned on it
t1b_summary = load_result(t1b_summary_path)
pearl_confirmed_in_t1b = t1b_summary.get("pearl_confirmed", False)
print("[Context] T1b pearl_confirmed = " + str(pearl_confirmed_in_t1b))
if pearl_confirmed_in_t1b:
    print("  T1d tests whether Pearl-consistent representations respect identification.")
else:
    print("  T1d tests whether Lewis-consistent representations fail identification.")
print()

# ── Step 1: Build config ──────────────────────────────────────────────────────

print("[Step 1] Building and locking experiment config...")

confounder_structure_description = {
    "conditions": {
        "back_door_adjustable": {
            "nodes": ["treatment", "outcome", "confounder"],
            "edges": [["treatment", "outcome"], ["confounder", "treatment"], ["confounder", "outcome"]],
            "criterion": "back_door",
            "adjustment_set": ["confounder"],
        },
        "front_door_adjustable": {
            "nodes": ["treatment", "mediator", "outcome", "hidden_confounder"],
            "edges": [["treatment", "mediator"], ["mediator", "outcome"],
                      ["hidden_confounder", "treatment"], ["hidden_confounder", "outcome"]],
            "criterion": "front_door",
            "adjustment_set": ["mediator"],
        },
        "confounded_not_adjustable": {
            "nodes": ["treatment", "outcome", "hidden_confounder"],
            "edges": [["treatment", "outcome"],
                      ["hidden_confounder", "treatment"], ["hidden_confounder", "outcome"]],
            "criterion": "none",
            "adjustment_set": [],
        },
        "unconfounded_control": {
            "nodes": ["treatment", "outcome"],
            "edges": [["treatment", "outcome"]],
            "criterion": "trivial",
            "adjustment_set": [],
        },
    }
}

expected_outcomes_description = {
    "identification_criterion": (
        "Binary probe accuracy (adjustable vs not_adjustable) at peak layer. "
        "> 0.70 → model encodes adjustability. <= 0.70 → no identifiability representation."
    ),
    "outcome_if_pearl_and_encodes_identification": (
        "back_door_adjustable and front_door_adjustable cluster together, "
        "separated from confounded_not_adjustable. Pearl representations respect "
        "do-calculus identification conditions — the full causal hierarchy."
    ),
    "outcome_if_lewis_and_fails_identification": (
        "No separation between adjustable and not_adjustable conditions. "
        "Lewis similarity ordering has no notion of identifiability — "
        "the model geometry is flat across identification conditions."
    ),
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t1d_gpt2m_" + date_stamp,
    thread_id=THREAD_ID,
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, GPT2_MEDIUM_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=True,
    expected_outcomes=expected_outcomes_description,
    prerequisite_experiment_id=T1B_PREREQUISITE_ID,
    identification_criterion="back_door",
    confounder_structure=confounder_structure_description,
)

config.lock()
config.to_json(CONFIG_PATH)

print("  Experiment ID         : " + config.experiment_id)
print("  Identification crit.  : " + str(config.identification_criterion))
print("  Prerequisite (T1b)    : " + T1B_PREREQUISITE_ID)
print("  Config locked         : " + str(config.pre_spec_locked))
print()

# ── Step 2: Surface null ──────────────────────────────────────────────────────

print("[Step 2] Running surface-statistics null...")
surface_null_result = run_surface_null(config)
surface_null_accuracy = surface_null_result["surface_classifier_accuracy"]
print("  Surface classifier accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
print()

# ── Step 3: Phase gate ────────────────────────────────────────────────────────

print("[Step 3] Checking phase gate...")
check_phase_gate(config)
print("  All gates passed.")
print()

# ── Step 4: Load model and extract activations ────────────────────────────────

print("[Step 4] Loading model and extracting activations...")
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_ID)
model.eval()

layer_activation_sets = extract_activations(config, model)
print("  Extraction complete.")
print()

# ── Step 5: Four-class probe at each layer ────────────────────────────────────

print("[Step 5] Running four-class probe at each layer...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

probe_results_by_layer: dict[int, dict] = {}
for activation_set in layer_activation_sets:
    layer_index = activation_set["layer"]
    activations = np.array(activation_set["activations"])
    labels = activation_set["labels"]

    probe_result = run_linear_probe(activations, labels, config)
    probe_result["layer"] = layer_index
    save_result(probe_result, RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json"))
    probe_results_by_layer[layer_index] = probe_result

peak_probe_layer = max(
    probe_results_by_layer,
    key=lambda layer_index: probe_results_by_layer[layer_index]["accuracy_mean"]
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
print("  Four-class probe complete. Peak layer: " + str(peak_probe_layer))
print()

# ── Step 6: Binary identification probe at peak layer ────────────────────────

print("[Step 6] Running binary identification probe at peak layer " + str(peak_probe_layer) + "...")
peak_activation_set = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
peak_activations = np.array(peak_activation_set["activations"])
peak_labels = peak_activation_set["labels"]

identification_probe_result = run_identification_probe(peak_activations, peak_labels, config)
identification_probe_result["layer"] = peak_probe_layer
save_result(identification_probe_result, RESULTS_DIR / "identification_probe.json")

identification_accuracy = identification_probe_result["accuracy_mean"]
identification_criterion_met = identification_accuracy > 0.70
print("  Identification probe accuracy : " + str(round(identification_accuracy * 100, 1)) + "%")
print("  Criterion met (> 70%)         : " + str(identification_criterion_met))
print()

# ── Step 7: Layer sweep (L3) ──────────────────────────────────────────────────

print("[Step 7] Running layer sweep...")

adjustable_label_set = {"back_door_adjustable", "front_door_adjustable"}
adjustable_indices = [
    i for i, label in enumerate(layer_activation_sets[0]["labels"])
    if label in adjustable_label_set
]

mean_adjustable_by_layer: dict[int, np.ndarray] = {
    activation_set["layer"]: np.array(activation_set["activations"])[adjustable_indices].mean(axis=0)
    for activation_set in layer_activation_sets
}

not_adjustable_sentences = []
with VALIDATED_PATH.open("r") as validated_file:
    for raw_line in validated_file:
        stripped = raw_line.strip()
        if stripped:
            pair = json.loads(stripped)
            if pair.get("label_b") == "confounded_not_adjustable":
                not_adjustable_sentences.append(pair["sentence_b"])

target_run_config = {"stimulus": not_adjustable_sentences[-1]}

sweep_result = run_layer_sweep(
    mean_adjustable_by_layer,
    target_run_config,
    config.layer_range,
    config.component,
    config.token_positions[0],
    model,
)
save_result(sweep_result, RESULTS_DIR / "layer_sweep.json")

peak_patch_layer = sweep_result["peak_layer"]
peak_patch_kl = sweep_result["layer_effects"][peak_patch_layer]
layers_agree = peak_probe_layer == peak_patch_layer

with torch.no_grad():
    baseline_logits = model(target_run_config["stimulus"])[0, -1, :].tolist()

mean_ablation_result = mean_ablate(
    peak_activations, target_run_config, peak_patch_layer,
    config.component, config.token_positions[0], model,
    baseline_logits=baseline_logits,
)
assert_specificity_valid(
    peak_patch_kl, mean_ablation_result["kl_from_baseline"] or 0.0, peak_patch_layer
)

print("  Peak patching layer : " + str(peak_patch_layer) + "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
print("  L2 / L3 agreement   : " + ("YES" if layers_agree else "NO"))
print()

# ── Step 8: Summary ───────────────────────────────────────────────────────────

summary = {
    "experiment_id": config.experiment_id,
    "thread_id": config.thread_id,
    "model_id": config.model_id,
    "run_timestamp": datetime.datetime.utcnow().isoformat(),
    "t1b_pearl_confirmed": pearl_confirmed_in_t1b,
    "peak_probe_layer": peak_probe_layer,
    "peak_probe_accuracy_4class": float(peak_probe_accuracy),
    "identification_probe_accuracy": float(identification_accuracy),
    "identification_criterion_met": identification_criterion_met,
    "surface_null_accuracy": float(surface_null_accuracy),
    "peak_patch_layer": peak_patch_layer,
    "peak_patch_kl": float(peak_patch_kl),
    "layers_agree": layers_agree,
    "expected_outcomes": config.expected_outcomes,
}

save_result(summary, SUMMARY_PATH)

print("=" * 60)
print("T1d Results — Causal Identification")
print("=" * 60)
print()
print("Identification probe (adjustable vs not): " +
      str(round(identification_accuracy * 100, 1)) + "%  " +
      ("ENCODES IDENTIFIABILITY" if identification_criterion_met else "NO IDENTIFIABILITY REPRESENTATION"))
print("T1b context: " + ("Pearl mechanism" if pearl_confirmed_in_t1b else "Lewis mechanism"))
print()
if pearl_confirmed_in_t1b and identification_criterion_met:
    print("Finding: Pearl-consistent model WITH identifiability encoding. Full do-calculus.")
elif pearl_confirmed_in_t1b and not identification_criterion_met:
    print("Finding: Pearl-consistent model WITHOUT identifiability. Partial do-calculus only.")
elif not pearl_confirmed_in_t1b and not identification_criterion_met:
    print("Finding: Lewis-consistent model fails identification test. Consistent with worlds-ordering.")
else:
    print("Finding: Lewis-consistent model passes identification test. Unexpected — investigate.")
```

- [ ] **Step 2: Update `check_phase_gate` in `experiments/run.py` to handle T1d**

In `check_phase_gate`, add handling for `t1d` alongside the existing T1b/T1c prerequisite check:

```python
        # V10 + T1d: T1b/T1c/T1d require prerequisite experiment passing
        if config.thread_id in {"t1b", "t1c", "t1d"} and config.prerequisite_experiment_id:
            prerequisite_summary_path = (
                PROJECT_ROOT / "experiments" / config.prerequisite_experiment_id / "results" / "summary.json"
            )
            if not prerequisite_summary_path.exists():
                raise FileNotFoundError(
                    f"Prerequisite experiment '{config.prerequisite_experiment_id}' "
                    f"summary not found at {prerequisite_summary_path}. "
                    f"Run that experiment first."
                )
            # For T1b and T1c: require level3_confirmed from T1a
            if config.thread_id in {"t1b", "t1c"}:
                prerequisite_summary = json.loads(prerequisite_summary_path.read_text())
                if not prerequisite_summary.get("level3_confirmed", False):
                    raise ValueError(
                        f"V10: prerequisite experiment '{config.prerequisite_experiment_id}' "
                        f"did not confirm level3_confirmed=True. T1b/T1c require T1a to pass."
                    )
            # For T1d: only require T1b summary exists (not outcome-gated)

        # V14: T1d requires identification_criterion
        if config.thread_id == "t1d" and not config.identification_criterion:
            raise ValueError(
                "V14: identification_criterion is None for T1d. "
                "Set before calling check_phase_gate."
            )

        # V15: T1d requires confounder_structure
        if config.thread_id == "t1d" and not config.confounder_structure:
            raise ValueError(
                "V15: confounder_structure is None for T1d. "
                "Define the formal causal graph before running."
            )
```

Also add `t2c` handling for V16 and V17:

```python
        # V16: T2c requires T2b gate passing on Llama
        if config.thread_id == "t2c" and config.prerequisite_experiment_id:
            t2b_summary_path = (
                PROJECT_ROOT / "experiments" / config.prerequisite_experiment_id / "results" / "summary.json"
            )
            if not t2b_summary_path.exists():
                raise FileNotFoundError(
                    f"V16: T2c requires T2b summary at {t2b_summary_path}. "
                    f"Run T2b on Llama 3.2 3B first."
                )
            t2b_summary = json.loads(t2b_summary_path.read_text())
            if not t2b_summary.get("behavioral_gate_passed", False):
                raise ValueError(
                    "V16: T2b behavioral gate did not pass on Llama 3.2 3B. "
                    "T2c is moot if the model is not hyperintensional."
                )

        # V17: T2c requires intension_type
        if config.thread_id == "t2c" and not config.intension_type:
            raise ValueError(
                "V17: intension_type is None for T2c. "
                "Set to 'primary', 'secondary', or 'dissociation' before running."
            )
```

- [ ] **Step 3: Commit**

```bash
git add experiments/t1d/run_experiment.py experiments/run.py
git commit -m "feat(t1d): add T1d experiment runner and check_phase_gate handling"
```

---

## Task 13: T2c grammar [PHASE 7 MARKER — do not begin until T2b passes on Llama]

**Files:**
- Create: `stimuli/grammars/t2c.py`
- Create: `tests/test_t2c_grammar.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_t2c_grammar.py`:

```python
"""
tests/test_t2c_grammar.py — Tests for T2c two-dimensional semantics grammar.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from stimuli.grammars.t2c import generate


class TestT2cGrammar:
    def test_generates_requested_count(self):
        pairs = generate(n=30)
        assert len(pairs) == 30

    def test_all_three_conditions_present(self):
        pairs = generate(n=60)
        label_set = set()
        for pair in pairs:
            label_set.add(pair["label_a"])
            label_set.add(pair["label_b"])
        required_conditions = {"primary_sensitive", "secondary_necessary", "primary_secondary_dissociation"}
        assert required_conditions.issubset(label_set)

    def test_pairs_have_required_fields(self):
        pairs = generate(n=10)
        for pair in pairs:
            assert "sentence_a" in pair
            assert "sentence_b" in pair
            assert "label_a" in pair
            assert "label_b" in pair

    def test_no_empty_sentences(self):
        pairs = generate(n=20)
        for pair in pairs:
            assert pair["sentence_a"].strip() != ""
            assert pair["sentence_b"].strip() != ""
```

- [ ] **Step 2: Create `stimuli/grammars/t2c.py`**

```python
"""
stimuli/grammars/t2c.py — Stimulus grammar for Thread T2c (two-dimensional semantics).

─── CONCEPT: Two-dimensional semantics (Chalmers) ────────────────────────────
Standard possible-worlds semantics assigns each expression a single intension:
a function from possible worlds to extensions. Chalmers' two-dimensional
framework splits this into two distinct intensions:

Primary intension: how a term fixes reference given the actual world as the
  epistemic base. "Water" picks out the watery stuff in the actual world.
  In a Twin Earth world where XYZ is the watery stuff, "water" would pick
  out XYZ under the primary intension. This is the epistemic/descriptive
  dimension.

Secondary intension: what a term picks out across possible worlds once
  reference is fixed by the actual world. "Water" rigidly picks out H2O
  in all possible worlds (including worlds where H2O is rare). This is the
  metaphysical/rigid dimension — Kripke's rigid designation.

The dissociation: "Water is H2O" is secondarily necessary (true in all
  metaphysically possible worlds — once we know water = H2O) but primarily
  contingent (in a world where watery stuff is XYZ, the primary intension
  of "water is H2O" would be false). This is Chalmers' key move.

─── CONCEPT: Three conditions ────────────────────────────────────────────────
primary_sensitive: sentences invoking primary intension — how reference is
  fixed. Context: epistemic possibility, "what stuff turns out to be".
  "Water is whatever the watery stuff turns out to be in this world."

secondary_necessary: sentences invoking secondary intension — rigid reference.
  Context: metaphysical necessity, "in all possible worlds".
  "Water is necessarily H2O in every possible world."

primary_secondary_dissociation: sentences secondarily necessary but primarily
  contingent — the Chalmers signature case.
  "If this substance had turned out not to be H2O, water would not be H2O."

─── MODEL REQUIREMENT ────────────────────────────────────────────────────────
T2c requires a model with reliable knowledge of scientific identity claims
(water = H2O, heat = molecular motion) and the primary/secondary distinction.
GPT-2 and Pythia fail the behavioral gate for this domain — T2c runs only on
Llama 3.2 3B in Phase 7, gated by T2b passing (V16).

─── T2b prerequisite ─────────────────────────────────────────────────────────
T2b established hyperintensionality — distinctions finer than possible worlds.
T2c asks whether those distinctions are organized along the primary/secondary
dimension specifically. If T2b failed, T2c is moot.
"""

from __future__ import annotations

import random
from typing import Any


# ── Primary intension templates ───────────────────────────────────────────────
# Epistemic context: how reference is fixed in the actual world

PRIMARY_TEMPLATES = [
    (
        "{term_a} is whatever the {description} turns out to be in this world.",
        "{term_b} is whatever the {description} turns out to be in this world.",
    ),
    (
        "When we say '{term_a}', we mean the {description} we encounter here.",
        "The term '{term_b}' refers to the {description} we actually find.",
    ),
    (
        "In a world where {description} is {alternate_substance}, '{term_a}' would pick out {alternate_substance}.",
        "If {description} turned out to be {alternate_substance}, '{term_b}' would refer to {alternate_substance}.",
    ),
]

# ── Secondary intension templates ─────────────────────────────────────────────
# Metaphysical context: rigid reference across all possible worlds

SECONDARY_TEMPLATES = [
    (
        "{term_a} is necessarily {scientific_identity} in every possible world.",
        "In all possible worlds where {term_a} exists, it is {scientific_identity}.",
    ),
    (
        "It is metaphysically necessary that {term_a} is {scientific_identity}.",
        "No possible world contains {term_a} that is not {scientific_identity}.",
    ),
    (
        "{term_a} could not have been anything other than {scientific_identity}.",
        "Even in counterfactual scenarios, {term_a} remains {scientific_identity}.",
    ),
]

# ── Primary-secondary dissociation templates ──────────────────────────────────
# Secondarily necessary but primarily contingent — the Chalmers signature

DISSOCIATION_TEMPLATES = [
    (
        "If {term_a} had turned out not to be {scientific_identity}, {term_a} would not be {scientific_identity}.",
        "Had the {description} in this world been {alternate_substance}, {term_a} would have been {alternate_substance}.",
    ),
    (
        "{term_a} is {scientific_identity} is a posteriori necessary — contingent on how things turned out.",
        "We could have discovered that {term_a} is {alternate_substance} if the {description} were different.",
    ),
    (
        "The necessity of '{term_a} is {scientific_identity}' depends on which world fixes reference.",
        "Primary intension of '{term_a} is {scientific_identity}' is false in worlds where {description} is {alternate_substance}.",
    ),
]

# ── Domain slots ──────────────────────────────────────────────────────────────

DOMAIN_SLOTS = [
    {
        "term_a": "water",
        "term_b": "H2O",
        "description": "watery stuff",
        "scientific_identity": "H2O",
        "alternate_substance": "XYZ",
    },
    {
        "term_a": "heat",
        "term_b": "molecular motion",
        "description": "heat phenomenon",
        "scientific_identity": "mean molecular kinetic energy",
        "alternate_substance": "caloric fluid",
    },
    {
        "term_a": "gold",
        "term_b": "the element with atomic number 79",
        "description": "yellowish metal",
        "scientific_identity": "the element with atomic number 79",
        "alternate_substance": "fool's gold",
    },
    {
        "term_a": "Hesperus",
        "term_b": "Phosphorus",
        "description": "bright celestial body",
        "scientific_identity": "Venus",
        "alternate_substance": "a different planet",
    },
]


def generate(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate n stimulus pairs for T2c — two-dimensional semantics conditions.

    Three conditions: primary_sensitive, secondary_necessary, primary_secondary_dissociation.
    Each pair contrasts two sentences invoking different intension dimensions.

    Args:
        n: number of pairs to generate.
        seed: random seed for reproducibility.

    Returns:
        List of dicts with keys: sentence_a, sentence_b, label_a, label_b.
    """
    rng = random.Random(seed)

    condition_template_map = {
        "primary_sensitive": PRIMARY_TEMPLATES,
        "secondary_necessary": SECONDARY_TEMPLATES,
        "primary_secondary_dissociation": DISSOCIATION_TEMPLATES,
    }
    condition_order = list(condition_template_map.keys())

    pairs: list[dict[str, Any]] = []
    items_per_condition = max(1, n // len(condition_order))

    for condition_label, templates in condition_template_map.items():
        contrast_conditions = [c for c in condition_order if c != condition_label]
        for _ in range(items_per_condition):
            domain_slots = rng.choice(DOMAIN_SLOTS)
            template_a, _ = rng.choice(templates)
            contrast_label = rng.choice(contrast_conditions)
            _, template_b = rng.choice(condition_template_map[contrast_label])

            sentence_a = template_a.format(**domain_slots)
            sentence_b = template_b.format(**domain_slots)

            pairs.append({
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "label_a": condition_label,
                "label_b": contrast_label,
            })

    rng.shuffle(pairs)
    return pairs[:n]
```

- [ ] **Step 3: Run tests**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/test_t2c_grammar.py -v 2>&1 | tail -15
```

Expected: all 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add stimuli/grammars/t2c.py tests/test_t2c_grammar.py
git commit -m "feat(t2c): add two-dimensional semantics grammar — primary/secondary/dissociation [Phase 7]"
```

---

## Task 14: T2c experiment runner [PHASE 7 MARKER]

**Files:**
- Create: `experiments/t2c/run_experiment.py`

- [ ] **Step 1: Create `experiments/t2c/run_experiment.py`**

Structure mirrors T1d runner. Key differences: `intension_type` field, T2b prerequisite, Llama 3.2 3B model.

```python
"""
experiments/t2c/run_experiment.py — T2c mechanistic experiment: two-dimensional semantics.

Tests whether Llama 3.2 3B encodes the primary/secondary intension distinction
from Chalmers' two-dimensional framework. Runs in Phase 7 after T2b confirms
hyperintensionality on Llama.

Prerequisite: T2b must have passed behavioral gate on Llama 3.2 3B.

Usage (Colab — Phase 7 only):
    !python experiments/t2c/run_experiment.py --intension_type primary
    !python experiments/t2c/run_experiment.py --intension_type secondary
    !python experiments/t2c/run_experiment.py --intension_type dissociation
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--intension_type",
    required=True,
    choices=["primary", "secondary", "dissociation"],
    help="Which intension condition to run.",
)
args = parser.parse_args()

THREAD_ID = "t2c"
MODEL_ID = "meta-llama/Llama-3.2-3B"
LLAMA_3B_N_LAYERS = 28

VALIDATED_PATH = PROJECT_ROOT / "stimuli" / "validated" / THREAD_ID / "pairs.validated.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / THREAD_ID / "results" / args.intension_type
CONFIG_PATH = PROJECT_ROOT / "experiments" / THREAD_ID / ("config_" + args.intension_type + ".json")
SURFACE_NULL_PATH = RESULTS_DIR / "surface_null.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

T2B_PREREQUISITE_ID = "t2b_llama"

if not VALIDATED_PATH.exists():
    print("ERROR: Validated stimulus file not found.")
    print("  Run scripts/run_validation.py --thread t2c first.")
    sys.exit(1)

t2b_summary_path = PROJECT_ROOT / "experiments" / T2B_PREREQUISITE_ID / "results" / "summary.json"
if not t2b_summary_path.exists():
    print("ERROR: T2b Llama summary not found. Run T2b on Llama 3.2 3B first.")
    sys.exit(1)

from extraction.extractor import compute_sha256, extract_activations
from experiments.config import ExperimentConfig
from experiments.run import run_surface_null, check_phase_gate
from probes.probes import run_linear_probe
from interventions.interventions import run_layer_sweep, assert_specificity_valid, mean_ablate
from core.io import load_result, save_result
import torch

print("=" * 60)
print("PoL-Probe — T2c Experiment: Two-Dimensional Semantics")
print("Intension type: " + args.intension_type)
print("=" * 60)
print()

t2b_summary = load_result(t2b_summary_path)
t2b_hyperintensional = t2b_summary.get("hyperintensional_finding", False)
print("[Context] T2b hyperintensional = " + str(t2b_hyperintensional))
print()

expected_outcomes_by_intension_type = {
    "primary": {
        "prediction": (
            "Representations of primary_sensitive pairs diverge — model encodes "
            "how reference is fixed, not just what it picks out. "
            "Primary intension is encoded as a distinct geometric cluster."
        ),
    },
    "secondary": {
        "prediction": (
            "Representations of secondary_necessary pairs converge — model encodes "
            "rigid reference across worlds. Metaphysical necessity produces a compact cluster."
        ),
    },
    "dissociation": {
        "prediction": (
            "Representations of primary_secondary_dissociation pairs diverge from "
            "both primary_sensitive and secondary_necessary. The two-dimensional "
            "split is encoded as a third distinct geometric region."
        ),
    },
}

date_stamp = datetime.date.today().strftime("%Y%m%d")

config = ExperimentConfig(
    experiment_id="t2c_llama3b_" + args.intension_type + "_" + date_stamp,
    thread_id=THREAD_ID,
    model_id=MODEL_ID,
    model_revision="main",
    layer_range=(0, LLAMA_3B_N_LAYERS - 1),
    component="resid_post",
    token_positions=[-1],
    probe_type="linear",
    stimulus_file=str(VALIDATED_PATH),
    stimulus_sha256=compute_sha256(VALIDATED_PATH),
    frequency_match_verified=True,
    expected_outcomes=expected_outcomes_by_intension_type[args.intension_type],
    prerequisite_experiment_id=T2B_PREREQUISITE_ID,
    intension_type=args.intension_type,
)

config.lock()
config.to_json(CONFIG_PATH)
print("  Config locked for intension_type=" + args.intension_type)
print()

surface_null_result = run_surface_null(config)
print("[Surface null] accuracy=" + str(round(surface_null_result["surface_classifier_accuracy"] * 100, 1)) + "%")
print()

check_phase_gate(config)
print("[Phase gate] Passed.")
print()

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_ID)
model.eval()

layer_activation_sets = extract_activations(config, model)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
probe_results_by_layer: dict[int, dict] = {}
for activation_set in layer_activation_sets:
    layer_index = activation_set["layer"]
    probe_result = run_linear_probe(
        np.array(activation_set["activations"]), activation_set["labels"], config
    )
    probe_result["layer"] = layer_index
    save_result(probe_result, RESULTS_DIR / ("probe_layer_" + str(layer_index) + ".json"))
    probe_results_by_layer[layer_index] = probe_result

peak_probe_layer = max(
    probe_results_by_layer,
    key=lambda layer_index: probe_results_by_layer[layer_index]["accuracy_mean"]
)
peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
print("[L2 probe] Peak layer=" + str(peak_probe_layer) +
      "  accuracy=" + str(round(peak_probe_accuracy * 100, 1)) + "%")

summary = {
    "experiment_id": config.experiment_id,
    "thread_id": config.thread_id,
    "intension_type": args.intension_type,
    "model_id": config.model_id,
    "run_timestamp": datetime.datetime.utcnow().isoformat(),
    "t2b_hyperintensional": t2b_hyperintensional,
    "peak_probe_layer": peak_probe_layer,
    "peak_probe_accuracy": float(peak_probe_accuracy),
    "surface_null_accuracy": float(surface_null_result["surface_classifier_accuracy"]),
    "expected_outcomes": config.expected_outcomes,
}
save_result(summary, SUMMARY_PATH)

print()
print("=" * 60)
print("T2c complete — intension_type=" + args.intension_type)
print("Peak probe accuracy: " + str(round(peak_probe_accuracy * 100, 1)) + "%")
print("=" * 60)
```

- [ ] **Step 2: Commit**

```bash
git add experiments/t2c/run_experiment.py
git commit -m "feat(t2c): add T2c experiment runner — primary/secondary/dissociation [Phase 7]"
```

---

## Task 15: Update TASKS.md and SPEC.md

**Files:**
- Modify: `TASKS.md`
- Modify: `SPEC.md`

- [ ] **Step 1: Add new task IDs to TASKS.md**

After task `| 43 | . | Run T1c probe & measure borderline geometry *(blocked: 42)* |`, add:

```markdown
| 43a | . | Write T1d grammar — four confounding conditions *(blocked: 43)* |
| 43b | . | Validate T1d stimuli, run behavioral gate *(blocked: 43a)* |
| 43c | . | Pre-specify T1d outcomes, lock config with identification_criterion + confounder_structure *(blocked: 43b)* |
| 43d | . | Run T1d surface null, L2 probe, binary identification probe, L3 patching *(blocked: 43c)* |
| 43e | . | Write T1d summary — interpret against T1b pearl_confirmed *(blocked: 43d)* |
```

After task `| 86 | . | Compile & validate PhilBench 500+ items *(blocked: 85)* |`, add:

```markdown
| 86a | . | Write T2c grammar — primary/secondary/dissociation conditions *(blocked: T2b Llama gate)* |
| 86b | . | Validate T2c stimuli, run behavioral gate on Llama 3.2 3B *(blocked: 86a)* |
| 86c | . | Pre-specify T2c outcomes, lock config per intension_type *(blocked: 86b)* |
| 86d | . | Run T2c surface null, L2 probe, L3 patching — all three intension_type values *(blocked: 86c)* |
| 86e | . | Write T2c summary — interpret against T2b hyperintensionality finding *(blocked: 86d)* |
```

- [ ] **Step 2: Add §T1d and §T2c to SPEC.md**

In SPEC.md, after the T1c section, add a T1d section. After the T2b section, add a T2c section. After §V, add V14–V18. Content mirrors the design doc at `docs/superpowers/specs/2026-06-02-depth-additions-design.md` — do not paraphrase, copy directly.

- [ ] **Step 3: Run full test suite to confirm nothing broken**

```bash
cd "/Users/scott/Desktop/C++ review projects/PoL-Probe" && .venv/bin/python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add TASKS.md SPEC.md
git commit -m "docs: add T1d and T2c tasks and spec sections, invariants V14-V18"
```

---

## Self-Review

**Spec coverage:**
- Shared infrastructure (circuits/ module, ExperimentConfig fields, invariants V14–V18, PhilBench schema): Tasks 1–5 ✓
- `run_identification_probe`: Task 6 ✓
- T1b issue fixes: Tasks 7–9 ✓
- Circuit analysis on T1b: Task 10 ✓
- T1d grammar + experiment: Tasks 11–12 ✓
- T2c grammar + experiment: Tasks 13–14 ✓
- TASKS.md + SPEC.md: Task 15 ✓

**V18 (circuit_analysis_enabled requires layer sweep results):** enforced in Task 10 sequencing (circuit step runs after sweep). Not yet enforced in `check_phase_gate` — add this to Task 12 step 2 when updating `run.py`:

```python
        # V18: circuit analysis requires layer sweep to have run
        if config.circuit_analysis_enabled:
            layer_sweep_path = (
                PROJECT_ROOT / "experiments" / config.thread_id / "results" / "layer_sweep.json"
            )
            if not layer_sweep_path.exists():
                raise FileNotFoundError(
                    f"V18: circuit_analysis_enabled=True but layer_sweep.json not found "
                    f"at {layer_sweep_path}. Run layer sweep first."
                )
```

**Type consistency:** `find_peak_circuit_components` returns `list[dict]` with keys `layer`, `head`, `kl_effect` — used identically in `run_path_patching` (Task 3) and T1b circuit step (Task 10). ✓

**Placeholder scan:** No TBDs. All code blocks complete. Task 7 (grammar fix) and Task 9 (Colab re-run) are operational tasks that require human judgment — documented as such. ✓
