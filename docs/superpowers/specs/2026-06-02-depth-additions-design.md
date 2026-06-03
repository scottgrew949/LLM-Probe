# PoL-Probe Depth Additions — Design Outline
_2026-06-02 — subject to revision_

## Scope

Three additions, executed in phase sequence (no skipping):

1. **Mechanistic depth** — circuit-level analysis, T1 first
2. **T1d** — causal identification sub-thread (back-door / front-door)
3. **T2c** — two-dimensional semantics (Phase 7, Llama only)

---

## 1. Shared Infrastructure (builds before anything else)

### New module: `circuits/`
- `circuits/circuit_finder.py`
  - `run_head_sweep` — patch each (layer, head) via `hook_z`, return KL matrix
  - `run_path_patching` — patch only peak circuit components, measure fraction of total KL recovered
  - `find_peak_circuit_components` — filter KL matrix by `circuit_kl_threshold`
- `circuits/attribution.py`
  - `compute_logit_attribution` — direct logit attribution per head via TransformerLens

### `ExperimentConfig` additions
```python
circuit_analysis_enabled: bool = False
circuit_kl_threshold: float = 0.1
identification_criterion: Optional[str] = None   # 'back_door' | 'front_door'
confounder_structure: Optional[dict] = None
intension_type: Optional[str] = None             # 'primary' | 'secondary' | 'dissociation'
```
Valid thread IDs: add `t1d`, `t2c`.

### New invariants
| V# | Rule |
|----|------|
| V14 | `identification_criterion` non-null required for T1d |
| V15 | `confounder_structure` non-null required for T1d |
| V16 | T2c requires T2b gate passing on Llama — via `prerequisite_experiment_id` |
| V17 | `intension_type` non-null required for T2c |
| V18 | `circuit_analysis_enabled=True` requires layer sweep results to exist |

### `PhilBench` schema additions
- `circuit_components: list[{layer, head, kl_effect}]` — optional, populated when circuit analysis ran
- `identification_criterion: str | null` — populated for T1d entries

### `run_experiment` pipeline change
After layer sweep: if `circuit_analysis_enabled`, run head sweep → find components → path patching → logit attribution → write `circuit_components.json`, `logit_attribution.json`.

---

## 2. Mechanistic Depth — Circuit Analysis (T1 first)

**Sequencing within T1:**
1. Fix three open T1b issues (surface confound, data pipeline, L2/L3 agreement)
2. Clean T1b re-run with `circuit_analysis_enabled=False` — confirm clean signal
3. T1b re-run with `circuit_analysis_enabled=True` — characterize circuit
4. T1c proceeds with circuit analysis available

**New results files per T1b/T1c run (when enabled):**
- `circuit_components.json`
- `logit_attribution.json`

**Key output:** (layer, head) pairs that account for ≥0.80 of peak patching KL. This is the circuit for causal direction encoding.

---

## 3. T1d — Causal Identification Sub-thread

**Position in sequence:** After T1c (#43), before SAE stretch (#44).

**Prerequisite:** T1b complete (not outcome-gated — T1d is informative either way).

**Stimulus conditions:**
- `back_door_adjustable` — confounded, observed adjustment set blocks back-door path
- `front_door_adjustable` — hidden confounder, mediator present for front-door
- `confounded_not_adjustable` — confounded, no valid adjustment set
- `unconfounded_control` — direct causation, no confounding

**`confounder_structure` field:** captures formal graph (nodes, edges, criterion, adjustment_set) per condition.

**New probe:** `run_identification_probe` in `probes/probes.py` — binary: adjustable vs not-adjustable. Reuses `run_linear_probe` with new label grouping.

**Behavioral gate:** forced-choice — correct identification of whether back-door adjustment is needed. Threshold 0.70 (V6).

**Tasks (43a–43e):**
- 43a: T1d grammar
- 43b: validate + behavioral gate
- 43c: pre-specify outcomes + lock config
- 43d: surface null + L2 + L3
- 43e: summary (interpreted against T1b pearl_confirmed)

---

## 4. T2c — Two-Dimensional Semantics

**Position:** Phase 7, after T2b on Llama 3.2 3B. Never runs on GPT-2 or Pythia.

**Prerequisite:** T2b gate passing on Llama (V16).

**Stimulus conditions:**
- `primary_sensitive` — same secondary intension, different primary (water/H2O in epistemic context)
- `secondary_necessary` — metaphysically necessary identity statements
- `primary_secondary_dissociation` — secondarily necessary, primarily contingent

**Three experiment runs:** one per `intension_type` value. Same pattern as T1a/T1b/T1c.

**Mechanistic prediction:**
- L2: probe on primary vs secondary framing context exceeds chance
- L3: patching primary-framing activation shifts completions toward primary-intension responses
- Divergence in `primary_sensitive`, convergence in `secondary_necessary` → 2D structure encoded

**Interpretation against T2b:** if T2b found hyperintensionality, T2c tests whether it's 2D-organized. If T2b failed, T2c is moot — enforced by V16 prerequisite gate.

**Tasks (86a–86e):**
- 86a: T2c grammar
- 86b: validate + behavioral gate on Llama
- 86c: pre-specify outcomes + lock config (one per intension_type)
- 86d: surface null + L2 + L3 — all three conditions
- 86e: summary — interpret against T2b finding

---

## Phase Sequence Impact

```
Phase 0–1: unchanged
Phase 2: T1a → T1b (fix issues + circuit analysis) → T1c → T1d → SAE stretch → Pythia replication
Phase 3–6: unchanged
Phase 7: T2b → T2c → T1–T6 Llama sweep → PhilBench → reports
```

No phase skipping. T1d unlocks only after T1c. T2c unlocks only after T2b on Llama.
