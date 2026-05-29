# PoL-Probe Task List

Reference doc. Mirrors task tracker. Status: `.` todo | `~` wip | `x` done.
Phase gates enforce order — no phase starts until prior phase fully complete.

---

## Phase 0 — Foundation (Tasks 1–16)

| id | status | task |
|----|--------|------|
| 1 | . | Install & configure project environment |
| 2 | . | Read Tenney et al. 2019 edge probing paper |
| 3 | . | Read Elhage Mathematical Framework paper |
| 4 | . | Replicate Tenney edge probing data pipeline *(blocked: 2,3)* |
| 5 | . | Replicate Tenney linear probe classifier *(blocked: 4)* |
| 6 | . | Verify TransformerLens hooks on GPT-2 *(blocked: 5)* |
| 7 | . | Define stimulus & activation JSON schemas *(blocked: 6)* |
| 8 | . | Implement `core/io.py` *(blocked: 7)* |
| 9 | . | Implement `ExperimentConfig` dataclass *(blocked: 8)* |
| 10 | . | Implement `stimuli/pipeline.py` core functions *(blocked: 9)* |
| 11 | . | Implement `run_behavioral_gate` in pipeline.py *(blocked: 10)* |
| 12 | . | Implement `build_philbench_entry` in pipeline.py *(blocked: 11)* |
| 13 | . | Implement `extraction/extractor.py` *(blocked: 12)* |
| 14 | . | Implement `probes/probes.py` *(blocked: 13)* |
| 15 | . | Implement `interventions/interventions.py` *(blocked: 14)* |
| 16 | . | Implement `experiments/run.py` scaffold *(blocked: 15)* |

---

## Phase 1 — T2 Sense/Reference + T2b Hyperintensionality (Tasks 17–31)

*Blocked: Phase 0 complete (#16)*

**T2 — Frege sense/reference (opaque vs. transparent contexts)**

| id | status | task |
|----|--------|------|
| 17 | . | Generate T2 stimulus set *(blocked: 16)* |
| 18 | . | Validate T2 stimuli & run behavioral gate *(blocked: 17)* |
| 19 | . | Write T2 expected outcomes & lock config *(blocked: 18)* |
| 20 | . | Run T2 surface-statistics null *(blocked: 19)* |
| 21 | . | Run T2 linear probe L2 *(blocked: 20)* |
| 22 | . | Run T2 activation patching L3 *(blocked: 21)* |
| 23 | . | Write T2 summary & question log *(blocked: 22)* |

**T2b — Hyperintensionality (distinctions finer than possible worlds)**

| id | status | task |
|----|--------|------|
| 24 | . | Generate T2b stimulus set |
| 25 | . | Validate T2b stimuli & run behavioral gate *(blocked: 24)* |
| 26 | . | Write T2b expected outcomes & lock config *(blocked: 25)* |
| 27 | . | Run T2b surface-statistics null *(blocked: 26)* |
| 28 | . | Run T2b cosine distance analysis per class *(blocked: 27)* |
| 29 | . | Run T2b activation patching L3 *(blocked: 28)* |
| 30 | . | Write T2b summary & question log *(blocked: 29)* |

**T2/T2b Replication**

| id | status | task |
|----|--------|------|
| 31 | . | Replicate T2 & T2b on Pythia 1.4B *(blocked: 23, 30)* |

---

## Phase 2 — T1 Counterfactuals: Lewis/Pearl (Tasks 32–46)

*Blocked: Phase 1 complete (#31)*

| id | status | task |
|----|--------|------|
| 32 | . | Write T1 counterfactual stimulus grammar *(blocked: 31)* |
| 33 | . | Generate & frequency-match T1 stimulus pairs *(blocked: 32)* |
| 34 | . | Write T1a behavioral forced-choice items *(blocked: 33)* |
| 35 | . | Run T1a behavioral gate on GPT-2 *(blocked: 34)* |
| 36 | . | Pre-specify T1a outcomes & lock config *(blocked: 35)* |
| 37 | . | Extract T1a activations & run surface null *(blocked: 36)* |
| 38 | . | Run T1a L2 probe & L3 patching *(blocked: 37)* |
| 39 | . | Pre-specify T1b & T1c configs with T1a gate *(blocked: 38)* |
| 40 | . | Run T1b behavioral gate & extract activations *(blocked: 39)* |
| 41 | . | Run T1b L2 probe & layer-resolved patching *(blocked: 40)* |
| 42 | . | Run T1c behavioral gate & extract activations *(blocked: 39)* |
| 43 | . | Run T1c probe & measure borderline geometry *(blocked: 42)* |
| 44 | . | Run T1 SAE double dissociation [stretch goal] *(blocked: 38)* |
| 45 | . | Replicate T1a–T1c on Pythia 1.4B *(blocked: 43)* |
| 46 | . | Write T1 one-page question log *(blocked: 45)* |

---

## Phase 3 — Checkpoint Analysis (Tasks 47–54)

*Blocked: Phase 2 complete (#46)*

| id | status | task |
|----|--------|------|
| 47 | . | Enumerate & download Pythia checkpoints *(blocked: 46)* |
| 48 | . | Write T1 checkpoint sweep runner *(blocked: 47)* |
| 49 | . | Run T1 checkpoint sweep on Pythia *(blocked: 48)* |
| 50 | . | Write T2 checkpoint sweep runner *(blocked: 47)* |
| 51 | . | Run T2 checkpoint sweep on Pythia *(blocked: 50)* |
| 52 | . | Plot T1 & T2 developmental curves *(blocked: 49, 51)* |
| 53 | . | Compare T1 & T2 emergence timing *(blocked: 52)* |
| 54 | . | Write Phase 3 question log *(blocked: 53)* |

---

## Phase 4 — T3 Intensional Context Failure (Tasks 55–64)

*Blocked: Phase 3 complete (#54)*

| id | status | task |
|----|--------|------|
| 55 | . | Generate T3 belief-report stimulus set *(blocked: 54)* |
| 56 | . | Generate T3 intentional-inexistence stimulus set *(blocked: 55)* |
| 57 | . | Generate T3 modal-opacity stimulus set *(blocked: 55)* |
| 58 | . | Validate T3 stimuli & run behavioral gate *(blocked: 56, 57)* |
| 59 | . | Lock T3 config with pre-specified outcomes *(blocked: 58)* |
| 60 | . | Extract T3 activations & compute surface null *(blocked: 59)* |
| 61 | . | Train T3 L2 probes per context type per layer *(blocked: 60)* |
| 62 | . | Build T3 layer-of-failure curves *(blocked: 61)* |
| 63 | . | Run T3 L3 patching for causal verification *(blocked: 62)* |
| 64 | . | Write T3 question log & replicate on Pythia *(blocked: 63)* |

---

## Phase 5 — T4 Quinean Ontological Commitment / RSA (Tasks 65–71)

*Blocked: Phase 4 complete (#64)*

| id | status | task |
|----|--------|------|
| 65 | . | Select & document BFO/DOLCE ontology version *(blocked: 64)* |
| 66 | . | Build T4 theoretical similarity matrices *(blocked: 65)* |
| 67 | . | Generate T4 entity stimulus set *(blocked: 66)* |
| 68 | . | Validate T4 stimuli & lock config *(blocked: 67)* |
| 69 | . | Extract T4 activations & compute surface null *(blocked: 68)* |
| 70 | . | Run RSA & Mantel test against ontology matrices *(blocked: 69)* |
| 71 | . | Interpret T4 results & write question log *(blocked: 70)* |

---

## Phase 6 — T5 Grounding + T6 Rigid Designation (Tasks 72–82)

*Blocked: Phase 5 complete (#71)*

**T5 — Fine's grounding (asymmetry, irreflexivity, transitivity)**

| id | status | task |
|----|--------|------|
| 72 | . | Generate T5 stimulus grammars — all three domains *(blocked: 71)* |
| 73 | . | Build T5 three-node transitivity chain stimuli *(blocked: 72)* |
| 74 | . | Validate T5 stimuli & write config with thresholds *(blocked: 73)* |
| 75 | . | Implement T5 asymmetry & irreflexivity patching *(blocked: 74)* |
| 76 | . | Implement T5 transitivity chain patching *(blocked: 75)* |
| 77 | . | Run T5 full experiment across three domains *(blocked: 76)* |

**T6 — Kripke rigid designation**

| id | status | task |
|----|--------|------|
| 78 | . | Generate T6 modal-context stimulus grammar |
| 79 | . | Validate T6 stimuli & lock config *(blocked: 78)* |
| 80 | . | Implement T6 cross-context patching intervention *(blocked: 79)* |
| 81 | . | Run T6 full experiment & write question log *(blocked: 80)* |

**Replication**

| id | status | task |
|----|--------|------|
| 82 | . | Replicate T5 & T6 on Pythia 1.4B *(blocked: 77, 81)* |

---

## Phase 7 — Llama 3.2 3B Sweep + PhilBench (Tasks 83–90)

*Blocked: Phase 6 complete (#82)*

| id | status | task |
|----|--------|------|
| 83 | . | Run T1–T6 behavioral gates on Llama 3.2 3B *(blocked: 82)* |
| 84 | . | Extract Llama 3.2 3B activations T1–T6 *(blocked: 83)* |
| 85 | . | Run L2 probes & L3 patching T1–T6 on Llama *(blocked: 84)* |
| 86 | . | Compile & validate PhilBench 500+ items *(blocked: 85)* |
| 87 | . | Write PhilBench datasheet *(blocked: 86)* |
| 88 | . | Release PhilBench to HuggingFace Datasets *(blocked: 87)* |
| 89 | . | Write cross-architecture comparison report *(blocked: 85)* |
| 90 | . | Write Llama 3.2 3B one-page question logs per thread *(blocked: 85)* |

---

## Dependency Summary

```
Phase 0 (1–16) → Phase 1 (17–31) → Phase 2 (32–46) → Phase 3 (47–54)
→ Phase 4 (55–64) → Phase 5 (65–71) → Phase 6 (72–82) → Phase 7 (83–90)
```

T1b/T1c (40–43) gated on T1a passing (#38) — enforced by V10 & prerequisite_experiment_id.
T3 belief-report (#55) runs before intentional-inexistence (#56) & modal-opacity (#57) in parallel.
Checkpoint sweep T1 (#48–49) & T2 (#50–51) run in parallel after checkpoints downloaded (#47).
T6 stimulus grammar (#78) unlocks independently within Phase 6 alongside T5.
PhilBench (#86–88) & cross-arch report (#89) & question logs (#90) all unblock after Llama sweep (#85).
T1 SAE stretch goal (#44) never blocks phase completion — V9.
