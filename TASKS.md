# PoL-Probe Task List

Reference doc. Mirrors task tracker. Status: `.` todo | `~` wip | `x` done.
Phase gates enforce order — no phase starts until prior phase fully complete.

---

## Phase 0 — Foundation (Tasks 1–16)

| id | status | task |
|----|--------|------|
| 1  | x | Install & configure project environment |
| 2  | x | Read Tenney et al. 2019 edge probing paper |
| 3  | x | Read Elhage Mathematical Framework paper |
| 4  | x | Replicate Tenney edge probing data pipeline |
| 5  | x | Replicate Tenney linear probe classifier |
| 6  | x | Verify TransformerLens hooks on GPT-2 |
| 7  | x | Define stimulus & activation JSON schemas |
| 8  | x | Implement `core/io.py` |
| 9  | x | Implement `ExperimentConfig` dataclass |
| 10 | x | Implement `stimuli/pipeline.py` core functions |
| 11 | x | Implement `run_behavioral_gate` in pipeline.py |
| 12 | x | Implement `build_philbench_entry` in pipeline.py |
| 13 | x | Implement `extraction/extractor.py` |
| 14 | x | Implement `probes/probes.py` |
| 15 | x | Implement `interventions/interventions.py` |
| 16 | x | Implement `experiments/run.py` scaffold |

---

## Phase 1 — T1 Counterfactuals: Lewis / Pearl / Stalnaker (Tasks 17–41)

*Unlocked: Phase 0 complete*

**T1a — Causal hierarchy levels (L3 causal vs L1 associative)**

| id | status | task |
|----|--------|------|
| 17 | x | Write T1a stimulus grammar & generate pairs |
| 18 | x | Frequency-match T1a stimulus pairs |
| 19 | x | Write T1a behavioral forced-choice items |
| 20 | x | Run T1a behavioral gate on GPT-2 |
| 21 | x | Pre-specify T1a outcomes & lock config |
| 22 | x | Extract T1a activations & run surface null |
| 23 | x | Run T1a L2 probe & L3 patching |

**T1b — Lewis/Stalnaker possible-worlds vs Pearl do-calculus**

| id | status | task |
|----|--------|------|
| 24 | x | Pre-specify T1b & T1c configs with T1a gate *(blocked: 23)* |
| 25 | x | Fix T1b surface confound — equalize forward/backtracking sentence lengths across all 10 domains |
| 26 | x | Fix T1b data pipeline — validation counts both label_a and label_b; fix t1c max pairs 300→200 |
| 27 | x | Add T1b L3 direction check — baseline top-10 completions printed before and after patch |
| 28 | . | Rerun T1b behavioral gate & extract activations on Colab *(blocked: 25, 26, 27 → all cleared)* |
| 29 | . | Rerun T1b L2 probe & layer-resolved patching, accept results *(blocked: 28)* |

**T1c — Lewis vs Stalnaker near-miss / tie conditions**

| id | status | task |
|----|--------|------|
| 30 | x | Write T1c stimulus grammar & generate pairs |
| 31 | x | Write T1c behavioral forced-choice items (4 items) |
| 32 | x | Implement T1c experiment runner |
| 33 | . | Run T1c behavioral gate & extract activations on Colab *(blocked: 32, 29)* |
| 34 | . | Run T1c L2 probe & L3 patching, accept results *(blocked: 33)* |

**T1d — Causal identification (do-calculus: back-door / front-door)**

| id | status | task |
|----|--------|------|
| 35 | x | Write T1d grammar — four confounding conditions |
| 36 | x | Validate T1d stimuli & run behavioral gate |
| 37 | x | Pre-specify T1d outcomes & lock config with `identification_criterion` + `confounder_structure` |
| 38 | x | Run T1d surface null, L2 probe, binary identification probe, L3 patching |
| 39 | x | Write T1d summary — interpret against T1b `pearl_confirmed` |

**T1 completion**

| id | status | task |
|----|--------|------|
| 40a | . | Run T1 SAE double dissociation *(stretch goal — never blocks)* |
| 40 | . | Replicate T1a–T1d on Pythia 1.4B *(blocked: 29, 34, 39)* |
| 41 | . | Write T1 one-page question log *(blocked: 40)* |

---

## Phase 2 — T2 Sense/Reference + T2b Hyperintensionality + T2c Context-Sensitivity (Tasks 42–56)

*Blocked: Phase 1 complete (#41)*

**T2 — Frege sense/reference (opaque vs. transparent contexts)**

| id | status | task |
|----|--------|------|
| 42 | x | Generate T2 stimulus set |
| 43 | x | Validate T2 stimuli & run behavioral gate — **GPT-2 FAIL (50%); queued for Pythia** |
| 44 | . | Run T2 on Pythia 1.4B: surface null, L2 probe, L3 patching *(blocked: 43)* |
| 45 | . | Write T2 summary & question log *(blocked: 44)* |

**T2b — Hyperintensionality (distinctions finer than possible worlds)**

| id | status | task |
|----|--------|------|
| 46 | x | Generate T2b stimulus set |
| 47 | x | Validate T2b stimuli & run behavioral gate — **GPT-2 FAIL; deferred to Llama 3.2 3B (Phase 7)** |
| 48 | . | Run T2b on Llama 3.2 3B: surface null, cosine distance analysis, L3 patching *(blocked: 47, Phase 7)* |
| 49 | . | Write T2b summary & question log *(blocked: 48)* |

**T2c — Two-dimensional semantics / context-sensitivity**

| id | status | task |
|----|--------|------|
| 50 | x | Generate T2c stimulus set |
| 51 | . | Validate T2c stimuli & run behavioral gate *(blocked: 50)* |
| 52 | . | Pre-specify T2c outcomes, lock config per `intension_type` *(blocked: 51)* |
| 53 | . | Run T2c surface null, L2 probe, L3 patching — all three `intension_type` values *(blocked: 52)* |
| 54 | . | Write T2c summary — interpret against T2b hyperintensionality finding *(blocked: 53)* |

**T2 replication**

| id | status | task |
|----|--------|------|
| 55 | . | Replicate T2 & T2c on Llama 3.2 3B *(blocked: 45, 54, Phase 7)* |
| 56 | . | Write T2 one-page question log *(blocked: 55)* |

---

## Phase 3 — Checkpoint Developmental Analysis (Tasks 57–64)

*Blocked: Phase 2 complete (#56)*

| id | status | task |
|----|--------|------|
| 57 | . | Enumerate & download Pythia checkpoints *(blocked: 56)* |
| 58 | . | Write T1 checkpoint sweep runner *(blocked: 57)* |
| 59 | . | Run T1 checkpoint sweep on Pythia *(blocked: 58)* |
| 60 | . | Write T2 checkpoint sweep runner *(blocked: 57)* |
| 61 | . | Run T2 checkpoint sweep on Pythia *(blocked: 60)* |
| 62 | . | Plot T1 & T2 developmental curves *(blocked: 59, 61)* |
| 63 | . | Compare T1 & T2 emergence timing *(blocked: 62)* |
| 64 | . | Write Phase 3 question log *(blocked: 63)* |

---

## Phase 4 — T3 Intensional Context Failure (Tasks 65–74)

*Blocked: Phase 3 complete (#64)*

| id | status | task |
|----|--------|------|
| 65 | . | Generate T3 belief-report stimulus set *(blocked: 64)* |
| 66 | . | Generate T3 intentional-inexistence stimulus set *(blocked: 65)* |
| 67 | . | Generate T3 modal-opacity stimulus set *(blocked: 65)* |
| 68 | . | Validate T3 stimuli & run behavioral gate *(blocked: 66, 67)* |
| 69 | . | Lock T3 config with pre-specified outcomes *(blocked: 68)* |
| 70 | . | Extract T3 activations & compute surface null *(blocked: 69)* |
| 71 | . | Train T3 L2 probes per context type per layer *(blocked: 70)* |
| 72 | . | Build T3 layer-of-failure curves *(blocked: 71)* |
| 73 | . | Run T3 L3 patching for causal verification *(blocked: 72)* |
| 74 | . | Write T3 question log & replicate on Pythia *(blocked: 73)* |

---

## Phase 5 — T4 Quinean Ontological Commitment / RSA (Tasks 75–81)

*Blocked: Phase 4 complete (#74)*

| id | status | task |
|----|--------|------|
| 75 | . | Select & document BFO/DOLCE ontology version *(blocked: 74)* |
| 76 | . | Build T4 theoretical similarity matrices *(blocked: 75)* |
| 77 | . | Generate T4 entity stimulus set *(blocked: 76)* |
| 78 | . | Validate T4 stimuli & lock config *(blocked: 77)* |
| 79 | . | Extract T4 activations & compute surface null *(blocked: 78)* |
| 80 | . | Run RSA & Mantel test against ontology matrices *(blocked: 79)* |
| 81 | . | Interpret T4 results & write question log *(blocked: 80)* |

---

## Phase 6 — T5 Grounding + T6 Rigid Designation (Tasks 82–92)

*Blocked: Phase 5 complete (#81)*

**T5 — Fine's grounding (asymmetry, irreflexivity, transitivity)**

| id | status | task |
|----|--------|------|
| 82 | . | Generate T5 stimulus grammars — all three domains *(blocked: 81)* |
| 83 | . | Build T5 three-node transitivity chain stimuli *(blocked: 82)* |
| 84 | . | Validate T5 stimuli & write config with thresholds *(blocked: 83)* |
| 85 | . | Implement T5 asymmetry & irreflexivity patching *(blocked: 84)* |
| 86 | . | Implement T5 transitivity chain patching *(blocked: 85)* |
| 87 | . | Run T5 full experiment across three domains *(blocked: 86)* |

**T6 — Kripke rigid designation**

| id | status | task |
|----|--------|------|
| 88 | . | Generate T6 modal-context stimulus grammar *(blocked: 81)* |
| 89 | . | Validate T6 stimuli & lock config *(blocked: 88)* |
| 90 | . | Implement T6 cross-context patching intervention *(blocked: 89)* |
| 91 | . | Run T6 full experiment & write question log *(blocked: 90)* |

**T5/T6 replication**

| id | status | task |
|----|--------|------|
| 92 | . | Replicate T5 & T6 on Pythia 1.4B *(blocked: 87, 91)* |

---

## Phase 7 — Llama 3.2 3B Sweep + PhilBench Release (Tasks 93–101)

*Blocked: Phase 6 complete (#92)*

| id | status | task |
|----|--------|------|
| 93  | . | Run T1–T6 behavioral gates on Llama 3.2 3B *(blocked: 92)* |
| 94  | . | Extract Llama 3.2 3B activations T1–T6 *(blocked: 93)* |
| 95  | . | Run L2 probes & L3 patching T1–T6 on Llama *(blocked: 94)* |
| 96  | . | Run T2b on Llama 3.2 3B (deferred from Phase 2) *(blocked: 94)* |
| 97  | . | Compile & validate PhilBench 500+ items *(blocked: 95, 96)* |
| 98  | . | Write PhilBench datasheet *(blocked: 97)* |
| 99  | . | Release PhilBench to HuggingFace Datasets *(blocked: 98)* |
| 100 | . | Write cross-architecture comparison report *(blocked: 95, 96)* |
| 101 | . | Write Llama 3.2 3B one-page question logs per thread *(blocked: 95, 96)* |

---

## Phase 8 — T7 Epistemic Contextualism + T8 Kratzer Modal Partition (Tasks TBD)

*Blocked: Phase 7 complete (#101)*

T7: DeRose/Lewis/Cohen — do "knows" representations shift with stakes context?
T8: Kratzer modal base — epistemic vs. circumstantial modal, same surface operator, distinct cluster?

Tasks TBD when Phase 7 complete.

---

## Phase 9 — T9 De Se Belief + PhilBench Update (Tasks TBD)

*Blocked: Phase 8 complete*

T9: Lewis/Perry self-locating belief — irreducible to propositional? Lingens/Sleeping Beauty. Llama only.

Tasks TBD when Phase 8 complete.

---

## Dependency Summary

```
Phase 0 (1–16) → Phase 1 (17–41) → Phase 2 (42–56) → Phase 3 (57–64)
→ Phase 4 (65–74) → Phase 5 (75–81) → Phase 6 (82–92) → Phase 7 (93–101)
→ Phase 8 (TBD) → Phase 9 (TBD)
```

T1b fixes (#25–27) complete — T1b Colab rerun (#28–29) unblocked.
T1c (#33–34) gated on T1b producing valid summary (#29).
T1b/T1c gated on T1a passing (#23) — enforced by `prerequisite_experiment_id`.
T1 SAE stretch goal (#40a) never blocks phase completion.
T3 belief-report (#65) runs before intentional-inexistence (#66) & modal-opacity (#67) in parallel.
Checkpoint sweep T1 (#58–59) & T2 (#60–61) run in parallel after checkpoints downloaded (#57).
T6 stimulus grammar (#88) unlocks independently within Phase 6 alongside T5.
T2b deferred to Phase 7 Llama sweep — GPT-2 gate fail + math prior requirement.
T2 GPT-2 gate FAIL — T2 runs on Pythia 1.4B instead (#44).
