# PoL-Probe — Philosophical Probing of Language Models

## Overview

Theory-driven mechanistic interpretability. Use formal theories from analytic philosophy as computational hypotheses. Test them via causal intervention. Convergence = finding. Divergence = deeper finding.

Central question: which formal philosophical theory of X describes a model's internal encoding of X — not whether X is encoded at all.

---

## §T Threads

### T1: Counterfactual Semantics — Worlds-Based vs. Causal-Graph

Tests whether model implements possible-worlds semantics (Lewis/Stalnaker) or causal-graph semantics (Pearl). These are not the same dispute. Lewis vs. Stalnaker is within worlds-based semantics. Worlds-based vs. Pearl is a deeper structural question.

**T1a — Level 3 Existence Test**

Establish counterfactual reasoning reaches Pearl's Level 3 before testing mechanism. Must use interventional stimuli — observational prompting tests rung-1 association, not rung-3.

Stimuli where P(C | A was prevented) ≠ P(C | A did not occur). Test which distribution governs model completion. Full-chain tracking → Level 3 confirmed, proceed to T1b. Proximate-only → Level 3 absent, T1b moot.

**T1b — Worlds-Based vs. Causal-Graph Mechanism**

Runs only if T1a confirms Level 3. Pre-specify the causal graph hypothesized (nodes, edges, structural equations) before running.

Three conditions:
- Forward (both agree true): `"If it had rained, the ground would be wet."`
- Backtracking (both false, different mechanism): `"If the ground were wet, it would have rained."`
- Would/might duality: same antecedent, test geometric separation

**T1c — Lewis vs. Stalnaker Within Worlds-Based**

Stalnaker: unique closest world (CEM holds). Lewis: similarity ordering, possible ties (CEM fails).

Borderline cases: if Lewisian, representations on tie cases are diffuse. If Stalnakerian, single centroid. Test geometric structure on cases designed to be equidistant from two possible worlds.

Minimum 30 minimal pairs per condition. Template grammar required — generate programmatically, do not hand-author.

**Primary mechanistic output: activation patching + layer-resolved curves**
SAE double dissociation is a stretch goal for T1 only — attempt after basic patching is working. Do not block T1 completion on SAE.

Expected outcomes:
- Similarity-weighted clustering, diffuse on borderlines → Lewisian
- Single centroid on borderlines → Stalnakerian
- Graph-structured clustering → Pearlian
- Flat/symmetric → neither

**T1d — Causal Identification (do-calculus)**

Runs after T1c. Tests whether model representations respect do-calculus identification conditions — specifically, whether the model internally distinguishes confounded structures with a valid adjustment set from those without one.

T1d is informative regardless of T1b outcome:
- If T1b = Pearl: tests whether Pearl-consistent representations respect identification constraints (strong claim — full do-calculus).
- If T1b = Lewis: tests whether Lewis-consistent representations fail identification tests (consistent with worlds-ordering having no notion of adjustability).

Four stimulus conditions:
- `back_door_adjustable` — confounded, observed covariate blocks back-door path. Back-door adjustment valid.
- `front_door_adjustable` — hidden confounder, mediator observed. Front-door adjustment valid.
- `confounded_not_adjustable` — confounded, no valid adjustment set. Effect not identified.
- `unconfounded_control` — direct causation, no confounding. Trivially identified.

Primary mechanistic output: binary identification probe (adjustable vs not_adjustable) + layer-resolved curves.

`identification_criterion` and `confounder_structure` required in config before lock (V14, V15). Prerequisite: T1b complete.

---

### T2: Sense and Reference — Frege, Not Quine

Quine's substitution failure is an argument against quantified modal logic, not a semantic theory. What this thread actually tests is Frege's sense/reference distinction.

Fregean prediction: representations in opaque contexts track mode of presentation (sense), not referent. Distinct senses of coreferential terms should produce distinct representations in opaque contexts even when referents are identical.

Stimuli: sense-distinct coreferential terms in transparent vs. opaque contexts.
- `"Hesperus is visible at dusk"` / `"Phosphorus is visible at dusk"` — transparent, same referent, should converge
- `"John believes Hesperus is a planet"` / `"John believes Phosphorus is a planet"` — opaque, different sense, should diverge

T2 causal intervention must specify: token position, layer range, component (residual stream vs. MLP output vs. attention output). Layer-resolved patching curves required — not a single patch.

Expected outcomes:
- Divergence in opaque, convergence in transparent → Fregean sense/reference encoded
- No divergence regardless of context → purely referential
- Divergence at specific layers only → localize where sense is computed

---

### T2b: Hyperintensionality

Tests whether model encodes distinctions finer than possible worlds — beyond what any worlds-based semantics can explain.

**Three stimulus classes — all required:**

1. **Syntactically distinct but logically equivalent:** `"P and Q"` / `"Q and P"` — NOT identical sentences. Representations must converge. Baseline distinguishes geometric convergence from literal token identity.
2. **Intensionally equivalent:** `"7 is prime"` / `"7 is not composite"` — same truth conditions all worlds, different cognitive content. Converge under Lewis/Stalnaker. May diverge under hyperintensional semantics.
3. **Intensionally distinct:** `"7 is prime"` / `"7 is even"` — different truth conditions. Must diverge. Positive control.

Finding: if class 2 cosine distance exceeds the 95th percentile of the cosine-ratio null distribution (labels permuted 1000×, computed before data collection) → model is hyperintensional. Threshold is null-derived, not post-hoc.

Stimulus domains: mathematical identities, analytic truths, definitional equivalences. Frequency-matched within one order of magnitude (see V7).

**Primary mechanistic output: activation patching + cosine distance curves**
SAE feature search is a stretch goal — attempt after basic patching is clean.

Expected outcomes:
- Class 2 converges with class 1 → Lewisian/Stalnakerian
- Class 2 diverges, magnitude correlates with cognitive difference → hyperintensional
- Some class 3 pairs converge → sub-intensional, model coarser than possible-worlds semantics

**Model requirement:** T2b requires a model with reliable mathematical knowledge (prime/composite/odd number theory). GPT-2 medium and Pythia 1.4B both fail the behavioral gate on this domain. T2b runs in Phase 7 on Llama 3.2 3B alongside the full cross-architecture sweep.

---

### T2c: Two-Dimensional Semantics — Chalmers

Runs after T2b on Llama 3.2 3B (Phase 7). Tests whether Llama encodes Chalmers' primary/secondary intension distinction.

T2b established hyperintensionality — distinctions finer than possible worlds. T2c asks whether those distinctions are organized along the primary/secondary intension axis specifically.

Three stimulus conditions (one experiment run per condition):
- `primary_sensitive` — terms with same secondary intension but different primary (water vs H2O in epistemic context). Expected: representations diverge.
- `secondary_necessary` — metaphysically necessary identity statements. Expected: representations converge.
- `primary_secondary_dissociation` — secondarily necessary but primarily contingent (Chalmers signature case). Expected: third distinct geometric cluster.

`intension_type` required in config before lock (V17). Prerequisite: T2b behavioral gate passing on Llama 3.2 3B (V16). Never runs on GPT-2 or Pythia — those models lack reliable scientific identity knowledge.

---

### T3: Intensional Context Failure

Not general compositionality — specifically where Fregean compositionality breaks under intensionality.

Three context types:
1. Belief reports: `"John believes Hesperus = Phosphorus"` — substitution of Phosphorus for Hesperus invalid
2. Intentional inexistence: `"looking for a unicorn"` — no referent to compose over
3. Modal opacity: coreferential terms under necessity operator

**Primary output: layer-of-failure curve per context type.**
Y-axis: probe accuracy predicting correct substitution behavior at each layer. X-axis: layer depth.

Expected outcomes:
- Failure at same layer across all three → general compositionality collapse
- Failure at different layers per type → context-specific
- Smooth degradation → compositionality approximate not categorical

---

### T4: Quinean Ontological Commitment

Stratified entity set:
- Concrete particulars: `"Socrates"`, `"Mt. Everest"`
- Abstract objects: `"the number 7"`, `"justice"`
- Properties: `"redness"`, `"being prime"`
- Events: `"the fall of Rome"`
- Tropes: `"Socrates' wisdom"`
- Fictional/intentional: `"Sherlock Holmes"`

RSA against theoretical similarity matrices for: Platonism, nominalism, trope theory, four-dimensionalism.

**Matrix construction protocol (required before Phase 5):** Matrices derived from BFO (Basic Formal Ontology) or DOLCE — published, citable, externally grounded formal ontologies. Not from intuition. Not from philosophical texts. Document which ontology version used and which relations drive each cell value.

**Statistical test:** Mantel permutation test (1000 permutations, p < 0.05) as significance criterion. Effect-size floor: observed RSA correlation must exceed the 95th percentile of the Mantel null distribution (the permuted-r values already computed by the test). Both required.

Pre-specified threshold: RSA correlation > null-95th-percentile AND p < 0.05 required to claim best-fit framework. null-95th-percentile recorded in mantel_result.json before any philosophical interpretation.

Expected outcomes:
- Abstract objects geometry-distant from concretes → Platonist signature
- No stable abstract clustering → nominalist signature
- No universal property cluster, each instance distinct → trope theory
- Divergence from all frameworks → novel emergent ontology

---

### T5: Grounding — Fine's Formal Conditions

Tests whether model representations satisfy all three of Fine's necessary conditions for grounding: asymmetry, irreflexivity, transitivity. Testing all three is what justifies calling this a grounding test — not just directionality. If representational dependence satisfies all three conditions in the mathematical domain (where grounding holds non-causally) and fails them in the causal control domain, the result is evidence for Fine's grounding structure specifically, not mere causal precedence.

**Three domains — required to distinguish grounding from causation and from circuit depth:**

1. **Mathematical domain (primary):** Number grounds its properties non-causally. `"7 is prime"`, `"7 is odd"`. No causal pathway exists. Intervene on number representation → measure cosine distance between clean and patched activation at property token position, averaged across stimuli. Intervene reverse direction.
2. **Physical domain (causal control):** Causal and grounding order align. Same metric.
3. **Unrelated pairs domain (depth control):** Semantically unrelated word pairs with matched co-occurrence statistics. If asymmetry here equals mathematical domain, the result measures circuit depth, not grounding.

**Three formal conditions — all required:**
- Asymmetry: intervention on A changes B; intervention on B does not change A
- Irreflexivity: patch property representation → effect on number representation should be near zero in mathematical domain. A does not ground itself.
- Transitivity: if A grounds B and B grounds C, intervention on A should propagate to C. Test three-node chains: number → property → derived fact. Verify full-chain intervention effect.

**Pre-specified asymmetry criterion:** mathematical domain asymmetry (cosine distance, L2 metric, specified layer range) must exceed the 95th percentile of the unrelated-pairs permutation null AND exceed the 95th percentile of the physical-domain permutation null. Both null distributions derived from 1000 label-permuted draws per domain before experiment runs. Both thresholds required.

Expected outcomes:
- Mathematical asymmetry exceeds both controls → grounding structure distinct from causation and circuit depth
- Mathematical ≈ physical, both exceed unrelated → measuring causation not grounding
- All three equal → no dependency structure encoded

---

### T6: Rigid Designation — Kripke

Kripke's *Naming and Necessity*: proper names are rigid designators — they pick out the same object in every possible world. Definite descriptions are non-rigid — they pick out whatever satisfies the description in each world.

Representational prediction: rigid designators should produce context-invariant representations across modal operators. Descriptions should shift representation with modal context.

Stimuli: proper names vs. definite descriptions that happen to be coreferential in the actual world.
- `"Aristotle was a philosopher"` vs. `"The student of Plato was a philosopher"` — transparent
- `"In world W, Aristotle might not have studied under Plato"` vs. `"In world W, the student of Plato might not have studied under Plato"` — modal context

Causal intervention: patch proper name representation into description processing position across modal contexts. Does context-invariance transfer?

Expected outcomes:
- Proper name reps stable across modal contexts, description reps shift → Kripkean rigid designation encoded
- Both shift → no rigidity distinction
- Both stable → model insensitive to modal context entirely

---

### T7: Epistemic Contextualism — DeRose / Lewis / Cohen

Contextualist claim: extension of "knows" shifts with salience of error-possibilities and stakes in attributor's context (Lewis "Elusive Knowledge" 1996; DeRose 1995; Cohen 1988). Invariantist competitor: standards fixed, context irrelevant to truth value of knowledge attribution (Hawthorne *Knowledge and Lotteries*). Relativist competitor: truth assessment-sensitive, not utterance-context-sensitive (MacFarlane).

Central question: do representations of "knows P" shift geometrically with contextual stakes framing, or remain stable?

Three conditions:
- `low_stakes` — everyday stakes, few relevant alternatives salient. `"We sometimes go on Saturdays. John knows the bank will be open."` Contextualist: "knows" true.
- `high_stakes` — same epistemic situation, high practical stakes. Deposit required or lose house. Contextualist: same claim false — skeptical alternative now relevant. Invariantist: same truth value as `low_stakes`.
- `skeptical` — radical skeptical alternative made explicit in context. Baseline for maximum standard elevation. Contextualist: "knows" false in both conditions above this threshold.

Minimum 30 minimal pairs per condition. Template grammar required — same propositional content, varied stakes framing only. Do not hand-author.

Primary mechanistic output: linear probe + layer-resolved activation patching on "knows" token position across conditions.

Expected outcomes:
- Representation of "knows P" shifts with stakes condition → contextualist signature
- Representation stable across conditions → invariantist / topic-neutral
- Third cluster distinct from low/high → relativist (assessment-sensitivity)

`stakes_level` required in config before lock (V19). Phase 8, primary on Pythia 1.4B + replication on Llama 3.2 3B.

---

### T8: Kratzer Modal Partition — Epistemic vs. Circumstantial

Kratzer: modals are quantifiers over accessible worlds; modal base parameter determines which worlds qualify (Kratzer "Modality" 1981, 2012). Epistemic base = worlds consistent with speaker's evidence. Circumstantial base = worlds consistent with relevant physical facts. Same surface operator "might" / "must" — different semantic value depending on base.

Competing: uniform modal force (no partition encoded — all "might" identical regardless of context). Distinct from T2c two-dimensional semantics — T2c tests intension type on terms; T8 tests modal base parameter on operators.

Three conditions (disambiguating context required in every stimulus — same surface operator across all three):
- `epistemic` — modal base = speaker's evidence. `"The keys might be in the drawer"` (I have not checked). Quantifies over epistemically accessible worlds.
- `circumstantial` — modal base = physical/causal circumstances. `"The bridge might hold under that load"` (given structural properties). Quantifies over circumstantially accessible worlds.
- `deontic` — modal base = normatively ideal worlds. `"You might leave early"` (it is permitted). Positive control — most distinct from epistemic; deontic and epistemic should diverge most.

Minimum 30 minimal pairs per condition. Frequency-match modal operator across conditions.

Primary mechanistic output: linear probe + activation patching on modal operator token position across conditions.

Expected outcomes:
- Geometric separation of epistemic from circumstantial at same surface form → Kratzer partition encoded
- No separation → uniform modal force, no base distinction
- Epistemic clusters with deontic → model conflates evidence-based and permission modality

`modal_base` required in config before lock (V20). Phase 8, primary on GPT-2 medium + replication on Pythia 1.4B.

---

### T9: De Se Belief — Lewis / Perry

Lewis: belief is relation to property, not proposition. Self-locating belief requires centered world ⟨world, agent, time⟩ — irreducible to any de dicto propositional content (Lewis "Attitudes De Dicto and De Se" 1979). Perry's essential indexical: `"I am making the mess"` triggers different downstream behavior than `"Scott is making the mess"` even when coreferential — indexical non-eliminable (Perry 1979).

Competing: propositionalism — de se reducible to de dicto via description. `"I believe I am the tallest"` paraphrases to `"John believes John is the tallest"` without residue.

Three conditions:
- `de_se` — self-locating indexical belief. `"I don't know who I am."` / `"I believe I am being followed."` Centered-world content irreducible to proposition. Lewis: no propositional equivalent.
- `de_dicto` — propositional belief, same content, no essential indexical. `"John doesn't know that he is John."` / `"Mary believes the followed person is in danger."` Propositionalist: equivalent to paired de_se stimulus.
- `de_re` — belief about object under acquaintance. `"John believes of Mary that she is the tallest."` Positive control — distinct from both de_se and de_dicto.

Stimulus domains: amnesia cases (Lingens library, Heimson), Sleeping Beauty variants, Perry grocery-store cases. Minimum 30 pairs per condition. Template grammar required.

Primary mechanistic output: linear probe + activation patching on belief-report verb + embedded clause position across conditions.

Expected outcomes:
- De se clusters distinct from de dicto for propositionally equivalent pairs → Lewis primitive encoded; centered-worlds irreducible
- De se clusters with de dicto → propositionalism, no self-locating primitive
- De se + de re cluster together, de dicto separate → direct-reference encoding, not centered-worlds

Model requirement: T9 requires strong indexical and propositional attitude comprehension. GPT-2 medium and Pythia 1.4B expected to fail behavioral gate on Sleeping Beauty / Lingens cases. T9 runs in Phase 9 on Llama 3.2 3B only — never on GPT-2 or Pythia.

`attitude_type` required in config before lock (V21). T9 `run.py` asserts `model_id` is Llama 3.2 3B before experiment proceeds (V22). Phase 9.

---

## §M Methodology

**Rigorous = L2 + L3 + surface-stats null. All three required.**

- L2 (linear probe): train classifier on layer activations → identifies which layer encodes distinction
- L3 (activation patching): replace representation mid-forward-pass → proves causal role
- Surface-stats null: measure how much geometric divergence is explained by unigram frequency, phrase length, embedding-space distance at frozen embedding layer (no transformer computation). If null accounts for most variance, philosophical interpretation unwarranted.

**Mean-ablation baseline required on every patching experiment.** Replace activation with mean over stimulus set. If mean ablation also breaks behavior, patch result is uninformative about specificity.

**Behavioral gate before mechanistic analysis (V8).** Model must pass >70% on forced-choice behavioral version of each stimulus set before probing internals. If model behavior is wrong, probing is uninterpretable.

**Pre-specification:** expected outcomes written per thread before data collection. Not post-hoc.

---

## §P Phases

Each phase ships something independently runnable. Next phase does not start until current phase passes all §V invariants. Cross-architecture replication runs per thread at end of that thread's phase — not batched at the end.

| Phase | Contents | Ships |
|-------|----------|-------|
| 1 | Core infrastructure + T2 + T2b complete + replication on Pythia | Extraction, RSA, probe, patching, Frege sense/reference, hyperintensionality results |
| 2 | T1 complete + replication on Pythia | Counterfactual semantics, worlds-based vs. causal-graph results |
| 3 | Checkpoint analysis on T1+T2 | Developmental curves, emergence timing |
| 4 | T3 complete + replication | Intensional context failure curves |
| 5 | T4 complete + replication | Ontological commitment RSA |
| 6 | T5 + T6 complete + replication | Grounding asymmetry, rigid designation |
| 7 | Full cross-architecture sweep | T1–T6 on Llama 3.2 3B |
| 8 | T7 + T8 complete + replication | Epistemic contextualism, Kratzer modal partition |
| 9 | T9 complete (Llama 3.2 3B only) | De se / centered-worlds vs. propositionalism |

---

## §R Roles

**Advisors design.** Every function spec, stimulus set, expected outcome, and experimental design decision comes from the SPEC.

**You write functions.** Given a spec for a function — inputs, outputs, what it must do — you implement it. Nothing more, nothing less.

---

## §L Learning Arc

| Stage | Task | Read before starting |
|-------|------|----------------------|
| 0 | Replicate Tenney edge probing | Tenney et al. 2019 |
| 1 | Transformer internals: residual stream, attention, MLP | Elhage et al. "Mathematical Framework" |
| 2 | Build core infrastructure | — |
| 3 | T2 + T2b experiments | Frege "Sense and Reference" + Hernandez et al. |
| 4 | T1 experiments | Lewis *Counterfactuals* ch.1 + Meng et al. ROME |
| 5 | Checkpoint analysis | — |
| 6 | T3 experiments | Stalnaker "A Theory of Conditionals" + Hupkes et al. |
| 7 | T4 experiments | Quine "On What There Is" + Tenney et al. |
| 8 | T5 + T6 experiments | Fine "Question of Ontology" + Kripke *Naming and Necessity* ch.1 + Nanda Othello-GPT |
| 9 | Full cross-architecture sweep | — |
| 10 | T7 experiments | DeRose "Solving the Skeptical Problem" + Hawthorne *Knowledge and Lotteries* ch.1 + Lewis "Elusive Knowledge" |
| 11 | T8 experiments | Kratzer "Modality" (in Arnim von Stechow & Wunderlich eds.) + von Fintel "Modality and Language" |
| 12 | T9 experiments | Lewis "Attitudes De Dicto and De Se" + Perry "The Problem of the Essential Indexical" + Elga "Self-Locating Belief and the Sleeping Beauty Problem" |

**Per thread — after running:**
Write one page: what did the results raise that the theory did not predict? That page is where originality lives.

---

## §D Design Principles

- **Modularity** — each module one job, clean interfaces. Swapping model or probe type touches one file.
- **Reproducibility** — every experiment produces config file capturing exact parameters. Results rerunnable from config alone.
- **Fail fast** — V8 behavioral gate runs before any mechanistic analysis. Bad stimuli caught early.
- **Separation of concerns** — stimuli, extraction, probing, intervention, visualization are independent layers. Philosophy work touches stimuli only.
- **Incremental verifiability** — every function testable in isolation before use in experiment.
- **Schema first** — stimulus schema, activation schema, result schema defined before code written. All data flows through them.
- **No premature abstraction** — build for nine threads that exist. Generalize only when third thread proves pattern.

---

## §I Infrastructure

**Directory:**
```
pol-probe/
├── core/
│   └── io.py                     # save_result, load_result, load_results — shared
├── stimuli/
│   ├── schemas/                  # stimulus.schema.json, philbench.schema.json
│   ├── grammars/                 # one file per thread, programmatic generation
│   ├── generated/{thread_id}/    # pairs.jsonl, behavioral_items.jsonl
│   ├── validated/{thread_id}/    # pairs.validated.jsonl — only path extraction accepts
│   ├── philbench/                # philbench.jsonl
│   └── pipeline.py               # owns: run_behavioral_gate, validate_set, generate_pairs,
│                                 #        check_frequency_match, build_philbench_entry
├── extraction/
│   └── extractor.py              # extract_activations — enforces validated/ path at runtime
├── probes/
│   └── probes.py                 # run_rsa, run_linear_probe, run_knife_mi, run_mantel_test
├── interventions/
│   └── interventions.py          # all patch/ablation functions, assert_specificity_valid
├── sae/                          # stretch goal — T1 only
├── visualizations/
└── experiments/
    └── {thread_id}/
        ├── config.json           # canonical ExperimentConfig — serialized before run
        ├── stimuli.jsonl         # symlink → stimuli/validated/{thread_id}/
        ├── run.py                # run_surface_null, check_phase_gate, run_experiment
        └── results/
            ├── surface_null.json # written FIRST — all other writes gated on its existence
            ├── probe_results.jsonl
            ├── mantel_result.json
            └── summary.json
```

**ExperimentConfig** (canonical schema, serializes to config.json):
```
experiment_id: str                # "{thread_id}_{run_timestamp}"
thread_id: str
model_id: str
model_revision: str
layer_range: tuple[int,int]
component: str
token_positions: list[int]
probe_type: str
rsa_permutations: int             # default 1000
seed: int
stimulus_file: Path               # must be under stimuli/validated/ — enforced by extractor
stimulus_sha256: str              # canonical gate; path restriction in extractor is removed
pre_spec_locked: bool             # must be True before run_experiment proceeds
frequency_match_verified: bool    # set only by validate_set — never by caller code
behavioral_gate_threshold: float  # floor hardcoded at 0.70 — run_behavioral_gate raises if below
t5_asymmetry_thresholds: dict | None  # required non-null for T5 — check_phase_gate asserts this
run_timestamp: str
expected_outcomes: dict
prerequisite_experiment_id: str | None  # T1b/T1c require T1a summary with level3_confirmed=True
ontology_version: str | None      # required non-null for T4 — specifies BFO/DOLCE version used
matrix_source: str | None         # required non-null for T4 — documents which relations drive RSA matrix
stakes_level: str | None          # required non-null for T7 — 'low' | 'high' | 'skeptical'
modal_base: str | None            # required non-null for T8 — 'epistemic' | 'circumstantial' | 'deontic'
attitude_type: str | None         # required non-null for T9 — 'de_se' | 'de_dicto' | 'de_re'
```

**Enforcement rules baked into code:**
- `run_experiment` asserts `pre_spec_locked=True` before any extraction
- `run_experiment` writes `surface_null.json` first; raises if write fails before continuing
- `assert_specificity_valid` called inside results-writing path, not optionally
- T1b/T1c `run.py` asserts prerequisite T1a `summary.json` has `level3_confirmed=True`
- `check_phase_gate` asserts `t5_asymmetry_thresholds` non-null before T5 runs
- `run_rsa` asserts `ontology_version` and `matrix_source` non-null for T4
- `behavioral_gate_threshold` floor: `run_behavioral_gate` raises `ValueError` if threshold < 0.70
- `frequency_match_verified` is set exclusively by `validate_set` — no external setter
- `check_phase_gate` asserts `stakes_level` non-null before T7 runs
- `check_phase_gate` asserts `modal_base` non-null before T8 runs
- `check_phase_gate` asserts `attitude_type` non-null before T9 runs
- T9 `run.py` asserts `model_id` contains "llama" (case-insensitive) before experiment proceeds

**Stack:**
```
transformer_lens    # activation extraction, patching
sae_lens            # SAE feature search (stretch goal)
torch               # tensor ops
sklearn             # linear probes
scipy               # Mantel permutation test
datasets            # benchmark hosting
matplotlib          # visualization
```

**Models:**
- GPT-2 medium/large — local development
- Pythia 1.4B — Colab, checkpoint analysis + per-thread replication
- Llama 3.2 3B — Colab, final cross-architecture sweep

---

## §B Benchmark Output

PhilBench: theory-annotated minimal pairs across all nine threads.

Each entry: sentence, philosophical category, formal theory source, competing prediction, falsification condition, corpus frequency of key terms.

Target: 500+ items. Released on HuggingFace Datasets with full datasheet.

---

## §V Invariants

V1: L2 probe ∀ thread — locates distinction
V2: L3 intervention ∀ thread — verifies causal role
V3: expected outcomes written before data collection ∀ thread
V4: specificity control ∀ intervention — mean-ablation baseline required, disruption must be selective
V5: T1b + T1c run only if T1a confirms Level 3
V6: T5 asymmetry thresholds pre-specified before experiment runs, both mathematical and depth controls required
V7: frequency matching within one order of magnitude verified before stimulus finalization ∀ thread
V8: behavioral pass-rate gate >70% on forced-choice version of each stimulus set before mechanistic analysis
V9: SAE double dissociation — stretch goal T1 only. Required only if basic patching is complete and clean. Never blocks phase completion.
V10: surface-statistics null hypothesis computed ∀ thread before philosophical interpretation
V14: `identification_criterion` non-null required before T1d config locks or runs.
V15: `confounder_structure` non-null required before T1d config locks or runs.
V16: T2c requires T2b behavioral gate passing on Llama 3.2 3B — enforced via `prerequisite_experiment_id`. T2c is moot if the model is not hyperintensional.
V17: `intension_type` non-null required before T2c config locks or runs. Valid values: 'primary' | 'secondary' | 'dissociation'.
V18: `circuit_analysis_enabled=True` requires layer_sweep.json to exist in the thread's results directory before head sweep runs.
V19: `stakes_level` non-null required before T7 config locks or runs. Valid values: 'low' | 'high' | 'skeptical'.
V20: `modal_base` non-null required before T8 config locks or runs. Valid values: 'epistemic' | 'circumstantial' | 'deontic'.
V21: `attitude_type` non-null required before T9 config locks or runs. Valid values: 'de_se' | 'de_dicto' | 'de_re'.
V22: T9 runs only on Llama 3.2 3B — `run.py` asserts model_id before experiment proceeds. GPT-2 and Pythia expected to fail behavioral gate on Lingens/Sleeping Beauty cases; never run T9 on those models.
