# T4 — Ontology Scaffold & Theory-Matrix Construction

The validity spine of T4. Documents (1) the published ontology that supplies entity categories, (2) how each stimulus maps to a category, (3) the explicit rule by which each metaphysical theory's similarity matrix is derived from those categories, and (4) the controls that keep the result an ontology measurement rather than a frequency artifact.

Satisfies V13 (`ontology_version` + `matrix_source` non-null). `matrix_source` → `stimuli/theoretical_matrices/t4_matrices.py` (built in #15 from the rules here).

---

## 1. Scaffold — DOLCE (DUL)

**Choice:** DOLCE, OWL operationalization **DOLCE+DnS Ultralite (DUL)**. Foundational ref: Gangemi, Guarino, Masolo, Oltramari, Schneider, *Sweetening Ontologies with DOLCE* (2002); DUL maintained by ISTC-CNR (namespace `http://www.ontologydesignpatterns.org/ont/dul/DUL.owl`).

**Why DOLCE not BFO:** DOLCE is *descriptive* — built to track how natural language and cognition carve the world, which is what a language model is trained on. It is metaphysically light-touch, so it does **not** pre-load a realist commitment; BFO is realist by design, and using a realist scaffold would subtly favor Platonist readings when nominalism is a rival. DOLCE also has the abstract/quality structure we need (below).

**Pinned version (V13 `ontology_version`):** `DOLCE+DnS Ultralite (DUL), owl:versionInfo 3.27`, pinned to the **immutable DBpedia Databus snapshot `2021-02-22`** (`https://databus.dbpedia.org/ontologies/ontologydesignpatterns.org/ont--dul--DUL--owl/2021.02.22-022820`). Rationale: the live `DUL.owl` is mutable; a dated snapshot is reproducible. Namespace `http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#`.

> **Category-IRI confirmation is deferred to code (#15):** `t4_matrices.py` resolves each category IRI against the pinned DUL file and **fails loud** on any missing/renamed IRI — so the mapping is verified by execution, not by assertion here.

**DUL categories we use** (`[verify]` IRIs):
- `dul:PhysicalObject` ⊂ `dul:Endurant` — wholly present at a time (concrete particulars).
- `dul:Event` / `dul:Process` ⊂ `dul:Perdurant` — happenings with temporal parts.
- `dul:Quality` — a **particular** quality *inhering in* an entity (note: DOLCE qualities are already trope-like — a specific instance, not a shared universal).
- `dul:Region` / `dul:Abstract` — abstract values / abstracta (numbers, regions in a quality space).
- `dul:Concept` / `dul:SocialObject` — universals-as-concepts, social/abstract objects (justice).

## 2. Entity strata → category mapping

| Stratum | Example stimuli | DUL category |
|---|---|---|
| Concrete particular | Socrates, Mt. Everest | `PhysicalObject` (Endurant) |
| Abstract object | the number 7, justice | `Region`/`Abstract`; justice → `SocialObject` |
| Property / universal | redness, being prime | universal reading → `Concept`; particular reading → `Quality` (the theories disagree exactly here) |
| Event | the fall of Rome | `Event` (Perdurant) |
| Trope | Socrates' wisdom | `Quality` (particular, inhering in Socrates) |
| Fictional / intentional | Sherlock Holmes | **no native DUL category** — treat as intentional/non-existent object; documented limitation, flagged in the summary, not forced into a category |

## 3. Theory-matrix rules (the V13 core)

Each matrix is a predicted **dissimilarity** over the entity set: `M_theory[i][j]` = distance under that theory, a documented function of entity i and j's DUL categories. Low = the theory says these are alike; high = unlike. Each rule names exactly which DUL distinction drives the cell.

**Platonism** — abstracta are real, mind-independent, a distinct realm; a universal is *one* entity instanced by many.
- low if both i,j ∈ {Abstract/Region, Concept(universal)} — the abstract realm clusters.
- high if one ∈ abstract-realm and the other ∈ {PhysicalObject, Event} — the master cleavage is abstract vs concrete.
- a trope (`Quality`) is read as an instance of its universal → close to the corresponding `Concept`.
- *Driver:* the abstract-vs-concrete category split; universals unified.

**Nominalism** — no abstracta, no universals; only particulars and predicates.
- abstracta do **not** cluster among themselves (no abstract realm).
- "redness" sits near its concrete instances (co-predication), not near "being prime".
- *Driver:* concrete co-instantiation / shared predication, **not** category membership. The abstract cleavage is absent — the key contrast with Platonism.

**Trope theory** (resemblance-class — Williams, Campbell; confirmed 2026-06-13) — properties are particular tropes; no universal *entity* exists, but exactly-resembling tropes form a **resemblance class** that does universals' work.
- same-kind tropes cluster (Socrates' wisdom ~ Plato's wisdom = 1) — the resemblance class.
- `property` (would-be universal) entities have no referent → property↔trope = 0, property↔property = 0.
- tropes also sit close to their **bearer**.
- *Drivers:* (a) resemblance class clusters same-kind tropes — this is the cell that separates trope theory from **nominalism** (which has no resemblance-class machinery → those pairs = 0); (b) property↔trope = 0 — the cell that separates it from **Platonism** (which unifies via the one universal → property↔trope = 1).

**4-dimensionalism (perdurantism)** — objects are spacetime worms with temporal parts; objects and events are the same kind of extended entity.
- low between `PhysicalObject` and `Event` (both 4D worms) — the Endurant/Perdurant split **collapses**.
- *Driver:* merging `Endurant(physical)` with `Perdurant` — the key contrast with every 3D view.

These produce structurally different matrices: Platonism's master cleavage = abstract/concrete; Nominalism = no abstract cleavage; Trope = properties fragmented to bearers (differs from Platonism on the property entities specifically); 4D = object/event cleavage collapses (differs from all).

## 4. Controls (from the #14 critique)

1. **Matrix discriminability (T1b lesson — propose as V24).** The four matrices share *some* structure (same entities), so a hard pairwise-decorrelation threshold like T1b's `|corr|<0.2` is the wrong instrument for four overlapping matrices. Instead:
   - report the full 4×4 inter-matrix correlation matrix as an artifact before any model run;
   - the **verdict requires the winning theory's RSA to exceed the runner-up's by a margin beyond the Mantel null** — a model-*comparison*, not merely winner > null. If two matrices are near-collinear, the comparison is reported as inconclusive between them, never force-picked.
2. **Geometry ≠ ontology (faithfulness).** Run `run_surface_null` first; frequency-match across strata (V7); verify the theory matrices don't themselves proxy word-frequency or co-occurrence (correlate each theory matrix with a frequency/co-occurrence RDM and report it). If a theory matrix correlates highly with the frequency RDM, a "match" is uninterpretable.
3. **n per stratum.** Pre-register a minimum n per DUL category so the model RDM is stable; report split-half RDM reliability alongside the Mantel result.
4. **Trope/fictional stimulus care.** Phrase tropes as bearer-bound ("Socrates' wisdom") and flag fictional entities as a documented out-of-scaffold category, not a forced cell.

## 5. Verdict criterion (pre-registered)

Best-fit theory requires: RSA correlation > Mantel null-95th-percentile **and** p < 0.05 **and** exceeds the runner-up theory beyond the null margin (control 1). All recorded in `mantel_result.json` before any philosophical interpretation. Expected-outcome shapes (Platonist / nominalist / trope / 4D / divergent-from-all) pre-written per the SPEC §T4.

### 5.1 Measured matrix discriminability (53-entity set, 2026-06-13)

Inter-matrix correlations on the generated entity web (`pairwise_theory_correlations`):

```
platonism↔nominalism 0.44   nominalism↔fourdim 0.37   platonism↔fourdim 0.48
platonism↔trope      0.65   trope↔fourdim      0.58   nominalism↔trope   0.71
```

**Pre-registered consequence — nominalism vs trope is not separable here.** The two are theoretically kin (both anti-Platonist about universals) and share almost all cells; the differing cells (abstract↔abstract, property↔instance, same-kind tropes under resemblance-class) are a thin minority. We do **not** engineer the correlation down — manufacturing separation by tuning the stimulus set would be construct substitution. Instead:

- T4 resolves **Platonist vs 4-dimensionalist vs anti-universalist camp** (those contrasts are well-separated, 0.37–0.48).
- **Nominalism vs trope is reported jointly** ("anti-universalist") unless the model's RSA to one beats the other beyond the Mantel null (control 1) — which the verdict rule already enforces. This is the adjudicability-map "convergent theories" case (`adjudicability-map.md §4 B`) at the matrix level: a documented resolution limit, reported as a finding.
