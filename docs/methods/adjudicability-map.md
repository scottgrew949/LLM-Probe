# The Adjudicability Map

*Which formal philosophical theories mechanistic interpretability can and cannot adjudicate in a language model — and why.*

Draft methodological contribution, PoL-Probe (2026-06-13). Status: argument, not theorem — contestable points flagged in §7.

---

## 1. Thesis

A mechanistic-interpretability study can adjudicate between two formal philosophical theories of some construct X only when the theories' **disagreement has a signature that one of the available tools can register**. When it cannot, the study does not test the theories — it tests something correlated and easier, and reports it under the theories' names. Most past failures in this project were exactly that substitution.

Two **independent** axes decide whether a thread is adjudicable:

- **Tool-blindness** — a property of the (method × theory) pair. The measurement instrument cannot register the construct *regardless of the model*. No model fixes it.
- **Capability-blindness** — a property of the (model × theory) pair. The model lacks the competence the test presupposes, *regardless of the method*. A more capable model fixes it.

Conflating these is the central error. "Bigger model" is the answer to capability-blindness and is irrelevant to tool-blindness.

## 2. The toolkit and what each instrument registers

Every tool in standard mechanistic interpretability measures a property of **representations of presented stimuli** (or of outputs for those stimuli). The unit of analysis is the stimulus, the stimulus pair, or the stimulus set.

| Tool | Measurement domain | Registers |
|---|---|---|
| Behavioral (next-token logprobs) | output distribution over completions for a prompt | truth-conditional / dispositional differences that surface as differential output |
| Linear probe (L2) | linear decodability of a label from a layer's per-stimulus activation | whether information about a distinction is linearly present per stimulus |
| Activation patching (L3) | causal effect on output of substituting an activation at (layer, position) between runs | whether a *localized* activation causally carries a distinction |
| RSA | correlation of the model's pairwise stimulus-dissimilarity matrix with a hypothesized matrix | the similarity *geometry* across a stimulus *set* |

The shared structure is decisive: each instrument sees a feature of, or a relation among, the representations of the **stimuli actually presented**. Anything a theory ranges over that is *not* a presented-stimulus representation is, by default, outside the instrument's domain.

## 3. Visibility criterion

**A construct is tool-visible iff it predicts a difference in one of:** the output distribution, a per-stimulus activation, the causal role of a localized activation, or the cross-stimulus geometry — i.e., the construct is a property of, or relation among, the model's representations of the presented stimuli.

When that holds, the theories are adjudicable *provided they also diverge* (see §4, case B) and *provided the model is competent* (capability axis).

## 4. Why a theory-discrimination fails — the taxonomy

A test of "theory A vs theory B for construct X" can fail in four distinct ways. Only the first is success.

**A. Visible and discriminating** → testable. The theories predict different stimulus-level signatures and a tool registers the difference. *(T2, T4, T6, T8.)*

**B. Visible but convergent** → not discriminable *here*. Both theories predict the *same* visible signature at the cases the model can handle, so the tool sees the phenomenon but cannot attribute it. Engineer a divergence case if one exists and is itself visible. *(Lewis ≡ Pearl on truth-conditions for simple recursive SCMs — Briggs 2012, Halpern 2013. The divergence case — backtracking / similarity-intrusion — exists but is capability-risky; see §5.)*

**C. Capability-blind** → fixable by a more capable model. The construct is visible in principle, but the model does not compute it. *(Backtracking counterfactuals, prime/composite knowledge, scientific-identity knowledge, stakes pragmatics — all plausibly above a 3B model.)*

**D. Tool-blind / type-mismatch** → no model fixes it. The construct ranges over something that is not a presented-stimulus representation. Four sub-types:

- **(i) Within-evaluation domain.** The construct ranges over entities compared *inside the evaluation of a single stimulus*, which are not themselves presented stimuli. *Lewis's comparative world-similarity* ranks possible worlds while evaluating one counterfactual; the instrument holds only that one stimulus's representation, not the space of compared worlds. The representation may encode the *result* of the comparison — but that result is the truth-value, which converges with Pearl (case B). The *ordering qua ordering* is not a feature current tools isolate. **Tool-class-invariant** (a better probe does not change the type of thing being measured).
- **(ii) Universal negative.** The construct is a claim quantifying over *all* possible representations. *De se irreducibility* asserts that **no** de dicto proposition captures the content. Probing can show de se is represented distinctly from any *candidate* reduction (strong inductive evidence of non-identity), but never that *no* reduction exists. The *strong* irreducibility thesis is unfalsifiable by these tools; only pairwise distinctness is visible. **Tool-class-invariant** (a logical-type limit, not an instrument limit).
- **(iii) Proxy-only / bridge.** The tool sees a causal or statistical shadow of a metaphysical relation. *Fine's grounding* is non-causal dependence; patching sees counterfactual dependence of output on activation, a causal/computational notion. A three-domain control (mathematical = non-causal) licenses the *inference* "asymmetric dependence in a non-causal domain ⇒ grounding-like structure" — but that is a **bridge assumption**, not a measurement. Report it as such; never assert "we measured grounding." **Tool-relative** (a feature-level tool such as an SAE might tighten the bridge).
- **(iv) Measurement-level confound.** The construct returns the *same instrument value* as a mundane quantity. *Stalnaker's tie* (genuine semantic indeterminacy between equidistant worlds) and high next-token entropy (frequency competition between completions) both yield high representational dispersion; the dispersion statistic cannot separate them. **Tool-relative** (a control that holds frequency fixed could, in principle, separate them — but the minimal-pair design must achieve it, and at 3B that is unproven).

**Tool-class-invariant (i, ii) vs tool-relative (iii, iv):** the first pair cannot be rescued by any instrument of this class (they are type/logic limits); the second pair might soften with better instruments (SAEs, tighter controls) but are not rescued by a bigger model.

## 5. Worked example — Lewis's world-similarity (type D-i), and its escape

Lewis (1973): "if A were the case, C would be" is true iff some A-and-C world is closer to actuality than any A-and-not-C world, on a comparative-similarity ordering of worlds. The semantics ranges over a **space of worlds compared during the evaluation of one sentence**.

Why RSA-over-stimuli cannot see it: RSA correlates the model's *between-stimulus* dissimilarity matrix with a hypothesized matrix. The redesigned T1b proposed `M_sim` (cluster by topic) for "Lewisian" and `M_graph` (cluster by causal topology) for "Pearlian." But:
- `M_sim` is topical clustering of the *prompts* — the dominant axis of variance in any LM's embeddings — not Lewis's world-similarity ordering. Tracking it shows the model behaves like a language model, nothing about Lewis.
- `M_graph` is causal-structure representation, which a Lewisian model *also* requires (Lewis's similarity weighting prioritizes matching laws and particular facts near the antecedent — that *encodes* the topology). It is necessary for both theories, so it discriminates neither.

The similarity ordering Lewis cares about is a relation among non-presented worlds *inside one evaluation*; no between-stimulus geometry is that relation. Type mismatch → invisible to RSA as operationalized.

**The escape (constructive corollary).** Make the compared entities into *presented stimuli*. Where Lewis's closest-world and Pearl's `do()`-surgery diverge — **backtracking and similarity-intrusion** cases — the two theories predict different *truth-values*, which is a behavioral (output-distribution) signature: visible. The within-evaluation relation is re-expressed as a between-stimulus behavioral contrast. This is case B's "engineer a divergence case," and it converts a D-i invisibility into an A-visibility — at the cost of capability risk (C), since backtracking reasoning is hard at 3B. Hence T1b is GATE-FIRST with null as a legitimate result.

The general move: **to see an invisible construct, re-operationalize so the thing the theory ranges over becomes a presented stimulus** — or accept it is out of reach and report that.

## 6. Per-construct adjudication

| Thread / construct | Failure mode | Tool-visible? | Verdict |
|---|---|---|---|
| T1 Lewis vs Pearl, *truth-conditions* | B (convergent) + escape via backtracking | escape is behaviorally visible | GATE-FIRST (capability-risky) |
| T1 Lewis world-similarity *ordering* | **D-i** | no (tool-class-invariant) | shelve qua ordering; test via §5 escape |
| T1c Lewis vs Stalnaker (tie) | **D-iv** confound + C | partially; dispersion confounds with entropy | GATE-FIRST; flag confound |
| T2 Frege sense vs reference | A | yes (opaque vs transparent; within-model control) | BUILD |
| T2b hyperintensional equivalence | A, gated by C (math) | yes (cosine vs null) | GATE-FIRST |
| T2c primary/secondary intension | C (steep) | yes in principle | shelve @3B |
| T3 intensional context failure | not theory-vs-theory | yes (localization) | DESCRIPTIVE |
| T4 Quinean ontology | A | yes (RSA vs externally-anchored matrices) | BUILD (strongest) |
| T5 Fine grounding | **D-iii** bridge + C | proxy only | GATE-FIRST; report bridge |
| T6 Kripke rigid designation | A | yes (cross-modal-context invariance via patching) | BUILD |
| T7 epistemic contextualism | A, gated by C (stakes) | yes (rep shift across stakes) | GATE-FIRST |
| T8 Kratzer modal partition | A | yes (epistemic vs circumstantial cluster) | BUILD |
| T9 de se irreducibility | **D-ii** universal negative + C | strong thesis no; distinctness yes | shelve (strong thesis) |

## 7. Caveats (not buried)

- This is an **argument about operationalization**, not a metaphysical impossibility proof. "Invisible" means "not registered by the standard toolkit as currently used."
- **Tool-relativity is real.** Sparse autoencoders or tighter controls could move D-iii and D-iv items toward visibility (e.g. a monosemantic feature, or a frequency-matched dispersion control). D-i and D-ii are more robust: within-evaluation ranging and universal negatives are type/logic limits, not instrument limits.
- The visible/discriminating verdicts (A) still require the model to be **competent** (capability axis) — visibility is necessary, not sufficient.
- The map is over the **standard toolkit** (probe, patching, RSA, logprobs). A genuinely new instrument warrants re-running it.

## 8. Why this is a contribution

The map turns "we couldn't get a result" into a *result*: a principled boundary between the philosophy mechanistic interpretability can adjudicate and the philosophy it cannot, with the reason typed for each case. It tells a researcher, before spending compute, whether a thread can succeed, fail informatively, or only produce a confound wearing a theory's name.
