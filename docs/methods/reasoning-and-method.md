# How This Project Reasons — A Walkthrough

A teaching companion to `adjudicability-map.md` (the formal version) and `../superpowers/specs/2026-06-13-rescope-decision.md` (the decisions). This one is for building intuition: read it top to bottom and you should hold the whole method in your head. It assumes you know the philosophy (Lewis, Stalnaker, Pearl, Frege, Fine, Kripke, Kratzer); it explains the interpretability side as it goes.

---

## 1. The question, and the trap inside it

We treat a formal philosophical theory as a **hypothesis about the model's internals**: not "does the model handle counterfactuals," but "*which* theory of counterfactuals describes how it does." Convergence with a theory is a finding; divergence from all of them is a deeper finding.

The trap — the one that produced every past mess — is **construct substitution**. A hard-to-test question ("does the model use Lewis's similarity ordering?") quietly becomes an easy-to-measure one ("do counterfactual sentences cluster by topic?"), and the easy number gets reported under the hard question's name. The whole method below exists to make that swap *visible and refused*.

## 2. The interpretability toolkit (plain version)

Four instruments. Each one only ever looks at the model's **representation of the sentences you feed it**. Keep that in mind — it's the key to everything in §4.

- **Behavioral test (logprobs).** Give the model a prompt and two endings; see which it assigns higher probability. This is just "what would it say." Cheapest. Sees *outputs*.
- **Linear probe.** Take the model's internal activation vector at some layer, train a simple classifier to predict a label (e.g. "is this the opaque-context sentence?"). If it succeeds, the distinction is *present* in that layer. Caveat (Pavlick's own point): "present" ≠ "the model uses it."
- **Activation patching.** Run sentence A, copy one internal activation, paste it into the run of sentence B, and see if B's output changes. If it does, that activation *causally carries* the thing you swapped. This is the closest we get to "the model actually relies on this."
- **RSA (representational similarity analysis).** Take many sentences, measure how similar the model's representations are pairwise, and ask which *theory's* predicted similarity-pattern that matches. Sees the *geometry of a whole set*.

## 3. Gate and discriminator — the load-bearing idea

Split every thread into two questions:

- **Gate:** is there anything here to theorize about? Built on what the rival theories **agree** predicts. It's deliberately *theory-neutral*.
- **Discriminator:** which theory is right? Built on where they **disagree**.

Why neutral gates matter: if your gate already speaks one theory's language, you've prejudged the contest. Example: Pearl's `do`/`see` distinction is *Pearl's* machinery. If T1's existence gate were "does the model tell `do` from `see`," a Lewisian model could fail a Pearl-flavored gate and we'd wrongly conclude "no counterfactual reasoning here." So the gate must rest on the *shared* prediction — both Lewis and Pearl agree that "if the match hadn't been struck, the fire wouldn't have started" should complete with the fire *absent*. Pass that, and you've earned the right to ask *how* (similarity ordering vs causal graph) — which is the discriminator's job.

The logic chains: **gate fails → no phenomenon → the theory question is moot → don't run the discriminator.** (This is the V10 prerequisite invariant in code.)

## 4. Can the tools even see it? — two axes, one taxonomy

Before building a thread, ask whether the theories' disagreement leaves a mark a tool can register. Two *independent* reasons it might not:

- **Capability-blind:** the model isn't smart enough. A bigger model fixes this.
- **Tool-blind:** the *method* can't register the construct, no matter the model. A bigger model does nothing.

Confusing these is the classic mistake. Now the four outcomes (formal version in `adjudicability-map.md §4`):

- **A — visible & discriminating:** theories predict different signatures, a tool sees them. *Testable.* (T2, T4, T6, T8.)
- **B — visible but convergent:** both theories predict the *same* signature at reachable cases, so the tool sees the phenomenon but can't attribute it. Need a constructed divergence case. (Lewis ≡ Pearl on simple counterfactuals.)
- **C — capability-blind:** visible in principle, model can't do it. (Backtracking reasoning, math facts, scientific identities.)
- **D — tool-blind:** the construct ranges over something that *isn't a presented sentence's representation*. Four flavors:
  - **D-i within-evaluation:** Lewis's similarity ranks *worlds inside one counterfactual's evaluation* — those worlds aren't sentences you fed in, so no per-sentence representation is them. RSA compares *between sentences*; it's the wrong kind of comparison. No probe of this class fixes it.
  - **D-ii universal negative:** de se *irreducibility* says "no proposition captures this." You can show de se differs from any *candidate* proposition, never that *none* works. Unfalsifiable in the strong form.
  - **D-iii proxy/bridge:** Fine's grounding is *non-causal* dependence; patching only sees *causal* dependence. You can argue from a non-causal domain (math) that asymmetric dependence there "looks like grounding" — but that's an *inference*, not a measurement. State the bridge; never claim you measured grounding.
  - **D-iv confound:** Stalnaker's "tie" and plain high next-token entropy produce the *same* dispersion number. The instrument can't separate them without a control that's unproven at 3B.

D-i and D-ii are *permanent* (logic/type limits). D-iii and D-iv *might* soften with better tools (sparse autoencoders, tighter controls) — but never with a bigger model.

**The escape move** (worth memorizing): to see a D-i construct, re-operationalize so the compared things become *presented sentences*. Lewis-vs-Pearl's real disagreement lives in *backtracking* cases (where "closest world" and "surgical `do()`" come apart); those give different *answers*, and answers are behaviorally visible. We turn a within-evaluation relation into a between-sentence behavioral contrast. (It's then capability-risky — hence "gate first, null is a real result.")

## 5. The cautionary tale, worked: how T1b fooled us

T1b was *supposed* to decide Lewis vs Pearl. The redesign measured representational geometry: cluster by causal topology (`M_graph`) → "Pearlian"; cluster by topic (`M_sim`) → "Lewisian." It looked rigorous (decorrelated matrices, partial-Mantel, layer bands, Holm correction). It is hollow. Two independent reasons:

1. **`M_sim` isn't Lewis.** Topic clustering is what *every* language model does — it's the biggest axis of variation in embeddings. Lewis's world-similarity is a within-evaluation ordering (D-i), not "do rain-sentences sit near rain-sentences." So "tracks `M_sim`" tells you the model is a language model. Nothing about Lewis.
2. **`M_graph` isn't uniquely Pearl.** A *Lewisian* model also has to represent causal structure — its similarity weighting prioritizes matching laws and facts near the antecedent, which encodes the topology. So topology-clustering is necessary for *both* theories; it discriminates neither.

The asymmetry-patching half fails too: it called directed influence "Pearl" and symmetric "Lewis," but Lewis *built* the asymmetry of counterfactual dependence into his similarity weighting deliberately. Both theories predict asymmetry. "Lewis = symmetric" isn't an idealization — it's wrong.

**Net:** any likely result would get stamped "Pearlian," but the stamp is unlicensed — a Lewisian model produces the same data. The honest reading of any outcome was "the model represents causal structure, asymmetrically," which neither philosopher disputes. That's the substitution trap (§1) in the wild: a hard question replaced by an easy, theory-neutral one wearing the hard question's name.

The lesson is the whole framework: **before building, run §4.** T1b is a case-B + case-D-i thread; the geometry proxy was the swap.

## 6. The framework applied — the 9-thread audit

Running §4 on each thread's *core* construct. "Visible?" = can a tool register the theories' disagreement; "Capable?" = could a ~3B model plausibly do it.

| Thread | The disagreement | Visible? | Capable @3B? | Verdict |
|---|---|---|---|---|
| **T2 Frege** | sense vs reference — coreferential terms diverge in *opaque* contexts, converge in transparent | **Yes (A)** — same terms, two contexts; the model is its own control | plausibly (use famous pairs) | **BUILD** |
| **T4 Quine** | which ontology (Platonist/nominalist/trope/4D) — each predicts a *different* similarity matrix over entities | **Yes (A)** — RSA against externally-anchored (BFO/DOLCE) matrices is exactly what RSA does | yes (just entity knowledge) | **BUILD** (strongest) |
| **T8 Kratzer** | epistemic vs circumstantial modal base — same "must," different cluster | **Yes (A)** | plausibly (modal flavor is common) | **BUILD** |
| **T6 Kripke** | rigid (name) vs flaccid (description) — name's representation stays invariant across modal contexts, description's shifts | **Yes (A)** — cross-context patching | moderate | **BUILD** |
| **T1b Lewis/Pearl** | truth-conditions diverge only in *backtracking* cases | B→A via the §4 escape | doubtful (hard reasoning) | **GATE-FIRST**, null ok |
| **T1c Lewis/Stalnaker** | unique closest world (no ties) vs ties allowed → dispersion on tie cases | partly — **D-iv** (dispersion confounds with entropy) | doubtful | **GATE-FIRST**, flag confound |
| **T2b hyperintensional** | distinctions finer than possible worlds (7 prime vs 7 not-composite) | Yes, but **C-gated** (needs reliable math) | doubtful @3B | **GATE-FIRST** |
| **T5 Fine grounding** | non-causal asymmetric dependence | **D-iii** — proxy only, bridge from math domain | subtle, needs all 3 conditions | **GATE-FIRST**, report bridge |
| **T7 contextualism** | "knows" shifts with stakes vs stable | Yes (A) | doubtful (stakes pragmatics @3B) | **GATE-FIRST** |
| **T1a** | (gate) counterfactual vs association | n/a — it's the gate | plausibly | **GATE only** |
| **T3** | where compositionality breaks | descriptive, no rival theories | — | **DESCRIPTIVE** |
| **T1d do-calculus** | not theory-A-vs-B (tests if reps respect identification) | — + capability wall | very doubtful | **SHELVE** |
| **T2c 2D semantics** | primary vs secondary intension | yes in principle, **C** steep | needs sci-identity + modal | **SHELVE @3B** |
| **T9 de se** | irreducible self-location | **D-ii** universal negative | steep | **SHELVE** (strong thesis) |

Read the table as the framework *working*: the BUILD threads are the ones where the disagreement is a difference in representations-of-sentences a tool registers; everything else is honestly capability-limited, tool-limited, or not a contest.

## 7. How this changes day-to-day work

- **Gate first.** Run the cheap behavioral gate before spending GPU on probing/patching. A failed gate ends the thread cheaply.
- **Null is a result.** For GATE-FIRST threads, "the model doesn't do this" is a finding, not a failure — especially on the deep ones (Bareinboim's causal questions, Fine's grounding).
- **Depth over breadth.** ~4 threads done defensibly beats 9 done thinly.
- **Name the bridge.** Where a verdict rests on an interpretation (grounding), say so in the summary; don't launder it into "we measured X."

If you internalize one thing: **every tool only sees representations of the sentences you presented — so a theory is testable only when its disagreement shows up *there*.** That single sentence generates the entire taxonomy.
