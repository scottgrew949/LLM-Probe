# Flagship: Compositionality and Systematicity (Fodor and Pylyshyn)

Design doc, 2026-06-13. New flagship thread (T10). Replaces the retired metaphysics-from-geometry threads (T4 ontology, T1 counterfactuals) as the lead.

## The question

Does the model have genuine recombinable constituent structure (a language of thought, Hypothesis A) or compositional-looking behavior from distributed pattern matching (Hypothesis B)?

## The core move

Behavior cannot decide this. Modern models pass behavioral systematicity by interpolation. So it must be tested in the mechanism. The operationalization: a genuinely structure-sensitive binding mechanism should be one causal mechanism whose strength is invariant to the content it binds and to how often that combination appeared in training. Invariance is the signature of structure. Frequency dependence is the signature of interpolation.

This is a clean case A on the adjudicability map (visible and discriminating), because the thing in dispute is the computation itself, which patching is built to see. It does not read metaphysics off a static geometry, which is what sank T4.

## Paradigm: attribute binding

Two-object scene, "the red cube and the blue sphere," then a retrieval probe, "What color is the sphere?". Correct answer requires binding blue to sphere. Precedent exists that models bind via abstract binding identifiers, which is the structured prediction. We push past existence of binding to its frequency and content invariance.

## Gate (neutral)

Model binds correctly on familiar two-object scenes above threshold. Both hypotheses predict pass. Precondition only.

## Discriminator (three tests, cleanest last)

1. **Separability and causal swap.** Locate the binding signal, patch it to rebind the attribute from cube to sphere, check the answer re-routes. Clean swap means a real manipulable constituent. Nothing isolable means holistic.
2. **Shared mechanism across role swap.** Same circuit for "red cube / blue sphere" as for "blue cube / red sphere", only bindings changed. One reusable mechanism is structure.
3. **Frequency invariance (the cut).** Vary combination frequency (common "red apple" to rare to novel) with individual word frequencies held fixed. Measure the causal binding strength across that range. Flat is structure. Negative slope is interpolation. Gentle slope is the honest middle.

## Stimuli

Programmatic two-object color-shape scenes plus retrieval question, binned by combination frequency. Critical control: individual word frequency matched across bins so only combination frequency varies. Combination frequency from corpus bigram counts, cross-checked against model surprisal. Minimal pairs swap the binding.

## Pre-registered verdict

- Structured: causal swap works, no significant frequency slope, fires on novel combinations.
- Pattern matching: binding effect only for frequent combinations, significant degradation with rarity.
- Graded (most likely): mechanism present but weakens with rarity; the slope is the finding.

## Failure modes (stated, not buried)

- **Superposition.** Binding signal may be smeared across overlapping directions; a null could be the tools failing, not absence of structure. Use multiple extraction methods, possibly an SAE, and flag any null as tool-limited.
- **Implementation objection.** Even flat invariance does not strictly refute pattern matching (the network could have learned to implement binding). Report the strongest honest claim, structure-sensitive content-independent processing, not "connectionism refuted."
- **Specificity.** Attribute binding may not generalize to relations. It is the first study, not the whole claim; scale to relational role swap only after it resolves.

## First build (minimal)

1. Stimulus grammar: two-object color-shape scenes with retrieval question, frequency-binned, word-frequency-controlled.
2. Behavioral gate on Llama 3.2 3B base.
3. Binding-signal probe plus causal swap.
4. Frequency-slope analysis.

Reuses existing extraction, probe, and patching code.
