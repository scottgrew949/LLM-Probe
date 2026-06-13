# Definition of Done

Quality gate for code in this repo. Goal: first-pass code is correct + efficient, not "just compiles." Past failure was shipping code that ran, then admitting on review it was crap — costly rewrite. Front-load the scrutiny to design.

## Routing rule

```
trivial/plumbing  → implement + adversarial test
risky/algorithmic → design → critique → implement (1 turn) + adversarial test
```

**Risky** (full loop): a silent bug would corrupt a finding — statistics, statistical inference, discriminator math, RDM construction, patch / causal direction, gate thresholds — OR a new algorithm OR multi-file change.

**Trivial** (skip to implement + test): IO wrappers, wiring, config, file moves, mechanical renames.

## The 3 parts (risky units only)

Written terse (fragment bullets), in a single turn, before any code:

1. **Design**
   - contract: inputs / outputs / shapes / invariants
   - algorithm + why correct
   - complexity + compute reuse (single model load, cached activations, vectorized — compute is the binding constraint: Colab T4)
   - validity holes to dodge: confound, leakage, train-test contamination, layer-selection double-dipping, class balance, majority-floor vs balanced accuracy, magic numbers
   - failure modes + boundary behavior

2. **Critique** — adversarial pass on the *design*: short findings list attacking correctness / efficiency / validity / edge cases. Resolve or flag each.

3. **Implement with fixes** — code the revised design.

## Exit gate (all code)

- adversarial tests green: boundary / empty / degenerate / adversarial inputs — NOT happy-path only
- fail loud at boundaries: validate shapes / label balance / non-empty / finite; no bare `except`; no silent fallback masking failure
- no magic numbers: every threshold justified or calibrated
- **verification tier** stated on every done-claim:

| tier | meaning | done-floor |
|------|---------|------------|
| `V0` | imports / compiles | ❌ never done |
| `V1` | pure-logic unit tests pass (no model) | stats / grammar / RDM / gate logic |
| `V2` | synthetic smoke — real code path, tiny random model / fake tensors | model-dependent code locally |
| `V3` | real-model Colab run, outputs sane | only true "verified" for model code |

Tag artifacts, e.g. `[V2 — unverified on real model]`. "Done" is forbidden below the floor.

## Cost note

This is not 2x tokens. The full loop runs on ~1/3 of units; the design block is terse (~10–15 lines). The old bounce-and-rewrite cycle already cost ~2x via rework — re-implementing a bounced unit + re-running compute. A terse up-front design is cheaper than that. Net usage drops. The wasteful failure mode is mis-gating: running the full loop on trivial code.
