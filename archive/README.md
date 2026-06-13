# Archive

Not dead code — superseded or shelved work, kept for revival and for the methodological record (the *why* of shelving is part of the contribution). See `docs/superpowers/specs/2026-06-13-rescope-decision.md`.

## experiments/
- `t1a_gpt2/`, `t1b_gpt2/`, `t1c_gpt2/`, `t1d_gpt2/` — GPT-2 runners. GPT-2 primacy dropped (free ceiling is now Llama 3.2 3B; GPT-2 failed behavioral gates on T1a/T1b/T2).
- `t1d_pythia/`, `t2c/` — shelved-thread runners.

## tests/
- `test_t1d_grammar.py`, `test_t2c_grammar.py` — dedicated tests for shelved threads. Import paths still resolve (the `t1d`/`t2c` grammars remain in `stimuli/grammars/`, dormant, because `tests/test_phase0.py` is still coupled to them — removing them is a refactor, deferred).

## Shelve reasons
- **T1d** — not a theory-vs-theory discriminator (tests whether reps respect do-calculus identification, not theory A vs B); capability wall at 3B.
- **T2c** — capability steep at 3B (reliable scientific-identity + modal competence).
- **T9** (no files yet) — de se *irreducibility* is construct-invisible to probing/patching (a universal negative); + capability.

## Revival
Move the dir/file back and re-point any config. Shared-module references (`config.py`, `run.py`, `probes.py`, `t1_experiments.py`) for `t1d`/`t2c` were left in place, so revival is low-friction.
