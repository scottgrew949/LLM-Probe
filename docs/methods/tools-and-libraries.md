# Tools & Libraries — Plain-English Guide

What every library, concept, and project function actually is and what it's *for*. Written for someone strong on the philosophy, newer to the ML/interpretability machinery.

---

## 1. The core mental model

A transformer language model processes a sentence as a stack of **layers**. At each layer, every **token** (roughly, word-piece) has a vector of numbers — its **activation**. The running vector that flows up through the layers is the **residual stream**; each layer reads it, computes, and writes back. Two sub-parts write to it at each layer: **attention** (mixes information across token positions) and the **MLP** (a per-token transformation).

Everything we do is: feed a sentence in, look at (or edit) these internal vectors, and reason about what they encode. The four "tools" from `reasoning-and-method.md §2` are all ways of doing that.

Key coordinates you'll see in configs:
- **layer** — depth (0 = embeddings, then 1..N). Layer 0 is just raw token/position lookup (no reasoning yet) — we exclude it from "where is the concept" claims.
- **token position** — which word's vector.
- **component** — *which* vector: `resid_post` (residual stream after the layer), `attn_out` (attention's contribution), `mlp_out` (MLP's contribution).

## 2. The libraries

| Library | What it is | What we use it for |
|---|---|---|
| **torch** (PyTorch) | the numerical engine — tensors (n-dim arrays) + autograd. Models *are* torch objects. | every model forward pass and tensor op |
| **transformer_lens** | an interpretability wrapper around transformers. Its `HookedTransformer` loads a named model and exposes **hooks** — points where you can *read or overwrite* any internal activation mid-computation. This is the library that makes mechanistic interp possible. | loading models (`HookedTransformer.from_pretrained("...")`), grabbing activations, patching |
| **scikit-learn (sklearn)** | classic (non-neural) ML toolkit. | the **linear probe** (logistic regression) + cross-validation + balanced-accuracy scoring |
| **scipy** | scientific computing — stats, distances, permutations. | distance matrices and the **Mantel** permutation test for RSA |
| **numpy** | fast array math; the substrate sklearn/scipy sit on. | matrix/vector math in probes and RSA |
| **wordfreq** | looks up how common a word is in a large corpus. | **frequency-matching** stimuli so a probe can't "win" by spotting rare words (the surface-null control) |
| **datasets** (HuggingFace) | dataset loading/hosting format. | publishing PhilBench |
| **sae_lens** | sparse-autoencoder toolkit (decomposes activations into interpretable features). | stretch goal only — mostly shelved |
| **matplotlib** | plotting. | layer-curves, RDM heatmaps |

A note on **hooks** (the heart of transformer_lens): a hook is a callback attached to a named internal site (e.g. "layer 8 residual stream"). On a forward pass, transformer_lens calls your hook with that activation. Read it → that's **extraction**. Return a modified version → that's **patching**. Same mechanism, two uses.

## 3. The statistical ideas you'll keep meeting

- **RDM (representational dissimilarity matrix):** take N stimuli, compute the pairwise distance between the model's representations → an N×N matrix describing the *geometry* of how the model arranges them.
- **RSA (representational similarity analysis):** correlate the model's RDM with a *hypothesized* RDM (what a theory predicts). High correlation = the model's geometry matches that theory's predicted structure.
- **Mantel test:** RSA's significance test. You can't use an ordinary p-value because matrix cells aren't independent. Mantel shuffles the stimulus labels many times (1000×), recomputes the correlation each time → a **null distribution** of "correlations you'd get by chance." Your real correlation must beat the **95th percentile** of that null *and* have p < 0.05. ("Beat a null distribution, not a round number" — why we don't use magic thresholds like 0.70.)
- **Balanced accuracy:** accuracy corrected for class imbalance — if 90% of items are class A, a dumb "always A" classifier scores 0.90 on plain accuracy but 0.50 balanced. We use balanced + a majority-class floor so a probe can't look good by exploiting imbalance.
- **KL divergence:** measures how much one probability distribution differs from another. In patching, we compare the model's output distribution before vs after a patch — a big KL means the patched activation mattered.
- **Selectivity / specificity control:** a real effect must be *specific*. We compare a targeted patch against a control (mean-ablation, or a norm-matched random patch). If random edits break things just as much, the targeted result is meaningless.

## 4. The project's own functions (where the above lives)

| Function (file) | What it does |
|---|---|
| `extract_activations` (`extraction/extractor.py`) | runs the model over the validated stimuli, pulls the activation at each (layer, position, component), returns them for probing/RSA. Enforces the stimulus-hash gate. |
| `run_behavioral_gate` (`stimuli/pipeline.py`) | the cheap gate: scores forced-choice items by summing log-probabilities over the answer tokens; returns pass/accuracy. No internals — pure behavior. |
| `run_linear_probe` (`probes/probes.py`) | trains a logistic-regression probe per layer (5-fold CV, standardized, balanced accuracy + optional control-task selectivity). Finds *where* a distinction is decodable. |
| `run_rsa` (`probes/probes.py`) | builds the model RDM and correlates it with a theory matrix (Spearman). |
| `run_mantel_test` (`probes/probes.py`) | the permutation significance test above; returns observed r, p, and the null-95th-percentile floor. |
| `patch_activation` / `run_layer_sweep` (`interventions/interventions.py`) | the causal test: paste one run's activation into another, measure KL on the output. The sweep does it across layers to find the causal peak. |
| `assert_specificity_valid` (`interventions/interventions.py`) | enforces the specificity control — raises if a targeted patch isn't meaningfully stronger than the mean-ablation baseline. |
| `run_surface_null` (`experiments/run.py`) | the baseline run *first*: how much of any separation is explainable by word frequency + length alone (no deep computation). If the null explains it, no philosophical claim is licensed. |
| `model_id_for_thread` (`experiments/config.py`) | maps a run id (e.g. `t4_llama`) to its HuggingFace model — the single place model identity lives (no hardcoding). |

## 5. For T4 specifically

T4 (ontology) leans on the **RSA + Mantel** path, not patching. The pipeline: `extract_activations` over the entity stimuli → build the model RDM → `run_rsa` against each *theory* matrix (Platonist / nominalist / trope / 4D) → `run_mantel_test` for significance → the theory whose matrix best matches (above the null floor) is the model's ontological signature. The whole validity question is whether those four theory matrices are (a) honestly derived from a published ontology and (b) different enough from each other to tell apart — which is exactly the #14/#15 design below.
