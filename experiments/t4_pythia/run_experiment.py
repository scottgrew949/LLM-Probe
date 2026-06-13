"""experiments/t4_pythia/run_experiment.py — T4 RSA pilot on Pythia 1.4B.

Runs on GPU automatically when Colab has one. Set PILOT=0 in the environment for
the full run (all layers, 1000 permutations); the default is a fast pilot.
"""

import os
from types import SimpleNamespace

import torch
from transformer_lens import HookedTransformer

from core.io import save_result
from experiments.t4_rsa import run_t4_rsa
from stimuli.grammars.t4 import generate

config = SimpleNamespace(
    thread_id="t4_pythia",
    model_id="EleutherAI/pythia-1.4b",
    ontology_version="DUL 3.27 (DBpedia snapshot 2021-02-22)",
    matrix_source="stimuli/theoretical_matrices/t4_matrices.py",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}. Using device: {device}.", flush=True)
if device == "cpu":
    print("WARNING: no GPU. Switch the Colab runtime to T4 GPU, or this is very slow.", flush=True)

print("Loading", config.model_id, "...", flush=True)
# float16 on GPU halves memory and avoids the fp32 double-load OOM. CPU stays
# float32 (float16 is poorly supported there). We use from_pretrained_no_processing
# because the default from_pretrained runs LayerNorm folding and centering, which is
# numerically unstable in float16 and memory heavy. We only read resid_post for RSA,
# so the unprocessed weights are fine and the hook names are identical.
load_dtype = torch.float16 if device == "cuda" else torch.float32
model = HookedTransformer.from_pretrained_no_processing(
    config.model_id, device=device, dtype=load_dtype
)
print("Loaded. n_layers =", model.cfg.n_layers, flush=True)

pilot = os.environ.get("PILOT", "1") == "1"
if pilot:
    layers = list(range(1, model.cfg.n_layers, 4))  # sampled layers
    n_perms = 200
    print(f"PILOT mode: layers {layers}, {n_perms} permutations. Set PILOT=0 for the full run.", flush=True)
else:
    layers = None  # all layers 1..n-1
    n_perms = 1000

records = generate()
print(f"Running T4 RSA over {len(records)} entities...", flush=True)
result = run_t4_rsa(records, model, config, layers=layers, n_perms=n_perms)

save_result(result, "experiments/t4_pythia/results/summary.json")

print("\n=== best-fit theory per layer (by observed_r) ===", flush=True)
for layer in result["layers"]:
    row = result["per_layer"][layer]
    ranked = sorted(row, key=lambda t: row[t]["observed_r"], reverse=True)
    best = ranked[0]
    runner_up = ranked[1]
    e = row[best]
    margin = e["observed_r"] - row[runner_up]["observed_r"]
    flag = "SIG" if (e["significant"] and e["exceeds_null_floor"]) else "ns"
    print(f"L{layer:2d}  {best:11s} r={e['observed_r']:+.3f} {flag}  "
          f"(2nd {runner_up} +{margin:.3f})", flush=True)

print("\ninter-matrix corr:", result["inter_matrix_corr"], flush=True)
print("\nDONE", flush=True)
