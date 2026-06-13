"""experiments/t4_pythia/run_experiment.py — T4 RSA pilot on Pythia 1.4B (CPU)."""

from types import SimpleNamespace

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

print("Loading", config.model_id, "(CPU)...", flush=True)
model = HookedTransformer.from_pretrained(config.model_id)
print("Loaded. n_layers =", model.cfg.n_layers, flush=True)

records = generate()
print("Running T4 RSA over", len(records), "entities...", flush=True)
result = run_t4_rsa(records, model, config, n_perms=1000)

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
