"""experiments/t10_gate/run_experiment.py — T10 behavioral binding gate.

Tests whether the model actually binds (answers the bound color, not the typical
prior) on atypical scenes. This is the V8 precondition before any mechanistic
work. Default model is Llama 3.2 3B base; override with MODEL env var. Llama is
gated on HuggingFace, so set HF_TOKEN first. Set PILOT_MODEL=pythia for the open
Pythia 1.4B instead (no token, weaker, useful as a capability check).
"""

import os

import torch
from transformer_lens import HookedTransformer

from stimuli.grammars.t10 import generate_behavioral_items
from stimuli.pipeline import run_behavioral_gate

MODEL_ID = os.environ.get("MODEL", "meta-llama/Llama-3.2-3B")
if os.environ.get("PILOT_MODEL") == "pythia":
    MODEL_ID = "EleutherAI/pythia-1.4b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}. device={device}. model={MODEL_ID}", flush=True)
if device == "cpu":
    print("WARNING: no GPU. Switch the Colab runtime to T4 GPU.", flush=True)

dtype = torch.float16 if device == "cuda" else torch.float32
model = HookedTransformer.from_pretrained_no_processing(MODEL_ID, device=device, dtype=dtype)
print("Loaded.", flush=True)

items = generate_behavioral_items()
print(f"Running binding gate on {len(items)} atypical items (bound color vs prior)...", flush=True)
result = run_behavioral_gate(items, model, threshold=0.70)

print(f"\nPASSED={result['passed']}  accuracy={result['accuracy']:.3f}  n={result['n_items']}", flush=True)
for d in result["details"]:
    print(d, flush=True)
print("\nDONE", flush=True)
