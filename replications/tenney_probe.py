"""
replications/tenney_probe.py — Tenney et al. 2019 edge probing replication.

Replicates the core methodology from "BERT Rediscovers the Classical NLP Pipeline"
(Tenney, Das, Pavlick 2019) using GPT-2 medium + TransformerLens.

Task: single-token POS classification (noun vs verb) — simplest edge probing task.

Two probe variants run side-by-side:

  Per-layer probe (PoL-Probe method)
    Train logistic regression independently at each layer.
    Answers: which single layer is the peak location for this distinction?

  Scalar mixing probe (Tenney method)
    Learn a weighted combination across all layers jointly.
    Answers: which layers contribute most when combined optimally?

The contrast is the lesson: per-layer isolates a peak; scalar mixing spreads
credit across layers. PoL-Probe uses per-layer because we want to locate
where a distinction lives, not just whether it exists.

Run with:
    .venv/bin/python replications/tenney_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from transformer_lens import HookedTransformer


# ── Dataset ───────────────────────────────────────────────────────────────────
# (sentence, target_word, label)
# Each pair shares a sentence — one example tags the noun, one tags the verb.
# This controls for sentence-level confounds: any probe difference is due to
# the word being probed, not the sentence itself.

EXAMPLES: list[tuple[str, str, str]] = [
    ("The dog barked loudly at the stranger.",       "dog",       "noun"),
    ("The dog barked loudly at the stranger.",       "barked",    "verb"),
    ("A small cat curled up beside the warm fire.",  "cat",       "noun"),
    ("A small cat curled up beside the warm fire.",  "curled",    "verb"),
    ("Her old book fell from the highest shelf.",    "book",      "noun"),
    ("The heavy book dropped onto the cold floor.",  "dropped",   "verb"),
    ("The wide river cuts through the valley.",      "river",     "noun"),
    ("The wide river floods every spring.",          "floods",    "verb"),
    ("An eagle drifted far above the pale clouds.",  "eagle",     "noun"),
    ("An eagle circled slowly above the mountains.", "circled",   "verb"),
    ("The ancient city sprawls along the coast.",    "city",      "noun"),
    ("The whole crowd cheered when play ended.",     "cheered",   "verb"),
    ("His closest friend arrived late that evening.","friend",    "noun"),
    ("She waited alone on the empty platform.",      "waited",    "verb"),
    ("A flat stone skipped across the still water.", "stone",     "noun"),
    ("The tired runner collapsed at the finish.",    "collapsed", "verb"),
    ("The small child laughed and ran away.",        "child",     "noun"),
    ("A young child wept quietly in the corner.",   "wept",      "verb"),
    ("The old engine groaned and then stopped.",     "engine",    "noun"),
    ("The old machine failed on its first test.",    "failed",    "verb"),
]


# ── Token position finder ─────────────────────────────────────────────────────

def find_token_position(model: HookedTransformer, sentence: str, target_word: str) -> int:
    """
    Return the index of the first token corresponding to target_word.

    Handles two cases:
      Single-token word: "dog" → exact match on decoded token.
      Multi-token word:  "barked" → "bark"+"ed"; returns position of "bark"
                         (first token of the span, which is Tenney's span
                         representation approach).

    Strips GPT-2's Ġ space-prefix before comparing.
    """
    token_ids = model.to_tokens(sentence)[0]
    decoded = [
        model.to_string(tid.item()).replace("Ġ", "").replace("Ċ", "").strip().lower()
        for tid in token_ids
    ]

    # Pass 1: exact match
    for pos, clean in enumerate(decoded):
        if clean == target_word.lower():
            return pos

    # Pass 2: target word starts with this token's text (first subword of span)
    for pos, clean in enumerate(decoded):
        if len(clean) >= 2 and target_word.lower().startswith(clean):
            return pos

    raise ValueError(
        f"Target '{target_word}' not found in '{sentence}'.\n"
        f"Decoded tokens: {decoded}"
    )


# ── Activation extraction ─────────────────────────────────────────────────────

def extract_all_layers(
    model: HookedTransformer,
    examples: list[tuple[str, str, str]],
) -> tuple[np.ndarray, list[str]]:
    """
    Extract resid_post activations at the target token position for every
    layer and every example.

    Returns:
        activations: np.ndarray of shape (n_examples, n_layers, hidden_dim)
        labels:      list of label strings, length n_examples
    """
    n_layers = model.cfg.n_layers
    hidden_dim = model.cfg.d_model
    n_examples = len(examples)

    activations = np.zeros((n_examples, n_layers, hidden_dim), dtype=np.float32)
    labels = []

    print(f"Extracting activations: {n_examples} examples × {n_layers} layers × {hidden_dim} dims")

    for ex_idx, (sentence, target_word, label) in enumerate(examples):
        token_pos = find_token_position(model, sentence, target_word)
        labels.append(label)

        # Capture resid_post at every layer in one forward pass
        captured: dict[int, np.ndarray] = {}

        hooks = []
        for layer_idx in range(n_layers):
            def make_hook(lidx: int):
                def hook_fn(val, hook):
                    captured[lidx] = val[0, token_pos, :].detach().cpu().numpy()
                    return val
                return hook_fn
            hooks.append((f"blocks.{layer_idx}.hook_resid_post", make_hook(layer_idx)))

        with torch.no_grad():
            model.run_with_hooks(sentence, fwd_hooks=hooks)

        for layer_idx in range(n_layers):
            activations[ex_idx, layer_idx, :] = captured[layer_idx]

        print(f"  [{ex_idx + 1:02d}/{n_examples}] '{target_word}' ({label}) @ token pos {token_pos}")

    return activations, labels


# ── Per-layer probe (PoL-Probe method) ────────────────────────────────────────

def run_per_layer_probes(
    activations: np.ndarray,
    labels: list[str],
) -> dict[int, float]:
    """
    Train a logistic regression probe independently at each layer.
    Uses leave-one-out CV to get unbiased accuracy estimate.
    (LOO is appropriate here because n=20 is too small for k-fold.)

    Returns dict mapping layer_index → LOO accuracy.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    n_layers = activations.shape[1]

    accuracy_by_layer: dict[int, float] = {}
    for layer_idx in range(n_layers):
        X = activations[:, layer_idx, :]
        probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        loo_scores = cross_val_score(probe, X, y, cv=LeaveOneOut())
        accuracy_by_layer[layer_idx] = float(loo_scores.mean())

    return accuracy_by_layer


# ── Scalar mixing probe (Tenney method) ───────────────────────────────────────

class ScalarMixingProbe(nn.Module):
    """
    Tenney's scalar mixing: learn a weighted combination of layer representations,
    then apply a linear classifier on the mixed vector.

    h_mixed = γ · Σ_l softmax(a)_l · h_l

    where a (mixing weights) and γ (scale) are learned jointly with the classifier.
    """
    def __init__(self, n_layers: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.mixing_weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, layer_activations: torch.Tensor) -> torch.Tensor:
        # layer_activations: (batch, n_layers, hidden_dim)
        weights = torch.softmax(self.mixing_weights, dim=0)          # (n_layers,)
        mixed = self.gamma * (layer_activations * weights[None, :, None]).sum(dim=1)
        return self.classifier(mixed)


def run_scalar_mixing_probe(
    activations: np.ndarray,
    labels: list[str],
    n_epochs: int = 300,
    lr: float = 0.01,
) -> tuple[float, np.ndarray]:
    """
    Train the scalar mixing probe on the full dataset and return:
      - final accuracy
      - learned per-layer mixing weights (softmaxed, sums to 1)

    Note: with n=20, train/test split would give noisy estimates.
    Training on the full set and reporting train accuracy is appropriate
    here since this is a replication/learning exercise, not a real experiment.
    """
    label_encoder = LabelEncoder()
    y_np = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)

    X = torch.tensor(activations, dtype=torch.float32)   # (20, 24, 1024)
    y = torch.tensor(y_np, dtype=torch.long)

    n_layers, hidden_dim = activations.shape[1], activations.shape[2]

    model = ScalarMixingProbe(n_layers, hidden_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=-1)
        accuracy = float((preds == y).float().mean().item())
        learned_weights = torch.softmax(model.mixing_weights, dim=0).numpy()

    return accuracy, learned_weights


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Tenney edge probing replication — noun vs verb")
    print("=" * 60)

    print("\nLoading GPT-2 medium...")
    model = HookedTransformer.from_pretrained("gpt2-medium")
    model.eval()
    print(f"Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}\n")

    # Extract activations
    activations, labels = extract_all_layers(model, EXAMPLES)
    print()

    # Per-layer probes
    print("Running per-layer probes (LOO CV)...")
    per_layer_acc = run_per_layer_probes(activations, labels)

    peak_layer = max(per_layer_acc, key=lambda l: per_layer_acc[l])
    chance = 0.5  # balanced classes

    print("\nPer-layer accuracy (PoL-Probe method):")
    print(f"  {'Layer':<8} {'Accuracy':>10}")
    print(f"  {'-'*20}")
    for layer_idx in range(model.cfg.n_layers):
        acc = per_layer_acc[layer_idx]
        marker = " ← peak" if layer_idx == peak_layer else ""
        print(f"  {layer_idx:<8} {acc:>10.3f}{marker}")
    print(f"\n  Chance baseline: {chance:.3f}")
    print(f"  Peak layer: {peak_layer}  (accuracy={per_layer_acc[peak_layer]:.3f})")

    # Scalar mixing probe
    print("\nTraining scalar mixing probe (Tenney method)...")
    scalar_acc, mixing_weights = run_scalar_mixing_probe(activations, labels)

    print(f"\nScalar mixing accuracy (train): {scalar_acc:.3f}")
    print("\nLearned mixing weights (higher = that layer contributed more):")
    print(f"  {'Layer':<8} {'Weight':>10}")
    print(f"  {'-'*20}")
    sorted_layers = sorted(range(len(mixing_weights)), key=lambda l: mixing_weights[l], reverse=True)
    for rank, layer_idx in enumerate(sorted_layers[:8]):  # top 8
        print(f"  {layer_idx:<8} {mixing_weights[layer_idx]:>10.4f}")
    print(f"  ... (showing top 8 of {model.cfg.n_layers})")

    # Comparison
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Per-layer peak:      layer {peak_layer}  (acc={per_layer_acc[peak_layer]:.3f})")
    print(f"  Scalar mixing acc:   {scalar_acc:.3f} (train)")
    print(f"  Top mixing layer:    layer {sorted_layers[0]}  (weight={mixing_weights[sorted_layers[0]]:.4f})")
    print()
    print("Takeaway for PoL-Probe:")
    if peak_layer == sorted_layers[0]:
        print("  Per-layer peak and scalar mixing top layer agree →")
        print("  this distinction is strongly localized to one layer.")
    else:
        print(f"  Per-layer peak (layer {peak_layer}) ≠ scalar mixing top (layer {sorted_layers[0]}) →")
        print("  distinction is spread across layers; mixing spreads credit.")
    print()
    print("PoL-Probe uses per-layer probing because we want to *locate*")
    print("where the distinction lives — not just prove it exists.")


if __name__ == "__main__":
    main()
