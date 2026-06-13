"""tests/test_t10_gate.py — synthetic smoke for the T10 binding gate.

Runs run_behavioral_gate over the real T10 items on a FAKE model so the gate
wiring is V2-verified without a download. Does not assert accuracy, only that it
runs and returns the right structure.
"""

from __future__ import annotations

import torch

from stimuli.grammars.t10 import generate_behavioral_items
from stimuli.pipeline import run_behavioral_gate

VOCAB = 64


def _word_id(word: str) -> int:
    return sum(ord(c) for c in word) % (VOCAB - 1) + 1  # deterministic, in [1, VOCAB)


class _FakeModel:
    """Minimal HookedTransformer stand-in for forced-choice scoring."""

    def to_tokens(self, text: str, prepend_bos: bool = True):
        ids = ([0] if prepend_bos else []) + [_word_id(w) for w in text.split()]
        return torch.tensor([ids])

    def __call__(self, tokens, return_type: str = "logits"):
        seq = tokens.shape[1]
        rng = torch.Generator().manual_seed(int(tokens.sum().item()))
        return torch.randn(1, seq, VOCAB, generator=rng)


def test_gate_runs_and_returns_structure():
    items = generate_behavioral_items()
    assert len(items) > 0
    result = run_behavioral_gate(items, _FakeModel(), threshold=0.70)
    assert set(result) >= {"passed", "accuracy", "n_items", "details"}
    assert isinstance(result["passed"], bool)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n_items"] == len(items)
    assert len(result["details"]) == len(items)
