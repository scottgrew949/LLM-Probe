"""
stimuli/theoretical_matrices/t1b_matrices.py — Theoretical RDMs for T1b.

─── CONCEPT: Two decorrelated theories, two matrices ─────────────────────────
T1b asks whether the model's counterfactual geometry is organized like a causal
GRAPH (cluster by causal topology, regardless of topic) or like a holistic
SIMILARITY metric (cluster by topic, regardless of structure). We encode each
theory as a pairwise SIMILARITY matrix over the stimulus sentences:

  M_graph[i,j] = 1.0 if sentence i and j share causal structure, else 0.0
  M_sim[i,j]   = 1.0 if sentence i and j share domain/topic,    else 0.0

Each sentence's label is "{structure}|{domain}" (e.g. "chain|weather"). Because
the stimulus set crosses structure x domain in balanced cells, these two matrices
are decorrelated by construction — that is what lets partial RSA attribute the
model's geometry to one theory net of the other. [INVARIANT V23] If they are not
decorrelated (|corr| >= 0.2), the discriminator is void and we refuse to proceed.

Matrices are SIMILARITY (1 = close), matching probes.run_rsa's convention
(cosine_similarity, higher = closer), so a positive RSA correlation means
"clusters by this factor".
"""

from __future__ import annotations

import numpy as np

MATRIX_DECORRELATION_THRESHOLD: float = 0.2
"""[V23] Max permitted |corr(M_graph, M_sim)| over off-diagonal pairs."""


def _structure_of(label: str) -> str:
    return label.split("|", 1)[0]


def _domain_of(label: str) -> str:
    return label.split("|", 1)[1]


def build_graph_similarity_matrix(labels: list[str]) -> np.ndarray:
    """
    Pairwise causal-structure similarity. M[i,j] = 1.0 iff labels i, j share the
    structure prefix ("chain", "fork", "direct", "collider"); else 0.0. Diagonal 1.0.
    """
    structures = [_structure_of(label) for label in labels]
    n_items = len(labels)
    matrix = np.zeros((n_items, n_items), dtype=float)
    for i in range(n_items):
        for j in range(n_items):
            matrix[i, j] = 1.0 if structures[i] == structures[j] else 0.0
    return matrix


def build_domain_similarity_matrix(labels: list[str]) -> np.ndarray:
    """
    Pairwise domain/topic similarity. M[i,j] = 1.0 iff labels i, j share the
    domain suffix; else 0.0. Diagonal 1.0.
    """
    domains = [_domain_of(label) for label in labels]
    n_items = len(labels)
    matrix = np.zeros((n_items, n_items), dtype=float)
    for i in range(n_items):
        for j in range(n_items):
            matrix[i, j] = 1.0 if domains[i] == domains[j] else 0.0
    return matrix


def corr_between_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Pearson correlation of the two matrices' off-diagonal upper triangles."""
    n_items = matrix_a.shape[0]
    upper_row, upper_col = np.triu_indices(n_items, k=1)
    a_flat = matrix_a[upper_row, upper_col]
    b_flat = matrix_b[upper_row, upper_col]
    if a_flat.std() == 0 or b_flat.std() == 0:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def assert_matrices_decorrelated(
    matrix_graph: np.ndarray,
    matrix_sim: np.ndarray,
    threshold: float = MATRIX_DECORRELATION_THRESHOLD,
) -> float:
    """
    [INVARIANT V23] Raise if the two theoretical matrices are collinear.

    Returns the observed |corr| if it passes. If it fails, the partial RSA cannot
    separate the theories, so the experiment must not run.
    """
    observed_corr = corr_between_matrices(matrix_graph, matrix_sim)
    if abs(observed_corr) >= threshold:
        raise ValueError(
            f"V23: M_graph and M_sim are collinear (|corr|={abs(observed_corr):.3f} "
            f">= {threshold}). The factorial domain x structure cross is unbalanced "
            f"or degenerate; partial RSA cannot attribute geometry to one theory. "
            f"Rebalance the stimulus cells before running T1b."
        )
    return observed_corr
