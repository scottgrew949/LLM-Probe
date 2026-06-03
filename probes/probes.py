"""
probes/probes.py — Linear probes, RSA, mutual information, Mantel test.

─── CONCEPT: Linear probe (L2 analysis) ──────────────────────────────────────
A linear probe is a simple linear classifier (logistic regression) trained on
the internal activations of a model to predict a target label. The key constraint
is linearity: if a linear classifier can decode the distinction from the
activations at layer L, that means the distinction is *linearly separable* in
the model's representation space at that layer.

Why linearity matters: a sufficiently complex classifier (deep neural net) can
decode almost any signal from almost any representation — that tells you nothing
interesting. A *linear* classifier can only succeed if the information is
structured in a specific, geometrically simple way. This is strong evidence that
the model has encoded the distinction explicitly, not just in some tangled
non-linear way.

Philosophical parallel: asking whether a property is "conceptually basic" vs.
"defined in terms of other concepts." A linear probe tests whether the concept
is a first-class resident of the representational geometry.

─── CONCEPT: RSA — Representational Similarity Analysis ──────────────────────
RSA compares the *structure* of two representations, not their raw values.

Step 1: For each stimulus pair, compute pairwise similarity (cosine similarity
        or correlation) between all item representations → model similarity matrix.
Step 2: For the theoretical model, define what similarity *should* look like
        according to the theory → theory similarity matrix.
        (For T4, this comes from BFO/DOLCE ontology distances.)
Step 3: Correlate the two matrices. High correlation = model geometry matches
        the theory's predicted structure.

This is philosophically significant: you're not asking "does the model have a
representation of X?" but "does the *relational structure* of the model's
representations match the relational structure predicted by theory X?"

─── CONCEPT: Mantel test ─────────────────────────────────────────────────────
The Mantel test asks: is the correlation between the model matrix and the theory
matrix higher than chance? It permutes the rows/columns of one matrix 1000 times,
recomputes the correlation each time, and reports a p-value (fraction of permuted
correlations >= the real correlation).

p < 0.05 means the structural match is unlikely by chance.

─── CONCEPT: KNIFE mutual information ────────────────────────────────────────
KNIFE (Kolchinsky-style neural information estimator) estimates the mutual
information between activations and labels without assuming Gaussianity.
Standard linear probes report accuracy, but accuracy conflates geometry with
threshold calibration. MI gives a cleaner signal of how much information about
the label is present in the activation.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr


# ── Linear probe ──────────────────────────────────────────────────────────────

def run_linear_probe(
    activations: np.ndarray,
    labels: list[str],
    config: Any,
) -> dict[str, Any]:
    """
    Train a logistic regression probe on activations and return accuracy + weights.

    Uses stratified k-fold cross-validation (k=5) to get an unbiased accuracy
    estimate. The probe weights (one weight vector per class) are returned in
    the result — they can be analyzed geometrically to understand what direction
    in activation space encodes the distinction.

    Args:
        activations: np.ndarray of shape (n_items, hidden_dim). Each row is the
                     activation vector for one stimulus at one layer.
        labels:      List of string labels, length n_items. E.g. ["opaque", "transparent", ...]
        config:      ExperimentConfig. Used for experiment_id, thread_id, layer metadata.

    Returns:
        Dict with:
          "experiment_id": str
          "thread_id": str
          "accuracy_mean": float        — mean cross-val accuracy
          "accuracy_std": float         — std across folds
          "chance_baseline": float      — largest class fraction (null model)
          "weights": list[list[float]]  — probe weight vectors, shape [n_classes, hidden_dim]
          "labels_order": list[str]     — which class corresponds to which weight row
          "n_items": int
          "n_folds": int

    Note:
        Use sklearn.linear_model.LogisticRegression with max_iter=1000, C=1.0.
        Use sklearn.model_selection.StratifiedKFold for cross-validation.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=config.seed)
    cross_val_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    cross_val_scores = cross_val_score(probe, activations, encoded_labels, cv=cross_val_splitter)

    # Fit once more on the full dataset to extract probe weights.
    # The accuracy above came from cross-val (held-out folds) — these weights
    # come from the full-data fit and are used for geometric analysis only,
    # not for evaluating accuracy.
    probe.fit(activations, encoded_labels)

    unique_label_counts = np.bincount(encoded_labels)
    chance_baseline = unique_label_counts.max() / len(encoded_labels)

    return {
        "experiment_id": config.experiment_id,
        "thread_id": config.thread_id,
        "accuracy_mean": float(cross_val_scores.mean()),
        "accuracy_std": float(cross_val_scores.std()),
        "chance_baseline": float(chance_baseline),
        "weights": probe.coef_.tolist(),
        "labels_order": label_encoder.classes_.tolist(),
        "n_items": len(labels),
        "n_folds": 5,
    }


# ── RSA ───────────────────────────────────────────────────────────────────────

def run_rsa(
    activations: np.ndarray,
    theory_matrix: np.ndarray,
    config: Any,
) -> dict[str, Any]:
    """
    Compute RSA between model activation geometry and a theoretical similarity matrix.

    For T4 (Quinean ontological commitment): theory_matrix comes from BFO or DOLCE
    formal ontology distances. [INVARIANT V13] Raises if config.ontology_version
    or config.matrix_source is None — these must be documented before RSA runs.

    Args:
        activations:   np.ndarray of shape (n_items, hidden_dim).
        theory_matrix: np.ndarray of shape (n_items, n_items). Pairwise similarity
                       scores predicted by the theory. Must be symmetric.
        config:        ExperimentConfig. For T4, must have ontology_version and
                       matrix_source set.

    Returns:
        Dict with:
          "spearman_r": float         — Spearman correlation between flattened matrices
          "model_matrix": list        — model pairwise similarity matrix (for inspection)
          "theory_matrix": list       — theory matrix as stored (for traceability)
          "ontology_version": str|None
          "matrix_source": str|None
          "n_items": int

    Raises:
        ValueError: if config.thread_id == "t4" and ontology_version or
                    matrix_source is None (V13).
    """
    # V13: T4 requires ontology provenance
    if getattr(config, "thread_id", "").startswith("t4"):
        if config.ontology_version is None or config.matrix_source is None:
            raise ValueError(
                "T4 RSA requires ontology_version and matrix_source non-null (V13). "
                "Set these in ExperimentConfig before running RSA."
            )

    n_items = activations.shape[0]
    model_similarity_matrix = cosine_similarity(activations)

    upper_triangle_row_indices, upper_triangle_col_indices = np.triu_indices(n_items, k=1)
    model_flat_upper = model_similarity_matrix[upper_triangle_row_indices, upper_triangle_col_indices]
    theory_flat_upper = theory_matrix[upper_triangle_row_indices, upper_triangle_col_indices]

    correlation_result = spearmanr(model_flat_upper, theory_flat_upper)

    # Use index access [0] for scipy version compatibility.
    # scipy < 1.9 returns a named tuple with .correlation;
    # scipy >= 1.9 returns a SpearmanrResult with .statistic.
    # Index [0] works on both.
    return {
        "spearman_r": float(correlation_result[0]),
        "model_matrix": model_similarity_matrix.tolist(),
        "theory_matrix": theory_matrix.tolist(),
        "ontology_version": getattr(config, "ontology_version", None),
        "matrix_source": getattr(config, "matrix_source", None),
        "n_items": n_items,
    }


# ── Mantel test ───────────────────────────────────────────────────────────────

def run_mantel_test(
    model_matrix: np.ndarray,
    theory_matrix: np.ndarray,
    n_perms: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Permutation test for significance of the RSA correlation.

    Shuffles row/column order of model_matrix n_perms times, recomputes
    Spearman r with theory_matrix each time, and reports the empirical p-value.

    Args:
        model_matrix:  np.ndarray (n_items, n_items). Model pairwise similarity.
        theory_matrix: np.ndarray (n_items, n_items). Theory pairwise similarity.
        n_perms:       Number of permutations. Default 1000.
        seed:          Random seed for reproducibility.

    Returns:
        Dict with:
          "observed_r": float             — Spearman r between original matrices
          "p_value": float                — fraction of permuted r >= observed_r
          "n_perms": int
          "significant": bool             — p_value < 0.05
          "null_95th_percentile": float   — 95th percentile of permuted-r null distribution
          "exceeds_null_floor": bool      — observed_r > null_95th_percentile

        null_95th_percentile is the effect-size floor. Both significant and
        exceeds_null_floor must be True to claim best-fit framework (§T4).

    Note:
        Permuting row i of model_matrix also permutes column i (symmetric matrix
        permutation). This preserves the structure of the matrix while randomizing
        which stimulus corresponds to which row.
    """
    n_items = model_matrix.shape[0]
    upper_triangle_row_indices, upper_triangle_col_indices = np.triu_indices(n_items, k=1)

    model_flat_upper = model_matrix[upper_triangle_row_indices, upper_triangle_col_indices]
    theory_flat_upper = theory_matrix[upper_triangle_row_indices, upper_triangle_col_indices]
    observed_r = float(spearmanr(model_flat_upper, theory_flat_upper)[0])

    rng = np.random.default_rng(seed)
    permuted_correlations = []
    for _ in range(n_perms):
        permutation_order = rng.permutation(n_items)
        permuted_model_matrix = model_matrix[permutation_order][:, permutation_order]
        permuted_model_flat = permuted_model_matrix[upper_triangle_row_indices, upper_triangle_col_indices]
        permuted_correlation = float(spearmanr(permuted_model_flat, theory_flat_upper)[0])
        permuted_correlations.append(permuted_correlation)

    permuted_correlations = np.array(permuted_correlations)
    empirical_p_value = float((permuted_correlations >= observed_r).mean())
    null_95th_percentile = float(np.percentile(permuted_correlations, 95))

    return {
        "observed_r": observed_r,
        "p_value": empirical_p_value,
        "n_perms": n_perms,
        "significant": empirical_p_value < 0.05,
        "null_95th_percentile": null_95th_percentile,
        "exceeds_null_floor": observed_r > null_95th_percentile,
    }


# ── KNIFE mutual information ──────────────────────────────────────────────────

def run_knife_mi(
    activations: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """
    Estimate mutual information between activations and labels using KNIFE.

    KNIFE (from Pimentel et al. 2020) estimates I(X; Y) between continuous
    activations X and discrete labels Y using a variational bound and learned
    encoder. Unlike linear probes, MI is invariant to invertible transformations —
    it captures *all* information, not just linearly decodable information.

    Comparing run_linear_probe accuracy and run_knife_mi gives insight into
    whether information is stored linearly vs. non-linearly. If probe accuracy
    is low but MI is high, the distinction is encoded non-linearly.

    Args:
        activations: np.ndarray of shape (n_items, hidden_dim).
        labels:      List of string labels, length n_items.

    Returns:
        Dict with:
          "mi_nats": float     — estimated mutual information in nats
          "mi_bits": float     — same in bits (mi_nats / log(2))
          "n_items": int
          "n_classes": int

    Note:
        Requires the `knife` package or a local implementation.
        See Pimentel et al. 2020 "A Pareto-Optimal Compositional Language Emerges
        with Hobson's Choice" for the estimator.
        If knife is not installed, raise ImportError with installation instructions.
    """
    # The full KNIFE estimator (Pimentel et al. 2020) requires installing the
    # `knife` package. Until that's available, we use a k-NN leave-one-out proxy:
    # estimate MI from LOO accuracy via an entropy decomposition.
    # This is an approximation — it will not match the paper's numbers.
    # To use the real estimator, install knife and replace this block.
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    loo_splitter = LeaveOneOut()
    correct_predictions = 0
    for train_indices, test_index in loo_splitter.split(activations):
        knn_classifier.fit(activations[train_indices], encoded_labels[train_indices])
        predicted_label = knn_classifier.predict(activations[test_index])[0]
        if predicted_label == encoded_labels[test_index][0]:
            correct_predictions += 1
    loo_accuracy = correct_predictions / len(labels)

    # MI proxy: H(Y) - H(Y|X_hat), where H(Y) is label entropy
    # and H(Y|X_hat) is estimated from LOO accuracy
    label_probs = np.bincount(encoded_labels) / len(encoded_labels)
    label_entropy_nats = -np.sum(label_probs * np.log(label_probs + 1e-10))

    # Approximate conditional entropy from aggregate accuracy (rough bound)
    error_rate = 1.0 - loo_accuracy
    error_spread = error_rate / max(n_classes - 1, 1)  # distribute errors evenly over wrong classes
    conditional_entropy_nats = -(
        loo_accuracy * np.log(loo_accuracy + 1e-10)
        + error_rate * np.log(error_spread + 1e-10)
    )
    mi_nats = max(0.0, label_entropy_nats - conditional_entropy_nats)
    mi_bits = mi_nats / np.log(2)

    return {
        "mi_nats": float(mi_nats),
        "mi_bits": float(mi_bits),
        "n_items": len(labels),
        "n_classes": int(n_classes),
        "estimator": "knn-loo-proxy",
        "note": "Approximate — install knife package for Pimentel et al. 2020 estimator",
    }


def run_identification_probe(
    activations: np.ndarray,
    labels: list[str],
    config: Any,
) -> dict[str, Any]:
    """
    Binary linear probe separating adjustable from not-adjustable causal structures.

    Used by T1d to test whether model representations distinguish confounded
    structures with a valid adjustment set from those without one. Collapses
    the four T1d conditions into two groups:
      adjustable:     back_door_adjustable | front_door_adjustable
      not_adjustable: confounded_not_adjustable | unconfounded_control

    The two-class grouping is fixed here — callers do not specify it. This
    enforces that the identification criterion is always tested as a binary
    distinction, not a four-way one.

    Args:
        activations: np.ndarray of shape (n_items, hidden_dim).
        labels: list of strings from T1d label set:
            "back_door_adjustable" | "front_door_adjustable" |
            "confounded_not_adjustable" | "unconfounded_control"
        config: ExperimentConfig. thread_id should be "t1d".

    Returns:
        All fields from run_linear_probe, plus:
          "probe_type": "identification_binary"
          "adjustable_class": "adjustable"
          "not_adjustable_class": "not_adjustable"
    """
    adjustable_label_set = {"back_door_adjustable", "front_door_adjustable"}

    binary_labels = [
        "adjustable" if label in adjustable_label_set else "not_adjustable"
        for label in labels
    ]

    probe_result = run_linear_probe(activations, binary_labels, config)
    probe_result["probe_type"] = "identification_binary"
    probe_result["adjustable_class"] = "adjustable"
    probe_result["not_adjustable_class"] = "not_adjustable"
    return probe_result
