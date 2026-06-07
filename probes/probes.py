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
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr


# ── T1d identifiability split (single source of truth) ───────────────────────
# Pearl identifiability collapses the four T1d conditions into two classes. Both
# the L2 identification probe (run_identification_probe) and the L3 layer sweep
# in experiments/t1d/run_experiment.py import these so they test the *same*
# contrast — identified vs not-identifiable — rather than drifting apart.
#
# unconfounded_control is *identified*: plain X -> Y with no confounding, so
# P(Y | do(X)) = P(Y | X), trivially recovered with the empty adjustment set.
# confounded_not_adjustable is the only genuinely not-identifiable condition.
IDENTIFIED_T1D_LABELS: frozenset[str] = frozenset({
    "back_door_adjustable",
    "front_door_adjustable",
    "unconfounded_control",
})
NOT_IDENTIFIABLE_T1D_LABELS: frozenset[str] = frozenset({
    "confounded_not_adjustable",
})


# ── Shared cross-validation ─────────────────────────────────────────────────

def _adaptive_grouped_cv_accuracy(
    estimator: Any,
    features: np.ndarray,
    encoded_labels: np.ndarray,
    seed: int,
    groups: list[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Leakage-safe, adaptive cross-validated accuracy. Single source of fold logic
    for every probe in this project — run_linear_probe (L2) and run_surface_null
    (the surface baseline) both call it so they cannot drift apart.

    ─── Minimal-pair leakage (S4) ───────────────────────────────────────────────
    When groups is given, every row sharing a group id (the two sentences of a
    minimal pair) is forced onto the same side of every fold. Without this, the
    near-collinear sentence_a / sentence_b of a pair can split across train and
    test, letting the probe "predict" an item it has effectively memorised and
    inflating accuracy. StratifiedGroupKFold keeps class balance across folds
    while respecting groups; if its constraints cannot be met (a class confined
    to too few groups) we fall back to plain GroupKFold — grouping is the
    correctness-critical constraint, so stratification is the part we drop.

    ─── Adaptive fold count (L1) ────────────────────────────────────────────────
    n_splits adapts to the rarest stratum as min(5, limiting_count). In the
    grouped path the binding constraint is the number of distinct groups per
    class, not rows, because a whole group moves into a single fold. If fewer
    than two folds can be formed, cross-validation is undefined, so scores is
    returned as None and the caller reports accuracy as NaN rather than a
    misleading single-fold number.

    Args:
        estimator:      Any sklearn classifier (cloned per fold by cross_val_score).
        features:       np.ndarray (n_items, n_features).
        encoded_labels: np.ndarray (n_items,) of integer-encoded labels.
        seed:           Random seed for the ungrouped shuffle (grouped splitters
                        are deterministic and ignore it).
        groups:         Optional group id per row. When provided, splits are
                        grouped; when None, plain StratifiedKFold is used.

    Returns:
        Dict with:
          "scores":   np.ndarray | None — per-fold accuracy, None if CV skipped
          "n_splits": int               — folds actually run (0 if skipped)
          "grouped":  bool              — whether grouped splitting was used
          "note":     str | None        — explanation, present only when skipped
    """
    grouped = groups is not None

    if grouped:
        groups = np.asarray(groups)
        limiting_count = min(
            len(np.unique(groups[encoded_labels == class_index]))
            for class_index in np.unique(encoded_labels)
        )
    else:
        limiting_count = int(np.bincount(encoded_labels).min())

    n_splits = min(5, limiting_count)

    if n_splits < 2:
        note = (
            "Cross-validation skipped: smallest "
            + ("class-group" if grouped else "class")
            + f" count is {limiting_count} (< 2 folds possible). accuracy_mean is NaN."
        )
        return {"scores": None, "n_splits": 0, "grouped": grouped, "note": note}

    if grouped:
        try:
            cross_val_splitter = StratifiedGroupKFold(n_splits=n_splits)
            cross_val_scores = cross_val_score(
                estimator, features, encoded_labels, cv=cross_val_splitter, groups=groups
            )
        except ValueError:
            cross_val_splitter = GroupKFold(n_splits=n_splits)
            cross_val_scores = cross_val_score(
                estimator, features, encoded_labels, cv=cross_val_splitter, groups=groups
            )
    else:
        cross_val_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cross_val_scores = cross_val_score(estimator, features, encoded_labels, cv=cross_val_splitter)

    return {"scores": cross_val_scores, "n_splits": n_splits, "grouped": grouped, "note": None}


# ── Linear probe ──────────────────────────────────────────────────────────────

def run_linear_probe(
    activations: np.ndarray,
    labels: list[str],
    config: Any,
    pair_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Train a logistic regression probe on activations and return accuracy + weights.

    Uses k-fold cross-validation to get an unbiased accuracy estimate. The probe
    weights (one weight vector per class) are returned in the result — they can
    be analyzed geometrically to understand what direction in activation space
    encodes the distinction.

    ─── Leakage across minimal pairs (S4) ──────────────────────────────────────
    Stimuli are minimal pairs: two near-identical sentences differing in one
    spot, linked by a shared pair_id (the extractor emits them interleaved, so
    sentence_a and sentence_b of a pair sit on adjacent rows). Their activation
    vectors are therefore almost collinear. A plain StratifiedKFold(shuffle=True)
    on individual rows can put sentence_a in the train fold and sentence_b in the
    test fold; the probe then "predicts" a held-out item it has effectively
    already memorised, inflating accuracy. To prevent this, when pair_ids is
    given we split by pair_id *group* so both members of a pair always land on
    the same side of every fold. StratifiedGroupKFold keeps class balance across
    folds while respecting groups; if its constraints cannot be met (e.g. a class
    confined to too few groups) we fall back to plain GroupKFold.

    When pair_ids is None we keep the legacy StratifiedKFold behaviour so existing
    callers (e.g. run.py) that do not yet thread pair_ids through keep working.

    ─── Adaptive fold count (L1) ────────────────────────────────────────────────
    k=5 is hardcoded nowhere: n_splits adapts to the rarest class (or rarest
    group, for the grouped path) as min(5, smallest_count). If even two folds
    cannot be formed (smallest_count < 2) cross-validation is impossible, so we
    skip it and report accuracy_mean as NaN with an explanatory "note" field.
    n_folds always reflects what actually ran.

    Args:
        activations: np.ndarray of shape (n_items, hidden_dim). Each row is the
                     activation vector for one stimulus at one layer.
        labels:      List of string labels, length n_items. E.g. ["opaque", "transparent", ...]
        config:      ExperimentConfig. Used for experiment_id, thread_id, layer metadata.
        pair_ids:    Optional list of minimal-pair identifiers, length n_items.
                     When provided, cross-val splits are grouped by pair_id to
                     prevent train/test leakage between the two sentences of a
                     pair. When None, falls back to ungrouped StratifiedKFold.

    Returns:
        Dict with:
          "experiment_id": str
          "thread_id": str
          "accuracy_mean": float        — mean cross-val accuracy (NaN if cross-val skipped)
          "accuracy_std": float         — std across folds (NaN if cross-val skipped)
          "chance_baseline": float      — largest class fraction (null model)
          "weights": list[list[float]]  — probe weight vectors, shape [n_classes, hidden_dim]
          "labels_order": list[str]     — which class corresponds to which weight row
          "n_items": int
          "n_folds": int                — folds actually run (0 if cross-val skipped)
          "grouped_by_pair_id": bool    — whether the leakage-safe grouped split was used
          "note": str                   — present only when cross-val was skipped

    Note:
        Use sklearn.linear_model.LogisticRegression with max_iter=1000, C=1.0.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=config.seed)

    # Leakage-safe, adaptive cross-validation. pair_ids is the group key: both
    # sentences of a minimal pair share one id and so stay on the same side of
    # every fold. None → ungrouped legacy behaviour. See _adaptive_grouped_cv_accuracy.
    cross_validation = _adaptive_grouped_cv_accuracy(
        probe, activations, encoded_labels, config.seed, groups=pair_ids
    )
    grouped_by_pair_id = cross_validation["grouped"]
    skipped_note = cross_validation["note"]

    if cross_validation["scores"] is None:
        accuracy_mean = float("nan")
        accuracy_std = float("nan")
        n_folds_run = 0
    else:
        accuracy_mean = float(cross_validation["scores"].mean())
        accuracy_std = float(cross_validation["scores"].std())
        n_folds_run = cross_validation["n_splits"]

    # Fit once more on the full dataset to extract probe weights.
    # The accuracy above came from cross-val (held-out folds) — these weights
    # come from the full-data fit and are used for geometric analysis only,
    # not for evaluating accuracy.
    probe.fit(activations, encoded_labels)

    unique_label_counts = np.bincount(encoded_labels)
    chance_baseline = unique_label_counts.max() / len(encoded_labels)

    result: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "thread_id": config.thread_id,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "chance_baseline": float(chance_baseline),
        "weights": probe.coef_.tolist(),
        "labels_order": label_encoder.classes_.tolist(),
        "n_items": len(labels),
        "n_folds": n_folds_run,
        "grouped_by_pair_id": grouped_by_pair_id,
    }
    if skipped_note is not None:
        result["note"] = skipped_note
    return result


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
        "note": (
            "'significant' (p < 0.05) and 'exceeds_null_floor' (r > 95th pct of "
            "the null) are two restatements of the SAME permutation test, not two "
            "independent pieces of evidence — both read off the one null "
            "distribution computed here. Do not count them as corroborating each other."
        ),
    }


# ── KNIFE mutual information ──────────────────────────────────────────────────

def run_knife_mi(
    activations: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """
    Variational LOWER BOUND on the mutual information I(X; Y) between continuous
    activations X and discrete labels Y. This is NOT the KNIFE estimator.

    ─── What this actually computes ─────────────────────────────────────────────
    For any classifier q(Y | X), the Barber–Agakov bound gives
        I(X; Y) = H(Y) - H(Y | X) >= H(Y) - E[ -log q(Y | X) ].
    We take q to be a k-NN classifier and estimate the expected held-out negative
    log-likelihood (cross-entropy) with cross-validated predict_proba, so the
    classifier never scores a point it trained on. mi_nats = max(0, H(Y) - CE) is
    therefore a defensible lower bound: a high value is real evidence the label is
    decodable from the activation, a low value only means *this* classifier could
    not decode it — the true MI may be higher. Unlike a linear probe this does not
    assume linear separability, but it is bounded by the kNN's expressiveness, so
    it is a lower bound, not the full information.

    The earlier implementation reported an ad-hoc entropy decomposition of LOO
    accuracy as if it were MI; that number had no estimator-theoretic meaning and
    was removed. For the true KNIFE estimator (Pimentel et al. 2020) install the
    `knife` package and swap it in here.

    Args:
        activations: np.ndarray of shape (n_items, hidden_dim).
        labels:      List of string labels, length n_items.

    Returns:
        Dict with:
          "mi_nats": float     — MI lower bound in nats (NaN if too few items to CV)
          "mi_bits": float     — same in bits (mi_nats / log(2))
          "n_items": int
          "n_classes": int
          "n_folds": int       — CV folds used (0 if estimation was skipped)
          "estimator": str     — "knn-cv-variational-lower-bound"
          "note": str          — states this is a lower bound, not KNIFE
    """
    from sklearn.model_selection import cross_val_predict

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)
    n_items = len(labels)

    label_probs = np.bincount(encoded_labels) / n_items
    label_entropy_nats = float(-np.sum(label_probs * np.log(label_probs + 1e-10)))

    # Held-out cross-entropy needs at least two folds, and a fold per class member.
    smallest_class_count = int(np.bincount(encoded_labels).min())
    n_splits = min(5, smallest_class_count)

    lower_bound_note = (
        "Variational LOWER bound on I(X;Y) via held-out k-NN cross-entropy "
        "(Barber-Agakov), NOT the KNIFE estimator. A low value means this "
        "classifier could not decode the label, not that the information is absent."
    )

    if n_splits < 2:
        return {
            "mi_nats": float("nan"),
            "mi_bits": float("nan"),
            "n_items": n_items,
            "n_classes": int(n_classes),
            "n_folds": 0,
            "estimator": "knn-cv-variational-lower-bound",
            "note": (
                lower_bound_note
                + f" Skipped: smallest class count is {smallest_class_count} (< 2 folds)."
            ),
        }

    n_neighbors = min(5, smallest_class_count)
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    cross_validation_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    held_out_class_probabilities = cross_val_predict(
        knn_classifier, activations, encoded_labels,
        cv=cross_validation_splitter, method="predict_proba",
    )
    probability_of_true_class = held_out_class_probabilities[np.arange(n_items), encoded_labels]
    cross_entropy_nats = float(-np.mean(np.log(probability_of_true_class + 1e-10)))

    mi_nats = max(0.0, label_entropy_nats - cross_entropy_nats)
    mi_bits = mi_nats / np.log(2)

    return {
        "mi_nats": float(mi_nats),
        "mi_bits": float(mi_bits),
        "n_items": n_items,
        "n_classes": int(n_classes),
        "n_folds": n_splits,
        "estimator": "knn-cv-variational-lower-bound",
        "note": lower_bound_note,
    }


def run_identification_probe(
    activations: np.ndarray,
    labels: list[str],
    config: Any,
    pair_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Binary linear probe separating *identified* from *not-identifiable* causal effects.

    Used by T1d to test whether model representations distinguish causal effects
    whose magnitude can be recovered from observational data from those that
    cannot. The discriminating concept is Pearl's identifiability, not the mere
    presence of an adjustment formula. Collapses the four T1d conditions into two
    groups:
      identified (adjustable): back_door_adjustable | front_door_adjustable |
                               unconfounded_control
      not identifiable:        confounded_not_adjustable

    Why unconfounded_control is *identified*, not not_adjustable: it is plain
    direct causation X -> Y with no confounding. The interventional distribution
    P(Y | do(X)) simply equals the observational P(Y | X) — the effect is
    *trivially* identified (the empty set is a valid adjustment set). It is the
    easiest case of identification, so it belongs with the identified class. The
    only genuinely not-identifiable condition is confounded_not_adjustable, where
    a hidden confounder leaves no valid adjustment set and P(Y | do(X)) cannot be
    recovered from observation at all.

    The two-class grouping is fixed here — callers do not specify it. This
    enforces that identifiability is always tested as a binary distinction, not a
    four-way one.

    Caller note — class balance: this grouping is imbalanced, 3 conditions vs 1
    (identified vs not-identifiable). Chance baseline is therefore not 0.5, and
    callers should ensure roughly balanced sampling between the two collapsed
    classes (e.g. oversample confounded_not_adjustable) before reading
    accuracy_mean as evidence of separation. Sampling is not adjusted here.

    Args:
        activations: np.ndarray of shape (n_items, hidden_dim).
        labels: list of strings from T1d label set:
            "back_door_adjustable" | "front_door_adjustable" |
            "confounded_not_adjustable" | "unconfounded_control"
        config: ExperimentConfig. thread_id should be "t1d".
        pair_ids: Optional minimal-pair identifiers, length n_items, passed
                  through to run_linear_probe to group cross-val folds and
                  prevent train/test leakage between paired sentences.

    Returns:
        All fields from run_linear_probe, plus:
          "probe_type": "identification_binary"
          "adjustable_class": "adjustable"
          "not_adjustable_class": "not_adjustable"
    """
    # Identified effects: a valid adjustment set exists. For the two adjustable
    # conditions this is the back-door / front-door set; for unconfounded_control
    # it is the empty set (direct causation needs no adjustment at all). The set
    # lives at module scope so the L3 sweep patches the same contrast (V6 audit).
    binary_labels = [
        "adjustable" if label in IDENTIFIED_T1D_LABELS else "not_adjustable"
        for label in labels
    ]

    probe_result = run_linear_probe(activations, binary_labels, config, pair_ids=pair_ids)
    probe_result["probe_type"] = "identification_binary"
    probe_result["adjustable_class"] = "adjustable"
    probe_result["not_adjustable_class"] = "not_adjustable"
    return probe_result
