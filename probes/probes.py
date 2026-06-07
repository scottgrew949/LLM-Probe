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
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import spearmanr


# ── T1d identifiability split (single source of truth) ───────────────────────
# Pearl identifiability collapses the four T1d conditions into two classes. Both
# the L2 identification probe (run_identification_probe) and the L3 layer sweep
# in experiments/t1d_gpt2/run_experiment.py import these so they test the *same*
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

    Two scores are returned per fold: raw accuracy and balanced accuracy (mean of
    per-class recall). Balanced accuracy is the honest number for imbalanced label
    sets — raw accuracy of a constant majority-class classifier equals the majority
    fraction, which can sit above a naive threshold, so any decision rule on
    imbalanced data must read balanced accuracy (or accuracy vs the majority floor).

    ─── Stratification transparency ─────────────────────────────────────────────
    When the StratifiedGroupKFold constraints cannot be met we fall back to plain
    GroupKFold and set "stratified": False, so the caller knows class balance
    across folds was not guaranteed (per-fold class counts may be uneven).

    Returns:
        Dict with:
          "accuracy_scores":          np.ndarray | None — per-fold raw accuracy
          "balanced_accuracy_scores": np.ndarray | None — per-fold balanced accuracy
          "n_splits":   int   — folds actually run (0 if skipped)
          "grouped":    bool  — whether grouped splitting was used
          "stratified": bool  — whether stratification held (False on GroupKFold fallback)
          "note":       str | None — explanation, present only when skipped
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
        return {
            "accuracy_scores": None, "balanced_accuracy_scores": None,
            "n_splits": 0, "grouped": grouped, "stratified": False, "note": note,
        }

    scoring = ["accuracy", "balanced_accuracy"]
    stratified = True
    if grouped:
        try:
            cross_val_splitter = StratifiedGroupKFold(n_splits=n_splits)
            cross_val_result = cross_validate(
                estimator, features, encoded_labels, cv=cross_val_splitter,
                groups=groups, scoring=scoring,
            )
        except ValueError:
            stratified = False
            cross_val_splitter = GroupKFold(n_splits=n_splits)
            cross_val_result = cross_validate(
                estimator, features, encoded_labels, cv=cross_val_splitter,
                groups=groups, scoring=scoring,
            )
    else:
        cross_val_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cross_val_result = cross_validate(
            estimator, features, encoded_labels, cv=cross_val_splitter, scoring=scoring,
        )

    return {
        "accuracy_scores": cross_val_result["test_accuracy"],
        "balanced_accuracy_scores": cross_val_result["test_balanced_accuracy"],
        "n_splits": n_splits, "grouped": grouped, "stratified": stratified, "note": None,
    }


# ── Linear probe ──────────────────────────────────────────────────────────────

def run_linear_probe(
    activations: np.ndarray,
    labels: list[str],
    config: Any,
    pair_ids: list[str] | None = None,
    selectivity_seed: int | None = None,
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

    # StandardScaler in a pipeline: resid_post scale varies across layers, so
    # comparing raw cross-layer probe accuracy conflates information content with
    # activation magnitude. Scaling is fit inside each CV fold (the pipeline is
    # cloned per fold by cross_validate), so there is no train→test leakage.
    def make_probe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=config.seed),
        )

    # Leakage-safe, adaptive cross-validation. pair_ids is the group key: both
    # sentences of a minimal pair share one id and so stay on the same side of
    # every fold. None → ungrouped legacy behaviour. See _adaptive_grouped_cv_accuracy.
    cross_validation = _adaptive_grouped_cv_accuracy(
        make_probe(), activations, encoded_labels, config.seed, groups=pair_ids
    )
    grouped_by_pair_id = cross_validation["grouped"]
    skipped_note = cross_validation["note"]

    if cross_validation["accuracy_scores"] is None:
        accuracy_mean = float("nan")
        accuracy_std = float("nan")
        balanced_accuracy_mean = float("nan")
        n_folds_run = 0
    else:
        accuracy_mean = float(cross_validation["accuracy_scores"].mean())
        accuracy_std = float(cross_validation["accuracy_scores"].std())
        balanced_accuracy_mean = float(cross_validation["balanced_accuracy_scores"].mean())
        n_folds_run = cross_validation["n_splits"]

    # Fit once more on the full dataset to extract probe weights (standardized
    # space). Used for geometric analysis only, not accuracy.
    full_pipeline = make_probe()
    full_pipeline.fit(activations, encoded_labels)
    probe_weights = full_pipeline.named_steps["logisticregression"].coef_

    unique_label_counts = np.bincount(encoded_labels)
    chance_baseline = unique_label_counts.max() / len(encoded_labels)

    result: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "thread_id": config.thread_id,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "balanced_accuracy_mean": balanced_accuracy_mean,
        "chance_baseline": float(chance_baseline),
        "weights": probe_weights.tolist(),
        "labels_order": label_encoder.classes_.tolist(),
        "n_items": len(labels),
        "n_folds": n_folds_run,
        "grouped_by_pair_id": grouped_by_pair_id,
        "stratified": cross_validation["stratified"],
    }
    if skipped_note is not None:
        result["note"] = skipped_note

    # Control task (Hewitt & Liang): re-run on randomly permuted labels. The probe
    # has capacity to fit some structure even from noise; selectivity = real minus
    # control accuracy bounds how much of the real accuracy reflects an actual
    # encoded distinction rather than probe expressiveness. Only computed on request.
    if selectivity_seed is not None and cross_validation["accuracy_scores"] is not None:
        shuffled_labels = np.random.default_rng(selectivity_seed).permutation(encoded_labels)
        control_cv = _adaptive_grouped_cv_accuracy(
            make_probe(), activations, shuffled_labels, config.seed, groups=pair_ids
        )
        if control_cv["accuracy_scores"] is not None:
            control_accuracy = float(control_cv["accuracy_scores"].mean())
            result["control_task_accuracy"] = control_accuracy
            result["selectivity"] = accuracy_mean - control_accuracy

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


# ── Calibrated probe null (replaces magic accuracy thresholds) ───────────────

def probe_beats_null(
    activations: np.ndarray,
    labels: list[str],
    config: Any,
    pair_ids: list[str] | None = None,
    n_shuffles: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Decide whether a probe encodes a real distinction by comparing its balanced
    accuracy against a CALIBRATED null distribution — not a hand-set 0.70 cutoff.

    The null is built by re-running the same probe (same activations, same grouped
    CV) on randomly permuted labels n_shuffles times. The permutation preserves the
    class multiset, so the null balanced accuracy centres on chance for THIS
    geometry and sample size. The real probe counts as encoding the distinction
    only if its balanced accuracy exceeds the 95th percentile of that null — a
    control-task test in the Hewitt & Liang sense, with the bar read off the data
    rather than chosen.

    Returns:
        Dict with the real balanced accuracy, the null mean / 95th percentile,
        n_shuffles, and beats_null (bool).
    """
    real_result = run_linear_probe(activations, labels, config, pair_ids=pair_ids)
    real_balanced_accuracy = real_result["balanced_accuracy_mean"]

    rng = np.random.default_rng(seed)
    null_balanced_accuracies: list[float] = []
    for _ in range(n_shuffles):
        shuffled_labels = list(rng.permutation(np.asarray(labels, dtype=object)))
        shuffled_result = run_linear_probe(activations, shuffled_labels, config, pair_ids=pair_ids)
        if np.isfinite(shuffled_result["balanced_accuracy_mean"]):
            null_balanced_accuracies.append(shuffled_result["balanced_accuracy_mean"])

    if not null_balanced_accuracies:
        return {
            "balanced_accuracy": real_balanced_accuracy,
            "null_balanced_p95": float("nan"),
            "null_balanced_mean": float("nan"),
            "n_shuffles": 0,
            "beats_null": False,
        }

    null_array = np.array(null_balanced_accuracies)
    null_p95 = float(np.percentile(null_array, 95))
    return {
        "balanced_accuracy": float(real_balanced_accuracy),
        "null_balanced_p95": null_p95,
        "null_balanced_mean": float(null_array.mean()),
        "n_shuffles": len(null_balanced_accuracies),
        "beats_null": bool(np.isfinite(real_balanced_accuracy) and real_balanced_accuracy > null_p95),
    }


# ── Dispersion analysis (T1c: Lewis vs Stalnaker) ────────────────────────────

def _dispersion_measures(activations: np.ndarray) -> dict[str, float]:
    """
    Three measures of how spread-out a set of activation vectors is. Used to
    contrast tie_case against clear_case for T1c: Lewis predicts ties are more
    dispersed (no unique closest world), Stalnaker predicts ties collapse to a
    centroid like clear cases.

    All three are reported because a single scalar can be gamed by outliers or by
    the raw scale of the activations:

      total_variance         — mean squared distance to the class centroid
                               (= trace of the covariance). Scale-dependent,
                               outlier-sensitive; the classic but fragile measure.
      median_pairwise_dist   — median Euclidean distance between item pairs.
                               Robust to outliers (median, not mean).
      participation_ratio    — (Σλ)² / Σλ² over the covariance eigenvalues λ.
                               The *effective dimensionality* of the cloud:
                               ~1 if the cloud collapses onto a single direction
                               (Stalnaker centroid), large if it fills many
                               dimensions (Lewis diffusion). Scale-free.
    """
    n_items = activations.shape[0]
    centroid = activations.mean(axis=0)
    centered = activations - centroid

    total_variance = float((centered ** 2).sum(axis=1).mean())

    # Singular values of the centered matrix give covariance eigenvalues:
    # λ_i = s_i² / (n - 1). Participation ratio is invariant to that constant.
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values ** 2
    sum_eigenvalues = float(eigenvalues.sum())
    participation_ratio = (
        float((sum_eigenvalues ** 2) / (eigenvalues ** 2).sum())
        if sum_eigenvalues > 0 else 0.0
    )

    # Median pairwise distance over upper-triangle pairs.
    if n_items > 1:
        row_index, col_index = np.triu_indices(n_items, k=1)
        pairwise_distances = np.linalg.norm(activations[row_index] - activations[col_index], axis=1)
        median_pairwise_dist = float(np.median(pairwise_distances))
    else:
        median_pairwise_dist = 0.0

    return {
        "total_variance": total_variance,
        "median_pairwise_dist": median_pairwise_dist,
        "participation_ratio": participation_ratio,
    }


def run_dispersion_analysis(
    activations: np.ndarray,
    labels: list[str],
    label_clear: str = "clear_case",
    label_tie: str = "tie_case",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Lewis vs Stalnaker discriminator for T1c via DISPERSION, not separability.

    A linear probe that separates clear from tie proves only that the two
    sentence-types differ — true under both theories, so it cannot discriminate
    them. The theories instead disagree on the GEOMETRY of the tie cloud:

      Lewis     → tie cloud is MORE dispersed than clear (indeterminacy: the
                  representation spreads over a set of equally-close worlds).
                  dispersion ratio tie/clear > 1.
      Stalnaker → tie cloud collapses to a centroid like clear (single world
                  selected). dispersion ratio ≈ 1.

    To make the ratio fair the two classes must have matched topical diversity —
    T1c enforces this with minimal pairs (same domain/template, adjective-only
    difference). Classes are subsampled to equal n here so neither sample size
    nor imbalance drives the dispersion.

    A bootstrap (resampling items within each class) gives a 95% CI on each
    ratio. The pre-specified read is threshold-free — the verdict is the CI's
    position relative to the null ratio = 1:
      lewis_confirmed     = participation-ratio CI lies entirely above 1.0
      stalnaker_confirmed = participation-ratio CI brackets 1.0 (cannot reject
                            equal dispersion)
      (CI entirely below 1.0 → inconclusive)

    Args:
        activations: np.ndarray (n_items, hidden_dim) at one layer.
        labels:      length-n_items condition labels.
        label_clear: the determinate (control) class name.
        label_tie:   the indeterminate (test) class name.
        n_bootstrap: bootstrap resamples for the CI. Default 1000.
        seed:        RNG seed.

    Returns:
        Dict with per-class dispersion measures, the tie/clear ratios with 95%
        CIs, the matched per-class n, and the lewis/stalnaker flags. Returns a
        "note" and NaN ratios if either class is too small (< 3 items) to have a
        meaningful covariance.
    """
    activations = np.asarray(activations, dtype=np.float64)
    clear_indices = np.array([i for i, label in enumerate(labels) if label == label_clear])
    tie_indices = np.array([i for i, label in enumerate(labels) if label == label_tie])

    measure_names = ("total_variance", "median_pairwise_dist", "participation_ratio")

    if len(clear_indices) < 3 or len(tie_indices) < 3:
        return {
            "note": (
                f"Dispersion analysis skipped: need >= 3 items per class, got "
                f"{len(clear_indices)} {label_clear} and {len(tie_indices)} {label_tie}."
            ),
            "n_per_class": int(min(len(clear_indices), len(tie_indices))),
            "lewis_confirmed": False,
            "stalnaker_confirmed": False,
            "dispersion_ratios": {name: float("nan") for name in measure_names},
        }

    rng = np.random.default_rng(seed)

    # Subsample the larger class so both clouds have equal n — dispersion measures
    # (especially participation ratio) depend on sample size, so an unequal n
    # would bias the ratio independently of geometry.
    matched_n = min(len(clear_indices), len(tie_indices))

    clear_activations_full = activations[clear_indices]
    tie_activations_full = activations[tie_indices]

    def ratio_for_sample(clear_sample: np.ndarray, tie_sample: np.ndarray) -> dict[str, float]:
        clear_measures = _dispersion_measures(clear_sample)
        tie_measures = _dispersion_measures(tie_sample)
        return {
            name: (tie_measures[name] / clear_measures[name] if clear_measures[name] > 0 else float("nan"))
            for name in measure_names
        }

    # Full-sample per-class measures (matched n, all distinct items) — reported
    # for transparency. NOT used directly as the point ratio: participation ratio
    # is sample-size sensitive, and a with-replacement bootstrap collapses
    # duplicate rows onto fewer effective dimensions, so a full-sample point
    # estimate can fall outside the bootstrap CI. To keep the point estimate and
    # its interval on the same footing, the reported point ratio is the bootstrap
    # MEDIAN and the CI is the 2.5/97.5 percentile of the SAME distribution.
    point_clear_measures = _dispersion_measures(clear_activations_full)
    point_tie_measures = _dispersion_measures(tie_activations_full)

    bootstrap_ratios: dict[str, list[float]] = {name: [] for name in measure_names}
    for _ in range(n_bootstrap):
        clear_sample = clear_activations_full[rng.choice(len(clear_activations_full), matched_n, replace=True)]
        tie_sample = tie_activations_full[rng.choice(len(tie_activations_full), matched_n, replace=True)]
        sample_ratios = ratio_for_sample(clear_sample, tie_sample)
        for name in measure_names:
            bootstrap_ratios[name].append(sample_ratios[name])

    point_ratios: dict[str, float] = {}
    confidence_intervals: dict[str, list[float]] = {}
    for name in measure_names:
        finite = np.array([value for value in bootstrap_ratios[name] if np.isfinite(value)])
        if finite.size:
            point_ratios[name] = float(np.median(finite))
            confidence_intervals[name] = [
                float(np.percentile(finite, 2.5)),
                float(np.percentile(finite, 97.5)),
            ]
        else:
            point_ratios[name] = float("nan")
            confidence_intervals[name] = [float("nan"), float("nan")]

    # Threshold-free verdict: the null is "equal dispersion" (ratio = 1). We read
    # it straight off the bootstrap 95% CI of the ratio — no hand-set constant.
    #   CI entirely above 1  → tie significantly MORE dispersed → Lewis.
    #   CI brackets 1         → cannot reject equal dispersion → consistent with
    #                           Stalnaker (ties collapse to a clear-like centroid).
    #   CI entirely below 1   → tie LESS dispersed than clear → neither prediction
    #                           (unexpected) → inconclusive.
    # The earlier "point < 1.15" was an arbitrary magic number and is removed.
    participation_ci = confidence_intervals["participation_ratio"]
    ci_lower, ci_upper = participation_ci[0], participation_ci[1]
    ci_finite = np.isfinite(ci_lower) and np.isfinite(ci_upper)
    lewis_confirmed = ci_finite and ci_lower > 1.0
    stalnaker_confirmed = ci_finite and ci_lower <= 1.0 <= ci_upper

    return {
        "n_per_class": int(matched_n),
        "clear_dispersion": point_clear_measures,
        "tie_dispersion": point_tie_measures,
        "dispersion_ratios": point_ratios,
        "dispersion_ratio_ci_95": confidence_intervals,
        "discriminating_measure": "participation_ratio",
        "lewis_confirmed": bool(lewis_confirmed),
        "stalnaker_confirmed": bool(stalnaker_confirmed),
        "criterion": (
            "participation-ratio dispersion ratio tie/clear, bootstrap 95% CI vs the "
            "null ratio = 1. CI entirely > 1 -> Lewis (ties diffuse). CI brackets 1 -> "
            "consistent with Stalnaker (cannot reject equal dispersion). CI entirely < 1 "
            "-> inconclusive. No hand-set threshold."
        ),
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


def run_partial_mantel_test(
    model_matrix: np.ndarray,
    theory_matrix: np.ndarray,
    covariate_matrix: np.ndarray,
    n_perms: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Partial Mantel: Spearman association between model and theory RDMs, controlling
    for a covariate RDM, with a permutation null.

    For T1b: model_matrix = activation similarity; theory_matrix = M_graph (or
    M_sim); covariate_matrix = M_sim (or M_graph). The partial correlation answers
    "does the geometry track this theory AFTER removing what the other theory
    explains?" — the discriminator that survives the Lewis/Pearl truth-convergence.

    Method: rank-transform the three flattened upper triangles (Spearman), linearly
    residualize model-ranks and theory-ranks on covariate-ranks, correlate the
    residuals (Pearson). Null: permute model_matrix rows+cols, recompute.

    Args:
        model_matrix:     (n, n) model pairwise similarity.
        theory_matrix:    (n, n) theory of interest.
        covariate_matrix: (n, n) the competing theory, partialled out.
        n_perms:          permutations for the null. Default 1000.
        seed:             RNG seed.

    Returns:
        Dict: partial_r, p_value, significant (p<0.05), null_95th_percentile,
              exceeds_null_floor (partial_r > null 95th pct), n_perms.
    """
    from scipy.stats import rankdata

    n_items = model_matrix.shape[0]
    upper_row, upper_col = np.triu_indices(n_items, k=1)
    theory_flat = theory_matrix[upper_row, upper_col]
    covariate_flat = covariate_matrix[upper_row, upper_col]

    def _residualize(target_values: np.ndarray, predictor_values: np.ndarray) -> np.ndarray:
        design = np.vstack([np.ones_like(predictor_values), predictor_values]).T
        beta, _residuals, _rank, _sv = np.linalg.lstsq(design, target_values, rcond=None)
        return target_values - design @ beta

    def _partial_r(model_flat: np.ndarray) -> float:
        model_ranks = rankdata(model_flat)
        theory_ranks = rankdata(theory_flat)
        covariate_ranks = rankdata(covariate_flat)
        model_residual = _residualize(model_ranks, covariate_ranks)
        theory_residual = _residualize(theory_ranks, covariate_ranks)
        if model_residual.std() == 0 or theory_residual.std() == 0:
            return 0.0
        return float(np.corrcoef(model_residual, theory_residual)[0, 1])

    observed_partial_r = _partial_r(model_matrix[upper_row, upper_col])

    rng = np.random.default_rng(seed)
    null_partial_correlations = []
    for _ in range(n_perms):
        permutation_order = rng.permutation(n_items)
        permuted_model = model_matrix[permutation_order][:, permutation_order]
        null_partial_correlations.append(_partial_r(permuted_model[upper_row, upper_col]))

    null_partial_correlations = np.array(null_partial_correlations)
    empirical_p_value = float((null_partial_correlations >= observed_partial_r).mean())
    null_95th = float(np.percentile(null_partial_correlations, 95))

    return {
        "partial_r": observed_partial_r,
        "p_value": empirical_p_value,
        "significant": empirical_p_value < 0.05,
        "null_95th_percentile": null_95th,
        "exceeds_null_floor": observed_partial_r > null_95th,
        "n_perms": n_perms,
    }
