"""Synthetic smoke: matrices + partial Mantel + asymmetry index assemble correctly
without loading a model. Proves the analysis path, not the model behavior.

Run: .venv/bin/python tests/smoke_t1b_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from stimuli.theoretical_matrices.t1b_matrices import (
    build_graph_similarity_matrix, build_domain_similarity_matrix, assert_matrices_decorrelated)
from probes.probes import run_partial_mantel_test
from interventions.interventions import asymmetry_index

structures = ["chain", "fork", "direct", "collider"]
domains = ["w", "m", "e", "p", "f"]
labels = [f"{s}|{d}" for s in structures for d in domains for _ in range(2)]

matrix_graph = build_graph_similarity_matrix(labels)
matrix_sim = build_domain_similarity_matrix(labels)
assert_matrices_decorrelated(matrix_graph, matrix_sim)  # V23 holds by construction

# Fake activations that cluster by STRUCTURE => should look graph-like (Pearlian).
rng = np.random.default_rng(0)
structure_id = np.array([structures.index(label.split("|")[0]) for label in labels])
activations = np.eye(len(structures))[structure_id] + 0.05 * rng.random((len(labels), len(structures)))
model_rdm = cosine_similarity(activations - activations.mean(axis=0))

graph_result = run_partial_mantel_test(model_rdm, matrix_graph, matrix_sim, n_perms=200, seed=1)
sim_result = run_partial_mantel_test(model_rdm, matrix_sim, matrix_graph, n_perms=200, seed=1)
print("graph partial_r", round(graph_result["partial_r"], 3), "sig", graph_result["significant"])
print("sim   partial_r", round(sim_result["partial_r"], 3), "sig", sim_result["significant"])
assert graph_result["partial_r"] > sim_result["partial_r"], \
    "structure-clustered activations should look graph-like"

# Directed KL => positive asymmetry; symmetric KL => ~0.
assert asymmetry_index(0.8, 0.2) > 0.5
assert asymmetry_index(0.5, 0.5) == 0.0
print("asymmetry directed:", asymmetry_index(0.8, 0.2), "symmetric:", asymmetry_index(0.5, 0.5))
print("SMOKE OK")
