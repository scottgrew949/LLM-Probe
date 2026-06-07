"""
experiments/t1_experiments.py — T1a / T1b / T1c / T1d experiment subclasses.

Each class inherits ThreadExperiment and implements three abstract methods:
  build_locked_config(n_layers) — pre-registers expected outcomes
  analyze(config, model, n_layers) — thread-specific pipeline (L2 + L3)
  print_results(summary) — formatted verdict table

Entry shims at experiments/t1{a,b,c,d}_{gpt2,pythia}/run_experiment.py are ~4
lines each — they instantiate the right class with the right thread_id and call .run().
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thread_experiment import ThreadExperiment


# ─────────────────────────────────────────────────────────────────────────────
# T1a — Pearl Level 3 counterfactual existence test
# ─────────────────────────────────────────────────────────────────────────────

class T1aExperiment(ThreadExperiment):
    """
    T1a: Does the model encode Pearl Level 3 (interventional vs observational)?

    L2 (linear probe) locates the layer. L3 (activation patching layer sweep)
    confirms causal role. level3_confirmed gates T1b and T1c.
    """

    MIDDLE_LATE_LAYER_FLOOR = 8  # layers >= 8 = "semantic/causal" in GPT-2 medium

    def build_locked_config(self, n_layers: int):
        from extraction.extractor import compute_sha256
        from experiments.config import ExperimentConfig
        from stimuli.pipeline import verify_stimulus_file_frequency_matched

        expected_outcomes = {
            "level3_confirmed_criterion": (
                "probe accuracy at peak layer > 0.70 "
                "AND probe accuracy > surface_null_accuracy + 0.10"
            ),
            "outcome_if_separable_middle_late_layers": (
                "causal_l3 and associative_l1 are linearly separable at layers 8-23 "
                "— model encodes Pearl Level 3 structure semantically, not just syntactically. "
                "T1b and T1c proceed."
            ),
            "outcome_if_separable_early_layers_only": (
                "Separable at layers 0-7 only — distinction is syntactic/surface. "
                "Model detects the past-perfect-subjunctive grammatical marker ('had not been'), "
                "not the causal structure. Level 3 confirmed but mechanism is surface, not semantic."
            ),
            "outcome_if_not_separable": (
                "causal_l3 and associative_l1 not linearly separable at any layer. "
                "Model treats interventional and observational framing identically internally. "
                "Level 3 absent. T1b and T1c moot."
            ),
            "outcome_if_layers_agree": (
                "Peak L2 probe layer == peak L3 patching layer — "
                "storage and causal use co-locate. Strong mechanistic finding."
            ),
            "outcome_if_layers_disagree": (
                "Peak L2 layer != peak L3 layer — "
                "distinction is stored at one depth and used at another."
            ),
        }

        date_stamp = datetime.date.today().strftime("%Y%m%d")
        config = ExperimentConfig(
            experiment_id=self.thread_id + "_" + date_stamp,
            thread_id=self.thread_id,
            model_id=self.model_id,
            model_revision="main",
            layer_range=(0, n_layers - 1),
            component="resid_post",
            token_positions=[-1],
            probe_type="linear",
            stimulus_file=str(self.validated_path),
            stimulus_sha256=compute_sha256(self.validated_path),
            frequency_match_verified=verify_stimulus_file_frequency_matched(self.validated_path),
            expected_outcomes=expected_outcomes,
        )
        config.lock()
        return config

    def analyze(self, config, model, n_layers: int) -> dict:
        from extraction.extractor import extract_activations
        from probes.probes import run_linear_probe
        from interventions.interventions import (
            run_layer_sweep_multi_target, assert_specificity_valid, norm_matched_control_kl,
        )
        from core.io import save_result, load_result

        print("=" * 60)
        print("PoL-Probe — " + self.thread_id + " — Pearl Level 3 Existence Test")
        print("=" * 60)
        print()

        # ── L2: linear probe at each layer ────────────────────────────────────
        print("[Step 1] Extracting activations and running linear probe at each layer...")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        layer_activation_sets = extract_activations(config, model)
        probe_results_by_layer: dict[int, dict] = {}

        for activation_set in layer_activation_sets:
            layer_index = activation_set["layer"]
            probe_result = run_linear_probe(
                np.array(activation_set["activations"]),
                activation_set["labels"],
                config,
                pair_ids=activation_set["pair_group_ids"],
            )
            probe_result["layer"] = layer_index
            probe_result["token_position"] = activation_set["token_position"]
            save_result(probe_result, self.results_dir / ("probe_layer_" + str(layer_index) + ".json"))
            probe_results_by_layer[layer_index] = probe_result

        peak_probe_layer = max(
            probe_results_by_layer,
            key=lambda l: probe_results_by_layer[l]["accuracy_mean"],
        )
        peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
        print("  Peak probe layer: " + str(peak_probe_layer) +
              "  (accuracy=" + str(round(peak_probe_accuracy * 100, 1)) + "%)")
        print()

        # ── L3: layer sweep ───────────────────────────────────────────────────
        print("[Step 2] Running activation-patching layer sweep (L3)...")

        stimulus_pairs: list[dict] = []
        with self.validated_path.open("r") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    stimulus_pairs.append(json.loads(stripped))

        mean_source_by_layer: dict[int, np.ndarray] = {}
        for activation_set in layer_activation_sets:
            all_acts = np.array(activation_set["activations"])
            mean_source_by_layer[activation_set["layer"]] = all_acts[::2].mean(axis=0)

        target_sentences = [pair["sentence_b"] for pair in stimulus_pairs]

        sweep_result = run_layer_sweep_multi_target(
            mean_source_by_layer,
            target_sentences,
            config.layer_range,
            config.component,
            config.token_positions[0],
            model,
            seed=config.seed,
        )
        save_result(sweep_result, self.results_dir / "layer_sweep.json")

        peak_patch_layer = sweep_result["peak_layer"]
        peak_patch_kl = sweep_result["mean_kl_by_layer"][peak_patch_layer]

        # Specificity (V2)
        control_result = norm_matched_control_kl(
            mean_source_by_layer[peak_patch_layer], target_sentences,
            peak_patch_layer, config.component, config.token_positions[0], model,
            seed=config.seed,
        )
        assert_specificity_valid(peak_patch_kl, control_result["control_kl_p95"], peak_patch_layer)

        print("  Peak patch layer: " + str(peak_patch_layer) +
              "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
        print()

        # ── level3_confirmed criterion (pre-specified) ────────────────────────
        surface_null = load_result(self.results_dir / "surface_null.json")
        surface_null_accuracy = surface_null["surface_classifier_accuracy"]

        probe_exceeds_threshold = peak_probe_accuracy > 0.70
        probe_exceeds_surface = peak_probe_accuracy > surface_null_accuracy + 0.10
        level3_confirmed = probe_exceeds_threshold and probe_exceeds_surface

        layers_agree = peak_probe_layer == peak_patch_layer

        expected_outcomes = config.expected_outcomes
        if level3_confirmed:
            interpretation = (
                expected_outcomes["outcome_if_separable_middle_late_layers"]
                if peak_probe_layer >= self.MIDDLE_LATE_LAYER_FLOOR
                else expected_outcomes["outcome_if_separable_early_layers_only"]
            )
        else:
            interpretation = expected_outcomes["outcome_if_not_separable"]

        layer_agreement_note = (
            expected_outcomes["outcome_if_layers_agree"]
            if layers_agree
            else expected_outcomes["outcome_if_layers_disagree"]
        )

        probe_layers_summary = [
            {
                "layer": l,
                "accuracy_mean": float(probe_results_by_layer[l]["accuracy_mean"]),
                "accuracy_std": float(probe_results_by_layer[l].get("accuracy_std", float("nan"))),
                "chance_baseline": float(probe_results_by_layer[l]["chance_baseline"]),
            }
            for l in sorted(probe_results_by_layer)
        ]

        return {
            "experiment_id": config.experiment_id,
            "thread_id": config.thread_id,
            "model_id": config.model_id,
            "run_timestamp": datetime.datetime.utcnow().isoformat(),
            "peak_probe_layer": peak_probe_layer,
            "peak_probe_accuracy": float(peak_probe_accuracy),
            "peak_patch_layer": peak_patch_layer,
            "peak_patch_kl": float(peak_patch_kl),
            "layers_agree": layers_agree,
            "level3_confirmed": level3_confirmed,
            "interpretation": interpretation,
            "layer_agreement_interpretation": layer_agreement_note,
            "surface_null_accuracy": float(surface_null_accuracy),
            "level3_confirmed_criterion": expected_outcomes["level3_confirmed_criterion"],
            "probe_exceeds_threshold": bool(probe_exceeds_threshold),
            "probe_exceeds_surface": bool(probe_exceeds_surface),
            "n_layers_probed": len(probe_results_by_layer),
            "probe_layers_summary": probe_layers_summary,
            "expected_outcomes": expected_outcomes,
        }

    def print_results(self, summary: dict) -> None:
        n_layers = summary["n_layers_probed"]
        peak_probe_layer = summary["peak_probe_layer"]
        peak_patch_layer = summary["peak_patch_layer"]
        peak_probe_accuracy = summary["peak_probe_accuracy"]
        peak_patch_kl = summary["peak_patch_kl"]
        surface_null_accuracy = summary["surface_null_accuracy"]
        level3_confirmed = summary["level3_confirmed"]

        print("[Step 3] Layer-by-layer probe accuracy (L2)")
        print()
        print("  Layer   Accuracy    Std    Chance")
        print("  -----   --------   -----   ------")
        for row in summary["probe_layers_summary"]:
            marker = "  <-- PEAK (L2)" if row["layer"] == peak_probe_layer else ""
            print(
                "  " + str(row["layer"]).rjust(5) +
                "   " + str(round(row["accuracy_mean"] * 100, 1)).rjust(6) + "%" +
                "   " + str(round(row["accuracy_std"] * 100, 1)).rjust(4) + "%" +
                "   " + str(round(row["chance_baseline"] * 100, 1)).rjust(5) + "%" +
                marker
            )
        print()
        print("  Peak patching layer (L3): Layer " + str(peak_patch_layer) +
              "   KL=" + str(round(peak_patch_kl, 4)))
        print()

        print("=" * 60)
        print("T1a Final Result [" + self.thread_id + "]")
        print("=" * 60)
        print()
        print("Peak probe layer    : Layer " + str(peak_probe_layer) +
              "  (accuracy=" + str(round(peak_probe_accuracy * 100, 1)) + "%)")
        print("Peak patching layer : Layer " + str(peak_patch_layer) +
              "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
        print("L2 / L3 agreement   : " + ("YES" if summary["layers_agree"] else "NO"))
        print()
        print("Surface null accuracy : " + str(round(surface_null_accuracy * 100, 1)) + "%")
        print("Probe accuracy        : " + str(round(peak_probe_accuracy * 100, 1)) + "%  (L2 peak)")
        print("Excess over surface   : +" +
              str(round((peak_probe_accuracy - surface_null_accuracy) * 100, 1)) + "%")
        print()
        print("Probe > 0.70          : " + str(summary["probe_exceeds_threshold"]))
        print("Probe > surface + 0.10: " + str(summary["probe_exceeds_surface"]))
        print("level3_confirmed      : " + str(level3_confirmed))
        print()
        print("Interpretation:")
        print("  " + summary["interpretation"])
        print()
        print("Layer agreement:")
        print("  " + summary["layer_agreement_interpretation"])
        print()
        print("=" * 60)
        if level3_confirmed:
            print("T1b and T1c: UNLOCKED")
        else:
            print("T1b and T1c: BLOCKED")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# T1b — mechanism-geometry (Pearl graph vs Lewis similarity)
# ─────────────────────────────────────────────────────────────────────────────

class T1bExperiment(ThreadExperiment):
    """
    T1b: Is counterfactual representation graph-structured (Pearl) or
    similarity-structured (Lewis)?

    L2: RSA partial-Mantel per layer — partial-r against decorrelated M_graph and
    M_sim. L3: asymmetry patching on direct-cause probe pairs.
    """

    ASYM_STRUCTURE_LABEL = "direct_asym"
    MATRIX_SOURCE = "stimuli/theoretical_matrices/t1b_matrices.py"

    def _pre_phase_gate_hook(self, config, model, n_layers: int) -> None:
        """Build theoretical matrices and write decorrelation artifact (V23)."""
        from stimuli.theoretical_matrices.t1b_matrices import (
            build_graph_similarity_matrix, build_domain_similarity_matrix,
            corr_between_matrices, assert_matrices_decorrelated,
            MATRIX_DECORRELATION_THRESHOLD,
        )
        from core.io import save_result

        print("[Pre-gate] Building theoretical matrices and checking decorrelation (V23)...")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        all_sentences: list[str] = []
        all_labels: list[str] = []
        with self.validated_path.open("r") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                pair = json.loads(stripped)
                all_sentences.append(pair["sentence_a"]); all_labels.append(pair["label_a"])
                all_sentences.append(pair["sentence_b"]); all_labels.append(pair["label_b"])

        rsa_indices = [
            i for i, label in enumerate(all_labels)
            if label.split("|", 1)[0] != self.ASYM_STRUCTURE_LABEL
        ]
        rsa_labels = [all_labels[i] for i in rsa_indices]

        matrix_graph = build_graph_similarity_matrix(rsa_labels)
        matrix_sim = build_domain_similarity_matrix(rsa_labels)
        observed_corr = corr_between_matrices(matrix_graph, matrix_sim)
        decorrelation_passed = abs(observed_corr) < MATRIX_DECORRELATION_THRESHOLD

        save_result(
            {
                "passed": bool(decorrelation_passed),
                "corr": float(observed_corr),
                "threshold": MATRIX_DECORRELATION_THRESHOLD,
                "n_items": len(rsa_labels),
            },
            self.results_dir / "matrix_decorrelation.json",
        )
        print("  corr(M_graph, M_sim) = " + str(round(observed_corr, 4)) +
              "  (passed: " + str(decorrelation_passed) + ")")

        # Hard-stop here too (V23 gate also checks, but fail fast before gate).
        assert_matrices_decorrelated(matrix_graph, matrix_sim)
        print()

        # Store for use in analyze()
        self._all_sentences = all_sentences
        self._all_labels = all_labels
        self._rsa_indices = rsa_indices
        self._matrix_graph = matrix_graph
        self._matrix_sim = matrix_sim

    def build_locked_config(self, n_layers: int):
        from extraction.extractor import compute_sha256
        from experiments.config import ExperimentConfig
        from stimuli.pipeline import verify_stimulus_file_frequency_matched

        expected_outcomes = {
            "geometry_criterion": (
                "Partial-Mantel RSA of the per-layer activation RDM against M_graph and "
                "M_sim, each partialling the other out. PEARLIAN: partial-RSA(graph) "
                "Holm-significant across a >=3-layer band AND > partial-RSA(sim) there AND "
                "asymmetry CI excludes 0. LEWISIAN: partial-RSA(sim) significant across a "
                ">=3-layer band AND > graph AND asymmetry CI includes 0. Else mixed/"
                "inconclusive. No single-layer argmax verdict."
            ),
            "outcome_if_pearlian": (
                "Representation clusters by causal STRUCTURE across domains and patching is "
                "directionally asymmetric — graph-structured counterfactual mechanism."
            ),
            "outcome_if_lewisian": (
                "Representation clusters by DOMAIN/topic and patching is symmetric — holistic "
                "similarity-metric mechanism (Lewis idealization: symmetric similarity)."
            ),
            "idealization_footnote": (
                "Treating Lewis comparative similarity as a SYMMETRIC relation is a defensible "
                "simplification, not a settled fact."
            ),
            "truth_convergence_note": (
                "Lewis and Pearl coincide on truth-conditions for simple SCMs; T1b is a "
                "MECHANISM/geometry test, not a verdict on which theory gives the right answer."
            ),
        }

        date_stamp = datetime.date.today().strftime("%Y%m%d")
        config = ExperimentConfig(
            experiment_id=self.thread_id + "_" + date_stamp,
            thread_id=self.thread_id,
            model_id=self.model_id,
            model_revision="main",
            layer_range=(0, n_layers - 1),
            component="resid_post",
            token_positions=[-1],
            probe_type="rsa",
            stimulus_file=str(self.validated_path),
            stimulus_sha256=compute_sha256(self.validated_path),
            frequency_match_verified=verify_stimulus_file_frequency_matched(self.validated_path),
            expected_outcomes=expected_outcomes,
            matrix_source=self.MATRIX_SOURCE,
        )
        config.lock()
        return config

    @staticmethod
    def _holm_significant(p_values_by_layer: dict, alpha: float = 0.05) -> set:
        """Holm-Bonferroni step-down correction. Returns layers that survive."""
        ordered = sorted(p_values_by_layer.items(), key=lambda item: item[1])
        n_tests = len(ordered)
        significant: set = set()
        for rank, (layer_index, p_value) in enumerate(ordered):
            if p_value <= alpha / (n_tests - rank):
                significant.add(layer_index)
            else:
                break
        return significant

    @staticmethod
    def _contiguous_band(layers: set, minimum_length: int = 3) -> list:
        """Return contiguous runs of length >= minimum_length in sorted layer set."""
        if not layers:
            return []
        ordered = sorted(layers)
        runs: list = []
        current = [ordered[0]]
        for layer_index in ordered[1:]:
            if layer_index == current[-1] + 1:
                current.append(layer_index)
            else:
                if len(current) >= minimum_length:
                    runs.append(current)
                current = [layer_index]
        if len(current) >= minimum_length:
            runs.append(current)
        return runs

    def analyze(self, config, model, n_layers: int) -> dict:
        from extraction.extractor import extract_activations
        from probes.probes import run_partial_mantel_test
        from interventions.interventions import (
            patch_activation, norm_matched_control_kl, assert_specificity_valid, asymmetry_index,
        )
        from core.io import save_result, load_result
        from sklearn.metrics.pairwise import cosine_similarity
        import torch

        print("=" * 60)
        print("PoL-Probe — " + self.thread_id + " — Mechanism Geometry")
        print("Graph-structured (Pearl) vs Similarity-structured (Lewis)")
        print("=" * 60)
        print()

        all_sentences = self._all_sentences
        all_labels = self._all_labels
        rsa_indices = self._rsa_indices
        matrix_graph = self._matrix_graph
        matrix_sim = self._matrix_sim

        # ── L2: RSA partial-Mantel per layer ──────────────────────────────────
        print("[Step 1] Extracting activations and running RSA partial-Mantel per layer...")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        layer_activation_sets = extract_activations(config, model)

        partial_r_graph_by_layer: dict[int, float] = {}
        partial_r_sim_by_layer: dict[int, float] = {}
        p_graph_by_layer: dict[int, float] = {}
        p_sim_by_layer: dict[int, float] = {}

        for activation_set in layer_activation_sets:
            layer_index = activation_set["layer"]
            full_acts = np.array(activation_set["activations"])
            rsa_acts = full_acts[rsa_indices]
            centered = rsa_acts - rsa_acts.mean(axis=0, keepdims=True)
            model_rdm = cosine_similarity(centered)

            graph_result = run_partial_mantel_test(
                model_rdm, matrix_graph, matrix_sim, seed=config.seed
            )
            sim_result = run_partial_mantel_test(
                model_rdm, matrix_sim, matrix_graph, seed=config.seed
            )

            partial_r_graph_by_layer[layer_index] = graph_result["partial_r"]
            partial_r_sim_by_layer[layer_index] = sim_result["partial_r"]
            p_graph_by_layer[layer_index] = graph_result["p_value"]
            p_sim_by_layer[layer_index] = sim_result["p_value"]

            save_result(
                {
                    "layer": layer_index,
                    "partial_r_graph": graph_result["partial_r"],
                    "p_graph": graph_result["p_value"],
                    "partial_r_sim": sim_result["partial_r"],
                    "p_sim": sim_result["p_value"],
                },
                self.results_dir / ("rsa_layer_" + str(layer_index) + ".json"),
            )

        # Exclude layer 0 (raw token/positional embeddings) from verdict
        non_zero_layers = [l for l in partial_r_graph_by_layer if l != 0]
        if not non_zero_layers:
            raise ValueError("No non-zero layers in RSA results — extraction may have failed.")

        graph_significant = self._holm_significant(
            {l: p_graph_by_layer[l] for l in non_zero_layers}
        )
        sim_significant = self._holm_significant(
            {l: p_sim_by_layer[l] for l in non_zero_layers}
        )
        graph_band_layers = {
            l for l in graph_significant
            if partial_r_graph_by_layer[l] > partial_r_sim_by_layer[l]
        }
        sim_band_layers = {
            l for l in sim_significant
            if partial_r_sim_by_layer[l] > partial_r_graph_by_layer[l]
        }
        graph_bands = self._contiguous_band(graph_band_layers)
        sim_bands = self._contiguous_band(sim_band_layers)

        patch_layer = max(
            non_zero_layers,
            key=lambda l: abs(partial_r_graph_by_layer[l]),
        )
        print("  Graph bands (>=3 contiguous Holm-sig): " + str(graph_bands))
        print("  Sim bands   (>=3 contiguous Holm-sig): " + str(sim_bands))
        print("  L3 patch layer (peak |graph_r|): " + str(patch_layer))
        print()

        # ── L3: asymmetry patching on direct-cause probe pairs ────────────────
        print("[Step 2] Asymmetry patching (cause→effect vs effect→cause)...")

        asym_pairs: list[tuple[int, int]] = []
        asym_cursor = 0
        with self.validated_path.open("r") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                pair = json.loads(stripped)
                a_idx, b_idx = asym_cursor, asym_cursor + 1
                asym_cursor += 2
                if pair["label_a"].split("|", 1)[0] == self.ASYM_STRUCTURE_LABEL:
                    asym_pairs.append((a_idx, b_idx))

        patch_act_set = next(s for s in layer_activation_sets if s["layer"] == patch_layer)
        patch_layer_acts = np.array(patch_act_set["activations"])

        kl_cause_to_effect_list: list[float] = []
        kl_effect_to_cause_list: list[float] = []
        effect_target_sentences: list[str] = []
        cause_source_activations: list[np.ndarray] = []

        for cause_idx, effect_idx in asym_pairs:
            cause_sentence = all_sentences[cause_idx]
            effect_sentence = all_sentences[effect_idx]
            source_cause = patch_layer_acts[cause_idx]
            source_effect = patch_layer_acts[effect_idx]

            with torch.no_grad():
                baseline_effect = model(effect_sentence)[0, -1, :].tolist()
                baseline_cause = model(cause_sentence)[0, -1, :].tolist()

            forward = patch_activation(
                source_cause, {"stimulus": effect_sentence}, patch_layer,
                config.component, config.token_positions[0], model,
                baseline_logits=baseline_effect,
            )
            backward = patch_activation(
                source_effect, {"stimulus": cause_sentence}, patch_layer,
                config.component, config.token_positions[0], model,
                baseline_logits=baseline_cause,
            )
            kl_cause_to_effect_list.append(float(forward["kl_from_baseline"]))
            kl_effect_to_cause_list.append(float(backward["kl_from_baseline"]))
            effect_target_sentences.append(effect_sentence)
            cause_source_activations.append(source_cause)

        # Guard: if no asymmetry pairs, asymmetry is undefined
        if not asym_pairs:
            mean_kl_ce = float("nan")
            mean_kl_ec = float("nan")
            asym = float("nan")
            asym_ci = [float("nan"), float("nan")]
            asym_ci_excludes_zero = False
            n_domains = 0
            control_mean_kl = float("nan")
        else:
            mean_kl_ce = float(np.mean(kl_cause_to_effect_list))
            mean_kl_ec = float(np.mean(kl_effect_to_cause_list))
            asym = asymmetry_index(mean_kl_ce, mean_kl_ec)

            rng = np.random.default_rng(config.seed)
            n_domains = len(asym_pairs)
            bootstrap_asymmetries: list[float] = []
            for _ in range(1000):
                sample = rng.integers(0, n_domains, n_domains)
                ce = float(np.mean([kl_cause_to_effect_list[i] for i in sample]))
                ec = float(np.mean([kl_effect_to_cause_list[i] for i in sample]))
                bootstrap_asymmetries.append(asymmetry_index(ce, ec))

            boot_arr = np.array(bootstrap_asymmetries, dtype=float)
            asym_ci = [
                float(np.nanpercentile(boot_arr, 2.5)),
                float(np.nanpercentile(boot_arr, 97.5)),
            ]
            # Excludes zero if both CI bounds are strictly positive (or both negative).
            asym_ci_excludes_zero = bool(
                (not np.isnan(asym_ci[0])) and (asym_ci[0] > 0 or asym_ci[1] < 0)
            )

            mean_cause_source = np.mean(np.array(cause_source_activations), axis=0)
            control_result = norm_matched_control_kl(
                mean_cause_source, effect_target_sentences, patch_layer,
                config.component, config.token_positions[0], model, seed=config.seed,
            )
            assert_specificity_valid(mean_kl_ce, control_result["control_kl_p95"], patch_layer)
            control_mean_kl = float(control_result["mean_control_kl"])

        print("  mean KL cause→effect : " + str(round(mean_kl_ce, 4) if not np.isnan(mean_kl_ce) else "nan"))
        print("  mean KL effect→cause : " + str(round(mean_kl_ec, 4) if not np.isnan(mean_kl_ec) else "nan"))
        print("  asymmetry index      : " + str(round(asym, 3) if not np.isnan(asym) else "nan") +
              "  CI [" + str(round(asym_ci[0], 3) if not np.isnan(asym_ci[0]) else "nan") + ", "
              + str(round(asym_ci[1], 3) if not np.isnan(asym_ci[1]) else "nan") + "]"
              + "  (excludes 0: " + str(asym_ci_excludes_zero) + ")")
        print()

        # ── Verdict ───────────────────────────────────────────────────────────
        pearlian = bool(graph_bands) and asym_ci_excludes_zero and (not np.isnan(asym)) and asym > 0
        lewisian = bool(sim_bands) and not asym_ci_excludes_zero
        if pearlian and not lewisian:
            verdict = "PEARLIAN (graph-structured)"
            interpretation = config.expected_outcomes["outcome_if_pearlian"]
        elif lewisian and not pearlian:
            verdict = "LEWISIAN (similarity-structured)"
            interpretation = config.expected_outcomes["outcome_if_lewisian"]
        else:
            verdict = "MIXED / INCONCLUSIVE"
            interpretation = (
                "Neither graph-geometry (band + directional asymmetry) "
                "nor similarity-geometry (domain band + symmetric patching) met. No verdict."
            )

        decorrelation_result = load_result(self.results_dir / "matrix_decorrelation.json")

        return {
            "experiment_id": config.experiment_id,
            "thread_id": config.thread_id,
            "model_id": config.model_id,
            "run_timestamp": datetime.datetime.utcnow().isoformat(),
            "matrix_decorrelation_corr": float(decorrelation_result["corr"]),
            "surface_null_accuracy": float(
                load_result(self.results_dir / "surface_null.json")["surface_classifier_accuracy"]
            ),
            "partial_r_graph_by_layer": {str(k): float(v) for k, v in partial_r_graph_by_layer.items()},
            "partial_r_sim_by_layer": {str(k): float(v) for k, v in partial_r_sim_by_layer.items()},
            "graph_verdict_bands": graph_bands,
            "sim_verdict_bands": sim_bands,
            "l3_patch_layer": patch_layer,
            "mean_kl_cause_to_effect": mean_kl_ce,
            "mean_kl_effect_to_cause": mean_kl_ec,
            "asymmetry_index": float(asym) if not np.isnan(asym) else None,
            "asymmetry_ci_95": asym_ci,
            "asymmetry_ci_excludes_zero": asym_ci_excludes_zero,
            "norm_matched_control_kl": control_mean_kl,
            "n_asym_domains": n_domains,
            "geometry_verdict": verdict,
            "interpretation": interpretation,
            "idealization_footnote": config.expected_outcomes["idealization_footnote"],
            "truth_convergence_note": config.expected_outcomes["truth_convergence_note"],
            "expected_outcomes": config.expected_outcomes,
        }

    def print_results(self, summary: dict) -> None:
        n_layers = len(summary["partial_r_graph_by_layer"])
        graph_r = summary["partial_r_graph_by_layer"]
        sim_r = summary["partial_r_sim_by_layer"]

        print("=" * 60)
        print("T1b Results — Mechanism Geometry [" + self.thread_id + "]")
        print("=" * 60)
        print()
        print("Per-layer partial RSA (graph_r | sim_r):")
        print("  Layer   graph_r   sim_r")
        print("  -----   -------   -----")
        for layer_str in sorted(graph_r, key=lambda s: int(s)):
            print("  " + str(layer_str).rjust(5) +
                  "   " + str(round(graph_r[layer_str], 3)).rjust(6) +
                  "   " + str(round(sim_r[layer_str], 3)).rjust(5))
        print()
        asym = summary["asymmetry_index"]
        ci = summary["asymmetry_ci_95"]
        print("Asymmetry index : " + (str(round(asym, 3)) if asym is not None else "nan") +
              "  CI [" + str(round(ci[0], 3)) + ", " + str(round(ci[1], 3)) + "]")
        print("Surface null    : " + str(round(summary["surface_null_accuracy"] * 100, 1)) + "%")
        print()
        print("=" * 60)
        print("GEOMETRY: " + summary["geometry_verdict"])
        print("=" * 60)
        print()
        print("Interpretation:")
        print("  " + summary["interpretation"])
        print()
        print("Footnote: " + summary["idealization_footnote"])
        print("Note: " + summary["truth_convergence_note"])


# ─────────────────────────────────────────────────────────────────────────────
# T1c — Lewis (similarity-set) vs Stalnaker (single-selection)
# ─────────────────────────────────────────────────────────────────────────────

class T1cExperiment(ThreadExperiment):
    """
    T1c: Within worlds-based mechanism, does the selection function choose
    a set (Lewis indeterminacy at ties) or a single world (Stalnaker Limit
    Assumption)?

    Primary measure: DISPERSION (participation ratio of tie_case vs clear_case),
    not separability — a probe separates the two under both theories.
    """

    def _t1a_thread_id(self) -> str:
        """T1a run for the V10 gate: same model variant, level3_confirmed required."""
        suffix = self.thread_id[len(self.base_thread):]  # "_gpt2" or "_pythia"
        return "t1a" + suffix

    def _prereq_thread_id(self) -> str:
        """T1b run for pearl_confirmed context: same model variant."""
        suffix = self.thread_id[len(self.base_thread):]
        return "t1b" + suffix

    def _t1b_summary_path(self) -> Path:
        return PROJECT_ROOT / "experiments" / self._prereq_thread_id() / "results" / "summary.json"

    def _check_guards(self) -> None:
        super()._check_guards()
        t1b_path = self._t1b_summary_path()
        if not t1b_path.exists():
            print("ERROR: T1b summary not found.")
            print("  Expected: " + str(t1b_path))
            print("  Run experiments/" + self._prereq_thread_id() + "/run_experiment.py first.")
            sys.exit(1)
        from core.io import load_result
        self._t1b_summary = load_result(t1b_path)

    def build_locked_config(self, n_layers: int):
        from extraction.extractor import compute_sha256
        from experiments.config import ExperimentConfig
        from stimuli.pipeline import verify_stimulus_file_frequency_matched

        expected_outcomes = {
            "lewis_vs_stalnaker_criterion": (
                "DISPERSION ratio (not separability). participation-ratio of tie_case "
                "activations / clear_case at the peak layer, bootstrap 95% CI vs null ratio=1. "
                "CI entirely above 1.0 (and above the layer-0 lexical baseline) → Lewis "
                "(tie cloud diffuse — indeterminacy encoded). CI brackets 1.0 → Stalnaker "
                "(cannot reject equal dispersion — tie collapses to a clear-like centroid). "
                "No hand-set threshold."
            ),
            "outcome_if_lewis": (
                "tie_case activations are MORE dispersed than clear_case (participation-ratio "
                "ratio > 1, CI excludes 1). The model represents the symmetric-worlds "
                "indeterminacy Lewis predicts."
            ),
            "outcome_if_stalnaker": (
                "tie_case activations are NO more dispersed than clear_case (ratio ≈ 1). "
                "Consistent with Stalnaker's Limit Assumption: a unique closest world is "
                "always selected."
            ),
            "probe_role": (
                "The clear-vs-tie linear probe is a SANITY CHECK only: confirms the two "
                "conditions are distinguishable at all. Not the Lewis/Stalnaker discriminator."
            ),
        }

        date_stamp = datetime.date.today().strftime("%Y%m%d")
        config = ExperimentConfig(
            experiment_id=self.thread_id + "_" + date_stamp,
            thread_id=self.thread_id,
            model_id=self.model_id,
            model_revision="main",
            layer_range=(0, n_layers - 1),
            component="resid_post",
            token_positions=[-1],
            probe_type="linear",
            stimulus_file=str(self.validated_path),
            stimulus_sha256=compute_sha256(self.validated_path),
            frequency_match_verified=verify_stimulus_file_frequency_matched(self.validated_path),
            expected_outcomes=expected_outcomes,
            # V10 gate checks level3_confirmed on this prereq — must be T1a, not T1b.
            # T1b is loaded separately in _check_guards() for pearl_confirmed context.
            prerequisite_experiment_id=self._t1a_thread_id(),
        )
        config.lock()
        return config

    def analyze(self, config, model, n_layers: int) -> dict:
        from extraction.extractor import extract_activations
        from probes.probes import run_linear_probe, run_dispersion_analysis
        from interventions.interventions import (
            run_layer_sweep_multi_target, assert_specificity_valid, norm_matched_control_kl,
        )
        from core.io import save_result, load_result

        pearl_confirmed = self._t1b_summary.get("pearl_confirmed", False)

        print("=" * 60)
        print("PoL-Probe — " + self.thread_id + " — Lewis vs Stalnaker")
        print("=" * 60)
        print()
        print("[Context] T1b geometry_verdict: " + self._t1b_summary.get("geometry_verdict", "unknown"))
        print("  " + ("T1b found Pearl. T1c probes partial worlds-structure." if pearl_confirmed
                      else "T1b found Lewis/Stalnaker. T1c discriminates the selection function."))
        print()

        # ── L2: 3-class probe at each layer (sanity check + peak locator) ─────
        print("[Step 1] Extracting activations and running 3-class probe at each layer...")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        layer_activation_sets = extract_activations(config, model)
        probe_results_by_layer: dict[int, dict] = {}

        for activation_set in layer_activation_sets:
            layer_index = activation_set["layer"]
            probe_result = run_linear_probe(
                np.array(activation_set["activations"]),
                activation_set["labels"],
                config,
                pair_ids=activation_set["pair_group_ids"],
            )
            probe_result["layer"] = layer_index
            save_result(probe_result, self.results_dir / ("probe_layer_" + str(layer_index) + ".json"))
            probe_results_by_layer[layer_index] = probe_result

        candidate_layers = [l for l in probe_results_by_layer if l != 0]
        peak_probe_layer = max(
            candidate_layers,
            key=lambda l: probe_results_by_layer[l]["balanced_accuracy_mean"],
        )
        peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
        chance_baseline = probe_results_by_layer[peak_probe_layer]["chance_baseline"]
        print("  3-class probe peak layer: " + str(peak_probe_layer))
        print()

        # ── Dispersion analysis at peak layer (THE Lewis/Stalnaker test) ──────
        print("[Step 2] Dispersion analysis at peak layer " + str(peak_probe_layer) + " (Lewis/Stalnaker test)...")

        peak_act_set = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
        peak_all_acts = np.array(peak_act_set["activations"])
        peak_all_labels = peak_act_set["labels"]
        peak_pair_ids = peak_act_set["pair_group_ids"]

        dispersion_result = run_dispersion_analysis(
            peak_all_acts, peak_all_labels,
            label_clear="clear_case", label_tie="tie_case",
            seed=config.seed,
        )
        save_result(dispersion_result, self.results_dir / "dispersion_analysis.json")

        participation_ratio = dispersion_result["dispersion_ratios"]["participation_ratio"]
        participation_ci = dispersion_result["dispersion_ratio_ci_95"]["participation_ratio"]

        # Layer-0 lexical baseline
        layer_0_act_set = next(s for s in layer_activation_sets if s["layer"] == 0)
        layer_0_dispersion = run_dispersion_analysis(
            np.array(layer_0_act_set["activations"]), layer_0_act_set["labels"],
            label_clear="clear_case", label_tie="tie_case", seed=config.seed,
        )
        layer_0_ratio = layer_0_dispersion["dispersion_ratios"]["participation_ratio"]
        save_result(layer_0_dispersion, self.results_dir / "dispersion_layer0_baseline.json")
        exceeds_lexical = bool(participation_ci[0] > layer_0_ratio)

        print("  participation ratio tie/clear : " + str(round(participation_ratio, 3)) +
              "  95% CI [" + str(round(participation_ci[0], 3)) + ", " +
              str(round(participation_ci[1], 3)) + "]")
        print("  layer-0 baseline ratio        : " + str(round(layer_0_ratio, 3)) +
              "  (CI lower exceeds it: " + str(exceeds_lexical) + ")")
        print()

        # Sanity-check pairwise probes
        def pairwise_probe_accuracy(la: str, lb: str) -> float:
            mask = [lbl in (la, lb) for lbl in peak_all_labels]
            filtered_acts = peak_all_acts[mask]
            filtered_labels = [lbl for lbl, keep in zip(peak_all_labels, mask) if keep]
            filtered_ids = [pid for pid, keep in zip(peak_pair_ids, mask) if keep]
            if len(set(filtered_labels)) < 2:
                return 0.5
            return run_linear_probe(filtered_acts, filtered_labels, config, pair_ids=filtered_ids)["accuracy_mean"]

        clear_vs_tie = pairwise_probe_accuracy("clear_case", "tie_case")
        clear_vs_near = pairwise_probe_accuracy("clear_case", "near_miss")
        tie_vs_near = pairwise_probe_accuracy("tie_case", "near_miss")
        save_result(
            {"clear_vs_tie": clear_vs_tie, "clear_vs_near_miss": clear_vs_near,
             "tie_vs_near_miss": tie_vs_near, "peak_layer": peak_probe_layer,
             "role": "sanity_check_only"},
            self.results_dir / "pairwise_probe_results.json",
        )
        print("  Sanity probes  clear/tie=" + str(round(clear_vs_tie * 100, 1)) +
              "%  clear/near=" + str(round(clear_vs_near * 100, 1)) +
              "%  tie/near=" + str(round(tie_vs_near * 100, 1)) + "%")
        print()

        # ── L3: layer sweep (clear_case → tie_case) ───────────────────────────
        print("[Step 3] Layer sweep (L3 — clear_case into tie_case)...")

        clear_indices = [
            i for i, lbl in enumerate(layer_activation_sets[0]["labels"])
            if lbl == "clear_case"
        ]
        mean_clear_by_layer: dict[int, np.ndarray] = {
            s["layer"]: np.array(s["activations"])[clear_indices].mean(axis=0)
            for s in layer_activation_sets
        }

        tie_sentences: list[str] = []
        with self.validated_path.open("r") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    pair = json.loads(stripped)
                    if pair.get("label_b") == "tie_case":
                        tie_sentences.append(pair["sentence_b"])

        sweep_result = run_layer_sweep_multi_target(
            mean_clear_by_layer, tie_sentences,
            config.layer_range, config.component, config.token_positions[0],
            model, seed=config.seed,
        )
        save_result(sweep_result, self.results_dir / "layer_sweep.json")

        peak_patch_layer = sweep_result["peak_layer"]
        peak_patch_kl = sweep_result["mean_kl_by_layer"][peak_patch_layer]
        peak_patch_kl_ci = sweep_result["kl_ci_95_by_layer"][peak_patch_layer]

        control_result = norm_matched_control_kl(
            mean_clear_by_layer[peak_patch_layer], tie_sentences,
            peak_patch_layer, config.component, config.token_positions[0], model, seed=config.seed,
        )
        assert_specificity_valid(peak_patch_kl, control_result["control_kl_p95"], peak_patch_layer)

        print("  Peak patching layer : " + str(peak_patch_layer) +
              "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
        print()

        # ── Verdict ───────────────────────────────────────────────────────────
        lewis_confirmed = dispersion_result["lewis_confirmed"] and exceeds_lexical
        stalnaker_confirmed = dispersion_result["stalnaker_confirmed"]
        layers_agree = peak_probe_layer == peak_patch_layer

        if lewis_confirmed:
            mechanism_label = "LEWIS (similarity-set, indeterminacy at ties)"
            mechanism_interpretation = config.expected_outcomes["outcome_if_lewis"]
        elif stalnaker_confirmed:
            mechanism_label = "STALNAKER (single-selection, Limit Assumption)"
            mechanism_interpretation = config.expected_outcomes["outcome_if_stalnaker"]
        elif dispersion_result["lewis_confirmed"] and not exceeds_lexical:
            mechanism_label = "INCONCLUSIVE (dispersion > 1 but not above layer-0 baseline)"
            mechanism_interpretation = (
                "Tie cloud more dispersed than clear, but gap attributable to adjective "
                "embeddings, not worlds computation."
            )
        else:
            mechanism_label = "INCONCLUSIVE (dispersion CI below 1)"
            mechanism_interpretation = (
                "Tie cloud less dispersed than clear — neither Lewis nor Stalnaker prediction met."
            )

        probe_layers_summary = [
            {
                "layer": l,
                "accuracy_mean": float(probe_results_by_layer[l]["accuracy_mean"]),
                "chance_baseline": float(probe_results_by_layer[l]["chance_baseline"]),
            }
            for l in sorted(probe_results_by_layer)
        ]

        surface_null = load_result(self.results_dir / "surface_null.json")

        return {
            "experiment_id": config.experiment_id,
            "thread_id": config.thread_id,
            "model_id": config.model_id,
            "run_timestamp": datetime.datetime.utcnow().isoformat(),
            "t1b_geometry_verdict": self._t1b_summary.get("geometry_verdict", "unknown"),
            "t1b_pearl_confirmed": pearl_confirmed,
            "peak_probe_layer": peak_probe_layer,
            "peak_probe_accuracy_3class": float(peak_probe_accuracy),
            "chance_baseline": float(chance_baseline),
            "surface_null_accuracy": float(surface_null["surface_classifier_accuracy"]),
            "dispersion_participation_ratio": float(participation_ratio),
            "dispersion_participation_ratio_ci": [float(participation_ci[0]), float(participation_ci[1])],
            "dispersion_total_variance_ratio": float(dispersion_result["dispersion_ratios"]["total_variance"]),
            "dispersion_median_dist_ratio": float(dispersion_result["dispersion_ratios"]["median_pairwise_dist"]),
            "dispersion_n_per_class": int(dispersion_result["n_per_class"]),
            "dispersion_layer0_baseline_ratio": float(layer_0_ratio),
            "dispersion_exceeds_lexical_baseline": exceeds_lexical,
            "sanity_clear_vs_tie": float(clear_vs_tie),
            "sanity_clear_vs_near_miss": float(clear_vs_near),
            "sanity_tie_vs_near_miss": float(tie_vs_near),
            "peak_patch_layer": peak_patch_layer,
            "peak_patch_kl_mean": float(peak_patch_kl),
            "peak_patch_kl_ci_95": [float(peak_patch_kl_ci[0]), float(peak_patch_kl_ci[1])],
            "norm_matched_control_kl": float(control_result["mean_control_kl"]),
            "layers_agree": layers_agree,
            "lewis_confirmed": lewis_confirmed,
            "stalnaker_confirmed": stalnaker_confirmed,
            "mechanism_label": mechanism_label,
            "mechanism_interpretation": mechanism_interpretation,
            "probe_layers_summary": probe_layers_summary,
            "expected_outcomes": config.expected_outcomes,
        }

    def print_results(self, summary: dict) -> None:
        peak_probe_layer = summary["peak_probe_layer"]
        participation_ratio = summary["dispersion_participation_ratio"]
        ci = summary["dispersion_participation_ratio_ci"]

        print("=" * 60)
        print("T1c Results — Lewis vs Stalnaker [" + self.thread_id + "]")
        print("=" * 60)
        print()
        print("3-class probe accuracy by layer:")
        print()
        print("  Layer   Accuracy    Chance")
        print("  -----   --------   ------")
        for row in summary["probe_layers_summary"]:
            marker = "  <-- PEAK" if row["layer"] == peak_probe_layer else ""
            print(
                "  " + str(row["layer"]).rjust(5) +
                "   " + str(round(row["accuracy_mean"] * 100, 1)).rjust(6) + "%" +
                "   " + str(round(row["chance_baseline"] * 100, 1)).rjust(5) + "%" +
                marker
            )
        print()
        print("DISPERSION test at peak layer " + str(peak_probe_layer) + " (tie/clear ratio):")
        print("  participation ratio  : " + str(round(participation_ratio, 3)) +
              "  95% CI [" + str(round(ci[0], 3)) + ", " + str(round(ci[1], 3)) + "]")
        print("  total variance       : " + str(round(summary["dispersion_total_variance_ratio"], 3)))
        print("  median pairwise dist : " + str(round(summary["dispersion_median_dist_ratio"], 3)))
        print("  layer-0 baseline     : " + str(round(summary["dispersion_layer0_baseline_ratio"], 3)))
        print()
        print("Sanity probes (NOT discriminator):")
        print("  clear/tie   : " + str(round(summary["sanity_clear_vs_tie"] * 100, 1)) + "%")
        print("  clear/near  : " + str(round(summary["sanity_clear_vs_near_miss"] * 100, 1)) + "%")
        print("  tie/near    : " + str(round(summary["sanity_tie_vs_near_miss"] * 100, 1)) + "%")
        print()
        print("Peak patching layer : Layer " + str(summary["peak_patch_layer"]) +
              "  (KL=" + str(round(summary["peak_patch_kl_mean"], 4)) + ")")
        print("L2 / L3 agreement   : " + ("YES" if summary["layers_agree"] else "NO"))
        print()
        print("=" * 60)
        print("MECHANISM: " + summary["mechanism_label"])
        print("=" * 60)
        print()
        print("Interpretation:")
        print("  " + summary["mechanism_interpretation"])


# ─────────────────────────────────────────────────────────────────────────────
# T1d — causal identification (back-door / front-door / unidentifiable)
# ─────────────────────────────────────────────────────────────────────────────

class T1dExperiment(ThreadExperiment):
    """
    T1d: Does the model internally distinguish causally identified structures
    (back-door adjustable, front-door adjustable) from non-identified ones?

    Informative regardless of T1b outcome:
      Pearl mechanism → tests whether representations respect full do-calculus.
      Lewis mechanism → tests whether similarity-ordered reps fail identification.
    """

    def _prereq_thread_id(self) -> str:
        suffix = self.thread_id[len(self.base_thread):]
        return "t1b" + suffix

    def _t1b_summary_path(self) -> Path:
        return PROJECT_ROOT / "experiments" / self._prereq_thread_id() / "results" / "summary.json"

    def _check_guards(self) -> None:
        super()._check_guards()
        t1b_path = self._t1b_summary_path()
        if not t1b_path.exists():
            print("ERROR: T1b summary not found.")
            print("  Expected: " + str(t1b_path))
            print("  Run experiments/" + self._prereq_thread_id() + "/run_experiment.py first.")
            sys.exit(1)
        from core.io import load_result
        self._t1b_summary = load_result(t1b_path)

    def build_locked_config(self, n_layers: int):
        from extraction.extractor import compute_sha256
        from experiments.config import ExperimentConfig
        from stimuli.pipeline import verify_stimulus_file_frequency_matched

        confounder_structure: dict = {
            "conditions": {
                "back_door_adjustable": {
                    "nodes": ["treatment", "outcome", "confounder"],
                    "edges": [["treatment", "outcome"], ["confounder", "treatment"], ["confounder", "outcome"]],
                    "criterion": "back_door",
                    "adjustment_set": ["confounder"],
                },
                "front_door_adjustable": {
                    "nodes": ["treatment", "mediator", "outcome", "hidden_confounder"],
                    "edges": [["treatment", "mediator"], ["mediator", "outcome"],
                              ["hidden_confounder", "treatment"], ["hidden_confounder", "outcome"]],
                    "criterion": "front_door",
                    "adjustment_set": ["mediator"],
                },
                "confounded_not_adjustable": {
                    "nodes": ["treatment", "outcome", "hidden_confounder"],
                    "edges": [["treatment", "outcome"],
                              ["hidden_confounder", "treatment"], ["hidden_confounder", "outcome"]],
                    "criterion": "none",
                    "adjustment_set": [],
                },
                "unconfounded_control": {
                    "nodes": ["treatment", "outcome"],
                    "edges": [["treatment", "outcome"]],
                    "criterion": "trivial",
                    "adjustment_set": [],
                },
            }
        }

        expected_outcomes: dict = {
            "identification_criterion": (
                "PRIMARY: balanced accuracy of the back_door_adjustable vs "
                "confounded_not_adjustable minimal-pair probe (chance 0.5) at the peak layer, "
                "calibrated against a shuffled-label null. Encodes adjustability iff balanced "
                "accuracy beats the null's 95th percentile (no fixed cutoff)."
            ),
            "outcome_if_pearl_and_encodes_identification": (
                "back_door_adjustable and front_door_adjustable cluster together, "
                "separated from confounded_not_adjustable. Pearl representations respect "
                "do-calculus identification conditions — the full causal hierarchy."
            ),
            "outcome_if_lewis_and_fails_identification": (
                "No separation between adjustable and not_adjustable conditions. "
                "Lewis similarity ordering has no notion of identifiability."
            ),
        }

        date_stamp = datetime.date.today().strftime("%Y%m%d")
        config = ExperimentConfig(
            experiment_id=self.thread_id + "_" + date_stamp,
            thread_id=self.thread_id,
            model_id=self.model_id,
            model_revision="main",
            layer_range=(0, n_layers - 1),
            component="resid_post",
            token_positions=[-1],
            probe_type="linear",
            stimulus_file=str(self.validated_path),
            stimulus_sha256=compute_sha256(self.validated_path),
            frequency_match_verified=verify_stimulus_file_frequency_matched(self.validated_path),
            expected_outcomes=expected_outcomes,
            prerequisite_experiment_id=self._prereq_thread_id(),
            identification_criterion="back_door",
            confounder_structure=confounder_structure,
        )
        config.lock()
        return config

    def analyze(self, config, model, n_layers: int) -> dict:
        from extraction.extractor import extract_activations
        from probes.probes import run_linear_probe, run_identification_probe, probe_beats_null
        from interventions.interventions import (
            run_layer_sweep_multi_target, assert_specificity_valid, norm_matched_control_kl,
        )
        from core.io import save_result, load_result

        pearl_confirmed = self._t1b_summary.get("pearl_confirmed", False)

        print("=" * 60)
        print("PoL-Probe — " + self.thread_id + " — Causal Identification")
        print("Back-Door vs Front-Door vs Unidentifiable")
        print("=" * 60)
        print()
        print("[Context] T1b geometry_verdict: " + self._t1b_summary.get("geometry_verdict", "unknown"))
        print("  " + ("T1d tests whether Pearl-consistent reps respect identification."
                      if pearl_confirmed
                      else "T1d tests whether Lewis-consistent reps fail identification."))
        print()

        # ── L2: four-class probe at each layer ────────────────────────────────
        print("[Step 1] Extracting activations and running 4-class probe at each layer...")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        layer_activation_sets = extract_activations(config, model)
        probe_results_by_layer: dict[int, dict] = {}

        for activation_set in layer_activation_sets:
            layer_index = activation_set["layer"]
            probe_result = run_linear_probe(
                np.array(activation_set["activations"]),
                activation_set["labels"],
                config,
                pair_ids=activation_set["pair_group_ids"],
            )
            probe_result["layer"] = layer_index
            save_result(probe_result, self.results_dir / ("probe_layer_" + str(layer_index) + ".json"))
            probe_results_by_layer[layer_index] = probe_result

        candidate_layers = [l for l in probe_results_by_layer if l != 0]
        peak_probe_layer = max(
            candidate_layers,
            key=lambda l: probe_results_by_layer[l]["balanced_accuracy_mean"],
        )
        peak_probe_accuracy = probe_results_by_layer[peak_probe_layer]["accuracy_mean"]
        peak_probe_balanced = probe_results_by_layer[peak_probe_layer]["balanced_accuracy_mean"]
        print("  4-class probe peak layer: " + str(peak_probe_layer))
        print()

        # ── Identification probes at peak layer ───────────────────────────────
        print("[Step 2] Identification probes at peak layer " + str(peak_probe_layer) + "...")

        peak_act_set = next(s for s in layer_activation_sets if s["layer"] == peak_probe_layer)
        peak_acts = np.array(peak_act_set["activations"])
        peak_labels = peak_act_set["labels"]
        peak_pair_ids = peak_act_set["pair_group_ids"]

        # Primary: back_door_adjustable vs confounded_not_adjustable (minimal pair)
        primary_mask = [lbl in ("back_door_adjustable", "confounded_not_adjustable") for lbl in peak_labels]
        primary_acts = peak_acts[primary_mask]
        primary_labels = [lbl for lbl, keep in zip(peak_labels, primary_mask) if keep]
        primary_ids = [gid for gid, keep in zip(peak_pair_ids, primary_mask) if keep]
        primary_result = run_linear_probe(
            primary_acts, primary_labels, config, pair_ids=primary_ids,
            selectivity_seed=config.seed,
        )
        primary_result["layer"] = peak_probe_layer
        primary_null = probe_beats_null(
            primary_acts, primary_labels, config, pair_ids=primary_ids, seed=config.seed,
        )
        primary_result["null_balanced_p95"] = primary_null["null_balanced_p95"]
        primary_result["beats_null"] = primary_null["beats_null"]
        save_result(primary_result, self.results_dir / "identification_probe_minimal.json")

        identification_balanced_acc = primary_result["balanced_accuracy_mean"]
        identification_criterion_met = primary_null["beats_null"]

        # Secondary: 3-vs-1 identified vs not
        identification_result = run_identification_probe(
            peak_acts, peak_labels, config, pair_ids=peak_pair_ids,
        )
        identification_result["layer"] = peak_probe_layer
        save_result(identification_result, self.results_dir / "identification_probe.json")
        identification_grouped_balanced = identification_result["balanced_accuracy_mean"]

        print("  PRIMARY back_door vs confounded (balanced) : " +
              str(round(identification_balanced_acc * 100, 1)) + "%   chance 50%")
        print("    null 95th pct                            : " +
              str(round(primary_null["null_balanced_p95"] * 100, 1)) + "%")
        print("  Criterion met (beats null)                 : " + str(identification_criterion_met))
        print("  SECONDARY identified-vs-not (balanced)     : " +
              str(round(identification_grouped_balanced * 100, 1)) + "%")
        print()

        # ── L3: layer sweep (identified → not-identified) ─────────────────────
        print("[Step 3] Layer sweep (back_door_adjustable → confounded_not_adjustable)...")

        back_door_indices = [
            i for i, lbl in enumerate(layer_activation_sets[0]["labels"])
            if lbl == "back_door_adjustable"
        ]
        mean_back_door_by_layer: dict[int, np.ndarray] = {
            s["layer"]: np.array(s["activations"])[back_door_indices].mean(axis=0)
            for s in layer_activation_sets
        }

        not_adjustable_sentences: list[str] = []
        with self.validated_path.open("r") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    pair = json.loads(stripped)
                    if pair.get("label_b") == "confounded_not_adjustable":
                        not_adjustable_sentences.append(pair["sentence_b"])

        sweep_result = run_layer_sweep_multi_target(
            mean_back_door_by_layer, not_adjustable_sentences,
            config.layer_range, config.component, config.token_positions[0],
            model, seed=config.seed,
        )
        save_result(sweep_result, self.results_dir / "layer_sweep.json")

        peak_patch_layer = sweep_result["peak_layer"]
        peak_patch_kl = sweep_result["mean_kl_by_layer"][peak_patch_layer]
        peak_patch_kl_ci = sweep_result["kl_ci_95_by_layer"][peak_patch_layer]
        layers_agree = peak_probe_layer == peak_patch_layer

        control_result = norm_matched_control_kl(
            mean_back_door_by_layer[peak_patch_layer], not_adjustable_sentences,
            peak_patch_layer, config.component, config.token_positions[0], model, seed=config.seed,
        )
        assert_specificity_valid(peak_patch_kl, control_result["control_kl_p95"], peak_patch_layer)

        print("  Peak patching layer : " + str(peak_patch_layer) +
              "  (KL=" + str(round(peak_patch_kl, 4)) + ")")
        print("  L2 / L3 agreement   : " + ("YES" if layers_agree else "NO"))
        print()

        probe_layers_summary = [
            {
                "layer": l,
                "accuracy_mean": float(probe_results_by_layer[l]["accuracy_mean"]),
                "balanced_accuracy_mean": float(probe_results_by_layer[l]["balanced_accuracy_mean"]),
                "chance_baseline": float(probe_results_by_layer[l]["chance_baseline"]),
            }
            for l in sorted(probe_results_by_layer)
        ]

        surface_null = load_result(self.results_dir / "surface_null.json")

        return {
            "experiment_id": config.experiment_id,
            "thread_id": config.thread_id,
            "model_id": config.model_id,
            "run_timestamp": datetime.datetime.utcnow().isoformat(),
            "t1b_geometry_verdict": self._t1b_summary.get("geometry_verdict", "unknown"),
            "t1b_pearl_confirmed": pearl_confirmed,
            "peak_probe_layer": peak_probe_layer,
            "peak_probe_accuracy_4class": float(peak_probe_accuracy),
            "peak_probe_balanced_accuracy_4class": float(peak_probe_balanced),
            "identification_primary_balanced_accuracy": float(identification_balanced_acc),
            "identification_primary_null_p95": float(primary_null["null_balanced_p95"]),
            "identification_primary_beats_null": bool(identification_criterion_met),
            "identification_primary_selectivity": float(primary_result.get("selectivity", float("nan"))),
            "identification_secondary_balanced_accuracy": float(identification_grouped_balanced),
            "identification_criterion_met": bool(identification_criterion_met),
            "surface_null_accuracy": float(surface_null["surface_classifier_accuracy"]),
            "peak_patch_layer": peak_patch_layer,
            "peak_patch_kl_mean": float(peak_patch_kl),
            "peak_patch_kl_ci_95": [float(peak_patch_kl_ci[0]), float(peak_patch_kl_ci[1])],
            "norm_matched_control_kl": float(control_result["mean_control_kl"]),
            "layers_agree": layers_agree,
            "probe_layers_summary": probe_layers_summary,
            "expected_outcomes": config.expected_outcomes,
        }

    def print_results(self, summary: dict) -> None:
        pearl_confirmed = summary["t1b_pearl_confirmed"]
        identification_met = summary["identification_criterion_met"]

        print("=" * 60)
        print("T1d Results — Causal Identification [" + self.thread_id + "]")
        print("=" * 60)
        print()
        print("4-class probe accuracy by layer (balanced):")
        print()
        print("  Layer   Accuracy   Balanced   Chance")
        print("  -----   --------   --------   ------")
        peak_probe_layer = summary["peak_probe_layer"]
        for row in summary["probe_layers_summary"]:
            marker = "  <-- PEAK" if row["layer"] == peak_probe_layer else ""
            print(
                "  " + str(row["layer"]).rjust(5) +
                "   " + str(round(row["accuracy_mean"] * 100, 1)).rjust(6) + "%" +
                "   " + str(round(row["balanced_accuracy_mean"] * 100, 1)).rjust(6) + "%" +
                "   " + str(round(row["chance_baseline"] * 100, 1)).rjust(5) + "%" +
                marker
            )
        print()
        print("Identification probe (back_door vs confounded):")
        print("  balanced accuracy : " + str(round(summary["identification_primary_balanced_accuracy"] * 100, 1)) + "%")
        print("  null 95th pct     : " + str(round(summary["identification_primary_null_p95"] * 100, 1)) + "%")
        print("  beats null        : " + str(identification_met))
        print()
        print("Peak patching layer : Layer " + str(summary["peak_patch_layer"]) +
              "  (KL=" + str(round(summary["peak_patch_kl_mean"], 4)) + ")")
        print("L2 / L3 agreement   : " + ("YES" if summary["layers_agree"] else "NO"))
        print()
        print("T1b context: " + ("Pearl mechanism" if pearl_confirmed else "Lewis mechanism"))
        print()
        print("=" * 60)
        if pearl_confirmed and identification_met:
            print("Finding: Pearl-consistent model WITH identifiability encoding. Full do-calculus.")
        elif pearl_confirmed and not identification_met:
            print("Finding: Pearl-consistent model WITHOUT identifiability. Partial do-calculus only.")
        elif not pearl_confirmed and not identification_met:
            print("Finding: Lewis-consistent model fails identification. Consistent with worlds-ordering.")
        else:
            print("Finding: Lewis-consistent model passes identification. Unexpected — investigate.")
        print("=" * 60)
