"""
experiments/config.py — ExperimentConfig dataclass.

─── CONCEPT: Pre-registration ────────────────────────────────────────────────
In philosophy of science, pre-registration means committing your hypotheses
and methods to paper *before* you collect data. This prevents a subtle form
of epistemic cheating: looking at the data first, then constructing a hypothesis
that fits it (HARKing — Hypothesising After Results are Known).

ExperimentConfig is the computational enforcement of pre-registration.
The field `pre_spec_locked` must be True before run_experiment() will execute.
You cannot set it True until you have written `expected_outcomes` — what you
*expect* to find. If your results diverge from expected_outcomes, that is a
real finding, not a failure.

─── CONCEPT: Reproducibility ─────────────────────────────────────────────────
This config gets serialized to experiments/{thread_id}/config.json and is
the single source of truth for the experiment. Given the config.json and the
stimulus file at the specified SHA256 hash, any result should be exactly
reproducible. The fields `model_revision`, `seed`, and `stimulus_sha256`
are what make this possible.

─── CONCEPT: Invariant enforcement ───────────────────────────────────────────
Several fields encode hard rules (invariants) from the SPEC. These are checked
in __post_init__ so violations fail immediately at object construction, not
silently at runtime. See the V-numbers — these map to SPEC §V.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentConfig:
    """
    Canonical configuration for a single PoL-Probe experiment run.

    Every experiment begins by constructing one of these, filling out
    expected_outcomes, then calling lock() to set pre_spec_locked = True.
    run_experiment() in run.py checks that flag and refuses to run if it's False.

    Fields marked [INVARIANT Vn] are enforced — violating them raises immediately.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    experiment_id: str
    """Unique run ID. Convention: {thread_id}_{model_short}_{YYYYMMDD}, e.g. t2_gpt2m_20260601"""

    thread_id: str
    """Which philosophical thread: t1 | t1a | t1b | t1c | t2 | t2b | t3 | t4 | t5 | t6"""

    # ── Model ─────────────────────────────────────────────────────────────────
    model_id: str
    """HuggingFace model string, e.g. 'gpt2-medium', 'EleutherAI/pythia-1.4b'"""

    model_revision: str
    """
    Pinned git revision or HuggingFace tag, e.g. 'main' or a commit hash.
    Required for reproducibility — 'latest' is not acceptable in final runs.
    """

    # ── Extraction parameters ─────────────────────────────────────────────────
    layer_range: tuple[int, int]
    """
    (start, end) layer indices to extract, inclusive. e.g. (0, 24) for all
    layers of GPT-2 medium. TransformerLens uses 0-indexed layers.
    """

    component: str
    """
    Which internal component to hook: 'resid_post' | 'attn_out' | 'mlp_out'.

    Concept: Transformers are composed of residual streams, attention heads, and
    MLP layers. resid_post is the full residual stream after each block — the
    running 'state of the model' at that depth. attn_out and mlp_out isolate
    which sub-component carries the information.
    """

    token_positions: list[int]
    """
    Which token positions to extract activations from. e.g. [-1] for the final
    token, [5, 6] for specific positions. The right choice depends on the thread:
    for T2 (Frege), the key token is likely the name being substituted.
    """

    probe_type: str
    """'linear' | 'cosine' | 'rsa' — which analysis to run on the activations."""

    # ── Statistical parameters ────────────────────────────────────────────────
    rsa_permutations: int = 1000
    """
    Number of permutations for the Mantel test. 1000 is standard — gives
    reliable p-values down to p=0.001. Increasing to 10000 for publication.

    Concept: The Mantel test asks 'is the correlation between two distance
    matrices higher than chance?' Permutation testing shuffles the labels
    1000 times to build a null distribution, then asks where the real
    correlation falls in that distribution.
    """

    seed: int = 42
    """Random seed for all stochastic operations. Fixed for reproducibility."""

    # ── Stimulus file ─────────────────────────────────────────────────────────
    stimulus_file: str = ""
    """
    Path to the validated stimulus file. [INVARIANT V12]
    Must be under stimuli/validated/ — extractor.py enforces this.
    Raw generated files under stimuli/generated/ are never accepted.
    """

    stimulus_sha256: str = ""
    """
    SHA256 hash of the stimulus file. [INVARIANT V12]
    extractor.py computes the hash at runtime and rejects mismatches.
    This ensures the exact stimulus set used is always known.
    """

    # ── Pre-registration fields ───────────────────────────────────────────────
    pre_spec_locked: bool = False
    """
    [INVARIANT V1] Must be True before run_experiment() executes.
    Set by calling lock() — which requires expected_outcomes to be non-empty.
    Never set manually.
    """

    expected_outcomes: dict[str, Any] = field(default_factory=dict)
    """
    Pre-specified predictions, written before data collection.
    e.g. {'peak_layer': '8-12', 'probe_accuracy': '>0.80', 'rsa_p': '<0.05'}
    These are your hypotheses. Results will be compared against them.
    """

    # ── Behavioral gate ───────────────────────────────────────────────────────
    frequency_match_verified: bool = False
    """
    [INVARIANT V5] Set only by validate_set() in pipeline.py, never manually.
    Ensures corpus frequency of sentence_a ≈ sentence_b before extraction,
    so probe differences reflect semantics not surface statistics.
    """

    behavioral_gate_threshold: float = 0.70
    """
    [INVARIANT V6] Floor is 0.70. Config cannot lower it.
    The model must achieve this accuracy on forced-choice behavioral items
    before any mechanistic analysis begins. Ensures the model actually
    exhibits the behavior you're trying to explain mechanistically.
    """

    # ── Thread-specific required fields ───────────────────────────────────────
    t5_asymmetry_thresholds: Optional[dict[str, float]] = None
    """
    [INVARIANT V4] Required non-null for T5 (Fine's grounding).
    Values must be derived from the 95th percentile of domain-specific permutation
    nulls (1000 label-permuted draws per domain) — not set from intuition.
    Expected keys: 'math_vs_unrelated_floor', 'math_vs_physical_floor'.
    Compute null distributions for unrelated-pairs and physical domains first,
    then set these fields to their respective 95th percentiles before lock().
    Must be pre-specified before T5 runs.
    """

    prerequisite_experiment_id: Optional[str] = None
    """
    [INVARIANT V10] Required for T1b and T1c.
    T1b (Lewis vs. Pearl mechanism comparison) and T1c (Lewis vs. Stalnaker)
    can only run if T1a (does causal hierarchy exist at all?) has passed.
    This field holds the experiment_id of the required prerequisite.
    """

    ontology_version: Optional[str] = None
    """
    [INVARIANT V13] Required non-null for T4.
    Specifies which version of BFO or DOLCE was used to construct the
    theoretical similarity matrix. Prevents circular construction
    (building matrix from model output, then comparing model to it).
    e.g. 'BFO-2020', 'DOLCE-Lite-1.0'
    """

    matrix_source: Optional[str] = None
    """
    [INVARIANT V13] Required non-null for T4.
    Path or identifier for the theoretical similarity matrix file.
    Must be constructed from the ontology independently of the model.
    """

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    run_timestamp: str = ""
    """ISO 8601 timestamp. Set automatically by run_experiment() at start."""

    def __post_init__(self) -> None:
        # V6: behavioral gate floor — config cannot lower it
        if self.behavioral_gate_threshold < 0.70:
            raise ValueError(
                f"behavioral_gate_threshold={self.behavioral_gate_threshold} "
                f"violates V6: floor is 0.70"
            )

        # Catch typos in component/probe_type early rather than failing deep in extractor
        valid_components = {"resid_post", "attn_out", "mlp_out"}
        if self.component not in valid_components:
            raise ValueError(
                f"component='{self.component}' is not valid. "
                f"Must be one of: {sorted(valid_components)}"
            )

        valid_probe_types = {"linear", "cosine", "rsa"}
        if self.probe_type not in valid_probe_types:
            raise ValueError(
                f"probe_type='{self.probe_type}' is not valid. "
                f"Must be one of: {sorted(valid_probe_types)}"
            )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_json(self, path: str | Path) -> None:
        """
        Serialize config to config.json at path. Call before lock().

        Creates parent directories if needed. The saved JSON is the record
        of what was planned — lock() will update pre_spec_locked and
        expected_outcomes in place.
        """
        config_as_dict = dataclasses.asdict(self)
        config_file_path = Path(path)
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with config_file_path.open("w") as config_file:
            json.dump(config_as_dict, config_file, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """
        Load ExperimentConfig from a saved config.json.

        Use this to reload an experiment's config in analysis scripts,
        or to check what was pre-specified before results were collected.
        """
        with Path(path).open("r") as config_file:
            data = json.load(config_file)
        data["layer_range"] = tuple(data["layer_range"])
        return cls(**data)

    def lock(self) -> None:
        """
        Pre-register: set pre_spec_locked = True and persist to config.json.

        This is the point of no return. After calling lock(), run_experiment()
        will accept this config. Before calling lock(), you must have written
        expected_outcomes — what you predict the results will show.

        Raises:
            ValueError: if expected_outcomes is empty (you haven't pre-specified)
            FileNotFoundError: if stimulus_file does not exist
            ValueError: if stimulus_sha256 is empty
        """
        if not self.expected_outcomes:
            raise ValueError(
                "expected_outcomes is empty. Write your pre-specified predictions before locking."
            )
        if not self.frequency_match_verified:
            raise ValueError(
                "frequency_match_verified is False (V7). "
                "Run validate_set() on stimulus pairs before locking. "
                "Do not set this field manually — only validate_set() may set it."
            )
        if self.stimulus_file and not Path(self.stimulus_file).exists():
            raise FileNotFoundError(f"stimulus_file not found: {self.stimulus_file}")
        if not self.stimulus_sha256:
            raise ValueError(
                "stimulus_sha256 is empty. Compute SHA256 of stimulus file and set it before locking."
            )
        self.pre_spec_locked = True
        # Caller must call to_json() after lock() to persist the updated config.
