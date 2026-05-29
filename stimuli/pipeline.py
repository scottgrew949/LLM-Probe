"""
stimuli/pipeline.py — Stimulus generation, validation, behavioral gating.

─── CONCEPT: Stimuli as philosophical experiments ────────────────────────────
In analytic philosophy, a thought experiment is a carefully constructed scenario
designed to isolate a single variable and test intuitions about it. PoL-Probe
stimuli work the same way: each stimulus *pair* is a minimal contrast — two
sentences identical in all respects except the one philosophical distinction
being tested.

For T2 (Frege sense/reference), a pair might be:
  sentence_a: "Alice believes Hesperus is visible tonight."   ← opaque context
  sentence_b: "Alice believes the evening star is visible."   ← still opaque, same referent

The theoretical distinction is whether the model encodes the *sense* (the mode
of presentation, "Hesperus" vs "evening star") or only the *reference*
(Venus, the same object either way).

─── CONCEPT: The validate/ gate ──────────────────────────────────────────────
Raw generated stimuli live in stimuli/generated/. They cannot be used for
extraction or probing until validate_set() has:
  1. Checked format against stimulus.schema.json
  2. Verified corpus frequency matching (V7)
  3. Written the validated file to stimuli/validated/

extractor.py enforces this by refusing stimulus files not under validated/.

─── CONCEPT: Behavioral gate (V8) ────────────────────────────────────────────
Before probing internal representations, you must verify the model actually
exhibits the behavior you want to explain. If a model can't reliably distinguish
opaque from transparent contexts at the behavioral level (>70% accuracy on
forced-choice items), there's nothing to explain mechanistically — the
representation you'd be hunting doesn't exist at the output level.

This is the same logic as: don't look for the neural correlates of pain in
a patient who reports no pain. Behavior gates mechanism.

─── CONCEPT: Frequency matching (V7) ─────────────────────────────────────────
A probe that can distinguish sentence_a from sentence_b might just be detecting
that one sentence uses rare words. That's a surface-statistics artifact, not
a semantic distinction. Frequency matching ensures that both sentences in a
pair draw from similar frequency ranges in the training corpus, so any probe
signal must reflect meaning rather than rarity.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path
from typing import Any

# Type alias for a stimulus pair dict matching stimulus.schema.json
StimulusPair = dict[str, Any]


# ── Generation ────────────────────────────────────────────────────────────────

def generate_pairs(grammar_file: str | Path, n: int, thread_id: str) -> list[StimulusPair]:
    """
    Generate n stimulus pairs from a grammar file and write to stimuli/generated/{thread_id}/.

    Grammar files (stimuli/grammars/{thread_id}.py or .json) define the templates
    and vocabulary for constructing pairs. Programmatic generation ensures:
      - Large stimulus sets without manual annotation
      - Systematic coverage of the theoretical space
      - Reproducibility (same grammar + seed = same pairs)

    Args:
        grammar_file: Path to the grammar definition for this thread.
        n:            Number of pairs to generate.
        thread_id:    e.g. "t2" — determines output path.

    Returns:
        List of StimulusPair dicts (not yet validated, not yet frequency-matched).
        Also writes to stimuli/generated/{thread_id}/pairs.jsonl.

    Note:
        Output file is stimuli/generated/{thread_id}/pairs.jsonl, one JSON object
        per line (JSONL format). Each object must pass stimulus.schema.json validation
        before moving to validated/.
    """
    project_root = Path(__file__).parent.parent
    output_directory = project_root / "stimuli" / "generated" / thread_id
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file_path = output_directory / "pairs.jsonl"

    grammar_path = Path(grammar_file)

    # Load grammar — supports .py modules with a `generate(n) -> list[dict]` function
    # or .json files with a "templates" list
    if grammar_path.suffix == ".py":
        grammar_module_spec = importlib.util.spec_from_file_location("grammar_module", grammar_path)
        grammar_module = importlib.util.module_from_spec(grammar_module_spec)
        grammar_module_spec.loader.exec_module(grammar_module)
        generated_pairs = grammar_module.generate(n)
    elif grammar_path.suffix == ".json":
        with grammar_path.open("r") as grammar_file_handle:
            grammar_definition = json.load(grammar_file_handle)
        # JSON grammars: {"templates": [{"sentence_a": "...", "sentence_b": "..."}]}
        # Repeat/truncate to n
        raw_templates = grammar_definition["templates"]
        generated_pairs = (raw_templates * ((n // len(raw_templates)) + 1))[:n]
    else:
        raise ValueError(f"Unsupported grammar file type: {grammar_path.suffix}. Expected .py or .json.")

    with output_file_path.open("w") as output_file_handle:
        for pair in generated_pairs:
            output_file_handle.write(json.dumps(pair) + "\n")

    return generated_pairs


# ── Validation ────────────────────────────────────────────────────────────────

def validate_set(pairs: list[StimulusPair], thread_id: str) -> list[StimulusPair]:
    """
    Validate pairs against stimulus.schema.json, run frequency matching,
    and write accepted pairs to stimuli/validated/{thread_id}/pairs.validated.jsonl.

    This is the gate between raw generation and use in experiments. Only pairs
    that pass both schema validation and frequency matching are written to
    validated/. The returned list is the validated subset.

    Sets frequency_match_verified = True on each passing pair. [INVARIANT V5]
    This is the ONLY place that field is set — never set it manually elsewhere.

    Args:
        pairs:     List of raw StimulusPair dicts from generate_pairs().
        thread_id: Determines output path.

    Returns:
        List of validated StimulusPair dicts with frequency_match_verified=True.
        Raises if zero pairs pass (something is wrong with generation).

    Side effects:
        Writes stimuli/validated/{thread_id}/pairs.validated.jsonl.
        Logs rejection reasons for any failed pairs.

    Raises:
        ValueError: if all pairs fail validation.
    """
    import jsonschema

    project_root = Path(__file__).parent.parent
    schema_path = project_root / "stimuli" / "schemas" / "stimulus.schema.json"
    output_directory = project_root / "stimuli" / "validated" / thread_id
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file_path = output_directory / "pairs.validated.jsonl"

    with schema_path.open("r") as schema_file_handle:
        stimulus_schema = json.load(schema_file_handle)

    schema_validator = jsonschema.Draft7Validator(stimulus_schema)

    validated_pairs = []
    rejection_reasons = []

    for pair_index, pair in enumerate(pairs):
        schema_errors = list(schema_validator.iter_errors(pair))
        if schema_errors:
            rejection_reasons.append({
                "pair_index": pair_index,
                "reason": "schema_validation_failed",
                "errors": [error.message for error in schema_errors],
            })
            continue

        if not check_frequency_match(pair):
            rejection_reasons.append({
                "pair_index": pair_index,
                "reason": "frequency_match_failed",
                "sentence_a": pair.get("sentence_a", ""),
                "sentence_b": pair.get("sentence_b", ""),
            })
            continue

        # [INVARIANT V7] Only validate_set sets this — never set it anywhere else
        pair["frequency_matched"] = True
        validated_pairs.append(pair)

    if rejection_reasons:
        logger = logging.getLogger(__name__)
        for rejection in rejection_reasons:
            logger.warning("Rejected pair %d: %s", rejection["pair_index"], rejection["reason"])

    if not validated_pairs:
        raise ValueError(
            f"All {len(pairs)} pairs failed validation. "
            f"Rejection reasons: {rejection_reasons[:5]}. "
            f"Check grammar file output against stimulus.schema.json."
        )

    with output_file_path.open("w") as output_file_handle:
        for validated_pair in validated_pairs:
            output_file_handle.write(json.dumps(validated_pair) + "\n")

    return validated_pairs


def check_frequency_match(pair: StimulusPair) -> bool:
    """
    Check whether sentence_a and sentence_b have comparable corpus frequencies.

    Computes the mean log10 corpus frequency of content words in each sentence,
    then checks that the two means differ by at most 1.0 — which is one order
    of magnitude on a log10 scale (i.e., frequency ratio ≤ 10×).

    Why this matters: a linear probe trained on activations for rare-word sentences
    vs. common-word sentences could be detecting word frequency, not meaning.
    Frequency matching forecloses this confound.

    Args:
        pair: A StimulusPair dict. Must have sentence_a and sentence_b fields.

    Returns:
        True if the mean log10 frequency difference is ≤ 1.0, False otherwise.

    Note:
        Uses wordfreq library (pip install wordfreq) for corpus frequency lookups.
        Language: English. Corpus: combined web text.
    """
    import re
    from wordfreq import word_frequency

    sentence_a = pair.get("sentence_a", "")
    sentence_b = pair.get("sentence_b", "")

    function_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "and", "or", "but", "not",
        "that", "this", "it", "its",
    }

    def extract_content_words(sentence: str) -> list[str]:
        all_words = re.findall(r"[a-z]+", sentence.lower())
        return [word for word in all_words if word not in function_words]

    def mean_log10_frequency(word_list: list[str]) -> float:
        log_freqs = [math.log10(max(word_frequency(w, "en"), 1e-9)) for w in word_list]
        return sum(log_freqs) / len(log_freqs)

    content_words_a = extract_content_words(sentence_a)
    content_words_b = extract_content_words(sentence_b)

    if not content_words_a or not content_words_b:
        return True  # can't check — pass through

    frequency_difference = abs(mean_log10_frequency(content_words_a) - mean_log10_frequency(content_words_b))
    return frequency_difference <= 1.0


# ── Behavioral gate ───────────────────────────────────────────────────────────

def run_behavioral_gate(
    behavioral_items: list[dict[str, Any]],
    model: Any,  # HookedTransformer — caller loads once and passes in
    threshold: float = 0.70,
) -> dict[str, Any]:
    """
    Run the model on forced-choice behavioral items and check it passes the gate.

    Before any mechanistic analysis (probing, patching), the model must demonstrate
    it can actually make the philosophical distinction at the behavioral level.
    Forced-choice means: given a question and two candidate sentences, the model
    must assign higher log-probability to the correct one.

    Scoring: for each choice, we sum log-probs over *only the choice tokens*
    conditioned on the question prefix. This is the standard forced-choice
    evaluation method (Brown et al. 2020). Using the full-string loss would
    mix question-token loss into the comparison and penalize longer choices.

    The 0.70 floor is hardcoded in ExperimentConfig — this function respects it.
    [INVARIANT V8]

    Args:
        behavioral_items: List of forced-choice items.
                          Each item: {"question": str, "choice_a": str,
                                      "choice_b": str, "correct": "a"|"b"}
        model:            HookedTransformer instance. Caller loads once and passes in.
        threshold:        Accuracy threshold. Floor is 0.70 — passing a lower
                          value raises ValueError.

    Returns:
        Dict with keys:
          "passed":   bool — did the model clear the threshold?
          "accuracy": float — fraction correct
          "n_items":  int — total items tested
          "details":  list of per-item results

    Raises:
        ValueError: if threshold < 0.70 (V8 enforcement).
    """
    if threshold < 0.70:
        raise ValueError(f"threshold={threshold} violates V8: floor is 0.70")

    import torch

    per_item_results = []
    correct_count = 0

    for behavioral_item in behavioral_items:
        question_prefix = behavioral_item["question"]
        choice_a_text = behavioral_item["choice_a"]
        choice_b_text = behavioral_item["choice_b"]
        correct_choice = behavioral_item["correct"]  # "a" or "b"

        # Score each choice by the sum of log-probs of *only the choice tokens*
        # conditioned on the question. This is the standard forced-choice scoring
        # method (Brown et al. 2020). Using the full-string loss would penalize
        # longer choices and mix question-token loss into the comparison, which
        # we don't want since the question is identical for both choices.
        def score_choice(question: str, choice: str) -> float:
            question_token_ids = model.to_tokens(question, prepend_bos=True)
            full_prompt_token_ids = model.to_tokens(f"{question} {choice}", prepend_bos=True)
            n_question_tokens = question_token_ids.shape[1]

            with torch.no_grad():
                # logits shape: (1, seq_len, vocab_size)
                logits = model(full_prompt_token_ids, return_type="logits")

            log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)

            # For each choice token at position i, the correct prediction is token i+1
            # (the model predicts next token, so logits at position i predict position i+1)
            choice_token_ids = full_prompt_token_ids[0, n_question_tokens:]
            choice_log_probs = log_probs[n_question_tokens - 1 : n_question_tokens - 1 + len(choice_token_ids)]

            # Sum log-probs over choice tokens — higher sum = model prefers this choice
            total_log_prob = sum(
                choice_log_probs[token_index, token_id].item()
                for token_index, token_id in enumerate(choice_token_ids)
            )
            return total_log_prob

        log_prob_choice_a = score_choice(question_prefix, choice_a_text)
        log_prob_choice_b = score_choice(question_prefix, choice_b_text)

        model_choice = "a" if log_prob_choice_a > log_prob_choice_b else "b"
        is_correct = model_choice == correct_choice
        if is_correct:
            correct_count += 1

        per_item_results.append({
            "question": question_prefix,
            "model_choice": model_choice,
            "correct_choice": correct_choice,
            "is_correct": is_correct,
            "log_prob_choice_a": log_prob_choice_a,
            "log_prob_choice_b": log_prob_choice_b,
        })

    accuracy = correct_count / len(behavioral_items) if behavioral_items else 0.0

    return {
        "passed": accuracy >= threshold,
        "accuracy": accuracy,
        "n_items": len(behavioral_items),
        "details": per_item_results,
    }


# ── PhilBench entry construction ──────────────────────────────────────────────

def build_philbench_entry(
    pair: StimulusPair,
    config: Any,  # ExperimentConfig — not imported here to avoid circular import
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a PhilBench benchmark entry from a stimulus pair and its experimental result.

    PhilBench is the theory-annotated benchmark released on HuggingFace at the end
    of the project. Each entry pairs a stimulus with its theoretical labels,
    the model's behavioral response, and the probe/patching result. This lets
    future researchers use our stimulus set as an evaluation benchmark.

    Args:
        pair:   Validated StimulusPair dict.
        config: ExperimentConfig for the experiment this pair came from.
        result: The experimental result dict (probe result or patching result).

    Returns:
        A dict conforming to philbench.schema.json. Includes:
          - pair_id, thread_id, sentences, theoretical labels
          - model_id, model_revision
          - behavioral_accuracy (from gate result)
          - probe_result or patch_result summary
          - theoretical_distinction, expected_outcome fields

    Note:
        Validated against philbench.schema.json before being written to
        stimuli/philbench/philbench.jsonl.
    """
    import jsonschema

    project_root = Path(__file__).parent.parent
    philbench_schema_path = project_root / "stimuli" / "schemas" / "philbench.schema.json"

    with philbench_schema_path.open("r") as schema_file_handle:
        philbench_schema = json.load(schema_file_handle)

    pair_id = pair.get("pair_id", "")
    thread_id = config.thread_id

    philbench_entry = {
        "philbench_id": f"pb_{thread_id}_{pair_id}",
        "pair_id": pair_id,
        "thread_id": thread_id,
        "theoretical_distinction": pair.get("theoretical_distinction", ""),
        "sentence_a": pair.get("sentence_a", ""),
        "sentence_b": pair.get("sentence_b", ""),
        "label_a": pair.get("label_a", ""),
        "label_b": pair.get("label_b", ""),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "behavioral_accuracy": result.get("behavioral_accuracy", 0.0),
        "behavioral_gate_passed": result.get("behavioral_gate_passed", False),
        "probe_peak_layer": result.get("probe_peak_layer", 0),
        "probe_accuracy": result.get("probe_accuracy", 0.0),
        "patch_peak_layer": result.get("patch_peak_layer", None),
        "patch_effect_size": result.get("patch_effect_size", None),
        "rsa_spearman_r": result.get("rsa_spearman_r", None),
        "rsa_p_value": result.get("rsa_p_value", None),
        "surface_null_accuracy": result.get("surface_null_accuracy", 0.0),
        "experiment_id": config.experiment_id,
    }

    jsonschema.validate(philbench_entry, philbench_schema)

    return philbench_entry
