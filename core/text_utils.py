"""
core/text_utils.py — Shared text and corpus frequency utilities.

Used by stimuli/pipeline.py (frequency matching) and experiments/run.py
(surface null classifier). Centralised here so the function_words set and
mean_log10_frequency computation stay in sync across both call sites.
"""

from __future__ import annotations

import math
import re

# Function words excluded from corpus frequency comparisons.
# Removing them focuses frequency matching on content words — the words that
# carry meaning and might introduce frequency confounds in stimulus pairs.
FUNCTION_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "and", "or", "but", "not",
    "that", "this", "it", "its",
})


def mean_log10_frequency(sentence: str) -> float:
    """
    Compute the mean log10 corpus frequency of content words in a sentence.

    Strips function words, computes wordfreq log10 frequency for each
    remaining word, returns the mean. Used by check_frequency_match (V7)
    and run_surface_null to build surface features.

    Args:
        sentence: Any string. Lowercased and tokenised by splitting on
                  non-alphabetic characters.

    Returns:
        Mean log10 frequency of content words. Returns 0.0 if no content
        words remain after stripping function words.
    """
    from wordfreq import word_frequency

    content_words = [
        word for word in re.findall(r"[a-z]+", sentence.lower())
        if word not in FUNCTION_WORDS
    ]
    if not content_words:
        return 0.0
    log_freqs = [math.log10(max(word_frequency(w, "en"), 1e-9)) for w in content_words]
    return sum(log_freqs) / len(log_freqs)
