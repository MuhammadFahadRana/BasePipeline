"""Lightweight Norwegian stemmer and morphological variant generator.

Handles common Norwegian (Bokmål/Nynorsk) suffixes so that different
inflected forms of the same word can match each other:
  brønnstrøm → brønnstrømmer, brønnstrømmen, brønnstrømmene
  ventil      → ventiler, ventilen, ventilene
"""

import re
from typing import List

# Ordered longest-first so we strip the most specific suffix first.
_SUFFIXES = (
    "ene",   # definite plural        (brønnstrømmene)
    "ane",   # nynorsk definite plural (brønnstrømmane)
    "ene",   # definite plural neuter
    "erte",  # past participle         (opererte)
    "erne",  # definite plural         (feltene → rare)
    "ing",   # nominalisation          (drilling → boring)
    "er",    # indefinite plural       (brønnstrømmer)
    "ar",    # nynorsk plural          (ventilaar)
    "en",    # definite singular       (brønnstrømmen)
    "et",    # definite singular neuter(systemet)
    "a",     # definite (some words)   (pumpa)
)

# Minimum stem length after stripping a suffix.
_MIN_STEM = 3


def norwegian_stem(word: str) -> str:
    """Strip common Norwegian inflectional suffixes to approximate a stem.

    >>> norwegian_stem("brønnstrømmene")
    'brønnstrømm'
    >>> norwegian_stem("brønnstrømmer")
    'brønnstrømm'
    >>> norwegian_stem("brønnstrømmen")
    'brønnstrømm'
    """
    w = word.lower().strip()
    for sfx in _SUFFIXES:
        if w.endswith(sfx) and len(w) - len(sfx) >= _MIN_STEM:
            return w[: -len(sfx)]
    return w


def norwegian_variants(stem: str) -> List[str]:
    """Generate probable Norwegian morphological forms from a stem.

    >>> sorted(norwegian_variants("brønnstrømm"))
    ['brønnstrømm', 'brønnstrømma', 'brønnstrømmane', 'brønnstrømmen',
     'brønnstrømmene', 'brønnstrømmer', 'brønnstrømmet']
    """
    return [
        stem,               # bare stem (may itself be a valid word)
        stem + "en",        # definite singular (common)
        stem + "et",        # definite singular (neuter)
        stem + "er",        # indefinite plural
        stem + "ar",        # nynorsk indefinite plural
        stem + "ene",       # definite plural
        stem + "ane",       # nynorsk definite plural
        stem + "a",         # definite (some dialects / words)
    ]


def words_share_stem(word_a: str, word_b: str) -> bool:
    """True if two Norwegian words reduce to the same stem.

    >>> words_share_stem("brønnstrømmer", "brønnstrømmen")
    True
    >>> words_share_stem("ventiler", "ventilen")
    True
    >>> words_share_stem("olje", "gass")
    False
    """
    return norwegian_stem(word_a) == norwegian_stem(word_b)


def stem_matches_in_text(keyword: str, text: str) -> bool:
    """Check if *any* morphological variant of `keyword` appears as a
    whole word in `text` (case-insensitive, Unicode-aware).

    This is the primary function used by the search engine to decide
    whether a transcript segment is 'anchored' to a query keyword.
    """
    stem = norwegian_stem(keyword)
    text_lower = text.lower()

    for variant in norwegian_variants(stem):
        # Unicode-aware whole-word check
        try:
            pattern = rf"(?<![\\w]){re.escape(variant)}(?![\\w])"
            if re.search(pattern, text_lower, flags=re.UNICODE):
                return True
        except re.error:
            if variant in text_lower:
                return True

    # Also try the original keyword as-is (handles already-stemmed or
    # unusual forms not covered by the suffix list).
    if keyword.lower() in text_lower:
        return True

    return False
