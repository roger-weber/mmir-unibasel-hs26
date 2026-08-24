"""
Text processing pipeline for retrieval demos.

Usage:
    from shared.text import tokenize, remove_stopwords, stem_tokens, pipeline, bag_of_words
    from shared.text import stopwords_for
"""

import re
from collections import Counter


# ─── Stopwords ──────────────────────────────────────────────────────────────

# Map language codes to NLTK corpus names
_LANG_TO_NLTK = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "ru": "russian",
    "ar": "arabic",
    "fi": "finnish",
    "sv": "swedish",
    "no": "norwegian",
    "da": "danish",
    "hu": "hungarian",
    "ro": "romanian",
    "tr": "turkish",
    "id": "indonesian",
}

# Cache for loaded stopword sets
_stopwords_cache: dict[str, set[str]] = {}


def stopwords_for(language: str) -> set[str]:
    """
    Get stopwords for a language.

    Args:
        language: ISO 639-1 code ("en", "de", "fr", "es", etc.)
                  or full NLTK name ("english", "german", ...).

    Returns:
        Set of stopword strings. Returns empty set for unsupported languages.
    """
    # Normalize: accept full name
    if len(language) > 3:
        for code, name in _LANG_TO_NLTK.items():
            if name == language.lower():
                language = code
                break

    if language in _stopwords_cache:
        return _stopwords_cache[language]

    # Load from NLTK
    nltk_name = _LANG_TO_NLTK.get(language)
    if nltk_name:
        try:
            from nltk.corpus import stopwords as nltk_stopwords
            words = set(nltk_stopwords.words(nltk_name))
            _stopwords_cache[language] = words
            return words
        except (LookupError, OSError):
            pass

    # Unsupported language — return empty set (no filtering)
    _stopwords_cache[language] = set()
    return set()


# ─── Tokenization ──────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters."""
    return [t for t in re.split(r'\W+', text.lower()) if t]


def remove_stopwords(tokens: list[str], language: str = "en") -> list[str]:
    """
    Remove stopwords from a token list.

    Args:
        tokens: List of tokens (already lowercased).
        language: Language code ("en", "de", "fr", ...),
                  "all" or "mixed" for all languages combined.

    Returns:
        Filtered token list.
    """
    if language in ("all", "mixed"):
        sw = _all_stopwords()
    else:
        sw = stopwords_for(language)
    return [t for t in tokens if t not in sw]


def _all_stopwords() -> set[str]:
    """Combined stopwords from all supported languages."""
    if "all" in _stopwords_cache:
        return _stopwords_cache["all"]
    combined = set()
    for lang in _LANG_TO_NLTK:
        combined.update(stopwords_for(lang))
    _stopwords_cache["all"] = combined
    return combined


# ─── Stemming ──────────────────────────────────────────────────────────────

def stem(token: str) -> str:
    """Stem a single token using the Porter stemmer."""
    return _get_porter().stem(token)


_porter_instance = None

def _get_porter():
    global _porter_instance
    if _porter_instance is None:
        from nltk.stem import PorterStemmer
        _porter_instance = PorterStemmer()
    return _porter_instance


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply stemming to all tokens."""
    return [stem(t) for t in tokens]


# ─── Convenience ───────────────────────────────────────────────────────────

def bag_of_words(tokens: list[str]) -> dict[str, int]:
    """Token frequency counts."""
    return dict(Counter(tokens))


def pipeline(text: str, language: str = "en", use_stemming: bool = True,
             use_stopwords: bool = True) -> list[str]:
    """
    Full extraction pipeline: tokenize → stopwords → stem.

    Args:
        text: Raw document or query string.
        language: Language code for stopwords ("en", "de", "fr", ...).
        use_stemming: Apply suffix stripping (Porter, English only).
        use_stopwords: Remove stopwords.

    Returns:
        List of processed tokens.
    """
    tokens = tokenize(text)
    if use_stopwords:
        tokens = remove_stopwords(tokens, language)
    if use_stemming:
        tokens = stem_tokens(tokens)
    return tokens
