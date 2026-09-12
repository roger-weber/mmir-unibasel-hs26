"""Ch01 helper - text-extraction primitives and a BM25 search engine.

The exercise composes the text-processing functions below into a pipeline,
then asks you to implement BM25Scorer.
"""
import math
import re
from collections import Counter

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', 'for', 'is',
    'are', 'was', 'were', 'at', 'by', 'on', 'with', 'that', 'this',
    'it', 'from', 'as', 'be', 'has', 'have', 'had', 'not', 'but',
    'its', 'can', 'do', 'does', 'did', 'will', 'would', 'should',
    'may', 'might', 'he', 'she', 'they', 'we', 'you', 'i', 'me',
}


# ─── Text-processing primitives ──────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Split text into raw tokens on non-word boundaries. Case is preserved."""
    return [t for t in re.split(r"\W+", text) if t]


def fold_case(tokens: list[str]) -> list[str]:
    """Lowercase every token."""
    return [t.lower() for t in tokens]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop very common English words (a, the, it, of, ...)."""
    return [t for t in tokens if t.lower() not in STOPWORDS]


def stem(tokens: list[str]) -> list[str]:
    """Reduce each token to its Porter stem, e.g. 'toys' -> 'toy', 'running' -> 'run'."""
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]


# ─── BM25 search engine ──────────────────────────────────────────────────────

class BM25Scorer:
    """BM25 ranking over {doc_id: raw_text} documents, using the positive (Lucene) IDF."""

    def __init__(self, documents: dict[str, str], extract_terms, k1: float = 1.2, b: float = 0.75):
        """Extract terms for every document and precompute df, N, and avgdl."""
        # YOUR CODE HERE
        raise NotImplementedError

    def idf(self, term: str) -> float:
        """Positive (Lucene) BM25 IDF for a term."""
        # YOUR CODE HERE
        raise NotImplementedError

    def _score(self, query_terms: list[str], doc_id: str) -> float:
        """BM25 score of one document for a list of already-extracted query terms."""
        # YOUR CODE HERE
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs, score > 0, sorted by -score then id."""
        # YOUR CODE HERE
        raise NotImplementedError
