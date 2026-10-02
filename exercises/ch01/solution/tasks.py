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
        self.k1 = k1
        self.b = b
        self.tokens = {doc_id: extract_terms(text) for doc_id, text in documents.items()}
        self.extract_terms = extract_terms
        self.N = len(self.tokens)
        self.df = Counter()
        for toks in self.tokens.values():
            self.df.update(set(toks))
        total = sum(len(toks) for toks in self.tokens.values())
        self.avgdl = total / self.N if self.N else 0.0

    def idf(self, term: str) -> float:
        """Positive (Lucene) BM25 IDF for a term."""
        d = self.df.get(term, 0)
        return math.log(1 + (self.N - d + 0.5) / (d + 0.5))

    def _score(self, query_terms: list[str], doc_id: str) -> float:
        """BM25 score of one document for a list of already-extracted query terms."""
        toks = self.tokens[doc_id]
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in set(query_terms):
            if term not in tf:
                continue
            f = tf[term]
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf(term) * num / denom
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs, score > 0, sorted by -score then id."""
        query_terms = self.extract_terms(query)
        if not query_terms:
            return []
        scored = [(doc_id, self._score(query_terms, doc_id)) for doc_id in self.tokens]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
