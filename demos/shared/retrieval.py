"""
Retrieval model building blocks.

Usage:
    from shared.retrieval import (
        document_frequencies, vocabulary,
        tfidf_vector, cosine_similarity, dot_product,
        bm25_score, boolean_and, boolean_or
    )
"""

import math
from collections import Counter


# ─── Corpus Statistics ───────────────────────────────────────────────────────

def term_frequencies(tokens: list[str]) -> dict[str, int]:
    """Term frequency counts for a single document."""
    return dict(Counter(tokens))


def document_frequencies(corpus: dict[str, list[str]]) -> dict[str, int]:
    """
    Document frequency for each term.

    Args:
        corpus: {doc_id: token_list}

    Returns:
        {term: number_of_documents_containing_term}
    """
    df = Counter()
    for tokens in corpus.values():
        df.update(set(tokens))
    return dict(df)


def vocabulary(df: dict[str, int]) -> list[str]:
    """Sorted vocabulary from document frequencies."""
    return sorted(df.keys())


def average_doc_length(corpus: dict[str, list[str]]) -> float:
    """Average number of tokens per document."""
    lengths = [len(tokens) for tokens in corpus.values()]
    return sum(lengths) / len(lengths) if lengths else 0.0


# ─── IDF Variants ────────────────────────────────────────────────────────────

def idf(term: str, df: dict[str, int], n: int) -> float:
    """Classic IDF: log(N / df)."""
    return math.log(n / df.get(term, 1))


def idf_bm25(term: str, df: dict[str, int], n: int) -> float:
    """BM25 IDF: log((N - df + 0.5) / (df + 0.5))."""
    d = df.get(term, 0)
    return math.log((n - d + 0.5) / (d + 0.5))


def idf_lucene(term: str, df: dict[str, int], n: int) -> float:
    """Lucene IDF: log(1 + (N - df + 0.5) / (df + 0.5))."""
    d = df.get(term, 0)
    return math.log(1 + (n - d + 0.5) / (d + 0.5))


# ─── Vector Space Model ──────────────────────────────────────────────────────

def tfidf_vector(tokens: list[str], df: dict[str, int], n: int, vocab: list[str]) -> list[float]:
    """
    TF-IDF vector for a document/query given a fixed vocabulary order.

    Args:
        tokens: Processed tokens for this document/query.
        df: Document frequency dict.
        n: Total number of documents.
        vocab: Ordered vocabulary list (defines vector dimensions).

    Returns:
        List of tf*idf values aligned with vocab.
    """
    tf = Counter(tokens)
    return [tf.get(t, 0) * idf(t, df, n) for t in vocab]


def dot_product(v1: list[float], v2: list[float]) -> float:
    """Inner product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def vector_norm(v: list[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two vectors."""
    n1 = vector_norm(v1)
    n2 = vector_norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot_product(v1, v2) / (n1 * n2)


# ─── BM25 ────────────────────────────────────────────────────────────────────

def bm25_score(query_tokens: list[str], doc_tokens: list[str],
               df: dict[str, int], n: int, adl: float,
               k: float = 1.2, b: float = 0.75) -> float:
    """
    BM25 score for a single query-document pair.

    Args:
        query_tokens: Processed query tokens.
        doc_tokens: Processed document tokens.
        df: Document frequency dict.
        n: Total number of documents.
        adl: Average document length.
        k: TF saturation parameter (default 1.2).
        b: Length normalization parameter (default 0.75).

    Returns:
        BM25 similarity score.
    """
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        if term not in tf:
            continue
        tf_val = tf[term]
        idf_val = idf_bm25(term, df, n)
        numerator = tf_val * (k + 1)
        denominator = tf_val + k * (1 - b + b * dl / adl)
        score += idf_val * numerator / denominator
    return score


# ─── Boolean Retrieval ───────────────────────────────────────────────────────

def boolean_and(query_terms: list[str], corpus: dict[str, list[str]]) -> list[str]:
    """
    Boolean AND: return doc_ids where ALL query terms are present.

    Args:
        query_terms: List of terms (already processed).
        corpus: {doc_id: token_list}

    Returns:
        List of matching doc_ids.
    """
    results = []
    for doc_id, tokens in corpus.items():
        token_set = set(tokens)
        if all(t in token_set for t in query_terms):
            results.append(doc_id)
    return results


def boolean_or(query_terms: list[str], corpus: dict[str, list[str]]) -> list[str]:
    """
    Boolean OR: return doc_ids where ANY query term is present.

    Args:
        query_terms: List of terms (already processed).
        corpus: {doc_id: token_list}

    Returns:
        List of matching doc_ids.
    """
    results = []
    for doc_id, tokens in corpus.items():
        token_set = set(tokens)
        if any(t in token_set for t in query_terms):
            results.append(doc_id)
    return results


# ─── Ranking Helpers ─────────────────────────────────────────────────────────

def rank_collection_vsm(query_tokens: list[str], corpus: dict[str, list[str]],
                        df: dict[str, int], vocab: list[str]) -> list[tuple[str, float]]:
    """
    Rank all documents by cosine similarity (TF-IDF VSM).

    Returns:
        List of (doc_id, score) sorted by decreasing score.
    """
    n = len(corpus)
    query_vec = tfidf_vector(query_tokens, df, n, vocab)
    results = []
    for doc_id, tokens in corpus.items():
        doc_vec = tfidf_vector(tokens, df, n, vocab)
        score = cosine_similarity(query_vec, doc_vec)
        results.append((doc_id, score))
    results.sort(key=lambda x: -x[1])
    return results


def rank_collection_bm25(query_tokens: list[str], corpus: dict[str, list[str]],
                         df: dict[str, int], k: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
    """
    Rank all documents by BM25.

    Returns:
        List of (doc_id, score) sorted by decreasing score.
    """
    n = len(corpus)
    adl = average_doc_length(corpus)
    results = []
    for doc_id, tokens in corpus.items():
        score = bm25_score(query_tokens, tokens, df, n, adl, k, b)
        results.append((doc_id, score))
    results.sort(key=lambda x: -x[1])
    return results
