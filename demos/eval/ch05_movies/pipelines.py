"""
Reusable retrieval pipelines for the chapter 5 semantic-search demo.

The book organises semantic search into five pipeline variants. This module
implements all five as small, composable functions so both the offline trial
runner and the eventual student notebook import the same code:

    A  bm25_search                 BM25 only (sparse keyword retrieval)
    B  bm25_cross_encoder          BM25 retrieve, then cross-encoder rerank
    C  dense_search                dense bi-encoder retrieval (cosine)
    D  hybrid_rrf                  BM25 + dense fused with Reciprocal Rank Fusion
    E  multistage_matryoshka       cheap truncated first stage, full-dim rerank

Index building is factored into build_bm25_index and encode_corpus so the
one-time cost is paid once and reused across queries and conditions.

All dense functions assume embeddings are L2-normalised, so a dot product
equals cosine similarity (the measure the book uses throughout). BM25 comes
from shared.retrieval; tokenisation from shared.text.
"""

from __future__ import annotations

import numpy as np

from shared.text import tokenize, remove_stopwords
from shared.retrieval import document_frequencies, rank_collection_bm25


# ─── Index building (one-time cost) ──────────────────────────────────────────

def build_bm25_index(collection) -> dict:
    """
    Build the sparse BM25 index: a tokenised corpus and document frequencies.

    Args:
        collection: iterable of retrieval units, each a dict with "id" and "text".

    Returns:
        {"corpus": {doc_id: tokens}, "df": {term: doc_count}, "n_terms": int}
    """
    corpus = {d["id"]: remove_stopwords(tokenize(d["text"])) for d in collection}
    df = document_frequencies(corpus)
    return {"corpus": corpus, "df": df, "n_terms": len(df)}


def encode_corpus(model, texts: list[str]) -> np.ndarray:
    """
    Encode documents into a normalised float32 embedding matrix.

    Documents are encoded with no prompt (symmetric for MiniLM, the plain
    document side for Qwen). Normalisation makes dot product equal cosine.
    """
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


# ─── Pipeline A: BM25 only ────────────────────────────────────────────────────

def bm25_search(query: str, bm25_index: dict, top_k: int = 10) -> list[tuple[str, float]]:
    """Rank by BM25; keep only positive scores. Returns (doc_id, score) pairs."""
    q_tokens = remove_stopwords(tokenize(query))
    ranked = rank_collection_bm25(q_tokens, bm25_index["corpus"], bm25_index["df"])
    return [(doc_id, score) for doc_id, score in ranked if score > 0][:top_k]


# ─── Cross-encoder reranking (used by B, optionally by D and E) ───────────────

def cross_encoder_rerank(query: str, candidate_ids: list[str],
                         id_to_text: dict[str, str], cross_encoder,
                         top_k: int = 10) -> list[tuple[str, float]]:
    """
    Rerank candidate documents with a cross-encoder relevance model.

    The cross-encoder scores each (query, document) pair jointly. It is the most
    precise and most expensive stage, so it only ever runs on a short candidate
    list produced by a cheaper first stage.
    """
    if not candidate_ids:
        return []
    pairs = [[query, id_to_text[cid]] for cid in candidate_ids]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: -float(x[1]))
    return [(cid, float(score)) for cid, score in ranked[:top_k]]


# ─── Pipeline B: BM25 retrieve then cross-encoder rerank ──────────────────────

def bm25_cross_encoder(query: str, bm25_index: dict, id_to_text: dict[str, str],
                       cross_encoder, retrieve_k: int = 50,
                       top_k: int = 10) -> list[tuple[str, float]]:
    """BM25 fetches a candidate pool; the cross-encoder reorders it precisely."""
    candidates = [doc_id for doc_id, _ in bm25_search(query, bm25_index, top_k=retrieve_k)]
    return cross_encoder_rerank(query, candidates, id_to_text, cross_encoder, top_k)


# ─── Pipeline C: dense bi-encoder retrieval ──────────────────────────────────

def dense_search(query: str, model, doc_embeddings: np.ndarray, doc_ids: list[str],
                 prompt_name: str | None = None, top_k: int = 10) -> list[tuple[str, float]]:
    """
    Retrieve top-k documents by cosine similarity on normalised embeddings.

    Args:
        prompt_name: name of the query prompt the model expects (Qwen uses
            "query"; MiniLM takes the raw query, so pass None).
    """
    kwargs = {"normalize_embeddings": True, "show_progress_bar": False}
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    q_emb = np.asarray(model.encode([query], **kwargs)[0], dtype=np.float32)
    sims = doc_embeddings @ q_emb
    idx = np.argsort(-sims)[:top_k]
    return [(doc_ids[i], float(sims[i])) for i in idx]


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]],
                           k: int = 60) -> list[tuple[str, float]]:
    """
    Merge ranked lists using only ranks, not raw scores.

    RRF(d) = sum over lists of 1 / (k + rank_of_d). The constant k (the book
    uses 60) dampens the influence of the very top ranks.
    """
    rrf_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: -x[1])


# ─── Pipeline D: hybrid BM25 + dense fused with RRF ───────────────────────────

def hybrid_rrf(query: str, bm25_index: dict, model, doc_embeddings: np.ndarray,
               doc_ids: list[str], prompt_name: str | None = None,
               retrieve_k: int = 50, k_rrf: int = 60,
               id_to_text: dict[str, str] | None = None, cross_encoder=None,
               top_k: int = 10) -> list[tuple[str, float]]:
    """
    Run BM25 and dense retrieval in parallel and fuse with RRF.

    If a cross_encoder (and id_to_text) is supplied, the fused candidate pool is
    reranked by the cross-encoder as a final precision stage.
    """
    bm = bm25_search(query, bm25_index, top_k=retrieve_k)
    de = dense_search(query, model, doc_embeddings, doc_ids, prompt_name, top_k=retrieve_k)
    fused = reciprocal_rank_fusion([bm, de], k=k_rrf)
    if cross_encoder is not None and id_to_text is not None:
        candidates = [doc_id for doc_id, _ in fused[:retrieve_k]]
        return cross_encoder_rerank(query, candidates, id_to_text, cross_encoder, top_k)
    return fused[:top_k]


# ─── Pipeline E: multi-stage retrieval with Matryoshka truncation ─────────────

def truncate_normalize(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Keep the first `dim` components of each row and renormalise to unit length."""
    sub = embeddings[:, :dim]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (sub / norms).astype(np.float32)


def multistage_matryoshka(query: str, model, doc_embeddings_full: np.ndarray,
                          doc_ids: list[str], prompt_name: str | None = None,
                          first_dim: int = 256, first_k: int = 50,
                          id_to_text: dict[str, str] | None = None, cross_encoder=None,
                          top_k: int = 10) -> list[tuple[str, float]]:
    """
    Cheap truncated first stage, then a precise full-dimension rerank.

    Stage 1 scans the whole collection with embeddings truncated to `first_dim`
    (cheap, since the model is Matryoshka-trained) and keeps the top `first_k`.
    Stage 2 reranks those survivors with the full-dimension embeddings. An
    optional cross-encoder adds a final precision stage.

    The embedding model must be Matryoshka-capable (truncating a prefix must
    still give a usable embedding); in this demo that is the Qwen model.
    """
    kwargs = {"normalize_embeddings": True, "show_progress_bar": False}
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    q_full = np.asarray(model.encode([query], **kwargs)[0], dtype=np.float32)

    # Stage 1: truncated prefix over the full collection.
    q1 = q_full[:first_dim]
    q1 = q1 / (np.linalg.norm(q1) or 1.0)
    docs1 = truncate_normalize(doc_embeddings_full, first_dim)
    sims1 = docs1 @ q1
    cand_idx = np.argsort(-sims1)[:first_k]

    # Stage 2: full-dimension rerank of the survivors.
    sub = doc_embeddings_full[cand_idx]
    sims2 = sub @ q_full
    order = np.argsort(-sims2)
    reranked = [(doc_ids[cand_idx[i]], float(sims2[i])) for i in order]

    if cross_encoder is not None and id_to_text is not None:
        candidates = [doc_id for doc_id, _ in reranked[:first_k]]
        return cross_encoder_rerank(query, candidates, id_to_text, cross_encoder, top_k)
    return reranked[:top_k]
