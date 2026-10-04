"""Ch05 helper - Semantic Search over the movies-small collection.

The retrieval blocks (bm25, dense, reciprocal_rank_fusion, cross_encoder_rerank) are
provided and match the chapter demo. You implement run_pipeline, search, genre_facets,
and rating_facets. Each ranking is a list of (movie_id, score) pairs, best first.
"""
from collections import Counter

import numpy as np

from shared.text import tokenize, remove_stopwords
from shared.retrieval import document_frequencies, rank_collection_bm25


class MovieSearch:
    """A linear-scan movie search engine over a list of movie dicts."""

    def __init__(self, movies, encode_query, model_key="emb_mini", cross_encoder=None):
        """Index the movies for BM25 and remember the query encoder and cross-encoder."""
        self.movies = movies
        self.encode_query = encode_query          # str -> normalised np.ndarray
        self.model_key = model_key                # "emb_mini" or "emb_qwen"
        self.cross_encoder = cross_encoder
        self.id_to_movie = {m["id"]: m for m in movies}
        # BM25 document frequencies over the whole collection (idf is collection-wide).
        self.df = document_frequencies({m["id"]: m["tokens"] for m in movies})

    # ─── Provided retrieval blocks (same as the demo) ────────────────────────

    def bm25(self, query, candidates, top_k=10):
        """Rank `candidates` by BM25; keep positive scores only."""
        q_tokens = remove_stopwords(tokenize(query))
        corpus = {m["id"]: m["tokens"] for m in candidates}
        ranked = rank_collection_bm25(q_tokens, corpus, self.df)
        return [(doc_id, score) for doc_id, score in ranked if score > 0][:top_k]

    def dense(self, query, candidates, top_k=10):
        """Rank `candidates` by cosine similarity on the chosen embedding."""
        if not candidates:
            return []
        q_emb = self.encode_query(query)
        matrix = np.array([m[self.model_key] for m in candidates])
        sims = matrix @ q_emb
        idx = np.argsort(-sims)[:top_k]
        return [(candidates[i]["id"], float(sims[i])) for i in idx]

    @staticmethod
    def reciprocal_rank_fusion(ranked_lists, k=60):
        """Merge ranked lists using ranks only: RRF(d) = sum 1 / (k + rank)."""
        scores = {}
        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    def cross_encoder_rerank(self, query, candidate_ids, top_k=10):
        """Reorder candidate ids by the cross-encoder's joint relevance score."""
        if not candidate_ids or self.cross_encoder is None:
            return []
        pairs = [[query, self.id_to_movie[cid]["text"]] for cid in candidate_ids]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: -float(x[1]))
        return [(cid, float(score)) for cid, score in ranked[:top_k]]

    # ─── Your code: pipelines ────────────────────────────────────────────────

    def run_pipeline(self, query, pipeline, candidates, top_k=10, retrieve_k=50):
        """Run one of the four pipelines over `candidates` and return the top_k ranking.

        pipeline is one of:
          "A"  BM25 only.
          "B"  BM25 fetches retrieve_k candidates, the cross-encoder reranks them.
          "C"  dense bi-encoder retrieval only.
          "D"  BM25 and dense each fetch retrieve_k, fuse with RRF, then the
               cross-encoder reranks the fused pool.
        """
        # YOUR CODE HERE
        raise NotImplementedError

    # ─── Your code: filtered search ──────────────────────────────────────────

    def search(self, query, pipeline="D", predicate=None, top_k=10, retrieve_k=50):
        """Search the collection, optionally restricted to movies matching `predicate`.

        predicate is a function movie_dict -> bool, or None for the whole collection.
        Apply it as a pre-filter: build the candidate list first, then run the pipeline
        over only those movies, so the ranking never contains a movie the filter excludes.
        """
        # YOUR CODE HERE
        raise NotImplementedError

    # ─── Your code: facets ───────────────────────────────────────────────────

    def genre_facets(self, results, top_n=5):
        """Count how often each genre occurs across the result movies.

        results is a list of (movie_id, score) pairs. Return the top_n
        (genre, count) pairs, most frequent first.
        """
        # YOUR CODE HERE
        raise NotImplementedError

    def rating_facets(self, results):
        """Group the result movies into three rating bands.

        Return a dict with keys "<6", "6-8", ">8" and the count of result movies in
        each band (rating below 6, from 6 to 8 inclusive, above 8).
        """
        # YOUR CODE HERE
        raise NotImplementedError
