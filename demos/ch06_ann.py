"""
Approximate nearest-neighbor search as three composable components.

A retriever answers "which stored vectors best match the query?" in three steps:

1. **Matryoshka truncation**: keep only the first `dim` components of every
   embedding (and renormalize). Fewer dimensions make every comparison cheaper.
2. **Coarse filter**: decide *which* vectors to examine. `AllVectors` examines
   every vector, `IVF` only the lists of the nearest cluster centers, `HNSW` only
   the vectors a greedy graph walk visits.
3. **Fine filter**: decide *how cheaply* to compare them. `Exact` uses the float
   vectors, `PQ` product-quantized codes, `LSH` bit strings and Hamming distance.

All vectors are unit length, so the dot product equals cosine similarity.

Cost is reported per query as the number of vectors examined and the bytes read,
which is machine-independent, unlike wall-clock time.

Usage:
    from ch06_ann import truncate, Retriever, IVF, PQ

    X = truncate(doc_embeddings, 256)
    r = Retriever(IVF(nlist=1024, nprobe=4), PQ(m=32)).fit(X)
    ids, cost = r.search(truncate(query_embedding, 256), k=50)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── 1. Matryoshka truncation ────────────────────────────────────────────────

def truncate(E: np.ndarray, dim: int) -> np.ndarray:
    """Keep the first `dim` components and rescale every vector to unit length."""
    X = np.ascontiguousarray(E[..., :dim], dtype=np.float32)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ─── Cost accounting ─────────────────────────────────────────────────────────

@dataclass
class Cost:
    """Work done for one query."""
    vectors: int = 0      # stored vectors scored by the fine filter
    bytes: int = 0        # bytes of vector data read (coarse + fine)

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.vectors + other.vectors, self.bytes + other.bytes)


# ─── 2. Coarse filters: which vectors to examine ─────────────────────────────

class AllVectors:
    """No filtering: every stored vector is a candidate (a full scan)."""
    name = "all"

    def fit(self, X: np.ndarray) -> "AllVectors":
        self.n = len(X)
        return self

    def candidates(self, q: np.ndarray) -> tuple[np.ndarray, Cost]:
        return np.arange(self.n), Cost()


class IVF:
    """Inverted file: k-means centers as index terms, one list per center.

    The query is compared with all centers; the lists of the `nprobe` best
    centers form the candidate set.
    """

    def __init__(self, nlist: int = 1024, nprobe: int = 4, seed: int = 0):
        self.nlist, self.nprobe, self.seed = nlist, nprobe, seed
        self.name = f"IVF{nlist} nprobe={nprobe}"

    def fit(self, X: np.ndarray) -> "IVF":
        import faiss
        km = faiss.Kmeans(X.shape[1], self.nlist, niter=20, seed=self.seed,
                          spherical=True, verbose=False)
        km.train(X)
        self.centers = km.centroids.astype(np.float32)
        assign = np.argmax(X @ self.centers.T, axis=1)
        order = np.argsort(assign, kind="stable")
        bounds = np.searchsorted(assign[order], np.arange(self.nlist + 1))
        self.lists = [order[bounds[c]:bounds[c + 1]] for c in range(self.nlist)]
        return self

    def with_nprobe(self, nprobe: int) -> "IVF":
        """Same trained index, different number of probed lists."""
        other = object.__new__(IVF)
        other.__dict__.update(self.__dict__)
        other.nprobe, other.name = nprobe, f"IVF{self.nlist} nprobe={nprobe}"
        return other

    def candidates(self, q: np.ndarray) -> tuple[np.ndarray, Cost]:
        best = np.argsort(-(self.centers @ q))[:self.nprobe]
        cand = np.concatenate([self.lists[c] for c in best])
        return cand, Cost(0, self.nlist * self.centers.shape[1] * 4)


# ─── 3. Fine filters: how cheaply to compare ─────────────────────────────────

class Exact:
    """Score candidates with their float32 vectors."""
    name = "exact"

    def fit(self, X: np.ndarray) -> "Exact":
        self.X = X
        self.bytes_per_vector = X.shape[1] * 4
        return self

    def scores(self, q: np.ndarray, cand: np.ndarray) -> np.ndarray:
        if len(cand) == len(self.X):          # full scan: no need to gather rows
            return self.X @ q
        return self.X[cand] @ q


class PQ:
    """Product quantization with `m` one-byte codes per vector.

    The vector is split into `m` sub-vectors; each is replaced by the index of
    its nearest center in a 256-entry codebook for that sub-space. A query first
    computes the dot product of each query sub-vector with every center
    (an m x 256 table); a stored vector's score is then a sum of m look-ups.
    """

    def __init__(self, m: int = 32, train_size: int = 50_000, seed: int = 0):
        self.m, self.train_size, self.seed = m, train_size, seed
        self.name = f"PQ {m} B"

    def fit(self, X: np.ndarray) -> "PQ":
        import faiss
        d = X.shape[1]
        assert d % self.m == 0, f"dimension {d} not divisible by m={self.m}"
        rng = np.random.default_rng(self.seed)
        train = X[rng.choice(len(X), min(self.train_size, len(X)), replace=False)]
        pq = faiss.ProductQuantizer(d, self.m, 8)
        pq.train(train)
        self.codes = pq.compute_codes(X).reshape(len(X), self.m)
        self.centers = faiss.vector_to_array(pq.centroids).reshape(self.m, 256, d // self.m)
        self.dsub = d // self.m
        self.bytes_per_vector = self.m
        return self

    def scores(self, q: np.ndarray, cand: np.ndarray) -> np.ndarray:
        table = np.einsum("mkd,md->mk", self.centers, q.reshape(self.m, self.dsub))
        return table[np.arange(self.m), self.codes[cand]].sum(axis=1)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


class LSH:
    """Random-hyperplane hashing: one bit per hyperplane, compared by Hamming distance.

    The expected Hamming distance between two codes is nbits * angle / pi, so a
    smaller Hamming distance means a smaller angle (a higher cosine).
    """

    def __init__(self, nbits: int = 512, seed: int = 0):
        self.nbits, self.seed = nbits, seed
        self.name = f"LSH {nbits} bit"

    def fit(self, X: np.ndarray) -> "LSH":
        rng = np.random.default_rng(self.seed)
        self.planes = rng.standard_normal((X.shape[1], self.nbits)).astype(np.float32)
        self.codes = np.packbits(X @ self.planes > 0, axis=1)
        self.bytes_per_vector = self.nbits // 8
        return self

    def scores(self, q: np.ndarray, cand: np.ndarray) -> np.ndarray:
        qc = np.packbits(q @ self.planes > 0)
        hamming = _POPCOUNT[np.bitwise_xor(self.codes[cand], qc)].sum(axis=1)
        return -hamming.astype(np.float32)


# ─── HNSW: graph search as a coarse filter ───────────────────────────────────

class HNSW:
    """Hierarchical navigable small-world graph (FAISS).

    The graph walk itself computes the scores, so HNSW combines a coarse and a
    fine filter: with `fine=None` it walks on float vectors, with `fine=PQ(m)`
    on product-quantized codes (FAISS IndexHNSWPQ).
    """

    def __init__(self, M: int = 32, ef: int = 64, pq_m: int | None = None):
        self.M, self.ef, self.pq_m = M, ef, pq_m
        self.name = f"HNSW M={M} ef={ef}"

    def fit(self, X: np.ndarray) -> "HNSW":
        import faiss
        d = X.shape[1]
        if self.pq_m is None:
            self.index = faiss.IndexHNSWFlat(d, self.M, faiss.METRIC_INNER_PRODUCT)
            self.bytes_per_vector = d * 4
        else:
            self.index = faiss.IndexHNSWPQ(d, self.pq_m, self.M, 8, faiss.METRIC_INNER_PRODUCT)
            self.index.train(X)
            self.bytes_per_vector = self.pq_m
        self.index.add(X)
        return self

    def with_ef(self, ef: int) -> "HNSW":
        other = object.__new__(HNSW)
        other.__dict__.update(self.__dict__)
        other.ef, other.name = ef, f"HNSW M={self.M} ef={ef}"
        return other

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, Cost]:
        import faiss
        self.index.hnsw.efSearch = max(self.ef, k)
        faiss.cvar.hnsw_stats.reset()
        _, I = self.index.search(q.reshape(1, -1), k)
        ndis = int(faiss.cvar.hnsw_stats.ndis)
        return I[0][I[0] >= 0], Cost(ndis, ndis * self.bytes_per_vector)


# ─── Composition ─────────────────────────────────────────────────────────────

class Retriever:
    """A coarse filter followed by a fine filter, returning the top-k vector indices."""

    def __init__(self, coarse, fine=None):
        self.coarse, self.fine = coarse, fine
        if isinstance(coarse, HNSW):
            fine_name = f"PQ {coarse.pq_m} B" if coarse.pq_m else "exact"
        else:
            fine_name = fine.name
        self.name = f"{coarse.name} + {fine_name}"

    def fit(self, X: np.ndarray) -> "Retriever":
        self.coarse.fit(X)
        if self.fine is not None:
            self.fine.fit(X)
        return self

    def search(self, q: np.ndarray, k: int = 50) -> tuple[np.ndarray, Cost]:
        if isinstance(self.coarse, HNSW):
            return self.coarse.search(q, k)
        cand, cost = self.coarse.candidates(q)
        s = self.fine.scores(q, cand)
        top = np.argpartition(-s, min(k, len(s) - 1))[:k] if len(s) > k else np.arange(len(s))
        top = top[np.argsort(-s[top])]
        return cand[top], cost + Cost(len(cand), len(cand) * self.fine.bytes_per_vector)


# ─── Evaluating the offline benchmark ────────────────────────────────────────

BENCHMARK_DIR = "eval/ch06_ann"
REFERENCE = "d=1024 | all + exact"   # exact search on full-length embeddings


def load_benchmark(folder: str = BENCHMARK_DIR) -> tuple[dict, dict]:
    """Return (meta, runs) from the offline benchmark artifacts."""
    import json
    from pathlib import Path
    data = json.loads((Path(folder) / "runs.json").read_text(encoding="utf-8"))
    return data["meta"], data["runs"]


def evaluate(runs: dict, assessments, reranker: str = "qwen3", mode: str = "standard",
             k: int = 10) -> dict[str, dict]:
    """Per-configuration measures, each a per-topic numpy array (except costs).

    recall10   overlap of the retrieved top 10 with exact search (the ann-benchmarks view)
    ndcg_ret   nDCG@10 of the retriever's own ranking
    ndcg_rr    nDCG@10 after the reranker reorders the retrieved top 50
    judged     fraction of the reranked top 10 that carries a judgment
    """
    from shared.assessments import ndcg_at_k, judged_at_k, overlap_at_k
    qids = assessments.query_ids()
    ref = runs[REFERENCE]["retrieved"]
    out = {}
    for name, r in runs.items():
        rr = r["reranked"][reranker]
        out[name] = {
            "dim": r["dim"], "coarse": r["coarse"], "fine": r["fine"], "bytes": r["bytes"],
            "recall10": np.array([overlap_at_k(r["retrieved"][q], ref[q], k) for q in qids]),
            "ndcg_ret": np.array([ndcg_at_k(r["retrieved"][q], assessments.qrels[q], k, mode) for q in qids]),
            "ndcg_rr": np.array([ndcg_at_k(rr[q], assessments.qrels[q], k, mode) for q in qids]),
            "judged": np.array([judged_at_k(rr[q], assessments.qrels[q], k) for q in qids]),
        }
    return out


def bootstrap_ci(diff: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval for the mean of per-topic differences."""
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
