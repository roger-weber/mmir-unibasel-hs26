"""Ch06 helper - filtered search on an inverted file (IVF).

The index (k-means centers and one list of vector ids per center) and the helpers
probe, score, top_k, and search are provided. You implement four filtered search
strategies. Each one returns a numpy array of vector ids, best first.

Cost model: every similarity computed through probe() or score() adds 1 to
self.scored per vector or center. Reading the boolean `admitted` array is free.
"""
import numpy as np


class FilteredIVF:
    """An inverted file over unit-length vectors, ranked by dot product (= cosine)."""

    def __init__(self, X, n_list=256, iters=15, seed=0):
        """Cluster X into n_list lists with k-means."""
        self.X = X
        self.n_list = n_list
        self.centers, assign = self._kmeans(X, n_list, iters, seed)
        self.lists = [np.flatnonzero(assign == j) for j in range(n_list)]
        self.scored = 0

    @staticmethod
    def _kmeans(X, n_list, iters, seed):
        """Spherical k-means: assign by dot product, recompute centers as means."""
        rng = np.random.default_rng(seed)
        centers = X[rng.choice(len(X), n_list, replace=False)].copy()
        for _ in range(iters):
            assign = np.argmax(X @ centers.T, axis=1)
            for j in range(n_list):
                members = assign == j
                if members.any():
                    centers[j] = X[members].mean(axis=0)
        return centers, np.argmax(X @ centers.T, axis=1)

    # ─── Provided helpers ────────────────────────────────────────────────────

    def probe(self, q, n_probe):
        """Ids of the n_probe lists whose centers are most similar to q, best first."""
        self.scored += self.n_list
        return np.argsort(-(self.centers @ q), kind="stable")[:n_probe]

    def score(self, q, ids):
        """Dot products of q with the vectors `ids`."""
        self.scored += len(ids)
        return self.X[ids] @ q

    @staticmethod
    def top_k(ids, sims, k):
        """The k ids with the highest similarity, best first."""
        return ids[np.argsort(-sims, kind="stable")[:k]]

    def search(self, q, k, n_probe):
        """Plain IVF search without a filter: scan the n_probe nearest lists."""
        ids = np.concatenate([self.lists[j] for j in self.probe(q, n_probe)])
        return self.top_k(ids, self.score(q, ids), k)

    # ─── Your code: three ways to filter ─────────────────────────────────────

    def search_post(self, q, k, admitted, n_probe, fetch=None):
        """Post-filtering.

        Run the plain IVF search, then drop every candidate that is not admitted, and
        keep at most k of the rest in their order. Two variants:
          - fetch=None: the filter sees every candidate the plain search scored, that is
            all entries of the n_probe lists, ranked by similarity.
          - fetch=<int>: the filter sees only the top `fetch` candidates, as when the
            index is a black box that returns only `fetch` results to the application.
        The result may hold fewer than k ids.
        """
        # YOUR CODE HERE
        raise NotImplementedError

    def search_during(self, q, k, admitted, n_probe):
        """Filtering during the search.

        Scan the same n_probe lists as the plain search, but check every entry against
        `admitted` first and compute the similarity only for admitted ids. Return the
        top k of the admitted ids found in the probed lists.
        """
        # YOUR CODE HERE
        raise NotImplementedError

    def search_pre(self, q, k, admitted):
        """Pre-filtering.

        Ignore the lists. Compute the similarity of q with every admitted vector and
        return the exact top k among them.
        """
        # YOUR CODE HERE
        raise NotImplementedError

    # ─── Your code: choosing per query ───────────────────────────────────────

    def search_auto(self, q, k, admitted, n_probe):
        """Pick a filtering strategy for this query and return its top k.

        Requirements, checked by the notebook on all five filters with k = 10:
          - mean recall@10 against the exact filtered result of at least 0.9 per filter,
          - mean cost (self.scored per query) below 10% of a full scan per filter,
          - mean cost below 2,500 for the "8% of tenants" filter,
          - min(k, number of admitted vectors) results, all admitted.
        """
        # YOUR CODE HERE
        raise NotImplementedError
