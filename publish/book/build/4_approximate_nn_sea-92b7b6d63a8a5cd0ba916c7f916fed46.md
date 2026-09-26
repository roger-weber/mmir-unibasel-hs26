---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Vector Search
section: Approximate Nearest-Neighbor Search
order: "6.4"
---

(vector-search-approximate-nn)=
# Approximate Nearest-Neighbor Search

The previous section left us with a dilemma. Exact nearest-neighbor search in high dimensions costs a full scan, and the distances it computes barely separate good matches from bad ones. Approximate nearest-neighbor (ANN) search resolves the dilemma by accepting a small, controlled loss of accuracy in exchange for a large gain in speed. This section covers the main families of ANN methods, how a modern library combines them, and how to measure the trade-off they make.

## Trading exactness for speed

The case for approximation is practical. In an embedding-based system, such as retrieval for a language model, what matters is retrieving a handful of passages that together answer the query, not the geometrically closest vector. A document ranked third is usually as useful as the first. Empirically, good ANN algorithms recover well over ninety percent of the true neighbors while running orders of magnitude faster than an exact scan. [Figure %s](#fig-exact-vs-ivf) contrasts the two: exact search compares the query to every vector, while an approximate method restricts the comparison to a promising region.

```{figure} images/figure_6_4.png
:name: fig-exact-vs-ivf
:width: 90%

Exact search compares the query to every vector (left). An approximate method partitions the space into clusters and searches only those near the query (right), trading a little recall for a large speed-up.
```

## Reducing the data before searching

One route to speed is to make each vector cheaper to compare. Classical dimensionality reduction, such as principal component analysis (PCA) or singular value decomposition (SVD), projects vectors onto the few directions that carry most of the variance. Newer embedding designs bake this in: Matryoshka embeddings arrange information so that a prefix of the vector is already a usable, lower-dimensional embedding. **Quantization** goes further and stores each component with only a few bits. The approximation error is small, and in practice it rarely disturbs the ordering of neighbors enough to change the top results. Reduction alone still reads every vector, though, so it is usually combined with a method that also reads fewer vectors.

## A composite index

No single trick wins on its own, so production libraries compose several. Facebook AI Similarity Search (FAISS) is the standard example, and it organizes a search into four stages that later methods slot into.

- **Vector transformers** prepare vectors: L2-normalization for cosine search, PCA to reduce dimensions, or padding to fit a later stage.
- **Coarse quantizers** cut the amount of data considered by selecting a promising subset of vectors, for example a few clusters.
- **Fine quantizers** cut the cost of each comparison by encoding vectors compactly so that distances can be estimated quickly.
- **Refiners** re-rank the survivors, typically by computing exact distances on the top candidates or by applying metadata filters.

The two middle stages carry the ANN methods. The coarse quantizers reduce *how many* vectors are examined; the fine quantizers reduce *how expensive* each examination is. A strong index often stacks them: cluster first, then encode each cluster compactly, then refine the best few.

## Coarse quantizers: clustering and graphs

The **inverted file** (IVF) borrows the inverted-index idea from text retrieval. A clustering algorithm, usually k-means, chooses $n$ centers and assigns every vector to its nearest center, as in [Figure %s](#fig-ivf). The centers act like index terms: a query is compared to the $n$ centers, and the search then scans only the list of the nearest center. This cuts the work by roughly a factor of $n$, at the risk of missing a neighbor that sits just across a cluster boundary. Probing the $m$ nearest clusters instead of one recovers most of those misses at $m$ times the cost, which is the tuning knob of IVF.

```{figure} images/figure_6_28.png
:name: fig-ivf
:width: 55%

An inverted file clusters the data around centers. The query is routed to its nearest center $\mathbf{c}_2$, and only that cluster's list is scanned; probing more clusters trades speed for recall.
```

A different coarse quantizer navigates a graph. A **navigable small-world** graph connects each vector to a few near neighbors plus a handful of long-range shortcuts, so that any two nodes are linked by a short path. Search starts at an entry point and greedily steps to whichever neighbor is closer to the query until no neighbor improves, as traced in [Figure %s](#fig-hnsw). The **hierarchical** version, HNSW, stacks several such graphs: sparse upper layers make long jumps toward the right region, and the dense base layer refines the result. HNSW is among the fastest ANN methods and appears in most vector databases.

```{figure} images/figure_6_29.png
:name: fig-hnsw
:width: 55%

Greedy search on a navigable small-world graph. From the entry point, each step moves to the neighbor closest to the query $\mathbf{q}$ until no neighbor is closer.
```

## Fine quantizers: hashing and product quantization

**Locality-sensitive hashing** (LSH) encodes a vector's position with a bit string. Each of $n$ hyperplanes contributes one bit, set by which side of the plane the vector falls on, as shown in [Figure %s](#fig-lsh). Nearby points tend to share bits, so the Hamming distance between two bit strings approximates their angular distance. Comparing compact bit strings is far cheaper than comparing full vectors: a $d$-dimensional float vector shrinks from $4d$ bytes to $n/8$ bytes, with a matching speed-up. When many points share a code, an exact re-ranking step resolves the ties.

```{figure} images/figure_6_30.png
:name: fig-lsh
:width: 50%

Locality-sensitive hashing. Three hyperplanes split the space; each point, and the query $\mathbf{q}$, gets a bit per plane, and similar points receive similar codes.
```

**Product quantization** (PQ) compresses more aggressively. It splits a vector into $m$ sub-vectors and quantizes each sub-vector separately against its own small codebook, so the whole vector becomes a short sequence of codebook indices, as in [Figure %s](#fig-pq). An eight-dimensional vector can shrink to a single byte. At query time, the distance from the query to every codebook entry is precomputed once, and each stored vector's distance is then a fast sum of table look-ups. PQ pairs naturally with IVF: cluster first, then store each cluster's vectors as product-quantized codes.

```{figure} images/figure_6_31.png
:name: fig-pq
:width: 80%

Product quantization splits a vector into sub-vectors and replaces each with the index of its nearest codebook entry, compressing the vector into a few bytes.
```

## Measuring the trade-off

Every ANN method exposes parameters that trade recall for speed: the number of probed clusters in IVF, the search width in HNSW, the number of bits in LSH. To compare methods fairly, we measure both axes at once. The quality axis is recall against the exact result.

```{important} Key Formula: Recall@k for Approximate Search

$$\text{recall@}k = \frac{\lvert \text{approximate top-}k \cap \text{exact top-}k \rvert}{k}$$

The fraction of the true $k$ nearest neighbors that the approximate search also returned; $1.0$ means the approximation matched exact search on this query.
```

The speed axis is throughput, in queries per second. Plotting throughput against recall traces a curve for each method, and the public benchmark at [ann-benchmarks.com](https://ann-benchmarks.com) collects these curves across many libraries on standard datasets. [Figure %s](#fig-ann-benchmarks) shows one such comparison. Reading it is straightforward: fix the recall you need, then pick the method with the highest throughput at that recall. A method whose curve sits above and to the right of another dominates it. There is no single best method, only the best operating point for a given recall target and dataset, which is exactly what the plot lets you choose.

```{figure} images/figure_6_32.png
:name: fig-ann-benchmarks
:width: 80%

Recall versus queries-per-second for many ANN methods on a standard dataset. Up and to the right is better; at a chosen recall, the highest curve is the fastest method.
```

```{hint} Hands-on: Approximate nearest-neighbor search
Build simplified versions of IVF, LSH, and product quantization, watch greedy search navigate an HNSW graph, and compare FAISS index configurations on the recall-throughput frontier.
[Open notebook ->](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch06-02-approximate-nn.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```
