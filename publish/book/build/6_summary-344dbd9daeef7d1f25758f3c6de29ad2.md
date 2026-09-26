---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Vector Search
section: Summary
order: "6.6"
---

(vector-search-summary)=
# Summary

## Method Comparison

The chapter's competing search methods trade recall, speed, and memory in different ways. Exact methods guarantee the true neighbors; the approximate methods relax that guarantee to run at scale.

| Method | Core idea | What it reduces | Recall knob | Memory | Best for |
|---|---|---|---|---|---|
| Flat (exact) | Scan every vector | Nothing | None (exact) | Full vectors | Small collections, ground truth |
| R-tree / grid file | Prune by bounding regions | Vectors examined | None (exact) | Full vectors | Low-dimensional data only |
| IVF | Cluster, scan nearest clusters | Vectors examined | Clusters probed ($m$) | Full vectors + centers | Large collections, tunable recall |
| HNSW | Greedy search on a layered graph | Vectors examined | Search width | Graph + full vectors | Low-latency, high-recall search |
| LSH | Hash by random hyperplanes | Comparison cost | Number of bits | Compact bit codes | Memory-limited, angular similarity |
| Product quantization | Encode sub-vectors via codebooks | Comparison cost and memory | Sub-vectors, codebook size | Very compact codes | Billion-scale, tight memory |

These are complementary rather than exclusive. A production index typically stacks a coarse quantizer (IVF or HNSW) to cut how many vectors are examined with a fine quantizer (product quantization) to cut the cost of each comparison, then refines the best candidates with exact distances.

## Key Takeaways

1. Embeddings are compared with a similarity where larger is better; media feature vectors are compared with a distance where smaller is better. Length normalization turns cosine similarity into a plain dot product and unifies the two views.
2. Classical spatial indexes (Voronoi, grid file, R-tree) share one optimal search algorithm and all work well only in low dimensions.
3. The curse of dimensionality is real and quantifiable: distances concentrate, the optimal algorithm ends up visiting every leaf, and exact nearest-neighbor search loses both efficiency and meaning.
4. Approximate search trades a little recall for orders-of-magnitude speed, which the applications that use embeddings can almost always afford.
5. In practice, vector search lives inside search engines and databases, must be combined carefully with metadata filters, and is often fused with sparse retrieval by reciprocal rank fusion.

## Key Formulas

```{important} Key Formula: Cosine Similarity and Normalization

$$\text{sim}_{\text{cos}}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d}_i \rVert} = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}}$$

Normalize vectors once, then compare them with a dot product.
```

```{important} Key Formula: Distance Concentration

$$\frac{\sigma_d}{\mu_d} \longrightarrow 0 \quad \text{as } d \to \infty$$

In high dimensions the nearest and farthest points are almost equidistant, which is why exact search stops paying off.
```

```{important} Key Formula: Recall@k

$$\text{recall@}k = \frac{\lvert \text{approximate top-}k \cap \text{exact top-}k \rvert}{k}$$

The quality axis of approximate search, plotted against throughput to compare methods.
```

Reciprocal rank fusion, $\text{RRF}(d) = \sum_{r} 1/(k + \text{rank}_r(d))$, is worth remembering as the standard way to blend dense and sparse rankings.

```{attention} Exam focus
- The difference between similarity ("larger is better") and distance ("smaller is better"), and how normalization connects cosine and dot product.
- Why dense embeddings cannot be pruned the way sparse term vectors can in an inverted file.
- Why the optimal nearest-neighbor algorithm visits every leaf once $l_{\max}$ falls below the expected NN-distance.
- The distinction between coarse quantizers (fewer vectors examined) and fine quantizers (cheaper comparisons), and why they are combined.
- Pre-filtering versus post-filtering, and the failure mode of post-filtering with a selective filter.
```

## Self-Check Questions

1. (Understand) Why does storing length-normalized vectors let a system use the dot product in place of cosine similarity?
2. (Understand) What does a coarse quantizer reduce, and what does a fine quantizer reduce?
3. (Analyze) A grid-file index performs well at $d = 4$ but no better than a full scan at $d = 100$. Explain, in terms of $l_{\max}$ and the expected nearest-neighbor distance, why this happens.
4. (Analyze) A query for the ten nearest neighbors that also match a rare metadata value returns only two results, though many matching documents exist. Which filtering strategy was used, and how would you fix it?
5. (Evaluate) Given a fixed recall target of $0.95$, describe how you would use an ann-benchmarks recall-throughput plot to choose between HNSW and IVF with product quantization.

```{hint} Test Your Knowledge
[Take the Chapter 6 Quiz ->](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)
```

## Further Reading

- Weber, R., Schek, H.-J., & Blott, S. (1998). **A quantitative analysis and performance study for similarity-search methods in high-dimensional spaces**. *Proceedings of VLDB*. [PDF](https://www.vldb.org/conf/1998/p194.pdf). Shows analytically that above a moderate dimension any partitioning index degrades to a scan, and introduces the VA-file as the response, the core argument behind this chapter's curse-of-dimensionality section.
- Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). **On the surprising behavior of distance metrics in high-dimensional space**. *Proceedings of ICDT*, LNCS 1973. [PDF](https://bib.dbvis.de/uploadedFiles/155.pdf). Explains why distance concentration hits some metrics harder than others and why lower-norm distances can behave better in high dimensions.
- Guttman, A. (1984). **R-trees: a dynamic index structure for spatial searching**. *Proceedings of ACM SIGMOD*. [ACM](https://dl.acm.org/doi/10.1145/971697.602266). The original R-tree, the bounding-rectangle index whose search algorithm and overlap problem this chapter builds on.
- Malkov, Y. A., & Yashunin, D. A. (2016). **Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs**. *IEEE TPAMI*. [arXiv](https://arxiv.org/abs/1603.09320). The HNSW method that underlies most modern vector databases.
- Jégou, H., Douze, M., & Schmid, C. (2011). **Product quantization for nearest neighbor search**. *IEEE TPAMI*. [PDF](https://inria.hal.science/inria-00514462v2/document). Introduces product quantization, the compact encoding that makes billion-scale vector search fit in memory.

**Tools and benchmarks**: [FAISS](https://github.com/facebookresearch/faiss) (composite index library), [ann-benchmarks](https://ann-benchmarks.com) (standard recall-throughput comparison), [pgvector](https://github.com/pgvector/pgvector) (PostgreSQL vector search), and the [OpenSearch k-NN plugin](https://docs.opensearch.org/latest/query-dsl/specialized/k-nn/index).
