---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Vector Search
section: Vector Search in Practice
order: "6.5"
---

(vector-search-in-practice)=
# Vector Search in Practice

The methods of the previous section rarely reach an application in raw form. They arrive packaged inside search engines and databases that also handle storage, updates, filtering, and scale. This section looks at how three widely used systems expose vector search, at the recurring problem of combining similarity with structured filters, and at how to fuse dense and sparse retrieval into one ranking.

## Vector search in search engines and databases

Search engines added vector fields to their existing text machinery. **Lucene** stores fixed-length float vectors and answers nearest-neighbor queries with an HNSW graph per segment, supporting cosine, dot product, and Euclidean distance. A vector query can be combined with an ordinary filter, for example "similar to this embedding and published before 2000", in a single Boolean query that the engine evaluates in one pass. **OpenSearch** builds on Lucene with a k-NN plugin, adding a vector field type and distributing the index across shards, each holding its own HNSW graph whose results merge at query time. Its main limitation is that vector scoring and keyword (BM25) scoring are separate signals that must be combined explicitly.

Relational databases took the same step. The **pgvector** extension adds a vector column type to PostgreSQL, with IVF, HNSW, or exact indexes and distance operators that plug into the query planner. A single SQL statement can filter rows by metadata and order the survivors by cosine distance, so vector similarity lives alongside ordinary `WHERE` clauses and joins. Because PostgreSQL runs on one node rather than a distributed cluster, pgvector suits collections that fit comfortably on a single machine, while OpenSearch and dedicated vector databases target larger scale.

## Combining similarity with filters

Real queries rarely ask for similarity alone. A user wants films similar to a query embedding *and* released after 2010, or documents close in meaning *and* written in German. Merging a metadata filter with a vector search is harder than it looks, and there are two basic strategies.

**Pre-filtering** applies the metadata condition first and searches only the matching vectors. This is exact and efficient when the filter is very selective, but it fights the ANN index: structures like HNSW and IVF are built over the whole collection, and restricting them to an arbitrary subset can force a fallback to a slower scan of the survivors.

**Post-filtering** runs the vector search first and then discards results that fail the filter. This keeps the ANN index intact, but it can return too few results, or none, when the filter is selective, because the top-$k$ neighbors may all be filtered out. Systems mitigate this by over-fetching, retrieving more candidates than requested so that enough survive the filter.

```{warning} Post-filtering can silently return too few results
If a query asks for the ten nearest neighbors and then removes those failing a selective filter, the final list may contain far fewer than ten items, or be empty, even though many matching documents exist deeper in the ranking. Over-fetch a larger candidate set before filtering, and verify that enough results survive.
```

The choice depends on filter selectivity and on what the index supports. A highly selective filter favors pre-filtering; a mild filter favors post-filtering with over-fetching. Many production systems implement both and choose per query based on an estimate of how many rows the filter admits.

## Hybrid retrieval: fusing dense and sparse

Dense embeddings capture meaning but are hard to interpret, while sparse lexical methods such as BM25 match exact terms and handle rare words and names well. Neither dominates, so systems increasingly run both and fuse the two rankings. The most common recipe is **reciprocal rank fusion**, which combines rankings by position rather than by raw score, sidestepping the problem that a cosine similarity and a BM25 score are not on the same scale.

```{important} Key Formula: Reciprocal Rank Fusion

For a document $d$ ranked by several retrievers, the fused score sums a small contribution from each ranking:

$$\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$$

where $\text{rank}_r(d)$ is the position of $d$ in ranking $r$ and $k$ is a constant (often $60$) that damps the influence of top positions.
```

Because RRF depends only on ranks, it fuses a dense embedding ranking with a sparse BM25 ranking without any score calibration, and a document that both methods rank highly rises to the top. This hybrid approach is now a common default: the sparse signal anchors exact matches and rare terms, the dense signal supplies semantic recall, and rank fusion blends them into a single result list.

```{hint} Hands-on: Vector search with pgvector and OpenSearch
Run similarity search in PostgreSQL and OpenSearch, combine vector queries with metadata filters, and compare when each framework is the right choice.
[Open notebook ->](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch06-03-vector-search-frameworks.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```
