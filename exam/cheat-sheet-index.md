# Cheat sheet: Indexing and Vector Search

How a search system avoids reading the whole collection for every query: the inverted index,
filtering and distribution for text (chapter 4), and similarity measures, the limits of exact
search structures, approximate nearest-neighbor search and hybrid search for vectors (chapter 6).

## Chapter 4: Index for Text Retrieval

### The inverted index and Boolean merging

- Boolean retrieval, the vector space model and BM25 score a document from its query terms alone,
  so a scan over all documents discards almost everything it reads.
- The inverted index turns the layout around, from documents with their terms to terms with their
  postings lists. Parts: the vocabulary (lookup key), the sorted postings lists of document
  identifiers, and a document table with per-document metadata.
- It stores the same term-document pairs as the document layout; what changes is the read per
  query. A query reads the postings of its own terms, so its cost depends on the query and grows far
  more slowly than the collection.
- Frequent terms are the weak spot: their postings are far longer than average. At web scale even
  discriminative terms occur in hundreds of millions of pages, so dropping stop words does not
  rescue a plain postings scan. Every term that query expansion adds is one more list to read.
- Boolean operators are merges of sorted postings read as streams: AND intersects, OR unites, AND
  NOT subtracts. Each merge emits a sorted stream, so merges compose. NOT
  stands only inside an AND with a positive operand and is applied last, since its complement
  covers almost the whole collection.
- Sorted order comes for free while documents are appended with increasing identifiers; deletions
  and in-place updates would break it. Lucene, for example, keeps immutable segments: a delete flips
  a bit in a list of live documents, an update is a delete followed by an add, and deleted documents
  are physically dropped when segments are merged in the background.

**Math / algorithm** (*for understanding; formulas are not asked in the exam*): AND merge

1. Compare the heads of both streams.
2. If they differ, advance the stream with the smaller head.
3. If they are equal, emit the document and advance both.
4. Stop as soon as one stream is exhausted.

Example: cat = 1, 4, 10 and dog = 3, 4, 8, 10, 12 give 4, 10; the entry 12 is never read.

**Go deeper:** [Inverted Files](https://roger-weber.github.io/mmir-unibasel-hs26/book/inverted-files/)

### Ranked retrieval over postings: TAAT and DAAT

- The retriever takes the union of the query terms' postings (a document without any query term
  scores zero); the ranker scores these candidates and keeps the top k.
- Term-at-a-time (TAAT) processes one list after the other and accumulates partial scores: simple,
  but the score table grows with common terms and it cannot stop early.
- Document-at-a-time (DAAT) advances all lists together, completes each candidate's score at once,
  keeps a heap of k results and can skip candidates that cannot reach the top k; production systems
  prefer it.
- The index stores what the model needs: document identifiers for Boolean retrieval and BIR, term
  frequencies for the vector space model and BM25, plus a per-document table of norms or lengths,
  since the postings never show a whole document.

### Filtering on metadata and facets

- A filter is a hard condition on metadata (released after 2005, rated at least 7.5); the question
  is when to evaluate it.
- Text: a priori through an index on the attribute, whose documents restrict the merge (best when
  the attribute is indexed); a posteriori on the ranked results when it is not; inline, with an
  attribute stream aligned with the postings.
- Facets filter on categories and report how many documents fall under each value. They work in
  two ways. Indexed category facets (product categories, genres) are a priori filters: one postings
  list per facet value, so filtering is an intersection and the facet counts are list lengths. As a
  post-processing step, for example in the user interface, facets only reduce or reorder the
  returned results (the retriever with facets of chapter 1).
- Vectors: the ANN index knows nothing about metadata.
  - Pre-filtering compares the query exactly with the admitted vectors: cheap for a selective
    filter, close to a full scan for a mild one.
  - Post-filtering keeps the index but can return far fewer than k results. The robust form
    consumes the index as a stream of candidates until k pass.
  - Filtering during the search skips non-admitted vectors while scanning.
- Under a very selective filter, an IVF that probes few lists sees few admitted vectors, while the
  best admitted matches lie in unprobed lists: the result is short or poor. Remedies: probe more
  lists, stream candidates, or switch to an exact scan over the admitted vectors.
- The filter belongs inside the vector query; placed next to it, it acts as a post-filter.

### Scaling out: segments, shards, replicas, regions

| Level | Buys | Does not |
|---|---|---|
| Segments in parallel | shorter query, up to a limit | remove the single point of failure |
| Shards (disjoint subsets) | collection outgrows one machine | spare any shard per query |
| Replicas (other zones) | survive failures; more concurrent queries | make one query faster |
| Regions (whole clusters) | low latency near users; capacity | split one query across regions |

- Text and vector search scale the same way (OpenSearch distributes both alike): scatter the query
  to all shards, merge their top lists.
- Each text shard keeps its own term statistics, so BM25 scores differ slightly by shard.
- Each vector shard runs its own ANN search; the global top k is always among the per-shard lists,
  so merging adds no error. Lucene builds one HNSW graph per segment, and merging segments means
  merging graphs, which makes indexing vectors expensive.

## Chapter 6: Vector Search

### Similarity measures

- Embeddings are compared by dot product or cosine, as the model was trained: dot product rewards
  aligned and long vectors, cosine alignment alone. Perceptual features usually use a distance
  (Euclidean, Manhattan, maximum, quadratic), where smaller is better.
- The comparison belongs to the feature or model; replacing it changes the ranking. A decreasing
  function such as 1/(1 + distance) turns a distance into a similarity and keeps the order.
- Normalizing the stored vectors once at indexing turns cosine into a dot product; on unit vectors
  Euclidean distance gives the same order. The query norm scales all its scores alike, so the query
  needs normalizing only for a fixed score threshold.
- Perceptual dimensions must be comparable: divide histogram counts by the number of pixels so image
  size drops out; rescale each dimension of a mixed feature (brightness 0 to 255 next to histogram
  fractions) by Gaussian normalization, with statistics estimated once on the collection and kept
  fixed. Correlated dimensions call for a quadratic distance, which a transformation at indexing
  turns into a Euclidean one.

**Math** (*for understanding; formulas are not asked in the exam*)

$$\text{sim}_{\cos}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d} \rVert} = \frac{1}{\lVert \mathbf{q} \rVert} \, \mathbf{q} \cdot \hat{\mathbf{d}}, \qquad \hat{\mathbf{d}} = \frac{\mathbf{d}}{\lVert \mathbf{d} \rVert}$$

$$\hat{v}_j = \frac{v_j - \mu_j}{\sigma_j} \qquad \text{(mean and standard deviation of dimension } j \text{ over the collection)}$$

### Why exact search structures fail on embeddings

- In two or three dimensions, a Voronoi diagram precomputes every nearest-neighbor answer (the
  geometric counterpart of the inverted file), and grid files and R-trees prune well for range
  queries, whose region is known in advance.
- A nearest-neighbor query must find a candidate and prove that nothing is closer. Even the optimal
  algorithm opens every region that the sphere through the nearest neighbor touches, often on
  several root-to-leaf paths.
- With growing dimension, rectangles overlap and distances concentrate: nearest and farthest point
  become almost equally far away (Euclidean, dot product and cosine alike). The sphere then touches
  every leaf, and beyond roughly ten dimensions a full scan beats a tree. No new tree design
  changed this; the cause lies in the space.
- Real data is clustered: related items stand out clearly from the background, which approximate
  methods exploit.
- Inside a cluster, the nearest neighbor is only marginally closer than the next candidates, so the
  vector search orders a cluster poorly; a reranker (a cross-encoder, or the full vectors after a
  compressed first stage) sets the final order. See the two-stage pipeline in
  [Semantic Search and RAG](cheat-sheet-semantic.md#bi-encoders-cross-encoders-and-colbert).

### The three levers of approximate search

- A full scan costs number of vectors times vector size. Each lever trades a little accuracy for
  speed.
- Fewer dimensions: PCA, or truncating a Matryoshka embedding, whose prefix is itself a usable
  embedding (no projection, a shorter read). A modest gain; every vector is still read.
- Fewer bits per vector: scalar quantization to small integers, product quantization with learned
  codebooks, binary codes or LSH bit strings compared by the Hamming distance. About 4 to 100 times
  less data, yet every vector is still read. Rescoring the best candidates with the full vectors
  repairs most compression errors.
- Fewer vectors read, inverted file (IVF): scan only the clusters nearest to the query. A neighbor
  across a cluster boundary is missed unless its list is probed; more probed lists recover misses at
  linear cost and shrinking gain.
- Fewer vectors read, graph (HNSW): walk greedily from neighbor to neighbor toward the query; the
  walk can end in a local minimum, so the result is approximate. Among the fastest methods, but the
  links and full vectors must sit in main memory, and building is slow.
- Production indexes combine the levers; their errors compound, so the settings are tuned together.
  HNSW is a strong default while the vectors fit in memory; beyond that, IVF with product
  quantization and a rescoring step.

**Math** (*for understanding; formulas are not asked in the exam, but compression factors may be
computed from small numbers*)

$$\text{compression factor} = \frac{32 \cdot d \text{ bits (floats)}}{\text{bits of the code}}$$

- int8 scalar: 32/8 = 4; 4-bit scalar: 8; binary (1 bit per component): 32.
- 768 dimensions (3,072 bytes): PQ with 96 one-byte codes gives 32; LSH with 256 bits (32 bytes)
  gives 96.

**Go deeper:** [Approximate Nearest-Neighbor Search](https://roger-weber.github.io/mmir-unibasel-hs26/book/approximate-nn-search/),
[demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch06-02-approximate-nn.ipynb)

### Choosing an operating point

- Each method trades recall for speed through its settings. Plotting recall@k against queries per
  second compares methods: fix the recall needed and take the fastest. The last points of recall are
  the most expensive.
- Recall against exact search compares retrievers, not what users see: the final result of a
  pipeline, judged for relevance (nDCG, see [Evaluation](cheat-sheet-eval.md#graded-relevance-dcg-and-ndcg)).
- The exact neighbors are only marginally nearer than the next candidates, and many of those are
  relevant too. A cheap retriever that finds a quarter of the exact top 10 can still return
  relevant documents, and a reranker on its top 50 often comes close to exact search.
- Fine-filter errors (distorted codes) are repairable by reranking while relevant documents remain
  among the candidates; coarse-filter errors (lists never probed) are not, as the reranker never
  sees those documents. So the coarse filter stays generous.
- Choose the operating point on end-to-end quality, per application: the cheapest retriever whose
  reranked result stays close to exact search.

**Math** (*for understanding; formulas are not asked in the exam, but recall@k may be computed for a
short list*)

$$\text{recall@}k = \frac{\lvert \text{approximate top-}k \cap \text{exact top-}k \rvert}{k}$$

**Go deeper:** [ANN benchmark demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch06-03-ann-benchmark.ipynb)

### Hybrid search and rank fusion

- One request often needs three kinds of match: the meaning of a plot (vector search on an
  embedding), an exact name or category (BM25 on a field), and hard constraints (filters).
- Each alone fails: in the book's movie example, vector search alone ranks the target movie 138th,
  BM25 misses a plot told in other words, and an embedding blurs a name into similar names.
- Reciprocal rank fusion (RRF) uses ranks alone, so it fuses a cosine and a BM25 ranking without
  calibrating their scores. A document found by both lists collects two contributions and rises; a
  document found by one list keeps its one contribution.
- Adding raw scores, as a Boolean query in Lucene does, lets the keyword part dominate: the vector
  clause contributes at most 1, a BM25 clause for a rare name several points. Normalizing and
  weighting the scores is the alternative, at the cost of choosing the weights.
- With more signals (freshness, popularity, a reranker score) and judged training data, a learned
  ranker can combine BM25, the vector score and further features with learned weights instead of a
  fixed rule; see learning to rank in [Text and Web Retrieval](cheat-sheet-text.md#the-ranking-stack-and-learning-to-rank).

**Math** (*for understanding; formulas are not asked in the exam*)

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Here $R$ is the set of result lists and $k$ a constant that dampens the influence of the top ranks.

**Go deeper:** [vector search frameworks demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch06-04-vector-search-frameworks.ipynb)
