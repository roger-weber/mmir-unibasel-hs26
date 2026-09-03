---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Semantic Search
section: Building Semantic Search
order: "5.4"
---

(building-semantic-search)=
# Building Semantic Search

The previous sections traced a progression from BM25 through LSI and word embeddings to transformer-based models. At each step, we reduced the independence assumption between tokens, gaining recall: documents that share no surface terms with the query can now be retrieved through semantic similarity. But this gain comes at three costs that classical retrieval did not face:

1. **Search is expensive.** Dense vectors cannot use inverted indices. Comparing a query against one million document embeddings requires one million dot products. We examine vector search acceleration in the next chapter, but no method matches the pruning efficiency of inverted files for sparse vectors.

2. **Precision drops.** Casting a wider semantic net retrieves more relevant documents, but also more noise. A query for "training large language models" will match papers about "optimizing neural networks" (relevant) and "training dogs with reinforcement signals" (not relevant). Cross-encoders address this through reranking, but at the cost of a full transformer forward pass per candidate.

3. **Compression loses information.** Encoding a multi-page document into a single 768-dimensional vector discards detail. Chunking the document into shorter passages mitigates this, but multiplies the number of vectors to store and search.

Every design decision in a semantic search pipeline is a tradeoff among three dimensions: **compute cost** (encoding, storage, infrastructure), **latency** (how fast results arrive), and **retrieval quality** (precision and recall for the use case). There is no single optimal pipeline. A factual search engine serving millions of short queries per second operates in a different region of this space than a patent lawyer who needs exhaustive recall and is willing to wait for a thorough assessment.

This section shows how to navigate these tradeoffs through pipeline design, model selection, and efficiency techniques.

## Embedding Model Selection

Before designing a pipeline, we need to choose the embedding model that will produce the dense vectors. The [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/spaces/mteb/leaderboard) provides a standardized evaluation framework comparing embedding models across retrieval, semantic textual similarity, classification, and clustering tasks. Retrieval-specific metrics (nDCG@10) are most relevant for search.

### Key selection criteria

| Criterion | Range | Tradeoff |
|-----------|-------|----------|
| Embedding dimension | 256-4096 | Higher dimensions capture more nuance but increase storage and search cost |
| Model size | 33M-8B parameters | Larger models produce better embeddings but require more compute for encoding |
| Context window | 512-32k tokens | Longer windows avoid chunking but compress more content into the same dimensions, degrading quality for long inputs |
| Multilingual support | 1-100+ languages | Multilingual models sacrifice some single-language quality for breadth |
| Instruction support | Yes/No | Instruction-tuned models adapt to specific tasks via prefixes |

### Model landscape (2025)

Representative models spanning the quality-efficiency spectrum:

| Model | Parameters | Dimensions | Context | MTEB Retrieval (nDCG@10) |
|-------|-----------|------------|---------|--------------------------|
| all-MiniLM-L6-v2 | 33M | 384 | 512 | ~49 |
| nomic-embed-text-v1.5 | 137M | 768 | 8192 | ~55 |
| jina-embeddings-v3 | 570M | 1024 | 8192 | ~58 |
| GTE-Qwen2-7B | 7B | 3584 | 32k | ~61 |
| Qwen3-Embedding-8B | 8B | 4096 | 32k | ~63 |

```{note}
Benchmark scores evolve rapidly. The specific numbers above reflect mid-2025 results. Always consult the current MTEB leaderboard when making production decisions.
```

### Cost model

To make the cost-latency-quality tradeoff concrete, consider encoding and searching a collection of 100,000 research papers (averaging 6,000 tokens each, chunked into passages of ~500 tokens, yielding roughly 1.2 million chunks):

| Operation | Small model (33M) | Large model (8B) |
|-----------|-------------------|------------------|
| Encode 1.2M chunks | ~15 min (1 GPU) | ~8 hours (1 GPU) |
| Storage (float32) | ~1.7 GB (384d) | ~18 GB (4096d) |
| Storage (int8 quantized) | ~0.4 GB | ~4.5 GB |
| 1 query: brute-force dot product | ~2 ms | ~15 ms |
| 1 query: cross-encoder rerank top-100 | ~0.5 s (33M) | ~10 s (8B) |
| 1 query: BM25 over inverted index | ~1 ms | ~1 ms |

These numbers are approximate and hardware-dependent, but the ratios are instructive: BM25 is orders of magnitude faster than dense search, and cross-encoder reranking is orders of magnitude slower than retrieval. This cost structure drives the pipeline architectures below.

## Pipeline Architectures

The [chapter introduction](#fig-retrieval-pipeline) presented the retriever-ranker blueprint and five pipeline variants (A through E). Here we examine why each exists and when it fails, following the natural progression as requirements grow.

**Pipeline A (BM25 only)** is the baseline from classical retrieval: an inverted index with BM25 scoring, no semantic component. It is fast, well-understood, and sufficient when queries and documents share vocabulary. Everything discussed in [Chapter 4](#) falls into this category. The remaining pipelines add semantic components to address its limitations.

### Starting simple: BM25 + reranker (Pipeline B)

The simplest semantic search system adds a cross-encoder reranker on top of existing BM25 infrastructure ([](#fig-pipeline-b)). BM25 retrieves the top-$k$ candidates (typically $k = 100\text{-}1000$), and the cross-encoder reorders them by semantic relevance.

This works well when BM25 recall is sufficient: if the relevant documents contain at least some of the query terms, BM25 will find them, and the cross-encoder improves the ordering. It fails when the vocabulary mismatch is severe. A search for "efficient training of large language models" will miss a paper titled "Scaling Laws for Neural Network Optimization" because BM25 cannot connect these terms.

```{figure} images/figure_5_22.png
:name: fig-pipeline-b
:width: 75%

Pipeline B: BM25 retrieves candidates from the inverted index, then a semantic ranker (embeddings and/or cross-encoder) reorders them.
```

### Adding dense retrieval: semantic search (Pipeline C)

To overcome the recall ceiling of BM25, we replace it with a bi-encoder that retrieves candidates from a vector index ([](#fig-pipeline-c)). The "Scaling Laws" paper is now retrieved because its embedding is semantically close to the query.

Dense retrieval finds documents that BM25 misses, but it has the opposite weakness: exact-match queries fail. A search for error code "NullReferenceException" or product ID "A1B2C3" depends on exact string matching; the bi-encoder may return semantically similar but wrong results.

```{figure} images/figure_5_23.png
:name: fig-pipeline-c
:width: 75%

Pipeline C: a bi-encoder retrieves candidates from a vector index by semantic similarity, with an optional cross-encoder for reranking.
```

### Going hybrid: sparse + dense (Pipeline D)

The most robust approach runs both retrievers in parallel and fuses the results:

1. BM25 retrieves top-$k_1$ candidates (handles exact matches, rare terms, proper nouns).
2. Dense retrieval retrieves top-$k_2$ candidates (handles semantic matches, paraphrases).
3. **Reciprocal Rank Fusion (RRF)** merges both result lists.
4. A cross-encoder reranks the merged top-$k$ for final ordering.

RRF assigns each document a fused score based on its rank in each result list:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$

where $R$ is the set of result lists and $k$ is a constant (typically 60) that dampens the influence of high ranks. Documents appearing in both lists receive contributions from both, naturally boosting results that both methods agree on.

```{figure} images/figure_5_21_hybrid_placeholder.png
:name: fig-hybrid-retrieval
:width: 85%

**[PLACEHOLDER]** Hybrid retrieval: BM25 and dense retrieval run in parallel, results are fused via RRF, then reranked by a cross-encoder.
```

### Chunking for different retrievers

All three pipelines require splitting long documents into shorter passages. The optimal chunk size differs by component:

- **BM25** tolerates large chunks (full sections, even full documents). It scores individual terms within the chunk; surrounding irrelevant text does not affect the score of matching terms. Traditional retrieval systems typically chunk at the page or section level.
- **Bi-encoders** need shorter chunks (256-1024 tokens). The entire chunk is compressed into a single vector, so irrelevant content dilutes the embedding. Paragraph-level or sliding-window chunking with overlap produces the best results.
- **Cross-encoders** can handle longer inputs (up to their context window) because they read query and document jointly with full token interaction. They are typically applied to the chunks retrieved by the earlier stages.

In hybrid systems, a common approach is to chunk at a granularity that works for the bi-encoder (the most sensitive component) and use the same chunks for BM25. The result metadata links chunks back to their source documents so that the final output can present either chunk-level or document-level results. We discuss chunking strategies (fixed-size, recursive, semantic, hierarchical) in detail in the [RAG chapter](#), where the interaction between chunk design and generation quality adds further constraints.

## Efficiency at Scale

As collections grow from thousands to millions of documents, the cost of encoding, storing, and searching becomes the dominant concern. Two techniques reduce this cost without changing the pipeline architecture.

### Matryoshka Representation Learning (MRL)

Traditional embeddings use a fixed number of dimensions. Matryoshka embeddings (Kusupati et al., 2022) train the model so that truncated prefixes of the embedding remain useful: the first 64 dimensions capture coarse semantics, the first 128 add detail, and the full vector provides maximum precision. The training objective adds loss terms at multiple truncation levels:

$$
\mathcal{L}_{\text{MRL}} = \sum_{d \in \mathcal{D}} \mathcal{L}_d(\mathbf{x}_{[:d]}, \mathbf{y}_{[:d]})
$$

where $\mathcal{D}$ is a set of target dimensions (e.g., {64, 128, 256, 512, 1024}) and $\mathbf{x}_{[:d]}$ denotes the first $d$ components of the embedding.

This enables **multi-stage retrieval within a single model** ([](#fig-matryoshka-pipeline)):
1. Use truncated embeddings (e.g., 128 dimensions) for fast coarse retrieval over the full collection.
2. Use full embeddings (e.g., 1024 dimensions) to rerank the top candidates with higher precision.

```{figure} images/figure_5_18.png
:name: fig-matryoshka-pipeline
:width: 85%

Multi-stage retrieval with Matryoshka embeddings: fast initial retrieval using small (truncated) embeddings narrows candidates, then full-dimension embeddings rerank with higher precision. A single model serves both stages.
```

### Binary and scalar quantization

Storage and comparison costs can be further reduced through quantization:

- **Binary quantization**: replace each float dimension with a single bit (positive to 1, negative to 0). Reduces storage by 32x and enables fast Hamming distance computation. Quality loss is typically 5-10% on retrieval benchmarks.
- **Scalar quantization** (int8): map each float to an 8-bit integer. Reduces storage by 4x with minimal quality loss (typically less than 2%).

Quantization is most effective as a first-stage filter: use quantized embeddings for coarse candidate retrieval, then rescore with full-precision embeddings.

## Practical Recommendations

There is no single best pipeline. The right design depends on where your application sits in the cost-latency-quality space. The following recommendations serve as starting points, not prescriptions.

### By collection size

| Collection | Recommended pipeline | Rationale |
|------------|---------------------|-----------|
| < 10k documents | Pipeline C (bi-encoder + cross-encoder) | Brute-force search is fast enough; no need for BM25 infrastructure or ANN indices |
| 10k - 1M documents | Pipeline D (hybrid + cross-encoder) | BM25 handles exact match; dense retrieval handles semantic match; cross-encoder refines |
| > 1M documents | Pipeline D with Matryoshka or quantized first stage | Multi-stage retrieval keeps latency manageable; ANN index required (see next chapter) |

### By use case

| Use case | Priority | Design emphasis |
|----------|----------|-----------------|
| Factual search (Q&A, support) | Low latency, moderate recall | Small bi-encoder (33M-137M), short chunks, fast reranker |
| Patent search, legal discovery | High recall, thoroughness | Large bi-encoder (7B+), hybrid retrieval, aggressive cross-encoder reranking, accept higher latency |
| E-commerce product search | Exact match + semantic, low latency | Hybrid mandatory (product IDs need BM25); lightweight reranker or no reranker |
| Academic literature search | Broad semantic recall, multilingual | Multilingual model (BGE-M3, Qwen3), hybrid retrieval, moderate reranking |

### Connection to later chapters

Dense embeddings create the vectors; [Chapter 6 (Vector Search)](#) addresses how to efficiently index and search them at scale using approximate nearest neighbor algorithms (HNSW, IVF, product quantization). [Chapter 7 (RAG)](#) builds on the retrieval pipeline by feeding retrieved chunks to a language model for answer generation, adding further constraints on chunk design and result quality.
