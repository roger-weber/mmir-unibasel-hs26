---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Semantic Search
section: Building Semantic Search
order: "5.5"
---

(building-semantic-search)=
# Building Semantic Search

The previous sections presented the models that produce semantic representations. This section addresses the engineering question: how do we assemble these components into a working retrieval system? A production semantic search pipeline must choose an embedding model, decide how to chunk documents, select a retrieval architecture, and manage efficiency at scale.

## Embedding Model Selection

The Massive Text Embedding Benchmark (MTEB) provides a standardized evaluation framework for comparing embedding models across multiple tasks: retrieval, semantic textual similarity, classification, clustering, pair classification, and reranking. As of mid-2025, MTEB evaluates models on 56+ datasets across these tasks, with retrieval-specific metrics (nDCG@10) being most relevant for search applications.

### Key selection criteria

When choosing an embedding model for a retrieval system, consider:

| Criterion | Range | Tradeoff |
|-----------|-------|----------|
| Embedding dimension | 256-4096 | Higher dimensions capture more nuance but increase storage and computation |
| Model size | 33M-8B parameters | Larger models produce better embeddings but require more compute for encoding |
| Context window | 512-32k tokens | Longer windows encode more context per chunk but increase encoding time |
| Multilingual support | 1-100+ languages | Multilingual models sacrifice some single-language quality for breadth |
| Instruction support | Yes/No | Instruction-tuned models adapt to specific tasks via prefixes |

### Instruction-tuned embeddings

Modern embedding models accept task-specific instructions that adapt the representation without retraining:

```
# For encoding documents
instruction = "Represent this document for retrieval: "
embedding = model.encode(instruction + document_text)

# For encoding queries
instruction = "Represent this query for retrieving relevant documents: "
embedding = model.encode(instruction + query_text)
```

This asymmetric encoding (different instructions for queries vs documents) improves retrieval quality by 3-5% on MTEB benchmarks compared to symmetric encoding.

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
Benchmark scores evolve rapidly. The specific numbers above reflect mid-2025 results. Always consult the current MTEB leaderboard (https://huggingface.co/spaces/mteb/leaderboard) when making production decisions.
```

## Document Chunking and Encoding

Embedding models have finite context windows, and retrieval granularity affects both precision and recall. Documents must be split into chunks that balance:
- **Semantic coherence**: each chunk should represent a complete thought or topic.
- **Size constraints**: chunks must fit within the model's context window.
- **Retrieval granularity**: smaller chunks enable precise matching but may lose context; larger chunks provide more context but dilute the embedding with irrelevant content.

### Chunking strategies

**Fixed-size chunking** splits text at regular token intervals (e.g., every 256 or 512 tokens) with overlap between adjacent chunks (typically 10-20%). Simple to implement but may split sentences or paragraphs mid-thought.

**Semantic chunking** uses structural cues (paragraph boundaries, section headers, sentence boundaries) to create naturally coherent chunks. More complex but produces chunks aligned with the document's information structure.

**Recursive chunking** starts with large structural splits (sections), then recursively subdivides any chunk exceeding the target size by progressively finer boundaries (paragraphs, sentences, fixed size).

### Encoding pipeline

A typical encoding pipeline for a document collection:

1. **Preprocessing**: extract text, normalize whitespace, optionally remove boilerplate.
2. **Chunking**: split into overlapping or non-overlapping segments.
3. **Metadata attachment**: each chunk retains a reference to its source document, position, and any relevant metadata (title, section heading, date).
4. **Batch encoding**: the embedding model encodes chunks in batches on GPU.
5. **Normalization**: L2-normalize all vectors for dot-product retrieval.
6. **Indexing**: store vectors in a vector database or approximate nearest neighbor index (covered in [Chapter 6](#)).

## Retrieval Architectures

Three dominant patterns structure modern semantic search pipelines:

### Option A: BM25 + semantic reranker

```{figure} images/figure_5_17.png
:name: fig-pipeline-options
:width: 85%

Two retrieval pipeline architectures. Top: BM25 retrieves candidates, then an embedding model or cross-encoder reranks them. Bottom: dense embedding retrieval with optional cross-encoder reranking.
```

**BM25 first-stage** retrieves the top-$k$ candidates (typically $k = 100\text{-}1000$) using the inverted index. A **semantic reranker** (cross-encoder or bi-encoder) then reorders these candidates by semantic similarity. This architecture leverages existing infrastructure (inverted indices are well-understood and fast) while adding semantic depth.

Strengths: fast, handles exact-match queries well, low infrastructure cost.
Weakness: BM25 recall ceiling limits semantic reranking; if the relevant document is not in the top-$k$ candidates, no reranker can recover it.

### Option B: Dense retrieval + optional reranker

A bi-encoder retrieves the top-$k$ documents directly from a vector index using approximate nearest neighbor search. An optional cross-encoder reranks the results for higher precision.

Strengths: captures semantic matches that BM25 misses entirely.
Weakness: poor on exact-match queries (proper nouns, codes, IDs); requires vector index infrastructure.

### Option C: Hybrid retrieval (sparse + dense)

The most robust modern approach fuses both BM25 and dense retrieval results:

1. BM25 retrieves top-$k_1$ candidates (handles exact matches, rare terms).
2. Dense retrieval retrieves top-$k_2$ candidates (handles semantic matches).
3. **Reciprocal Rank Fusion (RRF)** or linear score combination merges both result lists.
4. A cross-encoder reranks the merged top-$k$ for final ordering.

```{figure} images/figure_5_21_hybrid_placeholder.png
:name: fig-hybrid-retrieval
:width: 85%

**[PLACEHOLDER]** Hybrid retrieval: BM25 and dense retrieval run in parallel, results are fused via Reciprocal Rank Fusion, then reranked by a cross-encoder. This captures both lexical and semantic matches.
```

RRF assigns each document a fused score based on its rank in each result list:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$

where $R$ is the set of result lists and $k$ is a constant (typically 60) that dampens the influence of high ranks.

## Efficiency at Scale

### Matryoshka Representation Learning (MRL)

Traditional embeddings use a fixed number of dimensions. Matryoshka embeddings (Kusupati et al., 2022) train the model so that truncated prefixes of the embedding remain useful: the first 64 dimensions capture coarse semantics, the first 128 add detail, and the full vector provides maximum precision.

```{figure} images/figure_5_18.png
:name: fig-matryoshka-pipeline
:width: 85%

Multi-stage retrieval with Matryoshka embeddings: a fast initial retrieval using small (truncated) embeddings narrows candidates, then full-dimension embeddings provide precise reranking. A single model serves both stages.
```

The training objective adds loss terms at multiple truncation levels:

$$
\mathcal{L}_{\text{MRL}} = \sum_{d \in \mathcal{D}} \mathcal{L}_d(\mathbf{x}_{[:d]}, \mathbf{y}_{[:d]})
$$

where $\mathcal{D}$ is a set of target dimensions (e.g., {64, 128, 256, 512, 1024}) and $\mathbf{x}_{[:d]}$ denotes the first $d$ components of the embedding.

This enables **multi-stage retrieval within a single model**:
1. Use truncated embeddings (e.g., 128 dimensions) for fast coarse retrieval over the full collection.
2. Use full embeddings (e.g., 1024 dimensions) to rerank the top candidates with higher precision.

### Binary and scalar quantization

Storage and comparison costs can be further reduced through quantization:

- **Binary quantization**: replace each float dimension with a single bit (positive → 1, negative → 0). Reduces storage by 32x and enables fast Hamming distance computation. Quality loss is typically 5-10% on retrieval benchmarks.
- **Scalar quantization** (int8): map each float to an 8-bit integer. Reduces storage by 4x with minimal quality loss (typically < 2%).

Quantization is most effective as a first-stage filter: use quantized embeddings for coarse candidate retrieval, then rescore with full-precision embeddings.

## Practical Considerations

### Asymmetric retrieval

Queries are typically short (5-20 tokens) while documents may be long (hundreds to thousands of tokens). This asymmetry means:
- Query embeddings may not capture full intent in limited tokens.
- Document embeddings must compress extensive content into a fixed vector.
- Instruction-tuned models address this with different prefixes for queries vs documents.

### Domain adaptation

General-purpose embedding models may underperform on specialized domains (legal, medical, scientific) where vocabulary and language patterns differ from training data. Options for adaptation:
- **Fine-tuning**: train on domain-specific query-document pairs (requires labeled data).
- **Synthetic data**: use LLMs to generate query-document pairs from the target corpus.
- **Domain-specific models**: some domains have specialized embedding models (e.g., PubMedBERT for biomedical text).

### Embedding freshness

Unlike inverted indices where adding a document requires only updating postings lists, embedding models may drift: a model trained on 2024 data may produce suboptimal representations for 2026 content discussing new concepts. Re-encoding the full collection with an updated model requires significant compute but ensures consistent representation quality.

### Connection to vector search

Dense embeddings create the vectors; the next chapter ([Vector Search](#)) addresses how to efficiently index and search these vectors at scale using approximate nearest neighbor algorithms (HNSW, IVF, product quantization).
