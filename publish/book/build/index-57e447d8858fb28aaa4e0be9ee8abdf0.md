---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Semantic Search
order: "5"
---

# 5 - Semantic Search

Classical retrieval matches keywords; semantic search matches meaning. This chapter develops the full progression from matrix factorization methods that first revealed latent structure in text, through the neural revolution of word embeddings and transformers, to the production-ready retrieval pipelines deployed in modern search systems. By the end, you will understand how to design a system that retrieves documents about "cardiac conditions" when a user searches for "heart disease", even when no terms overlap.

```{admonition} What you'll learn
:class: tip
- Explain how SVD-based dimensionality reduction captures latent semantic relationships between terms
- Compare static word embeddings (Word2Vec, GloVe) with contextual embeddings (BERT, SBERT) and articulate why context matters
- Analyze the tradeoffs between bi-encoders, cross-encoders, and late interaction models in terms of quality, latency, and scalability
- Design a multi-stage semantic search pipeline that combines sparse and dense retrieval with reranking
- Evaluate embedding models using standardized benchmarks (MTEB) and select appropriate models for a given use case
```

```{admonition} Prerequisites
:class: seealso
- [Classical Text Retrieval (Ch. 1)](#): TF-IDF weighting, cosine similarity, the vector space model
- [Advanced Text Processing (Ch. 3)](#): tokenization, subword segmentation (BPE, WordPiece)
- [Indexing for Text Retrieval (Ch. 4)](#): inverted index, BM25 scoring
- Basic linear algebra: matrix multiplication, eigenvalues, orthogonality
- Basic neural networks: gradient descent, loss functions (see Appendix)
```
