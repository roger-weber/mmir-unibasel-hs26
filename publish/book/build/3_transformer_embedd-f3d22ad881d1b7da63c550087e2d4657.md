---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Semantic Search
section: Transformer Embeddings
order: "5.3"
---

(transformer-embeddings)=
# Transformer Embeddings

Static word embeddings assign a single vector to each word regardless of context. This fails for words with multiple meanings and ignores the compositional meaning that emerges from word combinations. The transformer architecture (Vaswani et al., 2017) solved this by introducing self-attention: a mechanism that lets each token's representation be influenced by every other token in the sequence. This section traces the path from the attention mechanism through BERT to the modern retrieval architectures: bi-encoders, cross-encoders, and late interaction models.

## The Attention Mechanism

The core idea of self-attention is that a word's meaning in context depends on its relationships with all other words in the sequence. In "The bank approved the loan", the meaning of "bank" is determined by its relationship with "approved" and "loan". Self-attention computes these relationships explicitly.

Consider a sequence of $n$ tokens (for example, the sentence "The bank approved the loan" has $n = 5$ tokens after tokenization). Each token at position $i$ in the sequence is represented by an embedding vector $\mathbf{x}_i \in \mathbb{R}^d$. These embeddings are learned during training, similar to Word2Vec, but they serve as the starting point that the transformer will refine using context. The subscript $i$ refers to the position in the sequence (not the index in the vocabulary): $\mathbf{x}_1$ is the embedding of "The", $\mathbf{x}_2$ is the embedding of "bank", and so on.

Self-attention transforms each token embedding $\mathbf{x}_i$ into a new, context-aware representation $\mathbf{z}_i \in \mathbb{R}^d$ that incorporates information from the entire sequence. The output $\mathbf{z}_i$ has the same dimensionality as the input $\mathbf{x}_i$, but now "bank" at position 2 will have a different $\mathbf{z}_2$ depending on whether the surrounding words are about finance or rivers. The transformation works in three steps:

1. **Project** each token embedding $\mathbf{x}_i \in \mathbb{R}^d$ into three vectors using learned linear mappings $\mathbf{W}_Q \in \mathbb{R}^{d_k \times d}$, $\mathbf{W}_K \in \mathbb{R}^{d_k \times d}$, $\mathbf{W}_V \in \mathbb{R}^{d_v \times d}$:

   $$
   \mathbf{q}_i = \mathbf{W}_Q \mathbf{x}_i \in \mathbb{R}^{d_k}, \quad \mathbf{k}_i = \mathbf{W}_K \mathbf{x}_i \in \mathbb{R}^{d_k}, \quad \mathbf{v}_i = \mathbf{W}_V \mathbf{x}_i \in \mathbb{R}^{d_v}
   $$

   The query $\mathbf{q}_i$ is used to compare (via dot product) against the keys $\mathbf{k}_j$ of all other tokens: a high dot product $\mathbf{q}_i^\top \mathbf{k}_j$ means token $j$ is relevant to token $i$. The value $\mathbf{v}_j$ is the contribution that token $j$ makes to the new representation of token $i$ when the attention score is high. Queries and keys must share the same dimensionality $d_k$ (so the dot product is defined). In practice, $d_k = d_v = d / h$ where $h$ is the number of attention heads (see multi-head attention below). For BERT-Base: $d = 768$, $h = 12$, so each head operates in $d_k = 64$ dimensions.

2. **Compute attention weights** for token $i$ over all positions $j$. The raw compatibility between query $i$ and key $j$ is their scaled dot product $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}$. To convert these raw scores into a probability distribution, we apply the softmax function, which exponentiates each score and normalizes by the sum of all exponentiated scores:

   $$
   \alpha_{ij} = \text{softmax}_j\!\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}\right) = \frac{\exp\!\left(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}\right)}{\sum_{l=1}^{n} \exp\!\left(\mathbf{q}_i^\top \mathbf{k}_l / \sqrt{d_k}\right)}
   $$

   The exponential function ensures all weights are positive, and dividing by the sum ensures they add up to 1: $\alpha_{ij} \geq 0$ and $\sum_j \alpha_{ij} = 1$. A high $\alpha_{ij}$ means token $i$ attends strongly to token $j$.

3. **Aggregate** by computing the weighted sum of all value vectors, using the attention weights as coefficients. This is attention pooling by similarity: each token $j$ contributes its value $\mathbf{v}_j$ proportionally to how relevant it is to token $i$. The result is the new, context-aware representation:

   $$
   \mathbf{z}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{v}_j
   $$

```{admonition} Key Formula: Scaled Dot-Product Attention
:class: important

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

This is the matrix form of steps 1-3 computed in one expression. $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ are $n \times d_k$ (or $n \times d_v$) matrices whose rows are the individual query, key, and value vectors. The product $\mathbf{Q}\mathbf{K}^\top$ is an $n \times n$ matrix of all pairwise dot products. The softmax is applied to each row independently (normalizing across columns), producing the attention weight matrix. Multiplying by $\mathbf{V}$ computes the weighted sum for all tokens in parallel.

The division by $\sqrt{d_k}$ prevents dot products from growing too large with increasing dimension, which would push the softmax into near-zero gradients and slow training.
```

[](#fig-attention-mechanism) illustrates this process for computing z₂ ("bank") in "The bank approved the loan". The query q₂ is compared against all keys; "approved" and "loan" receive the highest attention weights because their keys are most compatible with the query of "bank" in a financial context.

```{figure} images/figure_5_19.png
:name: fig-attention-mechanism
:width: 70%

Self-attention computing z₂ for "bank". The query q₂ is compared against all keys to produce attention weights. "Approved" (0.52) and "loan" (0.36) dominate, so z₂ is primarily a blend of their values, making "bank" context-aware as a financial institution.
```

**Multi-head attention** runs $h$ parallel attention operations, each with its own set of projection matrices $\mathbf{W}_Q^l, \mathbf{W}_K^l, \mathbf{W}_V^l$. Each head operates in $d/h$ dimensions and can learn to attend to different types of relationships (one head might capture syntactic dependencies, another semantic associations). The outputs of all heads are concatenated and projected back to $d$ dimensions ([](#fig-multihead-attention)):

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}_O
$$

```{figure} images/figure_5_20.png
:name: fig-multihead-attention
:width: 70%

Multi-head attention: h parallel heads each run self-attention in d/h dimensions with independent projections. Concatenating all heads recovers the full d-dimensional output, which W_O projects to the final representation.
```

A single self-attention layer produces one round of contextualization. A transformer stacks multiple such layers, each refining the representations further. Between attention layers, a feed-forward network (two linear transformations with a nonlinearity) processes each token independently, and residual connections with layer normalization (Add & Norm) stabilize training. [](#fig-transformers) shows the full architecture from the original "Attention Is All You Need" paper.

The left side is the **encoder**: it processes the full input sequence bidirectionally (each token attends to all others) and produces contextualized representations. The right side is the **decoder**: it generates output tokens one at a time, using masked self-attention (each token can only attend to previous positions) plus cross-attention to the encoder output. For embedding and retrieval, we use only the encoder side. BERT-Base stacks 12 encoder layers with 12 attention heads per layer; BERT-Large uses 24 layers with 16 heads.

```{figure} images/transformers.png
:name: fig-transformers
:width: 50%

The transformer architecture (Vaswani et al., 2017). Left: the encoder processes input bidirectionally through stacked layers of multi-head self-attention and feed-forward networks. Right: the decoder generates output auto-regressively with masked attention. For text embeddings, only the encoder is used.
```


## From Text to Transformer Input

The attention mechanism operates on embedding vectors $\mathbf{x}_1, \ldots, \mathbf{x}_n$, but raw text is a string of characters. Three steps transform text into the input the transformer expects: tokenization, embedding lookup, and positional encoding.

### Sub-word tokenization

Word-level tokenization (one token per space-separated word) creates vocabularies of millions of entries. A neural network with a million-dimensional input layer is impractical. Worse, any word not in the vocabulary (a new brand name, a misspelling, a word from another language) cannot be processed at all.

Sub-word tokenization solves both problems by splitting words into smaller pieces from a compact, fixed vocabulary (typically 30k-50k tokens). Any word, even one never seen during training, can be decomposed into known sub-word pieces. This connects directly to fastText's sub-word idea from the previous section, but here the sub-word units are determined by a data-driven algorithm rather than fixed character n-grams.

**Byte Pair Encoding (BPE)** builds its vocabulary iteratively:

1. Start with all individual characters as the initial vocabulary.
2. Count the frequency of all adjacent pairs across the corpus.
3. Merge the most frequent pair into a new vocabulary entry.
4. Update all word representations by replacing that pair with the new token.
5. Repeat steps 2-4 until the desired vocabulary size is reached.

```{admonition} Example
:class: example

For the sentence "this course is about this topic" (after lowercasing):

Initial vocabulary: `a, b, c, e, h, i, o, p, r, s, t, u`

Word representations:
- "this" (freq 2): `t, h, i, s`
- "course" (freq 1): `c, o, u, r, s, e`
- "is" (freq 1): `i, s`
- "about" (freq 1): `a, b, o, u, t`
- "topic" (freq 1): `t, o, p, i, c`

Most frequent pair: "i,s" (appears 3 times: twice in "this", once in "is"). Merge to create token `is`. After several more merges: `th`, `this`, `ou`, `cour`, `cours`, `course` ...

Final encoding of "this course is about this topic":
`this | course | is | a·b·ou·t | this | t·o·p·i·c`

Unknown words are decomposed into whatever sub-word pieces exist in the vocabulary.
```

**WordPiece** follows the same iterative process but differs in two ways:

1. It distinguishes word-initial characters from word-internal ones using a "##" prefix. The starting vocabulary becomes: `a, c, i, t, ##b, ##c, ##e, ##h, ##i, ##o, ##p, ##r, ##s, ##t, ##u`. This preserves prefix information: the "un" in "unhappy" is a different token than "##un" in "running".

2. Instead of merging the most frequent pair, it merges the pair whose components appear together disproportionately often compared to their individual frequencies. This is the same Pointwise Mutual Information (PMI) criterion used for n-gram extraction: $\text{score}(a, b) = tf(a, b) / (tf(a) \cdot tf(b))$. This prefers pairs that are strongly associated rather than merely frequent.

BERT uses WordPiece with a 30,000-token vocabulary. GPT models use BPE with 50,000 tokens. The word "playing" becomes ["play", "##ing"] in WordPiece or ["play", "ing"] in BPE.

### Token embedding lookup

Each token ID is mapped to a $d$-dimensional vector via a learned embedding matrix $\mathbf{E} \in \mathbb{R}^{|\mathbb{T}| \times d}$. This is the same mechanism as Word2Vec embeddings: row $j$ of $\mathbf{E}$ is the embedding for token $j$. The difference is that these embeddings are trained jointly with the transformer layers rather than separately. The embedding matrix is initialized randomly and refined during pre-training.

### Positional encoding

Self-attention is permutation-invariant: the formula $\mathbf{z}_i = \sum_j \alpha_{ij} \mathbf{v}_j$ depends on the content of tokens (via Q, K, V) but not on their position in the sequence. If we rearrange the input tokens, the attention weights between any two tokens remain unchanged. This means "dog bites man" and "man bites dog" would produce identical representations without additional information.

To preserve sequence order, a positional encoding $\mathbf{p}_i \in \mathbb{R}^d$ is added to each token embedding before the first attention layer:

$$
\mathbf{x}_i = \mathbf{E}[\text{token}_i] + \mathbf{p}_i
$$

The original transformer uses fixed sinusoidal functions at different frequencies for each dimension, allowing the model to learn relative positions. BERT uses learned position vectors (one per position up to 512). Both approaches add position information without increasing the dimensionality.

The result is a sequence of $d$-dimensional vectors that encode both the identity and the position of each token. These serve as the $\mathbf{x}_i$ inputs to the first self-attention layer.

## BERT: Bidirectional Context

BERT (Devlin et al., 2019) applies the transformer encoder to produce contextualized token representations. Unlike GPT, which processes text left-to-right (suitable for generation), BERT attends to context from both directions simultaneously, making it better suited for understanding and representation tasks.

### Input construction

BERT processes input through three summed embeddings:

1. **Token embeddings**: WordPiece sub-word units (vocabulary of ~30,000 tokens). The word "playing" becomes ["play", "##ing"].
2. **Positional encodings**: learned position vectors that preserve sequence order.
3. **Segment encodings**: markers distinguishing sentence A from sentence B when processing sentence pairs.

Two special tokens frame the input: `[CLS]` at the start (whose final hidden state serves as a sequence-level representation) and `[SEP]` separating segments. Sequences are padded to 512 tokens maximum.

```{figure} images/figure_5_15.png
:name: fig-bert-input
:width: 95%

BERT input construction for "the cat that chased the mouse was black". Token embeddings, positional encodings, and segment encodings are summed to form the encoder input. The 12/24-layer encoder produces contextualized output vectors. Option 1: use the `[CLS]` token encoding as the sequence embedding. Option 2: pool (average or max) over all non-masked token encodings.
```

### Pre-training

BERT is pre-trained on two self-supervised tasks:
- **Masked Language Modeling (MLM)**: randomly mask 15% of input tokens and predict them from context.
- **Next Sentence Prediction (NSP)**: given two sentences, predict whether the second follows the first in the original text.

After pre-training on large corpora (Wikipedia + BookCorpus), BERT's encoder produces 768-dimensional (Base) or 1024-dimensional (Large) contextualized vectors for each token. The same word receives different representations depending on its surrounding context.

### From BERT to sentence embeddings

Using raw BERT output for retrieval was initially disappointing. The `[CLS]` token, while designed for classification, produces poor sentence embeddings without task-specific fine-tuning. Mean pooling over token outputs performs marginally better but still underperforms even simple Word2Vec averaging on semantic similarity benchmarks. The breakthrough came with Sentence-BERT.

## Bi-Encoders: Sentence-BERT

Sentence-BERT (Reimers and Gurevych, 2019) restructures BERT into a **bi-encoder** (also called dual-encoder or siamese network): two sentences are encoded independently by the same transformer, then compared via cosine similarity.

```{figure} images/figure_5_16.png
:name: fig-four-architectures
:width: 95%

Four architectures for sentence similarity, progressing from simple to powerful: (1) pooling static word embeddings, (2) pooling BERT token outputs, (3) bi-encoder with independently encoded sentences compared via learned similarity, (4) cross-encoder that jointly processes both sentences for maximum accuracy.
```

### Architecture and training

The bi-encoder produces fixed-size sentence embeddings:
1. Pass a sentence through BERT (or any transformer encoder).
2. Apply mean pooling over the non-masked token outputs to produce a single $d$-dimensional vector.
3. Normalize the vector to unit length.

Training uses similarity-based losses on labeled sentence pairs:
- **Contrastive loss**: pulls similar pairs close, pushes dissimilar pairs apart with a margin.
- **Triplet loss**: given an anchor, a positive, and a negative, ensures $\text{sim}(a, p) > \text{sim}(a, n) + \epsilon$.
- **Multiple negatives ranking loss (MNRL)**: treats all other examples in the batch as negatives, maximizing efficiency of each training step.

Hard negative mining (selecting negatives that are challenging but incorrect) is critical for training high-quality bi-encoders.

### Inference efficiency

The bi-encoder's key advantage for retrieval: documents are encoded once during indexing. At query time, only the query needs encoding. Comparison is a dot product (equivalent to cosine similarity for normalized vectors), enabling sub-millisecond ranking over millions of pre-computed embeddings.

### Modern bi-encoder variants

Since SBERT, the field has advanced significantly:

- **Instruction-tuned embeddings**: models like E5-Mistral, GTE-Qwen2, and Nomic Embed accept a task instruction prefix (e.g., "Represent this document for retrieval:") that adapts the embedding to the downstream task without retraining.
- **Decoder-based encoders**: Qwen3-Embedding and similar models build on decoder transformers (using the final `[EOS]` token representation instead of `[CLS]`), achieving state-of-the-art quality at the cost of larger models (0.6B to 8B parameters).
- **Extended context**: modern embedding models support 8k-32k token inputs, enabling direct encoding of long documents without chunking.

## Cross-Encoders and Reranking

A cross-encoder processes the query and document together as a single input sequence, separated by `[SEP]`:

```text
[CLS] query [SEP] document [SEP]
```

The `[CLS]` token's final hidden state passes through a classification layer to produce a relevance score between 0 and 1. Because all tokens from both query and document attend to each other through every transformer layer, the cross-encoder captures fine-grained token-level interactions that bi-encoders miss.

### Quality vs efficiency tradeoff

| Property | Bi-Encoder | Cross-Encoder |
|----------|-----------|---------------|
| Encoding | Query and document separately | Query and document jointly |
| Pre-computation | Documents encoded at index time | Every pair must be processed at query time |
| Complexity per query | $O(1)$ per document (dot product) | $O(n \cdot L)$ per document ($L$ = sequence length) |
| Semantic depth | Good (independent representations) | Excellent (full token-level interaction) |
| Typical use | First-stage retrieval | Reranking top-$k$ candidates |

### Two-stage pipeline

Cross-encoders are too expensive for exhaustive search over large collections. The standard deployment pattern is:

1. **Retrieve**: a bi-encoder (or BM25) selects the top-$k$ candidates (typically $k = 100\text{-}1000$).
2. **Rerank**: a cross-encoder scores each candidate with full token interaction and reorders the list.

This hybrid approach achieves near cross-encoder quality at near bi-encoder speed.

## Late Interaction: ColBERT

ColBERT (Khattab and Zaharia, 2020) occupies the middle ground between bi-encoders and cross-encoders by computing interaction at the token level while still allowing document pre-computation.

### Architecture

Unlike bi-encoders (which compress each text into a single vector) or cross-encoders (which process both texts jointly), ColBERT:

1. **Encodes** query and document independently through a transformer, retaining all token-level embeddings (not pooling to a single vector).
2. **Interacts** via late interaction: each query token computes maximum similarity against all document tokens.
3. **Scores** by summing these maximum similarities across query tokens:

```{admonition} Key Formula: ColBERT MaxSim Scoring
:class: important

$$
\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{q}_i^\top \mathbf{d}_j
$$

Each query token finds its best-matching document token (MaxSim), and the total score sums these per-token matches. This captures fine-grained term-level alignment without requiring joint encoding.
```

```{figure} images/figure_5_20_colbert_placeholder.png
:name: fig-colbert-architecture
:width: 80%

**[PLACEHOLDER]** ColBERT architecture: query and document are encoded independently (allowing document pre-computation), but scoring uses token-level MaxSim interaction rather than a single-vector dot product.
```

### Why late interaction works

The MaxSim operation captures important phenomena that single-vector similarity misses:

- **Exact term matching**: a query token "Python" will have very high similarity with the document token "Python", directly contributing to the score.
- **Semantic matching**: a query token "automobile" will have high similarity with the document token "car" through their embedding proximity.
- **Partial matching**: not every query token needs to match strongly; the sum allows documents to score well by matching the most important query aspects.

### ColBERT v2 and efficiency

ColBERT v2 (Santhanam et al., 2022) addresses the storage cost of retaining per-token embeddings for every document through **residual compression**: token embeddings are quantized by storing only the residual from the nearest centroid, reducing storage by 6-10x while maintaining quality.

The PLAID engine further accelerates retrieval by using centroid interaction for candidate generation before computing full MaxSim only on the top candidates.

### Architecture comparison

| Architecture | Document representation | Query-time cost | Quality |
|---|---|---|---|
| Bi-encoder | 1 vector per document | 1 dot product per doc | Good |
| ColBERT | $n$ vectors per document | $|q| \times n$ dot products | Very good |
| Cross-encoder | None (must re-encode) | Full transformer pass per doc | Excellent |

ColBERT achieves 95-98% of cross-encoder quality with 100-1000x faster query processing, making it practical for direct retrieval (not just reranking) over moderately large collections.
