---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Semantic Search
section: Latent Semantic Indexing
order: "5.1"
---

(latent-semantic-indexing)=
# Latent Semantic Indexing

In classical retrieval, documents are sparse high-dimensional vectors with TF-IDF weights, and similarity depends on shared terms. This term independence assumption limits recall: a document about "automobile maintenance" will not match a query for "car repair" unless we explicitly expand the query with synonyms. Stemming and lemmatization reduce some mismatches, but they are language-specific, require manual effort, and cannot capture domain-specific relationships. A search for "tokens" in a text processing context should also retrieve paragraphs discussing "terms" or "words", but no stemmer will make that connection.

Latent Semantic Indexing (LSI) addresses this by projecting the high-dimensional term-document space into a compact, lower-dimensional representation where semantically related terms are brought closer together. The key insight is that terms appearing in similar document contexts are likely related, even if they never co-occur in the same document. By reducing dimensionality, LSI forces the model to merge these co-occurrence patterns into latent topics.

```{admonition} LSI in historical context (optional reading)
:class: note dropdown

LSI was developed at Bell Labs in the 1980s by Susan Dumais and Scott Deerwester. The first article appeared in 1988, and a patent was granted the same year (since expired). LSI demonstrated that unsupervised dimensionality reduction could improve retrieval quality, a foundational idea that influenced all subsequent embedding methods. Despite this conceptual contribution, LSI faced two practical barriers: the expensive SVD computation over large matrices, and the inability to use inverted indices for the resulting dense vectors. While computational costs have since become manageable, LSI has been superseded by neural embedding methods that capture finer-grained semantic associations and transfer across collections.
```

## Singular Value Decomposition

Recall from the vector space model that we represent a collection as a term-document matrix $\mathbf{A}$ of size $m \times n$, where $m$ is the vocabulary size and $n$ the number of documents. Each entry $a_{ij}$ holds the TF-IDF weight of term $i$ in document $j$. Most entries are zero because any single document uses only a small fraction of the vocabulary, making $\mathbf{A}$ extremely sparse.

LSI builds on the Singular Value Decomposition (SVD), which factorizes this matrix into three components. Given $\mathbf{A}$ of rank $r$, the SVD produces:

$$
\mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^\top
$$

where $\mathbf{U}$ is an $m \times r$ orthonormal matrix (each row is one term's representation in the $r$-dimensional latent space), $\mathbf{S}$ is an $r \times r$ diagonal matrix of singular values in decreasing order, and $\mathbf{V}$ is an $n \times r$ orthonormal matrix (each row is one document's representation in the latent space). Note that [](#fig-svd-full) shows $\mathbf{V}^\top$ rather than $\mathbf{V}$, so the document representations appear as columns of $\mathbf{V}^\top$ (equivalently, rows of $\mathbf{V}$).

```{figure} images/figure_5_3.png
:name: fig-svd-full
:width: 90%

SVD decomposes the $m \times n$ term-document matrix $\mathbf{A}$ into three factors: $\mathbf{U}$ (term representations), $\mathbf{S}$ (singular values capturing topic importance), and $\mathbf{V}^\top$ (document representations).
```

The singular values on the diagonal of $\mathbf{S}$ decrease rapidly in magnitude. By retaining only the $k$ largest singular values and their corresponding columns in $\mathbf{U}$ and $\mathbf{V}$, we obtain the best rank-$k$ approximation of $\mathbf{A}$ under the Frobenius norm:

$$
\mathbf{A}_k = \mathbf{U}_k \mathbf{S}_k \mathbf{V}_k^\top
$$

```{figure} images/figure_5_4.png
:name: fig-svd-truncated
:width: 85%

Truncated SVD retains only $k$ dimensions. The reduced matrices $\mathbf{U}_k$, $\mathbf{S}_k$, $\mathbf{V}_k^\top$ provide compact representations: columns of $\mathbf{V}_k$ are $k$-dimensional document vectors, and columns of $\mathbf{U}_k$ are $k$-dimensional term vectors.
```

The $k$ dimensions no longer correspond to individual terms. Each dimension represents a latent topic, a pattern of term co-occurrence across documents. We can interpret a topic by examining the terms with the largest weights in the corresponding column of $\mathbf{U}_k$.

## Application in Text Retrieval

In text retrieval, $\mathbf{A}$ is typically the TF-IDF weighted term-document matrix with $m$ vocabulary terms and $n$ documents. After computing the truncated SVD:

- **Document representations**: columns of $\mathbf{V}_k$ (equivalently, rows of $\mathbf{V}_k^\top$) give $k$-dimensional vectors for each document.
- **Term representations**: columns of $\mathbf{U}_k$ give $k$-dimensional vectors for each term.
- **Topic interpretation**: the $i$-th column of $\mathbf{U}_k$ describes which terms contribute to the $i$-th latent topic.

### Folding in new documents and queries

When a new document $\mathbf{d}$ arrives (represented as a term vector in the original $m$-dimensional space), we project it into the $k$-dimensional latent space without recomputing the full SVD:

```{admonition} Key Formula: LSI Projection
:class: important

$$
\bar{\mathbf{d}}^\top = \mathbf{d}^\top \mathbf{U}_k \mathbf{S}_k^{-1}
$$

A new document (or query) vector $\mathbf{d}$ is mapped to the latent topic space by multiplying with $\mathbf{U}_k \mathbf{S}_k^{-1}$. This is valid as long as the new document does not significantly alter the collection's topic structure.
```

```{figure} images/figure_5_5.png
:name: fig-lsi-folding
:width: 85%

Derivation of the folding-in formula in three steps: transpose the truncated SVD equation, multiply by the pseudo-inverse, and isolate a single document vector.
```

Queries are treated as short documents. The query vector $\mathbf{q}$ (with non-zero entries for query terms) is projected identically:

$$
\bar{\mathbf{q}}^\top = \mathbf{q}^\top \mathbf{U}_k \mathbf{S}_k^{-1}
$$

Retrieval then computes cosine similarity between the projected query and all projected documents:

$$
\text{sim}_{\cos}(Q, D_i) = \frac{\bar{\mathbf{q}} \cdot \bar{\mathbf{d}}_i}{\|\bar{\mathbf{q}}\| \cdot \|\bar{\mathbf{d}}_i\|}
$$

## Worked Example

Consider a collection of 9 documents split into two groups: five about human-computer interfaces (c1-c5) and four about graph algorithms (m1-m4):

| ID | Title |
|----|-------|
| c1 | Human machine interface for Lab ABC computer applications |
| c2 | A survey of user opinion of computer system response time |
| c3 | The EPS user interface management system |
| c4 | System and human system engineering testing of EPS |
| c5 | Relation of user-perceived response time to error measurement |
| m1 | The generation of random, binary, unordered trees |
| m2 | The intersection graph of paths in trees |
| m3 | Graph minors IV: Widths of trees and well-quasi-ordering |
| m4 | Graph minors: A survey |

A search for "human computer interaction" should retrieve c1-c5, but not every document contains those exact terms. Traditional retrieval requires explicit query expansion.

After constructing the term-document matrix (using term frequencies, excluding stop words and terms appearing only once), we compute the SVD and truncate to $k=2$ topics.

```{figure} images/figure_5_7.png
:name: fig-svd-k2
:width: 90%

Truncated SVD with $k = 2$: the matrices $\mathbf{U}_k$ (12 terms × 2 topics), $\mathbf{S}_k$ (2 × 2), and $\mathbf{V}_k^\top$ (2 × 9 documents). Each document is now a 2D vector.
```

Projecting the query "human computer interaction" (only "human" and "computer" are in the vocabulary) into the 2D topic space yields $\bar{\mathbf{q}} = (0.138,\ {-0.028})$ (after normalizing sign conventions).

```{figure} images/figure_5_9.png
:name: fig-lsi-cosine-cone
:width: 70%

Documents and query in the 2D latent space. Topic 1 (horizontal) aligns with the interface documents c1-c5, while topic 2 (vertical) aligns with the graph documents m1-m4. The shaded angular cone shows all documents within angle $\alpha$ of the query vector.
```

The query vector points toward the c-documents. Using cosine similarity, documents rank as: c3 (0.997) > c1 (0.997) > c4 (0.979) > c2 (0.895) > c5 (0.846), with all c-documents retrieved before any m-document. Notably, c3 ("The EPS user interface management system") ranks highest despite containing neither "human" nor "computer". The SVD has recognized that its terms ("EPS", "user", "interface", "system") co-occur with query-related terms in other documents, connecting them through the latent topic structure.

The two topics can be interpreted from $\mathbf{U}_k$:
- **Topic 1**: 0.64·system + 0.40·user + 0.30·eps + 0.27·time + 0.27·response + 0.24·computer
- **Topic 2**: 0.62·graph + 0.49·trees + 0.45·minors + 0.27·survey

```{admonition} Full numeric SVD matrices
:class: note dropdown

The complete $\mathbf{U}$, $\mathbf{S}$, and $\mathbf{V}^\top$ matrices for this example are available in the accompanying demo notebook. The notebook also provides interactive visualization of the 2D document space and allows experimentation with different values of $k$.
```

## Limitations of LSI

LSI demonstrated that latent semantic structure improves retrieval, but several limitations prevented widespread adoption:

1. **No inverted index acceleration**: projected document vectors are dense, so every query requires comparison against all documents. The pruning strategies that make inverted indices fast (early termination, skip pointers) do not apply.

2. **Corpus-specific topics**: the learned mapping depends on both terms and documents in the collection. An LSI model trained on IT articles performs poorly on biomedical text. Unlike modern embeddings, LSI representations do not transfer across domains.

3. **Static vocabulary**: new terms can only be approximated through the existing topic structure. Two new terms in the same document receive identical representations, since both depend solely on that document's projection into $\mathbf{V}_k$.

4. **No sub-word awareness**: LSI operates at the word level and cannot generalize to morphological variants or misspellings not already in the vocabulary.

5. **Linear assumption**: SVD captures linear relationships between terms and documents. Non-linear semantic patterns (metaphor, irony, context-dependent meaning) are beyond its reach.

Despite these limitations, LSI established the foundational principle that underpins all modern semantic search: by reducing dimensionality, we can discover latent structure that connects terms sharing similar usage patterns, even when they never co-occur directly. The methods that follow in this chapter, from Word2Vec to transformers, all build on this insight while addressing LSI's specific shortcomings.
