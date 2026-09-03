---
author: Roger Weber
edition: HS26
status: not-reviewed
book_part: Foundations
chapter: Classical Text Retrieval
section: Vector Space Retrieval
order: "1.4"
---

(classical-text-vsm)=
# Vector Space Retrieval

The Vector Space Model emerged from the SMART information retrieval project led by Gerard Salton and colleagues in the 1960s, formalized in Salton, Wong, and Yang's [**A Vector Space Model for Automatic Indexing**](http://ctp.di.fct.unl.pt/~jmag/ir/papers/1975.A%20vector%20space%20model%20for%20automatic%20indexing.pdf), listed in the Further Reading section of the chapter summary. SMART introduced many ideas that became central to classical text retrieval: weighted terms, document and query vectors, partial matching, and ranked output. The model had a major influence on both research and practical search systems because it replaced exact logical decisions with simple numerical operations.

Consider the free-text query `cat dog forest`. Boolean OR retrieves every document containing at least one of these terms but cannot order the ten matches. The Vector Space Model uses the same tokenized collection and the same term statistics, but represents the query and documents as vectors. Their similarity becomes a score, allowing the system to place stronger matches first.

### What Remains the Same, and What Changes?

As before, preprocessing determines the vocabulary, and each distinct token defines one dimension. Documents remain bag-of-words representations, so word order is ignored and lexical variants such as "cat" and "cats" remain separate. IDF still gives less weight to terms that occur in many documents.

The query changes more fundamentally. Instead of a Boolean expression, it is treated as a short document and processed through the same pipeline. No AND or OR operators are required. A document may receive a positive score when it shares only one query term, and the score determines its position in the ranked result list.

### Document and Query Vectors

For a vocabulary of $M$ terms, document $D_i$ becomes the sparse vector $\mathbf{d}_i=(d_{i,1},\ldots,d_{i,M})$ with

$d_{i,j}=\text{tf}(D_i,t_j)\,\text{idf}(t_j), \qquad 1\leq j\leq M.$

The query becomes a vector in the same space:

$q_j=\text{tf}(Q,t_j)\,\text{idf}(t_j), \qquad 1\leq j\leq M.$

Placing the document vectors as columns produces the term-document matrix $\mathbf{A}$. This matrix is conceptually useful, but practical systems store only non-zero entries because each document contains a small fraction of the complete vocabulary.

### Inner Product and Cosine Similarity

The inner product, also called the dot product, sums the weighted evidence from terms shared by query and document:

```{note} Key Formula: Inner Product
```

Only query dimensions contribute because $q_j=0$ for every non-query term. Repeating a query term increases the score linearly. The score has no fixed upper bound and is often called a retrieval status value rather than a similarity probability. For the full collection,

$\mathbf{sim}_{\text{dot}}(Q,\mathbb{D})=\mathbf{A}^{\top}\mathbf{q}.$

Cosine similarity divides the same inner product by both vector lengths:

```{note} Key Formula: Cosine Similarity
```

Cosine similarity compares vector direction rather than magnitude. For non-negative TF-IDF vectors, its value lies between 0 for no shared terms and 1 for identical directions.Document vectors can be normalized once during indexing. The system then computes cosine similarity as an inner product between unit-length vectors, avoiding repeated norm calculations at query time.

### Evaluating a Query

Evaluation begins like Boolean OR. The system looks up the posting list for every query term and takes their union; documents outside this union have score zero. For each candidate, it accumulates $q_jd_{i,j}$ from the matching postings. This sum is already the inner-product score. Cosine evaluation adds one step: divide by the query norm and the document norm stored with the index. Finally, sort candidates by decreasing score.

This procedure avoids constructing the dense matrix $\mathbf{A}$ or comparing the query with every document. Chapter 4 develops inverted files and posting-list traversal in detail.

### Running Example

For $Q=\text{"cat dog forest"}$ and $N=12$, the query terms have

$\text{idf}(\text{cat})=\ln(12/5)\approx0.875,$

$\text{idf}(\text{dog})=\ln(12/6)\approx0.693,$

$\text{idf}(\text{forest})=\ln(12/7)\approx0.539.$

Each query term occurs once, so these values are also the non-zero components of $\mathbf{q}$. Applying the same TF-IDF representation to every document gives the complete ranking evidence below.

| Document | TF (`cat`, `dog`, `forest`) | Inner product | Cosine |
|---|---:|---:|---:|
| $D_1$ | (2, 2, 1) | 2.784 | 0.659 |
| $D_2$ | (0, 0, 0) | 0.000 | 0.000 |
| $D_3$ | (0, 1, 1) | 0.771 | 0.102 |
| $D_4$ | (1, 0, 0) | 0.766 | 0.093 |
| $D_5$ | (0, 0, 2) | 0.581 | 0.059 |
| $D_6$ | (0, 0, 1) | 0.291 | 0.034 |
| $D_7$ | (0, 4, 0) | 1.922 | 0.232 |
| $D_8$ | (0, 0, 4) | 1.162 | 0.139 |
| $D_9$ | (1, 1, 1) | 1.537 | 1.000 |
| $D_{10}$ | (2, 2, 2) | 3.075 | 0.342 |
| $D_{11}$ | (0, 0, 0) | 0.000 | 0.000 |
| $D_{12}$ | (1, 1, 0) | 1.247 | 0.137 |

The inner product ranks $D_{10}>D_1>D_7>D_9$. Repeating all query terms makes $D_{10}$ the strongest match, while repeating only "dog" allows $D_7$ to outrank the compact exact match $D_9$. Cosine produces a different order: $D_9>D_1>D_{10}>D_7$. The vector for $D_9$ has exactly the query direction and therefore receives the maximum score of 1. The additional non-query terms in $D_{10}$ increase its norm and reduce its cosine score even though it contains every query term twice.

```{note} Geometry of inner product and cosine (optional reading)
:class: dropdown

For a fixed score threshold $\gamma$, the inner product accepts documents satisfying $\mathbf{q}^{\top}\mathbf{d}\geq\gamma$. The boundary is a hyperplane with the query vector as its normal. Documents farther from the origin in the query direction receive larger scores. Since dimensions with $q_j=0$ vanish from the sum, the inner product effectively projects each document onto the subspace spanned by the query terms.

Cosine similarity instead defines a hypercone around the query vector. A higher threshold produces a narrower cone, so accepted documents must point in a direction closer to the query. Scaling every component of a document by the same factor does not change its cosine score. The measure therefore rewards similar proportions rather than high absolute frequencies.

This distinction explains $D_9$ and $D_{10}$. Both have query-term proportions $1:1:1$, but cosine is measured in the full vocabulary space. The many non-query components of $D_{10}$ contribute to $\|\mathbf{d}_{10}\|$ even though they contribute nothing to the numerator. They rotate the full document vector away from the query and lower its score.

For a one-term query, the geometry becomes extreme. The inner product primarily ranks documents by the frequency of that term. Cosine rewards vectors concentrated on that one dimension, so additional vocabulary lowers the score. A page containing little except the query term can therefore outrank a richer document that discusses the same topic in broader language.
```

### Limitations of Vector Space Retrieval

TF-IDF weights and both similarity measures are heuristic. They rank effectively in many collections, but neither formula directly estimates relevance. The inner product grows linearly with term frequency, so long documents tend to receive more query-term occurrences and authors can manipulate rankings by repeating selected terms. In the example, four occurrences of "dog" push $D_7$ above $D_9$ despite the absence of "cat" and "forest".

Cosine removes this magnitude bias but introduces different assumptions. It prefers documents whose term-frequency ratios resemble the query and uses every document term in the normalization. Consequently, non-query terms can lower a score, although they provide no negative evidence in the inner-product numerator. This is counter-intuitive when a relevant passage is embedded in a longer document.

The representation also retains the earlier lexical limitations. Terms are independent dimensions, so the model neither connects "cat" with "feline" nor distinguishes the meanings of "forest". Word order and dependencies such as "random forest" are absent from the bag-of-words vector.

```{warning} Term independence assumption
The Vector Space Model treats terms as independent dimensions. For example, "New" and "York" contribute separately even when their combination denotes one place. Later chapters introduce phrases, latent representations, and dense embeddings that capture some relationships between terms.
```

**Advantages**: The model accepts natural free-text queries, supports partial matches, produces ranked output, and gives discriminating terms more influence through IDF. Sparse vectors and inverted files make evaluation simple and efficient.

**Disadvantages**: Weighting and similarity are heuristic. Inner products favour high term frequencies and longer documents, while cosine can penalize useful additional content and favours query-like term ratios. Neither measure includes term-frequency saturation, robust length normalization, or semantic relationships.

### Modern Applications

Classical TF-IDF vectors with cosine similarity remain useful as transparent baselines and in small or specialized collections. They also support document similarity, clustering, classification, recommendation, and near-duplicate detection where a sparse lexical representation is sufficient. For primary lexical ranking, production search systems now commonly prefer BM25 because it controls term-frequency growth and document-length effects more carefully.

The geometry has become even more important than the original representation. Modern semantic search encodes text as dense embedding vectors and retrieves neighbours with inner product or cosine similarity, often through approximate nearest-neighbour indexes. Hybrid systems combine this dense vector retrieval with sparse lexical methods such as BM25. Thus, modern systems frequently retain the Vector Space Model's scoring operations while replacing TF-IDF dimensions with learned semantic features.

The Vector Space Model establishes the geometry of ranked retrieval but leaves term-frequency growth and document-length effects to heuristic choices. The next section introduces probabilistic term evidence and develops BM25 as a practical response to these limitations.
