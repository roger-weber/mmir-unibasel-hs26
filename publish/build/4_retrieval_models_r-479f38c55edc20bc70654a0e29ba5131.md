---
author: Roger Weber
edition: HS26
status: not-reviewed
part: Foundations
chapter: Classical Text Retrieval
section: Vector Space Retrieval
order: "1.4"
---

(classical-text-vsm)=
# Vector Space Retrieval

The Vector Space Model emerged from the SMART information retrieval project led by Gerard Salton and colleagues in the 1960s. SMART introduced many ideas that became central to classical text retrieval: weighted terms, document and query vectors, partial matching, and ranked output. The model had a major influence on both research and practical search systems because it replaced exact logical decisions with simple numerical operations.

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

$\text{sim}_{\text{dot}}(Q,D_i)=\mathbf{q}^{\top}\mathbf{d}_i=\sum_{j=1}^{M}q_jd_{i,j}.$

Only query dimensions contribute because $q_j=0$ for every non-query term. Repeating a query term increases the score linearly. The score has no fixed upper bound and is often called a retrieval status value rather than a similarity probability. For the full collection,

$\mathbf{sim}_{\text{dot}}(Q,\mathbb{D})=\mathbf{A}^{\top}\mathbf{q}.$

Cosine similarity divides the same inner product by both vector lengths:

```{admonition} Key Formula: Cosine Similarity
:class: important

$\text{sim}_{\cos}(Q,D_i)=\frac{\mathbf{q}^{\top}\mathbf{d}_i}{\|\mathbf{q}\|\,\|\mathbf{d}_i\|}=\frac{\sum_{j=1}^{M}q_jd_{i,j}}{\sqrt{\sum_{j=1}^{M}q_j^2}\sqrt{\sum_{j=1}^{M}d_{i,j}^2}}$

Cosine similarity compares vector direction rather than magnitude. For non-negative TF-IDF vectors, its value lies between 0 for no shared terms and 1 for identical directions.
```

Document vectors can be normalized once during indexing. The system then computes cosine similarity as an inner product between unit-length vectors, avoiding repeated norm calculations at query time.

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

```{admonition} Geometry of inner product and cosine (optional reading)
:class: note dropdown

For a fixed score threshold $\gamma$, the inner product accepts documents satisfying $\mathbf{q}^{\top}\mathbf{d}\geq\gamma$. The boundary is a hyperplane with the query vector as its normal. Documents farther from the origin in the query direction receive larger scores. Since dimensions with $q_j=0$ vanish from the sum, the inner product effectively projects each document onto the subspace spanned by the query terms.

Cosine similarity instead defines a hypercone around the query vector. A higher threshold produces a narrower cone, so accepted documents must point in a direction closer to the query. Scaling every component of a document by the same factor does not change its cosine score. The measure therefore rewards similar proportions rather than high absolute frequencies.

This distinction explains $D_9$ and $D_{10}$. Both have query-term proportions $1:1:1$, but cosine is measured in the full vocabulary space. The many non-query components of $D_{10}$ contribute to $\|\mathbf{d}_{10}\|$ even though they contribute nothing to the numerator. They rotate the full document vector away from the query and lower its score.

For a one-term query, the geometry becomes extreme. The inner product primarily ranks documents by the frequency of that term. Cosine rewards vectors concentrated on that one dimension, so additional vocabulary lowers the score. A page containing little except the query term can therefore outrank a richer document that discusses the same topic in broader language.
```

### Limitations of Vector Space Retrieval

TF-IDF weights and both similarity measures are heuristic. They rank effectively in many collections, but neither formula directly estimates relevance. The inner product grows linearly with term frequency, so long documents tend to receive more query-term occurrences and authors can manipulate rankings by repeating selected terms. In the example, four occurrences of "dog" push $D_7$ above $D_9$ despite the absence of "cat" and "forest".

Cosine removes this magnitude bias but introduces different assumptions. It prefers documents whose term-frequency ratios resemble the query and uses every document term in the normalization. Consequently, non-query terms can lower a score, although they provide no negative evidence in the inner-product numerator. This is counter-intuitive when a relevant passage is embedded in a longer document.

The representation also retains the earlier lexical limitations. Terms are independent dimensions, so the model neither connects "cat" with "feline" nor distinguishes the meanings of "forest". Word order and dependencies such as "random forest" are absent from the bag-of-words vector.

```{admonition} Term independence assumption
:class: warning
The Vector Space Model treats terms as independent dimensions. For example, "New" and "York" contribute separately even when their combination denotes one place. Later chapters introduce phrases, latent representations, and dense embeddings that capture some relationships between terms.
```

**Advantages**: The model accepts natural free-text queries, supports partial matches, produces ranked output, and gives discriminating terms more influence through IDF. Sparse vectors and inverted files make evaluation simple and efficient.

**Disadvantages**: Weighting and similarity are heuristic. Inner products favour high term frequencies and longer documents, while cosine can penalize useful additional content and favours query-like term ratios. Neither measure includes term-frequency saturation, robust length normalization, or semantic relationships.

### Modern Applications

Classical TF-IDF vectors with cosine similarity remain useful as transparent baselines and in small or specialized collections. They also support document similarity, clustering, classification, recommendation, and near-duplicate detection where a sparse lexical representation is sufficient. For primary lexical ranking, production search systems now commonly prefer BM25 because it controls term-frequency growth and document-length effects more carefully.

The geometry has become even more important than the original representation. Modern semantic search encodes text as dense embedding vectors and retrieves neighbours with inner product or cosine similarity, often through approximate nearest-neighbour indexes. Hybrid systems combine this dense vector retrieval with sparse lexical methods such as BM25. Thus, modern systems frequently retain the Vector Space Model's scoring operations while replacing TF-IDF dimensions with learned semantic features.

(classical-text-probabilistic)=
## Probabilistic Retrieval

The primary criticism of the existing models lies in their heuristic nature. While they perform well, their correctness lacks a solid foundation. Probabilistic retrieval provides a formal approach based on probabilities. $P(R|D_i)$ is the probability that a document $D_i$ is relevant for a query $Q$, and $P(NR|D_i) = 1 - P(R|D_i)$ is the probability that it's not relevant. The similarity value between query $Q$ and document $D_i$ is then defined as:

$$\text{sim}(Q, D_i) = \frac{P(R|D_i)}{P(NR|D_i)} = \frac{P(R|D_i)}{1 - P(R|D_i)}$$

### The Binary Independence Model (BIR)

The BIR model is a straightforward approach grounded in several key assumptions:

- Term frequency does not matter (utilizing a set-of-words document model)
- Term independence (consistent with previous models)
- Terms absent from the query do not influence ranking (if a term is absent from the query, it's assumed to be equally distributed among relevant and non-relevant documents)

Given these assumptions, we apply Bayes' theorem to the conditional probabilities:

$$\text{sim}(Q, D_i) = \frac{P(R) \cdot P(D_i|R)}{P(NR) \cdot P(D_i|NR)}$$

These new probabilities can be interpreted as follows: $P(R)$ and $P(NR)$ represent the probabilities that a randomly selected document is relevant and not relevant, respectively. $P(D_i|R)$ and $P(D_i|NR)$ are the probabilities that document $D_i$ belongs to the set of relevant and non-relevant documents, respectively.

Leveraging the assumption of binary document vectors and term independence:

$$P(D_i|R) = \prod_{\forall j:\, d_{i,j}=1} P(d_{i,j}=1|R) \cdot \prod_{\forall j:\, d_{i,j}=0} P(d_{i,j}=0|R)$$

Let's use a compact notation. Define $r_j = P(d_{i,j}=1|R)$ as the probability of a relevant document having the term $t_j$, and $n_j = P(d_{i,j}=1|NR)$ as the probability of a non-relevant document having the term $t_j$:

$$\text{sim}(Q, D_i) = \frac{P(R)}{P(NR)} \cdot \prod_{\forall j:\, d_{i,j}=1} \frac{r_j}{n_j} \cdot \prod_{\forall j:\, d_{i,j}=0} \frac{1 - r_j}{1 - n_j}$$

It is important to observe that $P(R)$ and $P(NR)$ are solely determined by the query and do not affect the document ranking as they linearly scale the similarity values.

Applying the third assumption (non-query terms occur with equal probability in relevant and non-relevant documents), and taking the logarithm:

```{admonition} Key Formula: BIR Similarity
:class: important

$$\text{sim}(Q, D_i) \sim \sum_{\forall j:\, d_{i,j}=1,\, q_j=1} c_j \quad \text{with } c_j = \log \frac{r_j \cdot (1 - n_j)}{n_j \cdot (1 - r_j)}$$

Documents are scored by summing $c_j$-values only for query terms present in the document. Each $c_j$ captures how much more likely term $t_j$ is to appear in relevant vs. non-relevant documents.
```

```{admonition} Example
:class: example
Consider a collection of $N = 4$ documents and a query $Q$ = "information retrieval". Suppose we know from relevance feedback that:

- $r_{\text{information}} = 0.8$ (80% of relevant documents contain "information")
- $n_{\text{information}} = 0.4$ (40% of non-relevant documents contain "information")
- $r_{\text{retrieval}} = 0.9$ (90% of relevant documents contain "retrieval")
- $n_{\text{retrieval}} = 0.2$ (20% of non-relevant documents contain "retrieval")

Compute $c_j$ values:

$$c_{\text{information}} = \log \frac{0.8 \cdot (1 - 0.4)}{0.4 \cdot (1 - 0.8)} = \log \frac{0.8 \cdot 0.6}{0.4 \cdot 0.2} = \log \frac{0.48}{0.08} = \log 6 \approx 1.79$$

$$c_{\text{retrieval}} = \log \frac{0.9 \cdot (1 - 0.2)}{0.2 \cdot (1 - 0.9)} = \log \frac{0.9 \cdot 0.8}{0.2 \cdot 0.1} = \log \frac{0.72}{0.02} = \log 36 \approx 3.58$$

For a document $D_i$ containing both terms: $\text{sim}(Q, D_i) = 1.79 + 3.58 = 5.37$

For a document $D_k$ containing only "information": $\text{sim}(Q, D_k) = 1.79$

The term "retrieval" contributes more because it is much more likely to appear in relevant documents than in non-relevant ones.
```

### Estimating Parameters

To calculate the $c_j$-values, the BIR model starts with initial estimates for a first result list, and then refines these estimates based on user feedback.

**Initial Estimates** (no feedback available):

$$r_j = 0.5, \quad n_j = \frac{\text{df}(t_j) + 0.5}{N + 1} \quad \forall j: q_j = 1$$

**Estimates with Feedback**: In each iteration, we ask the user to rate the $K$ retrieved documents. Let $L$ be the number of documents marked as relevant, $k_j$ be the number of retrieved documents containing term $t_j$, and $l_j$ be the number of retrieved and relevant documents containing term $t_j$:

$$r_j = \frac{l_j + 0.5}{L + 1}, \quad n_j = \frac{k_j - l_j + 0.5}{K - L + 1} \quad \forall j: q_j = 1$$

We employ the values 0.5 and 1 in the formulas above to avoid numerical problems (division by zero). The more user feedback we gather, the more accurate the estimates for $r_j$ and $n_j$ become. However, users might be reluctant to provide feedback.

**Advantages**: The BIR model establishes similarity values on a probabilistic basis through basic assumptions. Document ranking depends on the likelihood of being relevant for the query. Only query terms are necessary for similarity calculations, and the inverted file method offers efficient evaluation. The model performs well, especially after some feedback iterations. It also accommodates partial match queries.

**Disadvantages**: The basic assumptions of the BIR model may not always be valid. Term independence is not universally applicable. The document ranking in BIR doesn't consider term frequencies or the discrimination power of terms. Not all users are willing to assist the system with feedback.

(classical-text-bm25)=
## Okapi Best Match 25 (BM25)

The Okapi BM25 ranking function was developed at London's City University and is rooted in Karen Spärck Jones' probabilistic framework from the 1970s and 1980s. It is notably applied in Lucene, the engine behind Solr, Elasticsearch, and OpenSearch — three widely used systems for observability, security analytics, and full-text search. BM25 builds on the vector space model, enhancing it with a probabilistic approach to relevance evaluation.

Some limitations in the previously discussed models stem from heuristic approaches to identify relevant documents. Researchers developed better frameworks for relevance assessment, driven by key observations:

- **Query Term Significance**: the presence or absence of query terms is crucial for relevance assessment
- **Partial Matches**: not all relevant documents contain every query term
- **Document Length**: longer documents have more terms, but shorter relevant ones should score well too
- **Term Specificity**: rare words often carry more meaning than common ones
- **Term Saturation**: while term frequency matters, overly frequent terms should not dominate
- **Fine Tuning**: flexibility to adjust ranking based on search context
- **Efficiency**: efficient retrieval and relevance assessment are essential
- **User Feedback**: if available, integration of user feedback for improved search quality
- **Term Proximity**: closeness of query terms in a document may indicate higher semantic relevance
- **Term Dependence**: recognizing term dependencies, like matching query 'animals' to 'cats' or 'dogs' in documents

BM25 addresses these observations or provides ways to consider them. We will cover Efficiency in the upcoming chapter on indexing structures and explore Term Proximity and Term Dependence in the chapter on natural language processing methods.

### Term Frequency Saturation

Term frequencies play a crucial role in determining document relevance. Typically, we assume that a document's relevance is linked to the frequency of query term occurrences within it. This notion led to the creation of the $\text{tf} \cdot \text{idf}$ vector component description. Nonetheless:

- A document with the search term 'cat' occurring a hundred times is certainly relevant, but it should not be considered twice as relevant as a document with 50 occurrences of 'cat'. In essence, the linear factor $\text{tf}$ exaggerates the relevance. It also makes the method vulnerable to spamming attacks.

- Shorter documents have fewer occurrences of terms compared to much longer documents. However, they can be equally or even more relevant. Yet, the $\text{tf} \cdot \text{idf}$ scheme tends to favor longer documents with higher term frequencies.

A simple adjustment like using $\sqrt{\text{tf}}$ instead of $\text{tf}$ does not provide significant improvement. We require a function that levels off after a certain occurrence threshold:

$$\widehat{\text{tf}}_k = \frac{\text{tf} \cdot (k+1)}{\text{tf} + k}$$

Typically, $k \in [1, 2]$ with Lucene using $k = 1.2$. The updated values $\widehat{\text{tf}}_k$ saturate relatively swiftly to the value 2.2 with $k = 1.2$, whereas unsaturated $\text{tf}$ and $\sqrt{\text{tf}}$ values increase without limit. The factor $(k+1)$ scales values but does not impact ranking.

```{figure} images/figure_1_23.png
:name: fig-tf-weighting-comparison
:width: 70%

Comparison of three term frequency weighting schemes as a function of raw term frequency: linear raw TF (blue), square root TF (orange), and BM25 saturated TF with $k = 1.2$ (green). BM25 saturation prevents any single term from dominating the score.
```

### Document Length Normalization

Now, let's examine document length. Lengthier documents include more terms and should saturate at a slower rate than shorter ones. BM25 employs a summation across all query terms, similar to the inner vector product, while modifying the core formula to account for document length:

$$\widehat{\text{tf}}_k(D) = \frac{\text{tf} \cdot (k+1)}{\text{tf} + k \cdot \left(1 - b + b \cdot \frac{|D|}{adl}\right)}$$

- with $b = 0.75$ (adjustable), $|D|$ the length of document $D$, and $adl$ the average length of documents in the collection
- If $|D|$ is smaller than $adl$ (short document), then $(1 - b + b \cdot |D|/adl) < 1$ and values $\widehat{\text{tf}}_k(D)$ saturate faster
- If $|D|$ is large (long document), then $(1 - b + b \cdot |D|/adl) > 1$ and values $\widehat{\text{tf}}_k(D)$ saturate slower
- $b \in [0, 1]$ is a hyperparameter that steers the impact of document length. Higher values prefer shorter documents
- $adl$ does not have to be the accurate average length; rather, we can consider it as another hyperparameter to define what 'long' / 'short' means

```{figure} images/figure_1_24.png
:name: fig-bm25-tf-length-normalization
:width: 70%

BM25 term frequency weighting curves for $k = 1.2$ and $b = 0.75$, comparing the baseline (no length normalization) with short-document and long-document curves. Short documents saturate faster; long documents saturate slower.
```

### BM25 IDF from the BIR Model

We previously discussed $\text{idf}$-weights without providing a rationale for using that specific formula. BM25 approaches term weighting probabilistically. From the BIR model, with no user feedback ($L = l_j = 0$, $K = N$, $k_j = \text{df}(t_j)$):

$$\text{idf}_{\text{BM25}}(t_j) = \log \frac{N - \text{df}(t_j) + 0.5}{\text{df}(t_j) + 0.5}$$

Note that for terms $t_j$ that appear in over 50% of the documents, the logarithm yields a negative value. Lucene uses a variant that avoids negative values:

$$\text{idf}_{\text{Lucene}}(t_j) = \log\left(1 + \frac{N - \text{df}(t_j) + 0.5}{\text{df}(t_j) + 0.5}\right) = \log\left(\frac{N + 1}{\text{df}(t_j) + 0.5}\right)$$

```{figure} images/figure_1_25.png
:name: fig-idf-variants-comparison
:width: 70%

Comparison of three IDF variants — IDF+1, BM25 IDF, and Lucene IDF — as a function of document frequency for $N = 1{,}000$. The Lucene variant avoids negative values while closely approximating the original IDF+1 curve.
```

### The Complete BM25 Formula

```{admonition} Key Formula: BM25
:class: important

$$\text{sim}_{\text{BM25}}(Q, D_i) = \sum_{j=1}^{M} \log \frac{N - \text{df}(t_j) + 0.5}{\text{df}(t_j) + 0.5} \cdot \frac{\text{tf}(D_i, t_j) \cdot (k+1)}{\text{tf}(D_i, t_j) + k \cdot \left(1 - b + b \cdot \frac{|D_i|}{adl}\right)}$$

BM25 combines probabilistically grounded IDF with saturating, length-normalized term frequency. Parameters: $k \in [1, 2]$ controls tf saturation (default 1.2), $b \in [0, 1]$ controls length penalty (default 0.75), $adl$ is average document length.
```

- Unlike the vector space retrieval model, the $\text{idf}$-values are applied only once and query term frequency is not considered. Later we will examine Lucene's scoring function, which expands the above formula with extra components, including query term frequencies and additional term and document weighting.

- In this fundamental formulation, BM25 encompasses three hyperparameters ($k$, $b$, $adl$) that allow fine-tuning the scoring function to match the requirements of the search context.

```{admonition} BM25 is not just "better TF-IDF"
:class: warning
Students often confuse BM25 with a simple TF-IDF variant. The key differences: (1) BM25 has a *saturation* function — doubling tf does not double the score; (2) document length normalization is built into the tf component, not applied as a separate step; (3) the IDF formula has a probabilistic derivation from the BIR model, not just a heuristic log ratio.
```

## Model Comparison

| Property | Boolean | Extended Boolean | Vector Space (TF-IDF) | BM25 |
|----------|---------|-----------------|----------------------|------|
| Ranking | No (filter only) | Yes (heuristic) | Yes (cosine/dot) | Yes (probabilistic) |
| Partial matches | No | Yes | Yes | Yes |
| Term weighting | Equal (all or nothing) | tf · idf (normalized) | tf · idf | Saturating tf · probabilistic idf |
| Document length | Not considered | Normalized to [0,1] | Cosine normalizes | Built-in length factor ($b$) |
| Query language | Boolean expression | Boolean expression | Free-text (bag of words) | Free-text (bag of words) |
| Theoretical basis | Set theory | Heuristic | Heuristic (geometric) | Probabilistic (BIR) |
| Typical use | Faceted filters, exact match | Rarely used alone | Legacy systems | Modern search (Lucene, ES) |
| Key strength | Precision control, explainability | Ranked Boolean | Simplicity, partial match | Best balance of all factors |
| Key weakness | No ranking, no partial match | Weak scoring theory | Length bias, no saturation | Requires tuning ($k$, $b$) |

```{admonition} Hands-on: Retrieval Models
:class: hint
Implement Boolean, TF-IDF, and BM25 retrieval on a small collection and compare their rankings.
[Open notebook -->](https://github.com/mmir-unibasel-hs26/mmir-unibasel-hs26/blob/main/05-IndexForTextRetrieval/01-boolean-retrieval.ipynb)

*Includes pre-run results. You can read through or download and experiment.*
```
