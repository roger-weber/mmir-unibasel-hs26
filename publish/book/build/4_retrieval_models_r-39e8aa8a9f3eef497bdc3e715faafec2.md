---
author: Roger Weber
edition: HS26
status: not-reviewed
part: Foundations
chapter: Classical Text Retrieval
section: Ranking Models
order: "1.4"
---

(classical-text-vsm)=
# Ranking Models

## Vector Space Retrieval

The initial version of the vector space retrieval model was introduced in the SMART retrieval system by Salton et al. It remains the most widely used classical retrieval model, and we will explore advanced extensions and implementations in this chapter for state-of-the-art retrieval performance.

Unlike Boolean methods, the vector space retrieval model treats documents and queries as vectors in a high-dimensional feature space. It employs vector-based similarity metrics for ranking. A document $D_i$ is represented as a vector $\mathbf{d}_i$, utilizing idf-weighted term frequencies. Unlike the extended Boolean models, we refrain from normalizing the term frequencies:

$$d_{i,j} = \text{tf}(D_i, t_j) \cdot \text{idf}(t_j) \quad \forall j: 1 \leq j \leq M$$

All document representations can be merged into the term-document matrix $\mathbf{A}$. Each column in $\mathbf{A}$ corresponds to a document, and each row represents a term in the vocabulary. Hence, matrix element $a_{j,i} = d_{i,j}$, following the convention of addressing matrix elements by rows and then columns.

While we illustrate the method using the term-document matrix and outline matrix-vector operations for score computation, practical implementations do not store or utilize matrix calculations due to the matrix's sparsity, where many elements are 0 as documents usually have only a few terms. We will explore more efficient evaluation techniques in subsequent parts of this chapter.

Queries are depicted as sparse vectors, denoted as $\mathbf{q}$. Unlike Boolean expressions, a query is treated as a mini-document or search prompt, following identical processing steps and vocabulary use as documents:

$$q_j = \text{tf}(Q, t_j) \cdot \text{idf}(t_j) \quad \forall j: 1 \leq j \leq M$$

### Inner Vector Product

The inner vector product uses the dot-product between the query and document vector. When applied to the entire collection, we multiply the term-document matrix by the query vector and then rank documents based on decreasing similarity values. It is important to note that similarity here is not confined to a range between 0 and 1, and literature often refers to it as retrieval status value (RSV):

$$\text{sim}(Q, D_i) = \mathbf{q} \cdot \mathbf{d}_i = \sum_{j=1}^{M} q_j \cdot d_{i,j}$$

$$\mathbf{sim}(Q, \mathbb{D}) = \mathbf{A}^\top \mathbf{q}$$

The formula shows that only query terms impact the similarity score, with terms absent in the query yielding a value of 0 for $q_j \cdot d_{i,j}$, irrespective of their frequency in documents. In contrast, documents with larger $d_{i,j}$ values for query terms, that is more term occurrences, receive higher ranks. Notably, significant terms with higher $\text{idf}$ values have more influence, and this influence is amplified due to $\text{idf}$ weighting in both queries and documents. Finally, we observe the 'partial-match' capability of the model. If a document shares at least one term with the query, then the score is positive.

### Cosine Similarity

The cosine measure calculates the angle between document and query vectors. It implies that documents need to contain query terms for high scores. Absence of query terms widens the angle between the vectors, leading to lower scores.

```{admonition} Key Formula: Cosine Similarity
:class: important

$$\text{sim}(Q, D_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \cdot \|\mathbf{d}_i\|} = \frac{\sum_{j=1}^{M} q_j \cdot d_{i,j}}{\sqrt{\sum_{j=1}^{M} q_j^2} \cdot \sqrt{\sum_{j=1}^{M} d_{i,j}^2}}$$

Measures the angle between query and document vectors. Normalizing by vector lengths removes the bias toward longer documents. Values range from 0 (orthogonal, no shared terms) to 1 (identical direction).
```

Similar to the inner vector product, scores for all documents can be calculated through matrix-vector multiplications. For this, we normalize the query vector by its size and introduce a diagonal matrix $\mathbf{L}$ with inverse document lengths to dynamically normalize document vectors:

$$\mathbf{sim}(Q, \mathbb{D}) = \mathbf{L} \mathbf{A}^\top \mathbf{q}' \quad \text{with } \mathbf{L} = \text{diag}\left(\frac{1}{\|\mathbf{d}_1\|}, \ldots, \frac{1}{\|\mathbf{d}_N\|}\right) \text{ and } \mathbf{q}' = \frac{\mathbf{q}}{\|\mathbf{q}\|}$$

Alternatively, we could normalize document and query vectors during the extraction step and save normalized versions. This makes the inner vector product and the cosine measure equivalent since vectors have a length of 1.

### Geometric Interpretation

For a simplified visualization of vector space retrieval, documents are projected into the smaller query vector space spanned by the query terms, while other dimensions have no effect on search order:

- Using the inner vector product, a hyperplane through the origin is established with the query vector as its normal. Documents farther from this plane are considered more relevant.

- On the other hand, the cosine measure creates hyper-cones with the query vector as their axis. Higher cosine values correspond to smaller angles of a hyper-cone embedding the document.

- Documents lacking query terms are placed at the origin, yielding a value of 0 with both measures. This allows us to disregard such documents and focus on those containing at least one query term. This leads to efficient retrieval methods explored later using inverted files.

- An issue arises when query terms are similar (e.g., 'house' and 'villa'), as they might not affect results unless pre-processing merges them. This limitation is common in classical retrieval techniques, often addressed by automatically expanding queries with related terms.

### Worked Example

```{figure} images/figure_1_21.png
:name: fig-toy-collection-query
:width: 80%

A toy document collection (D₁–D₃) and a three-term query Q = {gold, silver, truck}, with query term occurrences highlighted in red. This example motivates term-based relevance scoring.
```

```{admonition} Example
:class: example
Let's examine a simple collection of three documents:

- $D_1$: "Shipment of **gold** damaged in a fire"
- $D_2$: "Delivery of **silver** arrived in a **silver truck**"
- $D_3$: "Shipment of **gold** arrived in a **truck**"

Query: $Q$ = "gold silver truck"

We extract terms, find document frequencies, and compute IDF weights. Using $\text{idf}(t_j) = \log N - \log \text{df}(t_j)$ with $N = 3$:

With the inner vector product, the similarity scores yield:

$$\mathbf{sim}(Q, \mathbb{D}) = \begin{bmatrix} 0.031 \\ 0.486 \\ 0.062 \end{bmatrix}$$

**Ranking**: $D_2 > D_3 > D_1$

$D_2$ scores highest because it contains two of the three query terms with "silver" appearing twice, and "truck" is a rare term (high IDF).
```

```{figure} images/figure_1_22.png
:name: fig-vsm-worked-example
:width: 80%

TF-IDF vector space retrieval example with three documents and a three-term query, showing the term-document matrix, IDF weights, and resulting similarity scores.
```

**Advantages**: Extremely simple and intuitive query model. Term weights have a good impact on the scores and differentiate between query terms, e.g., reducing the impact of stop words in the query. Easy to implement and highly efficient in calculation. Outperforms Boolean models and can rival top retrieval methods. Naturally supports partial match queries, and documents do not have to include all query terms for high similarity values.

**Disadvantages**: Heuristic similarity scores with little intuition why they work well (no theoretical background for the model). The similarity measures are not robust and can be biased by authors (spamming of terms). If documents are of different lengths, scores can vary significantly due to the higher term occurrences in larger documents. Main assumption of retrieval model is independence of terms which may not hold true in typical scenarios (see synonyms and homonyms).

```{admonition} Term independence assumption
:class: warning
All models discussed so far assume that terms are statistically independent. This means "New" and "York" are treated as unrelated dimensions. In reality, term co-occurrences carry semantic information. This limitation is addressed in later chapters through phrase detection, n-grams, and eventually dense embeddings.
```

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
