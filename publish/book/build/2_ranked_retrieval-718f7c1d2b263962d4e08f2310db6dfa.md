---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Index for Text Retrieval
section: Ranked Retrieval over Inverted Files
order: "4.2"
---

(indexing-ranked-retrieval)=
# Ranked Retrieval over Inverted Files

Boolean retrieval returns an unordered candidate set. The ranked models from [Classical Text Retrieval](../ch01_classical_text_retrieval/0_index.md), the binary independence model (BIR), the vector space model, and BM25, go further: they assign each document a score so we can return the best matches first. This section shows that all three run over the same inverted index. What changes is the content of the postings and the function that turns them into a score.

## Retriever and ranker

Ranked retrieval separates into two stages, shown in [](#fig-retriever-ranker). The **retriever** uses the inverted index to gather candidates, taking the union of the postings lists of the query terms, the same merge as a Boolean `OR`. The **ranker** then scores each candidate with the model's scoring function and keeps the top results. An optional filter can drop candidates that fail a metadata condition before ranking; we return to filtering in the next section.

```{figure} images/figure_4_6.png
:name: fig-retriever-ranker
:width: 85%

The two-stage pipeline: a retriever gathers candidates from the index, and a filter-and-ranker scores them into a ranked list.
```

The retriever only needs candidates with at least one query term, because a document with no query term scores zero under all three models. This is the same efficiency argument as before: we score a small candidate set, not the whole collection.

## Two evaluation strategies

There are two natural orders in which to walk the postings and accumulate scores. We show both using the BIR model because its scoring function, a sum of per-term weights $c_j$, is additive and involves no term frequencies or document lengths, so the evaluation logic is fully visible without extra lookups.

### Term-at-a-time (TAAT)

The most direct extension of the set-based Boolean evaluation from the previous section processes one query term fully before moving to the next. For each term we walk its postings list and add that term's $c_j$ weight to a running score for the document. After all terms are processed, the score dictionary holds every candidate's final score, and we extract the top $k$.

```python
def search_TAAT(query, k):
    query_vector = analyzer.set_of_words(query)
    # filter terms and obtain c_j-weights
    # returns a list of pairs (term_j, c_j)
    term_weights = query_weights(query_vector)

    scores = defaultdict(int)

    # iterate over terms and their postings
    for (term, weight) in term_weights:
        for doc_id in index[term]:
            # add this term's c_j to the document's running score
            scores[doc_id] += weight

    # avoid a full sort: use a heap to extract the top k
    topk = TopKList(k)
    for doc_id, score in scores.items():
        topk.add(doc_id, score)
    return topk
```

TAAT mirrors what we did with Boolean sets: read one term's postings, then the next, then combine. The difference is that instead of intersecting or uniting, we accumulate weights. Its simplicity comes at a cost: the `scores` dictionary can grow very large when a query contains common terms with long postings lists, and we cannot prune early because a document's score is incomplete until the last term is done.

### Document-at-a-time (DAAT)

DAAT instead extends the sorted-stream merge from the end of the previous section. Rather than processing one term fully, it advances all query-term streams in parallel and produces one candidate at a time, in document-ID order. Each candidate receives its complete score the moment it appears, so we can feed it directly into a bounded top-$k$ heap.

```python
def search_DAAT(query, k):
    query_vector = analyzer.set_of_words(query)
    # filter terms and obtain c_j-weights
    term_weights = query_weights(query_vector)

    # get iterators for each term and fetch the first posting
    iters = [iter(index[term]) for (term, _) in term_weights]
    nexts = [next(it, None) for it in iters]

    topk = TopKList(k)

    while not all(e is None for e in nexts):
        # the smallest doc ID across all stream heads
        smallest = min(nexts, key=lambda x: x or math.inf)

        # score: sum c_j for every term whose head equals smallest
        score = 0
        for j in range(len(nexts)):
            if nexts[j] == smallest:
                score += term_weights[j][1]
        topk.add(smallest, score)

        # advance every stream whose head was the smallest
        for i, e in enumerate(nexts):
            if e == smallest:
                nexts[i] = next(iters[i], None)

    return topk
```

DAAT reads postings as streams and never builds a full score dictionary. Memory is bounded by the heap size $k$. Because the full score is known the moment a document is emitted, DAAT can also prune: once the heap is full, it can skip candidates whose maximum possible score cannot enter the top $k$ (techniques such as WAND exploit this). The price is slightly more complex bookkeeping in the inner loop.

### Comparison

Both strategies read the same postings and consider the same candidates, so their asymptotic cost is similar. TAAT is the clearer extension of the Boolean set approach and a fine choice for short queries. DAAT is generally preferred in production because of its bounded memory and its ability to prune.

## The three models over postings

The models differ only in what the postings store and how a candidate is scored.

For **BIR**, the postings need only document identifiers. Each query term $t_j$ carries a weight $c_j$ derived from relevance feedback, and a document's score is the sum of the $c_j$ over the query terms it contains. No term frequencies or document lengths are involved.

For the **vector space model**, the DAAT and TAAT patterns remain the same, but instead of summing a single $c_j$ per query term, we sum the product $d_j \cdot q_j$ for each query term present in the document. Expanding the TF-IDF weights, each term's contribution is $\text{tf}(D, t_j) \cdot \text{tf}(Q, t_j) \cdot \text{idf}(t_j)^2$. As a consequence, the postings for term $t_j$ must store not only the document identifier but also the term frequency $\text{tf}(D, t_j)$ so the ranker can compute that product.

Using the inner product alone (the sum above) already gives a useful ranking that favors documents sharing many weighted terms with the query. The **cosine measure** refines this by normalizing both vectors:

$$\text{sim}_{\cos}(Q, D) = \frac{\sum_{j} d_j \cdot q_j}{\lVert \mathbf{d} \rVert \cdot \lVert \mathbf{q} \rVert}.$$

The query norm $\lVert \mathbf{q} \rVert$ is not a problem: we have the full query vector and can compute it once per query. The document norm $\lVert \mathbf{d} \rVert$ is the difficulty. During evaluation, whether DAAT or TAAT, we see only the subset of document components that overlap with the query terms, not the full document vector, so we cannot compute the norm from what we read. Two options exist:

1. **Store the norm separately**, in a document-length table indexed by document identifier. This requires an extra random lookup per candidate and additional storage.
2. **Normalize all document vectors at index time**, dividing each stored TF-IDF weight by the document's norm before writing it. The postings then hold $\hat{d}_j = \text{idf}(t_j) \cdot \text{tf}(D, t_j) / \lVert \mathbf{d} \rVert$, and the cosine reduces to a dot product that the ranker can evaluate directly from the postings.

The second option is fast at query time but requires IDF values to remain fixed. If the collection grows or shrinks enough to significantly shift document frequencies, the stored weights and norms become stale and the collection must be reindexed. This is one reason why BM25 avoids the cosine measure altogether and uses a plain sum instead. BM25 normalizes by document length through an explicit parameter ($b$ and $\text{avgdl}$) that can be stored cheaply alongside each posting or looked up once per document, without requiring precomputed vector norms or assuming constant IDF values.

For **BM25**, the postings again store term frequencies, and the ranker additionally needs the document length $\lvert D \rvert$, the average document length $\text{avgdl}$, and the parameters $k$ and $b$.

```{admonition} Key Formula: BM25 over postings
:class: important

$$\text{sim}_{\text{BM25}}(Q, D) = \sum_{j} \text{idf}(t_j) \cdot \frac{\text{tf}(D, t_j)\,(k + 1)}{\text{tf}(D, t_j) + k\left(1 - b + b\,\dfrac{\lvert D \rvert}{\text{avgdl}}\right)}, \quad \text{idf}(t_j) = \log \frac{N - \text{df}(t_j) + 0.5}{\text{df}(t_j) + 0.5}$$

Each query term contributes a saturating term-frequency factor, scaled by an inverse document frequency and normalized by document length. Rare terms in short documents score highest.
```

```{admonition} IDF is not one fixed quantity
:class: warning

The symbol $\text{idf}$ appears in the vector space model, in BM25, and in Lucene, but the formulas differ. The classic form is $\log(N / \text{df})$; BM25 uses $\log\frac{N - \text{df} + 0.5}{\text{df} + 0.5}$; Lucene adds one inside the logarithm, $\log\left(1 + \frac{N - \text{df} + 0.5}{\text{df} + 0.5}\right)$, to keep the value positive. These variants track the same idea, rarer terms weigh more, but their numeric values are not interchangeable. When you compare scores, make sure they come from the same formula.
```

### A worked BM25 example

We index the titles of the 50-book library collection from [Performance Evaluation](../ch02_performance_evaluation/0_index.md), applying the standard pipeline of tokenization, stop-word removal, and stemming. This gives $N = 50$ documents with an average title length of $\text{avgdl} = 2.72$ tokens. We run the query "database systems", which the pipeline reduces to the stems "databas" and "system", with $k = 1.2$ and $b = 0.75$.

```{admonition} Example
:class: example

Document frequencies and BM25 IDF weights for the query terms:

- "databas": $\text{df} = 1$, so $\text{idf} = \log\frac{50 - 1 + 0.5}{1 + 0.5} = 3.4965$.
- "system": $\text{df} = 3$, so $\text{idf} = \log\frac{50 - 3 + 0.5}{3 + 0.5} = 2.6080$.

The top-scoring document is "Database Systems: The Complete Book", whose title stems to four tokens ("databas", "system", "complet", "book"), so $\lvert D \rvert = 4$. Both query terms occur once ($\text{tf} = 1$). The length-normalized denominator is the same for both terms:

$$\text{tf} + k\left(1 - b + b\,\frac{\lvert D \rvert}{\text{avgdl}}\right) = 1 + 1.2\left(0.25 + 0.75 \cdot \frac{4}{2.72}\right) = 2.6235.$$

Each term contributes $\text{idf} \cdot \frac{1 \cdot (1.2 + 1)}{2.6235}$, giving $3.4965 \cdot \frac{2.2}{2.6235} = 2.9320$ for "databas" and $2.6080 \cdot \frac{2.2}{2.6235} = 2.1869$ for "system". The document score is their sum, $5.1190$.

The next results score far lower because they match only "system": "Operating System Concepts" scores $2.5026$ and "Computer Systems: A Programmer's Perspective" scores $1.9420$. The rare term "databas" is what separates the top result from the rest.
```

## Precomputing to avoid extra lookups

The dot product at the heart of the vector space model can be computed straight from the postings. The cosine measure and BM25, however, need per-document data that is not in a single postings list: the vector norm $\lVert \mathbf{d} \rVert$ for cosine, and the document length $\lvert D \rvert$ for BM25. Fetching that data for every candidate adds a random lookup per document and can dominate query cost.

Two optimizations help. We can store the IDF weight next to each posting so the ranker does not consult the vocabulary during a query. And we can normalize the document vectors at index-build time: once we fix $\text{idf}$, $k$, $b$, $\lvert D \rvert$, and $\text{avgdl}$, all three measures reduce to a dot product between a normalized document vector and the query vector. The cost is flexibility. If any normalization parameter changes, the index must be rebuilt. The alternative, storing the raw per-document data with each posting, keeps parameters adjustable at the price of larger postings and more data to read.

```{admonition} Hands-on: Ranked Retrieval over an Inverted Index
:class: hint
Implement document-at-a-time and term-at-a-time evaluation, and compare BIR, vector space, and BM25 rankings on the library collection.
[Open notebook -->](https://github.com/mmir-unibasel-hs26/mmir-unibasel-hs26/blob/main/05-IndexForTextRetrieval/01-boolean-retrieval.ipynb)

*Includes pre-run results. You can read through or download and experiment.*
```
