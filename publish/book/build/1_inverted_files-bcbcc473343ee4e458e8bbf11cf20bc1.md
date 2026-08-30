---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Index for Text Retrieval
section: Inverted Files
order: "4.1"
---

(indexing-inverted-files)=
# Inverted Files

Consider a tiny collection of three documents:

- $D_1$: "The cat sat on the mat"
- $D_2$: "The dog chased the cat"
- $D_3$: "The cat and the dog played"

A query for "cat" should return $D_1$, $D_2$, and $D_3$; a query for "dog" should return $D_2$ and $D_3$. With three documents we could simply read each one and check. With three million documents we cannot. We need a structure that takes us straight from a query term to the documents that contain it, without touching the rest of the collection. That structure is the inverted index, and this section explains why it makes query cost depend on the query rather than on the size of the collection.

## Why scanning does not scale

Recall from [Classical Text Retrieval](../ch01_classical_text_retrieval/0_index.md) that a document is represented as a sparse vector over the vocabulary. Assume a collection of $N$ documents and a vocabulary of $M$ terms. On average a document contains $K$ distinct terms, with $K$ much smaller than $M$, and a typical query contains about $L$ terms, with $L$ much smaller than $K$ (five query terms is a common figure).

The most direct storage scheme reserves one entry for every term in every document: $N \cdot M$ entries in total. In the set-of-words model each entry is a single bit that records whether the term occurs; the bag-of-words model stores a term frequency instead. Almost all of these entries are zero, because each document uses only a small fraction $K/M$ of the vocabulary. Storing the full matrix is therefore wasteful, and a sparse layout that keeps only the $K$ non-zero entries per document reduces storage from $N \cdot M$ to $N \cdot K$ entries.

The sparse layout saves space, but it does not save time. To answer a query we still scan every document, and for each one we read terms that the query never asked about. Here we rely on a property of the classical models: Boolean retrieval, the vector space model, and BM25 all score a document from its query terms alone, under the assumption that terms contribute independently. Only the entries for the query terms affect the result, so almost everything we read is discarded. This is exactly the property the inverted index will exploit.

This property is what makes classical retrieval fast, and it is also its limit. It does not hold for dense retrieval, covered in [Semantic Search](../ch05_semantic_search/index.md), where a document's relevance depends on its whole embedding rather than on the presence of individual query terms. A dense system cannot restrict its attention to a handful of postings, which is why classical lexical retrieval remains orders of magnitude faster than dense retrieval and why the two are often combined.

## Inverting the layout

Since only the query terms matter, we can organize storage around terms instead of documents. Rather than storing, for each document, the list of terms it contains, we store, for each term, the list of documents that contain it. This is the inverted index, also called the inverted file. Each term points to a **postings list**: the document identifiers in which the term appears. [](#fig-inverted-index-example) shows the inverted index for the three-document collection above.

```{figure} images/figure_4_2.png
:name: fig-inverted-index-example
:width: 50%

An inverted index over a three-document collection: each term in the vocabulary points to its postings list.
```

Inverting the layout does not change how much we store. There are still $N \cdot K$ term-document pairs. What changes is how much we read per query. A query touches only the postings lists of its query terms. If the $N \cdot K$ pairs are spread over $M$ terms, an average postings list holds $N \cdot K / M$ entries, so a query of $L$ terms reads about $N \cdot K \cdot L / M$ entries instead of all $N \cdot K$.

```{admonition} Key Formula: Query read cost
:class: important

$$\text{entries read} \approx \frac{N \cdot K \cdot L}{M} = (N \cdot K) \cdot \frac{L}{M}$$

The inverted index reads only a fraction $L/M$ of the data a full scan would touch. The cost scales with the query length $L$ and grows far more slowly than the collection, though as we see below it does not vanish entirely at web scale.
```

```{admonition} Example
:class: example

With a query of $L = 5$ terms and a vocabulary of $M = 1{,}000{,}000$ terms, the fraction $L/M$ is five millionths: a query reads about five millionths of what a full scan would touch. To see what that means in bytes, assume four bytes per posting (a 32-bit document identifier) and estimate the data read per query across three collections.

| Collection | $N$ documents | $K$ terms/doc | $M$ vocabulary | Full index | Read per query |
|---|---:|---:|---:|---:|---:|
| E-book titles | 30 million | 6 | 1 million | ~0.7 GB | ~3.5 KB |
| Code search | 50 million | 8 | 500,000 | ~1.5 GB | ~16 KB |
| Web pages | 10 billion | 600 | 30 million | ~21.8 TB | ~3.8 MB |

The read per query is the average postings length $N \cdot K / M$ times the $L = 5$ query terms. Even though the web index is tens of thousands of times larger than the book index, an average query still reads only a few megabytes.
```

The read cost is proportional to $N \cdot K$, so inverted files scale linearly with the collection: double the documents and both the index and the average read roughly double. For small to large collections this is comfortable. The book and code indexes, under a gigabyte and around a gigabyte and a half, fit entirely in memory on one machine, so a query with a few keywords is answered almost instantly. The average-term assumption behind those reads often overstates the cost here, because someone searching a book catalog is usually after a specific title and types rare, discriminative words whose postings lists are shorter than average. Under practical conditions, inverted files work very well across this whole range.

The trouble is common words. Stop words and other high-frequency terms have postings lists far longer than average, and they are what makes large collections expensive. On the book and code collections they are a nuisance rather than a blocker: we can drop stop words from the query, searching "Lord of the Rings" as "lord" and "ring" while ignoring "of" and "the", and lean on the rare, discriminative terms. At web scale the same query behaves differently. A stop word on half of 10 billion pages carries roughly 19 GB of postings on its own, and even after dropping it, "lord" and "ring" each occur in hundreds of millions of pages, some close to 1 GB of postings apiece. No amount of stop-word removal rescues a plain postings scan when the discriminative terms are themselves frequent.

The web index also no longer fits on one machine at more than 20 TB, so it lives mostly on disk and must be distributed. Inverted files degrade gracefully rather than suddenly, remaining practical across a wide range of sizes, but the combination of an enormous collection and unavoidably frequent terms is what eventually breaks the simple scheme. Keeping web-scale queries fast despite multi-gigabyte postings lists is the job of the additional strategies we return to later: skip pointers and early termination, pruned and tiered indexes, caching, and the sharding covered in a later section.

Query length is a cost lever in the same way, which matters for query expansion. Expanding a query, introduced in [Advanced Text Processing](../ch03_advanced_text_processing/4_query_understanding.md), adds related terms to raise recall, but every added term is another postings list to read and merge, and the extra terms add noise that can lower precision. On a large collection, where a single frequent term can already cost gigabytes, expansion has to be weighed against query cost, not against precision alone.

## The structure of the index

An inverted index has three parts. The **vocabulary** (or dictionary) holds the $M$ distinct terms and serves as the lookup key. Each term points to its **postings list**. A separate **document table** stores per-document metadata, such as a title or a URL, that the ranker and the result page need but that is not itself searched term by term.

For Boolean retrieval over the set-of-words model, the postings lists need only the document identifiers; term frequencies and document frequencies are not required. As documents are added, their identifiers are appended to the postings of the terms they contain. When documents arrive in order, each postings list stays sorted by increasing document identifier. That sorted order is not incidental: it is what makes the evaluation below efficient.

## Boolean retrieval over postings

A Boolean query combines terms with `AND`, `OR`, and `NOT`. Because each postings list is a set of document identifiers, the operators map directly onto set operations:

- `expr1 AND expr2`: the intersection of the two postings sets.
- `expr1 OR expr2`: the union of the two postings sets.
- `expr1 AND NOT expr2`: the set difference, the documents of `expr1` that are not in `expr2`.

Nesting `AND` and `OR` combines these operations, and the same rules generalize to more than two operands. The `NOT` operator needs care. We never materialize `NOT expr2` on its own, because its complement can contain almost the whole collection. A pure negation, or an `OR` with a negated operand such as "cat OR NOT dog", would force us to enumerate millions of identifiers. Such a query is also rarely what a user means: "cat OR NOT dog" selects every document except those that contain "dog" but not "cat". We therefore allow `NOT` only inside an `AND` that has at least one non-negated operand, and we apply the negated parts last, as a subtraction from the candidates found so far.

### Merging sorted postings as streams

Loading whole postings lists into memory does not scale to lists with millions of entries. Instead we read them as sorted streams and merge them, advancing one entry at a time, much like merging sorted lists. Suppose the postings are:

- cat: 1, 4, 10
- dog: 3, 4, 8, 10, 12

To evaluate "cat AND dog" we look at the head of each stream and always advance the smaller one. When both heads are equal, that document satisfies the `AND`, so we emit it and advance both streams.

| cat head | dog head | comparison | action | output |
|---|---|---|---|---|
| 1 | 3 | 1 < 3 | advance cat | |
| 4 | 3 | 4 > 3 | advance dog | |
| 4 | 4 | equal | emit, advance both | 4 |
| 10 | 8 | 10 > 8 | advance dog | |
| 10 | 10 | equal | emit, advance both | 10 |
| exhausted | 12 | cat empty | stop | |

The result is the sorted list 4, 10. The merge stops as soon as one stream is exhausted: once cat runs out, no later document can match, even though dog still has the entry 12.

The other operators follow the same pattern with a different emit rule. For "cat OR dog" we emit the smaller head at each step and advance the stream it came from, producing the union 1, 3, 4, 8, 10, 12. For "cat AND NOT dog" we emit a cat entry only when it is strictly smaller than the current dog head, producing 1. Because every operator consumes sorted streams and emits a sorted stream, the operators compose: the output of one merge feeds directly into the next.

## From matching to ranking

Boolean evaluation answers a yes-or-no question: does a document satisfy the query? It produces a candidate set, but no order within it. For anything beyond exact filtering we want the best documents first, which means scoring each candidate. That splits retrieval into two stages: a **retriever** that uses the inverted index to gather candidates, and a **ranker** that scores them. The next section shows how the ranked models from Chapter 1, the binary independence model, the vector space model, and BM25, all run over the same postings lists.

```{admonition} Hands-on: Inverted Index and Boolean Retrieval
:class: hint
Build an inverted index over a small collection and evaluate Boolean queries by merging sorted postings.
[Open notebook -->](https://github.com/mmir-unibasel-hs26/mmir-unibasel-hs26/blob/main/05-IndexForTextRetrieval/01-boolean-retrieval.ipynb)

*Includes pre-run results. You can read through or download and experiment.*
```
