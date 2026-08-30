---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Index for Text Retrieval
section: Filtering, Facets, and Compression
order: "4.3"
---

(indexing-filtering-facets-compression)=
# Filtering, Facets, and Compression

Real queries rarely ask for text alone. A user searching a library wants "database systems" among books published after 2000, or filtered to the Computer Science shelf. And a production index must stay small enough to keep its hot data in memory. This section adds two capabilities to the inverted index: metadata filtering, including the faceted navigation built on top of it, and the compression that keeps postings compact.

## Filtering on metadata

A predicate is a condition on document metadata, such as `year > 2000` or `genre = "Computer Science"`. There are three ways to combine such a predicate with a text query, differing in when and how the predicate is evaluated.

- **A priori filtering** evaluates the predicate first. We keep document metadata in a structure that answers the condition efficiently, a B-tree or an inverted list on the attribute, and look up the set of document identifiers that satisfy it. That set is passed into the retriever, which drops any candidate not in it during the postings merge. This is the best option when the predicate can be indexed, because it removes non-matching documents before scoring.
- **A posteriori filtering** evaluates the predicate last. When no index exists for the attribute, we rank first and check the predicate only on the documents we are about to return, pulling from the top-$k$ heap in score order and skipping those that fail. For a loose predicate we test only the few documents we return plus a handful skipped. For a highly selective predicate we may have to walk deep into the heap, but that is still cheaper than testing every document in the collection.
- **Inline filtering** evaluates the predicate during the merge. We store the attribute in a stream aligned with the postings and ordered by document identifier, and advance it in lockstep with the postings streams, testing the condition as each candidate appears. This keeps filtering and scoring in a single pass.

In each case the retrieval algorithm itself is unchanged. Filtering slots into the same place where relevance feedback already removes documents: a candidate that fails the predicate is simply not scored or not returned.

## Faceted search

Faceted search is filtering turned into navigation. Instead of a free-form predicate, the user picks from categorical attributes, a genre, a publication decade, a content type, and the system both filters the results and reports how many documents fall under each option. This is the a priori approach specialized to categories: the index keeps a separate postings list for each facet value, so filtering by a facet is an intersection with that list, and the facet counts are the list lengths.

Two refinements reduce the storage this adds. **Clustered indexing** groups documents that share facet values, so their postings lists overlap heavily and can be encoded relative to a shared reference list. **Hierarchical facet compression** exploits nested facets: a geographic hierarchy like Country then State then City stores only the differences between levels rather than the full path for every document.

## Compressing postings lists

Compression is one of the highest-leverage optimizations in an inverted index. Cutting postings storage by a factor of four or more lets more of the index sit in memory, and reading fewer bytes from disk directly lowers query latency.

The key observation is that a postings list is a sorted sequence of document identifiers. Rather than storing each identifier, we store the gaps between consecutive identifiers, the **d-gaps**. In a list like 95673, 127088, ... we store the first identifier and then the gap 31415. Gaps are much smaller than absolute identifiers, especially for frequent terms whose postings are dense, and small integers compress well.

### Variable-byte encoding

Variable-byte (VByte) encoding is the most widely used byte-aligned scheme. It uses the most significant bit of each byte as a continuation flag and the remaining seven bits for data. Every byte except the last has its high bit set to 1; the final byte has its high bit set to 0 to mark the end of the number.

```{admonition} Example
:class: example

Encode the d-gap 31415. In binary (15 bits) this is `0000001 1110101 0110111`, already shown split into three seven-bit groups from the most significant end. The groups have the values 1, 117, and 55. We set the continuation bit (high bit = 1) on all bytes except the last:

| group value | seven bits | byte (with flag) | hex |
|---|---|---|---|
| 1 | `0000001` | `1000 0001` | `81` |
| 117 | `1110101` | `1111 0101` | `F5` |
| 55 | `0110111` | `0011 0111` | `37` |

The encoding is the three bytes `81 F5 37`. To decode, read bytes until one has a high bit of 0, strip the flags, and concatenate the seven-bit payloads: $1 \cdot 128^2 + 117 \cdot 128 + 55 = 31415$.
```

### Other schemes

VByte is simple and fast, but block-based schemes compress better. **PForDelta** (patched frame-of-reference delta) processes fixed blocks of, typically, 128 d-gaps. For each block it picks the smallest bit width $b$ that fits about 90% of the values and stores those in $b$ bits each; the roughly 10% that do not fit are recorded separately as exceptions. This packs the common case tightly while still handling outliers. Another byte-aligned family uses a two-bit length marker per value (one to four bytes), keeping data byte-aligned for fast decoding and reaching roughly 15 to 20 percent of the uncompressed size.

## Metadata alongside postings

Modern indexes store more than document identifiers in a postings list, and the same compression applies to the extra data. **Term frequencies** are written as small integers after each identifier, enabling TF-IDF and BM25 scoring. Because BM25's saturation function flattens the contribution of high term frequencies, values above a modest ceiling barely affect the score. This means term frequencies can be capped and encoded in just a few bits (four bits suffice in practice) without noticeably hurting retrieval accuracy. **Length-normalization data**, such as document lengths and per-field statistics used by BM25, is precomputed at index time and kept in compressed form, trading a little storage for less work per query. Storing these values next to the postings is what lets the ranker of the previous section avoid a separate lookup for every candidate.
