---
author: Roger Weber
edition: HS26
status: not-reviewed
part: Foundations
chapter: Classical Text Retrieval
section: Fundamental Flows
order: "1.1"
---

(classical-text-fundamental-flows)=
# Fundamental Flows

## Retrieval Architectures

The first computerized retrieval systems automated what librarians had always done: match a user's query against a catalog of documents. Today, the same principles apply far beyond libraries, from legal databases and patent archives to corporate knowledge bases and web search engines.

```{admonition} From card catalogs to computers (optional reading)
:class: note dropdown

The need to search large text collections has its roots in library science. For centuries, librarians and researchers relied on card catalogs, subject indexes, and classification systems to organize knowledge. As collections grew into the millions, manual methods became impractical. In the 1960s, Gerard Salton's SMART system at Cornell demonstrated that computers could match queries against document representations automatically, laying the foundation for the field of information retrieval.
```

Consider a university library with millions of articles and books. A researcher enters terms like "neural plasticity AND memory consolidation". The simplest architecture retrieves all documents whose catalog entry satisfies this Boolean expression and returns the matching set. This retriever-only approach can evaluate each document independently and return results as soon as it finds them. No scoring or sorting is required. [Figure %s](#fig-retriever-architecture) illustrates this basic architecture.

```{figure} images/figure_1_1.png
:name: fig-retriever-architecture
:width: 90%

Retriever-only architecture: query enters, matching documents come out.
```

As the library's collection grows, a Boolean query like "neural plasticity" might return hundreds of articles. The researcher needs a way to narrow results without reformulating the query. A post-processing step adds **faceted search**: the ability to filter and sort results by metadata such as publication year, journal, or language. The researcher can toggle these filters while browsing without resubmitting the original search. Faceted search does not change which documents are considered relevant. It simply helps navigate large result sets. [Figure %s](#fig-two-stage-filter-sort) shows this extended pipeline.

```{figure} images/figure_1_2.png
:name: fig-two-stage-filter-sort
:width: 90%

Retriever with faceted filtering: candidates are filtered and sorted by metadata.
```

Even with filters, an unranked list of 200 articles forces the researcher to scan them manually. In the 1970s, the Vector Space and Probabilistic retrieval models introduced a fundamentally different idea: instead of a binary relevant/not-relevant decision, the system estimates a degree of relevance and produces a ranked list. The most promising articles appear first. The architecture becomes a two-stage pipeline: a fast retriever selects candidates, and a ranker scores and orders them ([Figure %s](#fig-retriever-ranker-pipeline)).

```{figure} images/figure_1_3.png
:name: fig-retriever-ranker-pipeline
:width: 90%

Retriever-ranker pipeline: a scoring model orders candidates by estimated relevance.
```

These three architectural patterns, retriever-only, retriever with faceted filtering, and retriever-ranker, represent the historical progression of text retrieval. The same patterns appear today in systems of all scales, from a researcher querying PubMed to a customer searching an e-commerce catalog.

## Documents and Retrieval Granularity

Before we can search a collection, we need to define what a "document" is. In retrieval, a document is the unit that gets indexed, matched against queries, and returned to the user. It consists of three elements:

- **Document ID** - a unique identifier (ideally a UUID) for efficient storage and lookup.
- **Metadata attributes** - descriptive fields (author, publication date, language) used for filtering and faceted search, but not necessarily included in full-text matching.
- **Content attribute(s)** - the text fields indexed for search and ranking.

The critical design decision is: what constitutes a single retrieval unit?

Consider a pharmacology reference book with 800 pages. If we treat the entire book as one document, a search for "ibuprofen dosage" returns "the pharmacology textbook" and the user must search within the book manually. If instead we treat each section or page as a separate document, the system can return "page 47: Ibuprofen - Dosage and Administration" directly.

This decision about retrieval granularity defines the collection structure. Finer granularity gives more precise results but increases the number of entries the system must manage. Coarser granularity reduces index size but forces users to locate relevant passages themselves. [Figure %s](#fig-document-chunking-split) illustrates the splitting operation.

```{figure} images/figure_1_10.png
:name: fig-document-chunking-split
:width: 90%

A single document divided into multiple smaller retrieval units.
```

For classical text retrieval, the choice is typically straightforward: emails are individual documents, web pages are individual documents, book chapters or sections become individual documents. The retrieval models in this chapter assume that this splitting has already occurred and each document in the collection is a coherent, self-contained unit.

```{note}
Choosing the right retrieval granularity becomes significantly more complex when building systems that feed passages to language models. We revisit chunking strategies, overlapping windows, and hierarchical approaches in the chapter on Retrieval-Augmented Generation.
```

## Offline Phase: Indexing Pipeline

Many search systems scan through data at query time. File search on a local computer, for example, reads through every file on disk whenever the user types a query. This works for a personal laptop with a few thousand files but not for a library with millions of articles. Instead, retrieval is split into two phases: an offline indexing phase that processes documents before any query arrives, and an online querying phase that handles user requests in real time.

The offline phase ingests raw documents, applies the splitting decision described above, extracts text and metadata, transforms text into feature representations, and builds a searchable index ([Figure %s](#fig-offline-indexing-pipeline)).

```{figure} images/figure_1_4.png
:name: fig-offline-indexing-pipeline
:width: 60%

Document preprocessing pipeline: ingest, split, extract, summarize, and index.
```

## Online Phase: Query Processing

When a researcher types "neural plasticity AND memory consolidation" into the library search, the online phase begins. The system first tokenizes the query and generates feature representations using the same pipeline that processed documents during indexing. This symmetry is essential: queries and documents must live in the same representation space for comparison to work.

Before matching, the system may refine the query through spelling correction or synonym expansion. For example, it might recognize that "memory consolidation" relates to "long-term potentiation" and broaden the search accordingly.

The query's feature representation is then compared against the representations stored in the index. Documents with similar representations are treated as potential matches. Finally, a ranking step scores each candidate and sorts the result list so that the most relevant articles appear first.

```{figure} images/figure_1_5.png
:name: fig-online-query-pipeline
:width: 60%

A five-stage text-based multimedia retrieval pipeline. Starting from a user query ("fairy tale"), the system performs tokenization (b), query expansion with related terms (c), vector space model search with similarity-based retrieval (d), and finally presents a ranked list of results to the user (e).
```

The central challenge in the online phase is relevance ranking: accurately estimating how important a document is given the query. The next sections develop the feature extraction pipeline that creates document representations, followed by the retrieval models that use those representations for scoring.
