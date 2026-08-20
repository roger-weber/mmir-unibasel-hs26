---
author: Roger Weber
edition: HS26
status: not-reviewed
part: Foundations
chapter: Classical Text Retrieval
section: Feature Extraction Pipeline
order: "1.2"
---

(classical-text-feature-extraction)=
# Feature Extraction Pipeline

Before the system can index documents or match user queries, it must turn raw documents into a structured, searchable form. The feature extraction pipeline ([Figure %s](#fig-text-indexing-pipeline)) transforms each document through a sequence of stages: text extraction from the source format, splitting into retrieval units, tokenization into individual terms, lemmatization and stemming to normalize word forms, and summarization into weighted feature vectors. This results in two key artifacts:

- **Index** - high-dimensional feature vectors for every retrieval unit, stored together with source metadata.

- **Vocabulary** - the complete set of normalized terms used for both document and query processing.

```{figure} images/figure_1_6.png
:name: fig-text-indexing-pipeline
:width: 100%

Text indexing pipeline: extract, split, tokenize, stem, summarize, then build the index and vocabulary.
```



## Text Extraction

The first step extracts plain text and metadata from the source format. Documents arrive in many formats: PDF, EPUB, plain text, HTML, or office documents (DOCX, PPTX). Each format encodes content differently, mixing actual text with layout instructions, styling, and structural markup. The extraction step strips formatting and control sequences, isolates the character stream, and records metadata attributes (author, title, date) for later use in filtering.

Consider the HTML example in [Figure %s](#fig-html-extraction-example). The header holds structured metadata (title, keywords) that can enrich the index entry, while the body contains the main text interleaved with markup tags. Extracting useful text from HTML requires parsing the DOM tree and distinguishing content nodes from navigation, scripts, and boilerplate. Although HTML follows a well-defined standard, real-world pages use diverse layouts, making robust extraction (often called scraping) a non-trivial engineering problem.

```{figure} images/figure_1_26.png
:name: fig-html-extraction-example
:width: 70%

HTML document structure: metadata in the header, content in the body.
```

Different source formats require different extraction strategies:

- **PDF**: Text is stored as positioned glyphs, not logical paragraphs. Extraction must reconstruct reading order from coordinates. Tables and multi-column layouts are particularly challenging.
- **Office documents (DOCX, PPTX)**: ZIP archives containing XML. Libraries can parse the XML structure to extract text, but embedded images and charts require separate handling.
- **HTML/XML**: Well-structured but noisy. Useful text must be separated from navigation menus, advertisements, and script blocks.
- **Plain text and Markdown**: Minimal transformation needed, but encoding detection (UTF-8 vs. legacy encodings) and line-break conventions still require attention.

In practice, we rarely build extraction from scratch. Modern extraction pipelines rely on established libraries and frameworks:

- **Apache Tika**: Java-based toolkit from the Apache Lucene ecosystem that auto-detects file types and extracts text and metadata from over 1,000 formats (PDF, DOCX, EPUB, email archives, images via OCR). Commonly used as the ingestion layer in Solr and Elasticsearch pipelines.
- **Unstructured** (`unstructured` on PyPI): Open-source Python library designed for LLM/RAG pipelines. Handles PDFs, HTML, images, and office documents with built-in chunking support.
- **MarkItDown** (`markitdown` on PyPI): Open-source Python library from Microsoft that converts PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, images, and audio into clean Markdown. Lightweight and popular for RAG preprocessing.
- **Trafilatura**: Focused on web content extraction. Strips boilerplate from HTML pages and returns clean article text.
- **PyMuPDF (fitz)**: Fast PDF text and layout extraction in Python, preserving reading order and table structure.
- **BeautifulSoup / lxml**: HTML/XML parsers for custom extraction logic when off-the-shelf tools fall short.

The choice of extraction tool depends on the document mix in the collection, the required fidelity (do you need tables? images? footnotes?), and whether the pipeline must run at scale.

At this point, we must also decide on a character encoding for the index. UTF-8 is the dominant standard and handles virtually all scripts. Legacy collections may contain documents in older encodings (Latin-1, Shift-JIS) that require detection and conversion before indexing.

```{admonition} Web documents and link structure
:class: seealso
HTML documents carry additional structure beyond text: anchor texts, heading hierarchies, and hyperlink networks. These provide valuable signals for retrieval but require specialized treatment. We cover HTML-specific extraction and link-based relevance (including PageRank) in the chapter on Web Retrieval.
```

## Splitting

As discussed in the previous section, retrieval granularity determines what constitutes a single retrieval unit. Once we have extracted the raw text from a document, we must split it into the units that will be indexed and returned to the user. In classical text retrieval, the splitting strategies are pragmatic and structure-driven:

- **By document boundary**: Each file, email, or web page becomes one retrieval unit. This is the most common approach when documents are naturally self-contained.
- **By structural element**: Books are split at chapter or section headings. Legal texts are split by article or paragraph number. The document's own structure defines the boundaries.
- **By line or log entry**: In observability systems like Elasticsearch, each log line becomes an independent retrieval unit. This enables searching millions of events by timestamp, severity, or content.
- **By fixed page or paragraph count**: When no structural markup is available (e.g., scanned documents), we split at regular intervals such as every N paragraphs or every page.

These approaches share a common property: the splitting decision is made once during indexing and remains fixed. Each resulting unit is treated as an independent document by the retrieval models that follow. [Figure %s](#fig-document-chunking-split) illustrates this for a novel split into chapter-sized retrieval units.

```{figure} images/figure_1_10.png
:name: fig-document-chunking-split
:width: 90%

A single document divided into multiple smaller retrieval units.
```

```{admonition} Advanced chunking strategies for RAG
:class: seealso
When building systems that feed passages to language models, splitting becomes significantly more nuanced. Overlapping windows, semantic chunking, and hierarchical approaches are covered in the chapter on Semantic Search.
```


## Tokenization

Tokenization converts the extracted character stream into a sequence of discrete units called tokens. A **token** is an individual element in this sequence: it can be a word, a number, or a punctuation mark. A **term** is the normalized form of a token after processing (stemming, case folding). In classical text retrieval, tokens are typically whole words, and the two concepts are often used interchangeably. In later chapters, we will see that this equivalence breaks down when tokens become subword fragments or multi-word phrases.

For classical retrieval, we split text at whitespace and punctuation boundaries to produce word-level tokens. This requires decisions about edge cases: how to handle hyphenated words ("state-of-the-art"), numbers ("3.14"), abbreviations ("U.S.A."), and possessives ("Watson's"). In languages without explicit word boundaries (e.g., Japanese, Chinese), segmentation requires dictionary-based or statistical methods.

The most significant challenge is variation in word forms. Since retrieval models match documents to queries by comparing tokens, two tokens are either identical or unrelated. There is no notion of "almost the same". A query for "cats" will not match a document containing only "cat" because these are distinct tokens. Similarly, "go", "goes", "went", and "going" are four independent dimensions in the vocabulary despite sharing a meaning. The next subsection addresses this through stemming and lemmatization, which reduce inflected forms to a common token before indexing. [Figure %s](#fig-tokenization-process) shows the tokenization step applied to a document collection.

```{figure} images/figure_1_13.png
:name: fig-tokenization-process
:width: 80%

Tokenization: raw text split into individual word-level tokens.
```

```{admonition} Beyond word-level tokens
:class: seealso
Subword tokenization (BPE, WordPiece) and n-gram approaches are covered in the chapter on Advanced Text Processing. Word and sentence embeddings, which map tokens to dense vectors, are introduced in the chapter on Semantic Search.
```

### Lemmatization and Stemming

The previous subsection established that retrieval models treat each token as an independent symbol: "cat" and "cats" are unrelated unless we normalize them to the same form. The goal of this pipeline stage is to reduce surface variation so that semantically equivalent word forms map to a single canonical token in the vocabulary.

Two approaches exist:

- **Stemming** applies rule-based suffix stripping to produce a common pseudo-stem. The stem is not necessarily a real word (e.g., "computing", "computation", "computer" all reduce to "comput"). Stemming is fast, requires no dictionary, and works well enough for most retrieval tasks. Its errors go in both directions: it can merge unrelated words ("universal" and "university" both stem to "univers") or fail to merge related ones ("alumnus" and "alumni" remain distinct).

- **Lemmatization** uses a dictionary or morphological analysis to map each word to its actual root form (the lemma). "Better" maps to "good", "went" maps to "go". This produces more accurate results but requires language-specific resources and is computationally more expensive. Modern NLP libraries like spaCy provide lemmatization out of the box.

For classical text retrieval, stemming is the standard choice due to its speed and simplicity. The most widely used stemmer for English is the Porter Algorithm, which we examine next.

```{admonition} Beyond inflection: compounds, synonyms, and semantic relations
:class: seealso
Some languages form compound words of arbitrary length (e.g., German, Finnish), creating additional matching challenges. Compound decomposition, synonym expansion, and other linguistic transformations are covered in the chapter on Advanced Text Processing.
```

### The Porter Algorithm

The most well-known rule-based stemmer for English is the Porter Algorithm (1980). It works by stripping suffixes through a series of rules, producing a "pseudo-stem" that is not necessarily a real English word but serves as a common form for matching. Consider these examples:

| Input | Porter stem | True root |
|-------|------------|-----------|
| "computing", "computation", "computer" | comput | compute |
| "retrieval", "retrieving", "retrieved" | retriev | retrieve |
| "generalization", "generalizing" | general | generalize |
| "relational" | relat | relate |
| "agreed", "agreeable" | agre | agree |

The stems "comput", "retriev", and "agre" are not dictionary words. They exist only to ensure that related forms map to the same token during indexing. When a user queries "computation", the system stems it to "comput" and matches documents containing "computing" or "computer" because those were also stemmed to "comput".

This approach is fast (a few hundred lines of code), requires no dictionary, and works well enough for most English retrieval tasks. However, it is purely mechanical and makes errors in both directions: over-stemming merges words that should remain distinct ("operate" and "operating" correctly merge, but "operation" and "operational" may conflate with "opera"), while under-stemming fails to merge irregular forms ("be", "was", "been" remain three separate tokens). [Figures %s](#fig-porter-stemmer-steps-1-2) and [%s](#fig-porter-stemmer-steps-3-5) show selected rules from the algorithm.

```{figure} images/figure_1_14.png
:name: fig-porter-stemmer-steps-1-2
:width: 90%

Selected rules from the Porter Stemmer algorithm (Steps 1 and 2). Each rule specifies an optional applicability condition on the stem and a suffix-replacement mapping.
```

```{figure} images/figure_1_15.png
:name: fig-porter-stemmer-steps-3-5
:width: 90%

Selected rules from Steps 3–5 of the Porter Stemming Algorithm, illustrating suffix-stripping transformations conditioned on the stem's measure `m`.
```

```{note}
The internal mechanics of Porter (consonant/vowel patterns, the "measure" variable `m`, and rule ordering) are covered in detail in the chapter on Advanced Text Processing, along with modern alternatives like Snowball and dictionary-based lemmatization.
```

## Vocabulary

After tokenization and stemming, we have a set of normalized tokens across the entire collection. The **vocabulary** is the complete set of distinct terms that appear at least once. Each term becomes a dimension in the feature space used for retrieval. But not all terms are equally useful.

### Stop Words

Many terms are grammatically necessary but carry no content. The article "the" appears in virtually every English document but tells us nothing about what a document is about. A search for "the" would return the entire collection, unable to differentiate between relevant and non-relevant results. [Figure %s](#fig-top50-term-frequencies) shows that the 50 most frequent terms in a news corpus account for roughly one-third of all occurrences and appear in over 60% of documents.

```{figure} images/figure_1_17.png
:name: fig-top50-term-frequencies
:width: 90%

Document frequency and collection term frequency for the 50 most frequent terms in a news corpus (~20,000 documents).
```

Stop word lists for most languages are readily available, for example on [Kaggle (stop words in 28 languages)](https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages). However, eliminating stop words entirely creates problems. Consider the search for "it": if we remove this term, we lose the ability to find IT books or the novel "It" by Stephen King. The modern approach retains all terms but assigns them different weights based on their discriminating power.

### Zipf's Law

The distribution of term frequencies follows a remarkably regular pattern. Let $N$ be the total number of token occurrences in the collection and $M$ the number of distinct terms. If we rank terms by decreasing frequency, Zipf's law states that the probability of encountering the term at rank $r$ is:

$$p_{r}=\frac{c}{r}=\frac{\text{tf}(t)}{N}, \quad \text{term } t \text{ with } \text{rank}(t)=r$$

The constant $c$ depends only on the vocabulary size:

$$c = \frac{1}{\sum_{r=1}^{M}\frac{1}{r}} \approx \frac{1}{0.5772 + \ln M}$$

For a collection with $M=5{,}000$ terms, $c \approx 0.11$; with $M=100{,}000$, $c \approx 0.08$. The practical implication is shown in [Figure %s](#fig-zipf-law-discrimination): terms fall into three zones. The most frequent terms (above the upper cut-off) appear everywhere and cannot discriminate. The rarest terms (below the lower cut-off) are highly specific but unlikely to match any query. The terms with the highest retrieval value lie in between.

```{figure} images/figure_1_18.png
:name: fig-zipf-law-discrimination
:width: 80%

Zipf's law and term discrimination: the most useful terms for retrieval fall between the frequency extremes.
```

### Term Discrimination and IDF

The intuition behind term weighting is straightforward: imagine removing a single term from every document in the collection. If documents suddenly become harder to tell apart, that term was helping to discriminate between them. Conversely, removing a term that appears everywhere (like "the") changes nothing. Terms that appear in a moderate number of documents have the highest discriminating power, as confirmed empirically in [Figure %s](#fig-discrimination-rank-vs-df).

```{figure} images/figure_1_19.png
:name: fig-discrimination-rank-vs-df
:width: 70%

Discrimination rank vs. document frequency in the Medlars collection (450 documents). Peak discrimination occurs around df = 13.
```

Karen Spärck Jones (1972) captured this intuition in a single formula. The **document frequency** $\text{df}(t)$ counts how many documents contain term $t$ at least once. The **inverse document frequency** weights terms inversely to their commonness:

```{admonition} Key Formula: Inverse Document Frequency
:class: important

$$\text{idf}(t) = \log\frac{N+1}{\text{df}(t)+1} = \log(N+1) - \log(\text{df}(t)+1)$$

Terms that appear in fewer documents receive higher IDF weights. A term appearing in 10 out of 10,000 documents is far more informative than one appearing in 5,000.
```

IDF provides the weighting mechanism we need: common terms get low weights, rare terms get high weights. Combined with term frequency, it produces the **tf·idf** weighting that we will use throughout the retrieval models in the next sections. [Figure %s](#fig-idf-vs-discrimination-power) compares IDF weights with empirical discrimination power, confirming that IDF closely approximates the true discriminating value of terms.

```{figure} images/figure_1_20.png
:name: fig-idf-vs-discrimination-power
:width: 70%

IDF weights closely track empirical discrimination power as a function of document frequency (N = 1,000).
```

## Document Representation

With the vocabulary established and each term assigned an IDF weight, we can now represent documents as vectors. Each document becomes a point in an $M$-dimensional space where $M$ is the vocabulary size and each dimension corresponds to one term.

### Set-of-Words and Bag-of-Words

Let $D_{i}$ be a document. The feature representation $\mathbf{d}_{i}\in \mathbb{R}^{M}$ is a vector where component $d_{i,j}$ corresponds to term $t_{j}$. We use $\text{tf}(D_{i},t_{j})$ to denote the number of occurrences of term $t_{j}$ in document $D_{i}$.

The simplest option is the **set-of-words** model: for each term, record only whether it is present (1) or absent (0). A document mentioning "cat" three times looks identical to one mentioning it once:

$$d_{i,j}=\begin{cases}1 & \text{tf}(D_{i},t_{j})>0 \\ 0 & \text{tf}(D_{i},t_{j})=0\end{cases}$$

A richer option is the **bag-of-words** model: record how many times each term appears. A document mentioning "cat" five times is represented differently from one mentioning it once:

$$d_{i,j}=\text{tf}(D_{i},t_{j})$$

Both models ignore term proximity and order. A document containing "New York" is represented the same way as one containing "York New". Despite this limitation, these representations work remarkably well because the presence and frequency of specific terms are strong signals of relevance.

### TF·IDF Weighting

The bag-of-words model treats all terms equally: a document with 10 occurrences of "the" scores the same as one with 10 occurrences of "algorithm". By combining term frequency with IDF, we produce weighted vectors where informative terms contribute more:

$$d_{i,j} = \text{tf}(D_{i}, t_{j}) \cdot \text{idf}(t_{j})$$

This **tf·idf** weighting is the standard document representation for classical retrieval models. The retrieval models in the next sections build directly on these weighted vectors. [Figure %s](#fig-bag-of-words-pipeline) shows the complete pipeline from raw text to tf·idf vectors.

```{figure} images/figure_1_16.png
:name: fig-bag-of-words-pipeline
:width: 90%

From raw text to weighted feature vectors: tokenize, stem, count term frequencies, apply IDF weights.
```

### Sparsity

In practice, a vocabulary can include millions of terms. However, most documents contain only a few hundred or thousand unique terms. The feature vectors are therefore sparsely populated: the vast majority of components are zero. Efficient storage methods like the inverted file exploit this sparsity by recording only the non-zero entries. During retrieval, the system only needs to consider documents that share at least one term with the query.

```{admonition} Hands-on: Feature Extraction
:class: hint
Build a feature extraction pipeline in Python: tokenize text, apply stemming, compute term frequencies and IDF weights.
[Open notebook -->](https://github.com/mmir-unibasel-hs26/mmir-unibasel-hs26/blob/main/02-ClassicalTextRetrieval/01-extract-features.ipynb)

*Includes pre-run results. You can read through or download and experiment.*
```
