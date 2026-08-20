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


## Tokenization

A token is formed by a sequence of characters. Typically, we use complete words to create tokens, but there are other options which we will explore in later chapters. Here is a brief overview:

The simplest approach uses characters or subwords as tokens. For example, breaking "street" into 3-character fragments yields "str" and "eet". Large language models frequently use subword tokenization to maintain a small, fixed-size vocabulary while still encoding previously unseen words.

The primary approach in classical text retrieval is to use complete words as tokens. This requires decisions about how to handle special characters, numbers, and abbreviations. In certain languages, word boundaries may not be evident (e.g., Japanese and Chinese). The most significant challenge arises from variations in word forms. For instance, "cat" and "cats" are semantically related but count as different tokens. Stemming is a linguistic method to merge such variants, enabling better control over vocabulary size and term matching.

A third option is to treat multi-word expressions as single tokens. Phrases like "San Francisco", "Salt Lake City", or "Prime Minister" consistently appear together and carry a meaning distinct from their individual words. While such phrases can be added to the vocabulary manually, we will explore automated methods to detect them in later chapters.

For now, we use words as tokens. This is sufficient for classical retrieval methods, and we build on it with more advanced tokenization in later chapters.

```{figure} images/figure_1_13.png
:name: fig-tokenization-process
:width: 80%

Tokenization of a raw text corpus. Each document in the collection undergoes tokenization, where the text is split into individual tokens (words and punctuation).
```

## Lemmatization and Stemming

Lemmatization and linguistic transformation are essential for matching query terms with document terms, even when they have different inflections or spellings (e.g., "colour" vs. "color").

The most common technique is **stemming**. In most languages, words appear in various inflected forms based on tense, case, or gender. In English, we see "go", "goes", "went", and "going"; in German, "gehen", "gehst", "ging", and "gegangen". These forms differ on the surface but convey the same core meaning. Stemming reduces all inflected forms to a common stem and uses that stem as the canonical token. In English, Porter defined a simple suffix-stripping algorithm for this purpose, which we describe below. In highly inflected languages like German, stemming is more difficult because of irregular forms and strong conjugation patterns ("gehen" → "ging").

A related challenge arises from **compound words**. Some languages allow words of arbitrary length by combining smaller words into one. German is notorious for this: the law "Rinderkennzeichnungs- und Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz" (cattle marking and beef labeling supervision duties delegation law) was an actual legal term in Mecklenburg-Vorpommern until 2013. Finnish produces similar constructions.

Breaking compounds into their parts can improve the likelihood of matching query terms. Without decomposition, a query like "Rind Kennzeichnung" would never match the compound above. However, decomposition can also destroy meaning. Splitting "Gartenhaus" into "Garten" and "Haus" preserves the original sense reasonably well. Splitting "Wolkenkratzer" (skyscraper) into "Wolke" (cloud) and "Kratzer" (scratcher) loses the idiomatic meaning entirely.

```{admonition} Compound word decomposition
:class: warning
In practice, compound decomposition is a trade-off between recall and precision. Aggressive splitting helps match more queries but risks introducing false matches when the compound's meaning is non-compositional. There is no reliable automated way to distinguish compositional compounds ("Gartenhaus") from idiomatic ones ("Wolkenkratzer").
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

This approach is fast (a few hundred lines of code), requires no dictionary, and works well enough for most English retrieval tasks. However, it makes errors in both directions: it can merge unrelated words ("universal" and "university" both stem to "univers") or fail to merge related ones ("alumnus" and "alumni" remain distinct). [Figures %s](#fig-porter-stemmer-steps-1-2) and [%s](#fig-porter-stemmer-steps-3-5) show selected rules from the algorithm.

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

## Summarization

After tokenization and stemming, the system must represent each document as a data structure that retrieval models can compare against a query. The standard approach is a high-dimensional feature vector where each dimension corresponds to a term in the vocabulary.

Consider a simple collection of three documents about animals. After tokenization and stemming, the vocabulary contains terms like "cat", "dog", "fish", and "feed". How do we capture what each document is about?

The simplest option is the **set-of-words** model: for each term, record only whether it is present (1) or absent (0). A document mentioning "cat" three times looks identical to one mentioning it once. This discards useful information but produces a compact binary representation.

A richer option is the **bag-of-words** model: record how many times each term appears. A document that mentions "cat" five times is represented differently from one that mentions it once. The term order is still ignored, but frequency information is preserved.

Both models ignore term proximity and term order. A document containing "New York" is represented the same way as one containing "York New". Despite this limitation, these representations work remarkably well for retrieval because the presence and frequency of specific terms are strong signals of relevance.

### Formalization

Let $D_{i}$ be a document and $M$ be the size of the vocabulary. The feature representation $\mathbf{d}_{i}\in \mathbb{R}^{M}$ is a vector where component $d_{i,j}$ corresponds to term $t_{j}$. We use $\text{tf}(D_{i},t_{j})$ to denote the number of occurrences of term $t_{j}$ in document $D_{i}$.

For the set-of-words model:

$$d_{i,j}=\begin{cases}1 & \text{tf}(D_{i},t_{j})>0 \\ 0 & \text{tf}(D_{i},t_{j})=0\end{cases}$$

For the bag-of-words model:

$$d_{i,j}=\text{tf}(D_{i},t_{j})$$

```{figure} images/figure_1_16.png
:name: fig-bag-of-words-pipeline
:width: 90%

Text preprocessing pipeline for bag-of-words document representation. Raw tokenized text is processed through stemming and normalization to produce term-frequency vectors.
```

### Vocabulary and Sparsity

In practice, a vocabulary can include millions of terms. However, most documents contain only a few hundred or thousand unique terms. The feature vectors are therefore sparsely populated: the vast majority of components are zero. Efficient storage methods like the inverted file exploit this sparsity by recording only the non-zero entries. During retrieval, the system only needs to consider documents that share at least one term with the query.

Classical retrieval models also treat terms as independent. "Cat" and "cats" are different dimensions unless stemming merges them. A query for "cats" will not match a document containing only "cat" unless both are reduced to the same stem during tokenization. The same applies to spelling variations: "colour" does not match "color" without normalization.

### Stop Words and Term Discrimination

However, we notice many terms that are grammatically necessary but do not contribute significantly to the content description. For example, the article "the" in English is one of the most frequent terms in English texts but does not provide relevant information to describe the content. Since almost all English texts contain this article, a search with "the" would retrieve all documents making it unable to differentiate between relevant and non-relevant ones.

```{figure} images/figure_1_17.png
:name: fig-top50-term-frequencies
:width: 90%

Document frequency ($df$) and collection term frequency ($tf$) for the 50 most frequent terms in a large news corpus (~20,000 documents). The top-50 terms account for roughly one-third of all term occurrences yet appear in over 60% of documents, contributing little to content discrimination.
```

Stop word lists for most languages are readily available, for example on [Kaggle (stop words in 28 languages)](https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages).

Instead of manually maintaining stop word lists, a more pragmatic approach is based on Zipf's law. Let $N$ be the total number of term occurrences (tokens) in the collection and $M$ be the number of distinct terms in the vocabulary. We already used the term frequency $\text{tf}(t)$ to denote the number of occurrences of term $t$. Now, let us order all terms by decreasing term frequencies and assign $\text{rank}(t)$ to term $t$ based on that order. The central theorem of Zipf's law is that the probability $p(r)$ of randomly selecting the term $t$ with $\text{rank}(t)=r$ from the collection is $c/r$ with a constant $c$ that only depends on $M$:

$$p_{r}=\frac{c}{r}=\frac{\text{tf}(t)}{N}, \quad \text{term } t \text{ with } \text{rank}(t)=r$$

The sum of all $p(r)$ equals $1$ and plugging in $p(r)=c/r$ for all terms results in a closed formula to estimate $c$ based on the number of terms $M$:

$$c = \frac{1}{\sum_{r=1}^{M}\frac{1}{r}} \approx \frac{1}{0.5772 + \ln M}$$

For example, in a collection with $M=5{,}000$ different terms, $c=0.11$, while in a collection with $M=100{,}000$, $c=0.08$.

```{figure} images/figure_1_18.png
:name: fig-zipf-law-discrimination
:width: 80%

Zipf's Law and term discriminating power. The most frequent words (above the upper cut-off) hold minimal significance since they appear in nearly every text. The least frequent words (below the lower cut-off) are discriminative but unlikely to appear in queries. The range of meaningful words falls between the two cut-off points.
```

Initially, the idea was to establish cut-off thresholds and exclude words beyond those limits. This would save storage space and enhance search speed. Nowadays, the common practice is to retain all terms, including stop words, but consider the terms' discriminating power to determine their weight during relevance assessment.

Consider the search for "it" which is a stop word. If we were to eliminate this term, we would lose the ability to search for IT books or the book "It" by Stephen King. A query like "the cat" would still search for both terms in documents but would assign significantly higher weight to occurrences of "cat" to determine relevance.

### Discrimination Power and IDF

In their 1975 paper, Salton, Wong, and Yang took a different approach by exploring methods to quantify the discriminatory power of terms. Let's consider a collection with documents $D_{i}$ and the similarities between them given by $0\leq \text{sim}(D_{i}, D_{j})\leq 1$. We examine the collection twice, once with the term $t$ in documents and once with it removed, to analyze the impact of the term's presence on similarities. Removing a valuable term from the collection causes documents to become more similar to each other. This is because the valuable term helped to distinguish documents, resulting in lower similarities between them.

- Let $\text{tf}(D_{i},t_{j})$ represent the term frequency of term $t_{j}$ in document $D_{i}$

- We determine the centroid document $C$ by aggregating all $M$ terms with their average frequency $\text{tf}(C,t_{j})$ across the $N$ documents:

$$\text{tf}(C,t_{j})=\frac{1}{N}\cdot\sum_{i=1}^{N}\text{tf}(D_{i},t_{j}) \quad \text{for } \forall j$$

- Then, we define the density of the collection as the sum of all similarities between documents and their centroid $C$:

$$Q=\sum_{i=1}^{N}\text{sim}(D_{i},C)$$

- Finally, we compute the density $Q_{t}$ for the collection without the term $t$, and define the discrimination power of term $t$ as: $\text{dp}(t)=Q_{t}-Q$

    - $\text{dp}(t)$ is large: if we remove the term $t$ from the collection, similarities to the centroid increase. In other words, the term $t$ differentiates the collection and is hence a significant term.

    - $\text{dp}(t)$ is negative: if the term is present, documents are more similar to the centroid. This can happen, for instance, if a word occurs very frequently in all documents and thus dominates the similarity score.

- Sorting terms by their decreasing $\text{dp}(t)$-value assigns a discrimination rank to each term $t$.

```{figure} images/figure_1_19.png
:name: fig-discrimination-rank-vs-df
:width: 70%

Average discrimination rank of terms as a function of document frequency in the Medlars collection (450 documents). Terms occurring in approximately 13 out of 450 documents exhibit the highest average discrimination rank.
```

Karen Spärck Jones (1972) introduced a statistical interpretation for term discrimination called inverse document frequency (idf) which has evolved into the standard method for term weighting in relevance assessment. The document frequency $\text{df}(t)$ indicates how many documents contain the term $t$ at least once. Let $N$ be the collection's document count. The inverse document frequency $\text{idf}(t)$ is expressed as:

```{admonition} Key Formula: Inverse Document Frequency
:class: important

$$\text{idf}(t) = \log\frac{N+1}{\text{df}(t)+1} = \log(N+1) - \log(\text{df}(t)+1)$$

Terms that appear in fewer documents receive higher IDF weights. This captures their ability to discriminate between relevant and non-relevant documents for a given query.
```

We can utilize $\text{idf}$ to assign weights to components in both query and document feature vectors. As a simplification, let us assume that a term only occurs once in a query. Furthermore, we can estimate the probability that a term $t$ is part of the query to be proportional to $\text{df}(t)/N$ (we need to normalize by the sum over all terms to obtain probability values). Finally, the components of the weighted document vector for $D_{i}$ are given by $\text{idf}(t)\cdot\text{tf}(D_{i},t)$.

Comparing vectors in vector space retrieval relies on the inner vector product. We multiply query and document components and aggregate these values. Consequently, the term's discrimination power approximately equals $\text{idf}(t)^{2}\cdot\text{tf}(D_{i},t)\cdot p(t)$ over all queries and documents. This value predicts a term's contribution to the relevance assessment (here for the inner vector product), or in other words, how useful the term is to describe the content and to distinguish between relevant and non-relevant documents.

```{figure} images/figure_1_20.png
:name: fig-idf-vs-discrimination-power
:width: 70%

Comparison of IDF-weights and discrimination power as a function of document frequency $df$ for a corpus of N = 1,000 documents. Terms around $df = 100 = 0.1 \cdot N$ exhibit the highest discrimination power.
```

```{admonition} Hands-on: Feature Extraction
:class: hint
Build a feature extraction pipeline in Python: tokenize text, apply stemming, compute term frequencies and IDF weights.
[Open notebook -->](https://github.com/mmir-unibasel-hs26/mmir-unibasel-hs26/blob/main/02-ClassicalTextRetrieval/01-extract-features.ipynb)

*Includes pre-run results. You can read through or download and experiment.*
```
