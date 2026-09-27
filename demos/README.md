# Demos — Multimedia Retrieval (HS26)

Interactive notebooks that accompany the [course book](https://roger-weber.github.io/mmir-unibasel-hs26/book/). Each demo takes one technique from a chapter and makes it observable: you run the computation on a real collection, change the parameters, and see how the results move. Nothing here is graded — the demos are meant to be read, run, and modified.

Every notebook runs top to bottom without further preparation, and most end with a *Try It Yourself* section that suggests variations. The shared helper code (collections, text processing, plotting) lives in [shared/](shared/); downloaded PDFs and datasets are cached under `data/.cache/` and are not part of the repository.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook — both are safe to repeat. The [main README](../README.md) has the full explanation.

### Java kernel (Chapter 4 Lucene demo)

One demo, [ch04-04-lucene.ipynb](ch04-04-lucene.ipynb), is different: it runs real Apache Lucene, which is a Java library, so it needs a Java kernel instead of Python. This is the only notebook that needs it; everything else uses the `.venv` Python kernel. Set it up once:

1. Install a Java 21 JDK (for example [Amazon Corretto 21](https://docs.aws.amazon.com/corretto/latest/corretto-21-ug/downloads-list.html)). Lucene 10 requires Java 21 or newer.
2. Download the Ganymede kernel jar (`ganymede-2.1.1.20221231.jar`) from the [Ganymede releases](https://github.com/allen-ball/ganymede/releases).
3. Register the kernel using that Java 21 runtime:

   ```bash
   java -jar ganymede-2.1.1.20221231.jar --install --user
   ```

   Then check it appears: `jupyter kernelspec list` should show `ganymede-2.1.1-java-21`.
4. Open the notebook and pick that kernel. In VS Code: **Select Kernel**, then **Select Another Kernel...**, then **Jupyter Kernel...**, then **Ganymede 2.1.1 (Java 21)**. If it is not in the list, reload the window (Command Palette, *Developer: Reload Window*) and try again, because a freshly installed kernel is only picked up after a refresh.

Lucene itself is downloaded from Maven Central by the notebook's first cell (the `%%pom` magic), so the first run of that cell takes a little longer. A one-time "Java vector incubator module is not readable" message on the index-build cell is harmless: it is a performance note only.

## Chapter 1 — Classical Text Retrieval

| Notebook | What it shows |
|---|---|
| [ch01-00-explore-collection.ipynb](ch01-00-explore-collection.ipynb) | The starting point of every retrieval system: the collection. Load any of the registered collections, inspect the document structure (`id`, `source`, `text`, metadata), and look at basic text statistics — vocabulary size, term frequencies, and Zipf's law on real text. |
| [ch01-01-feature-extraction.ipynb](ch01-01-feature-extraction.ipynb) | The pipeline from raw text to vectors, one stage at a time: tokenization, stop word removal, stemming, vocabulary and document frequency, IDF. Ends with set-of-words, bag-of-words, and TF-IDF representations of the same collection, so you can see what each transformation keeps and what it throws away. |
| [ch01-02-retrieval-models.ipynb](ch01-02-retrieval-models.ipynb) | Four models on the same query and collection: standard Boolean, extended Boolean (P-norm), vector space with TF-IDF and cosine, and BM25. Steps through the formulas on concrete documents, shows each model's characteristic weakness (no ranking, repetition bias, length bias), and explores BM25's `k1` and `b`. |

## Chapter 2 — Performance Evaluation

| Notebook | What it shows |
|---|---|
| [ch02-01-retrieval-metrics.ipynb](ch02-01-retrieval-metrics.ipynb) | How retrieval quality becomes a number. Precision and recall on unranked sets, the F-measure trade-off, then the ranked metrics — P@k, reciprocal rank, average precision — plus interpolated precision-recall curves, MAP across queries, and nDCG with graded relevance. Two contrasting systems on the library collection make the disagreements between metrics visible. |
| [ch02-02-classification-and-roc.ipynb](ch02-02-classification-and-roc.ipynb) | Evaluation from the classifier side, using a spam filter. Confusion matrix and derived metrics, the accuracy paradox on imbalanced classes, a threshold sweep that turns scores into decisions, and the step-by-step construction of an ROC curve and its AUC. |

## Chapter 3 — Advanced Text Processing

| Notebook | What it shows |
|---|---|
| [ch03-01-tokenization.ipynb](ch03-01-tokenization.ipynb) | The first pipeline stage under a microscope. The naive regex breaking on a real sentence (possessives, currencies, abbreviations), the same sentence through the nltk and spaCy tokenizers, the retrieval cleanup filters, and the case, Unicode, and accent normalizations. Also handles text where word boundaries are not spaces (sub-word trigrams), detects language with a handful of rules, and measures how one normalization step changes a BM25 ranking. |
| [ch03-02-stemming.ipynb](ch03-02-stemming.ipynb) | Collapsing word forms so a query for "car repair" also matches "cars" and "repaired". The Porter measure and its five steps, Porter vs Lancaster vs Snowball aggressiveness, rule-based stemming vs dictionary lemmatization (went to go, better to good), the Snowball and spaCy divergence on German and French, and stemming closing the recall gap on a concrete match. |
| [ch03-03-phrases-and-compounds.ipynb](ch03-03-phrases-and-compounds.ipynb) | Multi-word units that a bag of words loses. Bi-grams from a Project Gutenberg book, why raw frequency fails, ranking by pointwise mutual information and by the likelihood ratio, what each measure favours (rare-exclusive vs well-attested pairs), and splitting German compounds with frequency-scored candidate parts. |
| [ch03-04-query-understanding.ipynb](ch03-04-query-understanding.ipynb) | Pulling structure out of a query. WordNet synonyms and the polysemy noise they bring, hypernym and hyponym expansion, POS tagging to disambiguate homonyms and filter stop words, named-entity recognition mapped to routing decisions, noun-phrase chunking and dependency parsing, edit distance and Soundex spell correction from scratch, and query expansion vs query rewriting. |
| [ch03-05-intent-routing.ipynb](ch03-05-intent-routing.ipynb) | Turning that structure into a routing decision. A Naive Bayes classifier built from scratch in log-space with Laplace smoothing, used first for language detection over character n-grams where the rules gave up, then to route queries to backends using POS, NER, and question-form features, ending with one query traced end-to-end through the whole pipeline. |

## Chapter 4 — Index for Text Retrieval

| Notebook | What it shows |
|---|---|
| [ch04-01-inverted-index.ipynb](ch04-01-inverted-index.ipynb) | The core search data structure built from scratch: tokenize a collection into a vocabulary, postings lists, and a document table, then evaluate Boolean AND, OR, and NOT with a two-pointer streaming merge over sorted postings. Shows why inverting the index turns a full scan into a lookup. |
| [ch04-02-ranked-retrieval.ipynb](ch04-02-ranked-retrieval.ipynb) | Two ways to evaluate a ranked query over the same index: Term-at-a-Time and Document-at-a-Time. Scores documents with BM25 straight from the postings, decomposes a score into per-term contributions, and explores how `k1` and `b` reshape the ranking; both strategies return identical results. |
| [ch04-03-database-search.ipynb](ch04-03-database-search.ipynb) | Full-text search without a dedicated engine. Builds an inverted index as a plain SQL table, answers Boolean and ranked queries in SQL, then switches to SQLite FTS5 (a real inverted index with `bm25()` ranking) and reads the equivalent PostgreSQL `tsvector`/`tsquery`/GIN/`ts_rank` syntax. |
| [ch04-04-lucene.ipynb](ch04-04-lucene.ipynb) | Real Apache Lucene 10 on a Java kernel. Compares analyzers, builds an index of typed fields, runs multi-field, boosted, and filtered queries ranked by BM25, and reads an `explain()` breakdown of every term in the score. Requires the Ganymede Java kernel: see [Java kernel](#java-kernel-chapter-4-lucene-demo) under Setup. |

More demos are added as the course progresses — see the [schedule](../README.md#schedule) for what comes next. The hands-on counterparts to these demos are in [exercises/](../exercises/).
