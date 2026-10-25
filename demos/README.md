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

### Java kernel (Lucene demos)

Two demos, [ch04-04-lucene.ipynb](ch04-04-lucene.ipynb) and [ch06-05-lucene-vector-search.ipynb](ch06-05-lucene-vector-search.ipynb), are different: they run real Apache Lucene, which is a Java library, so they need a Java kernel instead of Python. Everything else uses the `.venv` Python kernel. Set it up once:

1. Install a Java 21 JDK (for example [Amazon Corretto 21](https://docs.aws.amazon.com/corretto/latest/corretto-21-ug/downloads-list.html)). Lucene 10 requires Java 21 or newer.
2. Download the Ganymede kernel jar (`ganymede-2.1.1.20221231.jar`) from the [Ganymede releases](https://github.com/allen-ball/ganymede/releases).
3. Register the kernel using that Java 21 runtime:

   ```bash
   java -jar ganymede-2.1.1.20221231.jar --install --user
   ```

   Then check it appears: `jupyter kernelspec list` should show `ganymede-2.1.1-java-21`.
4. Open the notebook and pick that kernel. In VS Code: **Select Kernel**, then **Select Another Kernel...**, then **Jupyter Kernel...**, then **Ganymede 2.1.1 (Java 21)**. If it is not in the list, reload the window (Command Palette, *Developer: Reload Window*) and try again, because a freshly installed kernel is only picked up after a refresh.

Lucene itself is downloaded from Maven Central by the notebook's first cell (the `%%pom` magic), so the first run of that cell takes a little longer. A one-time "Java vector incubator module is not readable" message on the index-build cell is harmless: it is a performance note only.

### Docker (Chapter 6 frameworks demo)

[ch06-04-vector-search-frameworks.ipynb](ch06-04-vector-search-frameworks.ipynb) compares FAISS with two real servers, OpenSearch and PostgreSQL with pgvector. Both run as containers, so this notebook needs [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or another Docker engine with `docker compose`). Start the services from the `demos/` folder before running the notebook, and stop them afterwards:

```bash
docker compose -f docker/vector-search.yml up -d
docker compose -f docker/vector-search.yml down
```

The first start downloads the two images, which take about 3.5 GB of disk space. The FAISS parts of the notebook run without Docker.

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
| [ch04-04-lucene.ipynb](ch04-04-lucene.ipynb) | Real Apache Lucene 10 on a Java kernel. Compares analyzers, builds an index of typed fields, runs multi-field, boosted, and filtered queries ranked by BM25, and reads an `explain()` breakdown of every term in the score. Requires the Ganymede Java kernel: see [Java kernel](#java-kernel-lucene-demos) under Setup. |

## Chapter 5 — Semantic Search

| Notebook | What it shows |
|---|---|
| [ch05-01-latent-semantic-indexing.ipynb](ch05-01-latent-semantic-indexing.ipynb) | Latent Semantic Indexing on the book's nine-document Deerwester collection. Builds the term-document matrix, factorizes it with a Singular Value Decomposition, keeps the strongest latent topics, and projects documents and queries into that compact topic space so texts about the same subject match even when they share no words. |
| [ch05-02-word-embeddings.ipynb](ch05-02-word-embeddings.ipynb) | Word embeddings and the distributional hypothesis with pre-trained GloVe (100d): nearest neighbors, a PCA projection of semantic clusters, analogy arithmetic and where it breaks, GloVe versus fastText on out-of-vocabulary words, and the one-vector-per-word polysemy limit. |
| [ch05-03-tokenization.ipynb](ch05-03-tokenization.ipynb) | Sub-word tokenization, the step that turns arbitrary text into the fixed-size integer vocabulary a transformer reads. Compares several tokenizers, their vocabulary sizes, and how frequent words stay single tokens while rare, misspelled, or foreign words decompose into known sub-word pieces. |
| [ch05-04-sentence-embeddings.ipynb](ch05-04-sentence-embeddings.ipynb) | From word vectors to whole-sentence retrieval. The bi-encoder (Sentence-BERT), the cross-encoder, and late-interaction ColBERT side by side: a similarity matrix, vocabulary mismatch dissolving, the quality-versus-speed trade-off between bi- and cross-encoders, and MaxSim scoring stepped through on a small example. |
| [ch05-05-building-semantic-search.ipynb](ch05-05-building-semantic-search.ipynb) | A semantic search system end to end on a 500-movie collection. Builds five pipelines (BM25, dense bi-encoder, hybrid reciprocal-rank fusion, cross-encoder reranking, and Matryoshka-truncated embeddings), loads pre-shipped corpus embeddings, and evaluates them against a hand-built relevance benchmark, showing where the pipelines agree, where they differ, and the recall-versus-dimensionality trade-off. The offline evaluation harness and its artifacts live in [eval/ch05_movies/](eval/ch05_movies/). |

## Chapter 6 — Vector Search

| Notebook | What it shows |
|---|---|
| [ch06-01-distances-and-dimensionality.ipynb](ch06-01-distances-and-dimensionality.ipynb) | Similarity scores and distances side by side: dot product versus cosine, why the query length does not matter, how distance and score relate on the unit sphere, and how to keep the dimensions of a color histogram comparable. A second part reproduces the book's Monte Carlo numbers on the curse of dimensionality and ends with real TREC-COVID embeddings. |
| [ch06-02-approximate-nn.ipynb](ch06-02-approximate-nn.ipynb) | The three levers of approximate search on 400,000 GloVe vectors: fewer dimensions (PCA), fewer bits per vector (scalar quantization, VA-file, product quantization, LSH), and fewer vectors read (inverted file, navigable small-world graph). Each method is built in a few lines of numpy, then FAISS combines them on a recall-versus-throughput curve. |
| [ch06-03-ann-benchmark.ipynb](ch06-03-ann-benchmark.ipynb) | How much an approximate retriever costs the final ranking. Combines Matryoshka truncation, a coarse filter (full scan, IVF, HNSW), and a fine filter (floats, PQ, LSH) on TREC-COVID, reranks each top 50 with a cross-encoder, and compares recall against exact search with nDCG against relevance judgments. The precomputed runs and LLM judgments live in [eval/ch06_ann/](eval/ch06_ann/). |
| [ch06-04-vector-search-frameworks.ipynb](ch06-04-vector-search-frameworks.ipynb) | One realistic movie query sent through FAISS, OpenSearch, and pgvector on 10,000 movies: vector search, pre-, post-, and iterative filtering, and hybrid BM25 plus vector search fused with reciprocal rank fusion, including why the systems rank differently. Requires Docker: see [Docker](#docker-chapter-6-frameworks-demo) under Setup. Data in [eval/ch06_movies/](eval/ch06_movies/). |
| [ch06-05-lucene-vector-search.ipynb](ch06-05-lucene-vector-search.ipynb) | The same movie query in Lucene 10: a vector field next to text and numeric fields, an approximate k-NN query, filters applied inside versus next to the vector query, a BM25 plus vector score read with `explain()`, RRF fusion, and quantized vector formats with rescoring. Requires the Ganymede Java kernel: see [Java kernel](#java-kernel-lucene-demos) under Setup. |

## Chapter 7: Retrieval-Augmented Generation

The Chapter 7 demos call Claude on AWS Bedrock. Without AWS credentials they still run: every model call shows an answer or an agent run recorded earlier (in [data/ch07/](data/ch07/)), marked as such. The first run downloads the reader model of ch07-03 (about 260 MB) in addition to the models of Chapter 5.

| Notebook | What it shows |
|---|---|
| [ch07-01-chunking-strategies.ipynb](ch07-01-chunking-strategies.ipynb) | Every chunking method of the book on *A Study in Scarlet*: fixed-size, sentence, structural, semantic, hierarchical, and query-aware splitting with LangChain, their chunk-size distributions, and a comparison by answer recall at k and the context each strategy costs. |
| [ch07-02-rag-pipeline.ipynb](ch07-02-rag-pipeline.ipynb) | A RAG pipeline over pinned Wikipedia articles about the Nobel Prizes, stage by stage: two-level chunks, BM25 and dense retrieval, reciprocal rank fusion, cross-encoder reranking, and a grounded, citing answer to a question the model cannot answer from memory. Compares pipeline variants by quality and latency and rebuilds the pipeline in Haystack. |
| [ch07-03-result-generation.ipynb](ch07-03-result-generation.ipynb) | The three answer forms: a BERT reader that marks an answer span, a generator that writes a grounded answer, and a multi-hop synthesizer compared with a single retrieval pass of the same size. Ends with structured citations and a check that every quote is really in its passage. |
| [ch07-04-agentic-rag.ipynb](ch07-04-agentic-rag.ipynb) | The reason-act-observe agent loop with tool calling: three declared tools (knowledge base, people directory, web search), the agent's trajectory on a two-part question including a fallback, a step budget, and guardrails before the loop, before every tool call, and before the answer. |
| [ch07-05-rag-evaluation.ipynb](ch07-05-rag-evaluation.ipynb) | A pipeline, an agent, and the model alone on a 24-question benchmark: context recall and precision, p95 latency and tokens, correctness and faithfulness with an LLM judge checked against a rule, the diagnostic matrix, the paired sign test, and when RAG is not needed at all. |

More demos are added as the course progresses — see the [schedule](../README.md#schedule) for what comes next. The hands-on counterparts to these demos are in [exercises/](../exercises/).
