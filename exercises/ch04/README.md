# Exercise 04: Index for Text Retrieval

> **Chapter:** [Ch04 - Index for Text Retrieval](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-4/) · **Estimated time:** ~60 minutes · **Optional:** no submission, no grading

An inverted index is what lets a search engine ignore almost the whole collection and still find the right documents. This exercise follows that idea from cost to production. First you reason about what a query actually reads, and why adding terms is cheap on a small catalog but expensive on the web. Then you build a real search engine on Apache Lucene, the library behind Elasticsearch, Solr, and OpenSearch: you index a movie collection and give it keyword search, a metadata filter on the release year, and fuzzy matching. Finally you reason about what changes once the index no longer fits on one machine.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook. Both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"04 - Index for Text Retrieval"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: they ask you to reason about query cost, build a working search engine yourself, and think about what breaks at scale.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

The coding task uses **Apache Lucene 10**, the Java library behind Elasticsearch, Solr, and OpenSearch, over the full movie collection of about 45,000 titles. The notebook stays in Python: it prepares the data, then a small harness compiles and runs your Java, so you never leave the notebook. This task needs a **JDK, version 21 or newer** (Lucene 10 requires it); the Lucene libraries download automatically on first run.

| Task | Type | What you do |
|---|---|---|
| T1 | ✏ Reasoning | Using the read-cost model `entries ≈ N·K·L/M`, explain why turning on query expansion is nearly free on a 30-million-title catalog but pushes 95th-percentile latency into seconds on a 10-billion-page web index, whether dropping stop-words rescues it, and what expansion costs in result quality. |
| C1 | 💻 Code | Implement two methods in a Lucene `Search.java`: `buildDocument` maps a movie to typed fields (exact id, tokenized title/text, numeric year); `buildQuery` parses the query against the text field, adds a `year` range filter, and switches on fuzzy matching. A verify cell checks ranking and the `k` cap, that the year filter only returns recent films, that a misspelling matches only with fuzzy on, and the empty-query case. |
| T2 | ✏ Reasoning | Explain why the same document can receive a different BM25 score on different shards (the `idf` uses each shard's local `N` and `df`), why that is acceptable for ranked search and when it is not, and what sharding does and does not buy for a single query's latency. |

The Java code lives in [lucene/src/main/java/mmir/Search.java](lucene/src/main/java/mmir/Search.java). Open the [lucene/](lucene/) folder in an IDE for autocompletion; you only edit that one file. After saving it you re-run the cell, and the harness recompiles automatically, so no kernel restart is needed.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/solution.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
