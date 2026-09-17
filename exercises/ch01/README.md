# Exercise 01 — Classical Text Retrieval

> **Chapter:** [Ch01 — Classical Text Retrieval](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-1/) · **Estimated time:** ~55 minutes · **Optional:** no submission, no grading

Classical text retrieval turns raw text into terms, then ranks documents by how well their terms match a query. This exercise walks that whole path: first you explain a BM25 ranking that looks wrong, then you build both halves of a real search engine over a collection of movies — the extraction pipeline that decides what counts as a term, and the BM25 scorer that ranks the results.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections — including the movie dataset this exercise uses.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook — both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"01 - Classical Text Retrieval"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: explaining a ranking that looks wrong, and building a search engine yourself.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

| Task | Type | What you do |
|---|---|---|
| T1 | ✏ Reasoning | The book's BM25 ranking for the query `cat dog forest` puts a three-word document first, ahead of documents with more matches. Explain what in the BM25 formula produces that. |
| C1 | 💻 Code | Compose the text-processing primitives from `tasks.py` into an `extract_terms` pipeline, so that a set of title searches returns the movies a user obviously means. Shows how much the pipeline alone decides. |
| C2 | 💻 Code | Implement `BM25Scorer` in `tasks.py`: index the ~500-movie collection once, then answer ranked queries end to end, using the positive (Lucene) IDF. |
| T2 | ✏ Reasoning | Your `search` scores every document for every query. Explain why that breaks at a million movies, and describe what you would precompute instead — reasoning from first principles, before the index chapter. |

[tasks.py](tasks.py) holds the helper functions and the stubs you fill in. It is imported with autoreload enabled, so after saving it you only re-run the verify cell — no kernel restart needed.

The collection is the same `movies` dataset the demos use (about 500 movies with title, plot overview, tagline, top cast, and genres). It downloads on first use and is cached afterwards.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/exercise.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
