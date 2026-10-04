# Exercise 05: Semantic Search

> **Chapter:** [Ch05 - Semantic Search](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-5/) · **Estimated time:** ~50 minutes · **Optional:** no submission, no grading

Semantic search is more than ranking by meaning. A real search box also lets users narrow results by hard constraints (a release year, a minimum rating, a genre) and shows them which refinements are worth making. In this exercise you turn the retrieval building blocks from the chapter into a small movie search engine over 500 films, then extend it with two features every production system needs but the chapter only gestures at: filtered search and faceted navigation. You assemble the four pipelines, add a predicate filter, build genre and rating facets, and finally reason about where faceting belongs in the pipeline.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook. Both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"05 - Semantic Search"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: you build a working search engine, give it a filter and facets, and reason about a design choice the chapter leaves open.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

The exercise works over `movies-small`, the first 500 films of the Kaggle movie dataset. The two sentence-embedding sets are **pre-computed and shipped with the exercise**, so you never encode the 500 documents yourself; search is a plain linear scan over a list of movie objects. Only your live query is encoded. The default `MODEL = "mini"` setting downloads all-MiniLM-L6-v2 (about 90 MB) and the cross-encoder ms-marco-MiniLM-L-6-v2 (about 90 MB) on first run and caches them; switching to `qwen` downloads the stronger Qwen3-Embedding-0.6B (about 1.2 GB) instead and is optional.

| Task | Type | What you do |
|---|---|---|
| C1 | 💻 Code | Implement `run_pipeline`: compose the four provided blocks (BM25, dense bi-encoder, Reciprocal Rank Fusion, cross-encoder rerank) into pipelines A (BM25), B (BM25 then rerank), C (dense), and D (hybrid fusion then rerank). A verify cell checks each on an exact-title and a paraphrase query. |
| C2 | 💻 Code | Implement `search` with a `predicate` pre-filter: a `lambda` over a movie object (year, rating, genre) restricts the candidate set *before* scoring, so the ranking never contains a movie the filter excludes. |
| C3 | 💻 Code | Implement `genre_facets` (top-n genre counts over the result set) and `rating_facets` (counts in the bands `<6`, `6-8`, `>8`), the clickable filters a search UI would render. |
| T1 | ✏ Reasoning | Facet counts are computed over the top-k, not the whole collection: explain what that makes them, compare filtering the shown results against turning the clicked facet into a predicate and re-running the search, and say when each is the right choice. |

Your code lives in [tasks.py](tasks.py), a small `MovieSearch` class whose four retrieval blocks are already written; you fill in `run_pipeline`, `search`, `genre_facets`, and `rating_facets`. Autoreload means a saved edit takes effect on the next cell run, with no kernel restart.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/solution.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
