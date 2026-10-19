# Exercise 06: Vector Search

> **Chapter:** Ch06 - Vector Search · **Estimated time:** ~40 minutes · **Optional:** no submission, no grading

A real vector query rarely asks only for the nearest neighbors. It asks for the nearest neighbors that also satisfy a condition: movies after 2005, documents of one customer, products in stock. The chapter describes three ways to combine a filter with an approximate index and argues about their costs. In this exercise you measure them on an inverted file over 100,000 word vectors, run them against filters of very different selectivity, and build a search that picks the right strategy for each query. Along the way you meet a filter that breaks the chapter's cost estimate, and you work out why.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook. Both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"06 - Vector Search"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: you measure what the chapter only estimates, and you design a filtering rule that holds up on filters the estimate gets wrong.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

The exercise uses the GloVe-100 word vectors of the chapter demos (about 130 MB, downloaded on first run and cached under `~/gensim-data/`; already there if you ran demo ch05-02 or ch06-02). Every word gets two kinds of metadata to filter on: a random tenant id, as in a vector service shared by many customers, and a content attribute, whether the word contains a digit. The inverted file is plain numpy and built for you; no vector-search library is needed.

| Task | Type | What you do |
|---|---|---|
| C1 | 💻 Code | Implement post-filtering (over all scanned candidates, or over the top `fetch` results of a black-box index), filtering during the search, and pre-filtering, then compare their recall, result count, and cost on five filters. |
| T1 | ✏ Reasoning | Two filters admit the same 8% of the index, yet filtering during the search finds far fewer true neighbors for one of them. Explain why, using how the inverted file assigns vectors to lists. |
| C2 | 💻 Code | Implement `search_auto`, which decides per query how to filter and must meet a recall and cost budget on every filter. |

Your code lives in [tasks.py](tasks.py), a small `FilteredIVF` class whose index and helpers are already written; you fill in `search_post`, `search_during`, `search_pre`, and `search_auto`. Autoreload means a saved edit takes effect on the next cell run, with no kernel restart.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/solution.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
