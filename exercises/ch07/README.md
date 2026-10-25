# Exercise 07: Retrieval-Augmented Generation

> **Chapter:** Ch07 - Retrieval-Augmented Generation · **Estimated time:** ~55 minutes · **Optional:** no submission, no grading

A generator can only answer from what retrieval puts in front of it. If a fact never reaches the context, no model and no prompt can recover it. In this exercise you look at the retrieval side of a RAG system and ask what one retrieval pass can reach. You first measure how much a larger context budget helps, and find out why some facts stay out of reach. You then build a retriever with a second hop, the hard-wired loop the chapter calls a synthesizer, and test whether it is really better. No language model is involved, so everything runs offline and gives the same numbers on every machine.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook. Both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"07 - Retrieval Augmented Generation"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: you measure which facts a retrieval pass delivers, build the multi-hop retrieval of the synthesizer yourself, and test your improvement the way the chapter compares two configurations.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

The exercise uses the knowledge base of the chapter demos: 38 Wikipedia articles around the 2025 Nobel Prizes, pinned to fixed revisions and downloaded on first run (about 30 seconds). Retrieval is the chapter's hybrid chain: BM25 over parent passages, a bi-encoder over child chunks, reciprocal rank fusion, and cross-encoder reranking. The two models are about 180 MB, already cached if you ran the chapter demos. Building the index takes about a minute on a laptop CPU. A benchmark of 24 questions lists the facts each answer needs, half of them multi-hop.

| Task | Type | What you do |
|---|---|---|
| C1 | 💻 Code | Fill a context budget with ranked passages, then measure how context recall grows with the budget for single lookups and multi-hop questions. |
| T1 | ✏ Reasoning | Explain why sixteen times the context leaves multi-hop facts missing, using where the answer passages rank in each retrieval engine. |
| C2 | 💻 Code | Build a two-hop retriever that follows person names from the first pass and must meet a recall target within the same budget. |
| T2 | ✏ Reasoning | Your retriever is clearly better on average, yet the paired sign test is not significant. Explain why, and what a convincing benchmark would need. |
| T3 | ✏ Reasoning | Find where the hard-wired hop fails, and what the chapter's agent loop would do differently at what cost. |

C1 lives in the notebook. C2 lives in [tasks.py](tasks.py), where you implement `two_hop`; the knowledge base and its search are provided. Autoreload means a saved edit takes effect on the next cell run, with no kernel restart.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/solution.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
