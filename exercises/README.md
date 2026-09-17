# Exercises — Multimedia Retrieval (HS26)

One exercise per chapter, each about **55 minutes** of work. They are **optional**: there is no submission and no grading. What they do give you is the difference between recognising a formula and being able to implement it — good preparation for the exam.

Each exercise folder contains a notebook with two kinds of tasks:

- **✏ Reasoning tasks** — you write an explanation into a markdown cell. The solution notebook fills the same cell with a model answer, so you can compare your reasoning against it.
- **💻 Coding tasks** — you implement a function or class, either in the notebook or in the folder's `tasks.py`. A verify cell checks your implementation with assertions that fail with a specific message. Writing the code yourself and directing an AI to write it are both fine; the point is to understand *why* a given solution works.

Solution notebooks are published one to two weeks after the corresponding exercise session, in a `solution/` subfolder of the exercise. Until then the folder is not in the repository.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook — both are safe to repeat. The [main README](../README.md) has the full explanation.

## Quiz app

Every chapter has **20 multiple-choice questions** in the quiz app. Work through them before the exercise tasks: the quiz covers definitions and basic concepts, while the tasks assume you already have them and go further.

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

On the start screen, pick the topic that matches the chapter (for example *"01 - Classical Text Retrieval"*).

## Exercises

| Exercise | Chapter | Main topics |
|---|---|---|
| [ch01/](ch01/) | [Classical Text Retrieval](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-1/) | Reading a BM25 ranking that looks wrong; building a text-extraction pipeline that makes queries find their documents; implementing a full `BM25Scorer` over ~500 movies; why scoring every document does not scale. |
| [ch02/](ch02/) | [Performance Evaluation](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-2/) | Why average precision and nDCG can disagree about the same two systems; implementing a reusable `Evaluator` (P@k, AP, MAP, graded nDCG); why precision cannot be promised from recall and specificity alone; how a better ranker can score worse on an old benchmark. |

Exercises for the later chapters follow the [course schedule](../README.md#schedule). The demos that go with each chapter are in [demos/](../demos/).
