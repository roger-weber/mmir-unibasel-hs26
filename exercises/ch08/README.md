# Exercise 08: Web Search

> **Chapter:** Ch08 - Web Search · **Estimated time:** ~50 minutes · **Optional:** no submission, no grading

PageRank looks simple: a page is important if important pages link to it. But what does it do when people change the link graph, by accident or on purpose? In this exercise you take a real link graph of 4,604 Wikipedia articles and change it in seven ways. A spammer builds a link farm, a race report appears, a brand changes its name, an editor hoards links, and a template adds one link to every page. You first guess by gut feeling what each change does. Then you implement PageRank, run every change, and find out where your intuition was wrong. A last part asks whether the result of the link farm carries over to a web of billions of pages.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook. Both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"08 - Web Search"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: you watch PageRank react to changes of a real link graph and confront your intuition with the result.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

The exercise uses the Wikispeedia link graph from Stanford's SNAP collection: the 4,604 articles of the 2007 Wikipedia for Schools selection and their 119,772 links. The graph ships with the repository in `shared/data/` (about 0.4 MB), so nothing is downloaded. The last part generates synthetic webs of up to 200,000 pages, which takes up to a minute.

| Task | Type | What you do |
|---|---|---|
| T1 | ✏ Reasoning | Guess by gut feeling, about a minute each, what seven changes to the link graph do to PageRank. |
| C1 | 💻 Code | Implement PageRank by power iteration, including dangling pages, and check it against the chapter's seven-page web. |
| T2 | ✏ Reasoning | Run the seven changes, pick the guesses that were furthest off, and speculate why within a 15-minute time box. |
| T3 | ✏ Reasoning | Predict and explain how the same link farm fares as the web grows, and how that fits the chapter. |

C1 lives in the notebook. The cells for the seven changes and the synthetic webs are provided and run your `pagerank`.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/solution.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
