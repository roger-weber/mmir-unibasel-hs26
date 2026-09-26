# Exercise 03: Advanced Text Processing

> **Chapter:** [Ch03 - Advanced Text Processing](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-3/) · **Estimated time:** ~65 minutes · **Optional:** no submission, no grading

Before retrieval or routing can happen, raw text becomes features, and every step of that pipeline is a choice. This exercise puts you in a realistic position: you are standing up a new customer-support bot with only a handful of labelled utterances per intent, and you must push a classifier's accuracy as high as you can. The model is a fixed Naive Bayes classifier from the chapter. The interesting part is the text processing you feed it: how you tokenize, what you normalize, what you throw away. You discover which choices earn their keep when data is scarce, and which do not. Then you measure, rather than guess, whether an LLM is worth reaching for.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook. Both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"03 - Advanced Text Processing"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: they ask you to diagnose a broken router, build a working classifier, and reason about your own results.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

The coding task uses the **Bitext customer-support dataset**: about 27,000 real user utterances, each labelled with one of 27 fine intents such as `track_order`, `get_refund`, or `recover_password`. The realistic constraint is scarce labels, so the loader gives you a few-shot training set of just 10 utterances per intent, plus a large validation set to tune on and a held-out test set for the final score. The split is fixed by a hash of each utterance, so everyone trains and tests on identical data.

| Task | Type | What you do |
|---|---|---|
| T1 | ✏ Reasoning | An intent router trained on skewed query logs sends almost every query to the majority backend, even a clear people-search query. Trace the decision rule and explain why the empirical prior overrides the features, when a longer query would behave differently, and a fix with its cost. |
| C1 | 💻 Code | Implement `IntentClassifier` in `tasks.py`: multinomial Naive Bayes over bag-of-words features, with the feature pipeline under your control. `extract_features` lowercases and tokenizes, then optionally removes stop-words, stems, or lemmatizes; `train` counts per class with add-one smoothing; `predict` scores in log space under a uniform or observed prior. Tune the pipeline to lift accuracy on the scarce training set. |
| T2 | ✏ Reasoning | With only 10 utterances per intent, stop-word removal and stemming lift accuracy well above the plain baseline, yet switching the prior from uniform to observed changes nothing. Explain both effects, and predict how the improvement changes as you collect more labelled data. |
| T3 | ✏ Reasoning | An LLM classifies zero-shot and scores higher out of the box, but each call costs latency and money. Decide whether to deploy it for this support bot, how prompt caching changes the case, and whether the classifier and the LLM could work together. |

[tasks.py](tasks.py) holds the `IntentClassifier` stubs. It is imported with autoreload enabled, so after saving it you only re-run the verify cell. No kernel restart needed.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/solution.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
