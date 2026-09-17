# Exercise 02 — Performance Evaluation

> **Chapter:** [Ch02 — Performance Evaluation](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-2/) · **Estimated time:** ~55 minutes · **Optional:** no submission, no grading

A retrieval metric reduces a ranking to a single number so we can compare systems — and that creates two problems. Different metrics can rank the same two systems in opposite orders, and a number measured on a test set need not hold once the system meets real data. This exercise works through both: you explain a metric disagreement, implement the chapter's metrics as a reusable evaluator, and then look at cases where a correctly computed score still misleads.

## Setup

The notebooks run on Python 3.12 in an environment managed by [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Then open [setup.ipynb](../../setup.ipynb) in the repository root, select `.venv` as the kernel, and run all cells. It fetches what `uv sync` cannot: the NLTK corpora, the spaCy model, and the lecture PDFs and datasets behind the collections.

Later chapters add new dependencies, so this is not a one-time step. If a notebook fails with an unknown module or a missing corpus, run `git pull && uv sync` and re-run the setup notebook — both are safe to repeat. The [main README](../../README.md) has the full explanation.

## Quiz first

Work through the **20 questions** for this chapter before the tasks below. On the start screen, pick the topic **"02 - Performance Evaluation"**:

**→ [Quiz app](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)**

The quiz covers definitions and basic concepts. The tasks go further: explaining a metric disagreement, building the metrics yourself, and reasoning about when a correct score lies.

## The exercise

**→ [exercise.ipynb](exercise.ipynb)**

All tasks use the chapter's running example: a 50-book library, one information need, fifteen relevant books graded 3 (core), 2 (specialization) or 1 (peripheral), and two systems answering it — **System A** returns 25 books and favours coverage, **System B** returns 8 and favours a clean result list.

| Task | Type | What you do |
|---|---|---|
| T1 | ✏ Reasoning | Average precision and nDCG disagree about which of the two library systems is better. Explain what each metric rewards, and which property of the two runs produces the disagreement. |
| C1 | 💻 Code | Implement `Evaluator` in `tasks.py`: P@k, average precision, MAP, graded DCG and nDCG, with the chapter's exact formulas — binary AP with divisor \|Rel\|, the log₂(i+1) discount, nDCG normalized by the ideal DCG. One object scores any run against any need. |
| C2 | 💻 Code | Implement `precision_from_rates`. Recall and specificity are properties of the classifier; precision is not — it also depends on how rare the positives are. The explore cell then shows what that does to a classifier that looked good on a balanced test set. |
| T2 | ✏ Reasoning | A new neural ranker scores *lower* MAP than the old keyword baseline on an established TREC collection, yet its top-10 results are visibly better. Explain how, and which property of how such benchmarks were judged penalizes the newer system. |

[tasks.py](tasks.py) holds the `Evaluator` stubs. It is imported with autoreload enabled, so after saving it you only re-run the verify cell — no kernel restart needed.

## Solution

Not published yet. The solution notebook appears here one to two weeks after the exercise session, as `solution/exercise.ipynb`. It fills in every code stub and answers every reasoning task with a model answer, so you can compare it against your own.
