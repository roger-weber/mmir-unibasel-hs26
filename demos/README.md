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

More demos are added as the course progresses — see the [schedule](../README.md#schedule) for what comes next. The hands-on counterparts to these demos are in [exercises/](../exercises/).
