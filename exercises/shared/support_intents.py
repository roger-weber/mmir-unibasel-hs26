"""Bitext customer-support intent dataset loader (Ch03 exercise).

Downloads the public Kaggle dataset with kagglehub (anonymous, cached on first use),
strips the templated ``{{placeholders}}`` so the utterances read like real user text,
and produces a deterministic few-shot ``train`` / ``val`` / ``test`` split. The split
is fixed by a hash of each utterance, so every student trains and tests on exactly the
same data regardless of pandas version, OS, or row order.

    from shared.support_intents import load_support_intents
    data = load_support_intents(shots=10)      # 10 labelled examples per intent
    data.train        # list[(utterance, label)] - the few-shot training set
    data.val          # list[(utterance, label)] - tune on this
    data.test         # list[(utterance, label)] - final score only
    data.labels       # sorted list of class labels

The dataset: ~27,000 real customer-support utterances, each hand-labelled with one of
27 fine-grained intents (grouped into 11 coarse categories). See
https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset
"""
import csv
import glob
import hashlib
import os
import re

_KAGGLE_SLUG = "bitext/bitext-gen-ai-chatbot-customer-support-dataset"
_PLACEHOLDER = re.compile(r"\{\{.*?\}\}")


def _clean(text: str) -> str:
    """Remove templated {{...}} placeholders; collapse whitespace."""
    return " ".join(_PLACEHOLDER.sub(" ", text).split())


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _split_of(text: str) -> str:
    """Deterministic 70 / 15 / 15 split keyed on the utterance itself."""
    h = int(_hash(text), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def _load_rows() -> list[tuple[str, str, str]]:
    """Download (cached) and read the CSV as (utterance, intent, category) rows."""
    import kagglehub

    path = kagglehub.dataset_download(_KAGGLE_SLUG)
    csv_path = glob.glob(os.path.join(path, "*.csv"))[0]
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            utterance = _clean(r["instruction"])
            if utterance:
                rows.append((utterance, r["intent"], r["category"]))
    return rows


class Dataset:
    """A fixed few-shot split. Each item is a ``(utterance, label)`` tuple."""

    def __init__(self, train, val, test, label_kind):
        self.train = train
        self.val = val
        self.test = test
        self.label_kind = label_kind

    @property
    def labels(self) -> list[str]:
        return sorted({label for _, label in self.train})

    def __repr__(self):
        return (f"Dataset(label={self.label_kind!r}, train={len(self.train)}, "
                f"val={len(self.val)}, test={len(self.test)}, classes={len(self.labels)})")


def load_support_intents(shots: int = 10, label: str = "intent") -> Dataset:
    """Load the support dataset as a deterministic few-shot split.

    Args:
        shots: number of labelled examples per class in the training set. A small
            value simulates a realistic cold start, where feature engineering matters.
        label: ``"intent"`` (27 balanced classes) or ``"category"`` (11 coarse,
            imbalanced classes).

    Returns:
        A :class:`Dataset`. ``val`` and ``test`` hold every utterance in their split
        (they are not sub-sampled); only ``train`` is reduced to ``shots`` per class.
    """
    if label not in ("intent", "category"):
        raise ValueError("label must be 'intent' or 'category'")
    field = 1 if label == "intent" else 2

    buckets = {"train": [], "val": [], "test": []}
    for row in _load_rows():
        utterance = row[0]
        buckets[_split_of(utterance)].append((utterance, row[field]))

    # Few-shot training set: the first `shots` utterances per class, ordered by hash
    # so the choice is deterministic and independent of file order.
    per_class: dict[str, list] = {}
    for item in sorted(buckets["train"], key=lambda it: _hash(it[0])):
        bag = per_class.setdefault(item[1], [])
        if len(bag) < shots:
            bag.append(item)
    train = [item for bag in per_class.values() for item in bag]

    return Dataset(train, buckets["val"], buckets["test"], label)
