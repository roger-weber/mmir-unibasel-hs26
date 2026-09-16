"""Ch02 helper - a reusable evaluator for ranked retrieval metrics.

The exercise builds P@k, average precision, MAP, and (graded) nDCG into one class so a
single object can score any ranked run against any information need.
"""
import math


class Evaluator:
    """Scores ranked runs against graded relevance judgments for a set of needs."""

    def __init__(self, need_grades: dict[str, dict[str, int]]):
        """Store the per-need judgments: need id -> {doc_id: grade}."""
        # YOUR CODE HERE
        raise NotImplementedError

    def _grade(self, doc_id: str, need: str) -> int:
        """Relevance grade of a document for a need; 0 if unjudged."""
        # YOUR CODE HERE
        raise NotImplementedError

    def precision_at_k(self, run: list[str], need: str, k: int) -> float:
        """Fraction of the top-k results that are relevant (divisor is always k)."""
        # YOUR CODE HERE
        raise NotImplementedError

    def average_precision(self, run: list[str], need: str) -> float:
        """Mean precision at each relevant rank, divided by the total relevant count."""
        # YOUR CODE HERE
        raise NotImplementedError

    def dcg(self, run: list[str], need: str, k: int) -> float:
        """Graded DCG over the top k: sum of grade / log2(rank + 1)."""
        # YOUR CODE HERE
        raise NotImplementedError

    def ndcg(self, run: list[str], need: str, k: int) -> float:
        """DCG at k normalized by the ideal DCG (grades sorted descending)."""
        # YOUR CODE HERE
        raise NotImplementedError

    def map_score(self, runs: dict[str, list[str]]) -> float:
        """Macro-average of average_precision across the given need -> run mapping."""
        # YOUR CODE HERE
        raise NotImplementedError
