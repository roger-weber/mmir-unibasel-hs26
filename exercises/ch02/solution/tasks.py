"""Ch02 helper - a reusable evaluator for ranked retrieval metrics.

The exercise builds P@k, average precision, MAP, and (graded) nDCG into one class so a
single object can score any ranked run against any information need.
"""
import math


class Evaluator:
    """Scores ranked runs against graded relevance judgments for a set of needs."""

    def __init__(self, need_grades: dict[str, dict[str, int]]):
        """Store the per-need judgments: need id -> {doc_id: grade}."""
        self.need_grades = need_grades

    def _grade(self, doc_id: str, need: str) -> int:
        """Relevance grade of a document for a need; 0 if unjudged."""
        return self.need_grades[need].get(doc_id, 0)

    def precision_at_k(self, run: list[str], need: str, k: int) -> float:
        """Fraction of the top-k results that are relevant (divisor is always k)."""
        if k <= 0:
            return 0.0
        hits = sum(1 for doc_id in run[:k] if self._grade(doc_id, need) > 0)
        return hits / k

    def average_precision(self, run: list[str], need: str) -> float:
        """Mean precision at each relevant rank, divided by the total relevant count."""
        total_relevant = sum(1 for g in self.need_grades[need].values() if g > 0)
        if total_relevant == 0:
            return 0.0
        hits = 0
        summed = 0.0
        for rank, doc_id in enumerate(run, 1):
            if self._grade(doc_id, need) > 0:
                hits += 1
                summed += hits / rank
        return summed / total_relevant

    def dcg(self, run: list[str], need: str, k: int) -> float:
        """Graded DCG over the top k: sum of grade / log2(rank + 1)."""
        total = 0.0
        for rank, doc_id in enumerate(run[:k], 1):
            total += self._grade(doc_id, need) / math.log2(rank + 1)
        return total

    def ndcg(self, run: list[str], need: str, k: int) -> float:
        """DCG at k normalized by the ideal DCG (grades sorted descending)."""
        ideal_grades = sorted(self.need_grades[need].values(), reverse=True)
        idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_grades[:k], 1))
        if idcg == 0:
            return 0.0
        return self.dcg(run, need, k) / idcg

    def map_score(self, runs: dict[str, list[str]]) -> float:
        """Macro-average of average_precision across the given need -> run mapping."""
        if not runs:
            return 0.0
        aps = [self.average_precision(r, need) for need, r in runs.items()]
        return sum(aps) / len(aps)
