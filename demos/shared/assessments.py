"""
Queries and relevance judgments for evaluating retrieval runs.

An `Assessments` object holds the topics (information needs) of a benchmark and
its graded relevance judgments (qrels). Judgments are usually incomplete: only
documents that some system retrieved were ever judged. Two tools address this:

- `merge()` adds judgments from another source (for example an LLM judge)
  without overriding the original ones, and remembers where each label came from.
- The measures below accept a `mode`: "standard" treats unjudged documents as not
  relevant, "condensed" removes them from the ranking before scoring.

Usage:
    from shared.assessments import load_assessments, ndcg_at_k

    a = load_assessments("trec-covid")
    a.topics["1"].question           # "what is the origin of COVID-19"
    a.grade("1", "005b2j4b")         # 2, or None if unjudged
    ndcg_at_k(ranked_ids, a.qrels["1"], k=10)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Topic:
    """One information need: the question users ask, plus context for judges."""
    id: str
    question: str          # the query text a system receives
    keywords: str = ""     # short keyword form of the query (TREC "query" field)
    narrative: str = ""    # what counts as relevant (TREC "narrative" field)


class Assessments:
    """Topics and graded judgments {query_id: {doc_id: grade}} for one benchmark."""

    def __init__(self, name: str, topics: dict[str, Topic],
                 qrels: dict[str, dict[str, int]], sources: dict | None = None):
        self.name = name
        self.topics = topics
        self.qrels = qrels
        # (query_id, doc_id) -> label source; original judgments default to the benchmark
        self.sources = sources if sources is not None else {}

    def query_ids(self) -> list[str]:
        return list(self.topics)

    def grade(self, query_id: str, doc_id: str) -> int | None:
        """Relevance grade of a document, or None if it was never judged."""
        return self.qrels.get(query_id, {}).get(doc_id)

    def is_judged(self, query_id: str, doc_id: str) -> bool:
        return doc_id in self.qrels.get(query_id, {})

    def source(self, query_id: str, doc_id: str) -> str | None:
        if not self.is_judged(query_id, doc_id):
            return None
        return self.sources.get((query_id, doc_id), self.name)

    def unjudged(self, query_id: str, doc_ids) -> list[str]:
        """The documents of a ranking that have no judgment yet."""
        return [d for d in doc_ids if not self.is_judged(query_id, d)]

    def merge(self, extra: dict[str, dict[str, int]], source: str) -> "Assessments":
        """Return a copy completed with `extra` judgments; existing labels win."""
        qrels = {q: dict(labels) for q, labels in self.qrels.items()}
        sources = dict(self.sources)
        for q, labels in extra.items():
            for d, g in labels.items():
                if d not in qrels.setdefault(q, {}):
                    qrels[q][d] = int(g)
                    sources[(q, d)] = source
        return Assessments(f"{self.name}+{source}", self.topics, qrels, sources)

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.qrels.values())
        return f"Assessments('{self.name}', {len(self.topics)} topics, {n:,} judgments)"


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_assessments(name: str) -> Assessments:
    """Load the topics and test judgments of a BEIR dataset (e.g. "trec-covid")."""
    from shared.collections.beir import beir_dataset_dir
    folder = beir_dataset_dir(name)
    topics = {}
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            meta = q.get("metadata", {}) or {}
            topics[q["_id"]] = Topic(q["_id"], q["text"], meta.get("query", ""),
                                     meta.get("narrative", ""))
    qrels: dict[str, dict[str, int]] = {}
    with open(folder / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            q, d, g = line.rstrip("\n").split("\t")
            qrels.setdefault(q, {})[d] = max(0, int(g))  # a few -1 labels mean "not relevant"
    topics = {q: t for q, t in topics.items() if q in qrels}
    return Assessments(name, topics, qrels)


def load_judgments(path: str | Path) -> dict[str, dict[str, int]]:
    """Read extra judgments stored as {query_id: {doc_id: grade}} JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ─── Measures ────────────────────────────────────────────────────────────────

def _condense(ranked: list[str], labels: dict[str, int]) -> list[str]:
    return [d for d in ranked if d in labels]


def ndcg_at_k(ranked: list[str], labels: dict[str, int], k: int = 10,
              mode: str = "standard") -> float:
    """nDCG@k with linear gains (the BEIR / trec_eval convention).

    mode="standard": unjudged documents count as not relevant.
    mode="condensed": unjudged documents are removed before the cut at k.
    """
    if mode == "condensed":
        ranked = _condense(ranked, labels)
    dcg = sum(labels.get(d, 0) / math.log2(r + 2) for r, d in enumerate(ranked[:k]))
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def judged_at_k(ranked: list[str], labels: dict[str, int], k: int = 10) -> float:
    """Fraction of the top k that carries a judgment."""
    top = ranked[:k]
    return sum(d in labels for d in top) / len(top) if top else 0.0


def overlap_at_k(ranked: list[str], reference: list[str], k: int = 10) -> float:
    """|top-k ∩ reference top-k| / k: recall against a reference ranking."""
    return len(set(ranked[:k]) & set(reference[:k])) / k
