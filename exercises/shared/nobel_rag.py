"""Nobel knowledge base and retrieval chain for the Chapter 7 exercise.

A self-contained copy of the RAG retrieval chain of the book (Chapter 7, The RAG
Pipeline), so that the exercise does not depend on the chapter demos:

    index:  sections -> parents (<= 2000 chars) -> children (~400 chars)
            BM25 over parents, dense embeddings (all-MiniLM-L6-v2) over children
    query:  BM25 top 50 (parents) + dense top 50 (children -> resolved to parents)
            -> reciprocal rank fusion -> cross-encoder rerank of the top 30

`KnowledgeBase.search(query, k)` returns parent ids, best first, and counts every
call in `kb.searches`: one call is one retrieval pass. `kb.persons(pid)` returns
the person names that spaCy finds in a parent. The module also holds the
24-question benchmark with expected facts, and the rule that decides whether a
text contains a fact.

Usage:
    from shared.nobel_rag import KnowledgeBase, BENCHMARK, has, context_recall
    kb = KnowledgeBase()
    ids = kb.search("Who won the 2026 Nobel Prize in Physics?", k=5)
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
import warnings

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─── The benchmark ───────────────────────────────────────────────────────────

# (id, kind, question, expected facts). A complete context holds every fact.
BENCHMARK = [
    ("s01", "single", "Who won the 2026 Nobel Prize in Physics?", ["Halzen"]),
    ("s02", "single", "Who won the 2025 Nobel Prize in Physics?", ["Clarke", "Devoret", "Martinis"]),
    ("s03", "single", "Who received the 2024 physics Nobel together with Geoffrey Hinton?", ["Hopfield"]),
    ("s04", "single", "Which chemists shared the 2025 Nobel Prize in Chemistry?", ["Kitagawa", "Robson", "Yaghi"]),
    ("s05", "single", "Who received the 2025 Nobel Prize in Physiology or Medicine?", ["Brunkow", "Ramsdell", "Sakaguchi"]),
    ("s06", "single", "Who was awarded the Nobel Peace Prize in 2025?", ["Machado"]),
    ("s07", "single", "Who won the 2026 Nobel Prize in Literature?", ["Anne Carson"]),
    ("s08", "single", "Who won the 2026 Nobel Peace Prize?", ["Pillay"]),
    ("s09", "single", "Who shared the 2026 Nobel Prize in Chemistry?", ["Kagan", "Soai"]),
    ("s10", "single", "Who received the 2026 Nobel Prize in Economic Sciences?", ["Hall", "Woodford"]),
    ("s11", "single", "What is a transmon?", ["charge qubit"]),
    ("s12", "single", "In which year did John Martinis found Qolab?", ["2022"]),
    ("m01", "multi", "Who won the 2025 Nobel Prize in Physics, and which universities are they affiliated with?",
     ["Clarke", "Devoret", "Martinis", "Berkeley", "Santa Barbara"]),
    ("m02", "multi", "Where do the 2025 Nobel laureates in physics work today?", ["Berkeley", "Santa Barbara"]),
    ("m03", "multi", "Which universities are the 2024 physics laureates affiliated with?", ["Princeton", "Toronto"]),
    ("m04", "multi", "At which university is the 2026 physics laureate a professor?", ["Wisconsin"]),
    ("m05", "multi", "Where did the 2026 physics laureate study?", ["Leuven"]),
    ("m06", "multi", "Where do the 2025 chemistry laureates work?", ["Kyoto", "Melbourne"]),
    ("m07", "multi", "Which universities are the 2025 economics laureates affiliated with?",
     ["Northwestern", "Collège de France", "Brown"]),
    ("m08", "multi", "Where do the 2025 laureates in physiology or medicine work?",
     ["Institute for Systems Biology", "Sonoma", "Osaka"]),
    ("m09", "multi", "Who was John Martinis's doctoral advisor, and at which university is that "
     "advisor a professor emeritus?", ["Clarke", "Berkeley"]),
    ("m10", "multi", "Which 2025 physics laureate is chief scientist for quantum hardware at a "
     "technology company, and which company is it?", ["Devoret", "Google"]),
    ("m11", "multi", "Which 2025 Nobel laureates, in any field, are affiliated with Kyoto University?",
     ["Kitagawa", "Sakaguchi"]),
    ("m12", "multi", "Which 2025 economics laureate holds a chair at the Collège de France, and "
     "who is his long-time co-author?", ["Aghion", "Howitt"]),
]


def fold(text: str) -> str:
    """Lower case, accents removed: 'Collège' -> 'college'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c)).lower()


def has(text: str, fact: str) -> bool:
    """Does `text` contain `fact` as whole words (case and accents ignored)?"""
    return re.search(rf"\b{re.escape(fold(fact))}\b", fold(text)) is not None


def context_recall(texts: list[str], facts: list[str]) -> float:
    """Share of the expected facts that appear in at least one context text."""
    return sum(any(has(t, f) for t in texts) for f in facts) / len(facts)


# ─── Indexing: two-level chunks ──────────────────────────────────────────────

def build_units(collection, parent_size: int = 2000, child_size: int = 400,
                child_overlap: int = 50) -> tuple[list[dict], list[dict]]:
    """Split every section into parents of at most `parent_size` characters and every
    parent into children of about `child_size` characters.

    Each unit's `text` starts with "Title > Section:" so that a chunk which only says
    "He joined Yale in 2002" still tells the retriever who "he" is. `body` is the
    chunk without that header.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=0, separators=["\n", ". ", " "])
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=child_overlap, separators=["\n", ". ", " "])
    parents, children = [], []
    for unit in collection:
        header = f"{unit['title']} > {unit['section']}: "
        for p_text in parent_splitter.split_text(unit["text"]):
            parent = {"id": len(parents), "title": unit["title"], "section": unit["section"],
                      "url": unit["url"], "body": p_text, "text": header + p_text}
            parents.append(parent)
            for c_text in child_splitter.split_text(p_text):
                children.append({"id": len(children), "parent": parent["id"],
                                 "body": c_text, "text": header + c_text})
    return parents, children


# ─── Retrieval ───────────────────────────────────────────────────────────────

def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class KnowledgeBase:
    """The `nobel-2025` Wikipedia articles, indexed at two levels, with hybrid search.

    Attributes:
        parents, children: the two chunk levels (lists of dicts, `id` = list index)
        searches: number of `search` calls since the last reset (one retrieval pass each)
    """

    def __init__(self, parent_size: int = 2000, child_size: int = 400):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from transformers.utils import logging as hf_logging

        from shared.collections import load_collection
        from shared.retrieval import document_frequencies
        from shared.text import pipeline

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.collection = load_collection("nobel-2025")
        self.parents, self.children = build_units(self.collection, parent_size, child_size)
        self._pipeline = pipeline
        self._corpus = {p["id"]: pipeline(p["text"]) for p in self.parents}
        self._df = document_frequencies(self._corpus)
        self._bi = SentenceTransformer(EMBED_MODEL)
        self._ce = CrossEncoder(RERANK_MODEL)
        self._emb = self._bi.encode([c["text"] for c in self.children], normalize_embeddings=True,
                                    batch_size=64, show_progress_bar=False)
        self._rankings: dict[str, list[int]] = {}
        self._persons: dict[int, list[str]] = {}
        self._nlp = None
        self.searches = 0

    # ─── one retrieval pass ──────────────────────────────────────────────────

    def _rank(self, query: str, pool: int = 30) -> list[int]:
        """All candidate parents: the reranked top `pool`, then the rest in fused order."""
        from shared.retrieval import rank_collection_bm25

        if query not in self._rankings:
            ranked = rank_collection_bm25(self._pipeline(query), self._corpus, self._df)
            sparse = [(i, s) for i, s in ranked[:50] if s > 0]
            q = self._bi.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = self._emb @ q
            dense, seen = [], set()
            for c in np.argsort(-scores)[:50]:          # children -> their parents
                p = self.children[int(c)]["parent"]
                if p not in seen:
                    seen.add(p)
                    dense.append((p, float(scores[c])))
            fused = [p for p, _ in rrf([sparse, dense])]
            head = fused[:pool]
            ce = self._ce.predict([(query, self.parents[p]["text"]) for p in head],
                                  show_progress_bar=False)
            self._rankings[query] = [head[j] for j in np.argsort(-ce, kind="stable")] + fused[pool:]
        return self._rankings[query]

    def search(self, query: str, k: int = 10) -> list[int]:
        """Top-k parent ids for `query`, best first (hybrid retrieval + reranking).

        Every call counts as one retrieval pass in `self.searches`, also when the
        result comes from the cache.
        """
        self.searches += 1
        return list(self._rank(query)[:k])

    def engine_ranks(self, query: str, pid: int) -> dict:
        """Where parent `pid` ranks for `query` in each engine. Costs no retrieval pass.

        Returns {"bm25": rank of the parent in BM25, "final": rank after fusion and
        reranking (None if not among the candidates), "children": [(dense rank of the
        child among all children, child text), ...]}. Ranks start at 1.
        """
        from shared.retrieval import rank_collection_bm25

        bm25 = [i for i, _ in rank_collection_bm25(self._pipeline(query), self._corpus, self._df)]
        q = self._bi.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        position = np.empty(len(self.children), dtype=int)
        position[np.argsort(-(self._emb @ q))] = np.arange(1, len(self.children) + 1)
        final = self._rank(query)
        return {"bm25": bm25.index(pid) + 1 if pid in bm25 else None,
                "final": final.index(pid) + 1 if pid in final else None,
                "children": [(int(position[c["id"]]), c["body"]) for c in self.children
                             if c["parent"] == pid]}

    # ─── helpers on parents ──────────────────────────────────────────────────

    def text(self, pid: int) -> str:
        """The body of parent `pid`, the text a generator would receive."""
        return self.parents[pid]["body"]

    def label(self, pid: int) -> str:
        """Short name of a parent: 'Title > Section'."""
        return f"{self.parents[pid]['title']} > {self.parents[pid]['section']}"

    def persons(self, pid: int) -> list[str]:
        """Person names (two words or more) that spaCy finds in parent `pid`, in order of
        first appearance, without repeats. Costs no retrieval pass."""
        if pid not in self._persons:
            if self._nlp is None:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            names = []
            for ent in self._nlp(self.text(pid)).ents:
                name = " ".join(ent.text.split())
                if ent.label_ == "PERSON" and len(name.split()) >= 2 and name not in names:
                    names.append(name)
            self._persons[pid] = names
        return list(self._persons[pid])
