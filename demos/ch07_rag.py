"""
A hand-built RAG pipeline for Chapter 7, one function per stage of the book's figure.

    index:  sections -> parents (<= 2000 chars) -> children (~400 chars)
            BM25 over parents, dense embeddings over children
    query:  sparse top-k (parents) + dense top-k (children -> resolved to parents)
            -> reciprocal rank fusion -> cross-encoder rerank -> prompt -> generator

Every unit carries its title, section, and source URL, so citations survive to the end.

Usage:
    from shared.collections import load_collection
    from ch07_rag import build_units, SparseIndex, DenseIndex, resolve_to_parents, rrf

    parents, children = build_units(load_collection("nobel-2025"))
    sparse, dense = SparseIndex(parents), DenseIndex(children, model)
    fused = rrf([sparse.search(q, 50), resolve_to_parents(dense.search(q, 50), children)])
"""

import numpy as np

from shared.retrieval import document_frequencies, rank_collection_bm25
from shared.text import pipeline


# ─── Indexing: two-level chunks ──────────────────────────────────────────────

def build_units(collection, parent_size: int = 2000, child_size: int = 400,
                child_overlap: int = 50) -> tuple[list[dict], list[dict]]:
    """
    Split every section into parents of at most `parent_size` characters and every
    parent into children of about `child_size` characters.

    Each unit's `text` starts with "Title > Section:" so that a chunk which only says
    "He joined Yale in 2002" still tells the retriever who "he" is.
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


def label(unit: dict) -> str:
    """Short name of a parent: 'Title > Section'."""
    return f"{unit['title']} > {unit['section']}"


# ─── Retrieval ───────────────────────────────────────────────────────────────

class SparseIndex:
    """BM25 over parent chunks (tokenize, stopwords, Porter stemming)."""

    def __init__(self, units: list[dict]):
        self.corpus = {u["id"]: pipeline(u["text"]) for u in units}
        self.df = document_frequencies(self.corpus)

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        ranked = rank_collection_bm25(pipeline(query), self.corpus, self.df)
        return [(i, s) for i, s in ranked[:k] if s > 0]


class DenseIndex:
    """Bi-encoder embeddings over child chunks, cosine similarity."""

    def __init__(self, units: list[dict], model):
        self.model = model
        self.emb = model.encode([u["text"] for u in units], normalize_embeddings=True,
                                batch_size=64, show_progress_bar=False)

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


def resolve_to_parents(child_hits: list[tuple[int, float]], children: list[dict]) -> list[tuple[int, float]]:
    """Replace each child by its parent and keep only the best-ranked hit per parent."""
    seen, out = set(), []
    for c, score in child_hits:
        p = children[c]["parent"]
        if p not in seen:
            seen.add(p)
            out.append((p, score))
    return out


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def rerank(query: str, hits: list[tuple[int, float]], units: list[dict], cross_encoder,
           top_k: int = 5) -> list[tuple[int, float]]:
    """Score every (query, candidate) pair with a cross-encoder and keep the best `top_k`."""
    ids = [i for i, _ in hits]
    scores = cross_encoder.predict([(query, units[i]["text"]) for i in ids], show_progress_bar=False)
    order = np.argsort(-scores)[:top_k]
    return [(ids[j], float(scores[j])) for j in order]


def retrieve(query: str, sparse: "SparseIndex", dense: "DenseIndex", children: list[dict],
             parents: list[dict], cross_encoder, pool: int = 30,
             top_k: int = 5) -> list[tuple[int, float]]:
    """The full retrieval chain of ch07-02: BM25 + dense -> RRF -> rerank the top `pool`."""
    fused = rrf([sparse.search(query, 50),
                 resolve_to_parents(dense.search(query, 50), children)])
    return rerank(query, fused[:pool], parents, cross_encoder, top_k=top_k)


# ─── Reader: extractive span prediction ──────────────────────────────────────

def read_span(question: str, passage: str, tokenizer, model, max_answer_tokens: int = 30,
              max_length: int = 384, stride: int = 128) -> dict:
    """
    Extractive QA as in the book: score every passage token as a start and an end of
    the answer and return the span with the highest start + end score.

    The span must lie inside the passage (not the question), end after it starts, and
    be at most `max_answer_tokens` long. Passages longer than the model's window are
    read in overlapping windows; the best span over all windows wins.

    Returns {"answer", "score", "start", "end"} with character offsets into `passage`.
    """
    import torch

    enc = tokenizer(question, passage, truncation="only_second", max_length=max_length,
                    stride=stride, return_overflowing_tokens=True,
                    return_offsets_mapping=True, padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    with torch.no_grad():
        out = model(**enc)

    best = {"answer": "", "score": float("-inf"), "start": 0, "end": 0}
    for w in range(len(offsets)):
        in_passage = [s == 1 for s in enc.sequence_ids(w)]
        start_logits, end_logits = out.start_logits[w], out.end_logits[w]
        for i in torch.argsort(start_logits, descending=True)[:20].tolist():
            if not in_passage[i]:
                continue
            for j in torch.argsort(end_logits, descending=True)[:20].tolist():
                if not in_passage[j] or j < i or j - i + 1 > max_answer_tokens:
                    continue
                score = float(start_logits[i] + end_logits[j])
                if score > best["score"]:
                    a, b = int(offsets[w][i][0]), int(offsets[w][j][1])
                    best = {"answer": passage[a:b], "score": score, "start": a, "end": b}
    return best


# ─── Generation ──────────────────────────────────────────────────────────────

PROMPT = """You answer questions about {domain} using only the passages below.
Cite the passages you rely on by id, like [1]. If the answer is not in
them, say you don't know rather than guessing.

<passages>
{passages}
</passages>

<question>
{query}
</question>

Answer in {format}, and base every claim on the passages above."""


def format_passages(ids: list[int], units: list[dict]) -> str:
    """Numbered passages with their source, in the order given."""
    return "\n\n".join(
        f"[{n}] ({label(units[i])})\n{units[i]['body']}" for n, i in enumerate(ids, start=1))


def build_prompt(query: str, ids: list[int], units: list[dict], *,
                 domain: str = "the Nobel Prizes", format: str = "two or three sentences") -> str:
    """Fill the book's grounded-answer prompt skeleton."""
    return PROMPT.format(domain=domain, passages=format_passages(ids, units),
                         query=query, format=format)


def ask(prompt: str, cached: str, **kwargs) -> tuple[str, bool]:
    """
    Call the generator; without AWS access, return the cached answer instead.

    Returns (answer, live), where `live` is False when the cached answer was used.
    """
    from shared.llm import invoke_claude, AWSConnectionError

    try:
        return invoke_claude(prompt, **kwargs), True
    except AWSConnectionError:
        return cached, False


# Answers recorded with Claude Sonnet 5.5 (ch07-02 on 2026-10-22, ch07-03 on 2026-10-25),
# shown when Bedrock is unavailable.
CACHED = {
    "memory-2025": (
        "The 2025 Nobel Prize in Physics went to **John Clarke, Michel H. Devoret, and John M. "
        "Martinis**. They were honored \"for the discovery of macroscopic quantum mechanical "
        "tunnelling and energy quantisation in an electric circuit.\""),
    "memory-2026": (
        "I don't know who won the 2026 Nobel Prize in Physics. My training data doesn't include "
        "it, and the announcement usually comes in early October. I don't want to guess and give "
        "you a wrong name."),
    "rag-2026": (
        "Francis Halzen won the 2026 Nobel Prize in Physics \"for decisive contributions to the "
        "IceCube Neutrino Observatory and the discovery of high-energy neutrinos of astrophysical "
        "origin\" [1][5]. The Royal Swedish Academy of Sciences credited his scientific leadership "
        "in developing IceCube as an instrument for neutrino astronomy [1]."),
    "haystack-2026": (
        "Francis Halzen won the 2026 Nobel Prize in Physics. The prize recognized \"decisive "
        "contributions to the IceCube Neutrino Observatory and the discovery of high-energy "
        "neutrinos of astrophysical origin.\""),
    # ch07-03 result generation
    "gen-2025": (
        "John Clarke, Michel Devoret, and John M. Martinis won the 2025 Nobel Prize in Physics "
        "\"for the discovery of macroscopic quantum mechanical tunnelling and energy "
        "quantisation in an electric circuit\" [1][4]. Passage [5] adds that the three worked "
        "together in a team Clarke led in the 1980s, and that their work on macroscopic quantum"
        " phenomena using the Josephson effect earned them the prize [5]."),
    "gen-contributions": (
        "The 2025 Nobel Prize in Physics was awarded to John Clarke, Michel Devoret, and John "
        "M. Martinis \"for the discovery of macroscopic quantum mechanical tunnelling and energy"
        " quantisation in an electric circuit\" [2][5]. The passages give only this prize "
        "citation. They do not describe the experiments or the work in more detail."),
    "gen-2027": (
        "I don't know who won the Nobel Prize in Physics in 2027. The passages only cover "
        "laureates up to 2026. For 2026, [2] lists Francis Halzen, and for 2025, [2] and [3] "
        "list John Clarke, Michel Devoret, and John M. Martinis. None of the passages mention a"
        " 2027 prize or its winner."),
    "synth-names": (
        "```json\n[\"John Clarke\", \"Michel Devoret\", \"John M. Martinis\"]\n```"),
    "synth-answer": (
        "{\"answer\": \"John Clarke: He is a Professor Emeritus at the University of California, "
        "Berkeley [5]. He retired from Lawrence Berkeley National Laboratory as a faculty "
        "senior scientist in 2010 [4]. The passages do not say where else he currently "
        "works.\\n\\nMichel Devoret: He is Professor of Physics at the University of California, "
        "Santa Barbara, and Professor Emeritus of Applied Physics at Yale. He also serves as "
        "Chief Scientist for Quantum Hardware at Google Quantum AI [7].\\n\\nJohn M. Martinis: He"
        " is Professor of Physics at the University of California, Santa Barbara [9]. He joined"
        " the start-up Silicon Quantum Computing in 2020, and in 2026 he was appointed to the "
        "President's Council of Advisors on Science and Technology (PCAST) [8]. The passages do"
        " not say whether he still works at Silicon Quantum Computing.\", \"citations\": [{\"id\": "
        "5, \"quote\": \"John Clarke (born 10 February 1942) is a British experimental physicist "
        "and Professor Emeritus at the University of California, Berkeley.\"}, {\"id\": 4, "
        "\"quote\": \"eventually retiring as a faculty senior scientist in the Materials Sciences "
        "Division in 2010\"}, {\"id\": 7, \"quote\": \"He is Professor of Physics at the University "
        "of California, Santa Barbara, and Professor Emeritus of Applied Physics at Yale "
        "University.\"}, {\"id\": 7, \"quote\": \"He serves as the Chief Scientist for Quantum "
        "Hardware at Google Quantum AI.\"}, {\"id\": 9, \"quote\": \"is an American physicist and "
        "Professor of Physics at the University of California, Santa Barbara.\"}, {\"id\": 8, "
        "\"quote\": \"In 2020, Martinis joined Silicon Quantum Computing, a start-up founded in "
        "Australia by Professor Michelle Simmons.\"}, {\"id\": 8, \"quote\": \"In 2026, he was "
        "appointed to the President's Council of Advisors on Science and Technology (PCAST) by "
        "President Donald Trump.\"}]}"),
    "single-pass": (
        "**John Clarke:** The passages don't say where he works. They list him as a 2025 "
        "laureate with the United Kingdom as his country [1], but that doesn't tell me his "
        "employer, so I don't know.\n\n**Michel Devoret:** The passages list him among the "
        "\"current and past\" UC Santa Barbara Nobel laureate faculty, with his 2025 Physics "
        "prize [7]. They don't say whether he is currently at UCSB or was there in the past, so"
        " I can't confirm where he works today. [1] lists him as a 2025 laureate with the "
        "United States as his country.\n\n**John M. Martinis:** He is an American physicist and "
        "Professor of Physics at the University of California, Santa Barbara [6]. He is also "
        "listed among UCSB's Nobel laureate faculty [7]. The passages say he led a team at the "
        "Quantum AI Lab that developed a superconducting quantum computer [6], but they don't "
        "say whether he is still there."),
}
