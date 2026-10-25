"""
Chunking helpers for Chapter 7 (Retrieval Augmented Generation).

The fixed-size, sentence, structural, and Markdown-header splitters come straight
from `langchain_text_splitters` in the notebook. This module holds the parts that
LangChain does not provide in the form the book describes:

- `size_stats(chunks)`, `plot_sizes(...)`: the chunk-length summaries and histograms.
- `merge_small(paragraphs)`: merge adjacent paragraphs up to a size cap.
- `add_sentence_overlap(chunks)`: carry the neighbouring sentences across each boundary.
- `semantic_split(sentences, model)`: iterative similarity merge of neighbours.
- `fold_small_chunks(chunks, model)`: fold leftover fragments into their closest neighbour.
- `to_markdown(text)`: turn the Gutenberg PART / CHAPTER lines into Markdown headings.
- `embed`, `embed_queries`, `top_k`: encoding and dense search over chunks.
- `answer_recall(...)`: does the top-k of a chunking strategy contain the answer?
- `boundary_cohesion(chunks, model)`: label-free check of where the boundaries fall.

Usage:
    from shared.data import load_gutenberg_book
    from ch07_chunking import semantic_split
    text = load_gutenberg_book(244).page_content     # A Study in Scarlet
"""

import re

import numpy as np

COLOR = "#4C72B0"


# ─── Chunk sizes ─────────────────────────────────────────────────────────────

def size_stats(chunks: list[str]) -> dict:
    """Count, mean, median, min, max, and share under 200 characters."""
    xs = sorted(len(c) for c in chunks)
    n = len(xs)
    return {
        "chunks": n,
        "mean": round(sum(xs) / n),
        "median": xs[n // 2],
        "min": xs[0],
        "max": xs[-1],
        "under 200": f"{100 * sum(x < 200 for x in xs) / n:.0f}%",
    }


def stats_rows(named_chunks: dict[str, list[str]]) -> tuple[list[list], list[str]]:
    """Rows and headers for `print_table`, one row per named chunk list."""
    rows = [[name, *size_stats(c).values()] for name, c in named_chunks.items()]
    headers = ["Strategy", "Chunks", "Mean", "Median", "Min", "Max", "Under 200"]
    return rows, headers


def plot_sizes(named_chunks: dict[str, list[str]], xmax: int | None = None) -> None:
    """One chunk-length histogram per entry, stacked vertically (the book's plots)."""
    import matplotlib.pyplot as plt

    n = len(named_chunks)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), squeeze=False)
    for ax, (title, chunks) in zip(axes[:, 0], named_chunks.items()):
        ax.hist([len(c) for c in chunks], bins=40, color=COLOR, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{title} ({len(chunks)} chunks)", fontsize=10)
        ax.set_xlabel("chunk length (characters)", fontsize=9)
        ax.set_ylabel("number of chunks", fontsize=9)
        if xmax:
            ax.set_xlim(0, xmax)
    fig.tight_layout()
    plt.show()


# ─── Structural and semantic merging ─────────────────────────────────────────

def merge_small(units: list[str], max_size: int = 1000) -> list[str]:
    """Greedily join adjacent units with a space while the result stays under `max_size`."""
    merged, buf = [], ""
    for u in units:
        if not buf:
            buf = u
        elif len(buf) + 1 + len(u) <= max_size:
            buf = buf + " " + u
        else:
            merged.append(buf)
            buf = u
    if buf:
        merged.append(buf)
    return merged


def add_sentence_overlap(chunks: list[str]) -> list[str]:
    """Prefix each chunk with the last sentence of the previous one and append the next one's first."""
    import nltk

    sents = [nltk.sent_tokenize(c) for c in chunks]
    out = []
    for i, s in enumerate(sents):
        before = sents[i - 1][-1] + " " if i > 0 else ""
        after = " " + sents[i + 1][0] if i < len(sents) - 1 else ""
        out.append(before + chunks[i] + after)
    return out


def _pack(sentences: list[str], size: int) -> list[str]:
    """Pack sentences into groups of about `size` characters (no overlap)."""
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > size:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        out.append(cur)
    return out


def semantic_split(
    sentences: list[str],
    model,
    *,
    threshold: float,
    max_size: int = 1000,
    start_size: int = 100,
    merge_share: float = 0.2,
) -> list[str]:
    """
    Group sentences by meaning (the book's semantic splitting).

    Start from groups of about `start_size` characters. In each pass, embed the
    chunks, score every adjacent pair whose merge stays under `max_size`, and merge
    the best `merge_share` of the pairs whose similarity exceeds `threshold`. The
    score prefers small merges, so chunks grow evenly. Stop when a pass shrinks the
    chunk count by less than 10%.
    """
    chunks = _pack(sentences, start_size)
    previous = len(chunks) * 2
    while len(chunks) * 1.1 < previous:
        previous = len(chunks)
        emb = embed(chunks, model)
        candidates = []
        for i in range(len(chunks) - 1):
            size = len(chunks[i]) + len(chunks[i + 1])
            if size >= max_size:
                continue
            sim = float(emb[i] @ emb[i + 1])
            if sim > threshold:
                candidates.append((i, (max_size - size) / max_size * sim))
        candidates.sort(key=lambda c: c[1], reverse=True)
        chosen = sorted((i for i, _ in candidates[: int(merge_share * len(chunks))]), reverse=True)
        for i in chosen:  # right to left, so earlier indices stay valid
            if len(chunks[i]) + len(chunks[i + 1]) < max_size:
                chunks[i : i + 2] = [chunks[i] + " " + chunks[i + 1]]
    return chunks


def fold_small_chunks(chunks: list[str], model, *, small: int = 200, rounds: int = 20) -> list[str]:
    """
    Fold every chunk shorter than `small` into its semantically closer neighbour.

    Smallest chunks go first, and each chunk takes part in at most one merge per
    round, so runs of fragments (dialogue) coalesce with each other before they
    attach to a large neighbour.
    """
    chunks = list(chunks)
    for _ in range(rounds):
        smalls = [i for i, c in enumerate(chunks) if len(c) < small]
        if not smalls or len(chunks) <= 1:
            break
        emb = embed(chunks, model)
        used, merges = set(), []
        for i in sorted(smalls, key=lambda k: len(chunks[k])):
            if i in used:
                continue
            options = [
                (j, float(emb[i] @ emb[j]))
                for j in (i - 1, i + 1)
                if 0 <= j < len(chunks) and j not in used
            ]
            if not options:
                continue
            options.sort(key=lambda o: (o[1], -len(chunks[o[0]])), reverse=True)
            j = options[0][0]
            used.update((i, j))
            merges.append((min(i, j), max(i, j)))
        if not merges:
            break
        for lo, hi in sorted(merges, reverse=True):
            chunks[lo : hi + 1] = [chunks[lo] + " " + chunks[hi]]
    return chunks


def to_markdown(text: str) -> str:
    """
    Mark the book's structure as Markdown headings.

    `PART I.` becomes `# PART I.` and `CHAPTER III.` (plus its title on the next
    line) becomes `## CHAPTER III. THE LAURISTON GARDENS MYSTERY`. The table of
    contents at the top is dropped.
    """
    body = text[text.index("A STUDY IN SCARLET.", 200):]  # skip the table of contents
    body = re.sub(r"^(PART [IVX]+\.)[^\n]*\n+(?:([A-Z][A-Z ,.'’-]+)\n)?", r"# \1 \2\n", body, flags=re.M)
    body = re.sub(r"^(CHAPTER [IVX]+\.)\n+([A-Z][^\n]+)\n", r"## \1 \2\n", body, flags=re.M)
    return body


# ─── Dense search over chunks ────────────────────────────────────────────────

def embed(chunks: list[str], model) -> np.ndarray:
    """Unit-length document embeddings, one row per chunk."""
    return model.encode(chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32)


def embed_queries(queries: list[str], model) -> np.ndarray:
    """Unit-length query embeddings; uses the model's `query` prompt if it has one (Qwen3)."""
    kwargs = {"prompt_name": "query"} if "query" in (model.prompts or {}) else {}
    return model.encode(queries, normalize_embeddings=True, show_progress_bar=False, **kwargs)


def top_k(query_emb: np.ndarray, chunk_emb: np.ndarray, k: int = 3) -> list[int]:
    """Indices of the `k` chunks most similar to the query embedding."""
    return [int(i) for i in np.argsort(-(chunk_emb @ query_emb))[:k]]


def contains(chunk: str, answer: str) -> bool:
    """Case- and whitespace-insensitive substring test."""
    norm = lambda s: re.sub(r"\s+", " ", s).lower()
    return norm(answer) in norm(chunk)


def answer_recall(
    answers: list[str],
    query_emb: np.ndarray,
    chunks: list[str],
    chunk_emb: np.ndarray,
    k: int = 3,
    context: list[str] | None = None,
) -> tuple[float, int]:
    """
    Answer recall at k and the mean number of context characters per question.

    A question counts as answered if any of its top-k chunks contains the answer
    string. With `context`, chunk i is matched but `context[i]` (its parent) is
    what the generator would receive, so recall and size are measured on that.
    """
    hits, chars = 0, 0
    for answer, q in zip(answers, query_emb):
        ids = top_k(q, chunk_emb, k)
        texts = [context[i] for i in ids] if context else [chunks[i] for i in ids]
        texts = list(dict.fromkeys(texts))  # two children of one parent: send it once
        hits += any(contains(t, answer) for t in texts)
        chars += sum(len(t) for t in texts)
    return hits / len(answers), round(chars / len(answers))


def boundary_cohesion(chunks: list[str], model) -> tuple[float, float]:
    """
    Mean similarity of adjacent sentences inside a chunk and across a chunk boundary.

    Chunks must not overlap, otherwise the sentences on both sides of a boundary
    are copies of each other.
    """
    import nltk

    per_chunk = [nltk.sent_tokenize(re.sub(r"\s+", " ", c)) for c in chunks]
    sentences = [s for chunk in per_chunk for s in chunk]
    emb = embed(sentences, model)
    within, across, pos = [], [], 0
    for chunk in per_chunk:
        within += [float(emb[pos + j] @ emb[pos + j + 1]) for j in range(len(chunk) - 1)]
        pos += len(chunk)
        if pos < len(sentences):
            across.append(float(emb[pos - 1] @ emb[pos]))
    return float(np.mean(within)), float(np.mean(across))
