"""Semantic-search movie collection for the Chapter 5 exercise.

Loads the 500-movie `movies-small` slice together with its pre-computed sentence
embeddings, so the exercise runs offline without a Kaggle download and without
encoding 500 documents at student runtime. Only the student's live *query* is
encoded by a model in the notebook.

Each movie is a plain dict with this surface:

    id, title, year, rating, runtime, genres (list[str]), cast (str),
    tagline, text, tokens (list[str]),
    emb_mini  (np.ndarray, 384d, L2-normalised, all-MiniLM-L6-v2),
    emb_qwen  (np.ndarray, 1024d, L2-normalised, Qwen3-Embedding-0.6B)

The movies are returned as a list (no index structure): semantic search here is
a linear scan over the list, which is the point of the exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared.text import tokenize, remove_stopwords

_DATA = Path(__file__).resolve().parent / "data"


def load_movies(data_dir: Path | None = None) -> list[dict]:
    """Load the 500 movies-small movies with both embedding sets attached.

    Returns a list of movie dicts in the embedding matrices' row order, so
    `movies[i]` corresponds to row `i` of each `.npy` file.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA

    bundle = json.loads((data_dir / "movies_small.json").read_text(encoding="utf-8"))
    records = bundle["movies"]

    emb_mini = np.load(data_dir / "minilm_movies_small.npy")   # (500, 384)
    emb_qwen = np.load(data_dir / "qwen_movies_small.npy")     # (500, 1024)
    if not (len(records) == emb_mini.shape[0] == emb_qwen.shape[0]):
        raise ValueError("movie metadata and embedding matrices disagree in length")

    movies: list[dict] = []
    for i, rec in enumerate(records):
        movie = dict(rec)
        movie["tokens"] = remove_stopwords(tokenize(movie["text"]))
        movie["emb_mini"] = emb_mini[i]
        movie["emb_qwen"] = emb_qwen[i]
        movies.append(movie)
    return movies


# Which embedding field each model choice uses.
EMBEDDING_FIELD = {"mini": "emb_mini", "qwen": "emb_qwen"}
