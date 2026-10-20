"""
Build the movie collection of demo ch06-04 (book section 6.5, running example).

Selects the 10,000 most-voted movies of the Kaggle movies dataset (the ch04 movies
table), keeps title, overview, main actors, genres, year, and rating, and embeds every
overview with Qwen3-Embedding-0.6B (1,024 dimensions, unit length). The demo queries are
embedded with the model's query prompt, so the notebook runs without torch.

Outputs (next to this file):
    movies.jsonl     one movie per line, `id` = row number of the embedding matrix
    embeddings.npz   docs (n x 1024, float16), queries (q x 1024, float16), query_texts

Run once from demos/ in an environment with sentence-transformers and a GPU:
    python -m eval.ch06_movies.make_movies
"""

import csv
import os
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SOURCE = Path(__file__).parents[2] / "data" / ".cache" / "movies" / "movies.jsonl"
KAGGLE = Path.home() / ".cache/kagglehub/datasets/rounakbanik/the-movies-dataset/versions"
N_MOVIES = 10_000
MODEL = "Qwen/Qwen3-Embedding-0.6B"

QUERIES = [
    "a mind-bending heist thriller",
    "a lonely robot falls in love",
    "astronauts travel through a wormhole to save humanity",
    "a con artist pretends to be a pilot and a doctor",
    "a shark terrorizes a beach town",
]


def vote_counts() -> dict[str, int]:
    meta = next(KAGGLE.glob("*/movies_metadata.csv"))
    counts = {}
    with open(meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                counts[row["imdb_id"]] = int(float(row["vote_count"]))
            except (KeyError, TypeError, ValueError):
                continue
    return counts


def main_actors(cast: str, n: int = 5) -> list[str]:
    """'Leonardo DiCaprio as Dom Cobb, Joseph Gordon-Levitt as Arthur' -> names."""
    return re.findall(r"(?:^|, )([^,]+?) as ", cast)[:n]


def main() -> None:
    votes = vote_counts()
    records = [json.loads(line) for line in SOURCE.open(encoding="utf-8")]
    records = [r for r in records if r["overview"].strip() and main_actors(r["cast"])]
    records.sort(key=lambda r: votes.get(r["imdb_id"], 0), reverse=True)

    movies, seen = [], set()
    for r in records:
        if r["imdb_id"] in seen:
            continue
        seen.add(r["imdb_id"])
        movies.append({
            "id": len(movies), "imdb_id": r["imdb_id"], "title": r["title"],
            "year": r["year"], "rating": r["rating"], "votes": votes.get(r["imdb_id"], 0),
            "genres": r["genres"], "actors": main_actors(r["cast"]), "overview": r["overview"],
        })
        if len(movies) == N_MOVIES:
            break

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() and os.environ.get("CH06_DEVICE") != "cpu" else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    docs = model.encode([m["overview"] for m in movies], batch_size=64,
                        normalize_embeddings=True, show_progress_bar=True)
    queries = model.encode(QUERIES, prompt_name="query", normalize_embeddings=True)

    with open(HERE / "movies.jsonl", "w", encoding="utf-8") as f:
        for m in movies:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    np.savez_compressed(HERE / "embeddings.npz", docs=docs.astype(np.float16),
                        queries=queries.astype(np.float16), query_texts=np.array(QUERIES))
    print(f"wrote {len(movies):,} movies; min votes {movies[-1]['votes']}")


if __name__ == "__main__":
    main()
