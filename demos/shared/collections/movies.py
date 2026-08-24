"""
Movies collection — Kaggle movie metadata as retrieval units.

Downloads the "rounakbanik/the-movies-dataset" from Kaggle (via kagglehub),
processes it into retrieval units with rich metadata for filtering.

Each movie becomes one retrieval unit whose `text` field combines title,
overview, tagline, cast, and genres into a searchable stream.

Registers:
    - "movies"       → full collection (~45,000 movies)
    - "movies-small" → first 500 movies (fast loading for demos)

Requires: kagglehub (+ Kaggle API credentials for first download)
"""

import ast
import csv
import json
import os
from pathlib import Path

from shared.collections import Collection, register, register_variant, CACHE_DIR


DATASET_ID = "rounakbanik/the-movies-dataset"
CACHE_FILE = CACHE_DIR / "movies" / "movies.jsonl"


def _download_dataset() -> Path:
    """Download dataset via kagglehub and return the local path."""
    import kagglehub
    return Path(kagglehub.dataset_download(DATASET_ID))


def _load_cast(dataset_path: Path, max_records: int) -> dict[str, str]:
    """Load cast data, keeping only top 2% actors for relevance."""
    credits_path = dataset_path / "credits.csv"
    if not credits_path.exists():
        return {}

    actor_counts: dict[str, int] = {}
    rows = []

    with open(credits_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cast = ast.literal_eval(row["cast"])
            except (ValueError, SyntaxError):
                continue
            rows.append({"id": row["id"], "cast": cast})
            for person in cast:
                name = person.get("name", "")
                if name:
                    actor_counts[name] = actor_counts.get(name, 0) + 1
            if len(rows) >= max_records:
                break

    # Keep top 2% of actors
    n_top = max(1, len(actor_counts) // 50)
    top_actors = set(
        sorted(actor_counts, key=actor_counts.get, reverse=True)[:n_top]
    )

    cast_lookup = {}
    for row in rows:
        top_cast = [
            f"{p['name']} as {p.get('character', '?')}"
            for p in row["cast"]
            if p.get("name", "") in top_actors
        ]
        if top_cast:
            cast_lookup[row["id"]] = ", ".join(top_cast)

    return cast_lookup


def _build_cache(max_records: int) -> list[dict]:
    """Download, process, and cache the movie dataset."""
    print("  Downloading movie dataset from Kaggle...")
    dataset_path = _download_dataset()

    print("  Loading cast data...")
    cast_lookup = _load_cast(dataset_path, max_records)

    print("  Processing movie metadata...")
    records = []
    metadata_path = dataset_path / "movies_metadata.csv"

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip adult content, entries without runtime/date
            if row.get("adult") == "True":
                continue
            if not row.get("runtime") or not row.get("release_date"):
                continue
            try:
                genres = [g["name"] for g in ast.literal_eval(row["genres"])]
                runtime = int(float(row["runtime"]))
                rating = float(row["vote_average"])
                year = int(row["release_date"][:4])
            except (ValueError, SyntaxError, KeyError):
                continue

            record = {
                "imdb_id": row.get("imdb_id", ""),
                "title": row.get("title", ""),
                "overview": row.get("overview", ""),
                "tagline": row.get("tagline", ""),
                "cast": cast_lookup.get(row["id"], ""),
                "genres": genres,
                "runtime": runtime,
                "rating": rating,
                "year": year,
            }
            records.append(record)
            if len(records) >= max_records:
                break

    # Write cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Cached {len(records)} movies to {CACHE_FILE}")

    return records


def _load_from_cache() -> list[dict] | None:
    """Load from JSONL cache if it exists and has enough data."""
    if not CACHE_FILE.exists():
        return None
    records = []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records if records else None


def _movie_to_text(record: dict) -> str:
    """Combine movie fields into a searchable text stream."""
    parts = [
        record.get("title", ""),
        record.get("overview", ""),
        record.get("tagline", ""),
        record.get("cast", ""),
        " ".join(record.get("genres", [])),
    ]
    return " ".join(p for p in parts if p)


@register("movies", description="Kaggle movie dataset (~45,000 movies with metadata)")
class MoviesCollection(Collection):
    """
    Movie metadata as retrieval units.

    Each movie's text field combines title, overview, tagline, cast, and genres.
    Rich metadata (year, rating, runtime, genres) enables filtering demos.

    Args:
        max_records: Maximum movies to load (default 45000).
    """

    description = "Kaggle movie dataset (~45,000 movies with metadata)"
    language = "en"

    def __init__(self, max_records: int = 45000):
        self._max_records = max_records
        super().__init__()

    def _load(self):
        # Try cache first
        records = _load_from_cache()
        if records is None or len(records) < self._max_records:
            records = _build_cache(self._max_records)

        # Trim to requested size
        records = records[: self._max_records]

        for i, rec in enumerate(records):
            doc_id = rec.get("imdb_id") or f"movie-{i}"
            self._docs[doc_id] = {
                "id": doc_id,
                "source": "kaggle/the-movies-dataset",
                "text": _movie_to_text(rec),
                # Metadata for filtering
                "title": rec.get("title", ""),
                "year": rec.get("year", 0),
                "rating": rec.get("rating", 0.0),
                "runtime": rec.get("runtime", 0),
                "genres": rec.get("genres", []),
                "tagline": rec.get("tagline", ""),
                "cast": rec.get("cast", ""),
            }


# ─── Register small variant for demos ───────────────────────────────────────

def _make_small_factory():
    def factory():
        c = MoviesCollection(max_records=500)
        c.name = "movies-small"
        c.description = "500 movies (fast loading for demos)"
        return c
    return factory


register_variant(
    name="movies-small",
    description="500 movies (fast loading for demos)",
    factory=_make_small_factory(),
)
