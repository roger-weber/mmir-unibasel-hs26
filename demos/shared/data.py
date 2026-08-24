"""Data loading utilities (Project Gutenberg, etc.)."""

import os
import json
from typing import Optional
from pathlib import Path


GUTENBERG_CACHE_DIR = Path("data/gutenberg")


def load_gutenberg_book(book_id: int, reload: bool = False):
    """
    Load a book from Project Gutenberg (cached locally).

    Args:
        book_id: Gutenberg book ID.
        reload: Force re-download even if cached.

    Returns:
        LangChain Document with book text and metadata.
    """
    from langchain_core.documents import Document
    from urllib.request import urlopen
    from bs4 import BeautifulSoup

    GUTENBERG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = GUTENBERG_CACHE_DIR / f"{book_id}.json"
    text_path = GUTENBERG_CACHE_DIR / f"{book_id}.txt"

    # Return from cache if available
    if not reload and meta_path.exists() and text_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        with open(text_path, "rb") as f:
            content = f.read().decode("utf-8").replace("\r", "")
        return Document(page_content=content, metadata=metadata)

    # Download metadata
    print(f"Downloading metadata from https://www.gutenberg.org/ebooks/{book_id}")
    with urlopen(f"https://www.gutenberg.org/ebooks/{book_id}") as response:
        page = BeautifulSoup(response, "html.parser")

    rows = page.find("table", {"class": "bibrec"}).find_all("tr")
    rows.reverse()
    fields = {
        row.find("th").get_text().lower().strip(): row.find("td").get_text().strip()
        for row in rows
        if row.find("th")
    }
    author_parts = fields.get("author", "Unknown").split(",")
    yearspan = author_parts.pop().strip() if len(author_parts) > 1 else ""

    metadata = {
        "id": book_id,
        "title": fields.get("title", "Unknown"),
        "author": ",".join(author_parts).strip(),
        "yearspan": yearspan,
        "language": fields.get("language", "Unknown"),
    }

    # Download text
    text = _download_gutenberg_text(book_id)
    if text is None:
        raise RuntimeError(f"Could not download text for Gutenberg book {book_id}")

    # Cache
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    with open(text_path, "wb") as f:
        f.write(text.encode("utf-8"))

    return Document(page_content=text, metadata=metadata)


def _download_gutenberg_text(book_id: int) -> Optional[str]:
    """Try multiple Gutenberg URLs to download the book text."""
    from urllib.request import urlopen

    START_MARKERS = ["*** START OF"]
    END_MARKERS = ["*** END OF"]

    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in urls:
        try:
            print(f"  Trying {url}")
            with urlopen(url) as response:
                text_lines = []
                started = False
                for line in response:
                    line = line.decode("utf-8-sig").strip()
                    if not started:
                        if any(line.startswith(m) for m in START_MARKERS):
                            started = True
                        continue
                    if any(line.startswith(m) for m in END_MARKERS):
                        break
                    text_lines.append(line)
                if text_lines:
                    return "\n".join(text_lines).strip()
        except Exception:
            continue
    return None


def get_cached_book_ids() -> list[int]:
    """Return list of Gutenberg book IDs available in local cache."""
    GUTENBERG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return [
        int(f.stem)
        for f in GUTENBERG_CACHE_DIR.glob("*.json")
    ]
