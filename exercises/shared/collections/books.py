"""
Gutenberg books collection — chapters as retrieval units.

Downloads books from Project Gutenberg, splits by chapter markers,
and produces one retrieval unit per chapter.

Registers:
    - "books"          → all books combined
    - "books-en"       → English books only
    - "books-de"       → German books only
    - "books-fr"       → French books only
    - "books-es"       → Spanish books only
    - "books-pl"       → Polish book(s)
    - "books-zh"       → Chinese text(s)
    - "books-scarlet"  → A Study in Scarlet (individual)
    - "books-alice"    → Alice in Wonderland (individual)
    - ... (one per English book)

Usage:
    from shared.collections import load_collection

    collection = load_collection("books-en")
    collection = load_collection("books-scarlet")
"""

import re
import json
import urllib.request
from pathlib import Path

from shared.collections import Collection, register, register_variant, CACHE_DIR


# ─── Book catalogue ─────────────────────────────────────────────────────────

BOOK_CATALOGUE = {
    # English books
    "scarlet": {
        "gutenberg_id": 244,
        "title": "A Study in Scarlet",
        "author": "Arthur Conan Doyle",
        "language": "en",
    },
    "alice": {
        "gutenberg_id": 11,
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "language": "en",
    },
    "moby-dick": {
        "gutenberg_id": 2701,
        "title": "Moby Dick",
        "author": "Herman Melville",
        "language": "en",
    },
    "pride": {
        "gutenberg_id": 1342,
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "language": "en",
    },
    "frankenstein": {
        "gutenberg_id": 84,
        "title": "Frankenstein",
        "author": "Mary Shelley",
        "language": "en",
    },
    # German
    "buddenbrooks": {
        "gutenberg_id": 34811,
        "title": "Buddenbrooks",
        "author": "Thomas Mann",
        "language": "de",
    },
    # French
    "mousquetaires": {
        "gutenberg_id": 13951,
        "title": "Les Trois Mousquetaires",
        "author": "Alexandre Dumas",
        "language": "fr",
    },
    # Spanish
    "don-quixote": {
        "gutenberg_id": 2000,
        "title": "Don Quijote",
        "author": "Miguel de Cervantes",
        "language": "es",
    },
    # Polish
    "bajki": {
        "gutenberg_id": 27729,
        "title": "Bajki",
        "author": "Adam Mickiewicz",
        "language": "pl",
    },
    # Chinese
    "sutra-42": {
        "gutenberg_id": 23585,
        "title": "佛說四十二章經",
        "author": "Unknown",
        "language": "zh",
    },
}


# ─── Download and caching ───────────────────────────────────────────────────

BOOKS_CACHE_DIR = CACHE_DIR / "books"


def _download_book_text(gutenberg_id: int) -> str:
    """Download a book's plain text from Project Gutenberg."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
    ]

    for url in urls:
        try:
            with urllib.request.urlopen(url) as resp:
                raw = resp.read().decode("utf-8-sig")
            # Strip Gutenberg header/footer
            start_markers = ["*** START OF THE PROJECT", "*** START OF THIS PROJECT"]
            end_markers = ["*** END OF THE PROJECT", "*** END OF THIS PROJECT"]

            start_idx = 0
            for marker in start_markers:
                idx = raw.find(marker)
                if idx != -1:
                    start_idx = raw.index("\n", idx) + 1
                    break

            end_idx = len(raw)
            for marker in end_markers:
                idx = raw.find(marker)
                if idx != -1:
                    end_idx = idx
                    break

            body = raw[start_idx:end_idx].strip()
            if len(body) > 100:
                return body
        except Exception:
            continue

    raise RuntimeError(f"Could not download Gutenberg book {gutenberg_id}")


def _get_book_text(slug: str) -> str:
    """Get book text from cache or download."""
    info = BOOK_CATALOGUE[slug]
    cache_path = BOOKS_CACHE_DIR / f"{slug}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    print(f"  Downloading: {info['title']} (Gutenberg #{info['gutenberg_id']})")
    text = _download_book_text(info["gutenberg_id"])
    cache_path.write_text(text, encoding="utf-8")
    return text


# ─── Chapter splitting ──────────────────────────────────────────────────────

# Patterns for chapter markers across languages
CHAPTER_PATTERNS = [
    # English
    r"^CHAPTER\s+[IVXLCDM\d]+\.?",
    r"^Chapter\s+[IVXLCDM\d]+\.?",
    # German: "Erstes Kapitel", "Zweiter Teil", etc. — match any word ending in
    # -es/-er/-te/-tes followed by Kapitel/Teil/Abschnitt
    r"^\w+\s+Kapitel\s*$",
    r"^\w+\s+Teil\s*$",
    # French
    r"^CHAPITRE\s+[IVXLCDM\d]+",
    r"^Chapitre\s+[IVXLCDM\d]+",
    # Spanish
    r"^CAPÍTULO\s+[IVXLCDM\d]+",
    r"^Capítulo\s+[IVXLCDM\d]+",
]

CHAPTER_RE = re.compile("|".join(f"({p})" for p in CHAPTER_PATTERNS), re.MULTILINE)


def _split_chapters(text: str, min_length: int = 200) -> list[str]:
    """
    Split book text into chapters using chapter heading markers.

    Falls back to splitting on multiple blank lines if no chapter markers found.
    Returns list of chapter texts (cleaned).
    """
    lines = text.split("\n")

    # Find chapter boundary lines
    chapter_starts = []
    for i, line in enumerate(lines):
        if CHAPTER_RE.match(line.strip()):
            chapter_starts.append(i)

    # If we found chapters, split on them
    if len(chapter_starts) >= 3:
        chapters = []
        for idx, start in enumerate(chapter_starts):
            end = chapter_starts[idx + 1] if idx + 1 < len(chapter_starts) else len(lines)
            chapter_text = "\n".join(lines[start:end]).strip()
            chapter_text = re.sub(r"\n{3,}", "\n\n", chapter_text)
            if len(chapter_text) >= min_length:
                chapters.append(chapter_text)
        return chapters

    # Fallback: split on 4+ blank lines
    chunks = re.split(r"\n{4,}", text)
    chapters = []
    for chunk in chunks:
        chunk = chunk.strip()
        chunk = re.sub(r"\n{3,}", "\n\n", chunk)
        if len(chunk) >= min_length:
            chapters.append(chunk)

    # If still too few chunks, split on 3+ blank lines
    if len(chapters) < 3:
        chunks = re.split(r"\n{3,}", text)
        chapters = [c.strip() for c in chunks if len(c.strip()) >= min_length]

    return chapters


def _clean_chapter_text(text: str) -> str:
    """Clean chapter text for indexing: collapse whitespace, strip artifacts."""
    # Collapse multiple spaces/newlines into single space for searchable text
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Collection class ───────────────────────────────────────────────────────

@register("books", description="Gutenberg books split by chapter (~10 books, multilingual)")
class BooksCollection(Collection):
    """
    Gutenberg books as chapter-level retrieval units.

    Args:
        slugs: List of book slugs to include. Default: all books.
        languages: Filter by language codes (e.g., ["en", "de"]).
    """

    description = "Gutenberg books split by chapter (~10 books, multilingual)"
    language = "mixed"

    def __init__(self, slugs: list[str] | None = None, languages: list[str] | None = None):
        self._slugs = slugs
        self._languages = languages
        super().__init__()

    def _load(self):
        slugs = self._slugs or list(BOOK_CATALOGUE.keys())

        # Filter by language if specified
        if self._languages:
            slugs = [s for s in slugs if BOOK_CATALOGUE[s]["language"] in self._languages]

        for slug in slugs:
            info = BOOK_CATALOGUE[slug]
            text = _get_book_text(slug)
            chapters = _split_chapters(text)

            for i, chapter_text in enumerate(chapters):
                chapter_num = i + 1
                doc_id = f"{slug}-ch{chapter_num:02d}"
                self._docs[doc_id] = {
                    "id": doc_id,
                    "source": slug,
                    "text": _clean_chapter_text(chapter_text),
                    "title": info["title"],
                    "author": info["author"],
                    "language": info["language"],
                    "chapter": chapter_num,
                }


# ─── Register sub-collections ───────────────────────────────────────────────

# Language sub-collections
_LANGUAGE_MAP = {
    "en": "English books",
    "de": "German books",
    "fr": "French books",
    "es": "Spanish books",
    "pl": "Polish books",
    "zh": "Chinese texts",
}

for _lang, _desc in _LANGUAGE_MAP.items():
    def _make_lang_factory(lang=_lang, desc=_desc):
        def factory():
            c = BooksCollection(languages=[lang])
            c.name = f"books-{lang}"
            c.description = desc
            c.language = lang
            return c
        return factory

    register_variant(
        name=f"books-{_lang}",
        description=_desc,
        factory=_make_lang_factory(_lang, _desc),
    )

# Individual English book sub-collections
for _slug, _info in BOOK_CATALOGUE.items():
    if _info["language"] == "en":
        def _make_book_factory(slug=_slug, info=_info):
            def factory():
                c = BooksCollection(slugs=[slug])
                c.name = f"books-{slug}"
                c.description = f"{info['title']} ({info['author']})"
                c.language = info["language"]
                return c
            return factory

        register_variant(
            name=f"books-{_slug}",
            description=f"{_info['title']} ({_info['author']})",
            factory=_make_book_factory(_slug, _info),
        )
