"""
Document collections for retrieval demos.

A collection is a set of retrieval units — documents ready for indexing and
search. Each retrieval unit is a flat dict with at least:

    {"id": str, "source": str, "text": str, ...optional metadata...}

Collections handle data loading, caching, and splitting automatically.

Usage:
    from shared.collections import load_collection, available_collections

    # See what's available
    available_collections()  # → [{"name": ..., "description": ...}, ...]

    # Load by name — no kwargs needed
    collection = load_collection("mini")
    collection = load_collection("slides-classical-text-retrieval")
    collection = load_collection("slides")  # all slides combined

    # Iterate over retrieval units
    for doc in collection:
        print(doc["id"], doc["text"][:50])

    # Access by ID
    doc = collection["b1"]

    # Basic info
    print(collection.name)
    print(len(collection))
"""

from pathlib import Path
from typing import Iterator, Callable

# Cache root for downloaded data
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / ".cache"


class Collection:
    """
    Base class for document collections.

    Subclasses implement `_load()` which populates `self._docs` — an ordered
    dict of {id: flat_dict} where each dict has at least id, source, text.
    """

    name: str = ""
    description: str = ""
    language: str = ""  # ISO 639-1 code ("en", "de", ...) or "mixed" for multilingual

    def __init__(self):
        self._docs: dict[str, dict] = {}
        self._load()

    def _load(self):
        """Populate self._docs. Subclasses must override."""
        raise NotImplementedError

    # ─── Sequence interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._docs.values())

    def __getitem__(self, doc_id: str) -> dict:
        return self._docs[doc_id]

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    # ─── Convenience accessors ──────────────────────────────────────────

    def ids(self) -> list[str]:
        """All document IDs in collection order."""
        return list(self._docs.keys())

    def texts(self) -> dict[str, str]:
        """Mapping {id: text} for all documents."""
        return {doc_id: doc["text"] for doc_id, doc in self._docs.items()}

    def documents(self) -> list[dict]:
        """All retrieval units as a list."""
        return list(self._docs.values())

    def metadata_keys(self) -> list[str]:
        """All metadata keys present (beyond id, source, text)."""
        reserved = {"id", "source", "text"}
        keys = set()
        for doc in self._docs.values():
            keys.update(doc.keys() - reserved)
        return sorted(keys)

    def __repr__(self) -> str:
        return (
            f"Collection('{self.name}', {len(self)} docs"
            f"{', ' + self.description if self.description else ''})"
        )


# ─── Registry ───────────────────────────────────────────────────────────────

# Each entry: {"description": str, "factory": callable that returns a Collection}
_REGISTRY: dict[str, dict] = {}


def register(name: str, description: str = ""):
    """
    Decorator to register a collection class under a name.

    The class can also call `register_variant()` to add sub-collections
    during module load.
    """
    def decorator(cls):
        desc = description or cls.description
        _REGISTRY[name] = {"description": desc, "factory": cls}
        cls.name = name
        return cls
    return decorator


def register_variant(name: str, description: str, factory: Callable[[], Collection]):
    """
    Register a sub-collection (variant) as a first-class entry.

    Args:
        name: Unique name for this variant (e.g., "slides-classical-text-retrieval").
        description: Human-readable description.
        factory: Zero-argument callable that returns a Collection instance.
    """
    _REGISTRY[name] = {"description": description, "factory": factory}


def available_collections() -> list[dict]:
    """
    List all registered collections (including sub-collections).

    Returns:
        List of dicts: {"name": str, "description": str}
    """
    _ensure_registered()
    return [
        {"name": name, "description": entry["description"]}
        for name, entry in sorted(_REGISTRY.items())
    ]


def load_collection(name: str) -> Collection:
    """
    Load a collection by name.

    Args:
        name: Registered collection name (e.g., "mini", "slides",
              "slides-classical-text-retrieval").

    Returns:
        A Collection instance ready to iterate.
    """
    _ensure_registered()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown collection '{name}'. Available: {available}")
    return _REGISTRY[name]["factory"]()


_IMPORTS_DONE = False


def _ensure_registered():
    """Import all collection modules to trigger @register decorators."""
    global _IMPORTS_DONE
    if _IMPORTS_DONE:
        return
    _IMPORTS_DONE = True
    from shared.collections import mini, slides, movies, books  # noqa: F401
