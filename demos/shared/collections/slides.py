"""
Lecture slides collection — PDF pages as retrieval units.

Downloads lecture PDFs (or uses local copies) and splits them into one
retrieval unit per page. Text is extracted and cleaned automatically.

Registers:
    - "slides"                          → all lectures combined (~500+ pages)
    - "slides-classical-text-retrieval" → single lecture
    - "slides-semantic-search"          → single lecture
    - ... (one per lecture in the catalogue)

Usage:
    from shared.collections import load_collection

    collection = load_collection("slides")
    collection = load_collection("slides-classical-text-retrieval")
"""

import re
import urllib.request
from pathlib import Path
from functools import partial

from shared.collections import Collection, register, register_variant, CACHE_DIR


# ─── Slide catalogue ────────────────────────────────────────────────────────
# HS25 lecture PDFs from https://dmi.unibas.ch/.../lecture-multimedia-retrieval/

BASE_URL = "https://dmi.unibas.ch/fileadmin/user_upload/dmi/Studium/Computer_Science/Vorlesungen_HS23/Multimedia_Retrieval/HS25"

SLIDE_CATALOGUE = {
    "introduction": {
        "url": f"{BASE_URL}/01_Introduction.pdf",
        "title": "Introduction",
        "chapter": 0,
    },
    "classical-text-retrieval": {
        "url": f"{BASE_URL}/02_ClassicalTextRetrieval.pdf",
        "title": "Classical Text Retrieval",
        "chapter": 1,
    },
    "performance-evaluation": {
        "url": f"{BASE_URL}/03_PerformanceEvaluation.pdf",
        "title": "Performance Evaluation",
        "chapter": 2,
    },
    "advanced-text-processing": {
        "url": f"{BASE_URL}/04_AdvancedTextProcessing.pdf",
        "title": "Advanced Text Processing",
        "chapter": 3,
    },
    "index-text-retrieval": {
        "url": f"{BASE_URL}/05_IndexForTextRetrieval.pdf",
        "title": "Index for Text Retrieval",
        "chapter": 4,
    },
    "semantic-search": {
        "url": f"{BASE_URL}/06_SemanticSearch.pdf",
        "title": "Semantic Search",
        "chapter": 5,
    },
    "vector-search": {
        "url": f"{BASE_URL}/07_VectorSearch.pdf",
        "title": "Vector Search",
        "chapter": 6,
    },
    "rag": {
        "url": f"{BASE_URL}/08_RetrievalAugmentedGeneration.pdf",
        "title": "Retrieval Augmented Generation",
        "chapter": 7,
    },
    "web-search": {
        "url": f"{BASE_URL}/09_WebSearch.pdf",
        "title": "Web Search",
        "chapter": 8,
    },
    "multimodal-content-analysis": {
        "url": f"{BASE_URL}/10_MultimodalContentAnalysis.pdf",
        "title": "Multimodal Content Analysis",
        "chapter": 9,
    },
    "visual-features": {
        "url": f"{BASE_URL}/11_VisualFeatures.pdf",
        "title": "Visual Features",
        "chapter": 10,
    },
    "acoustic-features": {
        "url": f"{BASE_URL}/12_AcousticFeatures.pdf",
        "title": "Acoustic Features",
        "chapter": 11,
    },
    "spatiotemporal-features": {
        "url": f"{BASE_URL}/13_SpatiotemporalFeatures.pdf",
        "title": "Spatiotemporal Features",
        "chapter": 12,
    },
    "structural-features": {
        "url": f"{BASE_URL}/14_StructuralFeatures.pdf",
        "title": "Structural Features",
        "chapter": 13,
    },
    "ml-methods": {
        "url": f"{BASE_URL}/99_MLMethods.pdf",
        "title": "ML Methods",
        "chapter": 99,
    },
}


def _extract_pages(pdf_path: str | Path) -> list[str]:
    """Extract cleaned text per page from a PDF file."""
    from PyPDF2 import PdfReader

    pages = []

    def visitor_text(text, cm, tm, fontDict, fontSize):
        y = tm[5]
        if y > 20 and text:
            text = text.replace("\n", " ")
            text = re.sub(r"\[\d+\]|➢|•", "", text)
            parts.append(text)

    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        parts = []
        page.extract_text(visitor_text=visitor_text)
        page_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        pages.append(page_text)

    return pages


def _download_if_needed(url: str, local_path: Path) -> Path:
    """Download a file if not already cached."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, str(local_path))
    return local_path


@register("slides", description="All lecture slides combined (~500+ pages)")
class SlidesCollection(Collection):
    """
    Lecture slides split into page-level retrieval units.

    Each page becomes one document with cleaned text ready to tokenize.
    Pages with insufficient text (< min_length chars) are skipped.

    Args:
        pdf: Which slides to load. Options:
             - "all" (default): all lecture PDFs combined
             - A catalogue name (e.g., "classical-text-retrieval")
             - A URL to a PDF
             - A local file path
        min_length: Minimum text length to include a page (default 20).
    """

    description = "All lecture slides combined (~500+ pages)"
    language = "en"

    def __init__(self, pdf: str = "all", min_length: int = 20):
        self._pdf = pdf
        self._min_length = min_length
        super().__init__()

    def _load(self):
        if self._pdf == "all":
            self._load_all()
        else:
            self._load_single(self._pdf)

    def _load_all(self):
        """Load all lecture PDFs as one combined collection."""
        print(f"Loading {len(SLIDE_CATALOGUE)} lecture slide sets...")
        for name, info in SLIDE_CATALOGUE.items():
            pdf_path = self._resolve_named(name)
            self._add_pages(pdf_path, source_name=name, chapter=info["chapter"])
        print(f"  Total: {len(self._docs)} pages from {len(SLIDE_CATALOGUE)} lectures")

    def _load_single(self, pdf: str):
        """Load a single PDF (by catalogue name, URL, or path)."""
        if pdf in SLIDE_CATALOGUE:
            pdf_path = self._resolve_named(pdf)
            chapter = SLIDE_CATALOGUE[pdf]["chapter"]
            self._add_pages(pdf_path, source_name=pdf, chapter=chapter)
        elif pdf.startswith("http://") or pdf.startswith("https://"):
            filename = Path(pdf).name or "slides.pdf"
            cache_path = CACHE_DIR / "slides" / filename
            pdf_path = _download_if_needed(pdf, cache_path)
            self._add_pages(pdf_path, source_name=filename)
        else:
            path = Path(pdf)
            if path.exists():
                self._add_pages(path, source_name=path.name)
            else:
                available = ", ".join(sorted(SLIDE_CATALOGUE.keys()))
                raise FileNotFoundError(
                    f"Cannot resolve PDF: '{pdf}'. "
                    f"Known: {available}. Or provide a URL or local path."
                )

    def _add_pages(self, pdf_path: Path, source_name: str, chapter: int | None = None):
        """Extract pages and add to self._docs."""
        pages = _extract_pages(pdf_path)
        for i, text in enumerate(pages):
            if len(text) < self._min_length:
                continue
            page_num = i + 1
            doc_id = f"{source_name}-p{page_num:02d}"
            entry = {
                "id": doc_id,
                "source": source_name,
                "text": text,
                "page": page_num,
            }
            if chapter is not None:
                entry["chapter"] = chapter
            self._docs[doc_id] = entry

    def _resolve_named(self, name: str) -> Path:
        """Download a named slide set if not cached."""
        info = SLIDE_CATALOGUE[name]
        filename = name + ".pdf"
        cache_path = CACHE_DIR / "slides" / filename
        return _download_if_needed(info["url"], cache_path)


# ─── Register sub-collections for each lecture ──────────────────────────────

def _make_single_factory(catalogue_key: str):
    """Create a factory function for a single slide set."""
    def factory():
        c = SlidesCollection(pdf=catalogue_key)
        c.name = f"slides-{catalogue_key}"
        c.description = f"Lecture: {SLIDE_CATALOGUE[catalogue_key]['title']}"
        return c
    return factory


for _key, _info in SLIDE_CATALOGUE.items():
    register_variant(
        name=f"slides-{_key}",
        description=f"Lecture: {_info['title']}",
        factory=_make_single_factory(_key),
    )
