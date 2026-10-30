"""
BEIR collections — benchmark corpora with queries and relevance judgments.

Downloads a dataset from the BEIR benchmark (Thakur et al., 2021) on first use and
caches it under `data/.cache/beir/<name>/`. Each corpus document becomes one
retrieval unit; the dataset's queries and relevance judgments are available
through `shared.assessments.load_assessments(<name>)`.

Registers:
    - "trec-covid" → 171,332 CORD-19 abstracts; 50 TREC topics with deep,
                     graded judgments (0 = not, 1 = partially, 2 = fully relevant)

The zip is about 74 MB; the first load downloads and unpacks it.
"""

import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

from shared.collections import Collection, register, CACHE_DIR


BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
BEIR_DIR = CACHE_DIR / "beir"


def beir_dataset_dir(name: str) -> Path:
    """Return the unpacked dataset folder, downloading it on first use."""
    target = BEIR_DIR / name
    if (target / "corpus.jsonl").exists():
        return target
    BEIR_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BEIR_DIR / f"{name}.zip"
    if not zip_path.exists():
        # certifi's CA bundle: some Windows certificate stores lack the server's root CA.
        # Download to .part first, so an interrupted download is not mistaken for the zip.
        context = ssl.create_default_context(cafile=certifi.where())
        part = zip_path.with_suffix(".zip.part")
        with urllib.request.urlopen(BEIR_URL.format(name=name), context=context) as r,                 open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.replace(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(BEIR_DIR)
    return target


class BeirCollection(Collection):
    """A BEIR corpus as retrieval units: {"id", "source", "title", "text"}.

    `text` joins title and abstract, the input that dense retrievers and
    rerankers see. The plain abstract is kept as `abstract`.
    """

    dataset: str = ""

    def _load(self):
        folder = beir_dataset_dir(self.dataset)
        with open(folder / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                title = d.get("title", "").strip()
                abstract = d.get("text", "").strip()
                self._docs[d["_id"]] = {
                    "id": d["_id"],
                    "source": self.dataset,
                    "title": title,
                    "abstract": abstract,
                    "text": f"{title}. {abstract}" if title and abstract else (title or abstract),
                }


@register("trec-covid", "TREC-COVID (BEIR): 171k CORD-19 abstracts, 50 topics, graded judgments")
class TrecCovid(BeirCollection):
    dataset = "trec-covid"
    language = "en"
