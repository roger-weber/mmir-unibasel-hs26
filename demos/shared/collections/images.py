"""
Image collections: freely licensed photos from Wikimedia Commons, downloaded on first use.

A manifest under `data/images/` lists each image with its group, content flags, a one-line
description, content tags, the Commons file page, the thumbnail URL, the pixel size, and the
author and license. The first load downloads the thumbnails into `data/.cache/images/<name>/`
(not committed); later loads read the cache.

Collections:
    - "ch10" → 43 photos for the chapter 10 "then vs. now" comparison
               (`data/images/ch10_collection.json`, 1280 px wide, about 16 MB):
               matterhorn (8), mountain (12), color_decoy (9), other (14).
               Groups: the Matterhorn from several sides, seasons, and light; other
               snow-covered peaks; scenes without mountains but with the query's palette
               (pale blue sky, white clouds, dark green); unrelated categories and palettes.

Not registered with `load_collection()`: that registry serves text retrieval units, and the
ch01 demos list every registered collection for text feature extraction.

Usage:
    from shared.collections.images import load_image_collection

    images = load_image_collection()                # list of dicts, manifest order
    images[0]["path"]                               # Path to the cached JPEG
    images[0]["group"], images[0]["tags"]           # "matterhorn", ["matterhorn", "snow", ...]
    [im["id"] for im in images if im["is_mountain"]]
"""

import json
import time
from pathlib import Path

from shared.collections import CACHE_DIR


USER_AGENT = "mmir-unibasel-course/1.0 (https://github.com/roger-weber/mmir-unibasel-hs26)"
MANIFEST_DIR = Path(__file__).parent.parent.parent / "data" / "images"
IMAGE_CACHE_DIR = CACHE_DIR / "images"
MANIFESTS = {"ch10": MANIFEST_DIR / "ch10_collection.json"}


def _download(url: str, target: Path, retries: int = 4) -> None:
    """Fetch one file to `target`; write to .part first so an interrupted download is not kept."""
    import requests

    for attempt in range(retries):
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        if r.status_code == 429 and attempt < retries - 1:          # rate limited: back off
            time.sleep(float(r.headers.get("Retry-After", 5 * (attempt + 1))))
            continue
        r.raise_for_status()
        if not r.headers.get("Content-Type", "").startswith("image/"):
            raise ValueError(f"{url} returned {r.headers.get('Content-Type')}, not an image")
        part = target.with_suffix(target.suffix + ".part")
        part.write_bytes(r.content)
        part.replace(target)
        return


def load_image_collection(name: str = "ch10", cache_dir: Path | None = None) -> list[dict]:
    """
    Load an image collection, downloading missing files on first use.

    Args:
        name: Collection name, a key of MANIFESTS ("ch10").
        cache_dir: Folder for the image files; defaults to data/.cache/images/<name>/.

    Returns:
        The manifest entries in order, each with an added "path" (Path to the JPEG):
        {"id", "group", "is_matterhorn", "is_mountain", "description", "tags",
         "file_page", "download_url", "width", "height", "author", "license",
         "license_url", "path"}
    """
    if name not in MANIFESTS:
        raise ValueError(f"Unknown image collection '{name}'. Available: {', '.join(MANIFESTS)}")
    entries = json.loads(MANIFESTS[name].read_text(encoding="utf-8"))
    folder = Path(cache_dir) if cache_dir else IMAGE_CACHE_DIR / name
    for e in entries:
        e["path"] = folder / f"{e['id']}.jpg"
    missing = [e for e in entries if not e["path"].exists()]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(missing)} images from Wikimedia Commons to {folder} ...")
        for e in missing:
            _download(e["download_url"], e["path"])
    return entries
