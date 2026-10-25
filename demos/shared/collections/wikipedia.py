"""
Wikipedia collections: fixed revisions of a set of articles, one retrieval unit per section.

Each article is pinned to a revision id, so every student downloads exactly the same
text and the notebooks reproduce the same numbers. The first load fetches the pinned
revisions from the Wikipedia API (one request per article) and caches the parsed
sections under `data/.cache/wikipedia/<name>.json`.

The HTML of each revision is reduced to text by section: headings start a new unit,
paragraphs and list items become lines, and table rows become `cell | cell | cell`
lines (prize tables carry the facts). Infoboxes, references, navigation boxes, and
reference-like sections ("References", "See also", ...) are dropped.

Registers:
    - "nobel-2025" → 38 articles around the 2025 Nobel Prizes: the physics laureates,
                     their research and universities, the other 2025 prizes, the 2026
                     prizes, the 2024 physics laureates, and general Nobel articles.

Retrieval units:
    {"id": "Michel Devoret#Career", "source": "wikipedia", "title": "Michel Devoret",
     "section": "Career", "url": ..., "revid": 1380998550, "text": ...}
"""

import json
import re

from shared.collections import Collection, register, CACHE_DIR


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "mmir-unibasel-course/1.0 (https://github.com/roger-weber/mmir-unibasel-hs26)"
WIKI_DIR = CACHE_DIR / "wikipedia"

SKIP_SECTIONS = {
    "References", "External links", "See also", "Notes", "Further reading",
    "Bibliography", "Sources", "Citations", "Works cited", "Footnotes",
    "Selected publications", "Publications", "Notes and references",
}
DROP_SELECTORS = (
    "sup.reference, .mw-editsection, style, script, .navbox, .reflist, .infobox, "
    ".metadata, .hatnote, .thumb, figure, .sidebar, .mw-empty-elt, .shortdescription, "
    ".noprint, .mw-references-wrap"
)

# Pinned revisions (fetched 2026-10-22).
NOBEL_2025 = {
    # 2025 physics prize, laureates, research, universities
    "2025 Nobel Prizes": 1379966130,
    "Nobel Prize in Physics": 1380149086,
    "List of Nobel laureates in Physics": 1381322896,
    "John Clarke (physicist)": 1380998955,
    "Michel Devoret": 1380998550,
    "John M. Martinis": 1380998727,
    "Josephson effect": 1378857258,
    "Quantum tunnelling": 1380327394,
    "Superconducting quantum computing": 1380480077,
    "Transmon": 1377984572,
    "Google Quantum AI": 1379553066,
    "University of California, Berkeley": 1380523751,
    "Yale University": 1381304127,
    "University of California, Santa Barbara": 1373610073,
    # the other 2025 prizes
    "Omar M. Yaghi": 1379674073,
    "Susumu Kitagawa": 1351199256,
    "Richard Robson (chemist)": 1374829291,
    "Metal–organic framework": 1380735518,
    "Mary E. Brunkow": 1379081949,
    "Fred Ramsdell": 1380701571,
    "Shimon Sakaguchi": 1372290666,
    "Regulatory T cell": 1371253989,
    "2025 Nobel Prize in Literature": 1376512909,
    "László Krasznahorkai": 1376279570,
    "2025 Nobel Peace Prize": 1379415461,
    "María Corina Machado": 1379335470,
    "Joel Mokyr": 1379629575,
    "Philippe Aghion": 1381419766,
    "Peter Howitt (economist)": 1380324939,
    # other years: same prize, wrong answer
    "2026 Nobel Prizes": 1380470315,
    "Francis Halzen": 1381473977,
    "IceCube Neutrino Observatory": 1379318357,
    "John Hopfield": 1379322601,
    "Geoffrey Hinton": 1380282819,
    # the Nobel Prize in general
    "Nobel Prize": 1380674954,
    "Alfred Nobel": 1379875247,
    "Royal Swedish Academy of Sciences": 1369334966,
    "Nobel Prize medal": 1380338238,
}


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+([,.;:!?)\]])", r"\1", s)   # "Foundation ," -> "Foundation,"


def parse_sections(html: str) -> list[dict]:
    """Split the rendered HTML of one article into [{"section", "text"}]."""
    from bs4 import BeautifulSoup

    root = BeautifulSoup(html, "html.parser").select_one(".mw-parser-output")
    for el in root.select(DROP_SELECTORS):
        el.decompose()
    sections, heading, parts = [], "Introduction", []

    def flush():
        text = "\n".join(p for p in parts if p)
        if text and heading not in SKIP_SECTIONS:
            sections.append({"section": heading, "text": text})

    for el in root.find_all(["h2", "h3", "p", "ul", "ol", "table"]):
        if el.find_parent(["table", "ul", "ol"]):
            continue                                   # nested: handled by its container
        if el.name in ("h2", "h3"):
            flush()
            heading, parts = _clean(el.get_text(" ")), []
        elif el.name == "table":
            rows = [" | ".join(_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"]))
                    for tr in el.find_all("tr")]
            parts.append("\n".join(r for r in rows if r.strip(" |")))
        elif el.name in ("ul", "ol"):
            parts.append("\n".join("- " + _clean(li.get_text(" "))
                                   for li in el.find_all("li", recursive=False)))
        else:
            parts.append(_clean(el.get_text(" ")))
    flush()
    return sections


def fetch_revision(revid: int) -> str:
    """Rendered HTML of one pinned revision."""
    import requests

    r = requests.get(API_URL, params={
        "action": "parse", "oldid": revid, "prop": "text", "format": "json",
        "formatversion": 2, "disableeditsection": 1,
    }, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.json()["parse"]["text"]


def load_articles(name: str, revisions: dict[str, int]) -> list[dict]:
    """Sections of all articles, downloaded on first use and cached."""
    path = WIKI_DIR / f"{name}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("revisions") == revisions:
            return cached["units"]
    units, seen = [], set()
    for title, revid in revisions.items():
        for s in parse_sections(fetch_revision(revid)):
            uid, n = f"{title}#{s['section']}", 2
            while uid in seen:                         # repeated heading in one article
                uid, n = f"{title}#{s['section']} ({n})", n + 1
            seen.add(uid)
            units.append({
                "id": uid,
                "source": "wikipedia",
                "title": title,
                "section": s["section"],
                "url": f"https://en.wikipedia.org/w/index.php?oldid={revid}",
                "revid": revid,
                "text": s["text"],
            })
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"revisions": revisions, "units": units}, ensure_ascii=False),
                    encoding="utf-8")
    return units


@register("nobel-2025", "38 Wikipedia articles around the 2025 Nobel Prizes (pinned revisions), one unit per section")
class Nobel2025(Collection):
    language = "en"

    def _load(self):
        for unit in load_articles("nobel-2025", NOBEL_2025):
            self._docs[unit["id"]] = unit
