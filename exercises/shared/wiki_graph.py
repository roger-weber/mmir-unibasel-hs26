"""Link graphs for the Ch08 exercise (PageRank): a Wikipedia graph and synthetic webs.

    from shared.wiki_graph import load_wiki_graph, synthetic_web, rank_of, times_average
    graph = load_wiki_graph()          # {"United States": ["Abraham Lincoln", ...], ...}
    web = synthetic_web(50_000)        # {0: [17, 4, ...], 1: [...], ...}, page 0 most popular

Both return a graph as a dict: page -> list of the pages it links to. Every link target is
also a key, a page appears at most once in a list, and no page links to itself. Each call
returns a fresh graph, so a scenario can change it without affecting the next one.

The Wikipedia graph is the Wikispeedia link graph (West & Leskovec, WWW 2012; West, Pineau &
Precup, IJCAI 2009), https://snap.stanford.edu/data/wikispeedia.html: 4,604 articles of the
2007 Wikipedia for Schools selection and their 119,772 links. The article texts are from
Wikipedia, licensed CC BY-SA. `data/wikispeedia_links.tsv.gz` holds one line per article:
the title, then the titles it links to, tab-separated. Titles are URL-decoded with spaces
for underscores, and the 110 self-links of the original are dropped.
"""
import gzip
from pathlib import Path

import numpy as np

_WIKI_FILE = Path(__file__).parent / "data" / "wikispeedia_links.tsv.gz"


def load_wiki_graph() -> dict[str, list[str]]:
    """The Wikispeedia link graph: article title -> titles it links to."""
    graph = {}
    with gzip.open(_WIKI_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            title, *targets = line.rstrip("\n").split("\t")
            graph[title] = targets
    return graph


def synthetic_web(n: int, links_per_page: int = 20, skew: float = 0.9, seed: int = 0) -> dict[int, list[int]]:
    """A random web of `n` pages with a heavy-tailed in-degree, like the real web.

    Each page gets about `links_per_page` outlinks (Poisson). Each link picks its target with
    a Zipf popularity: page i is chosen with probability proportional to (i + 1) ** -skew, so
    page 0 is the most linked-to page. Duplicate links and self-links are dropped.
    """
    rng = np.random.default_rng(seed)
    popularity = np.arange(1, n + 1, dtype=float) ** -skew
    popularity /= popularity.sum()
    src = np.repeat(np.arange(n), rng.poisson(links_per_page, n))
    dst = rng.choice(n, size=len(src), p=popularity)
    keep = src != dst
    pairs = np.unique(src[keep].astype(np.int64) * n + dst[keep])
    graph = {i: [] for i in range(n)}
    for s, t in zip((pairs // n).tolist(), (pairs % n).tolist()):
        graph[s].append(t)
    return graph


def rank_of(pr: dict, page) -> int:
    """Rank of `page` by PageRank: 1 + the number of pages with a strictly higher score."""
    score = pr[page]
    return 1 + sum(1 for v in pr.values() if v > score)


def times_average(pr: dict, page) -> float:
    """PageRank of `page` as a multiple of the average page (1 / N)."""
    return pr[page] * len(pr)
