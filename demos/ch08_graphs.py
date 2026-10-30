"""Small directed graphs of Chapter 8 (link analysis) and a matplotlib drawing helper."""

import matplotlib.pyplot as plt
import numpy as np

# Seven-page web of the book (figures 8_10 and 8_11): page -> pages it links to.
PAGERANK_GRAPH = {
    "A": ["B", "E", "G"],
    "B": ["E"],
    "C": ["A", "B", "E"],
    "D": ["B", "C", "E", "G"],
    "E": ["C", "G"],
    "F": ["B", "D", "E", "G"],
    "G": ["E"],
}

# Eight-page HITS base set of the book (figure 8_14).
HITS_GRAPH = {
    "A": ["B", "C", "D", "E", "G"],
    "B": ["D", "G"],
    "C": [],
    "D": ["C"],
    "E": ["D"],
    "F": ["D", "E", "G"],
    "G": ["C", "D"],
    "H": ["D", "F", "G"],
}

# Node positions that follow the layout of the book figures.
PAGERANK_POS = {"A": (0.8, 2.85), "B": (2.6, 4.1), "C": (2.5, 1.2), "D": (4.8, 0.9),
                "E": (4.6, 3.1), "F": (6.4, 4.05), "G": (6.6, 1.8)}
HITS_POS = {"A": (0.8, 3.4), "B": (1.0, 2.0), "C": (2.65, 1.2), "D": (2.7, 3.1),
            "E": (2.05, 4.25), "F": (3.6, 4.0), "G": (4.2, 1.95), "H": (5.8, 3.3)}


def draw_graph(links, pos, sizes=None, labels=None, title=None, ax=None):
    """Draw a directed graph; node area follows `sizes` (dict), `labels` go inside the nodes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.2))
    if sizes:
        top = max(sizes.values())
        area = {p: 1100 + 2900 * sizes[p] / top for p in links}  # marker area in points^2
    else:
        area = {p: 1100 for p in links}
    radius = {p: np.sqrt(area[p]) / 2 for p in links}           # circle radius in points

    for src, targets in links.items():
        for dst in targets:
            ax.annotate("", xy=pos[dst], xytext=pos[src],
                        arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1, mutation_scale=12,
                                        connectionstyle="arc3,rad=0.08",
                                        shrinkA=radius[src], shrinkB=radius[dst] + 1))
    for p, (x, y) in pos.items():
        ax.scatter(x, y, s=area[p], marker="o", color="#dce8f8", edgecolor="#2a78d6",
                   linewidth=1.5, zorder=3)
        ax.annotate(p, (x, y), xytext=(0, 6 if labels else 0), textcoords="offset points",
                    ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
        if labels:
            ax.annotate(labels[p], (x, y), xytext=(0, -8), textcoords="offset points",
                        ha="center", va="center", fontsize=9, color="#52514e", zorder=4)
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax
