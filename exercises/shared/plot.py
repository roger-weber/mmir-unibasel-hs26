"""Visualization utilities for retrieval demos."""


def plot_pr_curve(
    precisions: list[float],
    recalls: list[float],
    *,
    title: str = "Precision-Recall Curve",
    ax=None,
) -> None:
    """
    Plot a precision-recall curve.

    Args:
        precisions: List of precision values.
        recalls: List of recall values.
        title: Plot title.
        ax: Optional matplotlib axes to plot on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, marker="o", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_venn_diagram(
    retrieved: set,
    relevant: set,
    *,
    labels: tuple[str, str] = ("Retrieved", "Relevant"),
) -> None:
    """
    Plot a Venn diagram of retrieved vs relevant documents.

    Args:
        retrieved: Set of retrieved document IDs.
        relevant: Set of relevant document IDs.
        labels: Tuple of (retrieved_label, relevant_label).
    """
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    fig, ax = plt.subplots(figsize=(5, 4))
    venn2([retrieved, relevant], set_labels=labels, ax=ax)
    plt.title("Retrieved vs. Relevant")
    plt.tight_layout()
    plt.show()
