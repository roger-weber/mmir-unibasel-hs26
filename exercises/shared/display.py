"""Display utilities for notebook output formatting."""

from itertools import islice
from IPython.display import display, Markdown
from tabulate import tabulate


def print_table(
    rows: list[list],
    headers: list[str],
    *,
    max_rows: int = 100,
    fmt: str = "pipe",
    floatfmt: str | tuple[str, ...] = "g",
) -> None:
    """
    Render a table as Markdown in notebook output.

    Args:
        rows: List of row lists.
        headers: Column header strings.
        max_rows: Maximum rows to display.
        fmt: Table format ('pipe' for Markdown, 'github' for plain text).
        floatfmt: Number format for float cells, one string or one per column
            (e.g. ".2f" or ("", ".3f", ".2f")). Numeric strings are parsed as
            numbers too, so pass floats and set floatfmt to keep trailing zeros.
    """
    if not rows or not headers:
        return
    if fmt == "text":
        print(tabulate(islice(rows, max_rows), headers, tablefmt="github", floatfmt=floatfmt))
    else:
        display(Markdown(tabulate(islice(rows, max_rows), headers, tablefmt=fmt, floatfmt=floatfmt)))


def display_md(text: str) -> None:
    """Display text as rendered Markdown."""
    display(Markdown(text))


def print_wrapped(text: str, width: int = 80) -> None:
    """Print text wrapped to the given width."""
    import textwrap
    for line in textwrap.wrap(text, width):
        print(line)
