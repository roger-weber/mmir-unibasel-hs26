"""Anonymous usage tracker: reports which demo or exercise notebook was run, once per kernel session.

Rows look like `hs26 | demo | ch05-02-word-embeddings.ipynb` or `hs26 | exercise | ch01-exercise`
(exercise notebooks share their file name across chapters, so the chapter folder is prefixed).
Only the file name is sent, never the full path. Author runs in the content repo are skipped.
"""
import os
import re
import threading
import urllib.parse
import urllib.request
from pathlib import Path

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfHwbtS2DLnG_f-_D4VE4i3VmSoXme7ivQiiiOyB8aaeJl9-w/formResponse"
FIELD_IDS = {
    "course": "entry.744480106",
    "kind": "entry.55505068",
    "notebook": "entry.661379773",
}
KINDS = {"demos": "demo", "exercises": "exercise"}   # folder holding shared/ -> kind


def notebook_path() -> Path | None:
    """Path of the running notebook (VS Code or Jupyter), or None."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        path = ip.user_ns.get("__vsc_ipynb_file__") if ip else None
        path = path or os.environ.get("JPY_SESSION_NAME")
        return Path(path) if path else None
    except Exception:
        return None


def notebook_name(kind: str, path: Path | None) -> str:
    """Demo file name, or `chNN-<stem>` for exercises; 'unknown' if the path is not available."""
    if path is None:
        return "unknown"
    if kind == "exercise":
        chapter = next((p.name for p in path.parents if re.fullmatch(r"ch\d\d", p.name)), "ch??")
        return f"{chapter}-{path.stem}"
    return path.name


def _in_content_repo(path: Path) -> bool:
    return any((p / "tools" / "build_exercise.py").is_file() for p in path.parents)


def _send(data: dict) -> None:
    try:
        req = urllib.request.Request(FORM_URL, data=urllib.parse.urlencode(data).encode())
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass   # never disturb the notebook


def track_run() -> None:
    """Submit one 'hs26 / <kind> / <notebook>' event in the background; silent on any failure."""
    here = Path(__file__).resolve()
    kind = KINDS.get(here.parents[1].name)
    if kind is None or _in_content_repo(here):
        return   # unknown layout, or an author run (content repo, local build, hs26 submodule)
    data = {
        FIELD_IDS["course"]: "hs26",
        FIELD_IDS["kind"]: kind,
        FIELD_IDS["notebook"]: notebook_name(kind, notebook_path()),
    }
    threading.Thread(target=_send, args=(data,), daemon=True).start()
