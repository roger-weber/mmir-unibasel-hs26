"""Compile-and-run bridge for the Chapter 4 Lucene exercise.

The exercise notebook stays on the Python kernel. This module lets it drive a small
Java program that uses Apache Lucene: it downloads the pinned Lucene jars from Maven
Central once (cached under the user's home), compiles the project's Java sources with
`javac`, and runs the resulting command-line tool with `java`, returning the program's
JSON output parsed into Python objects.

Only a JDK needs to be installed. Lucene 10 requires Java 21 or newer. Everything else
is downloaded automatically. The module works on Windows, macOS, and Linux; it uses the
platform path separator for the classpath and never assumes Maven is installed.

Typical use from the notebook:

    from shared.lucene_harness import LuceneProject
    proj = LuceneProject("lucene")     # the skeleton folder next to the notebook
    proj.index("movies.tsv")           # (re)build the index from a TSV data file
    hits = proj.query("space adventure", k=10, year_min=2000, fuzzy=False)

`index` and `query` recompile the Java sources automatically whenever a source file has
changed since the last build, so the edit-and-rerun loop needs no kernel restart.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

LUCENE_VERSION = "10.5.1"
MIN_JAVA = 21

# Full compile and runtime closure of lucene-queryparser plus lucene-analysis-common.
# Every artifact below depends only on other Lucene artifacts (verified against the
# 10.5.1 POMs), so there are no third-party jars to resolve.
_ARTIFACTS = [
    "lucene-core",
    "lucene-analysis-common",
    "lucene-queryparser",
    "lucene-queries",
    "lucene-sandbox",
    "lucene-facet",
]
_MAVEN_BASE = "https://repo1.maven.org/maven2/org/apache/lucene"


class JavaNotFound(RuntimeError):
    """Raised when no suitable JDK is on PATH, with an install hint in the message."""


def _jar_cache(version: str) -> Path:
    """Where the downloaded Lucene jars live, shared across every notebook."""
    cache = Path.home() / ".cache" / "mmir-lucene" / version
    cache.mkdir(parents=True, exist_ok=True)
    return cache


class LuceneProject:
    """Compiles and runs a small Java/Lucene project from the Python kernel."""

    def __init__(self, project_dir: str | os.PathLike = "lucene", version: str = LUCENE_VERSION):
        self.project_dir = Path(project_dir).resolve()
        if not self.project_dir.is_dir():
            raise FileNotFoundError(f"Lucene project folder not found: {self.project_dir}")
        self.version = version
        self.src_dir = self.project_dir / "src" / "main" / "java"
        self.classes_dir = self.project_dir / "target" / "classes"
        self.main_class = "mmir.Search"
        self._javac, self._java = self._find_jdk()
        self._jars = self._ensure_jars()
        self._index_dir: str | None = None

    # ── JDK discovery ─────────────────────────────────────────────────────────

    def _find_jdk(self) -> tuple[str, str]:
        javac = shutil.which("javac")
        java = shutil.which("java")
        if not javac or not java:
            raise JavaNotFound(self._install_hint("No JDK found on PATH (need both `javac` and `java`)."))
        result = subprocess.run([javac, "-version"], capture_output=True, text=True)
        reported = (result.stdout + result.stderr).strip()
        match = re.search(r"\b(\d+)", reported)
        major = int(match.group(1)) if match else 0
        if major < MIN_JAVA:
            raise JavaNotFound(self._install_hint(
                f"Found '{reported}', but Lucene {self.version} needs Java {MIN_JAVA} or newer."))
        return javac, java

    @staticmethod
    def _install_hint(message: str) -> str:
        if sys.platform == "darwin":
            how = "macOS:  brew install openjdk@21  (then follow brew's note to put it on PATH)"
        elif sys.platform.startswith("win"):
            how = "Windows:  install Temurin 21 from https://adoptium.net/ and open a new terminal"
        else:
            how = "Linux:  sudo apt install openjdk-21-jdk  (or install a JDK via https://sdkman.io/)"
        return (f"{message}\n\n"
                f"This exercise needs a Java Development Kit (JDK) {MIN_JAVA} or newer.\n"
                f"{how}\n"
                f"After installing, restart the notebook kernel so it picks up the new PATH.")

    # ── dependency jars ───────────────────────────────────────────────────────

    def _ensure_jars(self) -> list[Path]:
        cache = _jar_cache(self.version)
        jars: list[Path] = []
        for artifact in _ARTIFACTS:
            jar = cache / f"{artifact}-{self.version}.jar"
            if not jar.exists():
                url = f"{_MAVEN_BASE}/{artifact}/{self.version}/{artifact}-{self.version}.jar"
                tmp = jar.with_name(jar.name + ".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(jar)
            jars.append(jar)
        return jars

    def _classpath(self) -> str:
        parts = [str(p) for p in self._jars] + [str(self.classes_dir)]
        return os.pathsep.join(parts)

    # ── compile and run ─────────────────────────────────────────────────────────

    def _needs_compile(self) -> bool:
        if not self.classes_dir.exists():
            return True
        newest_src = max((p.stat().st_mtime for p in self.src_dir.rglob("*.java")), default=0.0)
        newest_cls = max((p.stat().st_mtime for p in self.classes_dir.rglob("*.class")), default=-1.0)
        return newest_src > newest_cls

    def compile(self) -> None:
        """Compile every .java file under src/main/java into target/classes."""
        sources = sorted(str(p) for p in self.src_dir.rglob("*.java"))
        if not sources:
            raise FileNotFoundError(f"No .java sources found under {self.src_dir}")
        self.classes_dir.mkdir(parents=True, exist_ok=True)
        cmd = [self._javac, "-cp", self._classpath(), "-d", str(self.classes_dir), *sources]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("Java compilation failed:\n\n" + proc.stdout + proc.stderr)

    def run(self, *args, timeout: int = 180) -> str:
        """Run `java mmir.Search <args>` and return its stdout (recompiling if needed)."""
        if self._needs_compile():
            self.compile()
        cmd = [self._java, "-cp", self._classpath(), self.main_class, *map(str, args)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            joined = " ".join(map(str, args))
            raise RuntimeError(f"`{self.main_class} {joined}` failed:\n\n{proc.stdout}{proc.stderr}")
        return proc.stdout

    # ── convenience wrappers around the CLI contract ────────────────────────────

    def index(self, data_file: str | os.PathLike, index_dir: str | os.PathLike | None = None) -> dict:
        """Build the Lucene index from a TSV file (id, title, text, year per line)."""
        if index_dir is None:
            index_dir = self.project_dir / "target" / "index"
        self._index_dir = str(Path(index_dir).resolve())
        out = self.run("index", "--data", str(Path(data_file).resolve()), "--dir", self._index_dir)
        return json.loads(out or "{}")

    def query(self, q: str, k: int = 10, year_min: int | None = None, fuzzy: bool = False) -> list[dict]:
        """Run a search and return the hits as a list of {id, title, score, year} dicts."""
        if self._index_dir is None:
            raise RuntimeError("Call index(...) before query(...).")
        args = ["query", "--dir", self._index_dir, "--q", q, "--k", str(k)]
        if year_min is not None:
            args += ["--year-min", str(year_min)]
        if fuzzy:
            args += ["--fuzzy"]
        return json.loads(self.run(*args) or "[]")
