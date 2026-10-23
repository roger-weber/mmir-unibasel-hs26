# Lucene search project

This is the Java project for the Chapter 4 exercise. You edit one file,
[`src/main/java/mmir/Search.java`](src/main/java/mmir/Search.java), and implement two
methods: `buildDocument` and `buildQuery`. The notebook drives everything else.

## Requirements

- A **JDK, version 21 or newer** (Lucene 10 needs Java 21+). Check with `javac -version`.
- Nothing else. The notebook's Python harness downloads the Lucene jars for you.

## Running it from the notebook (the normal way)

You do not compile or run this by hand. In the notebook:

```python
from shared.lucene_harness import LuceneProject
proj = LuceneProject("lucene")   # this folder
proj.index("movies.tsv")         # build the index
proj.query("space adventure", k=10)
```

Edit `Search.java`, save it, and rerun the cell. The harness recompiles automatically.

## Editing in an IDE (optional, for autocompletion)

The `pom.xml` makes this a standard Maven project, so IntelliJ, Eclipse, or VS Code
(with the Java extension) will resolve the Lucene classes and give you completion.
Open the folder as a Maven project. You do not need Maven on the command line; the
notebook harness handles compiling and running.

## The command-line contract

The notebook talks to the compiled program through two subcommands:

- `index --data <tsv> --dir <indexdir>` reads a tab-separated file
  (`id`, `title`, `text`, `year` per line) and builds the index. Prints `{"indexed": n}`.
- `query --dir <indexdir> --q "<text>" --k <n> [--year-min <year>] [--fuzzy]` searches
  and prints a JSON array of `{"id", "title", "score", "year"}`, best match first.
