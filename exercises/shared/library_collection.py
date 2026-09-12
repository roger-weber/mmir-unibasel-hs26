"""
Library collection for evaluation demos (Chapter 2).

A 50-book university library: 15 computer science titles and 35 works of
fiction, drama, poetry, and general non-fiction. One information need, graded
relevance judgments, and two contrasting retrieval runs.

Usage:
    from shared.library_collection import (
        LIBRARY, NEED, GRADES, RUNS,
        documents, record, text, ids, texts, doc2text,
        grade, is_relevant, relevant_ids,
    )
"""

# ─── Collection ──────────────────────────────────────────────────────────────
# Each entry maps a short key to a metadata record. Keeping metadata as fields
# (rather than one pre-joined string) lets demos filter by genre or year and
# lets doc2text decide what counts as searchable text.

LIBRARY = {
    # Computer science titles
    "database-systems": {
        "title": "Database Systems: The Complete Book",
        "author": "Hector Garcia-Molina, Jeffrey D. Ullman, and Jennifer Widom",
        "genre": "Computer Science", "year": 2008},
    "pattern-recognition": {
        "title": "Pattern Recognition and Machine Learning",
        "author": "Christopher Bishop",
        "genre": "Computer Science", "year": 2006},
    "data-science-scratch": {
        "title": "Data Science from Scratch",
        "author": "Joel Grus",
        "genre": "Computer Science", "year": 2015},
    "operating-systems": {
        "title": "Operating System Concepts",
        "author": "Abraham Silberschatz, Peter B. Galvin, and Greg Gagne",
        "genre": "Computer Science", "year": 2018},
    "ai-modern-approach": {
        "title": "Artificial Intelligence: A Modern Approach",
        "author": "Stuart Russell and Peter Norvig",
        "genre": "Computer Science", "year": 2020},
    "programming-languages": {
        "title": "Concepts of Programming Languages",
        "author": "Robert W. Sebesta",
        "genre": "Computer Science", "year": 2015},
    "theory-of-computation": {
        "title": "Introduction to the Theory of Computation",
        "author": "Michael Sipser",
        "genre": "Computer Science", "year": 2012},
    "everyday-things": {
        "title": "The Design of Everyday Things",
        "author": "Don Norman",
        "genre": "Computer Science", "year": 2013},
    "computer-organization": {
        "title": "Computer Organization and Design: The Hardware/Software Interface",
        "author": "David A. Patterson and John L. Hennessy",
        "genre": "Computer Science", "year": 2017},
    "computer-networks": {
        "title": "Computer Networks",
        "author": "Andrew Tanenbaum and David Wetherall",
        "genre": "Computer Science", "year": 2010},
    "cryptography": {
        "title": "Cryptography and Network Security: Principles and Practice",
        "author": "William Stallings",
        "genre": "Computer Science", "year": 2016},
    "sicp": {
        "title": "Structure and Interpretation of Computer Programs",
        "author": "Harold Abelson and Gerald Jay Sussman",
        "genre": "Computer Science", "year": 1996},
    "mythical-man-month": {
        "title": "The Mythical Man-Month: Essays on Software Engineering",
        "author": "Frederick P. Brooks Jr.",
        "genre": "Computer Science", "year": 1975},
    "computer-systems": {
        "title": "Computer Systems: A Programmer's Perspective",
        "author": "Randal E. Bryant and David R. O'Hallaron",
        "genre": "Computer Science", "year": 2015},
    "algorithms": {
        "title": "Introduction to Algorithms",
        "author": ("Thomas H. Cormen, Charles E. Leiserson, "
                   "Ronald L. Rivest, and Clifford Stein"),
        "genre": "Computer Science", "year": 2022},

    # Fiction, drama, poetry, and general non-fiction
    "blind-assassin": {
        "title": "The Blind Assassin", "author": "Margaret Atwood",
        "genre": "Fiction", "year": 2000},
    "angels-in-america": {
        "title": "Angels in America", "author": "Tony Kushner",
        "genre": "Drama", "year": 1991},
    "selected-poems": {
        "title": "Selected Poems", "author": "Gwendolyn Brooks",
        "genre": "Poetry", "year": 1963},
    "handmaids-tale": {
        "title": "The Handmaid's Tale", "author": "Margaret Atwood",
        "genre": "Fiction", "year": 1985},
    "penguin-history": {
        "title": "The Penguin History of the World", "author": "J.M. Roberts",
        "genre": "Non-Fiction", "year": 1980},
    "ragtime": {
        "title": "Ragtime", "author": "E.L. Doctorow",
        "genre": "Fiction", "year": 1975},
    "buried-child": {
        "title": "Buried Child", "author": "Sam Shepard",
        "genre": "Drama", "year": 1978},
    "ariel": {
        "title": "Ariel", "author": "Sylvia Plath",
        "genre": "Poetry", "year": 1965},
    "mockingbird": {
        "title": "To Kill a Mockingbird", "author": "Harper Lee",
        "genre": "Fiction", "year": 1960},
    "new-architecture": {
        "title": "Towards a New Architecture", "author": "Le Corbusier",
        "genre": "Non-Fiction", "year": 1923},
    "catcher-in-the-rye": {
        "title": "The Catcher in the Rye", "author": "J.D. Salinger",
        "genre": "Fiction", "year": 1951},
    "death-of-a-salesman": {
        "title": "Death of a Salesman", "author": "Arthur Miller",
        "genre": "Drama", "year": 1949},
    "on-the-road": {
        "title": "On the Road", "author": "Jack Kerouac",
        "genre": "Fiction", "year": 1957},
    "brave-new-world": {
        "title": "Brave New World", "author": "Aldous Huxley",
        "genre": "Fiction", "year": 1932},
    "room-of-ones-own": {
        "title": "A Room of One's Own", "author": "Virginia Woolf",
        "genre": "Non-Fiction", "year": 1929},
    "great-gatsby": {
        "title": "The Great Gatsby", "author": "F. Scott Fitzgerald",
        "genre": "Fiction", "year": 1925},
    "waste-land": {
        "title": "The Waste Land", "author": "T.S. Eliot",
        "genre": "Poetry", "year": 1922},
    "swanns-way": {
        "title": "Swann's Way", "author": "Marcel Proust",
        "genre": "Fiction", "year": 1913},
    "demonology": {
        "title": "Letters on Demonology and Witchcraft", "author": "Walter Scott",
        "genre": "Non-Fiction", "year": 1830},
    "wind-in-the-willows": {
        "title": "The Wind in the Willows", "author": "Kenneth Grahame",
        "genre": "Fiction", "year": 1908},
    "songs-of-innocence": {
        "title": "Songs of Innocence and of Experience", "author": "William Blake",
        "genre": "Poetry", "year": 1794},
    "dracula": {
        "title": "Dracula", "author": "Bram Stoker",
        "genre": "Fiction", "year": 1897},
    "interpretation-of-dreams": {
        "title": "The Interpretation of Dreams", "author": "Sigmund Freud",
        "genre": "Non-Fiction", "year": 1899},
    "huckleberry-finn": {
        "title": "Adventures of Huckleberry Finn", "author": "Mark Twain",
        "genre": "Fiction", "year": 1884},
    "dolls-house": {
        "title": "A Doll's House", "author": "Henrik Ibsen",
        "genre": "Drama", "year": 1879},
    "flowers-of-evil": {
        "title": "Flowers of Evil", "author": "Charles Baudelaire",
        "genre": "Poetry", "year": 1857},
    "crime-and-punishment": {
        "title": "Crime and Punishment", "author": "Fyodor Dostoevsky",
        "genre": "Fiction", "year": 1866},
    "origin-of-species": {
        "title": "On the Origin of Species", "author": "Charles Darwin",
        "genre": "Non-Fiction", "year": 1859},
    "madame-bovary": {
        "title": "Madame Bovary", "author": "Gustave Flaubert",
        "genre": "Fiction", "year": 1856},
    "leaves-of-grass": {
        "title": "Leaves of Grass", "author": "Walt Whitman",
        "genre": "Poetry", "year": 1855},
    "jane-eyre": {
        "title": "Jane Eyre", "author": "Charlotte Brontë",
        "genre": "Fiction", "year": 1847},
    "frederick-douglass": {
        "title": "Narrative of the Life of Frederick Douglass, an American Slave",
        "author": "Frederick Douglass",
        "genre": "Non-Fiction", "year": 1845},
    "macbeth": {
        "title": "Macbeth", "author": "William Shakespeare",
        "genre": "Drama", "year": 1623},
    "prometheus-unbound": {
        "title": "Prometheus Unbound", "author": "Percy Bysshe Shelley",
        "genre": "Poetry", "year": 1820},
    "frankenstein": {
        "title": "Frankenstein", "author": "Mary Shelley",
        "genre": "Fiction", "year": 1818},
}


# ─── Information Need and Relevance Judgments ────────────────────────────────

GRADE_LABELS = {
    3: "core foundations",
    2: "important specialization",
    1: "useful peripheral reading",
    0: "not relevant",
}

# The primary need carries graded judgments (3/2/1) and drives the worked
# examples. The secondary needs are judged binary (grade 1) and exist so that
# averaging across needs can be demonstrated on real numbers.
PRIMARY_NEED = "cs-foundations"

NEEDS = {
    "cs-foundations": {
        "text": ("Which books give a beginning master's student solid "
                 "computer science foundations?"),
        "graded": True,
        "grades": {
            "database-systems": 3,
            "pattern-recognition": 3,
            "data-science-scratch": 3,
            "operating-systems": 2,
            "ai-modern-approach": 2,
            "programming-languages": 2,
            "theory-of-computation": 2,
            "everyday-things": 1,
            "computer-organization": 1,
            "computer-networks": 1,
            "cryptography": 1,
            "sicp": 1,
            "mythical-man-month": 1,
            "computer-systems": 1,
            "algorithms": 1,
        },
    },
    "stage-plays": {
        "text": "Which works in the collection are stage plays?",
        "graded": False,
        "grades": {
            "macbeth": 1,
            "death-of-a-salesman": 1,
            "dolls-house": 1,
            "angels-in-america": 1,
            "buried-child": 1,
        },
    },
    "atwood": {
        "text": "Which books were written by Margaret Atwood?",
        "graded": False,
        "grades": {
            "blind-assassin": 1,
            "handmaids-tale": 1,
        },
    },
    "russian-novels": {
        "text": "Which nineteenth-century Russian novels does the library hold?",
        "graded": False,
        "grades": {
            "crime-and-punishment": 1,
        },
    },
}

# Convenience aliases for the primary need.
NEED = NEEDS[PRIMARY_NEED]["text"]
GRADES = NEEDS[PRIMARY_NEED]["grades"]


# ─── Retrieval Runs ──────────────────────────────────────────────────────────
# Result lists per need, ordered by rank. System A favours coverage, System B
# favours a clean result set. All runs are hand-designed to illustrate the
# precision-recall tension; no live retrieval produced them.

RUNS = {
    "cs-foundations": {
        "A": [
            "computer-organization", "operating-systems", "handmaids-tale",
            "frederick-douglass", "new-architecture", "data-science-scratch",
            "programming-languages", "leaves-of-grass", "ai-modern-approach",
            "ragtime", "angels-in-america", "database-systems",
            "theory-of-computation", "madame-bovary", "dracula", "sicp",
            "macbeth", "pattern-recognition", "huckleberry-finn",
            "room-of-ones-own", "computer-systems", "flowers-of-evil",
            "computer-networks", "crime-and-punishment", "algorithms",
        ],
        "B": [
            "database-systems", "pattern-recognition", "algorithms",
            "origin-of-species", "theory-of-computation", "operating-systems",
            "penguin-history", "data-science-scratch",
        ],
    },
    "stage-plays": {
        "A": [
            "macbeth", "prometheus-unbound", "death-of-a-salesman",
            "selected-poems", "dolls-house", "angels-in-america",
        ],
        "B": [
            "macbeth", "death-of-a-salesman", "dolls-house",
            "prometheus-unbound",
        ],
    },
    "atwood": {
        "A": [
            "handmaids-tale", "room-of-ones-own", "blind-assassin",
            "jane-eyre",
        ],
        "B": [
            "blind-assassin", "handmaids-tale",
        ],
    },
    "russian-novels": {
        "A": [
            "madame-bovary", "swanns-way", "flowers-of-evil",
        ],
        "B": [
            "crime-and-punishment",
        ],
    },
}


# ─── Access Functions ────────────────────────────────────────────────────────

def doc2text(doc, fields=None) -> str:
    """
    Flatten a document into one searchable string.

    Keeps retrieval code independent of the metadata schema: a collection of
    dictionaries, a collection of plain strings, or a collection with entirely
    different fields all reduce to text through this one function.

    Args:
        doc: A metadata dictionary, or a plain string.
        fields: Field names to include, in order. Defaults to every field.

    Returns:
        The selected field values joined into a single string.
    """
    if isinstance(doc, str):
        return doc
    keys = fields if fields is not None else list(doc.keys())
    parts = [str(doc[k]) for k in keys if doc.get(k) not in (None, "")]
    return ". ".join(parts)


def ids(collection=None) -> list[str]:
    """All document keys in collection order."""
    return list((collection or LIBRARY).keys())


def documents(collection=None) -> list[tuple[str, dict]]:
    """Iterate (key, record) pairs."""
    return list((collection or LIBRARY).items())


def record(key: str, collection=None) -> dict:
    """Look up the full metadata record for one document."""
    return (collection or LIBRARY)[key]


def text(key: str, collection=None, fields=None) -> str:
    """Searchable text for one document."""
    return doc2text((collection or LIBRARY)[key], fields)


def texts(collection=None, fields=None) -> dict[str, str]:
    """Searchable text for every document: {key: text}."""
    coll = collection or LIBRARY
    return {k: doc2text(v, fields) for k, v in coll.items()}


def need_ids() -> list[str]:
    """All information need identifiers, primary need first."""
    return list(NEEDS.keys())


def need_text(need: str = PRIMARY_NEED) -> str:
    """The information need, stated as a question."""
    return NEEDS[need]["text"]


def grades(need: str = PRIMARY_NEED) -> dict[str, int]:
    """Graded relevance judgments for one need."""
    return NEEDS[need]["grades"]


def grade(key: str, need: str = PRIMARY_NEED) -> int:
    """Graded relevance of a document, 0 if unjudged or not relevant."""
    return NEEDS[need]["grades"].get(key, 0)


def is_relevant(key: str, need: str = PRIMARY_NEED) -> bool:
    """Binary relevance: any positive grade counts as relevant."""
    return grade(key, need) > 0


def relevant_ids(need: str = PRIMARY_NEED) -> list[str]:
    """Keys of all relevant documents for a need, highest grade first."""
    g = NEEDS[need]["grades"]
    return sorted(g, key=lambda k: -g[k])


def system_ids() -> list[str]:
    """All system identifiers that have runs."""
    return sorted({s for runs in RUNS.values() for s in runs})


def run(need: str = PRIMARY_NEED, system: str = "A") -> list[str]:
    """The ranked result list one system returned for one need."""
    return RUNS[need][system]
