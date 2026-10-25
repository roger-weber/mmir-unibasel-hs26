"""Ch07 helper - a two-hop retriever (the hard-wired loop of a synthesizer).

The knowledge base `kb` (shared/nobel_rag.py) provides:
    kb.search(query, k)  ids of the top-k parent passages, best first (one retrieval pass)
    kb.text(pid)         the text of a parent passage
    kb.label(pid)        "Article > Section" of a parent passage
    kb.persons(pid)      person names in a passage, in order of appearance (no retrieval pass)
"""


def two_hop(question, kb, budget, fill):
    """Retrieve in two hops and return the context for a generator.

    Hop 1 retrieves passages for `question`. Person names found in hop-1 passages are
    the bridges: hop 2 retrieves passages for them. The passages of both hops are
    assembled into one context with `fill(ranked_ids, budget)`.

    Returns {"context": list of parent ids, "bridges": list of the names followed}.

    Requirements, checked by the notebook on all 24 benchmark questions at budget 8000:
      - the context is what `fill` returns: within the budget, no repeats,
      - at most 6 calls of kb.search per question,
      - mean context recall of at least 0.85 on the multi-hop questions,
      - mean context recall on the single lookups at least that of a single pass,
        fill(kb.search(question, 60), budget).
    """
    # YOUR CODE HERE
    raise NotImplementedError
