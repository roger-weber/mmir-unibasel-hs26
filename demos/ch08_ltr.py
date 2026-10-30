"""Learning to rank on TREC-COVID (ch08-03): one feature vector per judged topic-document pair.

The features are the content signals of Chapter 8, computed on the two fields of a paper
(title and abstract) for the topic's short keyword query:

    bm25_title, bm25_abstract   BM25 on one field (k1 = 1.2, b = 0.75)
    bm25f                       BM25F with per-field length normalization (title weight 3)
    log_proximity               log(1 + binned proximity score of ch08-01, merged hit lists)
    coverage                    fraction of query terms that occur in the paper
    dense                       cosine of all-MiniLM-L6-v2 embeddings (topic question vs paper)
    log_title_len, ...          log(1 + field length in tokens), query-independent

The proximity score and the lengths are very skewed (a few abstracts have thousands of
tokens), so they enter as logarithms; trees do not care, linear models do.

The dense scores take about 30 minutes to compute on a CPU (35,480 papers), so they are
precomputed for all judged pairs and stored in eval/ch08_ltr/dense_scores.npz. Recompute
them with `python ch08_ltr.py`.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from shared.text import stem, stopwords_for

EVAL_DIR = Path(__file__).parent / "eval" / "ch08_ltr"
DENSE_FILE = EVAL_DIR / "dense_scores.npz"
DENSE_MODEL = "all-MiniLM-L6-v2"

FEATURES = ["bm25_title", "bm25_abstract", "bm25f", "log_proximity", "coverage", "dense",
            "log_title_len", "log_abstract_len"]
K1, B = 1.2, 0.75
FIELD_WEIGHTS = {"title": 3.0, "abstract": 1.0}
PROX_WEIGHTS = (89, 55, 34, 21, 13, 8, 5, 3, 2, 1)
_STOP = stopwords_for("en")


@lru_cache(maxsize=None)
def _stem(token: str) -> str:
    return stem(token)


def analyze(text: str) -> list[str]:
    """Lowercase word tokens, Porter-stemmed; positions are list index + 1."""
    return [_stem(t) for t in re.findall(r"\w+", text.lower())]


def query_terms(query: str) -> list[str]:
    """Stemmed query terms without stopwords and duplicates, in query order."""
    terms = [_stem(t) for t in re.findall(r"\w+", query.lower()) if t not in _STOP]
    return list(dict.fromkeys(terms))


# ─── Proximity (same rule as ch08-01) ────────────────────────────────────────

def proximity(tokens: list[str], terms: list[str], weights=PROX_WEIGHTS) -> int:
    """Merge the hit lists of all query terms; bin neighbouring hits of different terms."""
    merged = [(k, t) for k, t in enumerate(tokens) if t in terms]
    score = 0
    for (p1, t1), (p2, t2) in zip(merged, merged[1:]):
        if t1 != t2 and p2 - p1 <= len(weights):
            score += weights[p2 - p1 - 1]
    return score


# ─── Corpus statistics ───────────────────────────────────────────────────────

def corpus_stats(collection, terms: set[str]) -> dict:
    """Document frequencies of the query terms and average field lengths over the corpus."""
    df = Counter()
    length = {"title": 0, "abstract": 0}
    n = 0
    for doc in collection._docs.values():
        n += 1
        title, abstract = analyze(doc["title"]), analyze(doc["abstract"])
        length["title"] += len(title)
        length["abstract"] += len(abstract)
        df.update(terms & (set(title) | set(abstract)))
    return {"n": n, "df": df, "avg": {f: length[f] / n for f in length}}


def idf(term: str, stats: dict) -> float:
    n, df = stats["n"], stats["df"][term]
    return math.log((n - df + 0.5) / (df + 0.5) + 1)


def bm25_field(tf: Counter, length: int, avg: float, terms, stats) -> float:
    s = 0.0
    for t in terms:
        if tf[t]:
            norm = K1 * (1 - B + B * length / avg)
            s += idf(t, stats) * tf[t] * (K1 + 1) / (tf[t] + norm)
    return s


def bm25f(tfs: dict, lengths: dict, terms, stats) -> float:
    """BM25F: length-normalize and weight each field, add, then saturate once."""
    s = 0.0
    for t in terms:
        tf_tilde = sum(FIELD_WEIGHTS[f] * tfs[f][t] / (1 - B + B * lengths[f] / stats["avg"][f])
                       for f in FIELD_WEIGHTS)
        if tf_tilde:
            s += idf(t, stats) * tf_tilde * (K1 + 1) / (tf_tilde + K1)
    return s


# ─── Feature matrix ──────────────────────────────────────────────────────────

def build_features(collection, assessments, dense=None):
    """Feature matrix X, grades y, topic id per row, and document ids for all judged pairs."""
    topics = assessments.query_ids()
    terms = {q: query_terms(assessments.topics[q].keywords) for q in topics}
    stats = corpus_stats(collection, set().union(*terms.values()))
    dense = dense if dense is not None else load_dense_scores()

    rows, y, qids, dids = [], [], [], []
    for q in topics:
        for d, grade in assessments.qrels[q].items():
            doc = collection._docs[d]
            title, abstract = analyze(doc["title"]), analyze(doc["abstract"])
            tfs = {"title": Counter(title), "abstract": Counter(abstract)}
            lengths = {"title": len(title), "abstract": len(abstract)}
            present = sum(1 for t in terms[q] if tfs["title"][t] or tfs["abstract"][t])
            rows.append([
                bm25_field(tfs["title"], len(title), stats["avg"]["title"], terms[q], stats),
                bm25_field(tfs["abstract"], len(abstract), stats["avg"]["abstract"], terms[q], stats),
                bm25f(tfs, lengths, terms[q], stats),
                math.log1p(proximity(title + abstract, terms[q])),
                present / len(terms[q]),
                dense.get((q, d), 0.0),
                math.log1p(len(title)),
                math.log1p(len(abstract)),
            ])
            y.append(grade)
            qids.append(q)
            dids.append(d)
    return np.array(rows), np.array(y), np.array(qids), np.array(dids)


# ─── Dense scores (precomputed) ──────────────────────────────────────────────

def load_dense_scores() -> dict:
    """{(topic id, doc id): cosine} from the precomputed file."""
    data = np.load(DENSE_FILE)
    return {(q, d): float(s) for q, d, s in zip(data["qids"], data["dids"], data["scores"])}


def compute_dense_scores(collection, assessments, batch_size=64):
    """Embed every judged paper and topic question; store the cosine of each judged pair."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL)
    doc_ids = sorted({d for labels in assessments.qrels.values() for d in labels})
    doc_vecs = model.encode([collection._docs[d]["text"] for d in doc_ids], batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=True)
    topics = assessments.query_ids()
    q_vecs = model.encode([assessments.topics[q].question for q in topics], normalize_embeddings=True)
    row = {d: i for i, d in enumerate(doc_ids)}
    qids, dids, scores = [], [], []
    for qi, q in enumerate(topics):
        for d in assessments.qrels[q]:
            qids.append(q)
            dids.append(d)
            scores.append(float(doc_vecs[row[d]] @ q_vecs[qi]))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DENSE_FILE, qids=np.array(qids), dids=np.array(dids),
                        scores=np.array(scores, dtype=np.float32))
    return len(scores)


if __name__ == "__main__":
    from shared.collections import load_collection
    from shared.assessments import load_assessments

    n = compute_dense_scores(load_collection("trec-covid"), load_assessments("trec-covid"))
    print(f"stored {n:,} dense scores in {DENSE_FILE}")
