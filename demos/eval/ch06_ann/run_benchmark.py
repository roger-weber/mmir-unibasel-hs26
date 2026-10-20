"""
Offline benchmark harness for the chapter 6 ANN demo.

Runs every (Matryoshka dimension, coarse filter, fine filter) configuration over
the 50 TREC-COVID topics, retrieves the top 50 per query, reranks them with a
cross-encoder to a top 10, and writes the artifacts next to this script:

    runs.json            per configuration: retrieved top 10, reranked top 10,
                         overlap of the retrieved top 50 with exact search, and cost
    pool_to_judge.json   unjudged (query, doc) pairs that appear in any top 10

Needs a GPU for encoding the 171k abstracts with Qwen3-Embedding-0.6B (minutes
on a recent GPU, hours on a CPU); embeddings are cached under data/.cache/ch06_ann/.
Run from the demos/ directory:

    python -m eval.ch06_ann.run_benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from shared.collections import load_collection, CACHE_DIR
from shared.assessments import load_assessments
from ch06_ann import truncate, Retriever, AllVectors, IVF, HNSW, Exact, PQ, LSH

HERE = Path(__file__).parent
EMB_DIR = CACHE_DIR / "ch06_ann"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"      # Matryoshka-trained, 1024 dimensions
RERANKERS = {"qwen3": "Qwen/Qwen3-Reranker-0.6B",            # main pipeline: stronger than the retriever
             "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2"}  # the ch05 reranker, for comparison
DIMS = [1024, 512, 256, 128, 64]
NPROBES = [1, 4, 16]
HNSW_EFS = [50, 200]
RETRIEVE_K, FINAL_K = 50, 10


def encode(collection, assessments):
    """Encode corpus and queries once; cache as float16 / float32 .npy files."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    doc_file, q_file = EMB_DIR / "docs_qwen3_1024_fp16.npy", EMB_DIR / "queries_qwen3_1024.npy"
    if doc_file.exists() and q_file.exists():
        return np.load(doc_file).astype(np.float32), np.load(q_file)
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512
    texts = [d["text"] for d in collection]
    order = np.argsort([len(t) for t in texts])          # length-sorted batches run faster
    t0 = time.time()
    E = model.encode([texts[i] for i in order], batch_size=64, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    docs = np.empty_like(E)
    docs[order] = E
    print(f"encoded {len(texts):,} documents in {time.time() - t0:.0f} s")
    queries = model.encode([assessments.topics[q].question for q in assessments.query_ids()],
                           prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)
    np.save(doc_file, docs.astype(np.float16))
    np.save(q_file, queries.astype(np.float32))
    return docs.astype(np.float32), queries.astype(np.float32)


def configurations(X):
    """All retrievers for one dimension, sharing trained components."""
    exact = Exact().fit(X)
    fines = [exact] + [PQ(m).fit(X) for m in (8, 16, 32)] + [LSH(b).fit(X) for b in (128, 512)]
    ivf = IVF(nlist=1024).fit(X)
    out = []
    for fine in fines:
        out.append(Retriever(AllVectors().fit(X), fine))
        for p in NPROBES:
            out.append(Retriever(ivf.with_nprobe(p), fine))
    for pq_m in (None, 32):
        h = HNSW(M=32, pq_m=pq_m).fit(X)
        for ef in HNSW_EFS:
            out.append(Retriever(h.with_ef(ef)))
    return out


def main():
    collection = load_collection("trec-covid")
    assessments = load_assessments("trec-covid")
    doc_ids, qids = collection.ids(), assessments.query_ids()
    docs, queries = encode(collection, assessments)

    runs = {}
    for dim in DIMS:
        X, Q = truncate(docs, dim), truncate(queries, dim)
        t0 = time.time()
        for r in configurations(X):
            res = [r.search(Q[i], RETRIEVE_K) for i in range(len(qids))]
            runs[f"d={dim} | {r.name}"] = {
                "dim": dim, "coarse": r.coarse.name, "fine": r.name.split(" + ")[1],
                "retrieved": {q: [doc_ids[j] for j in ids] for q, (ids, _) in zip(qids, res)},
                "vectors": float(np.mean([c.vectors for _, c in res])),
                "bytes": float(np.mean([c.bytes for _, c in res])),
            }
        print(f"d={dim}: {len(runs)} configurations so far ({time.time() - t0:.0f} s)")

    # Rerank every retrieved top 50 with each reranker; score each distinct pair once.
    import torch
    from sentence_transformers import CrossEncoder
    pairs = sorted({(q, d) for run in runs.values() for q, ds in run["retrieved"].items() for d in ds})
    text_of = {q: assessments.topics[q].question for q in qids}
    inputs = [(text_of[q], collection[d]["text"]) for q, d in pairs]
    score = {}
    for key, model_id in RERANKERS.items():
        t0 = time.time()
        ce = CrossEncoder(model_id, device="cuda" if torch.cuda.is_available() else "cpu", max_length=512)
        s = ce.predict(inputs, batch_size=64, show_progress_bar=False)
        score[key] = {p: float(v) for p, v in zip(pairs, s)}
        del ce
        torch.cuda.empty_cache()
        print(f"{key}: scored {len(pairs):,} distinct pairs in {time.time() - t0:.0f} s")

    reference = runs["d=1024 | all + exact"]["retrieved"]
    out, pool = {}, set()
    for name, run in runs.items():
        reranked = {key: {q: sorted(ds, key=lambda d: -score[key][(q, d)])[:FINAL_K]
                          for q, ds in run["retrieved"].items()} for key in RERANKERS}
        overlap50 = float(np.mean([len(set(run["retrieved"][q]) & set(reference[q])) / RETRIEVE_K for q in qids]))
        out[name] = {k: run[k] for k in ("dim", "coarse", "fine", "vectors", "bytes")}
        out[name].update({
            "overlap50_exact": overlap50,
            "retrieved": run["retrieved"],          # top 50, in retriever order
            "reranked": reranked,                   # {reranker: {query: top 10}}
        })
        for q in qids:
            for d in [d for key in RERANKERS for d in reranked[key][q]] + run["retrieved"][q][:FINAL_K]:
                if not assessments.is_judged(q, d):
                    pool.add((q, d))

    meta = {"collection": "trec-covid", "embed_model": EMBED_MODEL, "rerankers": RERANKERS,
            "retrieve_k": RETRIEVE_K, "final_k": FINAL_K, "n_docs": len(doc_ids),
            "bytes_full_scan": len(doc_ids) * 1024 * 4}
    (HERE / "runs.json").write_text(json.dumps({"meta": meta, "runs": out}), encoding="utf-8")
    (HERE / "pool_to_judge.json").write_text(json.dumps(
        {"n_pairs": len(pool), "pairs": sorted(pool)}, indent=0), encoding="utf-8")
    print(f"{len(out)} configurations written; {len(pool):,} unjudged pairs to judge")


if __name__ == "__main__":
    main()
