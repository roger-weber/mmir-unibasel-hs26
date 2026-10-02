"""
Offline trial runner for the chapter 5 semantic-search evaluation.

Runs every (pipeline, model) condition over the query set on the movies-small
collection, collects the top-10 ranked results per condition, and writes four
JSON artifacts next to this script:

    raw_results.json       ranked top-10 (id + title + score) per query per condition
    pool_to_judge.json     deduplicated (query, movie) pairs across all conditions
    costs.json             index build time, per-query latency, storage per method
    divergence.json        top-k overlap between conditions (how much they diverge)
    matryoshka_sweep.json  recall@10 vs full-dim and storage for truncated dims

It also writes the committed corpus embeddings (embeddings/*.npy + manifest.json)
that the student notebook loads instead of re-encoding the corpus.

Run from the demos/ directory so both `shared` and `eval` are importable:

    python -m eval.ch05_movies.run_trial

This is an offline harness. It does not judge relevance and does not build a
student notebook; it produces artifacts for the author to inspect.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from shared.collections import load_collection, CACHE_DIR
from eval.ch05_movies import pipelines as P

HERE = Path(__file__).parent
QUERIES_FILE = HERE / "queries.json"
EMB_CACHE_DIR = CACHE_DIR / "ch05_eval"
# Committed (tracked) embeddings the student notebook loads instead of re-encoding.
# This directory is NOT gitignored, so the matrices ship with the repo.
EMB_COMMIT_DIR = HERE / "embeddings"

# Embedding models: (label, hf_id, query_prompt_name, matryoshka_capable)
SMALL_MODEL = ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2", None, False)
LARGE_MODEL = ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", "query", True)
CROSS_ENCODER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 10
RETRIEVE_K = 50          # candidate pool for rerank / fusion stages
MATRYOSHKA_FIRST_DIM = 256
MATRYOSHKA_FIRST_K = 50
TRUNCATION_DIMS = [1024, 512, 256, 128, 64]


def load_queries() -> list[dict]:
    data = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    return data["queries"]


def short_context(doc: dict, limit: int = 260) -> str:
    """A compact context string for the later judging step."""
    text = doc.get("text", "")
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def mean_ms(times: list[float]) -> float:
    return 1000.0 * sum(times) / len(times) if times else 0.0


def encode_or_load(model, label: str, texts: list[str]) -> tuple[np.ndarray, float, bool]:
    """
    Encode the corpus, caching the embedding matrix on disk.

    The embedding build is a one-time cost (and Qwen on CPU is slow), so the
    normalised float32 matrix and the measured encode time are cached under the
    gitignored data/.cache. A cache hit returns instantly and reuses the
    originally measured encode time, so costs.json always reports the true cost.
    """
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace("/", "_")
    npy_path = EMB_CACHE_DIR / f"movies-small__{safe}.npy"
    meta_path = EMB_CACHE_DIR / f"movies-small__{safe}.meta.json"
    if npy_path.exists() and meta_path.exists():
        emb = np.load(npy_path)
        if emb.shape[0] == len(texts):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return emb.astype(np.float32), float(meta["encode_s"]), True
    t0 = time.time()
    emb = P.encode_corpus(model, texts)
    encode_s = time.time() - t0
    np.save(npy_path, emb)
    meta_path.write_text(json.dumps({"encode_s": encode_s, "dim": int(emb.shape[1]),
                                     "n_docs": int(emb.shape[0])}), encoding="utf-8")
    return emb, encode_s, False


def export_committed_embeddings(doc_ids: list[str], matrices: dict[str, dict]) -> dict:
    """
    Write the corpus embedding matrices to the committed embeddings/ directory
    plus a small manifest, so the student notebook loads them (sub-second) instead
    of re-encoding the corpus (the Qwen encode is ~12 min on CPU).

    Args:
        matrices: {filename_stem: {"matrix": ndarray, "model": hf_id, "label": str}}
    """
    EMB_COMMIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collection": "movies-small",
        "n_docs": len(doc_ids),
        "doc_id_order": doc_ids,
        "note": "L2-normalised float32 corpus embeddings, row i corresponds to "
                "doc_id_order[i]. Load with numpy.load(...). Dot product equals cosine.",
        "files": {},
    }
    for stem, info in matrices.items():
        mat = info["matrix"].astype(np.float32)
        path = EMB_COMMIT_DIR / f"{stem}.npy"
        np.save(path, mat)
        manifest["files"][f"{stem}.npy"] = {
            "model": info["model"],
            "label": info["label"],
            "dim": int(mat.shape[1]),
            "shape": [int(mat.shape[0]), int(mat.shape[1])],
            "dtype": "float32",
            "mb": round(mat.nbytes / 1e6, 3),
        }
    (EMB_COMMIT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def matryoshka_truncation_sweep(queries: list[dict], model_large, emb_large: np.ndarray,
                                doc_ids: list[str], large_prompt: str | None,
                                dims: list[int], n_docs: int) -> dict:
    """
    Truncation-dim exhibit for the Matryoshka (large) embeddings.

    For every query, the full-dimension dense top-10 is the reference ranking.
    Each truncated dim re-ranks the whole collection on its prefix (query and docs
    truncated to that dim, then renormalised) and we measure recall@10 against the
    full-dim top-10: how many of the full top-10 the truncated ranking still finds.
    Dim equal to the full dimension is included as a 1.0 sanity check. Storage per
    dim is reported alongside, so quality and cost sit in one exhibit.
    """
    full_dim = emb_large.shape[1]
    # Encode each query once at full dimension, then truncate the same vector.
    kwargs = {"normalize_embeddings": True, "show_progress_bar": False}
    if large_prompt is not None:
        kwargs["prompt_name"] = large_prompt
    q_full = {q["id"]: np.asarray(model_large.encode([q["text"]], **kwargs)[0],
                                  dtype=np.float32) for q in queries}

    # Reference: full-dim dense top-10 per query.
    full_top10 = {}
    for q in queries:
        sims = emb_large @ q_full[q["id"]]
        full_top10[q["id"]] = {doc_ids[i] for i in np.argsort(-sims)[:10]}

    per_dim = {}
    for dim in dims:
        docs_t = P.truncate_normalize(emb_large, dim)
        recalls = []
        for q in queries:
            qv = q_full[q["id"]][:dim]
            qv = qv / (np.linalg.norm(qv) or 1.0)
            sims = docs_t @ qv
            top10 = {doc_ids[i] for i in np.argsort(-sims)[:10]}
            ref = full_top10[q["id"]]
            recalls.append(len(top10 & ref) / len(ref) if ref else 1.0)
        per_dim[str(dim)] = {
            "dim": dim,
            "pct_of_full_dim": round(100.0 * dim / full_dim, 1),
            "mean_recall_at_10_vs_full": round(float(np.mean(recalls)), 3),
            "bytes_per_doc_float32": dim * 4,
            "total_mb_float32": round(dim * 4 * n_docs / 1e6, 3),
            "storage_pct_of_full": round(100.0 * dim / full_dim, 1),
        }
    return {
        "model": LARGE_MODEL[1],
        "label": LARGE_MODEL[0],
        "full_dim": full_dim,
        "reference": "full-dimension dense top-10 per query",
        "metric": "mean recall@10 of the truncated top-10 against the full-dim top-10, "
                  "averaged over the query set",
        "n_queries": len(queries),
        "per_dim": per_dim,
    }


def main() -> None:
    t_start = time.time()

    # ─── Load collection ─────────────────────────────────────────────────────
    print("Loading movies-small collection...")
    collection = load_collection("movies-small")
    docs = collection.documents()
    doc_ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    id_to_doc = {d["id"]: d for d in docs}
    id_to_text = {d["id"]: d["text"] for d in docs}
    id_to_title = {d["id"]: d["title"] for d in docs}
    n_docs = len(docs)
    avg_tokens = float(np.mean([len(t.split()) for t in texts]))
    print(f"  {n_docs} movies, mean {avg_tokens:.0f} whitespace tokens per movie")

    # ─── Build indexes (one-time cost) ───────────────────────────────────────
    print("Building BM25 index...")
    t0 = time.time()
    bm25_index = P.build_bm25_index(docs)
    bm25_build_s = time.time() - t0
    print(f"  BM25: {bm25_index['n_terms']:,} terms in {bm25_build_s:.2f}s")

    print("Loading embedding models (first load pulls from local cache)...")
    from sentence_transformers import SentenceTransformer, CrossEncoder

    small_label, small_id, small_prompt, _ = SMALL_MODEL
    large_label, large_id, large_prompt, _ = LARGE_MODEL

    model_small = SentenceTransformer(small_id)
    model_large = SentenceTransformer(large_id)
    cross_encoder = CrossEncoder(CROSS_ENCODER_ID)

    print(f"Encoding corpus with {small_label}...")
    emb_small, enc_small_s, hit_s = encode_or_load(model_small, small_label, texts)
    dim_small = emb_small.shape[1]
    print(f"  {small_label}: dim {dim_small}, {enc_small_s:.2f}s"
          f"{' (cached)' if hit_s else ''}")

    print(f"Encoding corpus with {large_label} (slow on CPU, first run only)...")
    emb_large, enc_large_s, hit_l = encode_or_load(model_large, large_label, texts)
    dim_large = emb_large.shape[1]
    print(f"  {large_label}: dim {dim_large}, {enc_large_s:.2f}s"
          f"{' (cached)' if hit_l else ''}")

    # Export committed embeddings so the student notebook skips the slow encode.
    manifest = export_committed_embeddings(doc_ids, {
        "minilm_movies_small": {"matrix": emb_small, "model": small_id, "label": small_label},
        "qwen_movies_small": {"matrix": emb_large, "model": large_id, "label": large_label},
    })
    sizes = ", ".join(f"{k} {v['mb']}MB" for k, v in manifest["files"].items())
    print(f"  committed embeddings -> {EMB_COMMIT_DIR} ({sizes})")

    # ─── Define conditions ───────────────────────────────────────────────────
    # Each condition: (name, callable(query) -> list[(doc_id, score)])
    conditions: dict[str, callable] = {
        "A-bm25":
            lambda q: P.bm25_search(q, bm25_index, top_k=TOP_K),
        "B-bm25+crossenc":
            lambda q: P.bm25_cross_encoder(q, bm25_index, id_to_text, cross_encoder,
                                           retrieve_k=RETRIEVE_K, top_k=TOP_K),
        "C-dense-small":
            lambda q: P.dense_search(q, model_small, emb_small, doc_ids,
                                     prompt_name=small_prompt, top_k=TOP_K),
        "C-dense-large":
            lambda q: P.dense_search(q, model_large, emb_large, doc_ids,
                                     prompt_name=large_prompt, top_k=TOP_K),
        "D-hybrid-small":
            lambda q: P.hybrid_rrf(q, bm25_index, model_small, emb_small, doc_ids,
                                   prompt_name=small_prompt, retrieve_k=RETRIEVE_K,
                                   top_k=TOP_K),
        "D-hybrid-large":
            lambda q: P.hybrid_rrf(q, bm25_index, model_large, emb_large, doc_ids,
                                   prompt_name=large_prompt, retrieve_k=RETRIEVE_K,
                                   top_k=TOP_K),
        "D-hybrid-large+crossenc":
            lambda q: P.hybrid_rrf(q, bm25_index, model_large, emb_large, doc_ids,
                                   prompt_name=large_prompt, retrieve_k=RETRIEVE_K,
                                   id_to_text=id_to_text, cross_encoder=cross_encoder,
                                   top_k=TOP_K),
        "E-matryoshka-large":
            lambda q: P.multistage_matryoshka(q, model_large, emb_large, doc_ids,
                                              prompt_name=large_prompt,
                                              first_dim=MATRYOSHKA_FIRST_DIM,
                                              first_k=MATRYOSHKA_FIRST_K, top_k=TOP_K),
        "E-matryoshka-large+crossenc":
            lambda q: P.multistage_matryoshka(q, model_large, emb_large, doc_ids,
                                              prompt_name=large_prompt,
                                              first_dim=MATRYOSHKA_FIRST_DIM,
                                              first_k=MATRYOSHKA_FIRST_K,
                                              id_to_text=id_to_text,
                                              cross_encoder=cross_encoder, top_k=TOP_K),
    }

    queries = load_queries()

    # ─── Run every condition over every query ────────────────────────────────
    print(f"\nRunning {len(conditions)} conditions over {len(queries)} queries...")
    raw_results: dict[str, dict] = {}
    latency: dict[str, list[float]] = {name: [] for name in conditions}

    for q in queries:
        qid, qtext = q["id"], q["text"]
        raw_results[qid] = {"text": qtext, "probe": q["probe"], "conditions": {}}
        for name, fn in conditions.items():
            t0 = time.time()
            ranked = fn(qtext)
            latency[name].append(time.time() - t0)
            raw_results[qid]["conditions"][name] = [
                {"rank": i + 1, "id": doc_id, "title": id_to_title.get(doc_id, "?"),
                 "score": round(score, 4)}
                for i, (doc_id, score) in enumerate(ranked)
            ]
        print(f"  {qid} done")

    # ─── Build the pool to judge (deduplicated query/movie pairs) ────────────
    pool: dict[tuple[str, str], dict] = {}
    for qid, qdata in raw_results.items():
        for name, ranked in qdata["conditions"].items():
            for entry in ranked:
                key = (qid, entry["id"])
                if key not in pool:
                    doc = id_to_doc[entry["id"]]
                    pool[key] = {
                        "query_id": qid,
                        "query_text": qdata["text"],
                        "movie_id": entry["id"],
                        "title": doc["title"],
                        "year": doc.get("year", 0),
                        "genres": doc.get("genres", []),
                        "context": short_context(doc),
                        "found_by": [],
                    }
                pool[key]["found_by"].append(name)
    pool_list = sorted(pool.values(), key=lambda r: (r["query_id"], r["title"]))

    # ─── Costs: storage and latency ──────────────────────────────────────────
    def storage_block(dim: int) -> dict:
        return {
            "dim": dim,
            "bytes_per_doc_float32": dim * 4,
            "total_bytes_float32": dim * 4 * n_docs,
            "total_mb_float32": round(dim * 4 * n_docs / 1e6, 3),
        }

    matryoshka_storage = {}
    for dim in TRUNCATION_DIMS:
        matryoshka_storage[str(dim)] = {
            "bytes_per_doc_float32": dim * 4,
            "total_bytes_float32": dim * 4 * n_docs,
            "total_mb_float32": round(dim * 4 * n_docs / 1e6, 3),
            "pct_of_full": round(100.0 * dim / dim_large, 1),
        }
    # Quantization of the full large vector (storage only; quality not measured here).
    quant = {
        "float32_full": {"bytes_per_doc": dim_large * 4,
                         "total_mb": round(dim_large * 4 * n_docs / 1e6, 3)},
        "int8_full": {"bytes_per_doc": dim_large * 1,
                      "total_mb": round(dim_large * 1 * n_docs / 1e6, 3),
                      "reduction_vs_float32": "4x"},
        "binary_full": {"bytes_per_doc": dim_large // 8,
                        "total_mb": round((dim_large // 8) * n_docs / 1e6, 3),
                        "reduction_vs_float32": "32x"},
    }

    costs = {
        "collection": {"name": "movies-small", "n_docs": n_docs,
                       "mean_tokens_per_doc": round(avg_tokens, 1)},
        "index_build": {
            "bm25": {"build_s": round(bm25_build_s, 3), "n_terms": bm25_index["n_terms"],
                     "storage_note": "sparse inverted index; dominated by postings, "
                                     "not stored as dense vectors"},
            small_label: {"encode_s": round(enc_small_s, 3), **storage_block(dim_small)},
            large_label: {"encode_s": round(enc_large_s, 3), **storage_block(dim_large)},
        },
        "matryoshka_storage_large_model": matryoshka_storage,
        "quantization_large_model": quant,
        "per_query_latency_ms": {name: round(mean_ms(times), 2)
                                 for name, times in latency.items()},
    }

    # ─── Divergence: top-k overlap between conditions ────────────────────────
    cond_names = list(conditions.keys())

    def topk_set(qid: str, name: str, k: int = TOP_K) -> set[str]:
        return {e["id"] for e in raw_results[qid]["conditions"][name][:k]}

    # Mean Jaccard overlap of top-10 sets across all queries, per condition pair.
    pair_overlap = {}
    for a, b in combinations(cond_names, 2):
        jaccs = []
        for qid in raw_results:
            sa, sb = topk_set(qid, a), topk_set(qid, b)
            union = sa | sb
            jaccs.append(len(sa & sb) / len(union) if union else 1.0)
        pair_overlap[f"{a} vs {b}"] = round(float(np.mean(jaccs)), 3)

    # Per-query top-1 agreement across conditions and small-vs-large top-1.
    per_query = {}
    for q in queries:
        qid = q["id"]
        top1 = {name: (raw_results[qid]["conditions"][name][0]["title"]
                       if raw_results[qid]["conditions"][name] else "(none)")
                for name in cond_names}
        distinct_top1 = sorted(set(top1.values()))
        per_query[qid] = {
            "text": q["text"],
            "probe": q["probe"],
            "open": bool(q.get("open", False)),
            "top1_by_condition": top1,
            "n_distinct_top1": len(distinct_top1),
        }

    divergence = {
        "conditions": cond_names,
        "mean_jaccard_top10_between_condition_pairs": pair_overlap,
        "per_query": per_query,
    }

    # ─── Matryoshka truncation-dim sweep (quality vs storage exhibit) ─────────
    print("Computing Matryoshka truncation sweep...")
    sweep = matryoshka_truncation_sweep(
        queries, model_large, emb_large, doc_ids, large_prompt, TRUNCATION_DIMS, n_docs)

    # ─── Write artifacts ─────────────────────────────────────────────────────
    (HERE / "matryoshka_sweep.json").write_text(
        json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "raw_results.json").write_text(
        json.dumps(raw_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "pool_to_judge.json").write_text(
        json.dumps({"n_pairs": len(pool_list), "pairs": pool_list},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "costs.json").write_text(
        json.dumps(costs, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "divergence.json").write_text(
        json.dumps(divergence, indent=2, ensure_ascii=False), encoding="utf-8")

    # ─── Console summary ─────────────────────────────────────────────────────
    print("\n=== PER-QUERY TOP-1 (A / C-small / C-large / D-large) ===")
    for q in queries:
        qid = q["id"]
        c = raw_results[qid]["conditions"]
        def t1(name):
            lst = c[name]
            return lst[0]["title"] if lst else "(none)"
        print(f"{qid} [{q['probe']:>20}] {q['text'][:48]:<48} | "
              f"A: {t1('A-bm25'):<22} | Csm: {t1('C-dense-small'):<22} | "
              f"Clg: {t1('C-dense-large'):<22} | Dlg: {t1('D-hybrid-large')}")

    print("\n=== MEAN JACCARD top-10 overlap (selected pairs) ===")
    for pair in ["A-bm25 vs C-dense-small", "A-bm25 vs C-dense-large",
                 "C-dense-small vs C-dense-large", "A-bm25 vs D-hybrid-small",
                 "C-dense-small vs D-hybrid-small", "A-bm25 vs B-bm25+crossenc",
                 "D-hybrid-large vs D-hybrid-large+crossenc",
                 "E-matryoshka-large vs C-dense-large",
                 "E-matryoshka-large vs E-matryoshka-large+crossenc"]:
        if pair in pair_overlap:
            print(f"  {pair:<50} {pair_overlap[pair]}")

    print("\n=== PER-QUERY LATENCY (mean ms) ===")
    for name in cond_names:
        print(f"  {name:<32} {mean_ms(latency[name]):8.1f} ms")

    print("\n=== STORAGE ===")
    print(f"  BM25 build: {bm25_build_s:.2f}s, {bm25_index['n_terms']:,} terms")
    print(f"  {small_label}: encode {enc_small_s:.1f}s, "
          f"{dim_small*4} B/doc, {dim_small*4*n_docs/1e6:.2f} MB total")
    print(f"  {large_label}: encode {enc_large_s:.1f}s, "
          f"{dim_large*4} B/doc, {dim_large*4*n_docs/1e6:.2f} MB total")
    for dim in TRUNCATION_DIMS:
        s = matryoshka_storage[str(dim)]
        print(f"    trunc {dim:>4}d: {s['total_mb_float32']:.3f} MB "
              f"({s['pct_of_full']}% of full)")

    print("\n=== MATRYOSHKA TRUNCATION SWEEP (large model, recall@10 vs full dim) ===")
    for dim in TRUNCATION_DIMS:
        d = sweep["per_dim"][str(dim)]
        print(f"  {dim:>4}d: recall@10 {d['mean_recall_at_10_vs_full']:.3f}  "
              f"storage {d['total_mb_float32']:.3f} MB ({d['storage_pct_of_full']}% of full)")

    print(f"\nArtifacts written to {HERE}")
    print(f"Total trial time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
