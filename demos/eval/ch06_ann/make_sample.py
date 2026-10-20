"""
Write the small embedding sample that the ch06 notebook uses for live experiments.

15,000 random TREC-COVID abstracts and the 50 topic queries, Qwen3 embeddings
truncated to 256 dimensions and stored as float16 (about 8 MB, committed), so
the notebook can build IVF, PQ, and LSH indexes on a laptop CPU in seconds.

    python -m eval.ch06_ann.make_sample
"""

import json
from pathlib import Path

import numpy as np

from shared.collections import load_collection
from eval.ch06_ann.run_benchmark import EMB_DIR
from ch06_ann import truncate

HERE = Path(__file__).parent
N_SAMPLE, DIM = 15_000, 256


def main():
    ids = load_collection("trec-covid").ids()
    docs = np.load(EMB_DIR / "docs_qwen3_1024_fp16.npy").astype(np.float32)
    queries = np.load(EMB_DIR / "queries_qwen3_1024.npy")
    pick = np.sort(np.random.default_rng(0).choice(len(ids), N_SAMPLE, replace=False))
    np.savez_compressed(HERE / "sample_256.npz",
                        docs=truncate(docs[pick], DIM).astype(np.float16),
                        queries=truncate(queries, DIM).astype(np.float16),
                        doc_ids=np.array([ids[i] for i in pick]))
    print(f"wrote {N_SAMPLE:,} x {DIM} sample: {(HERE / 'sample_256.npz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
