"""
LLM relevance judge for the chapter 6 ANN benchmark.

TREC-COVID judgments are deep but not complete: approximate retrievers surface
documents that no TREC assessor ever saw. This script grades those documents with
an LLM judge on the TREC-COVID scale and first measures how well the judge agrees
with the human assessors.

    calibration.json       judge vs. TREC on 300 already judged pairs (100 per grade)
    judgments_llm.json     {query_id: {doc_id: grade}} for the pool of unjudged pairs
    judgments_llm_detail.jsonl   one line per judged pair, with the judge's reason

Run after run_benchmark.py, from the demos/ directory (needs AWS Bedrock access):

    python -m eval.ch06_ann.judge
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from shared.collections import load_collection
from shared.assessments import load_assessments
from shared.llm import invoke_claude, JUDGE_MODEL_ID

HERE = Path(__file__).parent
CACHE_FILE = HERE / "judgments_llm_detail.jsonl"
N_PER_GRADE = 100
WORKERS = 16
CAL_TAG = "calibration-v2"   # v1 (strict prompt) kept in the cache for comparison

SYSTEM = """You are an expert relevance assessor for the TREC-COVID track, judging \
scientific articles from the CORD-19 collection against COVID-19 information needs.

Grade the article on this scale, using the TREC-COVID definitions:
2 = relevant: the article is fully responsive to the information need described by \
the question and the narrative.
1 = partially relevant: the article answers part of the question, or would be useful \
in combination with other articles.
0 = not relevant: everything else, including articles on the general topic that do not \
address the specific information need.

How to read the narrative: it lists the range of aspects that count as relevant. An article does not need to cover all of them. An article that substantially addresses one aspect of the information need is relevant (2). An article that touches the information need only in passing, or addresses a closely related question (for example another coronavirus, or COVID-19 in general with some useful findings on the need), is partially relevant (1). Reserve 0 for articles that would not help a researcher with this information need at all.

Judge only from the title and abstract shown. Reply with JSON only, in the form \
{"grade": <0, 1 or 2>, "reason": "<one short sentence>"}."""


def prompt(topic, doc) -> str:
    abstract = doc["abstract"][:4000] or "(no abstract available)"
    return (f"Question: {topic.question}\n"
            f"Keyword query: {topic.keywords}\n"
            f"Narrative: {topic.narrative}\n\n"
            f"Article title: {doc['title']}\n"
            f"Article abstract: {abstract}")


def judge_one(topic, doc, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            reply = invoke_claude(prompt(topic, doc), system=SYSTEM, max_tokens=1024,
                                  model_id=JUDGE_MODEL_ID)
            m = re.search(r"\{.*\}", reply, re.S)
            data = json.loads(m.group(0))
            grade = int(data["grade"])
            if grade in (0, 1, 2):
                return {"grade": grade, "reason": str(data.get("reason", ""))}
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"judge failed for topic {topic.id}, doc {doc['id']}")


def judge_pairs(pairs, collection, assessments, tag: str) -> dict:
    """Judge (query, doc) pairs; results are cached line by line and reused."""
    done = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["query"], r["doc"], r["tag"])] = r
    todo = [(q, d) for q, d in pairs if (q, d, tag) not in done]

    def work(pair):
        q, d = pair
        r = judge_one(assessments.topics[q], collection[d])
        return {"query": q, "doc": d, "tag": tag, **r}

    with ThreadPoolExecutor(WORKERS) as pool, open(CACHE_FILE, "a", encoding="utf-8") as f:
        for i, r in enumerate(pool.map(work, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[(r["query"], r["doc"], tag)] = r
            if i % 200 == 0:
                print(f"  {tag}: {i}/{len(todo)} judged")
    return {(q, d): done[(q, d, tag)]["grade"] for q, d in pairs}


def cohen_kappa(a, b, labels=(0, 1, 2), weighted: bool = False) -> float:
    a, b = np.asarray(a), np.asarray(b)
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[labels.index(x), labels.index(y)] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = (np.abs(np.subtract.outer(range(k), range(k))) / (k - 1)) if weighted else (1 - np.eye(k))
    return float(1 - (w * obs).sum() / (w * exp).sum())


def calibrate(collection, assessments) -> dict:
    rng = random.Random(0)
    by_grade = {g: [] for g in (0, 1, 2)}
    for q, labels in assessments.qrels.items():
        for d, g in labels.items():
            if d in collection:
                by_grade[g].append((q, d))
    sample = [p for g in (0, 1, 2) for p in rng.sample(sorted(by_grade[g]), N_PER_GRADE)]
    judged = judge_pairs(sample, collection, assessments, tag=CAL_TAG)
    human = [assessments.grade(q, d) for q, d in sample]
    llm = [judged[p] for p in sample]
    confusion = [[sum(1 for h, l in zip(human, llm) if h == i and l == j) for j in (0, 1, 2)]
                 for i in (0, 1, 2)]
    hb, lb = [int(h > 0) for h in human], [int(l > 0) for l in llm]
    result = {
        "judge_model": JUDGE_MODEL_ID, "n": len(sample), "per_grade": N_PER_GRADE,
        "confusion_trec_rows_llm_cols": confusion,
        "exact_agreement": float(np.mean(np.array(human) == np.array(llm))),
        "kappa": cohen_kappa(human, llm),
        "kappa_linear_weighted": cohen_kappa(human, llm, weighted=True),
        "binary_agreement": float(np.mean(np.array(hb) == np.array(lb))),
        "binary_kappa": cohen_kappa(hb, lb, labels=(0, 1)),
    }
    (HERE / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    collection = load_collection("trec-covid")
    assessments = load_assessments("trec-covid")
    cal = calibrate(collection, assessments)
    print(f"calibration: exact {cal['exact_agreement']:.2f}, kappa {cal['kappa']:.2f}, "
          f"weighted kappa {cal['kappa_linear_weighted']:.2f}, binary kappa {cal['binary_kappa']:.2f}")

    pool = [tuple(p) for p in json.loads((HERE / "pool_to_judge.json").read_text(encoding="utf-8"))["pairs"]]
    judged = judge_pairs(pool, collection, assessments, tag="pool")
    out: dict[str, dict[str, int]] = {}
    for (q, d), g in sorted(judged.items()):
        out.setdefault(q, {})[d] = g
    (HERE / "judgments_llm.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    grades = list(judged.values())
    print(f"judged {len(grades):,} pool pairs: " + ", ".join(f"grade {g}: {grades.count(g)}" for g in (0, 1, 2)))


if __name__ == "__main__":
    main()
