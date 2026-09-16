---
author: Roger Weber
edition: HS26
status: finished
book_part: Foundations
chapter: Performance Evaluation
section: Summary
order: "2.8"
---

# Summary

## Metric Lookup Table

| Metric | Formula | What it rewards | When to use |
|--------|---------|----------------|-------------|
| Precision | $\frac{TP}{TP+FP}$ | Clean results (few false positives) | Any binary decision; especially when FP is costly |
| Recall | $\frac{TP}{TP+FN}$ | Completeness (few misses) | Exhaustive search; patent, legal, medical |
| $F_\beta$ | $\frac{(\beta^2+1) \cdot p \cdot r}{\beta^2 \cdot p + r}$ | Balances precision and recall with tuneable weight | Single-number summary; $\beta=1$ default |
| $P@k$ | Precision in top $k$ | Clean top-of-list | Web search, short result pages |
| MRR | Mean of $1/\text{rank}_{\text{first relevant}}$ | First relevant result high | QA, known-item search |
| R-Precision | $p@R$ | Balance at a natural cutoff | Self-adjusting single-point summary |
| AP | Mean of $p_k$ at relevant positions, divided by $|\text{Rel}|$ | Both high precision and high recall, position-weighted | Full ranking evaluation per query |
| MAP | Mean of per-query AP | System-level ranking quality | Benchmark comparison (TREC standard) |
| $nDCG@k$ | $DCG_k / IDCG_k$ | High-grade documents placed early | Web search, graded relevance, user experience |
| Accuracy | $\frac{TP+TN}{P+N}$ | Overall correct decisions | Balanced classification; equals micro-precision and micro-recall |
| Specificity (TNR) | $\frac{TN}{TN+FP}$ | Few false alarms | Medical tests, biometric security |
| AUC | Area under ROC curve | Score separation across all thresholds | Comparing classifiers without committing to a threshold |

## Key Takeaways

1. Evaluation requires a benchmark: a collection, information needs, relevance judgments, and a performance goal. Without any component, a number is not interpretable.
2. Precision and recall are in tension. Improving one typically costs the other, and the $F_\beta$-measure makes the trade-off explicit.
3. Ranked evaluation adds position: AP and nDCG reward systems that place relevant (or high-grade) material where users will see it.
4. Graded relevance distinguishes "somewhat useful" from "essential". nDCG captures what binary metrics cannot: the quality of the relevant documents, not only their presence.
5. Classification evaluation reuses precision and recall but adds accuracy, specificity, and the confusion matrix. Accuracy equals micro-precision equals micro-recall in single-label settings.
6. Prevalence distorts accuracy: when one class dominates, a trivial baseline can score higher than a real classifier. Always check class balance before trusting accuracy.
7. A continuous score becomes a binary decision only after choosing a threshold. The ROC curve shows all possible trade-offs; AUC summarizes overall score separation.
8. No benchmark score is a claim about generalization. Incomplete judgments, data contamination, prevalence mismatch, and overfitting to static test sets all limit what a number can mean.

## Key Formulas

$$p = \frac{TP}{TP+FP} \qquad r = \frac{TP}{TP+FN}$$

Precision and recall: the fundamental pair from which all other metrics derive.

$$AP = \frac{1}{|\text{Rel}|} \sum_{k: d_k \in \text{Rel}} p_k$$

Average precision: area under the precision-recall curve, the standard per-query measure.

$$nDCG_k = \frac{\sum_{i=1}^k \frac{rel_i}{\log_2(i+1)}}{\text{IDCG}_k}$$

Normalized discounted cumulative gain: the standard for graded relevance evaluation.

$$AUC = P(s(x_+) > s(x_-))$$

Area under the ROC curve: the probability that a random positive scores higher than a random negative.

```{attention} Exam focus
- The difference between precision and recall, and why both are always reported together
- How AP integrates both precision and recall in one number, and why missed documents hurt AP
- Why nDCG and AP can disagree on which system is better (top-k user experience vs. full-list completeness)
- Why accuracy misleads under class imbalance, and how prevalence affects precision
- How moving a decision threshold trades TPR against FPR, and what the ROC curve visualizes
- The difference between macro-averaging (each query/class counts equally) and micro-averaging (each item counts equally)
```

## Self-Check Questions

1. (Understand) A system achieves 95% accuracy on a dataset where 95% of items belong to class A. Is this system useful? What additional metric would reveal the problem?
2. (Analyze) System X has MAP = 0.60 and System Y has MAP = 0.55, both measured on 50 queries. Under what circumstances might Y actually be the better system?
3. (Evaluate) A face-unlock system reports AUC = 0.99 and a spam filter reports AUC = 0.99. Are they equally good? What additional information do you need to judge each for its intended use?
4. (Apply) Given a ranked list of 10 documents where relevant documents appear at positions 1, 4, and 7 (out of 8 relevant in the collection), compute $P@5$, AP, and explain why AP is less than the average of the three precision values.
5. (Analyze) Explain why macro-averaged recall across 4 information needs can disagree with micro-averaged recall, and give a concrete scenario where the disagreement is large.

```{hint} Test Your Knowledge
[Take the Chapter 2 Quiz →](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/)
```

## Further Reading

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). **Introduction to information retrieval**, Chapter 8: Evaluation in information retrieval. Cambridge University Press. [Read online](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html). A concise foundation for precision, recall, MAP, and benchmark design.
- Järvelin, K., & Kekäläinen, J. (2002). **Cumulated gain-based evaluation of IR techniques**. *ACM Transactions on Information Systems*, 20(4), 422–446. [ACM record](https://dl.acm.org/doi/10.1145/582415.582418). The paper that introduced DCG and nDCG for graded relevance evaluation.
- Fawcett, T. (2006). **An introduction to ROC analysis**. *Pattern Recognition Letters*, 27(8), 861–874. [Read the paper](https://people.inf.elte.hu/kiss/11dwhdm/roc.pdf). A practical guide to ROC curves, AUC, and threshold selection for binary classifiers.
- Armstrong, T. G., Moffat, A., Webber, W., & Zobel, J. (2009). **Improvements that don't add up: Ad-hoc retrieval results since 1998**. In *Proceedings of CIKM 2009* (pp. 601–610). [ACM record](https://dl.acm.org/doi/10.1145/1645953.1646031). An influential warning that static benchmarks can conceal a lack of cumulative progress.
- **[TREC (Text REtrieval Conference)](https://trec.nist.gov/)**. NIST’s continuing evaluation campaigns provide the pooled collections, shared topics, and comparative methodology behind much of this chapter’s retrieval evaluation practice.
