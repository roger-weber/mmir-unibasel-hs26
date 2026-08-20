---
author: Roger Weber
edition: HS26
status: updated
book_part: Foundations
chapter: Performance Evaluation
section: Precision and Recall
order: "2.2"
---

(performance-evaluation-precision-recall)=
# Precision and Recall

Boolean retrieval returns a set of documents with no order attached: a document either matches the query or it does not. This section evaluates that kind of result. Ranking comes in [](#performance-evaluation-ranked-results); until then, a result list is treated purely as a set.

Recall System A and System B from [](#performance-evaluation-benchmark-design), both answering the same need: which books give a beginning master's student solid computer science foundations? System A returned 25 books, System B returned 8. To turn "returned" and "useful" into numbers, every book in the 50-book collection falls into exactly one of four groups, depending on whether the system retrieved it and whether it is actually relevant.

## Counting Outcomes

| | Relevant | Not relevant |
|---|---|---|
| **Retrieved** | True Positive (TP) | False Positive (FP) |
| **Not retrieved** | False Negative (FN) | True Negative (TN) |

A true positive is a relevant book the system retrieved: a hit. A false positive is a book the system retrieved that turns out not to be relevant: noise in the result. A false negative is a relevant book the system failed to retrieve: a miss. A true negative is a book the system correctly left out. [Figure %s](#fig-retrieval-confusion-sets) shows this partition as overlapping sets for both systems: the 50-book collection, the 15 relevant books, the retrieved books, and their intersection.

```{figure} images/figure_2_3.png
:name: fig-retrieval-confusion-sets
:width: 60%

Set-theoretic view of retrieval performance for Systems A and B over the 50-book collection.
```

Counting each book in the collection against the two systems gives:

| | TP | FP | FN | TN |
|---|---|---|---|---|
| System A | 12 | 13 | 3 | 22 |
| System B | 6 | 2 | 9 | 33 |

Every row sums to 50, the size of the collection, and every count came directly from checking the 15 known relevant books against each system's result list.

## Precision and Recall

The raw counts alone are hard to compare. System A has 12 true positives and System B has 6, but A also retrieved three times as many books and operates in a collection where 15 are relevant. A system retrieving the entire library would produce 15 true positives and look best by that count alone, which shows that an absolute count cannot separate a good system from one that simply returns everything. Ratios solve this: dividing by the appropriate total produces a value between 0 and 1 that is independent of how many books the system returned or how many relevant books the collection contains.

```{admonition} Key Formula: Precision and Recall
:class: important

$$p = \frac{TP}{TP + FP} \qquad r = \frac{TP}{TP + FN}$$

Precision is the fraction of retrieved documents that are relevant. Recall is the fraction of relevant documents that were retrieved.
```

```{admonition} Example: Precision and Recall for System A and B
:class: example

System A retrieved 25 books and found 12 of the 15 relevant ones:

$$p_A = \frac{12}{12+13} = \frac{12}{25} = 48\% \qquad r_A = \frac{12}{12+3} = \frac{12}{15} = 80\%$$

System B retrieved 8 books and found 6 of the 15 relevant ones:

$$p_B = \frac{6}{6+2} = \frac{6}{8} = 75\% \qquad r_B = \frac{6}{6+9} = \frac{6}{15} = 40\%$$
```

This is the trade-off from [](#performance-evaluation-benchmark-design) stated in numbers rather than in words. System B's list is three-quarters useful material; System A's list finds four-fifths of everything useful but buries it among thirteen novels. Whether 48% precision at 80% recall is better than 75% precision at 40% recall still depends on what the user prefers (quick fact check or exhaustive library scan).

A third quantity, fallout, measures the opposite kind of mistake: how much of the irrelevant material got pulled in.

$$f = \frac{FP}{FP + TN}$$

Fallout is the fraction of non-relevant documents that the system retrieved. For our two systems, $f_A = 13/35 = 37.1\%$ and $f_B = 2/35 = 5.7\%$. Fallout measures the reading cost imposed on someone who inspects every result: the patent lawyer from Section 1 needs high recall, because missing a document is expensive, but also wants low fallout, because every irrelevant document in the list costs time and attention. System A delivers recall of 80% but pulls in more than a third of the irrelevant collection; the ideal for an exhaustive search would push recall toward 1 while keeping fallout near 0.

```{admonition} Precision and recall are in tension
:class: warning

Retrieving more documents raises recall, since additional relevant ones may be included, and a system that returns the entire collection reaches perfect recall by construction. The cost is that more non-relevant documents come in as well, which lowers precision. In the other direction, a smaller result set tends to have higher precision, but at the risk of missing many of the relevant documents. In practice, improving one measure typically comes at the expense of the other, which is why both are always reported together.
```

## Balancing the Two

Comparing two numbers side by side, as above, does not say which system wins. The $F_\beta$-measure collapses precision and recall into a single value, letting a scenario's priority set the trade-off explicitly.

```{admonition} Key Formula: F-beta Measure
:class: important

$$F_\beta = \frac{(\beta^2 + 1) \cdot p \cdot r}{\beta^2 \cdot p + r}$$

The weighted harmonic mean of precision and recall. $\beta < 1$ favours precision, $\beta > 1$ favours recall, and $\beta = 1$ weighs them equally.
```

At $\beta = 0$ the formula reduces to precision alone; as $\beta \to \infty$ it reduces to recall alone. $F_1$, the case $\beta = 1$, is the most common default and appears throughout machine learning wherever a single number is needed to compare classifiers.

```{admonition} Example: F-beta for System A and B
:class: example

Using $p_A = 0.48$, $r_A = 0.80$ and $p_B = 0.75$, $r_B = 0.40$ from above:

| $\beta$ | Weighting | $F_\beta(A)$ | $F_\beta(B)$ | Winner |
|---|---|---|---|---|
| 0.5 | favours precision | 0.522 | 0.638 | B |
| 1.0 | equal weight | 0.600 | 0.522 | A |
| 2.0 | favours recall | 0.706 | 0.441 | A |
```

The winner changes between $\beta = 0.5$ and $\beta = 1$. Neither system is better in general: a fact-checker who wants a short, clean list would set $\beta$ below 1 and prefer System B, while a patent lawyer would set $\beta$ above 1 and prefer System A. $F_1$ is a convenient default, not a neutral one, since it silently commits to weighing a false positive and a false negative equally.

## Averaging Across Needs

One need produces one precision and one recall value. A real benchmark evaluates many needs, and summarising all of them into a single score requires an averaging method. The natural first idea is to compute precision and recall for each need separately and then take the arithmetic mean.

Example: Consider two additional needs alongside the CS-foundations query, answered by the same two systems:

| Need | Relevant | System A: retrieved / found | System B: retrieved / found |
|---|---|---|---|
| CS foundations | 15 | 25 / 12 | 8 / 6 |
| Stage plays | 5 | 6 / 4 | 4 / 3 |
| Russian novels | 1 | 3 / 0 | 1 / 1 |

The straightforward summary is the arithmetic mean of the per-need values:

```{admonition} Key Formula: Macro Averaging
:class: important

$$p_{\text{macro}} = \frac{1}{N}\sum_{i=1}^{N} p_i \qquad r_{\text{macro}} = \frac{1}{N}\sum_{i=1}^{N} r_i$$

Every need counts equally, regardless of how many relevant documents it has.
```

```{admonition} Example: Macro Averaging
:class: example

Per-need recall for both systems:

| Need | $r_A$ | $r_B$ |
|---|---|---|
| CS foundations | 12/15 = 0.800 | 6/15 = 0.400 |
| Stage plays | 4/5 = 0.800 | 3/5 = 0.600 |
| Russian novels | 0/1 = 0.000 | 1/1 = 1.000 |

$$r_{\text{macro},A} = \frac{0.800 + 0.800 + 0.000}{3} = 53.3\%$$

$$r_{\text{macro},B} = \frac{0.400 + 0.600 + 1.000}{3} = 66.7\%$$

System B wins. Yet looking at the table, System A finds more relevant documents on two of the three needs and retrieves 16 of 21 relevant books in total, against B's 10. The single Russian-novels miss, a zero out of one, drags A's average down by 27 percentage points.
```

The problem is visible: macro-averaging gives each need the same weight $1/N$ in the sum, so a need with one relevant document contributes as much to the average as a need with fifteen. A single zero from the Russian-novels miss lowers the three-need average by a full $1/3$, which is a large negative impact relative to the system's overall performance across 21 relevant documents. This is intentional when every need matters equally, but it can misrepresent a system's aggregate capability.

Micro-averaging addresses this by pooling all outcome counts before computing the ratio. Instead of averaging per-need ratios, it sums the numerators and denominators across needs:

$$p_{\text{micro}} = \frac{\sum_{i=1}^{N} TP_i}{\sum_{i=1}^{N} (TP_i + FP_i)} \qquad r_{\text{micro}} = \frac{\sum_{i=1}^{N} TP_i}{\sum_{i=1}^{N} (TP_i + FN_i)}$$

Every document counts equally. Needs with more relevant documents contribute more to the result.

```{admonition} Example: Micro Averaging
:class: example

Pooling the recall counts across all three needs:

| | $\sum TP$ | $\sum (TP + FN)$ | $r_{\text{micro}}$ |
|---|---|---|---|
| System A | 12 + 4 + 0 = 16 | 15 + 5 + 1 = 21 | 16/21 = 76.2% |
| System B | 6 + 3 + 1 = 10 | 15 + 5 + 1 = 21 | 10/21 = 47.6% |

System A now wins on recall, because the CS-foundations need, where A finds 12 of 15 books, contributes 15 of the 21 relevant documents to the pool and dominates the sum. The single Russian-novels miss adds only one false negative to a denominator of 21 and barely registers.
```

Same systems, same needs, opposite conclusions on recall: macro says B is better, micro says A is better. The difference reflects a genuine choice about what matters.

```{admonition} Macro averaging is the standard in retrieval evaluation
:class: warning

In retrieval evaluation, the standard practice is macro-averaging: MAP, mean nDCG, and mean MRR all compute a per-topic score and then take the arithmetic mean. Micro-averaging across topics is rarely used, because weighting each document equally would let one large topic dominate all others. The macro/micro choice becomes more relevant in classification evaluation, where classes are often imbalanced and the two methods can disagree sharply; Section 5 returns to this point for the intent-routing example. When reporting retrieval results, specify both the measure and the number of topics it was averaged over.
```


## Architecture and the Metrics

[](#classical-text-fundamental-flows) introduced retriever-only, retriever-with-filter, and retriever-ranker architectures. Precision and recall explain why each stage exists and how each is evaluated. The retriever is optimized for recall: it casts a wide net over the collection, accepting low precision because a false positive can still be removed later, while a false negative is lost for good. The ranker or filter is optimized for precision: it reorders or removes candidates so that the top of the list is dense with relevant material. Even a user who cares only about a precise top-ten result depends on the retriever's recall, because the ranker can only surface documents that the retriever found in the first place.

```{admonition} Hands-on: Precision and Recall
:class: hint
Recompute every number in this section from the library collection, then edit a run and watch precision, recall, and $F_\beta$ respond.
[Open notebook -&gt;](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch02-01-performance-evaluation.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```

Precision, recall, and their averages describe a result set with no notion of order. The next section brings rank back into the picture: whether a system found the right documents matters less if they never appear near the top of the list.
