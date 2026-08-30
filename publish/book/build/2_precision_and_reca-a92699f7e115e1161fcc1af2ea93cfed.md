---
author: Roger Weber
edition: HS26
status: updated
part: Foundations
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

```{admonition} Example: Precision and Recall for Two Systems
:class: example

System A retrieved 25 books and found 12 of the 15 relevant ones:

$$p_A = \frac{12}{12+13} = \frac{12}{25} = 48\% \qquad r_A = \frac{12}{12+3} = \frac{12}{15} = 80\%$$

System B retrieved 8 books and found 6 of the 15 relevant ones:

$$p_B = \frac{6}{6+2} = \frac{6}{8} = 75\% \qquad r_B = \frac{6}{6+9} = \frac{6}{15} = 40\%$$
```

This is the trade-off from [](#performance-evaluation-benchmark-design) stated in numbers rather than in words. System B's list is three-quarters useful material; System A's list finds four-fifths of everything useful but buries it among thirteen novels. Whether 48% precision at 80% recall is better than 75% precision at 40% recall still depends on the need, exactly as it did before either number existed.

A third quantity, fallout, measures the opposite kind of mistake: how much of the irrelevant material got pulled in.

$$f = \frac{FP}{FP + TN}$$

Fallout is the fraction of non-relevant documents that the system retrieved anyway. For our two systems, $f_A = 13/35 = 37.1\%$ and $f_B = 2/35 = 5.7\%$: System A lets through six times the noise System B does, which is the direct cost of its higher recall.

```{admonition} Precision and recall move independently
:class: warning

A system can raise recall by retrieving more documents, but only by also retrieving more of the non-relevant ones, which lowers precision. The reverse holds for a system that retrieves fewer documents to raise precision. Neither number can be improved in isolation without changing what the system returns, which is why both are reported together rather than as a single score.
```

## Balancing the Two

Comparing two numbers side by side, as above, does not say which system wins. The $F_\beta$-measure collapses precision and recall into a single value, letting a scenario's priority set the trade-off explicitly.

```{admonition} Key Formula: F-beta Measure
:class: important

$$F_\beta = \frac{(\beta^2 + 1) \cdot p \cdot r}{\beta^2 \cdot p + r}$$

The weighted harmonic mean of precision and recall. $\beta < 1$ favours precision, $\beta > 1$ favours recall, and $\beta = 1$ weighs them equally.
```

At $\beta = 0$ the formula reduces to precision alone; as $\beta \to \infty$ it reduces to recall alone. $F_1$, the case $\beta = 1$, is the most common default and appears throughout machine learning wherever a single number is needed to compare classifiers.

```{admonition} Example: F-beta for Two Systems
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

One need produces one precision and one recall value. Real evaluation uses many needs, and averaging them takes two different forms that can disagree.

Add three more needs to the comparison: which books are stage plays, which were written by Margaret Atwood, and which nineteenth-century Russian novels the library holds. These have 5, 2, and 1 relevant books respectively.

| Need | Relevant | System A: retrieved / found | System B: retrieved / found |
|---|---|---|---|
| CS foundations | 15 | 25 / 12 | 8 / 6 |
| Stage plays | 5 | 6 / 4 | 4 / 3 |
| Margaret Atwood | 2 | 4 / 2 | 2 / 2 |
| Russian novels | 1 | 3 / 0 | 1 / 1 |

Macro-averaging treats every need as equally important, computing precision and recall per need and then averaging those values.

```{admonition} Key Formula: Macro and Micro Averaging
:class: important

$$p_{\text{macro}} = \frac{1}{N}\sum_{i=1}^{N} p_i \qquad p_{\text{micro}} = \frac{\sum_{i=1}^{N} TP_i}{\sum_{i=1}^{N} (TP_i + FP_i)}$$

Macro-averaging gives every need equal weight. Micro-averaging pools the outcome counts first, so larger needs dominate. Recall is computed the same way, replacing $FP_i$ with $FN_i$.
```

```{admonition} Example: Macro versus Micro Averaging
:class: example

System A finds none of the single relevant Russian novel, contributing a precision and recall of 0 for that need:

$$p_{\text{macro},A} = \frac{0.480 + 0.667 + 0.500 + 0.000}{4} = 41.2\% \qquad p_{\text{micro},A} = \frac{12+4+2+0}{25+6+4+3} = \frac{18}{38} = 47.4\%$$

System B finds it, along with strong precision on the smaller needs:

$$p_{\text{macro},B} = \frac{0.750 + 0.750 + 1.000 + 1.000}{4} = 87.5\% \qquad p_{\text{micro},B} = \frac{6+3+2+1}{8+4+2+1} = \frac{12}{15} = 80.0\%$$
```

The two averages can even disagree on recall. Macro-averaged recall favours System B, 75.0% against 65.0%, because it weighs the Russian-novels miss as heavily as the 15-book need. Micro-averaged recall favours System A, 78.3% against 52.2%, because pooling all outcome counts lets the large CS-foundations need, where A finds far more books, dominate the sum. The same two systems, the same four needs, and opposite conclusions, depending only on whether needs or documents are the unit being averaged.

```{admonition} Choosing between macro and micro
:class: warning

Macro-averaging is the right choice when small, rare needs matter as much as common ones, such as a benchmark of specialist queries. Micro-averaging is the right choice when overall document-level effectiveness is what matters, since it reflects how a typical user's larger, more common needs are served. Reporting only one of the two without saying which was used leaves the reader unable to tell whether rare cases were protected or diluted.
```

## Architecture and the Metrics

[](#classical-text-fundamental-flows) introduced retriever-only, retriever-with-filter, and retriever-ranker architectures. Precision and recall explain why each stage exists. The retriever is built for recall: it casts a wide net over the collection, accepting a low precision because a false positive can still be removed later, while a false negative is lost for good. A subsequent filter or ranker then raises precision by removing or reordering candidates, at little further cost to recall since the relevant documents the retriever found are still in the pool.

```{admonition} Hands-on: Precision and Recall
:class: hint
Recompute every number in this section from the library collection, then edit a run and watch precision, recall, and $F_\beta$ respond.
[Open notebook -&gt;](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch02-01-performance-evaluation.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```

Precision, recall, and their averages describe a result set with no notion of order. The next section brings rank back into the picture: whether a system found the right documents matters less if they never appear near the top of the list.
