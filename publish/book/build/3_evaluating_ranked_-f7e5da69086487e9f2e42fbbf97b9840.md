---
author: Roger Weber
edition: HS26
status: updated
part: Foundations
chapter: Performance Evaluation
section: Evaluating Ranked Results
order: "2.3"
---

(performance-evaluation-ranked-results)=
# Evaluating Ranked Results

Precision and recall treat a result set as a bag: a document is either in the set or not, and the metrics do not change if the order is shuffled. A ranked retrieval system does more than decide which documents to return; it decides which ones to show first. A user who reads from the top and stops after five items gets a very different experience depending on whether the relevant books are at ranks 1 and 2 or at ranks 16 and 18. Evaluation must account for position, not only membership.

Recall System A's ranked result for the CS-foundations need. It retrieves 25 books and finds 12 of the 15 relevant ones. The first five positions are:

| Rank | Title | Relevant? |
|------|-------|-----------|
| 1 | Computer Organization and Design | yes |
| 2 | Operating System Concepts | yes |
| 3 | The Handmaid's Tale | no |
| 4 | Narrative of the Life of Frederick Douglass | no |
| 5 | Towards a New Architecture | no |

Two relevant books at the top, then three misses. System B's first five:

| Rank | Title | Relevant? |
|------|-------|-----------|
| 1 | Database Systems: The Complete Book | yes |
| 2 | Pattern Recognition and Machine Learning | yes |
| 3 | Introduction to Algorithms | yes |
| 4 | On the Origin of Species | no |
| 5 | Introduction to the Theory of Computation | yes |

Four of five are relevant. The set-based precision numbers from [](#performance-evaluation-precision-recall) already showed that B is more precise overall, but ranking adds a second dimension: how quickly does the user reach useful material?

## Precision at Rank k

The simplest rank-aware metric evaluates only the first $k$ positions. Precision at rank $k$ counts how many of the top-$k$ documents are relevant, ignoring everything below.

```{admonition} Key Formula: Precision at k
:class: important

$$P@k = \frac{|\{d_1, \ldots, d_k\} \cap \text{Rel}|}{k}$$

The fraction of relevant documents among the first $k$ results.
```

$P@k$ answers one practical question: if the user reads only the first $k$ items, what proportion is useful? It does not require knowing how many relevant documents exist in the collection, which makes it easy to compute and interpret. It also does not penalize a system for missing relevant documents below rank $k$: a system that places one relevant book at rank 1 and nothing else achieves $P@1 = 1.0$ regardless of how many relevant books it missed.

```{admonition} Example: P@k for System A and B
:class: example

| $k$ | $P@k$ (A) | $P@k$ (B) |
|-----|-----------|-----------|
| 3 | 2/3 = 66.7% | 3/3 = 100% |
| 5 | 2/5 = 40.0% | 4/5 = 80.0% |
| 10 | 5/10 = 50.0% | 6/10 = 60.0%* |

*System B returns only 8 documents. Positions 9 and 10 are empty, counted as non-relevant.
```

System B dominates at every cutoff. A student who reads the first five results gets four useful books from B but only two from A. $P@k$ captures this user experience directly.

```{admonition} P@k ignores recall
:class: warning

$P@k$ rewards a system for being precise at the top but says nothing about how many relevant documents exist below rank $k$. Two systems with identical $P@5$ can differ drastically in total recall: one may have surfaced all relevant documents, the other may have missed most of them. Report $P@k$ alongside recall or use metrics that integrate both.
```

## Reciprocal Rank and MRR

Some information needs have a single correct answer, or the user is satisfied as soon as one relevant result appears. A library patron asking "Who wrote Dracula?" needs exactly one fact; seeing it at rank 1 is ideal, at rank 20 is frustrating. The reciprocal rank captures this notion.

```{admonition} Key Formula: Reciprocal Rank and MRR
:class: important

$$RR = \frac{1}{\text{rank of first relevant document}} \qquad MRR(\mathbb{Q}) = \frac{1}{|\mathbb{Q}|} \sum_{i=1}^{|\mathbb{Q}|} RR_i$$

The reciprocal of the position where the user first finds what they need, averaged across a set of queries.
```

$RR = 1$ when the first result is relevant; $RR = 0.5$ when the first relevant result is at rank 2; $RR \to 0$ as the relevant document is pushed further down. If no relevant document appears in the result list at all, $RR = 0$.

```{admonition} Example: MRR across four needs
:class: example

Both systems place a relevant document at rank 1 for the CS-foundations need, so $RR = 1$ for both on that query alone. The difference emerges over multiple needs:

| Need | Relevant docs | System A: first relevant rank | $RR_A$ | System B: first relevant rank | $RR_B$ |
|------|--------------|-------------------------------|--------|-------------------------------|--------|
| CS foundations | 15 | 1 | 1.000 | 1 | 1.000 |
| Stage plays | 5 | 1 | 1.000 | 1 | 1.000 |
| Atwood books | 2 | 1 | 1.000 | 1 | 1.000 |
| Russian novels | 1 | none found | 0.000 | 1 | 1.000 |

$$MRR_A = \frac{1 + 1 + 1 + 0}{4} = 0.750 \qquad MRR_B = \frac{1 + 1 + 1 + 1}{4} = 1.000$$
```

System A fails the Russian-novels need entirely (it retrieves "Madame Bovary", "Swann's Way", and "Flowers of Evil", none of which is a nineteenth-century Russian novel). That single miss drags MRR from a perfect 1.0 to 0.75. System B returns only "Crime and Punishment" for that need, which is the one correct answer, at rank 1.

MRR is widely used in question answering and known-item search, where one good answer suffices. It ignores what happens after the first relevant document, so it is inappropriate when multiple relevant documents matter, as in our CS-foundations need where 15 books are relevant.

## The Precision-Recall Curve

$P@k$ evaluates one fixed cutoff. A more complete picture plots precision against recall at every rank where a relevant document appears. As the system returns more documents, recall increases (more relevant documents are found) and precision typically decreases (more non-relevant documents accumulate). Plotting the two against each other produces the precision-recall curve.

At each rank $i$ in the result list, define:

$$P_i = \frac{|\{d_1, \ldots, d_i\} \cap \text{Rel}|}{i} \qquad R_i = \frac{|\{d_1, \ldots, d_i\} \cap \text{Rel}|}{|\text{Rel}|}$$

A point $(R_i, P_i)$ is plotted only at ranks where a relevant document is retrieved (since only those ranks change recall). Between relevant documents, recall stays flat and precision drops, producing a characteristic sawtooth shape.

```{admonition} Example: Precision-Recall curve for System A
:class: example

The relevant documents in System A's ranking appear at ranks 1, 2, 6, 7, 9, 12, 13, 16, 18, 21, 23, and 25. At each of those positions:

| Rank | Relevant found | $P_i$ | $R_i$ |
|------|---------------|--------|--------|
| 1 | 1 | 1/1 = 1.000 | 1/15 = 0.067 |
| 2 | 2 | 2/2 = 1.000 | 2/15 = 0.133 |
| 6 | 3 | 3/6 = 0.500 | 3/15 = 0.200 |
| 7 | 4 | 4/7 = 0.571 | 4/15 = 0.267 |
| 9 | 5 | 5/9 = 0.556 | 5/15 = 0.333 |
| 12 | 6 | 6/12 = 0.500 | 6/15 = 0.400 |
| 13 | 7 | 7/13 = 0.538 | 7/15 = 0.467 |
| 16 | 8 | 8/16 = 0.500 | 8/15 = 0.533 |
| 18 | 9 | 9/18 = 0.500 | 9/15 = 0.600 |
| 21 | 10 | 10/21 = 0.476 | 10/15 = 0.667 |
| 23 | 11 | 11/23 = 0.478 | 11/15 = 0.733 |
| 25 | 12 | 12/25 = 0.480 | 12/15 = 0.800 |

The curve starts at $(0.067, 1.0)$ and ends at $(0.800, 0.480)$. After rank 2, three consecutive non-relevant books push precision from 1.0 down to 0.4 before the next relevant book at rank 6 raises it to 0.5. The curve never reaches recall 1.0 because System A misses 3 of the 15 relevant books.
```

The raw curve is jagged: precision jumps up when a relevant document appears and sags between. To compare systems fairly, the standard technique is interpolation: at any recall level $r$, define the interpolated precision as the maximum precision achieved at any recall level $r' \geq r$.

$$P_{\text{interp}}(r) = \max_{r' \geq r} P(r')$$

This produces a non-increasing step function. Interpolated precision at recall 0 equals the highest precision the system ever achieves; at recall 1 it equals the precision at the last relevant document (if the system finds all relevant documents) or 0 (if recall never reaches 1.0).

## R-Precision

R-precision links precision to recall through a single number. If the collection contains $R$ relevant documents for a need, R-precision is the precision after exactly $R$ documents have been retrieved.

$$\text{R-Prec} = P@R = \frac{|\{d_1, \ldots, d_R\} \cap \text{Rel}|}{R}$$

A perfect system would place all $R$ relevant documents in the first $R$ positions, achieving R-precision of 1.0. The measure adjusts itself to the difficulty of the need: a need with 15 relevant documents (our CS-foundations example) evaluates at cutoff 15, while a need with 2 relevant documents evaluates at cutoff 2.

```{admonition} Example: R-Precision
:class: example

For the CS-foundations need, $R = 15$. System A places 7 relevant books in its first 15 positions:

$$\text{R-Prec}(A) = \frac{7}{15} = 0.467$$

System B returns only 8 documents total. Positions 9 through 15 contain nothing (treated as non-relevant), so the 6 relevant books it returned are all that count:

$$\text{R-Prec}(B) = \frac{6}{15} = 0.400$$

System A wins here because it retrieved more relevant books overall (12 vs. 6), even though B placed them more densely at the top.
```

R-precision is simple and self-adjusting, but it reduces an entire ranking to one point. The metrics that follow evaluate the full curve.

## Average Precision

Average precision (AP) summarizes the precision-recall curve into a single number. It is the mean of the precision values computed at each rank where a relevant document is retrieved, divided by the total number of relevant documents in the collection.

```{admonition} Key Formula: Average Precision
:class: important

$$AP = \frac{1}{|\text{Rel}|} \sum_{k=1}^{n} P_k \cdot \mathbb{1}[d_k \in \text{Rel}]$$

The average of precision-at-rank values, taken only at positions where a relevant document appears, normalized by the total number of relevant documents.
```

The indicator $\mathbb{1}[d_k \in \text{Rel}]$ is 1 when document $d_k$ is relevant and 0 otherwise, so the sum includes exactly one term per relevant document found in the ranking. Dividing by $|\text{Rel}|$ rather than by the number of relevant documents found means that relevant documents the system missed contribute implicit zeros: they would have added a precision term had the system retrieved them, but they did not appear, so nothing is added to the numerator while the denominator still counts them.

```{admonition} Example: Average Precision for System A and B
:class: example

**System A** finds 12 of 15 relevant documents. The precision at each relevant rank:

$$AP_A = \frac{1}{15}\left(\frac{1}{1} + \frac{2}{2} + \frac{3}{6} + \frac{4}{7} + \frac{5}{9} + \frac{6}{12} + \frac{7}{13} + \frac{8}{16} + \frac{9}{18} + \frac{10}{21} + \frac{11}{23} + \frac{12}{25}\right)$$

$$= \frac{1}{15}(1.000 + 1.000 + 0.500 + 0.571 + 0.556 + 0.500 + 0.538 + 0.500 + 0.500 + 0.476 + 0.478 + 0.480)$$

$$= \frac{7.100}{15} = 0.473$$

The three relevant books that System A missed (ranks would be needed beyond 25, or they were never retrieved) each contribute 0 to the sum, pulling the average down.

**System B** finds 6 of 15 relevant documents, all placed high:

$$AP_B = \frac{1}{15}\left(\frac{1}{1} + \frac{2}{2} + \frac{3}{3} + \frac{4}{5} + \frac{5}{6} + \frac{6}{8}\right)$$

$$= \frac{1}{15}(1.000 + 1.000 + 1.000 + 0.800 + 0.833 + 0.750)$$

$$= \frac{5.383}{15} = 0.359$$

System B's precisions at the positions where it finds relevant documents are higher (it averages 0.897 over its 6 hits vs. A's 0.592 over its 12 hits), but it misses 9 of 15 relevant books. Those 9 zeros in the denominator dominate: $AP_B < AP_A$.
```

AP rewards both placing relevant documents high (each term in the sum is larger when fewer non-relevant documents precede it) and finding more relevant documents (more terms contribute to the sum). A system cannot achieve high AP by finding only a few relevant documents and placing them perfectly; the denominator $|\text{Rel}|$ ensures that missed documents count against it.

```{admonition} AP penalizes missed documents
:class: warning

A system that retrieves 3 relevant documents at ranks 1, 2, and 3 with perfect precision of 1.0 at each still scores only $AP = 3/15 \cdot 1.0 = 0.200$ when 15 relevant documents exist. High AP requires both high precision at each relevant position and high recall overall. This is why AP is considered the single best summary of a ranking for a given need.
```

## Mean Average Precision

Evaluating a system on one need gives one AP value. A benchmark with multiple needs produces an AP per need, and mean average precision (MAP) averages them.

$$MAP = \frac{1}{|\mathbb{Q}|}\sum_{i=1}^{|\mathbb{Q}|} AP_i$$

MAP is a macro-average: each need contributes equally regardless of how many relevant documents it contains. This is the same averaging strategy discussed in [](#performance-evaluation-precision-recall), and the same caveats apply: a need with one relevant document has as much influence on MAP as a need with fifteen.

```{admonition} Example: MAP across four needs
:class: example

Using the four needs from the library benchmark:

| Need | Relevant | $AP_A$ | $AP_B$ |
|------|----------|--------|--------|
| CS foundations | 15 | 0.473 | 0.359 |
| Stage plays | 5 | 0.587 | 0.600 |
| Atwood books | 2 | 0.750 | 1.000 |
| Russian novels | 1 | 0.000 | 1.000 |

$$MAP_A = \frac{0.473 + 0.587 + 0.750 + 0.000}{4} = 0.452$$

$$MAP_B = \frac{0.359 + 0.600 + 1.000 + 1.000}{4} = 0.740$$

System B wins on MAP despite finding far fewer relevant documents on the largest need. Two factors combine: B has perfect AP on two smaller needs, while A's zero on Russian novels (it returns three French/German works instead) costs 0.25 in the average, and A also fails to find one of the five plays (buried child) in its stage-plays run.
```

MAP is the standard primary metric in TREC and most retrieval benchmarks. A single MAP value summarizes an entire system's behaviour across all evaluated needs. When two systems are compared, a paired test (such as a paired $t$-test or Wilcoxon signed-rank test) on the per-need AP values determines whether the difference is statistically significant, not just the result of a few easy or hard queries.

## Choosing a Metric

Each metric answers a different question about the ranking:

| Metric | Question it answers | Sensitive to |
|--------|--------------------| -------------|
| $P@k$ | How clean are the first $k$ results? | Precision at a fixed cutoff |
| $MRR$ | How quickly does the user find one good result? | Position of the first relevant document |
| $\text{R-Prec}$ | How well does the system fill the first $R$ slots? | Balance of precision and recall at a natural cutoff |
| $AP$ | How good is the overall ranking for one need? | Both precision and recall, position-weighted |
| $MAP$ | How good is the system across many needs? | Per-need AP, macro-averaged |

$P@k$ and MRR suit settings where users inspect only the top few results: web search, question answering, voice assistants. AP and MAP suit settings where the full ranking matters: patent search, systematic reviews, benchmark comparisons. R-precision provides a single summary that adjusts to the number of relevant documents per need.

All of these metrics treat relevance as binary: a document is relevant or it is not. The next section relaxes this assumption, introducing graded relevance where some documents are more useful than others, and metrics that reward a system for placing highly relevant documents above marginally relevant ones.

```{admonition} Hands-on: Ranked Evaluation
:class: hint
Compute P@k, AP, and MAP from the library rankings, then reorder documents and watch the metrics respond.
[Open notebook ->](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch02-01-performance-evaluation.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```
