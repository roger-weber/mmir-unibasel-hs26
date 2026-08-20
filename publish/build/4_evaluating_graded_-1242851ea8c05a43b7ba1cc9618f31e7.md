---
author: Roger Weber
edition: HS26
status: updated
part: Foundations
chapter: Performance Evaluation
section: Evaluating Graded Relevance
order: "2.4"
---

(performance-evaluation-graded-relevance)=
# Evaluating Graded Relevance

All metrics so far treat relevance as binary: a document is relevant or it is not. Average precision, for instance, gives System A's rank-1 result ("Computer Organization and Design") the same credit as System B's rank-1 result ("Database Systems: The Complete Book"), because both are relevant. Yet for a student seeking CS foundations, the two books are not equally useful. The grading from [](#performance-evaluation-benchmark-design) distinguishes core foundations (grade 3), important specializations (grade 2), and useful peripheral reading (grade 1). "Database Systems" is a grade-3 core text; "Computer Organization and Design" is grade-1 peripheral reading. A metric that sees only "relevant" cannot tell these apart.

Of the 15 relevant books in the collection, 3 are graded as core foundations, 4 as important specializations, and 8 as useful peripheral reading. The complete grade distribution:

| Grade | Meaning | Count | Examples |
|-------|---------|-------|----------|
| 3 | Core foundations | 3 | Database Systems, Pattern Recognition, Data Science from Scratch |
| 2 | Important specialization | 4 | Operating Systems, AI: A Modern Approach, Programming Languages, Theory of Computation |
| 1 | Useful peripheral reading | 8 | Algorithms, Computer Networks, SICP, Cryptography, ... |
| 0 | Not relevant | 35 | Fiction, poetry, drama, general non-fiction |

Consider the first five results from each system, now labelled with grades:

| Rank | System A | Grade | System B | Grade |
|------|----------|-------|----------|-------|
| 1 | Computer Organization and Design | 1 | Database Systems: The Complete Book | 3 |
| 2 | Operating System Concepts | 2 | Pattern Recognition and Machine Learning | 3 |
| 3 | The Handmaid's Tale | 0 | Introduction to Algorithms | 1 |
| 4 | Narrative of the Life of Frederick Douglass | 0 | On the Origin of Species | 0 |
| 5 | Towards a New Architecture | 0 | Introduction to the Theory of Computation | 2 |

Binary AP cannot distinguish these two top-5 lists beyond counting "2 relevant at top for A, 4 for B". Graded evaluation does more: it rewards B for placing its two highest-value books (both grade 3) in the most visible positions, while A placed a peripheral text (grade 1) at rank 1 and pushed its grade-3 titles to ranks 6, 12, and 18.

A second limitation of AP is subtler. AP rewards a system for placing any relevant document higher, but it does not penalize a system that places relevant documents in a sub-optimal order among themselves. If two relevant documents appear at ranks 3 and 5, AP does not care which of the two is more important. A graded metric can: it assigns more credit when a highly relevant document sits above a marginally relevant one.

## Cumulative Gain

The simplest graded metric just sums the relevance grades of the documents returned up to rank $k$:

```{admonition} Key Formula: Cumulative Gain
:class: important

$$CG_k = \sum_{i=1}^{k} rel_i$$

The total relevance value accumulated in the first $k$ positions. Higher grades contribute more.
```

$CG_k$ measures how much total value the user has gathered after reading $k$ results. It is easy to compute but ignores order entirely: swapping a grade-3 book from rank 10 to rank 1 does not change $CG_{10}$.

```{admonition} Example: Cumulative Gain at k = 5
:class: example

From the table above:

$$CG_5(A) = 1 + 2 + 0 + 0 + 0 = 3$$
$$CG_5(B) = 3 + 3 + 1 + 0 + 2 = 9$$

System B delivers three times the accumulated value in the first five positions. But if we shuffled B's list to put grade-0 at rank 1 and grade-3 at rank 5, $CG_5$ would remain 9. Rank does not matter to CG.
```

## Discounted Cumulative Gain

To make position matter, we discount contributions from lower ranks. The idea is that a user reading from the top gets diminishing benefit from each additional position: finding a grade-3 book at rank 1 is more valuable than finding the same book at rank 10, because by rank 10 the user may have already stopped reading, or the earlier results have already partially satisfied the need.

```{admonition} Key Formula: Discounted Cumulative Gain
:class: important

$$DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}$$

Each relevance grade is divided by a logarithmic discount factor that grows with rank. Rank 1 has no discount ($\log_2 2 = 1$); rank 10 discounts by $\log_2 11 \approx 3.46$.
```

The logarithmic discount encodes a model of user patience: the benefit of each additional rank position decreases, but never reaches zero. A document at rank 10 still contributes, just less than the same document at rank 1.

```{admonition} Example: DCG at k = 5
:class: example

| Rank | Discount $\frac{1}{\log_2(i+1)}$ | A: grade | A: contribution | B: grade | B: contribution |
|------|----------------------------------|----------|-----------------|----------|-----------------|
| 1 | 1/1.000 = 1.000 | 1 | 1.000 | 3 | 3.000 |
| 2 | 1/1.585 = 0.631 | 2 | 1.262 | 3 | 1.893 |
| 3 | 1/2.000 = 0.500 | 0 | 0.000 | 1 | 0.500 |
| 4 | 1/2.322 = 0.431 | 0 | 0.000 | 0 | 0.000 |
| 5 | 1/2.585 = 0.387 | 0 | 0.000 | 2 | 0.774 |

$$DCG_5(A) = 1.000 + 1.262 + 0 + 0 + 0 = 2.262$$
$$DCG_5(B) = 3.000 + 1.893 + 0.500 + 0 + 0.774 = 6.167$$

The discount makes position visible. Both systems have a grade-2 book in the top 5, but A places it at rank 2 (contribution $2 \times 0.631 = 1.262$) while B places its grade-2 book at rank 5 (contribution $2 \times 0.387 = 0.774$). Same grade, same book quality, but A earns 63% more credit from it by showing it earlier.
```

### Discounting with Binary Relevance

DCG works even when all relevance grades are binary (0 or 1). In that case, it reduces to summing the discount factors at positions where a relevant document appears. This isolates the effect of rank position from the effect of graded relevance, which helps clarify what each component contributes.

```{admonition} Example: DCG@10 with binary relevance
:class: example

Treating every relevant document as grade 1 and every non-relevant as grade 0:

| | System A | System B |
|---|---|---|
| Relevant in top 10 | ranks 1, 2, 6, 7, 9 | ranks 1, 2, 3, 5, 6, 8 |
| $DCG_{10}$ (binary) | 1.000 + 0.631 + 0.356 + 0.333 + 0.301 = 2.621 | 1.000 + 0.631 + 0.500 + 0.387 + 0.356 + 0.315 = 3.190 |

System B scores higher because it places relevant documents more densely near the top (3 in the first 3 ranks vs. A's 2 in the first 2 ranks, then a gap). With graded relevance, B's lead grows further because its top-ranked documents also carry higher grades.
```

## Normalized DCG

DCG values depend on the number of relevant documents that exist and on the magnitude of the grades, so raw DCG scores cannot be compared across different queries. A query with five grade-3 documents in the collection can produce a much higher DCG than one with two grade-1 documents, regardless of system quality.

Normalization divides the actual DCG by the best possible DCG for that query: the score achieved by an ideal ranking that places documents in descending order of their grades.

```{admonition} Key Formula: Normalized DCG
:class: important

$$nDCG_k = \frac{DCG_k}{IDCG_k}$$

where $IDCG_k$ is the DCG of the ideal ranking: all relevant documents sorted by decreasing grade, placed at ranks 1 through $k$.
```

$nDCG$ always falls between 0 and 1. A value of 1 means the system produced the best possible ranking for that query at cutoff $k$. A value of 0 means no relevant document appeared in the first $k$ positions.

```{admonition} Example: nDCG@10 for System A and B
:class: example

The ideal ranking for the CS-foundations need places the three grade-3 books first, then the four grade-2 books, then grade-1 books:

| Rank | Ideal grade | Discount | Contribution |
|------|-------------|----------|--------------|
| 1 | 3 | 1.000 | 3.000 |
| 2 | 3 | 0.631 | 1.893 |
| 3 | 3 | 0.500 | 1.500 |
| 4 | 2 | 0.431 | 0.861 |
| 5 | 2 | 0.387 | 0.774 |
| 6 | 2 | 0.356 | 0.712 |
| 7 | 2 | 0.333 | 0.667 |
| 8 | 1 | 0.315 | 0.315 |
| 9 | 1 | 0.301 | 0.301 |
| 10 | 1 | 0.289 | 0.289 |

$$IDCG_{10} = 10.313$$

The actual DCG@10 values from above (with graded relevance):

$$nDCG_{10}(A) = \frac{4.599}{10.313} = 0.446 \qquad nDCG_{10}(B) = \frac{7.825}{10.313} = 0.759$$

System B achieves 76% of the ideal ranking quality; System A achieves only 45%.
```

## What nDCG Reveals That AP Does Not

The contrast between AP and nDCG on our running example is instructive:

| Metric | System A | System B | Winner |
|--------|----------|----------|--------|
| AP (binary, full list) | 0.473 | 0.359 | A |
| nDCG@10 (binary) | 0.577 | 0.702 | B |
| nDCG@10 (graded) | 0.446 | 0.759 | B |

AP favours System A because AP rewards finding many relevant documents (A finds 12 vs. B's 6), and the denominator $|\text{Rel}| = 15$ heavily penalizes B's 9 misses. The nDCG metrics tell a different story because they evaluate only the top 10 positions: within that window, B places more relevant, higher-grade material near the top.

Switching from binary to graded nDCG widens B's lead further. Under binary evaluation, A's rank-1 book and B's rank-1 book both score 1.0 (both are "relevant"). Under graded evaluation, B's grade-3 book at rank 1 scores 3.0 while A's grade-1 book scores only 1.0. Graded relevance captures what a user experiences: not all useful results are equally useful.

```{admonition} nDCG and AP answer different questions
:class: warning

AP asks: "How good is the complete ranking at placing relevant documents high and finding all of them?" It integrates precision over the full result list and penalizes missed documents. nDCG@$k$ asks: "How much value does the user get from reading the first $k$ results?" It evaluates a fixed window, rewards quality ordering within that window, and does not penalize a system for what it fails to retrieve beyond position $k$. Choose AP when exhaustive recall matters; choose nDCG when top-$k$ user experience matters.
```

## Practical Notes

nDCG is the standard metric in web search evaluation (where users rarely look past the first page) and in leaderboard-style benchmarks like MS MARCO and BEIR. The cutoff $k$ is chosen to match the application: $k = 10$ for a web search results page, $k = 3$ for a voice assistant that reads out results, $k = 100$ for a recall-oriented first stage.

The grading scale affects what nDCG rewards. A 4-point scale (0, 1, 2, 3) treats the jump from grade 0 to grade 1 as equivalent to the jump from grade 2 to grade 3. An exponential gain variant, $2^{rel_i} - 1$ instead of $rel_i$, emphasizes highly relevant documents more aggressively: a grade-3 document contributes $2^3 - 1 = 7$ while a grade-1 document contributes only $2^1 - 1 = 1$. This variant is common in practice but not used in our examples.

```{admonition} Hands-on: Graded Evaluation
:class: hint
Compute CG, DCG, and nDCG from the library rankings with graded judgments. Swap documents between positions and observe how discounting amplifies or dampens the effect.
[Open notebook ->](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch02-01-performance-evaluation.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```
