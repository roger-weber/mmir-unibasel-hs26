# Cheat sheet: Evaluation

How to show that one retrieval system or classifier is better than another: benchmarks and
judgments, set-based, ranked and graded measures, classifier rates and thresholds, and the ways a
correctly computed score still misleads (chapter 2).

## Benchmarks and set-based measures

### Designing a benchmark

- "Better" has no meaning until the use is named: a fact check wants a short, precise list; a
  prior-art search wants broad coverage.
- Relevance is a judgment about one need, not a property of the document. Evaluation isolates
  this subjective step: once needs and judgments are recorded, the counting is reproducible.
- A vendor claim or a one-query demonstration is an anecdote: for almost any two systems there is
  a query where each one wins.
- A benchmark fixes four parts: a stable, representative **collection**; written **information
  needs** (the need, not the query string); **relevance judgments** by trained assessors, frozen
  once published; a **measurement procedure** (measure, cutoff, averaging) stated before any
  results are seen.
- At scale, **pooling** judges only the documents the participating systems retrieved; unjudged
  documents count as not relevant. The order between the participants survives, since all are
  scored against the same judged subset.
- Classifier test sets have complete labels and a split into training, validation and test data.

### Precision and recall

- Outcomes: true positive (hit), false positive (noise), false negative (miss), true negative.
- **Precision**: share of the returned documents that are relevant. **Recall**: share of the
  relevant documents that are returned. **Fallout**: share of the non-relevant documents pulled
  in.
- Returning more raises recall and usually lowers precision; returning the whole collection gives
  perfect recall. Report both.
- Precision can be read off a judged result list; recall also needs the number of relevant
  documents in the collection.
- The **F-measure** combines both with a chosen balance ($\beta < 1$ favours precision,
  $\beta > 1$ recall). $F_1$ is a default, not a neutral choice: it weighs a false positive like a
  false negative.
- In a pipeline, the **retriever** is tuned for recall (a missed document is lost for good, a false
  positive can still be removed later), the **ranker or filter** for precision.

*Worked count* (book library example, 50 books, 15 relevant):

| System | Returned | Relevant among them | Precision | Recall |
|---|---|---|---|---|
| A | 25 | 12 | 12/25 = 48 % | 12/15 = 80 % |
| B | 8 | 6 | 6/8 = 75 % | 6/15 = 40 % |

Fallout of A: 13 of 35 non-relevant books, 37.1 %.

**Math** (*for understanding; formulas are not asked in the exam*):

$$p = \frac{TP}{TP + FP} \qquad r = \frac{TP}{TP + FN} \qquad F_\beta = \frac{(\beta^2 + 1) \cdot p \cdot r}{\beta^2 \cdot p + r}$$

**Go deeper:** [retrieval metrics demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch02-01-retrieval-metrics.ipynb)

### Averaging across needs and classes

- **Macro average**: the measure per need or class, then the mean; every need or class counts
  equally. **Micro average**: pool the counts first; every document or item counts equally, so
  large needs and frequent classes dominate.
- The two can disagree: on three library needs, macro recall prefers B (A's 0 of 1 on one need
  weighs a full third), micro recall prefers A (16/21 against 10/21).
- Retrieval benchmarks use macro averages (MAP, mean nDCG, MRR), so one large topic does not
  dominate.
- In single-label classification, micro precision, micro recall and accuracy are the same number.
  Accuracy therefore hides a failing rare class; the macro average shows it.
- Report a per-class value with its number of test items: with a dozen items, one changed label
  moves it by several points.

## Ranked and graded measures

### Ranked measures: P@k, MRR, AP

- **P@k**: how clean are the first k results? It ignores everything below k, including missed
  documents.
- **Reciprocal rank**: 1 over the rank of the first relevant result (0 if none); **MRR** averages
  it over queries. Right when one answer suffices (known-item search, question answering); wrong
  when many documents matter, since it ignores everything after the first hit.
- **Precision-recall curve**: a point at each rank with a relevant document. Its high-precision end
  serves a fact check, its high-recall end a prior-art search.
- **R-precision**: precision at cutoff R, the number of relevant documents (there precision equals
  recall).
- **Average precision (AP)**: the precisions at the ranks of the relevant documents, divided by the
  number of all relevant documents, so missed ones count as zeros. It rewards high placement and
  high recall. **MAP** is its macro average over needs.
- P@k and MRR suit users who read only the top (web search, voice assistants); AP and MAP suit
  tasks where the full ranking matters (patent search, systematic reviews).

*Worked count* (top 5: A = yes, yes, no, no, no; B = yes, yes, yes, no, yes): P@5 is 2/5 = 40 %
for A and 4/5 = 80 % for B. A first relevant result at rank 4 gives RR = 1/4. AP over the full
lists: A 0.473 (12 of 15 found), B 0.359 (9 zeros for the missed books), so AP prefers A although
P@k prefers B.

**Math** (*for understanding; formulas are not asked in the exam*):

$$AP = \frac{1}{|\text{Rel}|} \sum_{k \text{ with } d_k \in \text{Rel}} P_k \qquad MAP = \frac{1}{|Q|} \sum_{i=1}^{|Q|} AP_i$$

Recall at k for approximate search: see [cheat-sheet-index.md](cheat-sheet-index.md#choosing-an-operating-point).

### Graded relevance: DCG and nDCG

- Grades (here 0 to 3) express that relevant documents differ in value.
- **Cumulative gain** sums the grades in the top k and ignores their order (top-5 grades of B,
  3, 3, 1, 0, 2, give CG = 9 in any order).
- **Discounted cumulative gain** divides each grade by a logarithmic discount of its rank, a model
  of a reader who may stop.
- **Normalized DCG** divides by the DCG of the ideal ranking, giving 0 to 1. Only nDCG is
  comparable across queries with different numbers and grades of relevant documents.
- The cutoff k follows how far users read: 10 for a web result page, 3 for a voice assistant; very
  deep ranks add almost nothing.
- nDCG@k and AP answer different questions: the graded value of the first k results, against the
  whole ranking including missed documents. In the book example AP prefers A, nDCG@10 prefers B
  (0.759 against 0.446).
- Scores with linear and exponential gains ($2^{rel} - 1$) are not comparable.
- nDCG as a learning-to-rank target: see [cheat-sheet-text.md](cheat-sheet-text.md#pointwise-pairwise-and-listwise-training).

**Math** (*for understanding; formulas are not asked in the exam*):

$$DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)} \qquad nDCG_k = \frac{DCG_k}{IDCG_k}$$

**Go deeper:** [Evaluating Graded Relevance](https://roger-weber.github.io/mmir-unibasel-hs26/book/evaluating-graded-relevance/)

## Classifiers and thresholds

### Confusion matrix and asymmetric errors

- A classifier assigns one label per item: no ranking, the share of positives matters, and the
  two errors rarely cost the same.
- Rates: precision (of the items called positive, how many are), recall or sensitivity (of the
  positives, how many were found), specificity (of the negatives, how many were left alone; its complement is the
  fallout), accuracy (share of correct decisions).
- Which error to minimize is a design decision taken before evaluation, set by who uses the
  labels. A spam filter must not lose legitimate mail: precision and specificity first. Face unlock
  keeps the false-positive rate very low, at the price of more owner rejections.
- With many classes, evaluate each class one-versus-rest: an overall figure hides a class that
  fails. An intent router with 76.5 % accuracy finds only 5 of 20 `clarify` queries; most go to
  `knowledge_base`, which answers confidently instead of asking back.

*Worked count* (book spam example, 1000 messages):

| | Actual spam (400) | Actual legitimate (600) |
|---|---|---|
| Predicted spam | TP = 360 | FP = 30 |
| Predicted legitimate | FN = 40 | TN = 570 |

Precision 360/390 = 92.3 %, recall 360/400 = 90 %, specificity 570/600 = 95 %, accuracy
930/1000 = 93 %.

**Math** (*for understanding; formulas are not asked in the exam*):

$$\text{Specificity} = \frac{TN}{TN + FP} \qquad \text{Accuracy} = \frac{TP + TN}{P + N} \qquad \text{Fallout} = 1 - \text{Specificity}$$

**Go deeper:** [classification and ROC demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch02-02-classification-and-roc.ipynb)

### Prevalence and the accuracy trap

- With rare positives, true negatives dominate accuracy. Screening test, 30 sick among 2030, TP =
  20, FP = 180: accuracy 90.6 %, precision 10 %. Always answering "healthy" scores 98.5 %.
- Judge an accuracy against the share of the majority class; for imbalanced tasks report
  precision, recall and specificity, or an F-score, which true negatives do not inflate.
- Precision depends on prevalence; recall and specificity do not. A spam filter with 90 % recall
  and 98 % specificity reaches precision 450/460 = 97.8 % on a balanced test set, but 450/640 =
  70.3 % in a mailbox with 5 % spam: 2 % of a much larger legitimate pool gives 190 false
  positives.
- A test set whose class balance differs from deployment therefore reports a precision that
  deployment will not reproduce.

### Thresholds, ROC and AUC

- Every score that becomes a yes/no decision needs a threshold. When the score distributions of
  the two classes overlap, no threshold removes the errors; it only distributes them. A lower
  threshold raises both the true-positive and the false-positive rate.
- The **ROC curve** plots TPR (sensitivity) against FPR (1 − specificity) over all thresholds:
  the diagonal is random guessing, the upper-left corner perfect separation.
- **AUC** is the probability that a random positive scores above a random negative. It compares
  how well two scorers separate the classes, but does not choose the operating point; a lower AUC
  can win in the region that matters.
- The operating point follows from the costs, not from accuracy: face unlock fixes a hard limit on
  false accepts and takes the highest TPR within it.
- Tune the threshold on validation data from the target setting, never on the test set.

## When evaluation misleads

### Incomplete and noisy ground truth

- Unjudged documents count as not relevant: a convention forced by cost, not a finding.
- A new system on a reused pool may retrieve relevant documents nobody judged and is penalized.
  On sparsely judged benchmarks (one or a few judged answers per query), a lower score can mean
  "different documents" rather than "worse documents".
- Recall suffers most, because the total number of relevant documents is never established.
- Judged labels are opinions: assessors disagree on borderline cases, which puts a floor under the
  differences worth interpreting. Complete labels are not necessarily correct labels.
- Per-class numbers from a handful of test items move with a single label.

### What a reported score supports

- A fixed benchmark becomes a target: changes kept because they raise the score absorb the topic
  set and do not transfer, with no dishonesty needed.
- A threshold or hyperparameter chosen on the test set inflates the result. A model trained on the
  test items (contamination, acute for language models) recalls answers instead of solving the
  task.
- Defences: new topic sets per cycle, withheld labels (each submitted score still leaks a little),
  and blind evaluation, where the system is sent to the data.
- One number is not a result: per-need scores vary widely, so a small difference in the mean needs
  a significance test; the best of many tried variants is partly luck.
- Effectiveness is one axis; latency, throughput and cost per query belong in the goals too.
  Comparing approximate search configurations: see [cheat-sheet-index.md](cheat-sheet-index.md#choosing-an-operating-point);
  RAG evaluation: see [cheat-sheet-semantic.md](cheat-sheet-semantic.md#evaluating-rag).

**Go deeper:** [When Evaluation Misleads](https://roger-weber.github.io/mmir-unibasel-hs26/book/when-evaluation-misleads/)
