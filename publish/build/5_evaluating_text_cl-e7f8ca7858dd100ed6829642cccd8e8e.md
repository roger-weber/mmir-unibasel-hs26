---
author: Roger Weber
edition: HS26
status: updated
part: Foundations
chapter: Performance Evaluation
section: Evaluating Text Classifiers
order: "2.5"
---

(performance-evaluation-text-classifiers)=
# Evaluating Text Classifiers

Retrieval systems do not exist in isolation. A search engine for images relies on classifiers that detect faces, identify objects, or assign scene categories before any query arrives. An audio retrieval system uses a classifier to segment speech from music. A filtering stage in a retrieval pipeline decides "does this candidate pass the criteria?" — a binary classification decision embedded inside retrieval. These classifiers are components of the retrieval pipeline, and they need their own evaluation. The confusion matrix and its metrics appear every time a system makes a binary or multiclass decision, whether that decision is the final answer or a preprocessing step that feeds into ranking.

The underlying evaluation question is familiar: how often does the system make the right decision? Precision and recall still apply. What changes is the framing. There is no "retrieved set" to measure against a collection; instead, the system assigns exactly one label to each item in a test set, and we compare those predictions against ground-truth labels. Three properties make classification evaluation distinct from retrieval evaluation:

1. **No ranking.** The system outputs a label, not a position. There is no notion of "rank 3 is better than rank 10".
2. **Prevalence matters.** The proportion of positive items in the test set affects how we interpret every metric. A rare disease affects 1% of patients; a common spam folder holds 40% junk.
3. **Error costs are often asymmetric.** A false positive in spam filtering deletes a legitimate email; a false negative lets spam through. These are not equally bad.

## The Confusion Matrix

The confusion matrix organizes every prediction the system made into a 2x2 table for binary classification. Each item in the test set falls into exactly one cell, depending on the system's prediction and the true label.

```{figure} images/figure_2_11.png
:name: fig-confusion-matrix-binary
:width: 50%

The binary confusion matrix. Rows represent predictions, columns represent actual conditions. Green cells are correct predictions; orange cells are errors.
```

The row sums give the number of items the system labelled positive ($T = TP + FP$) and negative ($F = FN + TN$). The column sums give the actual counts: $P = TP + FN$ positive items and $N = FP + TN$ negative items in the test set. The total population is $P + N$.

From these four counts, we derive the same metrics introduced for retrieval, plus several that become important only in classification:

```{admonition} Key Formula: Classification Metrics
:class: important

$$\text{Precision (PPV)} = \frac{TP}{TP + FP} \qquad \text{Recall (TPR)} = \frac{TP}{TP + FN}$$

$$\text{Specificity (TNR)} = \frac{TN}{TN + FP} \qquad \text{Accuracy} = \frac{TP + TN}{P + N}$$

Precision asks: of all items the system called positive, how many truly are? Recall asks: of all truly positive items, how many did the system find? Specificity asks: of all truly negative items, how many did the system correctly leave alone? Accuracy asks: what fraction of all decisions is correct?
```

Precision and recall are identical to the retrieval definitions from [](#performance-evaluation-precision-recall), just viewed from a different angle: "retrieved" becomes "predicted positive", and "relevant" becomes "actually positive". Specificity is new; in retrieval we called its complement "fallout" ($FPR = 1 - \text{Specificity}$). Accuracy had no retrieval equivalent because retrieval operates on a ranked list, not a fixed yes/no decision for every item.

```{figure} images/figure_2_15.png
:name: fig-confusion-matrix-metrics
:width: 80%

The confusion matrix and its derived metrics. Column-normalized rates (TPR, FPR, FNR, TNR), row-normalized predictive values (PPV, NPV), and overall accuracy, with all complementary pairs summing to one.
```

## Example: Spam Filtering

A university email server processes 1000 messages in one day. Of these, 400 are spam ($P = 400$) and 600 are legitimate ($N = 600$). The spam filter makes the following decisions:

| | Actual Spam (400) | Actual Legitimate (600) |
|---|---|---|
| **Predicted Spam** | TP = 360 | FP = 30 |
| **Predicted Legitimate** | FN = 40 | TN = 570 |

```{admonition} Example: Spam filter metrics
:class: example

$$\text{Precision} = \frac{360}{360 + 30} = \frac{360}{390} = 92.3\%$$

$$\text{Recall} = \frac{360}{360 + 40} = \frac{360}{400} = 90.0\%$$

$$\text{Specificity} = \frac{570}{570 + 30} = \frac{570}{600} = 95.0\%$$

$$\text{Accuracy} = \frac{360 + 570}{1000} = 93.0\%$$
```

All four numbers look healthy: the filter catches 90% of spam, and 92% of what it flags is genuinely spam. But consider the 30 false positives: those are 30 legitimate emails moved to the junk folder. If one of them is an acceptance letter for a grant proposal, the cost of that single FP dwarfs the annoyance of 40 spam messages reaching the inbox.

```{admonition} Which error is worse depends on the application
:class: warning

In spam filtering, a false positive (legitimate email deleted) is typically worse than a false negative (spam reaches inbox). The user can ignore spam in the inbox; they cannot read an email they never saw. This asymmetry means that optimizing for high recall (catching all spam) at the cost of precision (more legitimate emails lost) is the wrong trade-off. A spam filter should favour high precision and high specificity, accepting that some spam slips through.
```

## When Accuracy Misleads

Accuracy counts all correct decisions equally. When the two classes are roughly balanced (as in the spam example with 400 vs. 600), accuracy gives a fair summary. When one class dominates, accuracy becomes misleading.

```{figure} images/figure_2_16.png
:name: fig-confusion-matrix-imbalanced
:width: 70%

Confusion matrix for an imbalanced dataset ($P = 30$, $N = 2000$). High accuracy (91%) masks low precision (10%) and moderate recall (67%).
```

Consider a screening test for a rare disease. In a population of 2030 people, 30 have the disease ($P = 30$) and 2000 do not ($N = 2000$). The test produces:

| | Actually Sick (30) | Actually Healthy (2000) |
|---|---|---|
| **Test Positive** | TP = 20 | FP = 180 |
| **Test Negative** | FN = 10 | TN = 1820 |

```{admonition} Example: The accuracy paradox
:class: example

$$\text{Accuracy} = \frac{20 + 1820}{2030} = \frac{1840}{2030} = 90.6\%$$

$$\text{Precision} = \frac{20}{200} = 10\% \qquad \text{Recall} = \frac{20}{30} = 66.7\%$$

$$\text{Specificity} = \frac{1820}{2000} = 91\% \qquad \text{NPV} = \frac{1820}{1830} = 99.5\%$$

The test is 91% accurate, yet only 10% of positive results are correct. Of every 10 people told they might be sick, 9 are healthy. A trivial baseline that always predicts "healthy" achieves $2000/2030 = 98.5\%$ accuracy, which is better than the actual test.
```

The problem is prevalence: only $30/2030 = 1.5\%$ of the population is positive. With so few positives, even a small false-positive rate ($FPR = 180/2000 = 9\%$) produces many more false positives than true positives in absolute terms. Accuracy is dominated by the 2000 true negatives and hides the fact that the test is nearly useless for confirming the disease.

```{admonition} When to distrust accuracy
:class: warning

If prevalence is below 10% or above 90%, accuracy alone is uninformative. Report precision, recall, and specificity separately. The $F_1$-score (or $F_\beta$ with a chosen $\beta$) combines precision and recall without being inflated by true negatives, which makes it a better single-number summary for imbalanced problems.
```

## Asymmetric Error Costs

The spam and screening examples already show that FP and FN carry different costs. A third scenario makes this even starker: biometric face unlock on a smartphone.

The system decides whether the face in front of the camera matches the phone's owner. Two errors are possible:
- **False Positive**: an impostor's face is accepted. The phone unlocks for a stranger. This is a security breach.
- **False Negative**: the owner's face is rejected. The owner must retry or use a PIN. This is a minor inconvenience.

The costs are radically different: a single FP compromises all data on the device, while a FN costs five seconds. The system must therefore achieve an extremely low false-positive rate (FPR), even if this means a noticeable false-negative rate (FNR).

```{admonition} Example: Biometric face unlock
:class: example

A face-unlock system is tested on 10,000 unlock attempts: 9,000 by the legitimate owner and 1,000 by impostors.

| | Owner (9000) | Impostor (1000) |
|---|---|---|
| **Unlocked** | TP = 8820 | FP = 2 |
| **Rejected** | FN = 180 | TN = 998 |

$$\text{FPR} = \frac{2}{1000} = 0.2\% \qquad \text{FNR} = \frac{180}{9000} = 2.0\%$$

$$\text{Precision} = \frac{8820}{8822} = 99.98\% \qquad \text{Recall} = \frac{8820}{9000} = 98.0\%$$

The system accepts a 2% owner-rejection rate to keep impostor acceptance at 0.2%. This trade-off is deliberate: the security requirement demands $FPR < 0.5\%$ regardless of the cost to FNR.
```

The choice of which error to minimize is not a mathematical question; it is a design decision that precedes evaluation. The metrics merely reveal whether the system meets the requirement. In the next section, we explore how moving a decision threshold trades FPR against FNR continuously, and how ROC curves visualize this trade-off.

## Multiclass Classification

Binary classification assigns one of two labels. Many tasks assign one of $K$ labels: an image classifier distinguishes "woman", "man", and "child"; a document router assigns "sports", "politics", "technology", or "culture". The confusion matrix generalizes to a $K \times K$ table where rows are predicted classes and columns are actual classes.

```{figure} images/figure_2_18.png
:name: fig-confusion-matrix-multiclass
:width: 60%

Confusion matrix for a three-class person-type classifier on 100 instances. Diagonal cells (green) are correct; off-diagonal cells (orange/red) are errors.
```

```{admonition} Example: Person-type classifier
:class: example

A classifier labels images as "Woman", "Man", or "Child". Evaluated on 100 images (20 women, 20 men, 60 children):

| | Actual: Woman (20) | Actual: Man (20) | Actual: Child (60) |
|---|---|---|---|
| **Predicted: Woman (19)** | 13 | 4 | 2 |
| **Predicted: Man (18)** | 2 | 15 | 1 |
| **Predicted: Child (63)** | 5 | 1 | 57 |

Overall accuracy: $(13 + 15 + 57) / 100 = 85\%$.
```

### Per-Class Metrics via One-vs-Rest

To compute precision, recall, and specificity for one class, collapse the $K \times K$ matrix into a binary view: "belongs to class $C$" vs. "does not belong to class $C$". For the "Woman" class:

| | Actual Woman (20) | Actual Not-Woman (80) |
|---|---|---|
| **Predicted Woman** | TP = 13 | FP = 6 |
| **Predicted Not-Woman** | FN = 7 | TN = 74 |

$$\text{Precision}_\text{Woman} = \frac{13}{19} = 68.4\% \qquad \text{Recall}_\text{Woman} = \frac{13}{20} = 65.0\%$$

$$\text{Specificity}_\text{Woman} = \frac{74}{80} = 92.5\%$$

The high specificity (92.5%) and accuracy look encouraging, but precision and recall are both below 70%. This is the same prevalence effect: "Woman" is only 20% of the dataset, so specificity is easy to achieve while precision suffers from even a few false positives.

For the "Child" class (60% of the data):

$$\text{Precision}_\text{Child} = \frac{57}{63} = 90.5\% \qquad \text{Recall}_\text{Child} = \frac{57}{60} = 95.0\%$$

The classifier handles children well because the class is large (high prevalence) and visually distinct.

### Aggregating Across Classes

With $K$ classes, we have $K$ precision values and $K$ recall values. Summarizing them into one number uses the same macro/micro distinction from [](#performance-evaluation-precision-recall):

**Macro-averaging** computes the metric per class, then takes the arithmetic mean:

$$\text{Precision}_\text{macro} = \frac{1}{K}\sum_{k=1}^{K} \text{Precision}_k$$

Each class contributes equally, regardless of size. For our example:

$$\text{Precision}_\text{macro} = \frac{0.684 + 0.833 + 0.905}{3} = 0.807$$

**Micro-averaging** pools TP, FP, and FN across all classes before computing the ratio:

$$\text{Precision}_\text{micro} = \frac{\sum_k TP_k}{\sum_k (TP_k + FP_k)} = \frac{13 + 15 + 57}{19 + 18 + 63} = \frac{85}{100} = 0.850$$

With micro-averaging, precision equals recall equals accuracy (all are 85/100) because every item is positive for exactly one class and predicted as exactly one class. Micro-averaging is dominated by the largest class (Child), while macro-averaging gives equal voice to the small Woman and Man classes, revealing that the system struggles with those.

```{admonition} Macro vs. micro in multiclass settings
:class: warning

If all classes matter equally regardless of size (e.g., each intent in a routing system must work well), report macro-averaged metrics. If overall throughput matters more (e.g., a production system where most items are of one type), micro-averaging is appropriate. Always report both when class sizes differ substantially.
```

```{admonition} Hands-on: Classification Evaluation
:class: hint
Build confusion matrices for spam filtering and multiclass scenarios, compute per-class metrics, and observe how prevalence affects accuracy.
[Open notebook ->](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch02-01-performance-evaluation.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```

All metrics in this section assume a fixed decision: the system outputs a label and we count outcomes. But most classifiers internally produce a continuous score (a probability, a distance, a similarity), and the label emerges only after applying a threshold. Moving that threshold changes the balance between FP and FN. The next section explores this continuous perspective through score distributions, threshold selection, and ROC curves.
