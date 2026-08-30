---
author: Roger Weber
edition: HS26
status: updated
part: Foundations
chapter: Performance Evaluation
section: Scores, Thresholds, and ROC Curves
order: "2.6"
---

(performance-evaluation-thresholds)=
# Scores, Thresholds, and ROC Curves

The previous section evaluated classifiers after the decision was already made: the system assigned a label, and we counted outcomes. But most classifiers do not jump directly to a label. They produce a continuous score, a confidence value between 0 and 1, and the label emerges only after comparing that score to a threshold. The spam filter assigns each email a spam probability: 0.92 for obvious junk, 0.03 for a message from a colleague, 0.47 for a newsletter that could go either way. At threshold 0.5, the newsletter stays in the inbox; at threshold 0.4, it moves to junk.

Different thresholds produce different confusion matrices, different precision/recall values, and different TPR/FPR trade-offs. A single classifier is not one point in metric space; it is a family of points, one for each threshold. This section explores how to visualize that family and choose a single operating point.

## Score Distributions and the Threshold

A binary classifier assigns a score $s(x) \in [0, 1]$ to each item $x$. Items from the positive class tend to receive higher scores; items from the negative class tend to receive lower scores. If the two score distributions were perfectly separated (all positives above some value, all negatives below), any threshold between them would classify perfectly. In practice, the distributions overlap.

```{figure} images/figure_2_20.png
:name: fig-classification-score-distribution
:width: 80%

Score distributions for the negative class (grey) and positive class (yellow) with a decision threshold. The overlap region produces classification errors regardless of where the threshold is placed.
```

The threshold $T$ partitions the score axis into two regions: items with $s(x) \geq T$ are predicted positive, items with $s(x) < T$ are predicted negative. Moving the threshold changes the balance between true positives, false positives, true negatives, and false negatives:

- **Shifting $T$ to the left** (lower threshold): more items cross into the "predicted positive" region. TPR increases (fewer false negatives), but FPR also increases (more false positives). The system becomes more sensitive but less specific.
- **Shifting $T$ to the right** (higher threshold): fewer items are predicted positive. FPR decreases (fewer false positives), but TPR also decreases (more false negatives). The system becomes more specific but less sensitive.

This is the fundamental trade-off underlying every threshold-based classifier. No single threshold is universally best; the right choice depends on the error costs of the application.

## Formalizing the Threshold as Areas Under the Distributions

Let $f_n(x)$ denote the score distribution of the negative class and $f_p(x)$ the score distribution of the positive class. Each classification rate corresponds to an area under one of these curves, partitioned by the threshold $T$:

```{figure} images/figure_2_21.png
:name: fig-threshold-rates
:width: 95%

Decomposition of the score distributions into the four classification rates. TNR and FPR partition the negative-class curve $f_n(x)$; FNR and TPR partition the positive-class curve $f_p(x)$.
```

```{admonition} Key Formula: Classification Rates as Integrals
:class: important

$$TPR(T) = \int_T^{\infty} f_p(x)\, dx \qquad FNR(T) = \int_{-\infty}^{T} f_p(x)\, dx$$

$$TNR(T) = \int_{-\infty}^{T} f_n(x)\, dx \qquad FPR(T) = \int_T^{\infty} f_n(x)\, dx$$

Each pair partitions one distribution: $TPR + FNR = 1$ and $TNR + FPR = 1$. Moving $T$ shifts the boundary between the two areas in each pair.
```

The integrals make the trade-off precise. Shifting $T$ to the left expands the integration range for TPR (the area under $f_p$ to the right of $T$ grows), but simultaneously expands the integration range for FPR (the area under $f_n$ to the right of $T$ also grows). No threshold eliminates the overlap region; it can only choose how to distribute the overlap errors between false positives and false negatives.

## The ROC Curve

Instead of evaluating one threshold, the ROC (Receiver Operating Characteristic) curve evaluates all of them simultaneously. It plots $TPR(T)$ on the vertical axis against $FPR(T)$ on the horizontal axis, sweeping $T$ from high to low. The result is a curve from $(0, 0)$ to $(1, 1)$ that shows the full trade-off landscape.

````{admonition} Example: ROC curve construction for a spam filter
:class: example

[Figure %s](#fig-roc-construction) shows 20 emails (10 spam, 10 legitimate) sorted by descending spam score. At each score, we set the threshold there and count cumulative TP, FP, FN, TN. The right panel plots the resulting ROC curve.

```{figure} images/figure_2_22.png
:name: fig-roc-construction
:width: 100%

ROC curve construction from a ranked list of 20 emails. Each row sets the threshold at that score; the curve traces TPR vs. FPR as the threshold sweeps from high to low. The highlighted point (T = 0.54) has the highest accuracy but is not necessarily the best operating point.
```

The highlighted row at score 0.54 gives $TPR = 50\%$, $FPR = 10\%$, accuracy $= 70\%$. This is the highest-accuracy threshold. But is it the best choice for a spam filter?

- At threshold 0.54: half the spam is caught, and only 10% of legitimate emails are falsely flagged.
- At threshold 0.38: $TPR = 80\%$, $FPR = 50\%$. Most spam is caught, but half the legitimate mail goes to junk.
- At threshold 0.80: $TPR = 20\%$, $FPR = 0\%$. No legitimate email is lost, but 80% of spam reaches the inbox.

For a spam filter, a false positive (legitimate email deleted) is far more costly than a false negative (spam reaching the inbox). The right operating point is not the one that maximizes accuracy but the one that keeps FPR very low while maintaining acceptable TPR. [Figure %s](#fig-roc-construction) shows that no threshold achieves both goals well: the classifier's score separation is simply too weak. This is the point where evaluation tells us to go back and improve the model itself rather than searching for a better threshold on a mediocre curve.
````



### Key Landmarks

| Point | Meaning |
|-------|---------|
| $(0, 0)$ | Threshold so high that nothing is predicted positive (always predict negative) |
| $(1, 1)$ | Threshold so low that everything is predicted positive (always predict positive) |
| $(0, 1)$ | Perfect classifier: all positives detected, no false positives |
| Diagonal | Random classifier: TPR = FPR at every threshold (no discrimination) |

A curve that hugs the upper-left corner represents a strong classifier: it achieves high TPR before FPR rises appreciably. A curve that follows the diagonal adds no information beyond random guessing.

## Reading the ROC Space

```{figure} images/figure_2_23.png
:name: fig-roc-space-classifiers
:width: 100%

Four classifiers (A-D) mapped from their confusion matrices to ROC space. The upper-left region represents high sensitivity with low false-positive rate; the diagonal represents random performance.
```

[Figure %s](#fig-roc-space-classifiers) places four classifiers at different operating points:

- **Classifier A** (TPR = 95%, FPR = 30%): high sensitivity, moderate false-positive rate. If a negative prediction from A is almost certainly correct (high NPV), A can "rule out" the condition: a negative result is trustworthy.
- **Classifier B** (TPR = 40%, FPR = 80%): below the diagonal. This classifier is worse than random. Flipping its predictions gives B' (TPR = 60%, FPR = 20%), which is above the diagonal and usable.
- **Classifier C** (TPR = 90%, FPR = 70%): high sensitivity but very high false-positive rate. It catches most positives but at enormous cost in false alarms.
- **Classifier D** (TPR = 60%, FPR = 5%): high specificity, moderate sensitivity. If a positive prediction from D is almost certainly correct (high PPV), D can "rule in" the condition: a positive result is trustworthy.

The biometric face-unlock from [](#performance-evaluation-text-classifiers) would appear as a point near the left edge of ROC space: $FPR = 0.2\%$, $TPR = 98\%$. The requirement "FPR below 0.5%" constrains the operating point to a narrow vertical strip on the left side of the plot.

## AUC: Area Under the ROC Curve

The ROC curve shows performance across all thresholds. Summarizing it into a single number gives the Area Under the Curve (AUC).

```{admonition} Key Formula: AUC
:class: important

$$AUC = \int_0^1 TPR(FPR)\; d(FPR)$$

Geometrically: the area under the ROC curve. Probabilistically: the probability that a randomly chosen positive item receives a higher score than a randomly chosen negative item.
```

| AUC | Interpretation |
|-----|---------------|
| 1.0 | Perfect separation: every positive scores higher than every negative |
| 0.5 | Random: scores carry no discriminative information |
| < 0.5 | Worse than random (flip predictions to improve) |

```{admonition} Example: AUC interpretation
:class: example

The AUC can be computed from the step-function ROC curve by summing rectangles: each time a negative item is encountered (FPR steps right by $1/N$), the rectangle's height is the current TPR. For the 20-email spam filter in [Figure %s](#fig-roc-construction), the ROC curve hugs the left edge for the first two spam emails (TPR rises to 20% with FPR still at 0%), then begins stepping right. The resulting AUC is approximately 0.74.

This means: if we pick one random spam email and one random legitimate email, there is a 74% chance the spam email receives the higher score. The imperfect separation arises because several spam and legitimate emails have scores in the 0.30-0.55 range where the distributions overlap heavily.
```

AUC is useful for comparing classifiers without committing to a threshold: a system with higher AUC has better score separation overall. However, AUC does not say which operating point the system will use in production, and two classifiers with the same AUC can have very different curves (one may be better at low FPR, the other at high TPR).

```{admonition} AUC summarizes but does not prescribe
:class: warning

A classifier with AUC = 0.95 is not necessarily better than one with AUC = 0.90 for a specific application. If the requirement is $FPR < 0.1\%$ (as in biometric unlock), the only relevant part of the curve is the far-left edge. A system with AUC = 0.90 that achieves $TPR = 97\%$ at $FPR = 0.1\%$ is better for that use case than a system with AUC = 0.95 that only reaches $TPR = 85\%$ at the same FPR constraint. Always examine the curve at the operating region that matters.
```

## Choosing an Operating Point

The ROC curve shows what is possible. Selecting one point on it requires a decision criterion from outside the mathematics:

**Maximize accuracy.** Compute $(TP + TN) / (P + N)$ at each threshold and pick the maximum. This works when classes are balanced and error costs are symmetric. In the spam example from [Figure %s](#fig-roc-construction), threshold 0.54 gives $TP = 5$, $TN = 9$, accuracy $= 14/20 = 70\%$, the highest value in the table. But as we discussed, the spam filter should rather optimize for a low FPR while maintaining acceptable TPR, not simply maximize accuracy.

**Constrained optimization.** Fix one rate and optimize the other. The face-unlock requirement "$FPR \leq 0.2\%$" defines a vertical line on the ROC plot; the operating point is the highest TPR value at or to the left of that line. This is the standard approach when one error type has a hard cost limit.

**Youden's J-statistic.** Maximize $J = TPR - FPR$, which is the vertical distance from the diagonal (random classifier). The point with the highest $J$ is geometrically closest to the upper-left corner and balances sensitivity against specificity without committing to a cost model.

**Equal error rate (EER).** Find the threshold where $FPR = FNR$ (equivalently, $FPR = 1 - TPR$). This is the point where the ROC curve crosses the anti-diagonal from $(0, 1)$ to $(1, 0)$. EER is commonly reported in biometric and speaker-verification systems as a single-number summary that does not require choosing which error is worse.

In all cases, threshold selection should be performed on a validation set separate from the test set used to report final metrics. Optimizing the threshold on the test set inflates reported performance.

```{admonition} Hands-on: Thresholds and ROC
:class: hint
Sweep the decision threshold across the spam-filter scores and watch the confusion matrix, TPR, FPR, and ROC curve update in real time. Compute AUC and find the threshold that maximizes accuracy vs. the one that satisfies a FPR constraint.
[Open notebook ->](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch02-01-performance-evaluation.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```
