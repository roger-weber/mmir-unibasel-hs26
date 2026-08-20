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

The previous section evaluated classifiers that output a fixed label: spam or not spam, owner or impostor. But most classifiers do not produce a label directly. They produce a continuous score: a probability, a confidence, a distance. The spam filter assigns each email a number between 0 and 1 representing how likely it is to be spam. The face-unlock system computes a similarity score between the face in front of the camera and the stored template. The label only emerges after comparing the score to a threshold: if the score exceeds the threshold, predict positive; otherwise, predict negative.

Different thresholds produce different confusion matrices. At a spam threshold of 0.3, the filter catches nearly all spam but also flags many legitimate emails (high TPR, high FPR). At a threshold of 0.8, almost nothing is falsely flagged, but much spam slips through (low FPR, low TPR). The entire performance profile of a classifier is not a single point but a curve across all possible thresholds.

## Score Distributions and the Threshold

A binary classifier assigns a score $x$ to each item. The scores for truly negative items follow one distribution $f_n(x)$, and the scores for truly positive items follow another distribution $f_p(x)$. If the classifier is useful, these distributions are separated: positive items tend to get higher scores. But they always overlap to some degree; perfect separation would mean perfect classification.

```{figure} images/figure_2_20.png
:name: fig-classification-score-distribution
:width: 80%

Score distributions for the positive and negative classes. A threshold (dashed vertical line) partitions the score axis into four confusion-matrix regions. Striped areas represent classification errors.
```

A threshold $T$ divides the score axis into two halves. Items scoring above $T$ are predicted positive; items scoring below are predicted negative. This creates the four familiar outcomes:

- **True Positives (TP)**: positive items with score $\geq T$ (correctly caught)
- **False Positives (FP)**: negative items with score $\geq T$ (falsely flagged)
- **False Negatives (FN)**: positive items with score $< T$ (missed)
- **True Negatives (TN)**: negative items with score $< T$ (correctly left alone)

Moving $T$ to the left catches more positive items (TPR rises) but also sweeps in more negatives (FPR rises). Moving $T$ to the right reduces false alarms (FPR drops) but misses more positives (TPR drops). There is no free lunch: improving one rate always costs the other.

```{figure} images/figure_2_21.png
:name: fig-threshold-rates
:width: 80%

The four classification rates as areas under the class-conditional distributions. TPR and FNR partition the positive distribution $f_p(x)$; TNR and FPR partition the negative distribution $f_n(x)$. Moving the threshold trades sensitivity against specificity.
```

Formally, the rates are integrals over the respective distributions:

$$TPR(T) = \int_{T}^{\infty} f_p(x)\, dx \qquad FPR(T) = \int_{T}^{\infty} f_n(x)\, dx$$

$$FNR(T) = \int_{-\infty}^{T} f_p(x)\, dx \qquad TNR(T) = \int_{-\infty}^{T} f_n(x)\, dx$$

Each complementary pair sums to 1: $TPR + FNR = 1$ and $FPR + TNR = 1$. The threshold $T$ is a single knob that moves continuously between "predict everything positive" ($T = -\infty$, giving $TPR = FPR = 1$) and "predict everything negative" ($T = +\infty$, giving $TPR = FPR = 0$).

## The ROC Curve

Instead of picking one threshold and reporting a single confusion matrix, we can characterize the full behaviour of the classifier by plotting $TPR$ against $FPR$ for every possible threshold. This is the Receiver Operating Characteristic (ROC) curve.

```{admonition} Key Formula: ROC Curve
:class: important

The ROC curve plots $TPR(T)$ on the vertical axis against $FPR(T)$ on the horizontal axis as the threshold $T$ sweeps from $+\infty$ (lower-left corner) to $-\infty$ (upper-right corner).

Each point on the curve corresponds to one threshold and one confusion matrix. The entire curve shows all achievable trade-offs between sensitivity and false-alarm rate.
```

### Construction

To construct the ROC curve from a test set, sort all items by their scores in descending order. Start with the threshold above the highest score: everything is predicted negative, so $TPR = 0$ and $FPR = 0$ (the origin). Lower the threshold one step at a time. Each time a positive item is crossed, TPR increases by $1/P$. Each time a negative item is crossed, FPR increases by $1/N$. Plot the resulting $(FPR, TPR)$ point at each step.

```{admonition} Example: ROC construction for a spam filter
:class: example

A spam filter evaluates 10 emails (5 spam, 5 legitimate) and assigns scores:

| Email | True label | Score |
|-------|-----------|-------|
| 1 | spam | 0.95 |
| 2 | spam | 0.88 |
| 3 | legit | 0.82 |
| 4 | spam | 0.77 |
| 5 | spam | 0.65 |
| 6 | legit | 0.54 |
| 7 | legit | 0.43 |
| 8 | spam | 0.38 |
| 9 | legit | 0.25 |
| 10 | legit | 0.12 |

Sweeping the threshold from high to low:

| Threshold between | Items above | TP | FP | TPR | FPR | Point |
|---|---|---|---|---|---|---|
| (start) | none | 0 | 0 | 0.0 | 0.0 | origin |
| 0.95–0.88 | 1 | 1 | 0 | 0.2 | 0.0 | |
| 0.88–0.82 | 1,2 | 2 | 0 | 0.4 | 0.0 | |
| 0.82–0.77 | 1,2,3 | 2 | 1 | 0.4 | 0.2 | |
| 0.77–0.65 | 1–4 | 3 | 1 | 0.6 | 0.2 | |
| 0.65–0.54 | 1–5 | 4 | 1 | 0.8 | 0.2 | |
| 0.54–0.43 | 1–6 | 4 | 2 | 0.8 | 0.4 | |
| 0.43–0.38 | 1–7 | 4 | 3 | 0.8 | 0.6 | |
| 0.38–0.25 | 1–8 | 5 | 3 | 1.0 | 0.6 | |
| 0.25–0.12 | 1–9 | 5 | 4 | 1.0 | 0.8 | |
| (end) | all | 5 | 5 | 1.0 | 1.0 | upper-right |

The curve rises steeply at first (the highest-scoring items are mostly spam) and flattens in the middle (a legitimate email at rank 3 causes a rightward step without gaining TPR). The curve reaches $TPR = 1.0$ at $FPR = 0.6$, meaning the filter must accept a 60% false-positive rate to catch all spam.
```

### Key Landmarks

Every ROC curve passes through $(0, 0)$ and $(1, 1)$:
- $(0, 0)$: threshold so high that nothing is predicted positive. $TPR = 0$, $FPR = 0$.
- $(1, 1)$: threshold so low that everything is predicted positive. $TPR = 1$, $FPR = 1$.

The diagonal line from $(0, 0)$ to $(1, 1)$ represents a random classifier: one that assigns scores uniformly, so positive and negative items are equally likely to appear at any rank. A useful classifier has a curve that bows above the diagonal toward the upper-left corner $(0, 1)$, which represents perfect classification: $TPR = 1$ with $FPR = 0$.

## Reading the ROC Curve

```{figure} images/figure_2_23.png
:name: fig-roc-space-classifiers
:width: 90%

Four classifiers (A–D) mapped from their confusion matrices to ROC space. The upper-left region represents high sensitivity with low false-alarm rate; the diagonal represents random performance.
```

[Figure %s](#fig-roc-space-classifiers) places four classifiers in ROC space:

- **A** ($TPR = 95\%$, $FPR = 30\%$): high sensitivity, moderate false-alarm rate. Useful for "ruling out" a condition: if the test is negative, we can trust it (high NPV), because it catches 95% of positives. The cost is a 30% false-positive rate.
- **D** ($TPR = 60\%$, $FPR = 5\%$): high specificity, moderate sensitivity. Useful for "ruling in" a condition: if the test is positive, we can trust it (high PPV), because only 5% of negatives are falsely flagged. The cost is missing 40% of positives.
- **C** ($TPR = 90\%$, $FPR = 70\%$): high sensitivity but unacceptably high false-alarm rate. Nearly as sensitive as A, but flags 70% of negatives. The classifier barely improves over random.
- **B** ($TPR = 40\%$, $FPR = 80\%$): below the diagonal. Worse than random. Flipping its predictions (producing B' at $TPR = 60\%$, $FPR = 20\%$) gives a useful classifier.

The biometric face-unlock from [](#performance-evaluation-text-classifiers) operates in the same space as classifier D: it requires $FPR < 0.5\%$ (the leftmost edge of the plot), accepting a moderate FNR. On the ROC curve, this corresponds to reading the TPR value at the vertical line $FPR = 0.005$.

## AUC: Area Under the ROC Curve

The ROC curve shows all possible operating points. A single number that summarizes the entire curve is the Area Under the Curve (AUC).

```{admonition} Key Formula: AUC
:class: important

$$AUC = \int_0^1 TPR(FPR)\, d(FPR)$$

The area under the ROC curve, ranging from 0 to 1. Equivalently, AUC is the probability that a randomly chosen positive item receives a higher score than a randomly chosen negative item.
```

- $AUC = 1.0$: perfect separation. Every positive scores higher than every negative. The ROC curve passes through $(0, 1)$.
- $AUC = 0.5$: random classifier. Scores for positives and negatives are indistinguishable. The ROC curve follows the diagonal.
- $AUC < 0.5$: worse than random. The score ranking is inverted; flipping predictions improves the classifier.

```{admonition} Example: AUC for the spam filter
:class: example

From the ROC construction above, the curve encloses a large area. Computing AUC by summing the rectangles under each step:

$$AUC = (0.2 \times 0) + (0.4 \times 0) + (0.4 \times 0.2) + (0.6 \times 0.2) + (0.8 \times 0.2) + (0.8 \times 0.2) + (0.8 \times 0.2) + (1.0 \times 0.2) + (1.0 \times 0.2)$$

Using the trapezoidal method on the 11 points: $AUC = 0.88$. This means that if we pick a random spam email and a random legitimate email, the filter assigns the higher score to the spam email 88% of the time.
```

```{admonition} AUC does not pick a threshold
:class: warning

AUC summarizes how well the classifier separates classes across all thresholds, but it does not say where on the curve the system should operate. Two classifiers with the same AUC can have very different curves: one may excel at low FPR (good for biometrics) while the other excels at high TPR (good for screening). Always inspect the curve shape, not just the area.
```

## Choosing an Operating Point

The ROC curve shows what is achievable; the operating point is a design decision. It depends on the application and its error costs, connecting directly to the asymmetric-cost discussion in [](#performance-evaluation-text-classifiers).

**Maximize accuracy.** Pick the threshold where $(TP + TN) / (P + N)$ is highest. This is appropriate when false positives and false negatives carry equal cost and prevalence is balanced.

**Constrained optimization.** Fix one rate and optimize the other. The biometric face-unlock requires $FPR \leq 0.2\%$; within that constraint, pick the threshold that maximizes $TPR$. A screening test for a dangerous disease requires $TPR \geq 95\%$; within that constraint, minimize $FPR$.

**Youden's J-statistic.** Maximize $J = TPR - FPR$, which selects the point on the curve farthest from the diagonal. Geometrically, this is the point closest to the upper-left corner, balancing sensitivity and specificity without reference to prevalence.

In practice, the operating point is selected on a validation set (not the test set used for final evaluation). The threshold found on validation data is then applied to the test set to produce the final confusion matrix and reported metrics.

```{admonition} Hands-on: ROC Curves and Thresholds
:class: hint
Sweep the spam filter threshold and watch the ROC curve form point by point. Experiment with different score distributions and observe how separation quality maps to AUC.
[Open notebook ->](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch02-01-performance-evaluation.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```
