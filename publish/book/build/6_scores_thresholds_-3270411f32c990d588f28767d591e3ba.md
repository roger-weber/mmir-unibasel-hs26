(performance-evaluation-thresholds)=
# Scores, Thresholds, and ROC Curves

In many binary classification algorithms, the output is not a direct assignment to a class, but rather a prediction score. Take the example on the top right, where there is a clear correlation between glucose concentration and the presence of diabetes:

  - Plotting the distribution of concentration results in two curves: green for healthy people and red for those with diabetes

  - The distributions overlap, causing uncertainty in determining whether a person is healthy or sick in this overlapping area

  - To make predictions, we need to choose a threshold, denoted as $x^{∗}$ in the figure. Scores (glucose concentration in this case) below $x^{∗}$ are classified as healthy while scores above $x^{∗}$ are classified as sick. $x^{∗}$ becomes a hyperparameter

But how do we select $x^{∗}$? Let’s consider the bottom right figure:

  - Scores below the threshold are predicted as negative cases which we referred to as "false" in the confusion matrix

  - Scores above the threshold are predicted as positive cases which we referred to as “true" in the confusion matrix

  - In general, the actual distributions of scores in the positive and negative classes overlap, making it impossible to perfectly separate the two classes. This results in false negatives (scores below threshold but actually positive cases) and false positives (scores above threshold but actually negative cases)

  - In the example on the top right, we can choose the threshold $x^{∗}$ as shown to safely rule out diabetes with minimal false negatives (high sensitivity). Alternatively, we could set $x^{∗}$ further to the right to reduce false positives and accurately identify people with diabetes (high specificity)

source: https://docs.aws.amazon.com/machine-learning/latest/dg/binary-classification.html

healthy

disease


Let’s have a closer look at how the setting of the threshold impacts the results of the prediction:

  - The plot on the bottom left shows the actual distribution of scores between 0 and 1 for the negative class as $f_{n}(x)$ and for the positive class as $f_{p}(x)$. When we introduce a threshold $T$, we classify a score $x<T$ as negative

  - By considering only $f_{n}(x)$, we can determine the true negatives (green) and false positives (red) as shown in the upper right plot. The true negative rate (TNR), or specificity, is then the integral of $f_{n}(x)$ over scores from $−\infty $ to $T$ (the green area under $f_{n}(x)$). Similarly, we obtain the false positive rate (FPR) as the integral of $f_{n}(x)$ over scores from $T$ to $+\infty $ (the red area under $f_{n}(x)$)

  - By considering only $f_{p}(x)$, we can determine the true positives (green) and false negatives (red) as shown in the lower right plot. The false negative rate (FNR) is then the integral of $f_{p}(x)$ over scores from $−\infty $ to $T$ (the red area under $f_{p}(x)$). Similarly, we obtain the true positive rate (TPR), or sensitivity, as the integral of $f_{p}(x)$ over scores from $T$ to $+\infty $ (the green area under $f_{p}(x)$)

  - When the two distributions, $f_{n}(x)$ and $f_{p}(x)$, overlap, shifting the threshold $T$ to the left will increase the true positive rate (TPR), or sensitivity, while decreasing the true negative rate (TNR), or specificity. Conversely, moving the threshold $T$ to the right will decrease the TPR, or sensitivity, while increasing the TNR, or specificity

  - Prevalence can significantly impact the outcome. For example, if we have 10 times more negative cases, having equal values for TNR and TPR would result in 10 times more true negatives than true positives

FPR

TNR

TPR

FNR

$f_{p}(x)$

$f_{n}(x)$

sensitivity

specificity
The Receiver Operating Characteristic curve, or  ROC curve, is a graph that visualizes the performance of a binary classifier by varying the threshold. It plots true positive rate (TPR, sensitivity) on the y-axis, and false positive rate (FPR, 1-specificity) on the x-axis. We can generate the ROC curve by sorting the data items in the validation set based on their scores in descending order, as illustrated in the table below:

  - The first column, labeled "Class“, represents the actual class of each item (ground truth). The second column, labeled "Score“, contains the scores assigned by the binary classifier to each item

  - The threshold for each row is set to the score value in that row. All items at and above the row's threshold are classified as "positives" by the classifier, while all items below the threshold are classified as "negatives"

  - Let's focus on the highlighted row in the table: using a threshold of 0.54, we can determine the true positives (TP=5, by counting all P’s above and including the row), false positives (FP=1, by counting all N’s above and including the row), false negatives (FN=5, by counting all P’s below the row), and true negatives (TN=9, by counting all N’s below the row). With a total of 10 P’s and N’s, we can calculate the TPR and FPR and then plot the results, as shown in the graph on the right-hand side.

1-specificity

sensitivity

Threshold (T) for this point

$TPR$ (sensitivity)

$FPR$ (1-specificity)

Ideal point with maximum sensitivity and specificity
Interpretation of the ROC-curve: consider the 4 methods A, B, C, D and their confusion matrices below:

  - A lies in an are with high sensitivity. If the $NPV$ is also high (consider the prevalence), then A can effectively “rule out” (negative predictions). A would be a good candidate for the diagnostic process to rule out diseases

  - B lies in the area below the diagonal. In such cases, we can construct B' which negates the outcome of B to obtain a better method: $TP$ and FN switch their values; $FP$ and $TN$ switch their values. This results in new values for the diagram as follows: $TPR′ = 1 − TPR = 60\%$ and $FPR′ = 1 − FPR = 20\%$

  - C has a high sensitivity but a lower $NPV$ than A due to its lower specificity. This affects its ability to "rule out," and a negative prediction is incorrect in 25% of the cases, making it unsuitable for many scenarios

  - D lies in an area with high specificity. If the $PPV$ is also high (consider the prevalence), then D can effectively “rule in” (positive predictions). D would be a good candidate to provide evidence for the presence of a disease

  - Finally, note that the points$ \left(TPR=0,FPR=0\right)$ and $(TPR=1,FPR=1)$ are the results of extreme thresholds that always return negative or positive predictions, respectively. Methods close to these areas, including C below, may exhibit a too strong bias towards negative or positive predictions

better

worse

High sensitivity  ability to “rule out” (neg, prediction) if NPV is high

High specificity  ability to “rule in” (pos. prediction) if PPV is high

$TPR$ (sensitivity)

$FPR$ (1-specificity)

B’
Back to the example from before, depicted again the bottom of the page:

  - To find the optimal threshold, we must consider the specific performance goal for our task. If we aim to "rule out" or "rule in" the condition of the task, we choose thresholds with corresponding $TPR$ and $FPR$ values in or close to that area as highlighted on the previous page

  - In machine learning classification tasks, accuracy is a commonly used performance measure. In this case, we would select the threshold that maximizes the accuracy value. In the example shown below, the threshold of 0.54 gives the highest accuracy and is thus a good choice for this scenario

  - If we don't have a specific performance goal, we can choose the threshold that is closest to the ideal point in the upper left corner. Alternatively, we can optimize for the sum of sensitivity and specificity to make the decision

  - The area under the ROC curve (AUC) is a comprehensive performance measure across all thresholds. It reflects how well a method can distinguish between positive and negative predictions based on the computed scores. In the example below, if a method consistently assigns higher scores to the P's than the N's, the AUC would cover the entire space

1-specificity

sensitivity

Threshold (T) for this point

$TPR$ (sensitivity)

$FPR$ (1-specificity)

Ideal point with maximum sensitivity and specificity