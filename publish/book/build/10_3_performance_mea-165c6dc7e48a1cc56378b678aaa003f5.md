# Performance Measure for Structural Features

Previously, we covered performance measures such as precision and recall, which evaluate how well a system ranks documents for a given query. For tasks that use structural features, such as extracting labels from images, we want to evaluate how well a classifier performs. We consider both binary and multi-class tasks with assignments of labels.

The confusion matrix is a common approach that presents correct and incorrect classifications in a tabular form, enabling the extraction of various metrics to assess the performance of a method. Rows represent the predicted conditions, also known as test results, while columns denote the observed actual conditions, also known as ground truth (the labels in the data sets). Let's examine the table below:

  - The term "condition" is a general description of the task's output value. For instance, it could be "it will rain," "it's a dog," "patient has the disease," "the object belongs to the class," or "the student passes the exam."

  - The "True" row contains all data items for which the method predicts that the item fulfills the condition. Let's compare these predictions with the actual values: first, we have the True Positives (TP) where the prediction is correct (matches the observation). Second, we have the False Positives (FP) where the prediction is wrong, and the method overestimates the condition.

  - The "False" row contains all data items for which the method predicts that the item does not fulfill the condition. If we compare these outcomes with the actual values, we observe the True Negatives (TN) for which the prediction is correct, and the False Negatives (FN) for which the prediction is wrong. In the latter case, the method is underestimating the condition.

There is also “confusion” regarding the placement of actual values in the confusion matrix. Earlier papers display the confusion matrix with actual values in the columns (like here on the left side), while more recent publications and popular software packages use the transposed notation, placing the actual values as rows. This does not change the interpretation of the confusion matrix but makes it difficult to read the table if it appears in the unfamiliar form.

Based on this basic table, we can derive further metrics. To this end, let us introduce the following notation:

  - Based on the ground truth, $P$ and $N$ represent the number of positive and negative items, respectively. $P+N$ is the total size of the dataset. In the 2x2 confusion matrix, the values in the columns sum up to $P$ and $N$.

  - Based on the predicted values, $T$ and $F$ represent the number of “true” and “false” outputs, respectively. $T+F$ is the total size of the dataset. In the 2x2 confusion matrix, the values in the rows sum up to $T$ and $F$.

  - The values in the cells correspond to the notion introduced on the previous page: True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN).

Let’s first consider the rows in more detail:

  - The Positive Predictive Value (PPV), or Precision, is calculated as the ratio $TP/T$. It represents the proportion of correctly predicted positive items. In the context of a disease test, it indicates the percentage of people with positive (‘true’) test results who are actually sick. A low PPV value means that the method would wrongly diagnose a disease from which the patient does not actually suffer. The False Discovery Rate (FDR) is the complement of PPV and is computed as $FP/T=1−PPV$.

  - The Negative Predicative Value (NPV) is calculated as the ratio $TN/F$. It represents the proportion of correctly predicted negative items. In the context of a disease test, it indicates how well a method can exclude a disease during diagnostics of symptoms. A low NPV value means that the method misses many sick people. The False Omission Rate (FOR) is the complement of NPV and is computed as $FN/F=1−NPV$.

Next, we look at the columns for further insights:

  - The True Positive Rate (TPR), also known as Recall, is calculated as $TP/P$. It represents the proportion of correctly predicted items among all positive cases. This measure is often referred to as Sensitivity, for example in the context of a disease test, as it indicates the percentage of actual sick people the test can detect. A high sensitivity in a disease test effectively “rules out” the disease in negative predictions, as it rarely misdiagnoses those who have the disease. However, a high sensitivity does not necessarily indicate the ability to "rule in" the disease. For example, a fake test that always returns positive results will have a sensitivity of 100%, but it is not effective. The False Negative Rate (FNR) is the complement of TPR and is computed as $FN/P=1−TPR$.

  - The True Negative Rate (TNR), also known as Specificity, is calculated as $TN/N$. It represents the proportion of correctly predicted items among all negative cases. This measure, for example in the context of a disease test, indicates the percentage of healthy people the test correctly classifies as "not sick" (false). A high specificity in a disease test effectively “rules in” the disease in positive predictions, as it rarely diagnoses the disease for healthy people. However, a high specificity does not necessarily indicate the ability to “rule out” the disease. For example, a fake test that always returns negative results will have a specificity of 100% but it is not effective. The False Positive Rate (FPR), or Fall-Out, is the complement of TNR and is computed as $FP/N=1−TNR$.

Finally, we consider the diagonals of the confusion matrix:

  - Accuracy (ACC) is calculated as $(TP+TN)/(P+N)$. It represents the percentage of correctly predicted items and has become a standard measure for many classification tasks in machine learning. However, high accuracy alone is not always a good indicator, as we will discuss later with an example.

  - The Error Rate (ERR), or Misclassification Rate, is the complement of the accuracy and measures the percentage of wrongly predicted items. It is calculated as $(FP+FN)/(P+N)=1−ACC$.

The literature has produced many more measures around the confusion matrix. A few examples include:

  - The Prevalence is the ratio of $P$ over the total population. In a balanced scenario where positive and negative cases are about equally frequent, $P$ is close to 0.5. An extreme value for $P$ (<0.1, >0.9) often suggests revisiting the applicability of some of the metrics, as we will see in examples later on.

  - The $F_{1}$-score, as we introduced earlier in this chapter, is a harmonic mean between precision and recall. It is computed as $F_{1}=2∙P∙R/(P+R)=2TP/(2TP+FP+FN)$. Note how the $F_{1}$-score does not take the true negative values $TN$ into account. The $F_{1}$-score is widely used in the natural language processing literature for tasks such as word segmentation or entity recognition.

Overview of popular metrics based on the confusion matrix for binary classifications:

  - To help remember the formulae below better, the core 2x2 matrix of the confusion matrix results from counting the predictions and comparing them with the actual values. $P$, $N$, $T$, and $F$ are the sums of their respective column and row. The cells in the 2x2 matrix to the right are calculated by taking the ratio of the value of the core cell in the same position and the row's total $T$ or $F$. The rows of this matrix sum up to 1. Similarly, the cells in the 2x2 matrix below are calculated by taking the ratio of the value of the core cell in the same position and the column's total $P$ or $N$. The columns of this matrix sum up to 1. Accuracy and Error Rate are the ratios of the sum of the diagonals and the total number of cases ($P+N=T+F$).

Example 1: Is this a good cancer test?

The prevalence value of P=30/2030=1.4% for this data set indicates a highly unbalanced distribution of positive and negative cases. This strongly suggests that we examine different values before drawing conclusions.

  - The test shows 91% accuracy, which looks good at first. However, precision (positive predictive value) is only 10%. Of 200 positive test results, only 20 people actually had cancer. That means 180 people would be wrongly told they have cancer if we follow the test results. The high accuracy comes mainly from the large number of true negatives. In fact, a test that always returns negative would reach 99% accuracy here, producing 2000 true negatives and 0 true positives.

  - The low precision already indicates that we should not rely on a positive test outcome. However, when the test is negative, it is correct in 99% of the cases (NPV). Furthermore, we observe a specificity of 91% (TNR). In other words, during the diagnostic process, we can successfully rule out cancer for 91% of the patients. If this is an affordable test, it can be used as an initial step to eliminate the possibility of this cancer type.

  - We might be concerned about the low sensitivity value of 67% (TPR, recall). Using only this test would miss one-third of the positive cases. Therefore, a doctor should consider other factors like symptoms or additional test results before reaching a conclusion. However, we would not recommend this test for a widely applied preventive campaign, as it could result in many false positives and unnecessary alerts, while still missing many positive cases.

Example 2: Can this test prevent further spreading of a contagious virus in its early stages?

The prevalence value of P=700/99,300=0.7% for this dataset shows a highly unbalanced distribution of positive and negative cases. As in the previous example, let's examine the details. Additionally, let's assume that contagious individuals need to isolate themselves, and only those with severe symptoms require further medical attention.

  - Once more, we see a high accuracy of 95% but a low precision of 11%. The test incorrectly indicates a positive result for many people who do not have the virus. Unlike the first example, a false positive in this case is not as impactful for the individual (unnecessary isolation compared to unnecessary cancer treatment).

  - With a high specificity of 95% (TNR), the test is effective in correctly identifying a large portion of people who do not carry the virus. However, as discussed earlier, about 5% of individuals may still be unnecessarily required to enter isolation.

  - The sensitivity value is crucial in this case. We observe a fairly good value of 85% (TPR), indicating that most people carrying the virus are correctly identified. However, whether the test is acceptable depends on other factors. If the virus is highly contagious, an 85% sensitivity may not be sufficient. As discussed later in this chapter, we may need to adjust certain parameters of the test to increase sensitivity at the cost of lower specificity. This means accepting more false positives in return for reducing false negatives.

  - To enhance the test's sensitivity, we can set a minimum threshold for specificity (e.g., it must be >90%), or we can create a weighted sum of specificity and sensitivity to optimize the test as we make adjustments.

In many classification tasks, there are multiple classes, such as labeling images with recognized animals or objects. The generalized confusion matrix compares each pair of the actual class and recognized class, forming an $K×K$ table where $K$ is the number of classes. Each cell represents the number of items with the actual class in the column and the recognized (or predicted) class in the rows. Some newer literature and software packages may transpose the table, but it does not affect the interpretation of the result. Let's consider the example below with 3 classes: "Woman", "Man", "Child":

  - The table's diagonal represents the correctly recognized classes, while all other cells indicate the prediction errors. For instance, out of the 20 women, 13 were correctly recognized. On the other hand, 2 women were wrongly recognized as men, and 5 women were wrongly recognized as children. To visualize correct and wrong classifications, we use different colors, which make it easier to identify areas of "confusion," especially with a large number of classes. Rearranging columns and rows to create "clusters of confusion" further helps pinpoint issues in the applied method

  - Accuracy is calculated by taking the sum of the cells on the diagonal and dividing it by the total population. In this example, $ACC=(13+15+57)/100=85\%$. On the other hand, the error rate is the complement of $ACC$, giving us $ERR=1−ACC=15\%$ for the example shown

  - But how do we calculate sensitivity, specificity, precision, and other metrics from the binary confusion matrix when dealing with multiple classes? We can use a simple trick: if we want to focus on a particular class $C_{k}$, we can collapse the multi-class view into a binary view with new conditions “$\in C_{k}$” and “$\notin C_{k}$” and then compute the measures as introduced before. Let’s consider an example on the next page

By collapsing the classes "Woman" and "Child", we can delve deeper into the performance of the example:

  - For the class "Woman," we observe a high specificity (correctly dismissing the class) but low precision and sensitivity values. A closer examination of the prevalence also indicates that the high specificity and accuracy values are a consequence of the class imbalance ($20/100=20\%$)

  - For the class "Child," we observe high values for sensitivity, specificity, and precision, indicating that the method successfully recognizes children in the images. The balanced prevalence of 60/100=60% and the high accuracy of 91% suggest that the method is performing well for this class

precision

precision

sensitivity

specificity

sensitivity

specificity


## 10.3.1 Optimizing Hyperparameters


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
