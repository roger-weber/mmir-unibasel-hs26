# Evaluating Text Classifiers

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