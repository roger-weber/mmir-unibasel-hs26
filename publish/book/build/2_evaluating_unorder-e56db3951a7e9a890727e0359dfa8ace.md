# Precision and Recall

First, we examine Boolean retrieval. In this simple case, documents are returned as an unordered set. The system gives a list of documents, but we do not consider their order when we evaluate it. Later, we will explore how to extend this approach to take document order into account.

Precision and recall are the earliest and still most important measures used for the evaluation of search algorithms. Precision denotes how many of the answers are relevant from a user's perspective. Recall describes the percentage of retrieved and relevant answers over all relevant documents in the collection. They form the key dimensions that covers the user’s interests:

  - Precision measures how easily relevant documents can be found without forcing users to sift through many results. Often, users need only a few documents that provide the required information, not every relevant item. A search engine that returns mostly relevant documents is well suited to knowledge tasks and fact checking, such as student research or professional fact-checking.

  - Recall is about finding most of the relevant documents. In these cases we want a comprehensive view of those documents and we want to keep the number of non-relevant items low (false positives). A good search engine retrieves most or all relevant documents while minimizing non-relevant results.

Notations:

Precision $p$, recall $r$, and fallout f are then defined as (see visualization on next page):

  - $𝔸$		Set of all documents

  - $ℝ_{Q}$  	Set of relevant documents for a query $Q$ in the collection $𝔸$

  - $𝔽_{Q}$  	Set of documents retrieved by a system for query $Q$

  - $\begin{matrix}p=\frac{\left|𝔽_{Q}∩ℝ_{Q}\right|}{\left|𝔽_{Q}\right|}&                         r=\frac{\left|𝔽_{Q}∩ℝ_{Q}\right|}{\left|ℝ_{Q}\right|}&                          f=\frac{\left|𝔽_{Q}∖ℝ_{Q}\right|}{\left|𝔸∖ℝ_{Q}\right|}\end{matrix}$

Visualization or precision and recall

Collection of Documents

$𝔸$

Relevant

Documents

$ℝ_{Q}$

Retrieved Documents

  - $𝔽_{Q}$

Relevant,

Retrieved

$𝔽_{Q}∩ℝ_{Q}$

  - Precision:  $p=\frac{\left|𝔽_{Q}∩ℝ_{Q}\right|}{\left|𝔽_{Q}\right|}$                  Recall: $r=\frac{\left|𝔽_{Q}∩ℝ_{Q}\right|}{\left|ℝ_{Q}\right|}$                  Fallout: $f=\frac{\left|𝔽_{Q}∖ℝ_{Q}\right|}{\left|𝔸∖ℝ_{Q}\right|}$

To better understand these metrics, we introduce some foundational related terms:

  - True Positives (TP): Relevant documents that the system correctly retrieves.

  - False Positives (FP): Non-relevant documents that the system incorrectly retrieves.

  - False Negatives (FN): Relevant documents that the system fails to retrieve.

  - True Negatives (TN): Non-relevant documents that the system correctly does not retrieve.

  - Fallout (f): The proportion of non-relevant documents retrieved out of all non-relevant documents in the collection. It helps measure how much “noise” the system produces.

Let’s illustrate these concepts with a concrete example. Suppose a query has 20 relevant documents in a collection of 1000 documents. Two different search methods are applied:

Comparison of Method A and Method B:

  - Precision: Method B (75%) is higher than Method A (67%). This means Method B returns a higher proportion of relevant documents, making it more efficient for users who want fewer irrelevant results.

  - Recall: Method A (50%) is slightly higher than Method B (45%). This indicates Method A retrieves more of the total relevant documents, which is useful when comprehensiveness is important.

  - Trade-off: Method A sacrifices precision for better recall, while Method B sacrifices recall to improve precision.

  - Recommendation: If the goal is to quickly find mostly relevant documents (e.g., fact-checking or quick research), Method B is preferable due to higher precision. If the goal is to find as many relevant documents as possible (e.g., exhaustive research or legal searches), Method A is preferable due to higher recall.

🔍 Search Method A

🔍 Search Method B

TP =10,  FP=5,  FN=10,  TN=975

$p=\frac{10}{15}=67\%$      $r=\frac{10}{20}=50\%$

TP=9,  FP=3,  FN=11,  TN=977

$p=\frac{9}{12}=75\%$      $r=\frac{9}{20}=45\%$

non-relevant document

relevant document

The F-Measure combines precision and recall into a single value, simplifying the comparison of different retrieval methods. The parameter 𝛽 determines the importance of recall over precision. When 𝛽=0, only precision is considered; when 𝛽=∞, only recall is considered.

  - The $𝑭_{𝟏}$-score is a common choice with 𝛽=1 and is also frequently used in machine learning tasks to optimize hyperparameters (see later in this course). Generally, 𝛽 should be selected thoughtfully depending on the retrieval task's performance goal. For example, for fact-checking tasks, precision is prioritized over recall, making a smaller 𝛽=0.5 suitable. On the other hand, a patent lawyer may choose 𝛽=2 to emphasize the importance of retrieving many relevant documents while maintaining reasonable precision.

Example: Comparing two methods (query has a total of 20 relevant documents)

  - $F_{\beta }=\frac{\left(\beta ^{2}+1\right)∙p∙r}{\beta ^{2}∙p+r}$

🔍 Search Method A

🔍 Search Method B

$p=\frac{7}{14}=50\%$     $r=\frac{7}{20}=35\%$      $F_{1}=\frac{2∙\frac{7}{14}∙\frac{7}{20}}{\frac{7}{14}+\frac{7}{20}}=0.41$

$p=\frac{4}{6}=67\%$       $r=\frac{4}{20}=20\%$       $F_{1}=\frac{2∙\frac{4}{6}∙\frac{4}{20}}{\frac{4}{6}+\frac{4}{20}}=0.31$

non-relevant document

relevant document

  - $F_{1}=\frac{2∙p∙r}{p+r}= $F-score

Typically, we have multiple queries, and for each query, we calculate a precision-recall pair. To evaluate the overall retrieval performance, we need to compute average precision and recall. Let 𝑁 represent the number of queries. For each query $Q_{i}$, we have a set $𝔽_{i}$ retrieved by the search method and a set $ℝ_{i}$ of relevant documents for that query. We use $p_{i}$ and $r_{i}$ to denote the precision and recall values, respectively, as explained earlier.

With macro evaluation, we simply compute the average over all precision and recall values as follows:

  - While the macro evaluation method is generally effective, it can have limitations when dealing with varying sizes of relevant documents for queries. For example, consider a query $Q_{i}$ that has only one relevant document in the entire collection while the other queries have hundreds of relevant documents. Not finding that relevant document for $Q_{i}$ would result in a precision value $p_{i}=0$. This can significantly lower the average precision, even if the method performs well and produces high precision values for the other queries.

Micro evaluation overcomes this issue by summing the true positives and the retrieved/relevant documents before calculating the average precision and recall. This ensures a fair evaluation regardless of the result sizes of queries:

  - With the example from above, the impact on missing out on the relevant document $Q_{i}$ is now much smaller and may better suit the retrieval benchmark’s design.

The choice between micro and macro evaluation depends on the nature of the retrieval task and the importance given to different queries. Micro evaluation tends to emphasize the performance on queries with more relevant documents since they contribute more to the overall counts. Macro evaluation, on the other hand, gives equal weight to each query, regardless of its size or importance, providing a more balanced view of the overall performance across all queries.

  - $p=\frac{1}{N}\sum_{i=1}^{N}p_{i}=\frac{1}{N}\sum_{i=1}^{N}\frac{\left|𝔽_{i}∩ℝ_{i}\right|}{\left|𝔽_{i}\right|}$

  - $r=\frac{1}{N}\sum_{i=1}^{N}r_{i}=\frac{1}{N}\sum_{i=1}^{N}\frac{\left|𝔽_{i}∩ℝ_{i}\right|}{\left|ℝ_{i}\right|}$

  - $p=\frac{\sum_{i=1}^{N}\left|𝔽_{i}∩ℝ_{i}\right|}{\sum_{i=1}^{N}\left|𝔽_{i}\right|}$

  - $r=\frac{\sum_{i=1}^{N}\left|𝔽_{i}∩ℝ_{i}\right|}{\sum_{i=1}^{N}\left|ℝ_{i}\right|}$

Example:

Macro Evaluation

  - Macro Precision: 	(0.0+0.8+0.5)/3 = 0.433

  - Macro Recall: 	(0.0+0.4+0.5)/3 = 0.3

  - Interpretation: Each query contributes equally, so Q1 (with only 1 relevant doc missed) heavily lowers the overall averages.

Micro Evaluation

  - Sum of TP: 	0 + 40 + 25 = 65

  - Sum of Retrieved Docs: 	1 + 50 + 50 = 101

  - Sum of Relevant Docs: 	1 + 100 + 50 = 151

  - Micro Precision: 	65 / 101 = 0.644

  - Micro Recall: 	65 / 151 = 0.430

  - Interpretation: Queries with more relevant documents (Q2 and Q3) dominate the evaluation, so missing Q1 has a smaller effect.

Takeaways:

  - Macro: Use when each query matters equally; highlights worst-case or low-volume queries.

  - Micro: Use when overall effectiveness is key; reflects impact of queries with more relevant documents.

## Retriever and Filter/Ranker

In a previous chapter, we introduced two common retrieval architectures: Retriever-Filter and Retriever-Ranker systems. These patterns show how information retrieval systems choose and present relevant documents to users and differ in how they balance coverage and ordering. Understanding how they affect precision and recall is important for building effective systems:

  - In both architectures the retriever first selects a broad set of candidate documents from the full collection. It aims for high recall so that most or all relevant documents are included. Precision at this stage is often lower because many irrelevant documents may also be retrieved. The goal is to avoid missing relevant documents and to shrink the pool of candidates that will be shown to the user.

  - In Retriever-Filter systems, users apply filters like year, rating, or price to narrow results. This faceted search removes irrelevant items and improves precision. Because the feedback loop can shift which criteria are prioritized, the top results may not always be the most relevant.

Retriever

query

doc 1

doc 2

doc 3

…

index

Filter & Sort

meta-data

criteria