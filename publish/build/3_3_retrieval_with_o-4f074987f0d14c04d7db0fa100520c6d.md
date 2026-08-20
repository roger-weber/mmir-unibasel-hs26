# Retrieval with Order

$RR_{1}=\frac{1}{2}=0.5$

In many retrieval scenarios, the order of results is crucial. For example, for web search engines or fact-checking tasks, users expect the answer to appear among the top results. In such cases, the Mean Reciprocal Rank (MRR) is the preferred metric. MRR is especially useful when dealing with benchmarks that have sparse assessments as it prevents meaningful calculations of precision and recall values. The definition for queries $Q_{i}\in ℚ$ is as follows:

  - with $rank_{i}$ being the rank of the first relevant document for query $Q_{i}$. Unlike precision and recall, MRR only focuses on the first relevant document and ignores the rest. It places a high importance on the top position in the ranking and assigns lower importance to later positions, converging quickly to 0 as the rank increases.

Example: consider the search method A and its results for queries $Q_{1}$ to $Q_{4}$, as shown in the visualization below. The Mean Reciprocal Rank (MRR) is calculated as the average of the reciprocal ranks $RR_{i}$ for all the queries $Q_{i}$.

  - $MRR(ℚ)=\frac{1}{\left|ℚ\right|}\sum_{Q_{i}\in ℚ}^{}\frac{1}{rank_{i}}$

🔍 Search Method A

$Q_{1}$

⑤

①

②

③

④

$RR_{2}=\frac{1}{4}=0.25$

$Q_{2}$

⑤

①

②

③

④

$RR_{3}=\frac{1}{1}=1$

$Q_{3}$

⑤

①

②

③

④

$RR_{4}=\frac{1}{5}=0.2$

$Q_{4}$

⑤

①

②

③

④

  - $MRR_{A}(ℚ)=0.49$

We can extend the definition of precision and recall to include the ranking of retrieved documents. The Precision-Recall Curve only considers the top-$k$ results in the ranking for $k=1..n$ (as shown in the picture below on the left side). For each top-$k$ result, it calculates the precision and recall values based on their relevance assessment. In the example below with $k=3$, the precision is $p_{3}=2/3$, as 2 out of 3 documents in the top-3 are relevant. If we have 5 relevant documents overall in the collection, then the recall is $r_{3}=2/5$, as the top-3 contains 2 out of 5 relevant documents. We can calculate all the other precision $p_{i}$ and recall $r_{i}$ values, forming the precision-recall curve as depicted on the right side below.

  - As we increase k, the precision increases when the next document is relevant and decreases if it is not relevant. On the other hand, recall values only increase whenever we find a new relevant document. This results in a characteristic "sawtooth" plot, which is often interpolated to simplify subsequent calculations such as the area under the precision-recall curve (blue area in the picture on the right side)

$Q_{1}$

⑤

①

②

③

④

⑩

⑥

⑦

⑧

⑨

$p_{1}=100\%$

$r_{1}=20\%$

$p_{2}=100\%$

$r_{2}=40\%$

$p_{3}=67\%$

$r_{3}=40\%$

$p_{4}=75\%$

$r_{4}=60\%$

$r_{5}=60\%$

$r_{6}=80\%$

$r_{7}=80\%$

$r_{8}=80\%$

$r_{9}=80\%$

$r_{10}=80\%$

$p_{5}=60\%$

$p_{6}=67\%$

$p_{7}=57\%$

$p_{8}=50\%$

$p_{9}=44\%$

$p_{10}=40\%$

⑬

⑪

⑫

$r_{11}=80\%$

$r_{12}=80\%$

$r_{13}=100\%$

$p_{11}=36\%$

$p_{12}=33\%$

$p_{13}=38\%$

⑭

$r_{14}=100\%$

$p_{14}=36\%$

ranking

Interpolated

Precision-Recall-Curve
Note that we can always achieve a recall value of 1 if we keep enumerating documents in the list until all relevant ones have been returned. Once we have all the relevant documents, the PR-curve becomes complete, and we can use it directly for visual comparisons between two methods or calculate simpler metrics for the comparison.

  - Near the point where recall is 0 and precision is 1, the PR-curve shows how well a method can answer fact-checker type queries where high precision for the top-10 documents is expected but recall does not really matter.

  - Near recall values of 1, the PR-curve identifies methods that can find most of the relevant documents in the collection. High precision values are preferred as they indicate low overhead when going through the result list.

  - The ideal system would achieve a precision and recall value of 1. The system efficiency is measured by calculating the distance $d$ of the PR-curve to this ideal point. The system efficiency $E$ is then given by $E=1−d/\sqrt{2}$.

  - The precision at k (P@k) is a commonly used measure, calculated as the precision $p_{k}$ of the top-$k$ results. It is often used when we are not interested in all relevant documents and, thus, do not consider recall values.

  - Similarly, the R-precision measures the precision once a threshold recall value $r_{t}$ is reached. For example, with $r_{t}=20\%$, the metric evaluates the precision once 1/5th of the relevant documents were found. This method requires knowing the total number of relevant documents in the collection.

  - Finally, the average precision (AP) measures the area under the PR-curve (blue area, formula on the right side). High AP values indicate that a method maintains high precision as more and more relevant documents are found.

  - By iterating over a set of queries $ℚ$, we can easily calculate the mean values for all the measures introduced above. The formula on the lower right side shows an example for the mean average precision (MAP).

  - [MATH_ERROR]

  - $MAP\left(ℚ\right)=\frac{1}{\left|ℚ\right|}\sum_{Q_{i}\in ℚ}^{}AP\left(Q_{i}\right)$

Precision

Recall

SystemEfficiency

fact-checker

patentlawyer

$r_{t}$

R-precision

$AP$
