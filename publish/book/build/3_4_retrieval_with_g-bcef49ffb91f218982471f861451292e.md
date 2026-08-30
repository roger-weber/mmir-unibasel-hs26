# Retrieval with Graded Relevance

Until now, we have only considered binary relevance assessments for documents. However, we can also use graded relevance assessment where the grade indicates varying degrees of match between the document and the query. For example, we can introduce relevance values between 0 and 3, where 0 means “not relevant“, 1 means “somewhat relevant“, 2 means “relevant“, and 3 means “highly relevant“. The grades now influence how we assess a search method: higher grades are preferred over lower grades at the top of the rankings.

The cumulative gain (CG-k) is a measure of how valuable the top-$k$ results are, similar to precision at $k$:

  - with $rel_{i}$ denoting the graded relevance of the document at position $i$. To obtain a value between 0 and 1, $CG_{i}$ needs to be normalized with $k∙rel_{max}$. In the case of binary relevance assessments, $\hat{CG}_{k}$ is equal to precision at $k$.

The discounted cumulative gain (DCG-k) incorporates the ranking by penalizing relevant documents in lower ranks:

  - The variant on the right emphasizes relevant documents more strongly than the left formula. Other variants may use different logarithm bases and slight modifications of this approach.

Search results for a given query may have varying lengths, making it challenging to interpret DCG-k values and compare them across queries. To address this, we compute a normalized DCG (nDCG-k) by first establishing an ideal ranking where documents are sorted by their graded relevance in descending order, and then calculating the DCG-k value for this ideal ranking. This yields the highest possible value, known as the ideal DCG-k (IDCG-k), which we use to normalize the DCG value of the actual result.

  - $CG_{k}=\sum_{i=1}^{k}rel_{i}$

  - $\hat{CG}_{k}=\frac{\sum_{i=1}^{k}rel_{i}}{k∙rel_{max}}$

  - $rel_{i}\in [0, rel_{max}]$

  - $DCG_{k}=\sum_{i=1}^{k}\frac{rel_{i}}{log_{2}\left(i+1\right)}$

  - $DCG_{k}^{`}=\sum_{i=1}^{k}\frac{2^{rel_{i}}−1}{log_{2}\left(i+1\right)}$

variant:

  - $nDCG_{k}=\frac{DCG_{k}}{IDCG_{k}}$

  - $𝐼𝐷𝐶 𝐺 𝑘 = max 𝐷𝐶 𝐺 𝑘$

with:                                                    (over all possible rankings of documents)

Example: let’s consider the first 10 documents of a search result with graded relevance values between 0 and 3

  - The table above displays 10 results, with the 2nd column indicating the graded relevance $rel_{k}$ for each document at position $k$. The cumulative gain $CG_{k}$ is the sum of these values up to position $k$. To obtain the normalized version, we divide $CG_{k}$ by $3∙k$. Consequently, we obtain $\hat{CG}_{10}=0.5$. For comparison, if we consider any $rel_{k}>0$ as relevant, then the “precision at 10” is 0.70 overrating the documents with low relevance grades.

  - The discounted cumulative gain $DCG_{k}$ is shown in the next columns: first, we have the discount factors for each ranking position. By multiplying them with the graded relevance $rel_{k}$ and summing them up to position $k$, we obtain the $DCG_{k}$ values. Since we have not normalized the relevance values, they are difficult to interpret.

  - For the normalized DCG, we assume there are a total of 5 documents with a graded relevance of 3 and 10 documents with a graded relevance of 2 in the collection. The "ideal $rel_{k}$" column shows an ideal ranking for this scenario, allowing us to compute the ideal DCG ($IDCG_{k}$) values and use them to obtain the normalized DCG ($nDCG_{k}$) values in the rightmost column. An $nDCG_{10}$ value of 0.49 illustrates solid performance in this example.
