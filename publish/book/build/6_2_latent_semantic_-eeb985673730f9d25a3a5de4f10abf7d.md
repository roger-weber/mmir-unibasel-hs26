# Latent Semantic Analysis

Previously, we viewed documents as sparse high-dimensional vectors, using either binary (set-of-words) or $tf∙idf$ values. We leveraged vector sparsity with the inverted index for fast document retrieval, and employed parallelism to enhance performance and concurrency. However, this approach assumed term independence, limiting matching for similar but different terms like “house”, “villa”, and “houses”. To address this, we used stemming techniques to reduce words to a common stem and lemmatization to identify synonyms and hypernyms. This improved matching between query and document terms, such as query expansion for “house” queries with synonyms like “villa”.

Stemming and lemmatization, while effective, are language-dependent, general across various documents, and demand substantial manual effort for high-quality results. Moreover, they often fail to address semantically similar terms specific to a collection. To illustrate this with an example from this course, we discussed tokens, words, and terms, which, though not identical, can be seen as closely related. For instance, a search for “tokens” might also yield paragraphs mentioning “terms”. To enhance term matching in such situations, we need a method capable of learning semantic relationships specific to each context, ideally without manual intervention.

Latent Semantic Indexing (LSI) is a method to understand the semantic meaning of terms in document collections through dimensionality reduction. The sparse, high-dimensional document vectors are transformed into a compact, lower-dimensional representation unique to that collection. These dimensions no longer align with individual terms but instead represent latent topics that characterize the collection. However, these topics may not precisely align with our conceptual understanding. Both documents and terms can be expressed as combinations of these topics, and it is this connection between terms and topics that partly explains the extracted topics. For instance, consider a retrieved topic explained as 3.45*airplane + 0.34*bird. We might interpret this as “flying”. However, the model can also generate a mathematically justified description as 3.45*airplane + 0.34*flower, which does not immediately align with a concept we commonly associate with our thinking.

LSI was developed at Bell Labs in the 1980s by Susan Dumais and Scott Deerwester to enhance information retrieval systems. The first article was published in 1988, and a patent was granted the same year (the patent has since expired). Although LSI found applications in various scenarios, it faced challenges due to the substantial computational requirements for topic learning and the inability to use inverted files with the compact lower-dimensional vectors. While computational challenges have been addressed, LSI now lacks the fine-grained semantic associations found in embeddings.

We briefly introduce the mathematical concepts before delving into their application in text retrieval. In linear algebra, eigenvector decomposition is a technique used to convert a quadratic $n×n$-matrix $𝐀$ into a set of  eigenvalues $\lambda _{i}$ and corresponding eigenvectors $v_{i}$ of length 1, satisfying the equation for matrix $𝐀$:

  - Eigenvalues are determined by solving the equation $det⁡(𝐀−\lambda 𝐈)=0$, equivalent to finding roots of a polynomial of degree 𝑛. Eigenvalues can be real or complex and may have multiplicity. The associated eigenvectors are orthonormal. The formula on the right illustrates the eigenvalue decomposition of matrix $𝐀$. Let $r\leq n$ be the rank of $𝐀$. We can express matrix $𝐀$ as the product of $𝐔$ (an $n×n$ matrix) containing the eigenvectors and $𝚲$ (an $n×n$ diagonal matrix) containing the corresponding eigenvalues (with $n−r$ values equal to $0$).

Eigenvectors describe the directions in which the matrix scales and stretches, valuable for characterizing latent topics in a text corpus. However, the document-term matrix is generally non-square. Therefore, we use the Singular Value Decomposition (SVD), a generalization of the eigenvalue decomposition. Let $𝐀$ be an $m×n$-matrix of rank r. There exists an $r×r$-diagonal matrix $𝐒$, an orthonormal $m×r$-matrix $𝐔$, and an orthonormal $n×r$-matrix $𝐕$ such that:

  - The connection between singular value and eigenvalue decomposition is shown with the following representations. Specifically, the singular values are the square roots of the eigenvalues for the matrices $𝐀^{⊤}𝐀$ and $𝐀𝐀^{⊤}$:

We can express $𝐀=𝐔𝐒𝐕^{⊤}$ as a sum of vector products, known as dyadic vector products.

  - By omitting one or more of these summands, we obtain an approximation for 𝐀. We get the best approximation (Frobenius norm) of rank $k<r$ by keeping the summands of the $k$ largest singular values and their corresponding columns in $𝐔$ and $𝐕$. This provides then a mapping from the original $m$-dimensional to a compact $k$-dimensional space.

$𝐀𝒗=\lambda 𝒗$

$𝐀=𝐔𝚲𝐔^{⊤}$

$𝐀=𝐔𝐒𝐕^{⊤}$

$𝐀^{⊤}𝐀 = \left(USV^{⊤}\right)^{⊤}\left(𝐔𝐒𝐕^{⊤}\right) = VSU^{⊤}US𝐕^{⊤} = VS2V^{⊤}$

$𝐀𝐀^{⊤} = \left(USV^{⊤}\right)\left(𝐔𝐒𝐕^{⊤}\right)^{⊤} = USV^{⊤}VS𝐔^{⊤} = US2U^{⊤}$

$𝐀=s_{1}\left(𝒖_{𝟏}𝒗_{𝟏}^{⊤}\right)+s_{2}\left(𝒖_{𝟐}𝒗_{𝟐}^{⊤}\right)+\cdots +s_{r}\left(𝒖_{𝒓}𝒗_{𝒓}^{⊤}\right)$


## 6.2.1 Application in Text Retrieval


In text retrieval, we can apply the Singular Value Decomposition (SVD) to the document-term matrix $𝐀$, typically using $tf∙idf$ weighted components. SVD decomposes $𝐀$ into matrices $𝐔$, $𝐒$, and $𝐕$, reducing them to the intrinsic rank $r \leq  min⁡(m,n)$. Matrix $𝐔$ represents the $m$ terms of the vocabulary in an $r$-dimensional space, $𝐒$ contains singular values (usually sorted by decreasing value on the diagonal), and $𝐕$ holds the $n$ documents in the collection as representations in an $r$-dimensional space.

As we enumerate singular values in decreasing order, the values quickly diminish in magnitude, allowing us to remove many of them while still be able to accurately reconstruct matrix $𝐀$. Removing singular values and their corresponding columns in $𝐔$ and $𝐕$ reduces the dimensionality of the new term and document representations.

$n$ documents

$m$ terms

=

$m×r$

columns of $𝐔$

are orthonormal

$𝐀$

$𝐔$

$m×n$

$r×r$

$𝐒$ diagonal,$𝑟 ≤ min 𝑚 , 𝑛$

$r×n$

rows of $𝐕^{⊤}$

are orthonormal

$𝐕^{⊤}$

x

x

x

x

x

$𝐒$

$𝐀=𝐔𝐒𝐕^{⊤}$

document

document

Note that we use $𝐕^{⊤}$ and thus columns of $𝐕$ are depicted as rows of $𝐕^{⊤}$

term

term

x

x

x

=

U

$k×k$

$𝐒_{k}$ diagonal$k ≤ 𝑟 ≤ min 𝑚 , 𝑛$

$k×n$

rows of $𝐕_{k}^{⊤}$ are

orthonormal

$𝐕_{k}^{⊤}$

$𝐒_{k}$

Dimensionality reduction: When we reduce the number of singular values in the dyadic vector product representation of $𝐀$, we also eliminate corresponding columns in $𝐔$ and $𝐕$, resulting in reduced matrices $𝐔_{k}$, $𝐒_{k}$, and $𝐕_{k}$, as illustrated below. Consequently, the new representations for documents and terms in the original document-term matrix are now more compact, with $k$ dimensions. Columns in $𝐕_{k}^{⊤}$ (equivalent to rows in $𝐕_{k}$) contain the new representations for documents, each dimension expressing a latent topic in the corpus. Consequently, the $i$-th row in $𝐕_{k}^{⊤}$ (or $i$-th column in $𝐕_{k}$) describes the $i$-th latent topic in terms of the documents. Similarly, columns in $𝐔_{k}$ contain the new representations for terms, again with each dimension expressing a latent topic in the corpus. Consequently, the $i$-th column in $𝐔_{k}$ portrays the relationship between the $i$-th topic and the vocabulary terms. This enables us to describe the topics identified by LSI.

When new documents, possibly with new terms, are added to the collection, we must repeat this process for the updated document-term matrix to adapt to topic changes. To avoid recalculations for each new document, the following pages detail an approximate method that delays the need for renewed SVD computations.

$𝐀_{k}$

$𝐔_{k}$

new, reduced representation of the document

document

document

term

term

$m×k$

columns of $𝐔_{k}$

are orthonormal

$m×n$

new, reduced representation of the term

$m$ terms

$n$ documents

Inserting new documents (approximation): Using the approximate representation $𝐀_{k}$ from the reduced singular value decomposition, we can derive a mapping from the original term space to the topic space in three steps as follows: 1) transpose both sides of the equation, 2) multiply first by $\left(𝐔_{k}^{⊤}\right)^{−𝟏}=𝐔_{𝒌}$ and then by $𝐒_{k}^{−1}$, and 3) focus on a single document in $𝐀_{k}^{⊤}$ (denoted as $𝒅^{⊤}$) and $𝐕_{k}$ (denoted as $\overline{𝒅}^{⊤}$).

  - As long as the new documents do not significantly alter the collection's characteristics, the latent topics remain relatively consistent, allowing us to delay the SVD recalculations. When the collection size has increased by a certain threshold percentage, we can initiate recalculations and operate with updated topics.

What about new terms? We can map new terms to their reduced space in $𝐔_{k}$ with the formula $\overline{𝒕}^{⊤}=𝒕^{⊤}𝐕_{k}𝐒_{k}^{−1}$. However, because a new term appears only in a new document, the term vector 𝒕 solely depends on the reduced representation of that new document in $𝐕_{k}$ and the values in $𝐒_{k}$. If a document has two new terms, they both receive the same approximate representation $\overline{𝒕}$ due to this. A better approach is to disregard terms not in the vocabulary and introduce them through a fresh SVD calculation.

documents

terms

=

U

x

x

x

$𝐕_{k}$T

$𝐒_{k}$

$𝐀_{k}$

$𝐔_{k}$

new document

new document

$\overline{𝒅}^{⊤}=𝒅^{⊤}𝐔_{k}𝐒_{k}^{−1}$

$𝐀_{k}=𝐔_{k}𝐒_{k}𝐕_{k}^{⊤}$

$𝐀_{k}^{⊤}=𝐕_{k}𝐒_{k}𝐔_{k}^{⊤}$

$𝐕_{k}=𝐀_{k}^{⊤}𝐔_{k}𝐒_{k}^{−1}$

$\overline{𝒅}^{⊤}=𝒅^{⊤}𝐔_{k}𝐒_{k}^{−1}$

1

2

3

Like vector space retrieval, LSI treats queries as miniature documents. To compare them with the documents in the collection, we must initially map the query, similar to newly added documents, to the reduced topic space:

We apply the same similarity functions as in Vector Space retrieval, using either the dot-product or the cosine measure, to compare the query with the document collection.

The main difference is that we are now comparing two dense vectors. Because the query vector usually has nonzero values in every dimension, we cannot use an inverted index to accelerate the search. Instead we must compute similarity with every document and then sort them by score. Although LSI vectors have fewer dimensions than the original vectors, we still must process much more data and cannot prune documents as efficiently as with inverted indexes. We will cover indexing methods for dense vector search in a later chapter. For now it is important to balance more topics for a richer semantic representation of the corpus's latent topics with fewer dimensions to reduce retrieval costs.

It is possible to reuse the mapping from document vectors to a compact low-dimensional representation when the same vocabulary appears in different collections. However, we usually have to run LSI separately on each corpus. LSI learns not only the important topics from terms but also how those terms are used in documents. Therefore the mapping for an IT article collection will be quite different from the mapping for news articles. Using a generic mapping would lower topic quality, require more topics to cover a broad range of applications, and could harm retrieval performance.

$\overline{𝒒}^{⊤}=𝒒^{⊤}𝐔_{k}𝐒_{k}^{−1}$

$sim_{dot}\left(Q,D_{i}\right)=𝒒∙𝒅_{i}=\sum_{j=1}^{M}q_{j}∙d_{i,j}$

$sim_{cos}\left(Q,D_{i}\right)=\frac{𝒒∙𝒅_{i}}{\left‖𝒒\right‖∙\left‖𝒅_{i}\right‖}=\frac{\sum_{j=1}^{M}q_{j}∙d_{i,j}}{\sqrt{\sum_{j=1}^{M}q_{j}^{2}}∙\sqrt{\sum_{j=1}^{M}d_{i,j}^{2}}}$


## 6.2.2 A Simple Example with LSI


Let’s consider a simple example to illustrate, step-by-step, how LSI works:

c1	Human machine interface for Lab ABC computer applications

c2	A survey of user opinion of computer system response time

c3	The EPS user interface management system

c4	System and human system engineering testing of EPS

c5	Relation of user-perceived response time to error measurement

m1	The generation of random, binary, unordered trees

m2	The intersection graph of paths in trees

m3	Graph minors IV: Widths of trees and well-quasi-ordering

m4	Graph minors: A survey

We see that the collection has two separate document groups: one about human interfaces and one about graph algorithms. If we search the collection for "human-computer interaction", not every relevant document c1 to c5 uses those exact words. Traditional retrieval models must expand the query to add synonyms and broader terms to improve search quality, for example changing "human" to "user" and "computer" to "system". Now we will assess how LSI performs on this example.

First, we create the document-term matrix. For simplicity, we consider only term frequencies and assume equal importance for all terms (i.e., $idf = 1$ for all terms). We also exclude stop words and terms that appear only once in the collection since they are unlikely to contribute to topics, being isolated to a single document.

A =(m=12, n=9)

Then we apply the singular value decomposition on the matrix $𝐀$ below. The results are shown below.

0.2214  -0.1132   0.2890  -0.4148  -0.1063  -0.3410   0.5227  -0.0605  -0.4067

0.1976  -0.0721   0.1350  -0.5522   0.2818   0.4959  -0.0704  -0.0099  -0.1089

0.2405   0.0432  -0.1644  -0.5950  -0.1068  -0.2550  -0.3022   0.0623   0.4924

0.4036   0.0571  -0.3378   0.0991   0.3317   0.3848   0.0029  -0.0004   0.0123

0.6445  -0.1673   0.3611   0.3335  -0.1590  -0.2065  -0.1658   0.0343   0.2707

0.2650   0.1072  -0.4260   0.0738   0.0803  -0.1697   0.2829  -0.0161  -0.0539

0.2650   0.1072  -0.4260   0.0738   0.0803  -0.1697   0.2829  -0.0161  -0.0539

0.3008  -0.1413   0.3303   0.1881   0.1148   0.2722   0.0330  -0.0190  -0.1653

0.2059   0.2736  -0.1776  -0.0324  -0.5372   0.0809  -0.4669  -0.0363  -0.5794

0.0127   0.4902   0.2311   0.0248   0.5942  -0.3921  -0.2883   0.2546  -0.2254

0.0361   0.6228   0.2231   0.0007  -0.0683   0.1149   0.1596  -0.6811   0.2320

0.0318   0.4505   0.1411  -0.0087  -0.3005   0.2773   0.3395   0.6784   0.1825

U =

3.3409

2.5417

2.3539

1.6445

1.5048

1.3064

0.8459

0.5601

0.3637

S =

0.1974   0.6060   0.4629   0.5421   0.2795   0.0038   0.0146   0.0241   0.0820

-0.0559   0.1656  -0.1273  -0.2318   0.1068   0.1928   0.4379   0.6151   0.5299

0.1103  -0.4973   0.2076   0.5699  -0.5054   0.0982   0.1930   0.2529   0.0793

-0.9498  -0.0286   0.0416   0.2677   0.1500   0.0151   0.0155   0.0102  -0.0246

0.0457  -0.2063   0.3783  -0.2056   0.3272   0.3948   0.3495   0.1498  -0.6020

-0.0766  -0.2565   0.7244  -0.3689   0.0348  -0.3002  -0.2122   0.0001   0.3622

0.1773  -0.4330  -0.2369   0.2648   0.6723  -0.3408  -0.1522   0.2491   0.0380

-0.0144   0.0493   0.0088  -0.0195  -0.0583   0.4545  -0.7615   0.4496  -0.0696

-0.0637   0.2428   0.0241  -0.0842  -0.2624  -0.6198   0.0180   0.5199  -0.4535

$𝐕^{⊤}$ =

To simplify document presentation later, we choose $k=2$ and adjust all matrices accordingly. The matrix $𝐕_{k}$T contains the reduced documents as 2-dimensional vectors in its columns, maintaining the same order as in the collection. On the next page, we use these vectors to illustrate the document positions in this 2-topic space.

Next, we project the query into the topic space. Since “interaction” is not in the vocabulary, the query vector contains only two 1s. This vector is then mapped to the 2-dimensional topic space. On the left side, we also display the approximate representation of $𝐀$ with $k=2$. While it may not closely resemble the original document-term matrix, $𝐀_{k}$ shown below is the best rank-2 approximation for $𝐀$ under the Frobenius norm.

0.2214  -0.1132

0.1976  -0.0721

0.2405   0.0432

0.4036   0.0571

0.6445  -0.1673

0.2650   0.1072

0.2650   0.1072

0.3008  -0.1413

0.2059   0.2736

0.0127   0.4902

0.0361   0.6228

0.0318   0.4505

3.3409

2.5417

0.1974  0.6060  0.4629  0.5421  0.2795  0.0038  0.0146  0.0241  0.0820

-0.0559  0.1656 -0.1273 -0.2318  0.1068  0.1928  0.4379  0.6151  0.5299

$𝐔_{k}$

$𝐒_{k}$

$𝐕_{k}$T

0.1621   0.4005   0.3790   0.4676   0.1760  -0.0527  -0.1151  -0.1591  -0.0918

0.1406   0.3698   0.3290   0.4004   0.1650  -0.0328  -0.0706  -0.0968  -0.0430

0.1524   0.5050   0.3579   0.4101   0.2362   0.0242   0.0598   0.0869   0.1240

0.2580   0.8411   0.6057   0.6974   0.3923   0.0331   0.0832   0.1218   0.1874

0.4488   1.2344   1.0509   1.2658   0.5563  -0.0738  -0.1547  -0.2096  -0.0489

0.1596   0.5817   0.3752   0.4169   0.2765   0.0559   0.1322   0.1889   0.2169

0.1596   0.5817   0.3752   0.4169   0.2765   0.0559   0.1322   0.1889   0.2169

0.2185   0.5496   0.5110   0.6281   0.2425  -0.0654  -0.1425  -0.1966  -0.1079

0.0969   0.5321   0.2299   0.2118   0.2665   0.1368   0.3146   0.4444   0.4250

-0.0613   0.2321  -0.1389  -0.2656   0.1449   0.2404   0.5461   0.7674   0.6637

-0.0647   0.3353  -0.1456  -0.3014   0.2028   0.3057   0.6949   0.9766   0.8487

-0.0431   0.2539  -0.0967  -0.2079   0.1519   0.2212   0.5029   0.7069   0.6155

$𝐀_{k}$=

1

0

1

0

0

0

0

0

0

0

0

0

$𝒒$

$𝐔_{k}𝐒_{k}^{−1}$

0.1382

-0.0276

$\overline{𝒒}$

reduced representation for c1

The right side visualizes the document collection. We notice that topic 1 (horizontal) aligns more with documents c1...c5, while topic 2 (vertical) aligns with documents m1...m4

The query is represented at (0.14, -0.03), pointing toward the c-documents. When we apply a cosine similarity measure, we select the green area, which encompasses the subspace with an angle of at most $\alpha $ to the query vector. This area includes all the c-documents, and we can arrange them as follows: c1 < c3 < c4 < c2 < c5.

Interestingly, c3 ranks as the second-best document despite lacking any of the query terms. Due to the SVD reduction, some of its terms align with topics similar to the query terms, making c3 highly relevant.

We can extract the meaning of topic 1 from the $𝐔_{k}$ matrix (first column).

  - 0.64*system 	+ 0.40*user + 0.30*eps 	+ 0.27*time + 0.27*response	+ 0.24*computer

  - and for topic 2:

  - 0.62*graph 	+ 0.49*trees + 0.45*minors 	+ 0.27*survey

0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

Q

c1

c3

c4

c5

m1

m2

m4

m3

c2

-0.3

-0.2

-0.1

0

0.1

0.2

0.3

0.4

0.5





acos(sim) < 

0.7

0.6
