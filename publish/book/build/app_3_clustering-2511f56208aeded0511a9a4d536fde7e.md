# Clustering

With unsupervised learning tasks, the machine learning algorithm observes a data set without targets and infers a function that captures the inherent structure and/or distribution of the data. In a clustering scenario, that function is a finite set of clusters and the ability to assign new data items to one (or several) of the clusters. In this chapter, we study the k-means clustering and the Expectation Maximization over a Gaussian mixture to infer a mapping of features to clusters. In the context of multimedia data, typical applications are:

  - Feature quantization, i.e., reducing a multivariate feature to a small number of discrete values. The quantized value serve as an approximated or smoothed version of the original ones much like histograms approximates the distribution of data values

  - Vector search techniques based on product quantization often use k-means clustering to map several dimensions to a small number of bits.

  - Cluster analysis, i.e., the validation of the cluster hypothesis and the extraction of clusters to infer labels for the clusters.

  - Image segmentation, i.e., the extraction of different areas in an image that “belong” to each other. In a first step, clustering reduces the number of features through quantization. In a second step, morphological operators build coherent regions for segmentation.

As we do not know the number of clusters that are present in the data (we have no labels), we need to guide clustering algorithms in the selection of the optimal number $K$ of clusters. Again,  poor choice for the number of clusters can lead to underfitting (extreme case is $K=1$) and overfitting (extreme case is $K=N$ with $N$ being the number of training items). As we have no targets, we cannot use a validation set to measure accuracy of prediction. Instead, we utilize a target function for the compactness of the clusters and the separation between clusters and must prevent, at the same time, an excessive number of clusters.

k-means clustering goes back to the 1960s as an approach to quantify vectors for signal processing. It subsequently became very popular in data mining for cluster analysis. k-means clusters the data set into $k$ clusters in such a way that each data point belongs to the cluster with the nearest centroid (or prototype of the cluster). The centroids are the mean position over all points in the cluster. The centroids divide the space into Voronoi diagrams defining the cluster shapes.

  - Although the computation of the optimal $K$ centroids is a NP-hard problem, there are very efficient heuristics that lead to a (local) optimum. We will first describe the classical approach using Lloyd’s algorithm and then re-interpret the approach with Expectation Maximization.

  - Let $N$ be the number of data items with the $d$-dimensional representations $𝒙_{1}, …, 𝒙_{N}$. We then want to partition the data items into $K$ sets $𝕊=\left\{𝕊_{1},…,𝕊_{K}\right\}$ such that the within-cluster sum of squares (WCSS, also called the variance) become minimal, i.e.:

    - with $𝝁_{k}$ denoting the mean of items in $𝕊_{k}$, and $\sigma _{k}^{2}$ being the variance of items in $𝕊_{k}$. With Lloyd’s algorithm, we obtain a local optimum with a simple iterative algorithm:

[MATH_ERROR]

Select an initial set of centroids $𝝁_{1}^{(0)}, …, 𝝁_{K}^{(0)}$    (see later how to select)

Assign each data point $𝒙$ to the set $𝕊_{k}^{(t)}$ if it is closest to $𝝁_{k}$, i.e., $\left‖𝒙−𝝁_{k}^{(t)}\right‖\leq \left‖𝒙−𝝁_{l}^{\left(t\right)}\right‖ ∀l:1\leq l\leq K$(if several centroids are closest, pick one randomly)

Calculate the new centroids for the next iteration $(t+1)$:

$𝝁_{k}^{(t+1)}=\frac{1}{\left|𝕊_{k}^{(t)}\right|}\sum_{𝒙\in 𝕊_{k}^{(t)}}^{}𝒙$

Repeat steps 2 and 3 until algorithm has converged

  - Initial choice of centroids

    - Random points: pick $K$ random items from the data set. This leads to a spread of centroids across the data space.

    - Random partition: assign each data item to a random cluster (1 to $K$) and compute centroids over these random clusters. These centroids tend to be closer together near the center of the data set.

    - k-means++: the first centroid is chosen randomly from the data set. Each subsequent centroid (up to $K$) is chosen from the remaining items with probabilities proportional to the their squared distance to closest centroid. Although more expensive, it leads to much smaller final errors and faster convergence during the iterative part.

Expectation Maximization (EM) (and interpretation of k-means algorithm)

  - Expectation maximization is an iterative method to estimate parameters in a statistical model than cannot be solved in closed form. It assumes that the observations (here: the training set) are obtained from probability distribution, typically a mixture of several distributions with a soft assignment. In k-means, we used a hard assignment, that is, every data point is assigned to exactly one cluster. In EM, soft assignment denotes that cluster assignment of a point follows a conditional distribution. Finally, the objective is to find the soft assignment and the parameters of the distributions (e.g., with Gaussian, these are the means and variances) that best explain the observations (maximum likelihood).

  - Solving above objective function in closed form is not always possible. The EM algorithm consists of two steps: in the expectation step, the distribution parameters are constant and we compute the best soft assignment. In the maximization step, we keep the soft assignment constant and choose the parameters that maximize the objective function. With each step, the objective function increases and eventually converges, but not necessarily to a global optimum.

  - Let us start with a simple one dimensional example with a mixture of two ($K=2$) Gaussian distributions $𝒩\left(\m _{k},\sigma _{k}^{2}\right)$. The picture on the right shows the two Gaussian distributions and their mixture. With an infinite number of Gaussians, a mixture can model any distribution. Each Gaussian represent a sub-population (cluster) of the data items that follow its distribution. In addition, a prior $P\left(C_{k}\right)$ defines how likely data items come from $k$-the cluster with $\sum_{}^{}P\left(C_{k}\right)=1$.

  - Now, assume we make the observations $𝕋=\{x_{1},…, x_{N}\}$. Further assume, we know that all $x\in 𝕊_{1}$ stem from the blue cluster $C_{1}$, and all $x\in 𝕊_{2}=𝕋∖𝕊_{1}$ stem from the red cluster $C_{2}$. We then can easily compute the parameters and the priors of the distributions using the (biased) estimators:

  - On the other side, assume we know the parameters $\m _{k},\sigma _{k}^{2}$ of the distributions and the priors $P\left(C_{k}\right)$, can we estimate the probability $P\left(x_{i}\right)$ that a point $x_{i}$ is part of cluster $C_{k}$?

$\m _{k}=\frac{\sum_{x\in 𝕊_{k}}^{}x}{\left|𝕊_{k}\right|}$

$\sigma _{k}^{2}=\frac{\sum_{x\in 𝕊_{k}}^{}\left(x−\m _{k}\right)^{2}}{\left|𝕊_{k}\right|}$

$P\left(C_{k}\right)=\frac{\left|𝕊_{k}\right|}{N}$

$P\left(x_{i}\right)=\frac{P\left(C_{k}\right)∙P\left(C_{k}\right)}{P\left(x_{i}\right)}=\frac{P\left(C_{k}\right)∙P\left(C_{k}\right)}{\sum_{k}^{}P\left(C_{k}\right)∙P\left(C_{k}\right)}$

with    $𝑃 𝑥 𝑖 𝐶 𝑘 = 𝑓 𝑥 𝑖 ; 𝜇 𝑘 , 𝜎 𝑘 2 = 1 2 𝜋 𝜎 𝑘 2 ∙ exp − 𝑥 𝑖 − 𝜇 𝑘 2 2 𝜎 𝑘 2$


  - Given the probabilities $P\left(x_{i}\right)$ that $x_{i}$ belongs to cluster $C_{k}$ we no longer have a hard assignment as above with $𝕋=𝕊_{1}∪𝕊_{2}$, and $𝕊_{1}∩𝕊_{2}=∅$, but utilize soft assignments. In other words,we are not entirely sure from which sub-population the points come from but have a fairly good understanding how likely they stem from each cluster. To estimate the parameters and the priors, we need to take the soft assignments into account:

  - Now we can summarize the EM algorithm: to this end, we introduce the responsibility $\gamma _{i,k}=P\left(x_{i}\right)$ denoting the soft assignment of data item $x_{i}$ to cluster $C_{k}$, and the weights $w_{k}=P\left(C_{k}\right)$ representing the prior of cluster $C_{k}$. The algorithm runs as follows:

$\m _{k}=\frac{\sum_{i}^{}P\left(x_{i}\right)∙x_{i}}{\sum_{i}^{}P\left(x_{i}\right)}$

$\sigma _{k}^{2}=\frac{\sum_{i}^{}P\left(x_{i}\right)∙\left(x−\m _{k}\right)^{2}}{\sum_{i}^{}P\left(x_{i}\right)}$

$P\left(C_{k}\right)=\frac{\sum_{i}^{}P\left(x_{i}\right)}{N}$

Select initial values for $\m _{k}^{(0)}, \sigma _{k}^{2}^{(0)}$ and $w_{k}^{(0)}$ for $1\leq k\leq K$

E-step: evaluate new responsibilities $\gamma _{i,k}^{(t)}$ for $1\leq i\leq N$ and $1\leq k\leq K$ using current parameters

M-step: evaluate new parameters $\m _{k}^{(t+1)}, \sigma _{k}^{2}^{(t+1)}$ and $w_{k}^{(t+1)}$ for $1\leq k\leq K$ using current responsibilities

Repeat E-step and M-step until the parameters stop changing

$\gamma _{i,k}^{(t)}=\frac{w_{k}^{(t)}∙f\left(x_{i}; \m _{k}^{(t)},\sigma _{k}^{2}^{(t)}\right)}{\sum_{k}^{}w_{k}^{(t)}∙f\left(x_{i}; \m _{k}^{(t)},\sigma _{k}^{2}^{(t)}\right)}$

$\m _{k}^{(t+1)}=\frac{\sum_{i}^{}\gamma _{i,k}^{(t)}∙x_{i}}{\sum_{i}^{}\gamma _{i,k}^{(t)}}$

$\sigma _{k}^{(t+1)}=\frac{\sum_{i}^{}\gamma _{i,k}^{(t)}∙\left(x_{i}−\m _{k}^{\left(t+1\right)}\right)^{2}  }{\sum_{i}^{}\gamma _{i,k}^{(t)}}$

$w_{k}^{(t+1)}=\frac{\sum_{i}^{}\gamma _{i,k}^{(t)}  }{N}$

  - Once convergence of EM is reached after $\vartheta $ iterations, we can (hard) assign a data item $x_{i}$ to its most likely cluster $C_{k^{∗}}$ by solving the following equation:

  - We can generalize this approach to $d$-dimensional spaces with $d=M$ being the number of features. We create a mixture of $K$ multi-variate (or multi-dimensional) Gaussian distribution $𝒩(𝝁_{k},𝚺_{k})$ with $𝝁_{k}=E\left[𝒙\in 𝕋_{k}\right]$ denoting the centroid of items of cluster $C_{k}$, and $𝚺_{k}=E_{𝒙\in 𝕋_{k}}\left[\left(𝒙−𝝁\right)\left(𝒙−𝝁\right)^{T}\right]$ the covariance matrix of items in cluster $C_{k}$.

  - Again, we obtain a hard assignment for a data item $𝒙_{i}$ to its most likely cluster $C_{k^{∗}}$ as follows:

[MATH_ERROR]

Select initial values for $𝝁_{k}^{(0)}, 𝚺_{k}^{2}^{(0)}$ and $w_{k}^{(0)}$ for $1\leq k\leq K$

E-step: evaluate new responsibilities $\gamma _{i,k}^{(t)}$ for $1\leq i\leq N$ and $1\leq k\leq K$ using current parameters

M-step: evaluate new parameters $𝝁_{k}^{(t+1)}, 𝚺_{k}^{2}^{(t+1)}$ and $w_{k}^{(t+1)}$ for $1\leq k\leq K$ using current responsibilities

Repeat E-step and M-step until the parameters stop changing

$\gamma _{i,k}^{(t)}=\frac{w_{k}^{(t)}∙f\left(𝒙_{i}; 𝛍_{k}^{(t)},𝚺_{k}^{2}^{(t)}\right)}{\sum_{k}^{}w_{k}^{(t)}∙f\left(𝒙_{i}; 𝛍_{k}^{(t)},𝚺_{k}^{2}^{(t)}\right)}$

$𝝁_{k}^{(t+1)}=\frac{\sum_{i}^{}\gamma _{i,k}^{(t)}∙𝒙_{i}}{\sum_{i}^{}\gamma _{i,k}^{(t)}}$

$𝚺_{k}^{(t+1)}=\frac{\sum_{i}^{}\gamma _{i,k}^{(t)}∙\left(𝒙_{i}−𝝁_{k}^{\left(t+1\right)}\right)  \left(𝒙_{i}−𝝁_{k}^{\left(t+1\right)}\right)^{T} }{\sum_{i}^{}\gamma _{i,k}^{(t)}}$

$w_{k}^{(t+1)}=\frac{\sum_{i}^{}\gamma _{i,k}^{(t)}  }{N}$

[MATH_ERROR]

$𝑓 𝒙 𝑖 ; 𝛍 𝑘 , 𝚺 𝑘 2 = 1 2 𝜋 𝑑 𝚺 ∙ exp − 1 2 𝒙 𝑖 − 𝝁 𝑘 𝑇 𝚺 k −1 𝒙 𝑖 − 𝝁 𝑘$

  - Where does the name Expectation Maximization come from? Let $𝕏=\left\{𝒙_{i}\right\}$ be the set of data items and $𝕐=\left\{w_{1},\m _{1}, \sigma _{1},…,w_{k}, \m _{K},\sigma _{K}\right\}$ be the set of unknown parameters of the mixture of K Gaussian distributions. In addition, we have the latent unobserved data items $ℤ=\left\{\gamma _{i,k}\right\}$ denoting the soft memberships of $x_{i}$ to cluster $C_{k}$. Given, $𝕏$ we want to find the parameters $𝕐$ that maximize the probability that the data items in $𝕏$ are observations from the mixture using these parameters. This is called the maximum likelihood estimate (MLE):

    - In other words, if $𝕐$ is known, how likely is it that data items in $𝕏$ follow the mixture of the K Gaussian distributions. Adding the soft memberships $ℤ$, $p(𝕏|𝕐)$ is given by the marginal probability of $p(𝕏,ℤ|𝕐)$ over all possible sets of $ℤ$. This equation, however, is often not solvable in closed forms. Instead, an iterative method is used, that improves $log 𝑝 ( 𝕏 | 𝕐 )$ with each iteration. EM uses a so-called Q-function that indirectly improves $log 𝑝 ( 𝕏 | 𝕐 )$ given current estimates $𝕐^{\left(t\right)}$:

    - The right hand side is the expectation function over $log 𝑝 ( 𝕏 , ℤ | 𝕐 )$ given the conditional distribution of $ℤ$ given $𝕏$ and the current estimates $𝕐^{\left(t\right)}$. Now, the E-step generates this expectation function by computing the probabilities $P\left(x_{i}\right)$ for $ℤ$ (soft assignment) given $𝕏$ and the current estimates $𝕐^{\left(t\right)}$ and uses Bayes’ rule as we have done above. Then, given $ℤ$, the M-step maximizes the Q-function over all possible $𝕐$ to obtain a new estimate $𝕐^{\left(t+1\right)}$. With log-probabilities and Gaussian distributions, we can cancel $log⁡$ and $exp$ in the equation, and solutions are found by solving for the maximum (partial derivative is zero). We omit proof for solutions and convergence.

[MATH_ERROR]

$𝑄 𝕐 𝕐 𝑡 = 𝐸 ℤ | 𝕏 , 𝕐 𝑡 log 𝑝 ( 𝕏 , ℤ | 𝕐 )$

  - Let us reconsider the k-means algorithm as an EM problem. We can re-write the objective function (within-cluster sum of squares, WCSS) as follows:

    - $\gamma _{i,k}$ are the hard assignments of $x_{i}$ to $C_{k}$, i.e., for each $1\leq i\leq N$ exactly one $\gamma _{i,k}=1$ and all others are $0$. We can transform k-means to an EM algorithm over a mixture of K Gaussian distributions with hard assignments as follows:

$J=\sum_{i=1}^{N}\sum_{j=1}^{k}\gamma _{i,k}\left‖𝒙_{i}−𝝁_{k}\right‖_{2}^{2}$

Select initial values for $𝝁_{k}^{(0)}$. Keep $𝚺=𝐈$ and $w_{k}=1/k$ constant

E-step: evaluate new responsibilities $\gamma _{i,k}^{(t)}$ for $1\leq i\leq N$ and $1\leq k\leq K$ using current parameters

M-step: evaluate new parameters $𝝁_{k}^{(t+1)}$ for $1\leq k\leq K$ using current responsibilities

Repeat E-step and M-step until the parameters stop changing

[MATH_ERROR]

$𝝁_{k}^{(t+1)}=\frac{\sum_{i}^{}\gamma _{i,k}^{(t)}∙𝒙_{i}}{\sum_{i}^{}\gamma _{i,k}^{(t)}}$

For both k-means and EM, we need to control then number $K$ of clusters. If the number is too small, the error value is high and the algorithms suffer from underfitting. If we select a large $K$, we can reduce the error but at risk of overfitting. Let $𝕊_{k}$ be the set of data items $𝒙$ that are assigned to cluster $C_{k}$. To control K, we determine the sum of squared errors $SSE$ over all clusters:

  - If we plot this SSE as a function of K, we obtain a graph like on the right side below. As we increase the number $K$, the SSE decreases. However, we cannot simply solve for $K$ that minimizes the SSE function as $K=N$ would have an $SSE=0$ but clearly overfits the data. Rather, we look for the so-called elbow point as highlighted in the figure where the SSE-functions “abruptly” levels out as is decreasing much slower than before the elbow. We can obtain an optimal $K$ in two ways:

    - Vary $K$ from 2 to an upper bound (here 20) and determine the point that lies farthest away from the line between the start and the end of the curve.

    - Start with $K=2$ and determine the distance to the point(2,0). While increasing $K$ observe the distance. Stop ifthe distance starts growing.

  - Method b) has the advantage of iterating less over $K$. For bothvariants to work, we need to normalize the two dimensions, forinstance with a min/max scaling, to obtain a meaningful result.

$SSE(k−means)=\sum_{k=1}^{K}\sum_{𝒙\in 𝕊_{k}}^{}\left‖𝒙−𝝁_{k}\right‖_{2}^{2}$

$SSE(EM)=\sum_{k=1}^{K}\sum_{𝒙\in 𝕊_{k}}^{}\left(𝒙−𝝁_{k}\right)^{T}𝚺_{k}^{−1}\left(𝒙−𝝁_{k}\right)$

(a)

(b)

elbow point
