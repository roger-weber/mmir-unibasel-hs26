---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Vector Search
section: Distance and Similarity Measures
order: "6.1"
---

(vector-search-distance)=
# Distance and Similarity Measures

Retrieval by meaning rests on a single operation: comparing two vectors. Before we can index millions of them, we need to agree on what "close" means. This section works through the two ways vectors are compared, why they are really the same idea seen from two sides, and how to keep the comparison meaningful when the individual dimensions live on different scales.

## Two kinds of vectors, two notions of closeness

Vectors reach the search engine along two routes. Learned models produce **embeddings**: dense vectors whose direction encodes meaning, as in the previous chapter. Feature extractors read raw media and produce **feature vectors**: a color histogram of an image, the cepstral coefficients of an audio clip, motion statistics of a video shot. The two routes lead to two habits of comparison.

Embeddings are compared with a **similarity**, most often the dot product or cosine, where a *larger* value means a better match. Feature vectors are compared with a **distance**, such as Euclidean distance, where a *smaller* value means a better match. The distinction is a matter of convention rather than of substance, and we will convert freely between the two. What never changes is the task: given a query vector, find the stored vectors that are most similar, or equivalently least distant.

To make the feature-vector side concrete, consider a color histogram. An image is reduced to the fraction of its pixels falling into each color bin. [Figure %s](#fig-color-histograms) shows the three per-channel histograms of one photograph; collecting such counts into a single vector gives a compact descriptor of the image's color content.

```{figure} images/figure_6_2.png
:name: fig-color-histograms
:width: 85%

A color image decomposed into its red, green, and blue channels, each summarized by an intensity histogram. Stacking the bin counts produces a feature vector that describes the image by its colors.
```

Two images with similar color distributions produce nearby histogram vectors, so color-based retrieval ranks candidates by histogram distance. Crucially, the feature and its distance are designed together: a color histogram is meaningless without a rule for comparing two histograms, and that rule is a distance where smaller means more alike.

## Similarity for embeddings: dot product and cosine

For embeddings, the natural comparison is the dot product. Writing the query embedding as $\mathbf{q}$ and a document embedding as $\mathbf{d}_i$, the dot product sums the products of matching components, while cosine similarity divides that sum by the two vector lengths so that only direction matters.

```{important} Key Formula: Dot Product and Cosine Similarity

$$\text{sim}_{\text{dot}}(\mathbf{q}, \mathbf{d}_i) = \mathbf{q} \cdot \mathbf{d}_i = \sum_{k=1}^{M} q_k\, d_{i,k}$$

$$\text{sim}_{\text{cos}}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d}_i \rVert}$$

The dot product rewards vectors that are both aligned and long; cosine similarity divides out the lengths and rewards alignment alone.
```

Both measures have a clean geometric reading. For the dot product, a plane perpendicular to the query separates the highest-scoring documents from the rest; [Figure %s](#fig-dot-product) shows this plane sliding along the query direction. For cosine similarity, the best matches lie inside a cone around the query, shown in [Figure %s](#fig-cosine-cone). A document in the quadrant opposite the query scores negatively under both and is never a good match.

```{figure} images/figure_6_7.png
:name: fig-dot-product
:width: 55%

Dot-product similarity in 2D. A hyperplane orthogonal to the query $\mathbf{q}$ separates the top-scoring documents from the rest; moving the plane outward selects fewer, higher-scoring documents.
```

```{figure} images/figure_6_8.png
:name: fig-cosine-cone
:width: 55%

Cosine similarity in 2D. The best matches lie within a cone around $\mathbf{q}$; the cone's half-angle is set by the similarity threshold, and vector length plays no role.
```

Unlike the mostly non-negative feature vectors of classical retrieval, embedding components can be positive or negative. A component pair increases the dot product when the two share a sign and decreases it when they differ. This has a practical consequence that separates embeddings from the sparse term vectors of [](#classical-text-retrieval): we cannot ignore a document just because it disagrees with the query on some component. With an inverted file, a query term that a document lacks contributes nothing and the document can be skipped. With dense embeddings, every component contributes, and a small query component $q_k$ still matters when the matching document component $d_{i,k}$ is large.

```{note} Example
Take the query $\mathbf{q} = (1, 2)$ and two documents $\mathbf{d}_1 = (2, 4)$ and $\mathbf{d}_2 = (5, 3)$.

The dot products are $\mathbf{q} \cdot \mathbf{d}_1 = 1\cdot 2 + 2\cdot 4 = 10$ and $\mathbf{q} \cdot \mathbf{d}_2 = 1\cdot 5 + 2\cdot 3 = 11$, so the dot product ranks $\mathbf{d}_2$ first.

The cosine values tell a different story. Since $\mathbf{d}_1 = 2\mathbf{q}$ points in exactly the query's direction, $\text{sim}_{\text{cos}}(\mathbf{q}, \mathbf{d}_1) = 10 / (\sqrt{5}\cdot\sqrt{20}) = 1.0$, whereas $\text{sim}_{\text{cos}}(\mathbf{q}, \mathbf{d}_2) = 11 / (\sqrt{5}\cdot\sqrt{34}) = 0.844$. Cosine ranks $\mathbf{d}_1$ first. The document $\mathbf{d}_2$ wins on the dot product only because it is longer, not because it is better aligned.
```

## Normalization: one measure instead of two

The example hints at a simplification. If we divide every vector by its length before storing it, all vectors lie on the unit sphere, their lengths are all $1$, and the dot product of two normalized vectors equals their cosine similarity. We can then use the dot product everywhere and forget about the division at query time.

```{important} Key Formula: Length Normalization

$$\hat{\mathbf{d}} = \frac{\mathbf{d}_i}{\lVert \mathbf{d}_i \rVert}, \qquad \hat{\mathbf{q}} = \frac{\mathbf{q}}{\lVert \mathbf{q} \rVert}, \qquad \text{sim}_{\text{cos}}(\mathbf{q}, \mathbf{d}_i) = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}}$$

Normalize once when building the index; cosine similarity then reduces to a plain dot product on the unit sphere.
```

This is the standard trick behind most vector indexes: store normalized embeddings, run dot products. It also makes the two vector worlds meet, because a dot product is trivially turned into a distance, as we see next.

## Distance for feature vectors

Feature vectors are compared with a distance $\delta(\mathbf{q}, \mathbf{p}_i)$, and several choices are common. Each induces a different notion of a neighborhood, that is, a different shape for the set of points within a fixed distance of a center. [Figure %s](#fig-distance-metrics) shows these shapes in 2D.

- **Euclidean distance** (L2) measures straight-line distance and gives a round neighborhood. It is the default for continuous features.
- **Manhattan distance** (L1) sums the absolute differences per dimension and gives a diamond neighborhood. It emphasizes many small differences.
- **Maximum distance** (L-infinity) takes the single largest component difference and gives a square neighborhood. It lets the worst dimension dominate.
- **Quadratic distance** weights and correlates the dimensions through a matrix, giving a tilted-ellipse neighborhood.

$$\delta_{\text{L2}}(\mathbf{q}, \mathbf{p}_i) = \sqrt{\sum_j (q_j - p_{i,j})^2}, \qquad \delta_{\text{L1}}(\mathbf{q}, \mathbf{p}_i) = \sum_j |q_j - p_{i,j}|$$

$$\delta_{\text{max}}(\mathbf{q}, \mathbf{p}_i) = \max_j |q_j - p_{i,j}|, \qquad \delta_{\text{quad}}(\mathbf{q}, \mathbf{p}_i) = (\mathbf{q} - \mathbf{p}_i)^\top \mathbf{A} (\mathbf{q} - \mathbf{p}_i)$$

```{figure} images/figure_6_9.png
:name: fig-distance-metrics
:width: 80%

Neighborhood shapes of four distance measures in 2D: Euclidean (round), Manhattan (diamond), maximum (square), and quadratic (tilted ellipse). The shape decides which points count as close.
```

```{note} Example
Represent three images by three-bin color histograms giving the fraction of red, green, and blue pixels. The query is a sunset $\mathbf{q} = (0.5, 0.4, 0.1)$, and the candidates are another sunset $\mathbf{p}_1 = (0.6, 0.3, 0.1)$ and an ocean scene $\mathbf{p}_2 = (0.1, 0.3, 0.6)$.

Euclidean distances: $\delta_{\text{L2}}(\mathbf{q}, \mathbf{p}_1) = \sqrt{0.01 + 0.01 + 0} = 0.141$ and $\delta_{\text{L2}}(\mathbf{q}, \mathbf{p}_2) = \sqrt{0.16 + 0.01 + 0.25} = 0.648$. The sunset $\mathbf{p}_1$ is far closer, exactly as color-based retrieval should decide. Manhattan distances agree: $0.2$ for $\mathbf{p}_1$ against $1.0$ for $\mathbf{p}_2$. Here the smaller distance marks the better match.
```

One awkward feature of distances is that they are hard to read on their own. A cosine of $0.9$ clearly signals a strong match, but a distance of $32.8$ says nothing until it is compared with other distances. This is one reason normalized similarities are convenient when a human or a downstream system needs to interpret the score.

### Keeping dimensions comparable

Distances break down when dimensions live on different scales. Suppose one feature ranges over $[0, 1]$ and another over $[0, 1000]$. Euclidean distance is then dominated by the second dimension, and the first barely influences the result. The fix is **Gaussian normalization**: for each dimension, subtract its mean $\mu_j$ and divide by its standard deviation $\sigma_j$, estimated once from the collection or a large sample.

$$\hat{v}_j = \frac{v_j - \mu_j}{\sigma_j}$$

Applying the same transform to queries and documents recentres the data and rescales every dimension to unit variance. Equivalently, it turns Euclidean distance into a weighted distance with weights $w_j = 1/\sigma_j$, so each dimension contributes in proportion to how much it actually varies.

Gaussian normalization treats the dimensions as independent. When they are correlated, the **quadratic distance** generalizes the idea through a positive semi-definite matrix $\mathbf{A}$, whose eigenvectors define the main axes of the data and whose eigenvalues scale them. Consider color histograms again: the bin for red is related to the bin for orange, because the two colors are perceptually close, while red and blue are not. The matrix $\mathbf{A}$ captures such relationships.

Computing $(\mathbf{q} - \mathbf{p}_i)^\top \mathbf{A} (\mathbf{q} - \mathbf{p}_i)$ for every candidate is expensive, but an eigenvalue decomposition avoids it. Rotating and scaling the space along the eigenvectors of $\mathbf{A}$, shown in [Figure %s](#fig-whitening), maps the quadratic distance in the original space onto a plain Euclidean distance in the transformed space. We rotate all vectors once when building the index, then compare them with ordinary Euclidean distance at query time.

```{figure} images/figure_6_10.png
:name: fig-whitening
:width: 75%

Whitening. Rotating along the eigenvectors of the weighting matrix and scaling by the eigenvalues turns an elongated, correlated distribution into an isotropic one, where quadratic distance becomes ordinary Euclidean distance.
```

```{warning} Normalize on statistics, not on a single query
Gaussian normalization uses the mean and standard deviation of the whole collection, not of one query or one result set. Estimate $\mu_j$ and $\sigma_j$ once from a representative sample and keep them fixed, so that the same vector always maps to the same normalized point regardless of which query retrieves it.
```

## The search problem, stated once

Whether we maximize a similarity or minimize a distance, the retrieval task is the same. For embeddings we seek the most similar vector; for feature vectors we seek the nearest neighbor. And because any similarity can be flipped into a distance, and any distance into a similarity, the two problems are interchangeable.

```{important} Key Formula: Best-Match and Nearest-Neighbor Search

Given a query $\mathbf{q}$ and a set $\mathbb{P}$ of stored vectors, the **best match** under a similarity and the **nearest neighbor** under a distance are

$$\mathbf{d}^{*} = \arg\max_{\mathbf{d}_i \in \mathbb{P}} \text{sim}(\mathbf{q}, \mathbf{d}_i), \qquad \text{NN}(\mathbf{q}) = \arg\min_{\mathbf{p}_i \in \mathbb{P}} \delta(\mathbf{q}, \mathbf{p}_i)$$

More generally we return the top $k$ vectors, the $k$ with the highest similarity or the smallest distance.
```

The rest of the chapter answers a single question about these definitions: how do we compute the arg-max or arg-min without comparing the query to every stored vector? For a few thousand vectors a full scan is fine. For millions or billions it is not, and the structures that speed up the search behave very differently depending on how many dimensions the vectors have.

```{hint} Hands-on: Distance measures and normalization
Compute dot product, cosine, and Euclidean and Manhattan distances on the same vectors, watch length normalization make cosine and dot product agree, and see Gaussian normalization rebalance dimensions on different scales.
[Open notebook ->](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch06-01-distances-and-dimensionality.ipynb)

*Includes pre-run results: you can read through or download and experiment.*
```
