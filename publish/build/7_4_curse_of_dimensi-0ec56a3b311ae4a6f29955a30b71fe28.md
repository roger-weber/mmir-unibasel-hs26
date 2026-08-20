# Curse of Dimensionality

The methods discussed earlier work well in lower-dimensional spaces, but as dimensionality increases, search performance declines noticeably. Beyond roughly 10 dimensions, a brute-force search, that is calculating distances or similarities between the query and every vector in the index, often outperforms more sophisticated methods, except in cases of heavy data clustering. This phenomenon is commonly referred to as the "curse of dimensionality", and many research efforts have attempted to overcome it, with limited success.

Several studies show that performance declines as the number of dimensions increases. One major reason is that humans find high-dimensional spaces hard to understand, so search methods that work well in low dimensions often fail in higher ones. Properties that are easy to visualize and reason about in two or three dimensions, such as what is near and what is far, behave very differently in high-dimensional vector spaces. This changes how common measures work, such as Euclidean distance and dot-product similarity. Similar problems appear with other distance or similarity measures, but each case may need a slightly different explanation.

High-dimensional spaces have several peculiarities that make search difficult. Distances between points tend to concentrate, meaning that the difference between the nearest and farthest neighbors becomes very small. This reduces the discriminative power of Euclidean distance. Dot-product similarities also tend to cluster around zero, making it harder to distinguish relevant matches. Other challenges include sparsity, where most points are far from each other, and increased storage and computational requirements, as each additional dimension adds complexity. Designing effective data structures and search algorithms in this context requires careful consideration of these high-dimensional effects.

On the following pages we show a few oddities that appear when we apply our 2D and 3D geometric intuition to high dimensional spaces without care. Before you review the solutions on the right, pause and reflect on your own intuition and on how high dimensional geometry differs from everyday experience in lower dimensions.

The mysterious escape of the green sphere

Examine the illustration:

  - The large square has a side length of 2.

  - Each corner of the square has a circle with radius 1, covering all edges of the square.

  - In the center of the square, there is a circle with a radius $r$ that touches all corner circles, captured inside the square

What is the radius $r$ of the inner circle?

What happens when we apply the same scheme in a $d$-dimensional space, with hyperspheres at each corner of the hyper-cube, and a centered (green) hypersphere that touches all corner hyperspheres?

1

1

1

1

$r$

Big is never big enough

Consider the illustration:

  - The gray square has a side length of 1.

  - $𝒄$ is at the center of the square with $𝒄=(0.5, 0.5)$.

  - $𝒑$ is slightly off-center within the square, located at $𝒑=(0.4, 0.4)$.

  - $𝒑$ serves as the center of a circle with a radius $r$ (choose a large value for $r > 1$) that significantly extends beyond the square's boundaries.

What is happening if we increase dimensionality and keep the radius $r$ constant? Assume that $𝒄$ and $𝒑$ have in all dimensions the same component value.

1

1

$𝒄$

$𝒑$

$r$

What means "same direction"?

Consider the illustration:

  - We have embeddings for the objects $𝒅$, $𝒆$, $𝒇$, $𝒈$, and $𝒉$. A query embedding $𝒒$ is provided.

  - For the cosine measure, data point $𝒅$ has the highest similarity due to its smallest angle to the query.

  - For the dot product, data point $𝒆$ holds the highest similarity as it lies on a hyperplane orthogonal to $𝒒$ and none of the other points are positioned "above" this plane.

Generally, with a large number of data points, we can discover strong matches in the same direction (in this case, the upper right quadrant) of the query.

How about in high-dimensional spaces?

$𝒅$

$𝒒$

$𝜶$

$𝒆$

$𝒇$

$𝒈$

$𝒉$

Where have all the data points gone?

Consider the illustration:

  - We have two squares here. The outer square has a side length of 1, and the inner square has a side length of $s$, where $s<1$. The inner square is centered within the outer square.

  - Let's choose $s=0.99$. The inner, green square covers most of the outer square, leaving only a small portion near the outer square's edge uncovered.

  - When data points are evenly distributed in the data space, it's reasonable to assume that the majority of data points are inside the inner, green square, with only a few exceptional points located outside.

Now, how does this change in higher dimensions?

1

1

$s$

$s$

Considering these observations, let's examine the behavior of index structures that partition space, such as the gridfile or the R-tree.

  - Normalization: To simplify the math, we assume a closed data space in the shape of a hypercube of side length 1 . We also assume independent dimensions and uniform distributions along these dimensions, which eliminates the need for dimensionality reduction. We only consider Euclidean distance for this analysis.

  - Observation: In the context of this space, the probability that a data point falls within a subspace is equivalent to the volume of that subspace. The total volume of the space remains 1, regardless of dimensionality.

  - Gridfile: As dimensionality grows, we encounter challenges with the cell-based approach. With increasing dimensions, we have at least $2^{d}$ cells, and even at moderate dimensionality, numerous cells remain empty. For example, with 1 billion data points and $d=10$, most of the $2^{100}$ cells are empty. In addition, the dictionary size expands with $2^{d}$, potentially exceeding available memory. A possible solution is to limit the number of dimensions involved in cell splits.

In the following, we estimate the costs associated with nearest neighbor (NN) searches in hierarchical structures. To do this, we will first calculate the expected distance between the query point and its nearest neighbor. Then, we will make an estimation of the average number of leaf nodes/grid cells retrieved during the search.

  - Since we are utilizing the optimal NN-search algorithm, we can identify the leaf nodes to be accessed as all nodes that intersect with the NN-sphere around the query point.

  - Expected NN-Distance: The expected NN-distance represents the average distance between a query point and its nearest neighbor. The figure on the right side shows the expected NN-distance as it grows with dimensions (for a fixed data set).

  - With the expected NN-distance, we can define a sphere around a randomly selected query point with that distance as the radius. If this sphere intersects with the minimum region of a leaf node, we must visit that node during the NN-search.

  - Finally, we compute the probability that we need to visit, for a random query, a node in the index structure.

Number of leaf nodes / grid cells to visit:

  - As  discussed, it is not feasible to split all dimensions in the gridfile or similar structures utilizing rectangular MBRs. The core reason behind this limitation is that each additional split along a dimension results in a halving of volumes and, in turn, a decrease in the likelihood of a point residing within that MBR. For instance, when dealing with 1 million data points and$ d=14$, the expected number of data points drops to roughly 60 after splitting. While it is possible to perform further splits in dimensions, leaf nodes become increasingly less populated, and their numbers grow almost exponentially with each additional split until each leaf contains exactly one data point

  - Considering the number of leaf nodes to visit, we make the following assumptions: if the tree uses rectangular MBRs, it undergoes splits along $d′$ axes, always dividing in the middle. Consequently, the MBRs of leaf nodes assume the shape shown in the figure on the right below. Let$ l_{max}⁡$ represent the maximum distance between a point in the space and such a leaf node. Given the MBR's shape, this distance can be calculated as $l_{max}=0.5⋅\sqrt{d′}$.

  - Now, let's compare this distance with the expected NN-distance, which leads to intriguing insights. When $d=40$, $l_{max}⁡$ is approximately the expected NN-distance. Furthermore, for $d=100$, $l_{max}⁡$ is much smaller than the expected NN-distance. This is due to the limited number of splits and ensuring we maintain non-empty leaves.

leaf node

$l_{max}$

$1$

$1$

$1$

  - When $l_{max}⁡< NNdist$, a query point lies closer to all leaf nodes than to its nearest neighbor. Consequently, the MBR of each leaf intersects with the NN-sphere. Therefore, an optimal NN-search must visit all leaves to obtain the nearest neighbor for any query in the data space.

This situation raises an important question: why do we employ hierarchical structures if we ultimately need to access all data?

  - It can be shown for most hierarchical data structures, that beyond a relatively small number of dimensions (say 20-50), the method no longer performs better than a brute force search for the nearest neighbor.

Finally, let us examine how dot product similarities and Euclidean distances change as dimensionality increases. Both sets of distance and similarity distributions were obtained through a Monte Carlo simulation using pairs of random vectors. For the dot product, component values lie within the range [-1, 1], and all vectors are normalized so that they lie on the unit hypersphere. For the Euclidean distance, component values lie within [0, 1], and vectors are not normalized.

  - With the dot product and dimensionality $d=2$, similarity scores range from -1 to 1, and many pairs of points have high similarity values above 0.95. As dimensionality increases, the distribution changes sharply. At $d=8192$, nearly all similarity values cluster around 0. As noted earlier, in very high dimensions most points become almost orthogonal, meaning their dot products approach 0. The small differences between values make it increasingly difficult to distinguish between good and poor matches.

  - A similar pattern appears with the Euclidean distance measure. In low dimensions, such as $d=2$, distances span a wide range, making it easy to separate nearby and distant points. As dimensionality increases, for example at $d=8192$, most distances concentrate around a mean value of 36.9 with a standard deviation of 0.24. This means that 99 percent of the distances fall between 36.22 and 37.67, making distance an unreliable indicator of how well an object matches a query.

