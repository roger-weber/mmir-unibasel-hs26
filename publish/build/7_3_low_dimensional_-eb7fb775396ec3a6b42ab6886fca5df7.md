# Low-dimensional Search Structures

A Voronoi diagram is a geometric structure that divides a space into regions based on the closest distance to a set of seed points. Voronoi diagrams are particularly useful in various spatial applications, such as geographical information systems (GIS), computational geometry, nearest neighbor search, and are widely used in algorithms and data structures for spatial data analysis and optimization.

  - Given a set of seed points, a Voronoi diagram partitions space so each region contains all locations closest to one seed point. Each region then represents precomputed results of nearest neighbor searches. Instead of storing data points directly, we compute the Voronoi diagram and index its regions. To find the nearest neighbor for a query point, we find which region that contains the query point and return that region's seed point as the answer. Voronoi diagrams can be built for different distance measures, not only for Euclidean distances shown below.

  - Two main challenges are computing the Voronoi diagram, especially in higher dimensions, and storing the cells so point-containment queries are fast. In the example below, some regions are bounded by six or more lines, which raises storage needs exponentially with dimensionality. To reduce storage costs, we can approximate a Voronoi cell by its minimum bounding rectangle. That may require checking several regions instead of just one to find the nearest point. The extra work is small in low dimensions but can grow quickly in higher dimensions, as we will see.

  - Voronoi diagrams function somewhat like inverted files in text retrieval, providing answers to all potential queries similar to inverted files for single-term queries. However, it is not an ideal data structure due to complex insertion and removal operations, and limited scalability in higher dimensions.

Voronoi Cell: all points in the cell are closest to blue center point
A gridfile is a spatial data structure used to organize and index multidimensional data efficiently. It works by dividing the multidimensional space into a grid of cells, where each cell is represented by a bucket:

  - Grid Partitioning: the initial step is to divide the multidimensional space into a grid, where the grid cells create a rectangular layout. The gridlines may not be equally spaced, but the division is designed to ensure that the resulting pages on disk hold roughly the same number of data points.

  - Dictionary: grid cells are organized in a dictionary, which associates a cell with a disk page holding the cell's data points. Disk pages can consolidate data from multiple cells to optimize storage when many cells remain empty due to data distribution.

  - Insertions: when adding new data points to the gridfile, disk pages may become full. In this scenario, two new pages are created, and cells are evenly distributed between them. If a cell grows too large for a disk page, the gridfile introduces a new partition line along a chosen dimension, creating new cells. This update affects the dictionary but only necessitates redistributing the data points from the overflowing cell to the new disk pages.

  - Query Processing: to perform a range query, the gridfile divides the query space into the corresponding grid cells and retrieves data points from these cells. The advantage of this approach is that it significantly reduces the search space, making queries faster. For nearest neighbor search, we will explore a generic algorithm after the next method that is applicable to the gridfile as well.

  - Gridfiles are particularly useful for efficient spatial data indexing in databases. They are commonly employed in geographic information systems (GIS), image databases, and various applications where multidimensional data needs to be quickly retrieved based on spatial attributes. Gridfiles are more suitable for moderate to high-dimensional data compared to Voronoi diagrams, as they offer better scalability and performance in higher dimensions. In lower dimensions, they offer quick responses with a few data page reads.

1

2

3

1

2

3

An R-tree is a tree-based spatial data structure used for indexing and searching multidimensional data in a space, particularly in geographic information systems (GIS) and database systems. It is designed to efficiently store and query spatial objects like points, rectangles, and polygons similar to a B-tree in conventional databases:

  - Structure: an R-tree is a balanced tree structure where each node represents a bounding rectangle that encloses a set of data points or other bounding rectangles. The root of the tree encompasses all the objects.

  - Hierarchical Organization: the tree's hierarchical organization means that as you traverse the tree from the root to the leaves, you progressively narrow down the search space. Data points typically reside in leave nodes only.

  - Insertion: when you insert a datapoint into an R-tree, the structure determines where to place it within the tree. It selects a leaf that can accommodate the new object without violating the tree's balance and spatial structure.

  - Splitting and Reorganization: if a node becomes too full, it may be split into two, and the parent node is updated. This may lead to consecutive splits up to the root if also parent nodes become full. If the root node overflows, it is split into two parts and a new root is created. R-trees are designed to keep their structure balanced. This means that each node contains a roughly equal number of data points or child nodes, which ensures that the tree remains efficient for search operations.

  - Query Processing: to perform a query, such as a range or nearest neighbor search, you start at the root and traverse the tree, visiting only those nodes that overlap or are close to the query region. This significantly reduces the number of objects to be considered, making queries much faster than a brute-force search.

R-Tree

Splitting nodes in an R-tree can lead to inefficiencies:

  - Splitting leaf nodes is straightforward: data points are divided along a dimension using the median value. This split guarantees that the resulting leaf nodes cover non-overlapping areas, ensuring efficient search operations by preventing overlaps that might require visiting multiple nodes during insertion and searches.

  - Splitting inner nodes can lead to overlaps, as it is often not feasible to consistently separate the minimum bounding rectangles of child nodes into 2 non-overlapping areas, as illustrated in the examples below:

  - Certain R-tree variants minimize overlaps by re-inserting data points from leaves with substantial overlap. Another approach involves bottom-up reconstruction of the R-tree to eliminate leaf-level overlaps, which subsequently prevents overlaps in inner nodes.

Option 1

Option 2

good case

‘ok’ case

bad case

The minimum bounding regions are overlapping

One region is completely contained within the other region

Overlaps are a problem: Overlaps force searches to examine multiple child nodes that share the same regions. Furthermore, overlaps make insertions unpredictable and can cause more overlaps if the insertion path is poor, because leaf nodes may grow and start overlapping with neighboring leaves.

R-tree extensions: Over time, several extensions and optimizations for R-trees have been introduced. Key optimization aspects include:

  - Minimum bounding region shapes (e.g., rectangles, spheres, or combinations)

  - Splitting methods to reduce leaf node overlaps

  - Adjusting node sizes, such as increasing page size if splitting is unbeneficial

  - Metric Trees, which rely on object metrics (e.g., edit distance) rather than a data space metric (like Euclidean)

  - Notable examples include R+-Tree (1987), R*-Tree (1990), P-Tree (1990), TV-Tree (1994), vp-Tree (1994), GiST (1995), X-Tree (1996), SS-Tree (1996), SS+-Tree (1997), SR-Tree (1997), M-Tree (1997), Pyramid-Tree (1998), DABS-Tree (2000), P-Sphere Tree (2000), and more.

Which path to follow?

New point to insert

good case: follow blue path

bad case: follow green path

Hjaltson and Samet proposed an optimal search algorithm for identifying the nearest neighbor within hierarchical structures. "Optimal" in this context implies that the algorithm minimizes the number of visited nodes, ensuring the correctness of the nearest neighbor found (which cannot be achieved with fewer visits).

  - The algorithm employs a priority queue for nodes and points. Priority is determined by the distance between the query point and data point or the minimal bounding region (MBR). The queue is sorted by distances in ascending order. The algorithm operates as follows for a given query object $𝒒$:

    - Initialization: the root node is added to the queue with the distance of its MBR to $𝒒$

    - As long as the queue is not empty, fetch the top element of the queue  $𝒑$

      - If $𝒑$ is a data object, then $𝒑$ is the nearest neighbor to $𝒒$

      - If $𝒑$ is a leaf node, insert all contained data points with their distances to $𝒒$

      - If $𝒑$ is an inner node, insert all its child nodes with their distances to $𝒒$

  - The algorithm uses only distance measurements between objects and between an object and a node. For example, if a node is represented as a minimum bounding rectangle (MBR), it computes the minimal distance between a point and that rectangle. The algorithm also works for generalized search trees (GiST) with arbitrary distance measures.

  - Proof of correctness: The priority queue orders entries by increasing distance. Because nodes use minimum bounding regions, every child node or object has a distance to the query that is equal to or greater than its parent node's distance. Thus, when a data object is at the top of the queue, every unvisited node and all their children and descendants are at least as far from the query and therefore not closer than that object at the top of the queue.

Nearest neighbor to q

q

NN-sphere

  - Proof of optimality: Assuming we already know the nearest neighbor of the query object $𝒒$, we define the Nearest Neighbor Sphere (NN-Sphere) as the circle centered at $𝒒$ that passes through that nearest neighbor. To be correct, the algorithm must consider all nodes that intersect the NN-Sphere, that is, nodes that contain points closer to $𝒒$ than the current nearest neighbor, because they might hold a better answer. In the example, the red rectangle must be considered, while the blue circle does not need to be checked.

  - The algorithm visits nodes in ascending order of their distance to $𝒒$. When an object reaches the top of the queue, it is the nearest neighbor. We can continue the search to find the k nearest neighbor by repeating the steps until k points were found at the top of the queue.

Grid cells and minimum bounding regions approximate data point positions. These approximations let us set lower and upper bounds on distances and similarities for objects inside those regions. This makes it possible to apply the nearest neighbor algorithm to any structure that underestimates distances or overestimates similarity for data points within larger "containers". The figure below illustrates this idea for the L2 metric and the dot product.

  - To compute lower and upper bounds on distances between a query and a minimum bounding region, the shape of the regions plays a crucial role. In the case of spheres and L2-distances, we derive the lower and upper bounds by taking the distance between the query and the sphere's center and then subtracting (lower bound) or adding (upper bound) the sphere's radius (lower bound is zero if query is within sphere). Likewise, we can establish bounds on the dot-product of points within a sphere (as shown in the right-hand figure). These bounds are determined by the hyperplane that is orthogonal to the query vector and touches the sphere on both sides.

  - We often consider rectangular bounding regions and utilize L1/L2/Lp-distances and dot-products. Because these measures rely on component-wise operations, we can determine lower and upper bounds by minimizing and maximizing these component-wise operations. Let $q_{j}$ represent the query component for dimension $j$, and $l_{j}$ and $u_{j}$ define the rectangle's boundaries. For the L2-distance, we establish boundaries as follows (if the query component falls in between the rectangular boundaries, the lower bound contribution is 0):

q

q

L2-distance

dot-product

$lBnd\left(q,MBR\right)=\sqrt{\sum_{j=0}^{d−1}\left\{\begin{matrix}\left(l_{j}−q_{j}\right)^{2}&q_{j}<l_{j}\\\left(q_{j}−u_{j}\right)^{2}&q_{j}>u_{j}\\0&l_{j}\leq q_{j}\leq u_{j}\end{matrix}\right.}$

$uBnd\left(q,MBR\right)=\sqrt{\sum_{j=0}^{d−1}\left\{\begin{matrix}\left(u_{j}−q_{j}\right)^{2}&q_{j}\leq \frac{l_{j}+u_{j}}{2}\\\left(q_{j}−l_{j}\right)^{2}&q_{j}>\frac{l_{j}+u_{j}}{2}\end{matrix}\right.}$

  - Similarly, we can under- and overestimate dot-product similarities as follows:

In conclusion, Voronoi diagrams, gridfiles, and R-trees are all important spatial data structures, each with its own set of advantages and applications. However, their performance can be significantly impacted as the dimensionality of the data increases, a challenge commonly referred to as the "curse of dimensionality".

  - Voronoi Diagrams are powerful tools for spatial partitioning and proximity analysis. They work effectively in lower dimensions, helping with tasks like nearest neighbor searches and spatial analysis. However, as dimensionality increases, Voronoi diagrams become less practical due to the rapid expansion of the space and the exponential growth in the number of regions and computational complexity.

  - Gridfiles provide a structured way to index multidimensional data, offering a balance between storage efficiency and query performance. They are useful for lower-dimensional data where grid partitioning can efficiently reduce search space. In higher dimensions, Gridfiles face limitations as the grid becomes overly fine, leading to extensive storage requirements for the dictionary and inefficient searches as we explain next.

  - R-trees are hierarchical spatial data structures that excel in organizing and searching spatial data. They are versatile and can be used for various types of spatial data, from points to polygons. While R-trees are suitable for moderate dimensions, they also struggle with high-dimensional data due to issues like overlaps and complex splits, making them less efficient as dimensionality increases.

Spatial data structures are valuable tools for managing and analyzing data in lower dimensions. However, as dimensionality grows, they face serious  limitations that makes them impractical for many of the scenarios in semantic search and similarity search in general. This phenomenon, known as the "curse of dimensionality", highlights the need for specialized techniques and data structures to handle high-dimensional data effectively.

$lBnd\left(q,MBR\right)=\sum_{j=0}^{d−1}\left\{\begin{matrix}&\\q_{j}∙l_{j}&q_{j}\geq 0\\q_{j}∙u_{j}&q_{j}<0\end{matrix}\right.$

$uBnd\left(q,MBR\right)=\sum_{j=0}^{d−1}\left\{\begin{matrix}&\\q_{j}∙u_{j}&q_{j}\geq 0\\q_{j}∙l_{j}&q_{j}<0\end{matrix}\right.$
