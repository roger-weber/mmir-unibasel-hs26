---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Vector Search
section: Low-Dimensional Search Structures
order: "6.2"
---

(vector-search-low-dimensional)=
# Low-Dimensional Search Structures

A full scan compares the query against every stored vector. For a small collection that is fine, but the cost grows with the collection size, so we look for index structures that visit only a fraction of the data. This section surveys the classical spatial indexes and the search algorithm that runs on all of them. They work well in a handful of dimensions. Understanding exactly how they work is what lets us see, in the next section, why they fail as dimensions grow, and that failure is what motivates approximate search.

## Precomputing every answer: Voronoi diagrams

The most direct idea is to precompute the answer for every possible query. A **Voronoi diagram** partitions the space into one cell per stored point, where a cell contains all locations closer to its point than to any other. [Figure %s](#fig-voronoi) shows such a partition. To answer a nearest-neighbor query, we find which cell the query falls into and return that cell's point. This is the geometric analogue of the inverted file: just as an inverted file precomputes the postings for every term, a Voronoi diagram precomputes the nearest neighbor for every location.

```{figure} images/figure_6_12.png
:name: fig-voronoi
:width: 60%

A Voronoi diagram. Every location inside a cell is closer to that cell's seed point than to any other, so locating the query's cell answers the nearest-neighbor query directly.
```

The trouble is building and storing the diagram. In two dimensions it is elegant, but the number of cell faces grows quickly with the dimension, and locating the containing cell becomes as expensive as the search it was meant to replace. Insertions and deletions are awkward too, since adding one point can reshape many neighboring cells. Voronoi diagrams are a clean mental model of exact nearest-neighbor search, not a practical index beyond low dimensions.

## Partitioning space: grid files

A **grid file** takes the opposite approach. Instead of one cell per point, it lays a coarse grid over the space and stores the points of each cell on a disk page, as shown in [Figure %s](#fig-grid-file). A directory maps grid cells to pages, and several sparse cells can share a page to avoid waste. A range query touches only the cells overlapping the query region, which prunes most of the collection cheaply.

```{figure} images/figure_6_13.png
:name: fig-grid-file
:width: 80%

A grid file partitions the space into cells (left) and maps each cell through a directory to a disk page holding its points (right). A query reads only the pages of the overlapping cells.
```

Grid files handle insertions gracefully by splitting a page and adding a partition line when a page overflows. They serve low-dimensional spatial data well. Their weakness is the directory: in $d$ dimensions a grid needs at least $2^{d}$ cells, so even at moderate dimension most cells are empty and the directory outgrows memory. This is the first hint of the trouble ahead.

## Bounding regions: R-trees

Rather than partition the whole space in advance, an **R-tree** groups nearby points into a **minimum bounding rectangle** (MBR), groups nearby rectangles into larger rectangles, and continues up to a single root, giving a balanced tree much like a B-tree for spatial data. [Figure %s](#fig-rtree) shows a small R-tree and the nested rectangles it represents. A query descends only into the branches whose bounding rectangle it could plausibly contain a good answer for, and prunes the rest.

```{figure} images/figure_6_14.png
:name: fig-rtree
:width: 85%

An R-tree groups points into nested minimum bounding rectangles (left) that form a balanced tree (right). Search descends only into rectangles that might hold a closer point.
```

R-trees index points, rectangles, and polygons, and are a mainstay of spatial databases. Their characteristic difficulty is overlap: when the bounding rectangles of sibling nodes overlap, a query may have to follow several branches, and the pruning weakens.

```{note} How R-trees split and where points go (optional reading)
:class: dropdown

The quality of an R-tree depends on how nodes are split when they overflow. Splitting a leaf is easy: divide the points along a dimension at the median, and the two resulting rectangles do not overlap. Splitting an inner node is harder, because its entries are already rectangles and separating them cleanly is often impossible. The outcomes range from a clean split with no overlap, through a tolerable split with partial overlap, to a pathological split where one rectangle sits entirely inside the other.

Insertion faces a related choice, shown when a new point falls inside several overlapping rectangles at once. The standard heuristic descends into the child whose rectangle needs the least enlargement to contain the point, keeping rectangles tight; a careless choice inflates a rectangle and creates overlap that slows every later query.

Decades of variants attack these problems: the R+-tree, R*-tree, X-tree, SS-tree, SR-tree, and M-tree, among others, differ in the shape of the bounding region, the split strategy, and whether they rely on a data-space metric or an object metric such as edit distance. They improve the constants but not the asymptotic behavior that the next section describes.
```

## One algorithm for all of them

Grid files and R-trees share a search algorithm, and it is provably optimal in the number of nodes it visits. The idea is to explore containers in order of their distance to the query, using a priority queue.

The algorithm keeps a queue of points and nodes, ordered by their distance to the query $\mathbf{q}$. It starts with the root and repeatedly removes the closest entry. If the entry is a data point, it is the nearest neighbor and the search stops. If it is a node, its children (or the points it contains) are inserted with their distances, and the loop continues. To find the $k$ nearest neighbors, we keep removing entries until $k$ data points have surfaced.

The reason this is correct and optimal is the **NN-sphere**: the sphere centered at the query passing through the current nearest neighbor, drawn in [Figure %s](#fig-nn-sphere). Any node whose bounding region does not intersect this sphere cannot hold a closer point, so it is pruned. Ordering the queue by distance guarantees that when a data point reaches the front, every unvisited node is at least as far away, so no closer point can remain hidden. The algorithm visits exactly the nodes whose regions intersect the NN-sphere, and no correct algorithm can visit fewer.

```{figure} images/figure_6_18.png
:name: fig-nn-sphere
:width: 60%

Nearest-neighbor search by pruning. Clusters whose bounding region lies outside the NN-sphere around $\mathbf{q}$ cannot hold a closer point and are skipped; only intersecting regions are searched.
```

The algorithm needs only one primitive: a distance from the query to a bounding region. For a rectangle and Euclidean distance, the closest possible point in the rectangle gives a lower bound, and the farthest gives an upper bound.

```{important} Key Formula: Distance Bound to a Bounding Rectangle

For a rectangle with per-dimension bounds $[l_j, u_j]$, the smallest possible L2 distance from the query $\mathbf{q}$ is

$$\text{lBnd}(\mathbf{q}, \text{MBR}) = \sqrt{\sum_{j} \begin{cases} (l_j - q_j)^2 & q_j < l_j \\ (q_j - u_j)^2 & q_j > u_j \\ 0 & l_j \le q_j \le u_j \end{cases}}$$

If this lower bound already exceeds the current nearest-neighbor distance, the whole rectangle can be skipped.
```

The same reasoning bounds a dot product: the hyperplane orthogonal to the query, touching the region on either side, gives the smallest and largest possible similarity. [Figure %s](#fig-mbr-bounds) contrasts the two cases. Any container that can under-estimate distance or over-estimate similarity works with this algorithm, which is why one implementation serves grid files, R-trees, and their many descendants.

```{figure} images/figure_6_19.png
:name: fig-mbr-bounds
:width: 80%

Bounding the query-to-region distance. For L2 distance (left) the nearest and farthest corners bound it; for the dot product (right) two hyperplanes orthogonal to the query bound the similarity.
```

These structures share more than an algorithm; they share a fate. Voronoi diagrams, grid files, and R-trees all excel in low dimensions and all degrade as dimensions rise: the Voronoi diagram explodes in size, the grid directory outgrows memory, and R-tree rectangles overlap until pruning stops helping. The next section explains why this is not a limitation of any one structure but a property of high-dimensional space itself.
