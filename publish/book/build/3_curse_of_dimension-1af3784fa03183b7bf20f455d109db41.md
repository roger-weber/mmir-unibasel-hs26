---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Vector Search
section: The Curse of Dimensionality
order: "6.3"
---

(vector-search-curse-of-dimensionality)=
# The Curse of Dimensionality

The structures of the previous section prune the search by reasoning about distances. That reasoning quietly assumes that some points are near and others are far. In high-dimensional spaces this assumption breaks down. Beyond roughly ten dimensions, a brute-force scan usually beats the clever index, and beyond a few dozen the gap is hopeless. This section explains why, first through geometry that defies intuition, then through the cost of the optimal algorithm, and finally through what happens to the distances themselves.

## When intuition stops working

We reason about space using two and three dimensions, where "near" and "far" behave as expected. High-dimensional geometry does not obey that intuition. Four short puzzles make the point; try to predict each answer before reading the resolution.

**The escaping sphere.** Place a hypercube of side $2$ with a unit sphere at each corner, and a further sphere at the center just touching them. In $d$ dimensions the center-to-corner distance is $\sqrt{d}$, so the central sphere has radius $r = \sqrt{d} - 1$, illustrated for the 2D case in [Figure %s](#fig-escaping-sphere). At $d = 2$ this is about $0.41$; at $d = 4$ it is exactly $1$ and the central sphere reaches the cube's faces; and for $d = 9$ it is $2$, so the "inner" sphere pokes outside the cube that supposedly contains it.

```{figure} images/figure_6_20.png
:name: fig-escaping-sphere
:width: 45%

The escaping-sphere puzzle in 2D. Corner spheres of radius $1$ leave a central sphere of radius $\sqrt{d}-1$, which grows past the cube's faces once $d > 4$.
```

**Fixed offset, growing distance.** Take two points that differ by $0.1$ in every dimension. Their Euclidean distance is $0.1\sqrt{d}$, so points with a tiny, fixed per-dimension gap drift arbitrarily far apart as $d$ grows, as suggested by [Figure %s](#fig-fixed-offset). A radius that captured a neighbor in low dimensions captures nothing in high ones.

```{figure} images/figure_6_21.png
:name: fig-fixed-offset
:width: 45%

A ball of fixed radius $r$ around a point $\mathbf{p}$ close to the center $\mathbf{c}$. As dimensions are added, the distance between two fixed points grows like $\sqrt{d}$ and outruns any fixed radius.
```

**Losing direction.** In low dimensions many points share the query's rough direction, and cosine similarity separates them cleanly, as in [Figure %s](#fig-same-direction). In high dimensions almost all random vectors are nearly orthogonal to the query, so "the same direction as the query" describes hardly any point at all.

```{figure} images/figure_6_22.png
:name: fig-same-direction
:width: 45%

Candidate vectors around a query. In two dimensions several points fall near the query's direction; in high dimensions almost every vector is close to orthogonal.
```

**Everything lives near the edge.** Put a smaller square of side $s = 0.99$ inside the unit square. It seems to cover almost everything, and in 2D it holds $0.99^2 \approx 98\%$ of the volume. In $d$ dimensions it holds $0.99^{d}$: about $37\%$ at $d = 100$ and $0.004\%$ at $d = 1000$. [Figure %s](#fig-points-gone) shows the setup. Nearly all of the volume, and so nearly all of the data, concentrates in the thin shell near the boundary.

```{figure} images/figure_6_23.png
:name: fig-points-gone
:width: 45%

An inner square of side $s$ inside the unit square. Its share of the volume is $s^{d}$, which collapses toward zero as $d$ grows, pushing the data into the boundary shell.
```

## Why the optimal algorithm ends up visiting everything

These oddities have a direct cost. Consider a space-partitioning index over uniform data in the unit hypercube, searched by the optimal algorithm from the previous section, which visits every leaf whose bounding region intersects the NN-sphere.

Two quantities compete. The first is the **expected nearest-neighbor distance**, which grows with dimension as [Figure %s](#fig-nn-distance) shows, because points spread out as dimensions are added. The second is the size of a leaf's bounding rectangle. A tree cannot split every dimension: each split halves the points in a cell, so after splitting along $d'$ dimensions in the middle, the maximum distance from a query to such a leaf is $l_{\max} = 0.5\sqrt{d'}$, sketched in [Figure %s](#fig-leaf-node). Because $d'$ is limited by the need to keep leaves non-empty, $l_{\max}$ cannot keep pace with the growing NN-distance.

```{figure} images/figure_6_24.png
:name: fig-nn-distance
:width: 60%

The expected nearest-neighbor distance under Euclidean distance grows roughly as $\sqrt{d}$: even the closest point drifts away as dimensions accumulate.
```

```{figure} images/figure_6_25.png
:name: fig-leaf-node
:width: 50%

A leaf cell in the unit cube. Splitting along $d'$ dimensions bounds the query-to-leaf distance by $l_{\max} = 0.5\sqrt{d'}$.
```

Once $l_{\max}$ falls below the expected NN-distance, every leaf lies closer to the query than the query's own nearest neighbor, so every leaf's bounding region intersects the NN-sphere and the algorithm must visit all of them. The crossover arrives early: around $d = 40$ the two are comparable, and by $d = 100$ the leaf distance is far smaller, forcing a full traversal. This is why, beyond twenty to fifty dimensions, a hierarchical index is no faster than the brute-force scan it was meant to replace, and often slower once its own overhead is counted.

## The distances themselves collapse

The deepest problem is not the index but the measure. As dimensions grow, distances and similarities between random points cluster so tightly that the nearest and farthest neighbors become almost indistinguishable.

```{important} Key Formula: Distance Concentration

As $d \to \infty$, the spread of pairwise distances shrinks relative to their mean:

$$\frac{\sigma_d}{\mu_d} \longrightarrow 0$$

When the standard deviation is negligible against the mean, "nearest" and "farthest" carry almost the same distance, and nearest-neighbor search loses its meaning.
```

Two Monte-Carlo experiments make this visible. For the dot product of random unit vectors, [Figure %s](#fig-dot-collapse) shows the similarity distribution narrowing from a broad spread at $d = 2$ to a spike at zero by $d = 8192$: almost all pairs are nearly orthogonal, and the values that should separate good matches from bad ones differ only in the third decimal. For Euclidean distance, [Figure %s](#fig-euclidean-collapse) shows the same collapse: at $d = 8192$ the distances concentrate around a mean of about $36.9$ with a standard deviation near $0.24$, so $99\%$ of all pairs fall within a band of width barely over one unit. Distance has stopped discriminating.

```{figure} images/figure_6_26.png
:name: fig-dot-collapse
:width: 90%

Dot-product similarity between random unit vectors, from $d = 2$ to $d = 8192$. The distribution concentrates on zero: in high dimensions almost every pair is nearly orthogonal.
```

```{figure} images/figure_6_27.png
:name: fig-euclidean-collapse
:width: 90%

Euclidean distances between random points across the same dimensions. The distances pile up around their mean, so the closest and farthest points differ by almost nothing.
```

There is a way out, and it is the subject of the rest of the chapter. Exact nearest-neighbor search in high dimensions is both expensive and, given how little distances vary, of questionable value. The applications that use embeddings rarely need the exact nearest neighbor; a result ranked third in the embedding space is usually just as useful to the user. Giving up exactness on purpose buys back the speed that the curse of dimensionality took away.
