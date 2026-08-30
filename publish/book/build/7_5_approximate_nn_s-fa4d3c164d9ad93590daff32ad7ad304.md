# Approximate NN-Search

For small datasets, doing a full vector scan, which means comparing the query embedding to every stored embedding, works well and gives exact results. As the dataset grows to millions or billions of vectors, this exhaustive method becomes computationally infeasible. Comparing every vector soon exceeds practical limits for both time and memory.

  - To address this scalability challenge, researchers began exploring approximate nearest neighbor (ANN) methods in the 1990s. One early approach was Locality Sensitive Hashing (LSH), which replaced exact search with an approximate one. The main idea of LSH and later approaches is to trade a small amount of accuracy for a large gain in speed, letting systems find results that are good enough much faster than an exact search.

  - In many modern applications, such as embedding search for retrieval-augmented generation (RAG) systems, finding the exact nearest neighbors is often unnecessary. What matters is retrieving a few relevant documents or passages that together have enough information to answer a user's query. Approximate methods usually achieve this and are far more efficient.

  - Early work on speeding up search focused on reducing vector dimensionality, for example using Principal Component Analysis (PCA) or Singular Value Decomposition (SVD). More recent embedding designs, like Matryoshka embeddings, get similar efficiency by arranging information hierarchically inside the vector space. Component-wise quantization encodes each vector component with a small number of bits. This causes a small approximation error, but in practice it does not change relative distances between vectors enough to affect nearest neighbor rankings.

Even with data reduction methods, we still must search through all the data, although we read less than for an exact search. In this chapter, we provide an overview of modern nearest neighbor and similarity search methods, using Facebook AI Similarity Search (FAISS) as an example. FAISS brings together the main techniques developed for nearest neighbor (NN) and most similar (MS) search, so it applies across many domains. It uses a composite index structure that combines several complementary methods into one efficient search index. FAISS implementations are also highly optimized for both CPU and GPU architectures.

  - FAISS includes the traditional methods described above and adds more recent improvements. For example, the SQ4 index performs approximate search by quantizing each component to 4 bits, while the Flat index does a brute force scan of all vectors. Applying PCA beforehand can reduce dimensionality. You can combine these methods with HNSW or IVF to greatly reduce the number of items that must be read.

The FAISS framework organizes the search process into four main phases:

  - Vector Transformers: Techniques for normalizing and transforming vectors. The choice of a vector transformer depends on the data, the distance or similarity measure you want, and the indexing method you will use. Some transformations prepare vectors for indexing, for example by normalizing for cosine measures, while others provide a rough initial result by reducing dimensionality.

    - L2-norm: Normalizes vectors by their L2 norm, which is beneficial for both cosine similarity and maintaining consistent distance and similarity values.

    - PCA: Applies principal component analysis to reduce dimensionality, typically to around 64 dimensions. This focuses vectors on the most discriminating directions and preserves the utility of distances/similarities.

    - Optimized Product Quantization (OPQ): Optimizes vectors for subsequent product quantization, reducing quantization errors and improving estimates.

    - Padding: Adjusts the dimensionality of vectors to align with the requirements of a subsequent method.

  - Non-exhaustive search components (coarse quantizers): Methods that use coarse-grained selection to reduce the amount of data considered during search, allowing early elimination of candidates.

    - See the following pages for implementations in the framework, such as HNSW and IVF.

  - Encodings (fine quantizers): Quantization techniques that lower the computational burden in estimating distances or similarities to the query.

    - See the following pages for implementations in the framework, such as LSH and PQ.

  - Refiners: While vector transformers prepare data before searching, refiners process results afterward. In many approximate search cases, refiners are unnecessary, and the outcomes of the approximate methods are used directly to generate user search results. However, there are situations where refiners can be beneficial:

    - Refiners re-rank search results using extra criteria. One method uses the original vector data to compute exact distances or similarities between the query and each item. To improve accuracy, we retrieve the vector representations of the top $n=m∙k$ results and pick the best $k$ matches. A parameter lets us balance higher search cost against better result quality.

    - Some methods do not compute scores directly, but we need those values to combine scores from different methods. In those cases, calculating distances or similarities for the top results gives the search engine and the user the required information. For example, we add extra constraints or predicates on metadata and apply them during retrieval. As the index returns more results, we filter out items that do not meet these criteria.

Non-exhaustive search components (coarse quantizers): The Inverted File (IVF) reduces search scope through clustering of data. In principle, we could use any scheme to cluster data, but we discuss here a k-means based clustering. This method is simple, fast and provides good results.

  - Consider the figure below: we can choose a number of clusters (e.g., $n=4096$), and the k-means algorithm selects $n$ centers, $𝒄_{𝟏},…,𝒄_{𝒏}$ that minimize the sum of distances between data points and their closest center.

  - Each data point is assigned to the nearest center, and the $n$ centers serve as a coarse-grained representation for all the data points assigned to them (indicated by the amber, red, and blue areas in the figure). These $n$ centers effectively cluster the dataset, providing a quantization code for each data point in the collection.

  - The inverted file is essentially a dictionary with the $n$ centers, each of which maps to a list containing the data points assigned to the respective clusters of these centers.

  - With a query $𝒒$, we determine the closest center among the $n$ centers (for example, in the figure, $𝒄_{𝟐}$ is the closest center). While it is probable that the nearest neighbor to $𝒒$ is in the same cluster, it is not guaranteed. Still, we reduce the search to the cluster of the nearest center to obtain an approximate result, even though it may miss points in nearby other clusters. By utilizing an inverted list, we scan through the list linked to that cluster, effectively reducing search time by a factor of $1/n$ (e.g., $1/4096$) at the cost of reduced quality (assuming a good balance of the k-means clustering with about equal data points per cluster).

$𝒄_{𝟏}$

$𝒄_{𝟐}$

$𝒄_{𝟑}$

$𝒒$

  - To enhance search accuracy, we can explore $m<n$ clusters based on the distances between the query and their centers. This involves scanning $m$ lists, increasing search time to $m/n$ while improving result quality.

  - Each list in the inverted file can be further indexed using additional methods to speed up the search within those lists. For example, a product quantizer can be used to estimate approximate distances. However, stacking multiple approximate methods can result in a rapid decline in result quality. Thus, it is crucial to choose hyperparameters thoughtfully to strike the right balance between time savings and quality preservation.

Non-exhaustive search components (coarse quantizers): The Hierarchical Navigable Small World (HNSW) can work stand-alone or in combination with a product quantization method to enhance navigation speed.

  - A Navigable Small World (NSW) is a graph structure with two key properties:

    - Navigability: NSW allows for efficient and effective traversal from one data point to another within the high-dimensional space. This navigability is achieved by creating connections between data points in a way that reflects the underlying geometric structure of the data, making it easier to find nearby neighbors during search.

    - Small-World Property: In high-dimensional spaces, most data points can be reached through relatively short paths (poly-logarithmic). This property is crucial for efficient nearest neighbor searches, enabling quick location of relevant data points without full dataset traversal.

  - In the lower-right figure, the graph consists of data points as nodes, with each node having a fixed number of connections or so-called "friends." Friends are chosen from the nearest neighbors within the dataset. Additionally, shortcuts to other areas are introduced to enhance navigation and prevent isolated sections. While the number of connections per node can differ, NSWs generally work towards maintaining balanced connections.

  - During a search, we start at specific entry points in the graph. For each node, we use a greedy navigation strategy by following connections that lead us closer to the query point. If we arrive at a node where all connected nodes are farther from the query, it indicates a local minimum in the graph, and we can consider this as an approximate answer. The search process for finding the most similar data point (dot-product) operates in a similar manner.

entry point

$𝒒$

  - The HNSW hierarchy consists of multiple layers: the base layer encompasses all data points, forming a full NSW structure. Above the base layer, multiple higher layers are established, each with its own NSW, using well-distributed data points. Each higher layer uses fewer data points and contributes to a search hierarchy: at higher layers, coarse regions of interest are identified. Navigating down to the base layer, we refine nearest neighbors with new connections from the lower layer.

  - HNSW can be combined with other techniques to reduce the cost of distance calculations, such as product quantization or PCA. It is one of the fastest methods for approximate search and is included in many frameworks to provide approximate search. Different implementations use different strategies to build and tune the graphs.

Encodings (fine quantizers): The Locality Sensitive Hashing (LSH) uses $n$ hyperplanes to represent the position of a data point with regard to each hyperplane. LSH encodes the position with a $n$-bit string: if a data point is located on the positive side of hyperplane $j$, it is assigned a value of 1 in the $j$-th position of the bit-string, and 0 otherwise. Consequently, the $n$ hyperplanes encode data points' spatial positions with $2^{n}$ potential combinations.

  - We can choose the hyperplanes randomly or employ a method to select an optimized set of hyperplanes. In the lower-right figure, we defined three hyperplanes that partition the data space into seven areas (one of the eight potential combinations is not possible). Each area is represented by a 3-bit string, where a "1" indicates that the area is above the corresponding hyperplane. For instance, the position of the query $𝒒$ can be described as "100" because it only lies above hyperplane 1.

  - As an alternative, we can use the principal axes of the data space to define $d$ hyperplanes. The bit string assigned to data points is then determined by the signs of their component values (1 for positive values, 0 for negative values).

  - We then use these bit-strings to quantize the data points that share the same representation. However, with even a moderate number of hyperplanes, the number of partitions exceeds the number of data points, resulting in many partitions being empty and others being sparsely populated.

  - Instead, we calculate the Hamming distance between the query's bit-representation and the bit-representations of all data points. Then, we choose the $k$ data points with the smallest Hamming distance as an approximate search result. Since we rely solely on the Hamming distance, this approach is most effective when not too many data points share the same representation.

$𝒒$

+

+

+

1

2

3

  - If many points have identical Hamming distances, an additional refinement steps becomes necessary to re-rank data points. During reranking, we retrieve the actual vector representations, calculate precise similarity/distance values, and arrange data points accordingly. We can combine this method with an inverted files method and apply the LSH on each list separately to further accelerate the search.

  - LSH offers an effective balance between search quality and retrieval costs. By adjusting the number of bits, we can control the retrieval speed and memory usage as a hyperparameter. For data points in a $d$-dimensional space and with the choice of $n$ hyperplanes, we can reduce the data size from $4∙d$ bytes to $n/8$ bytes, resulting in an equivalent speedup for the calculation of Hamming distances.

Encodings (fine quantizers): Product Quantization (PQ) is an extension of the component-wise quantization method. In PQ, vectors are divided into $m$ sub-vectors, each with a dimensionality of $d/m$. These sub-vectors are usually non-overlapping, and often require that the total dimensionality, $d$, is a multiple of $m$. Otherwise, we can employ padding to extend vectors to the next multiple of $m$.

  - In the figure at the bottom, an $8$-dimensional vector is divided into $m=4$ sub-vectors. Each sub-vector is quantized into a 2-bit string. When concatenated, this compresses the 8 floating-point numbers into 8 bits, achieving a 1:32 compression ratio in this example.

  - For each sub-vector, we employ a separate quantization scheme. Instead of scalar quantization, we typically use $k$-means clustering (or other quantization methods). This divides the sub-space into $2^{n}$ clusters, where $n$ represents the number of bits for each sub-vector. The cluster centers approximate the spatial locations of the data points.

  - At query time, we first compute the distances between the query's sub-vectors and all cluster centers. In our example, this requires $m∙2^{n}$ sub-vector distance calculations. Subsequently, we scan through the data points, utilizing the $n$-bit strings for each sub-vector as lookup values for the previously computed distances to the centers. Essentially, for each sub-vector, we employ the cluster center as an approximation for the query-to-data distance in that sub-space. Summing up all distances across sub-vectors provides the overall similarity/distance, and we use  these values to identify the $k$ best answers.

0.1

0.3

0.5

0.2

0.6

0.4

0.9

0.8

0.1

0.3

0.9

0.8

0.6

0.4

0.5

0.2

00

10

01

10

vector

sub-vectors

quantized

  - While scalar quantization optimizes each dimension individually, product quantization optimizes across multiple dimensions. Moreover, it allows for  greater data compression and the assignment of fewer bits than there are dimensions within that sub-vector. Achieving this level of compression is not possible with scalar quantization, unless dimensions are skipped.

  - The speedup of PQ is due to the use of small values for $n$, typically 8, 12, or 16, which determine the size of the codebooks representing the distances to cluster centers. Larger values for $n$ create excessively large codebooks that may not perform well on certain CPU/GPU architectures. In such cases, increasing the value of $m$ can be necessary.

  - PQ can be combined with techniques like inverted files (IVF), where IVF initially narrows the search scope, and each of its lists is encoded with a separate PQ index. PQ then expedites the scanning of these lists to obtain the final answer.

The website https://ann-benchmarks.com  offers a comparison of the performance of approximate search methods on real-world datasets. It also offers a benchmarking framework for testing your own method and submitting results for publication. The figure below displays the results for a 100-dimensional GloVe embeddings dataset with $k=10$.

  - The x-axis represents recall, which, in this context, signifies the proportion of approximate top-10 answers that are also present in the top-10 of an exact search.

  - The y-axis displays the query throughput, indicating the number of queries per second achievable at a specific recall threshold.
