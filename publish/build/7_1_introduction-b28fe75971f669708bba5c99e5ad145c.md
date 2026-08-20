# Introduction

The story of vector search does not begin with algorithms or system architectures but with a basic problem that has challenged information retrieval systems since the beginning: the semantic gap. This gap, the mismatch between low-level computational features and high-level human understanding, has driven decades of innovation in how we search for, retrieve, and make sense of digital information.

In the 1990s, as digital image collections grew rapidly, researchers ran into a key limit of traditional retrieval systems. A user searching for "vacation photos" expected the system to grasp the idea: beaches, sunsets, families together. Keyword-based metadata could only match exact text labels. Images without proper tags stayed hidden, and manual tagging was too slow and subjective to scale.

Content-based Image Retrieval (CBIR) was the first systematic effort to close this gap. IBM's QBIC system in 1995 introduced feature extraction, turning images into vectors that describe color, texture, and shape, and allowed comparisons based on visual traits rather than metadata alone. However, these low-level features captured only surface properties. For example, a blue ocean and a blue sky can have similar color histograms while meaning very different things. The semantic gap remained: machines could detect pixel patterns but could not grasp meaning.

This challenge went well beyond images. Text retrieval had similar limits. Traditional inverted indexes and TF-IDF systems worked for exact keyword matches but struggled with synonyms, paraphrases, and conceptual queries. For example, searching for “heart disease” did not find documents about “cardiology treatment”. The systems could not capture semantic relationships, which reduced their usefulness across all modalities: text, images, audio, and video.

Semantic Gap

source: https://theailearner.com/2019/02/09/understanding-image-histograms

Content-based Image Retrieval

Key-word vs. Semantic Matching


The breakthrough came with neural embedding models had matured enough to put meaning into numbers. In 2013, Word2vec showed that semantic relations can emerge from vector math. In 2018, BERT introduced contextualized embeddings that capture how a word's meaning shifts with its surrounding text. In 2021, CLIP created multimodal embeddings that place text and images in a shared semantic space, enabling zero-shot cross-modal search. These models learned representations that reflect human ideas of similarity and meaning.

Vector search became the infrastructure that lets embeddings work at scale. It turns queries and documents into dense vectors in high-dimensional space, so systems can judge meaning by how close vectors are to each other. A query vector for "vacation photos" will sit near image vectors for beaches and families, even when the images have poor tags. Nearest neighbor search became a key part of the retrieval pipeline.

As data volumes grew exponentially, a new barrier appeared: scale. Brute force search for the most similar vectors among millions or billions of candidates required prohibitive computing resources. Traditional spatial indexes like R-trees handle a few dimensions but fail under the curse of dimensionality. As vector dimensions increase, distances between points converge and the idea of a nearest neighbor loses meaning.

Exact nearest neighbor (NN) search looks ideal in theory, but in practice it provides little benefit for the applications that use it. Embedding-based systems care about finding relevant results, not exact geometric closeness in the embedding space. Approximate nearest neighbor (ANN) search optimizes for the metric that actually matters: how often the retrieved results meet the downstream task. A document or product that ranks third in the embedding space is often still relevant to a user's query. Empirically, ANN algorithms achieve over 95 percent of the recall of exact search, while running hundreds of times faster and scaling to billions of items.

Search in Embedding Space

source: https://projector.tensorflow.org

source: https://stacksweep.substack.com/p/smarter-search-at-airbnb-how-embedding

Trade-off with Approximate Nearest Neighbour Search

Today, vector search supports a wide range of applications. Spotify recommends music using embeddings of audio features. Financial firms detect fraud by finding unusual transaction patterns in high-dimensional behavioral spaces. Healthcare systems match patients with similar symptom profiles to guide treatment decisions. Autonomous vehicles process sensor streams as vectors to navigate in real time.

The infrastructure evolved in parallel. Specialized libraries such as FAISS (2017) offered GPU-accelerated similarity search with sophisticated compression techniques. Purpose-built vector databases like Milvus and Pinecone were optimized for high throughput and low latency retrieval. Traditional systems added vector capabilities, for example pgvector for PostgreSQL and Elasticsearch’s k-NN search, allowing organizations to adopt vector search without replacing their entire infrastructure.

Even though vector search is mature, it still faces technical hurdles. At very large scale, the curse of dimensionality makes distance measures less reliable, so work focuses on quantization and dimensionality reduction. Another challenge is interpretability. Dense embeddings capture meaning as a whole but are hard for humans to inspect, while sparse features as with BM25 are easier to understand but miss subtle meaning. A growing consensus favors hybrid retrieval, which blends dense and sparse signals using ranking fusion methods such as Reciprocal Rank Fusion (RRF).

The semantic gap that once separated human thought from machine retrieval is steadily closing. A new information paradigm is emerging: every kind of data, from legal contracts and jazz riffs to medical scans, can be represented, compared, and retrieved by meaning. Vector search is no longer just a technology; it is the foundation of semantic search, where similarity is measured by understanding rather than by words or symbols.

source: https://www.linkedin.com/posts/grusiya_systemdesign-spotify

Curse of Dimensionality

Spotify Recommendation Approach

