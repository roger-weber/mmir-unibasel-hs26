# Semantic Search for Documents

Modern information retrieval systems increasingly use semantic search to find documents by shared meaning rather than by shared words. This uses embeddings, which convert text into dense, continuous vectors so that similar meanings are close together. However, semantic methods can be computationally expensive, and purely lexical methods like BM25 often miss subtle relationships between words. The most effective pipelines combine the speed of classical search, the contextual insight of embeddings, and the precision of re-rankers into a single system.

  - The process starts with document preprocessing and chunking. Since embedding models work best with limited input lengths, long documents are split into overlapping segments, typically about 200 to 300 tokens for BERT-based models such as SBERT, and 500 to 1,000 tokens for larger decoder models like Qwen3. Each chunk is then embedded as a high-dimensional vector by a sentence or document encoder, creating a searchable semantic index.

  - When a user submits a query, the system first uses BM25 for lexical retrieval. This traditional ranking function quickly finds a small set of possibly relevant documents by matching words and counting how often they appear. BM25 does not understand meaning, but it is fast and keeps processing focused on the most promising candidates.

  - Next comes semantic retrieval. The query is embedded in the same vector space as the document chunks, and similarity, usually measured by cosine distance, is computed between the query embedding and the stored chunk embeddings. This returns the top-k chunks that are most semantically similar, finding passages that mean the same thing as the query even when they use different words.

  - To refine the results, a re-ranker, often a cross-encoder model, is applied to the top candidates. Unlike embedding models that encode query and document separately, a cross-encoder processes them together for improved semantic analysis. The re-ranker gives more accurate relevance scores and improves the final ranking of results.

  - Finally, the system can return the most relevant chunks for direct answer extraction, or combine chunk scores to rank whole documents. By combining these complementary techniques, the pipeline achieves both scalability and semantic depth. BM25 narrows the candidates, embeddings capture meaning, and re-rankers add precision. Together, they form the backbone of modern semantic search.

For small document collections, systems often skip BM25 lexical retrieval and start with semantic retrieval over embeddings to increase recall. A re-ranker is usually needed to raise precision and to sort the retrieved documents or chunks by how well they answer the query. Another option is to use embeddings instead of an expensive re-ranker to quickly re-rank BM25 results.

Retriever-Ranker Pipeline Options

  - The retriever-ranker has two main parts: the retriever finds a set of candidate documents in the index that match the query (focused on recall). The ranker assigns a relevance score to each candidate and returns the final ordered results (focused on precision).

  - Option 1: We can use a BM25 retriever to get an initial set of candidate documents or chunks. To broaden the search, we can expand the query with additional keywords from dictionaries or from embeddings. Then use embeddings or a cross-encoder to compute semantic similarity scores between the query and each candidate. The result is an updated top-k list of documents for the user.

  - Option 2: Alternatively, we can replace BM25 with semantic retrieval using embeddings. Compute embeddings for documents and queries and store them in a high-dimensional vector index. At query time, select chunks whose embeddings are close to the query embedding, for example by cosine similarity. Optionally, re-rank the top candidates with a cross-encoder ranker for better precision.

We will discuss more semantic search pipelines in Chapter 8. The next pages provides a summary overview over classical and semantic retrieval methods.

Retriever: BM25

query

doc 1

doc 2

doc 3

…

Ranker:embeddings and/orcross-encoder

Retriever: embeddings

query

doc 1

doc 2

doc 3

…

(optional)Ranker:cross-encoder

Matryoshka / MRL Embeddings

  - Recent research introduced Matryoshka Representation Learning (MRL), also called Matryoshka embeddings, to make semantic retrieval more flexible and efficient. Traditional embedding models produce a single vector of a fixed size for each text piece, often 384, 768, or 1,024 dimensions. These vectors are powerful but costly in computation and storage, especially for large vector databases that must store and compare billions of embeddings. MRL builds nested representations that keep their semantic meaning even when cut down to lower dimensions, like Russian Matryoshka dolls where each smaller layer fits inside a larger one.

  - The key idea is to train the model so partial embeddings remain useful. Instead of optimizing only the full vector, MRL adds extra loss terms at several truncation levels, for example the first 128, 256, and 512 dimensions. This forces the model to spread semantic information across the whole vector in a structured way. As a result, the leading dimensions capture the most important semantic features, while the later dimensions refine the representation for finer distinctions. This is similar to Latent Semantic Analysis. Low-dimensional projections capture the main concepts or topics, while extra dimensions capture finer variations and context. In both methods, the representation space is arranged hierarchically: the first components describe broad semantic structure and later components add detail. This hierarchy makes MRL embeddings behave like compressed versions of the same meaning space, preserving coherence when they are made smaller.

  - This property lets you balance accuracy and efficiency. For high-recall searches over large collections, smaller sub-vectors can run fast approximate queries (retriever step 1), cutting memory use and computation. After a small candidate set is found, full-dimension embeddings can re-rank results more precisely (retriever step 2), matching large-model performance at a fraction of the cost. MRL embeddings therefore enable multi-stage retrieval in a single model and remove the need for separate encoders of different sizes.

  - In the next chapter, we study vector search in more detail and examine ways to speed up searches through large sets of vectors.

Retriever Step 1: small embeddings

query

doc 1

doc 2

doc 3

…

(optional)Ranker:cross-encoder

Retriever Step 2: large embeddings

reranker, align with rag chapter
