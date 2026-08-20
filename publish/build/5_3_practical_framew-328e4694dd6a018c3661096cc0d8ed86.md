# Practical Frameworks

Modern information retrieval applications require reliable frameworks that hide the complexity of indexing and searching while providing the scalability and performance needed in production. In the rest of this chapter we review several practical text retrieval frameworks, each showing a different stage in the evolution of an application, from lightweight experimental tools to distributed, enterprise-grade systems. In this section:

  - We start with Whoosh, a pure Python search engine built for simplicity and ease of use. Whoosh is self-contained and easy to install, and it does not require external services or Java. Its index format and analyzer pipeline are written entirely in Python, making it suitable for learning, testing ranking algorithms, and adding search to small applications. While Whoosh does not provide the distributed scale of larger systems, it stands out for its clarity and transparency, which makes it a popular choice for teaching and rapid prototyping.

  - Next, we turn to PostgreSQL Full Text Search (FTS), a feature-rich part of the database that adds full-text search directly to SQL. PostgreSQL uses the tsvector and tsquery types to tokenize, normalize, and match text inside relational data. It supports stemming, stop words, ranking functions such as ts_rank, and GIN and GiST indexes for fast lookups. This integration lets developers build consistent, transactional systems with search living alongside the rest of the data model. It is an elegant choice to avoid running a separate search system.

  - Next, we consider Haystack, a higher-level framework that provides an abstraction layer for different search backends. Haystack offers a single API for indexing and querying data, no matter the underlying engine or backend service (including cloud-based services). This abstraction lets you start with a lightweight backend like Whoosh for development and then move to a more powerful one in production.

  - Together these frameworks reflect the goals of early-stage search development. They are lightweight, adaptable, and approachable, suited to small and mid-size applications, research prototypes, and teaching.

The next section looks at Lucene, a Java-based core library that enables most of today's advanced search platforms. Lucene supplies the basic tools of modern information retrieval: tokenization, inverted indexing, term weighting, and ranking methods. It is highly configurable and lets developers create custom analyzers, filters, and scoring models.

In the final section, we examine Solr, Elasticsearch, and OpenSearch. These distributed systems build on Lucene and add clustering, horizontal scalability, load balancing, and real time search. Solr uses a schema-driven enterprise model and supports faceting, multilingual text, and integration with existing data platforms. Elasticsearch and OpenSearch use a schema-flexible RESTful API that emphasizes easy integration, analytics, and near real time updates.

Whoosh is a pure Python library for full-text search that makes it easy to learn information retrieval and quickly prototype search applications:

  - Whoosh requires you to specify schema fields before you start indexing, like Lucene but using Python syntax. Fields can be set to TEXT for tokenized full-text search, KEYWORD for exact matches, ID for unique identifiers, or STORED to keep values without indexing.

  - Whoosh's inverted index follows standard information retrieval methods and exposes its internals to Python. For each term, the library builds a posting list that records which documents contain the term, along with term frequency and position data, based on the field settings. Whoosh's real-time updates add, delete, and modify documents through a writer interface that manages index segments and applies changes incrementally.

  - Whoosh supports several scoring methods, including term frequency counting, TF-IDF, BM25F, and cosine similarity. BM25F builds on basic BM25 to work with multiple document fields and different weights, so a match in a title can score higher than the same match in an overview field. Whoosh lets you build queries in code or parse them from strings. The parser handles Boolean operators AND, OR, NOT; quoted phrase searches; wildcards with * and ?; and field-specific searches using field:term. Fuzzy search can match misspelled terms within a specified edit distance.

PostgreSQL full-text search (FTS) capabilities demonstrate how relational databases have evolved to incorporate information retrieval functionality:

  - PostgreSQL uses the tsvector and tsquery data types for full-text search. tsvector stores text as a normalized list of lexemes. tsquery stores search expressions that allow Boolean operators and phrase matching.  The function to_tsvector converts plain text into a tsvector by applying stemming, removing stop words, and optionally assigning weights to parts of the text to affect ranking. The function to_tsquery converts a search string into a tsquery.

  - GIN (Generalized Inverted Index) is the main index type used for PostgreSQL's full-text search. GIN builds an inverted index that maps each lexeme to the documents containing it, speeding up full-text queries. The index uses a B-tree at the top level and compressed posting lists for each lexeme, saving space while keeping query performance high. This process closely follows the inverted index construction algorithms described earlier.

  - The match operator (@@) tests whether a document matches a query. PostgreSQL uses the ts_rank function for ranking and relevance scoring; it implements BM25-style scoring with document length normalization. A key advantage is its integration with relational queries, allowing complex queries that combine full-text search with standard SQL.

Haystack is a modern framework for building retrieval-augmented generation (RAG) pipelines and search systems. Its modular, pipeline-based design connects reusable components into flexible processing chains that handle both traditional keyword search and modern neural retrieval methods.

  - Document Stores serve as the foundation of Haystack's architecture, providing persistent storage for indexed documents and their associated metadata. The framework supports multiple document store implementations, from simple in-memory stores (InMemoryDocumentStore) for development to production-grade solutions like Elasticsearch and vector databases.

  - Retrievers run search algorithms that find documents relevant to user queries. Haystack provides several retriever types. InMemoryBM25Retriever uses keyword search with the BM25 ranking function. It follows the document at a time evaluation strategy described earlier and searches the full list of documents.

  - Haystack is commonly used to build longer pipelines composed of stores, embedders, retrievers, rerankers, generators, and other components to integrate different implementations. Pipeline orchestration links these parts into end-to-end workflows using explicit connection definitions. Those definitions make it easier to debug, test, and modify search workflows.
