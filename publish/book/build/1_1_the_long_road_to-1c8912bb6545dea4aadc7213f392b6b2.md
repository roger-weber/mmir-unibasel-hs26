# The Long Road to Modern Search

The study of information retrieval and knowledge management is closely linked to the broader story of how societies create, store, and access knowledge. From early library catalogues to today’s vast digital ecosystems, each technological advance has reshaped the ways in which people find and use information. Understanding this progression is essential for students of information science, computer science, and related disciplines because it highlights not only the technical breakthroughs but also the social and economic forces that drive innovation.

Before computers became common tools for research and communication, information retrieval was a slow, manual process. Scholars and librarians relied on carefully maintained card catalogues, subject indexes, and classification systems to locate relevant works. These methods were effective for the printed collections of the time, but they could not scale to meet the demands of a rapidly growing body of scientific literature and government documentation. By the mid-twentieth century, the need for more efficient retrieval methods had become urgent. Governments, universities, and industries all required faster ways to search large collections of technical documents, and this demand set the stage for the field’s first major transformation.

The arrival of digital computing in the post-war era provided the tools to address these challenges. Early experiments with computerized catalogues and automated indexing showed that machines could, in principle, manage and search much larger volumes of information than human indexers. Researchers began to formalize concepts such as “recall” and “precision” which remain core metrics today. Yet the transition from theory to practical systems was gradual. Early machines were expensive, text collections were limited, and few documents were available in machine-readable form. Nevertheless, the foundations laid during this period would guide the next decades of development.

The history that follows traces how information retrieval evolved from these early experiments into the sophisticated, large-scale systems of the present. Each decade introduced new models—Boolean retrieval, vector spaces, probabilistic approaches, and eventually neural networks—that improved how systems interpret queries and rank results. At the same time, the volume of data expanded exponentially, requiring continual advances in storage, processing power, and algorithmic efficiency. By exploring these developments chronologically, we can see how the field adapted to shifting technological landscapes, from the first keyword indexes to today’s transformer-based semantic search and retrieval-augmented generation. In the following, we provide the context needed to appreciate both the theoretical underpinnings and the practical achievements described in this course.

The 1960s: Foundations of Knowledge Management

  - The 1960s saw the rise of computerized information retrieval, built on centuries of manual cataloging. Calvin Mooers had coined the term "information retrieval" in 1950, but serious research did not begin until the 1960s. That decade became a "boom time for information retrieval" because of widespread research and development.

  - Important advances included H. P. Luhn's creation of KWIC (Key Word In Context) indexes and Calvin Mooers's development of edge-notched card systems. Western Reserve University's Searching Selector, built by Allen Kent, was one of the era's best-known machines. Most work, however, remained experimental, and there was little computerized retrieval because few texts were machine-readable.

  - During the 1960s, researchers established core evaluation metrics for retrieval systems, defining recall and precision. IBM developed STAIRS, the Storage and Information Retrieval System, one of the first large-scale, general-purpose information retrieval systems for text datasets. At Cornell, Gerard Salton began developing SMART, the System for the Mechanical Analysis and Retrieval of Text, which became foundational to modern search technology.

Characteristics:

  - Users: Researchers, government/industrial R&D groups

  - Use Cases: Experimental text search, bibliographic retrieval, indexing scientific literature, cataloging documents

  - Key Technologies: Boolean retrieval, KWIC (Key Word In Context) indexing, edge-notched cards, classification systems (Dewey Decimal)

  - Retrieval model:  Retriever-only, Retriever-Filter

  - Limitations: Small machine-readable text collections, slow response times, complex Boolean queries required, limited scope, largely experimental, few operational systems

The 1970s: The Rise of Computational Models

  - The 1970s marked a turning point when retrieval systems became practical. Widespread use of computer typesetting and word processing produced the machine-readable text needed for large-scale systems. By the end of the decade, most printed material passed through computer input stages, creating large repositories for retrieval systems.

  - This decade saw the rise of classical retrieval models that would shape the field for years. Gerard Salton's vector space model represented documents and queries as vectors in a high-dimensional space, allowing their similarity to be measured mathematically. During the same period, Stephen E. Robertson, Karen Sparck Jones, and others developed a probabilistic retrieval framework, which laid the groundwork for  probabilistic models. Both approaches were later combined into BM25, which still produces excellent retrieval results.

  - Large-scale commercial systems like the Lockheed Dialog system came into operation in the early 1970s. The MEDLARS (Medical Literature Analysis and Retrieval System) became operational, providing computerized access to biomedical literature. These systems demonstrated the practical viability of automated information retrieval on a commercial scale.

Characteristics:

  - Users: Researchers, Librarians & Government analysts

  - Use Cases: Academic & biomedical literature search, Library catalog/book search, Bibliographic & reference retrieval

  - Key Technologies: Boolean, vector space, probabilistic retrieval

  - Retrieval model:  Retriever-only, Retriever-Filter, Retriever-Ranker

  - Limitations: Limited coverage, Expensive access (subscriptions, terminals, telecom), Slow response times, Complex queries requiring trained intermediaries, Often abstract/metadata-only (not full-text)


The 1980s: Probabilistic Models and Boolean Systems

  - In the 1980s, retrieval models developed in the previous decade were refined and widely adopted. Boolean retrieval systems, derived from century-old indexing practices, became the dominant commercial approach. These systems let users combine search terms with logical operators (AND, OR, NOT), but they remained difficult for new users to operate effectively.

  - The probabilistic retrieval approach continued to develop, notably advancing the Binary Independence Retrieval (BIR) model. Robertson formalized its theoretical basis, the Probability Ranking Principle, showing that ordering documents by their probability of relevance from highest to lowest yields the best retrieval performance.

  - The Okapi information retrieval system was developed at London's City University during this decade; it later gave its name to the well known BM25 ranking function. Latent Semantic Indexing (LSI) appeared as a new method that uses singular value decomposition to reveal hidden relationships between terms and documents.

Characteristics:

  - Users: Researchers, librarians, and commercial search system operators

  - Use Cases: Academic literature retrieval, library book searches, bibliographic/reference lookup

  - Key Technologies: Boolean retrieval, vector space model, probabilistic retrieval (e.g., Binary Independence Model)

  - Retrieval model:  Retriever-only, Retriever-Filter, Retriever-Ranker

  - Limitations: Small-scale datasets, slow query response times, high learning curve for formulating queries, costly system implementation and operation

The 1990s: The Web Revolution and Search Engines

  - The 1990s brought a major change with the arrival of the World Wide Web and the first web search engines. When the Web opened to the public in 1991, available information expanded rapidly, and new methods were needed to find it.

  - Early search engines such as Archie, Gopher, and WebCrawler appeared to help users find their way through the new online world. Yahoo! Search became one of the first widely used web search services; it began as a directory-based system. Launched in 1995, AltaVista introduced full-text searching and indexed multimedia content.

  - The most significant development occurred when Larry Page and Sergey Brin created PageRank at Stanford University in 1996. The algorithm changed web search by ranking pages according to the authority of links pointing to them instead of relying only on keyword matching.

  - During this period, machine learning began to be used in information retrieval. Researchers explored neural networks, symbolic learning, and genetic algorithms for various IR tasks. Relevance feedback grew more sophisticated, enabling systems to learn from user interactions.

Characteristics:

  - Users: General public, e-commerce consumers, professionals and business users

  - Use Cases: web search, e-commerce search, business/market intelligence, academia

  - Key Technologies: Vector Space Retrieval, Probabilistic Retrieval, Web Search Engines, Query Expansion, separation of retrieval & sort

  - Retrieval model: Retriever-Ranker, (Retriever-Filter)

  - Limitations: data & index explosion due to exponential growth, large retrieval and indexing costs, good recall values but poor perceived precision (i.e., document not relevant for the user), quality of data


The 2000s: Machine Learning and Ranking Algorithms

  - In the 2000s, information retrieval systems began using advanced machine learning. Learning to Rank (LTR) became an important approach: supervised models were trained to sort search results by relevance. This shift moved ranking away from hand-designed features toward data-driven optimization.

  - Based on a probabilistic framework from earlier decades, BM25 became a standard ranking function. It combines term frequency and inverse document frequency with document-length normalization, delivering strong performance across diverse collections.

  - Deep learning reemerged after research largely stopped at the end of the 1990s. More powerful computers, especially GPUs, larger datasets, and better training methods brought neural networks back into use. Researchers developed deep belief networks (Hinton, 2006) and applied convolutional neural networks to vision tasks.

  - Search engines continued to evolve their algorithms, and Google's PageRank became the basis of its dominance in web search. User behavior signals and personalization began to influence the ranking.

Characteristics:

  - Users: Web search users, e-commerce consumers, data scientists & academics, social media users

  - Use Cases: Web search personalization, e-commerce search & recommendation, emergent semantic search research

  - Key Technologies: LTR, BM25, personalization via behavior logs

  - Retrieval model: Retriever-Ranker (enhanced with ML)

  - Limitations: Data/index explosion, high retrieval/indexing cost, data quality, model transparency, update latency

Geoffrey Hinton

Nobel Prize in Physics 2024


The 2010s: Deep Learning and Neural Information Retrieval

  - The 2010s saw a major shift as deep learning techniques were applied to information retrieval. Neural ranking models began using shallow and deep neural networks to rank search results, moving away from traditional handcrafted features toward learned representations.

  - Classical learning-to-rank models relied on manual feature engineering. Neural models learned representations that bridged the vocabulary gap between queries and documents; they required large training sets but captured semantic meaning far more effectively.

  - Google introduced BERT in 2018 as a major advance in natural language understanding. BERT's ability to capture context and relationships between words allowed for more accurate interpretation of complex queries. Google integrated BERT into its search algorithm in 2019.

  - Transformer architectures fundamentally changed how machines handle sequential data, enabling them to capture long-range dependencies in text. Capabilities in 2010s were still constrained lacking datasets, scalability, and performance.

Characteristics:

  - Users: Web search users, e-commerce consumers, mobile/app users, data scientists & academics

  - Use Cases: Web search personalization, e-commerce search & recommendation, conversational search, semantic search research

  - Key Technologies: Deep learning (CNNs, RNNs, Transformers), BERT, early neural ranking models, embeddings, large-scale behavior log personalization

  - Retrieval model: Neural retriever-ranker pipelines

  - Limitations: High computational cost, large data requirements, latency in updates, model interpretability, scalability of training & indexing

Word2Vec

https://opensearch.org/blog/ltr-with-opensearch-and-metarank


The 2020s: Semantic Search and Vector Databases

  - The 2020s marked the start of semantic search and vector-based retrieval. Transformer models improved information retrieval by enabling systems to understand context and meaning rather than rely solely on keyword matching.

  - Vector databases are now essential infrastructure for modern search systems. They allow fast similarity search over high-dimensional embeddings and can match semantically similar content even when keywords do not overlap, for example "car" with "automobile“.

  - Retrieval-Augmented Generation (RAG) combines large language models with external knowledge retrieval. It lets models access up-to-date information that is not in their training data, reducing hallucinations and improving factual accuracy.

  - Agentic information retrieval systems can understand complex information needs, create multiple search strategies, and combine results from different sources. They act as intelligent intermediaries that reason about information requirements, carry out multi-step retrieval, and deliver thorough responses tailored to each context.

Characteristics:

  - Users: Web search users, e-commerce consumers, mobile/app users, data scientists, knowledge workers, AI developers, enterprise teams

  - Use Cases: Semantic search, retrieval-augmented generation (RAG), conversational AI, personalized recommendations, knowledge bases, intelligent virtual assistants, multi-step query resolution

  - Key Technologies: Large language models, embeddings, vector search

  - Retrieval model: Neural retriever-ranker, retriever-generator

  - Limitations: Requires high-quality embeddings and up-to-date data, may still hallucinate if retrieval fails, computationally intensive, complexity in multi-source integration

https://opensearch.org/platform/vector-engine

query

response

agent

LLM

knowledge base 1

knowledge base 2

web search

code execution

tool A

tool B

reasoning

