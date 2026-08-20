# Query Transformation

Retrieving the right information depends not only on what users want to know but also on how they ask. Users often pose queries in informal, vague, or incomplete ways, while documents use different words or phrasing. This mismatch is known as the vocabulary gap. Query transformation is an umbrella term for methods that modify, expand, or reformulate a query to close this gap. It includes simple term-level changes and more advanced rephrasing driven by large language models. The main goal is to improve recall by finding more relevant documents while keeping precision by preserving the original intent.

To show how these methods work, consider the query: "Who won the Nobel Price in Physics in 2025?" Although clear in everyday speech, an information retrieval system must treat it as a search for recipients, their names, institutions, scientific contributions, and other related terms that may appear in relevant documents.

Method 1: Spell-checking

  - The first change to a user's query is often correcting spelling mistakes or typing errors. This may seem simple, but it is essential because even a small misspelling can prevent a system from finding relevant documents.

  - In the example query "Who won the Nobel Price in Physics in 2025?" the phrase "Nobel Price" is misspelled: Price should be Prize. If not corrected, the system could search for documents containing the word price, which is not about awards and could return irrelevant results.

  - Why this step matters:

    - Improves matching accuracy: Most retrieval systems rely on exact or near exact term matches. Spelling errors make it less likely that query terms will match words in documents.

    - Prevents recall loss: A misspelled word can reduce matches and cause many relevant documents to be missed.

    - Supports further transformation: Later steps, such as rephrasing, depend on clean and accurate input terms.

  - How spell-checking works: The system compares query terms to a list of correctly spelled words. If a term is not found, it suggests the closest match based on string similarity, e.g., by using edit distance between Price and Prize.

  - After spell-checking, the query becomes:

      - Who won the Nobel Prize in Physics in 2025?

  - Now the system can more confidently proceed to the next transformation tasks, such as identifying relevant entities, expanding terms, or generating alternative phrasings.

Method 2: Knowledge-Based Expansion

  - Once the query is clean and spelled correctly, expand it with structured knowledge sources. This helps when key terms link to several related concepts in curated resources. The aim is not to change the query intent but to cast a wider net and find documents that use different but related terms.

  - Documents do not always use the same words as the query. A newspaper might say "The Nobel was awarded to" instead of "Who won the Nobel Prize". By expanding the query with related terms from trusted knowledge sources, systems can match more of these variations, improving recall and keeping results relevant to the user's intent.

  - Types of knowledge sources used

    - Thesauri and lexical databases (e.g., WordNet) provide synonyms or closely related words. For example:

      - won  	  received 	was awarded	secured	prize  	  award		honor

      - This helps match documents that describe the same event using different wording.

    - Ontologies and knowledge graphs (e.g., ConceptNet, Wikidata, DBpedia) go beyond synonyms by revealing relationships between ideas and entities. For instance:

      - Nobel Prize  		scientific achievement	award ceremony	Royal Swedish Academy of Sciences

      - Physics		quantum mechanics	particle physics	theoretical physics

      - Laureate		winner

      - These structures are human-validated, which helps maintain high precision while broadening the search vocabulary.

  - With knowledge-based expansion, our running example becomes something like:

    - Nobel Prize, Nobel award in Physics, physical sciences, won, received, was awarded, laureate, 	scientific achievement, Royal Swedish Academy of Sciences, quantum mechanics

  - Internally, the system uses knowledge-based connections to recognize documents that express the same ideas in different words or with more technical terms, without showing this complexity to the user. This helps deliver more relevant results by capturing phrasing variations, while relying on carefully curated knowledge to avoid irrelevant matches. However, there is a balance. Adding too many related terms can introduce noise and weaken the original intent, so expansions must be managed carefully to stay focused and to avoid query drift.

Method 3: Embeddings

  - After cleaning the query and adding structured knowledge, the next step uses semantic representations learned by neural networks trained on large text collections. Unlike keyword matching or synonym expansion, these models capture subtle meanings and the relationships present in normal language.

  - Why embeddings and neural methods matter: Traditional expansion methods use fixed word lists or rigid relationships. Language is fluid, and words can mean different things depending on context. Embeddings place words in a multi-dimensional space where closer words have similar meanings. This lets retrieval systems find documents that do not use the same words but cover the same ideas.

  - Word embeddings like Word2Vec and GloVe learn vector representations of words by analyzing the words that appear near them in sentences. In this vector space, quantum and mechanics will be close, as will award and prize.

    - For the query “Who won the Nobel Prize in Physics in 2025?”, embeddings might suggest adding terms such as “laureate”, “quantum”, or “breakthrough”, based on semantic proximity.

Method 4: Large Language Models

  - Large language models such as GPT improve queries by using deep reasoning, paraphrasing, and creativity to rewrite them. Rather than only expanding or embedding a query, they understand the user's intent, infer what information is needed, and offer several ways to phrase the same question. They can also think through a likely answer internally to guide retrieval.

    - Multi-query generation: The LLM produces various paraphrases and alternative queries, such as “Physics Nobel winners 2025", “Who received the 2025 Nobel Prize in Physics?”, “John Clarke Nobel Prize quantum research”

    - Step-back prompting: The LLM first broadens the scope by abstracting the query to a general question, like “What are recent Nobel Prize winners in Physics?” This wider context helps the system reason before homing in on the 2025-specific answer.

    - Chain-of-thought decomposition: The model breaks the complex query into smaller parts: "What is the Nobel Prize in Physics?", "When is it awarded?", and "Who won it in 2025?". Each answer builds on the previous, allowing more precise and structured retrieval.

  - These LLM-driven changes go well beyond matching keywords. They create a richer understanding of user needs, narrative context, and logical structure, enabling more accurate and insightful information retrieval.

Method 5: Feedback-Based Transformation

  - Relevance feedback uses user input to improve the query based on real judgments. When users mark documents as relevant or irrelevant, the system updates the query vector to raise the weight of terms from relevant documents and lower the weight of terms from irrelevant ones. In the Nobel Prize example, if users mark documents that mention John Clarke, Michel Devoret, and John Martinis as relevant, those documents likely include terms such as "macroscopic quantum tunneling", "energy quantization", or "quantum computing". The system adds these frequent terms to the query and reduces terms from irrelevant documents that discuss prizes in other fields like Literature or Peace or awards from earlier years, keeping the search on topic.

  - Another useful type of feedback comes from using users' collective behavior by showing similar queries they have made before. Instead of relying only on explicit judgments about document relevance, this method uses patterns in how other people ask related questions. When users see alternative queries phrased differently but asking the same thing, such as "Physics Nobel winners 2025" or "Who received the Nobel Prize in Physics this year?", they can refine their search or choose a version that better matches the system's retrieval capabilities.

  - Pseudo-relevance feedback: Manual relevance feedback works well but is not always practical because it asks users to judge documents. Pseudo-relevance feedback automates this by assuming the top-ranked results from the initial search are relevant. The system extracts terms from those documents to expand the query.

    - For example, when the initial query "Who won the Nobel Prize in Physics in 2025?" yields a list of top documents, the system analyzes them to find terms that commonly occur with the query concepts and then adds those terms back into the query.

    - The main benefits are increased recall by adapting the query to the actual content returned. It requires no extra effort from the user and speeds up retrieval.

    - Risks include: if the first retrieval results are off-topic or noisy, the system may drift the query toward irrelevant content, a phenomenon known as query drift. This can cause the system to retrieve fewer relevant documents or to introduce unwanted biases.

  - Despite advances in neural and LLM-based methods, feedback-based approaches remain a cornerstone of many retrieval systems. They draw directly on the user's experience and retrieval results, so query modifications reflect practical relevance rather than just linguistic or statistical links.

Balancing Precision, Recall, and Intent

  - Query transformation is powerful, but it must be managed carefully to avoid losing sight of the user's real goal. A major risk is query drift. Adding too many new terms, or terms only loosely related to the original query, can gradually pull the search away from the user's true intent. This can produce many irrelevant documents, reducing precision even as recall increases.

  - Detecting query drift often means checking how query expansion affects retrieval performance. For example, if expansion makes the top results less relevant or shifts them to unrelated topics, such as prizes in other fields or different years, the system flags possible drift. Another method is to measure semantic similarity between the original and expanded queries. If similarity falls below a set threshold, the expansion may be too broad.

  - Preventing drift usually means assigning weights to query terms. The original words keep the highest weight so they anchor the search, while added expansion terms receive lower weights according to their relevance or confidence. For the Nobel Prize query, the weights might look like this:

    - Original terms: “Nobel Prize Physics 2025” with full weight (1.0)

    - Named entities from feedback: “John Clarke”, “Michel Devoret”, “John Martinis” at slightly lower weight (0.8)

    - Related scientific concepts: “quantum tunneling”, “energy quantization” at moderate weight (0.6)

    - Broader contextual terms: “Royal Swedish Academy”, “announcement” at low weight (0.4)

    - This hierarchy keeps the search centered while still exploring useful variations.

Hybrid Systems: Combining Methods for Best Results

  - No single query transformation method works for every situation. Different use cases, query types, and document collections need specific strategies. Modern search engines combine several approaches in layered pipelines to use each method's strengths.

  - A typical hybrid system might begin by using a large language model to clarify or paraphrase the user's query, making it more precise. Next, contextual embeddings such as BERT expand the query to include related meanings that a keyword system might miss. Then a knowledge graph adds domain terms and relationships to enrich the search with human curated insights. Finally, pseudo-relevance feedback reviews the initial search results and fine tunes the query to the corpus language and to emerging topics.

  - Combining these strategies, the system balances user intent, semantic nuance, and real document language, while preventing drift through weighted term importance and relevance checks.
