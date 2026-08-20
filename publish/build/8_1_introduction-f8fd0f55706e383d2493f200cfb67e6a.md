# Introduction

Information retrieval is not just about finding documents. It is about responding accurately to the user's underlying intent. Without understanding that intent, there is a real risk of giving irrelevant or incomplete answers. For example, when someone asks, "Who won the Nobel Prize for Physics in 2025?" they expect a clear, direct answer with names and context, not just a list of links to sort through.

Companies such as Google have invested heavily to provide direct answers using tools like the Knowledge Graph, featured snippets, and highlighted answers that show concise, authoritative information. Different types of search queries determine how retrieval systems respond:

  - Navigational queries try to find a specific site or brand, so systems favor exact matches and trusted sources.

  - Transactional queries show commercial intent, so systems combine product data and language understanding to display relevant offers and availability.

  - Informational queries seek clear, reliable answers and often use document search together with answer extraction or generation.

  - News queries emphasize freshness and real-time updates, integrating time-sensitive ranking and summarization to keep users informed.

  - Research or exploratory queries seek broad, authoritative coverage and may involve synthesizing information across multiple documents.

  - Local queries hinge on geographic context, using proximity and location data to deliver relevant results.

  - Comparative queries target structured comparisons and reviews.

  - Personal queries focus on privacy and tailoring results based on user profiles.

  - Entertainment queries highlight popularity, freshness, and multimedia content.

  - Feed based queries (often without explicit query) to explore more content on a social platform.

To serve user intent well, it is important to distinguish different query types because each needs a different strategy. For example, informational queries require clear, authoritative answers, while news queries require recent, up to date coverage. This chapter focuses on informational queries and shows how Retrieval Augmented Generation, or RAG, improves their handling. RAG lets models answer questions that go beyond the static knowledge in LLM training data, such as current facts, product details, company policies, or personal meeting notes.

The Knowledge Graph introduced by Google in 2012 is an early form of retrieval augmentation. Rather than returning raw documents, it supplies authoritative, structured facts that can be combined with retrieved data to produce concise answers. It helped shift search from lists of links to clear, context-aware responses.

The earliest form of RAG in web search is Google's Featured Snippets (2014). These snippets pulled short, relevant answers from web pages and showed them above regular search results, synthesizing information instead of just listing links. Typical steps include:

  - Query Understanding: The system interprets the user’s intent, recognizing that the query calls for a direct answer rather than just a list of links.

  - Document Retrieval: It pulls relevant pages using ranking algorithms based on keyword matching, relevance signals, and other ranking factors.

  - Passage Extraction: Within those documents, the system scans for passages or sentences that best address the query (so-called readers).

  - Answer Selection: The best candidate passage is selected as the snippet.

While not generation in the modern LLM sense, featured snippets represented an important step toward answering user intent with extracted, precise information. This approach was paving the road for more advanced RAG techniques that integrate retrieval with natural language generation today.

Knowledge Graph, Google (2012)

Featured Snippet, Google (2014)

Recently, Google integrated large language models to generate answers by combining retrieved information with generative AI. This produces responses that are accurate and contextually rich, going beyond simple extraction to include summarization and clarification.

  - In certain cases, Google combines AI generation with a reader approach, producing text and highlighting the direct answer to the user's question, as shown in the example on the right.

Retrieval-Augmented Generation (RAG) is useful beyond traditional web search. For questions that require non-public or very specific information, such as "Which insurance is best for me?", responses should be tied to a clear context, for example an individual insurance company's policies, rather than to general pre-learned facts.

  - Fine-tuning a large language model to add this context is possible but costly, slow, and inflexible. RAG is a more efficient option because it retrieves relevant, up-to-date information from specific sources when a question is asked.

  - Hallucinations happen when a language model tries to answer without enough information or context, causing it to give plausible but incorrect or made-up responses. By explicitly grounding answers in retrieved documents or data, RAG significantly reduces these hallucinations and ensures the output is tied to real, verifiable sources.

  - RAG implementations often combine generation and reader approaches and link back to sources.

AI Overview, Google (2024)

Retrieval Augmented Generation

Document Store

Snippets

User Query

Response

Generator

Generation-only is not always successful

In the past two years, Retrieval-Augmented Generation (RAG) has advanced significantly as model capabilities improved and the need to handle larger, more complex information spaces grew.

  - At first, a major limitation was the small context window of large language models. For example, OpenAI's GPT-2 had a maximum context of 1,024 tokens, about 750 to 800 English words. Multi-language support was also basic.

  - Because LLMs can process only a limited amount of text at once, documents are split into manageable chunks. This lets retrieval systems break large sources into pieces (chunks) that fit the model's input size, so they can retrieve and generate from relevant segments instead of overwhelming the model with entire documents.

  - With the rise of transformer models, vector search grew more popular because it captures meaning better and can find relevant documents even when they use different keywords than the query.

  - Using the same chunks for vector search and generation at first seemed practical for fitting documents into early generators' small context windows, but it caused major problems. Splitting documents into smaller pieces made it possible to embed and search manageable segments, yet vector search on these isolated chunks often returned fragments that were semantically incomplete or lacked enough context to be relevant. As a result, the retrieved pieces did not fully answer queries and sometimes even confused the generator.

  - Hierarchical chunking is a promising approach to the trade-off between retrieval and generation. It uses small chunks for vector search to keep strong semantic matches, since vector search works better on concise, focused text. It then combines those small pieces into larger, coherent units for the generation step. For example, recent Gemini models have context windows of up to 1,000,00 tokens, allowing the generation step to process these larger, aggregated chunks and produce more comprehensive, context-rich responses without losing coherence.

Recently the field has moved toward agentic RAG systems that combine retrieval, reasoning, and generation in a more interactive and flexible way. Rather than a simple retrieve then generate pipeline, these models can perform multiple retrieval steps, query external knowledge bases, or interact dynamically with structured data sources. This lets them handle more complex queries that need multi-step reasoning to combine data from different sources.

Moreover, there is a growing shift to include generic structured data alongside unstructured text. By integrating databases, APIs, or knowledge graphs as retrievable sources, RAG systems can give precise answers based on up-to-date, authoritative data. This change makes RAG more versatile and useful in fields where accuracy and context matter. In the past two years, RAG has moved from limited, chunk-based retrieval combined with keyword search to semantically rich, agent driven systems that dynamically integrate structured and unstructured data to produce deeper, more reliable information.


## 8.1.1 Pipeline Overview


The tolerance for latency in search systems has shifted significantly. Users are more willing to accept longer latency if it means getting better answers. The experience of receiving responses as a continuous text stream has also made slight delays feel more natural and less frustrating. In specialized areas like research, search sessions can last minutes, but this is only acceptable when the results are truly more valuable and accurate. Because large language models inherently take time to generate responses, users have grown more accepting of slightly longer retrieval times. This has relaxed the previously strict “sub-10ms recall at all costs” mindset that dominated traditional search, allowing systems to prioritize more thorough, higher-quality retrieval over ultra-fast but potentially less accurate results. This shift enables more complex retrieval strategies without sacrificing user experience.

In the following, we discuss the retrieval pipeline that are typically used for Retrieval Augmented Generation (RAG). Over the past two years, the pipeline has grown from a 2-step approach (retrieve, then generate) to a complex multi-step pipeline. Let us use the following query as a running example:

        - Who won the Nobel Prize for Physics in 2025?

  - Traditional keyword-based search systems have a major limitation: they rely on exact word matches with the query. For example, a relevant document that says "The 2025 Nobel Physics laureate was awarded to..." instead of "won the Nobel Prize" might be missed. The system does not recognize that "won" and "was awarded to" mean the same thing or that "2025" means the year, so it can miss relevant information, return unrelated documents that only match keywords, and downgrade highly-relevant documents that use wording similar to the query.

  - Vector search improves results by turning queries and documents into embeddings that capture overall meaning. This helps the system match different wording, for example recognizing that "recipient of the 2025 Nobel Prize in Physics" refers to the same idea. However, it can miss fine details. A word like "who" which signals a search for a person, becomes part of the general vector and loses that explicit role. The system may then return pages about the Nobel Prize event rather than the specific person who won in 2025.

  - To address these issues, retrieval systems should combine semantic search with deeper language understanding and answer extraction. For example, after a vector search narrows down documents related to the 2025 Nobel Prize, a specialized layer should recognize that the word "who" requires a person's name and extract that fact. This hybrid approach ensures the system does not just find broadly related information but provides the exact answer.

The pipeline on the right shows all the components of a possible retrieval strategy for informational queries. We will break down the role of each component:

  - Query Expansion: As discussed earlier, the query can be transformed by several methods: LLM rewrites, breaking it into multi-hop questions, or semantic expansion with embeddings. These changes help to better align the query with the documents.

  - Sparse Retrieval: BM25 is a sparse retrieval method that relies on keywords and how often they appear. With effective query expansion, it can find many documents relevant to the query.

  - Dense Retrieval: Vector search can find semantically similar documents without query expansion. However, the context must be short to produce useful embeddings. The common approach splits a long document into smaller parts, called chunks, and searches those chunks instead of the full document.

  - Hybrid Retrieval: Combines sparse and dense retrieval in a two-stage framework. Stage 1 performs efficient sparse retrieval, and Stage 2 refines the results with more advanced models.

  - Reranker: Bi-encoders embed the query and document separately. Cross-encoders combine the query and document during inference, allowing richer interaction and letting the model directly judge query-document relevance. For example, it can match "who" to people instead of other meanings. Because this requires expensive inference, rerankers are usually applied to only a small set of candidates.

Query Input

Query Expansion

Sparse Retrieval

Dense Retrieval

Hybrid Retrieval

Reranker

Extractive Reader

Generative Reader

Synthesizer

Output

multi-hop

guardrails

guardrails

guardrails

Extractive Reader: An extractive reader answers questions by finding the exact span of text in a passage that contains the answer. It encodes the query and the passage together, then checks each token in the passage to decide where the answer starts and ends. Because it uses the original text directly, the output is not a paraphrase or interpretation but the exact snippet from the passage.

  - For example, if the passage says "John Clarke, Michel Devoret, and John Martinis won the 2025 Nobel Prize in Physics", an extractive reader finds the start and end tokens around the names and extracts that exact span: "John Clarke, Michel Devoret, and John Martinis".

Generative Reader: A generative reader answers questions by generating responses one token at a time using a large language model. It takes both the query and the retrieved passages as input. This lets the model produce a synthesized, natural sounding answer.

  - For example, after reading several passages about the 2025 Nobel Prize in Physics, a generative reader might say: "The 2025 Nobel Prize in Physics was awarded to John Clarke, Michel Devoret, and John Martinis for their discoveries in quantum tunneling and energy quantization in electric circuits." Unlike extractive models, it does not just copy this sentence from one source; it builds the answer from the available evidence.

Synthesizer: For more thorough answers in research mode, repeat the retrieval steps several times. At each step, a trained reasoning model asks for more information to complete the generative answer. For example:

  - Hop 1: Retrieve documents about the 2025 Physics Nobel winners. This identifies Clarke, Devoret, and Martinis.

  - Hop 2: Use names to retrieve data about their research focus. This finds work on quantum computing applications.

  - Hop 3: Retrieve documents that connect that research focus to specific teams or organizations.

Guardrails are rules that make sure the system provides answers that are accurate, safe, and match the user's intent. They control how the system selects, generates, and delivers information after retrieval so the final answer is trustworthy and appropriate. They act at several stages of the process, from input to output, ensuring the final response is accurate, safe, and aligned with the user's expectations. Guardrails are implemented through a combination of:

  - Prompt instructions and fine-tuned model behavior

  - Rule-based filters and classifiers

  - Evidence grounding and verification mechanisms

  - Safe output formatting and fallback strategies

Step by step execution:

  - Query Input: "Who won the Nobel Prize in Physics in 2025?"

  - Query Expansion: generate variants like "2025 Physics Nobel winners", "physics Nobel laureates 2025"

  - Retrieval (parallel or sequential):

      - BM25: returns ~500 documents with exact keyword matches (Nobel, Prize, Physics, 2025, winner, laureates)

      - Bi-Encoder: returns ~500 documents with semantic similarity to query embeddings

      - Hybrid Merging: combine results from both retrievers and select ~100 top candidates

      - Alternative: BM25 for top-500, then bi-encoder over top-500 to select ~100 top candidates

  - Reranker: score all 100 candidates with a cross-encoder   top 20 reranked documents

  - Reader (one of the following):

      - Extractive Reader: finds a span with "Clarke, Devoret, and Martinis" in top documents

      - Generative Reader: generate answer: "The 2025 Nobel Prize in Physics was awarded jointly to John Clarke, Michel Devoret, and John Martinis for their discoveries in quantum computing."

  - Synthesizer: performs multiple hops (reasoning) and produces final response with top-ranked documents: "The 2025 Nobel Prize in Physics was awarded to John H. Clarke, Michel H. Devoret, and John M. Martinis for their groundbreaking discoveries on quantum tunneling and macroscopic quantum effects in electric circuits. Clarke, from UC Berkeley, Devoret from Yale University, and Martinis, now at Google Quantum AI, developed key technologies for quantum computing, including quantum error correction and superconducting qubits."

Modern RAG systems are modular pipelines that can be tuned for speed, accuracy, or both. Individual components, such as sparse retrieval like BM25, dense retrieval using embeddings, or cross-encoder reranking, can be swapped, skipped, or run in parallel. Simple queries may skip dense retrieval, while complex or domain-specific queries often use cross-encoders for the best results.

These systems trade off speed and quality. Sparse retrieval is fast but captures less meaning. Dense retrieval finds semantic matches but uses more compute. Cross-encoders raise accuracy by reranking results. Using all three gives the best output but at a higher cost. Because of this modular design, RAG systems can power anything from very fast search tools to precise, domain-specific question answering.
