# Multimedia Retrieval (HS26) — University of Basel

The **Multimedia Retrieval** course explores information retrieval systems spanning text, images, audio, and video content — from classical models to modern AI-driven approaches.

### Prerequisites

- Programming experience in Python
- Basic linear algebra (vectors, matrices, dot products)
- Basic probability and statistics

### Learning Goals

By the end of this course, you will be able to:

- Build a text retrieval system using inverted indexes and ranking models
- Evaluate retrieval quality with precision, recall, and ranking metrics
- Design semantic search systems using dense embeddings
- Construct a RAG pipeline that answers questions from a document collection
- Apply retrieval techniques to images, audio, and video
- Choose the right retrieval architecture for a given problem



## Schedule

Fridays, 15:00–18:00

Each session focuses on the listed chapter but may include review of the previous chapter or a brief preview of the next.

| Date | Title | Location | Notes |
|------:|-------|----------|-------|
| Sep 18 | [Overview](https://roger-weber.github.io/mmir-unibasel-hs26/book/) ([pdf](chapters/ch00-overview.pdf), [slides](slides/ch00-overview.pdf)) | University | Course intro |
| Sep 25 | [Classical Text Retrieval](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-1/) ([pdf](chapters/ch01-classical-text-retrieval.pdf)) | University | |
| Oct 02 | Performance Evaluation | University | |
| Oct 09 | Advanced Text Processing | University | |
| Oct 16 | Index for Text Retrieval | University | |
| Oct 23 | Semantic Search | University | |
| Oct 30 | Vector Search | Online (Zoom) | |
| Nov 06 | Retrieval-Augmented Generation | Online (Zoom) | |
| Nov 13 | Web Search | Online (Zoom) | Exam preparation |
| Nov 20 | Content Analysis | University | |
| ==Nov 27== | ==*Dies Academicus*== |  | ==No lecture== |
| Dec 04 | Visual Features | University | Evaluation |
| Dec 11 | Audio Features | University | |
| Dec 18 | Video & Structural Features | University | Feedback for course |



## Resources

The full course material is available as an interactive online book:
| Resource | Link |
|----------|------|
| Book | https://roger-weber.github.io/mmir-unibasel-hs26/book |
| Quiz App | https://roger-weber.github.io/mmir-unibasel-hs26/quiz/ |
| ADAM (students) | https://adam.unibas.ch/goto_adam_crs_2206931.html |
| Public course page | https://dmi.unibas.ch/de/studium/computer-science-informatik/lehrangebot-hs26/lecture-multimedia-retrieval/ |



## How the Content Is Organized

The book is structured in three parts that build on each other. Each chapter focuses on one retrieval capability and the techniques to implement it. The path is cumulative: later chapters assume you have worked through the earlier ones. Every chapter starts with a concrete scenario that motivates the problem, then develops the solution through several layers:

1. **Concepts and examples.** We introduce each technique through a running example with real data, so you can see what the method does before we formalize how it works.
2. **Formal foundations.** Key formulas and models are stated precisely, with plain-English intuition alongside the math. You will know both what to compute and why.
3. **Practical implementation.** Code examples show how techniques translate into working systems. Where relevant, we reference production tools (Lucene, FAISS, LangChain) so you can connect theory to practice.
4. **Hands-on notebooks.** Interactive demos let you run the techniques yourself, experiment with parameters, and observe how changes affect retrieval quality.
5. **Quiz questions.** Each chapter includes multiple-choice questions in the [Quiz App](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/) to test your understanding and prepare for the exam.


### Part I: Foundations

The first three chapters establish the fundamentals that every retrieval system relies on.

**Chapter 1: Classical Text Retrieval** introduces the core problem: given a query and a collection of documents, find the relevant ones. We build from Boolean retrieval (exact keyword matching) through TF-IDF weighting to BM25, the statistical ranking model that still powers production search engines today.

**Chapter 2: Performance Evaluation** asks the question every retrieval engineer must answer: does this system actually work? We cover precision, recall, ranked evaluation metrics (MAP, NDCG), and the experimental methodology for benchmarking retrieval systems.

**Chapter 3: Advanced Text Processing** looks at what happens before retrieval: how raw text is transformed into features that improve search quality. Tokenization, stemming, compound handling, query understanding, and intent classification.


### Part II: Search Systems

The middle chapters move from individual techniques to complete systems.

**Chapter 4: Index for Text Retrieval** covers the data structures that make retrieval fast: inverted files, posting lists, compression, and how systems scale from thousands to billions of documents.

**Chapter 5: Semantic Search** introduces the shift from keyword matching to meaning matching. We trace the path from latent semantic indexing through word embeddings to transformer-based dense retrieval.

**Chapter 6: Vector Search** addresses the infrastructure challenge: once documents are represented as vectors, how do we find nearest neighbors efficiently among millions of embeddings? Approximate nearest neighbor algorithms, quantization, and vector databases.

**Chapter 7: Retrieval-Augmented Generation** combines retrieval with large language models. We cover chunking strategies, query transformation, retrieval pipelines, and how to build systems that generate answers grounded in retrieved evidence.

### Part III: Advanced Topics

The final chapters extend retrieval beyond text.

**Chapter 8: Web Search** adds link analysis (PageRank, HITS) and web-specific ranking signals to the retrieval stack.

**Chapter 9: Content Analysis** covers structural and metadata-based features for multimedia documents.

**Chapter 10: Visual Features** applies retrieval to images: color histograms, texture descriptors, shape features, and modern deep visual features.

**Chapter 11: Audio Features** extends retrieval to audio: perceptual features, musical features, and fingerprinting for music recognition.

**Chapter 12: Video Structural Features** addresses the temporal dimension: shot detection, motion features, and how to search within video.



