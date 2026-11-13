# Exam Guide: Multimedia Retrieval (HS26)

The written exam of the course *Multimedia Retrieval* (HS26, University of Basel) checks whether you understand how retrieval systems for text, images, audio and video work, and whether you can apply that understanding to new situations.

**Contents**

- [Introduction](#introduction)
- [Exam content](#exam-content)
- [Content outline](#content-outline)
- [Group 1: Text and Web Retrieval](#group-1-text-and-web-retrieval)
- [Group 2: Evaluation](#group-2-evaluation)
- [Group 3: Indexing and Vector Search](#group-3-indexing-and-vector-search)
- [Group 4: Semantic Search and RAG](#group-4-semantic-search-and-rag)
- [Group 5: Multimedia Content and Features](#group-5-multimedia-content-and-features)
- [Out of scope](#out-of-scope)
- [How to prepare](#how-to-prepare)

## Introduction

The exam validates that you can:

- explain the challenges of a retrieval task and the trade-offs between the methods that address them;
- choose a method, a measure or an architecture for a given scenario and justify the choice;
- combine several necessary concepts, sometimes from different chapters, into a solution;
- recognize when a result, a measure or a design does not support the conclusion drawn from it.

The exam asks for understanding, not for memorized details. You need to know what a method does, why it works, what it gains and what it costs, but not its formulas or parameters.

## Exam content

### Format

- Paper exam, in English, **2 hours**.
- Date, room and permitted aids: *to be announced*.
- Number of questions and points: *to be announced*.
- Every question belongs to one of the five groups of the [content outline](#content-outline). The order of the questions may vary, and easy and hard questions are mixed.

### Response types

| Type | Format | Points |
|---|---|---|
| Single choice | 4 options, exactly 1 correct | 1 |
| Name or term | a name or a term, written into a box | 1 or 2 |
| Multiple choice | 5 options, 2 or 3 correct; the number of correct options is not stated | 2 or 3 |
| Kprim | 4 statements, each true or false | 2 or 3 |
| Short explanation | at most 20 words, written into a box; only a few per exam | 3 |

- **Points follow time.** A question's points reflect the time it needs if you know the answer, about 1 to 1.5 minutes per point. Questions at higher levels and of harder types give more points. You may answer the questions in any order.

- **Single choice:** mark exactly one box.
- **Multiple choice:** mark every option that applies. Each wrong mark costs twice what a correct mark earns, and a question never scores below 0. Example: in a 2-point question with 2 correct options, marking both plus 1 wrong option scores 1 + 1 − 2 = 0 of 2 points. If you do not know, marking fewer options is safer than marking more.
- **Kprim:** decide true or false for each statement. 4 correct decisions earn full points, 3 earn half points, fewer earn nothing.
- **Name or term:** write the name or term the question asks for, inside the box.
- **Short explanation:** write one sentence of at most 20 words inside the box: the reason or mechanism the question asks for.
- Text outside the box is not graded. Written answers score 0, half or full points; a wrong answer costs nothing.
- Wrong options are plausible on purpose: they describe what someone with incomplete understanding might believe.

### Levels of understanding

Every question targets one of four levels. All groups contain questions at several levels.

| Level | What you do | Approximate share of points |
|---|---|---|
| Recall | state one taught name, definition, property, convention or abstract purpose | 25 % |
| Apply | apply one taught concept to a familiar problem with new details | 40 % |
| Synthesize | combine at least two necessary taught concepts into a new solution | 25 % |
| Extend | derive a new consequence from taught principles under an unfamiliar retrieval setting or constraint | 10 % |

- Solid recall and application are enough to pass.
- Extend questions are few and demanding. They distinguish excellent answers; do not worry if you cannot answer them all.
- No calculator is needed. Some questions ask for a simple calculation from counts, such as the precision of a short result list.

### Results

- Grades on the Swiss scale from 1 to 6; 4.0 is a pass.
- Points needed per grade: *to be announced*.

## Content outline

The exam covers five content groups. The share is the approximate share of questions per group.

| Group | Chapters | Share |
|---|---|---|
| [1. Text and Web Retrieval](#group-1-text-and-web-retrieval) | 1, 3, 8 | 37.5 % |
| [2. Evaluation](#group-2-evaluation) | 2 | 12.5 % |
| [3. Indexing and Vector Search](#group-3-indexing-and-vector-search) | 4, 6 (and parts of 5) | 12.5 % |
| [4. Semantic Search and RAG](#group-4-semantic-search-and-rag) | 5, 7 | 12.5 % |
| [5. Multimedia Content and Features](#group-5-multimedia-content-and-features) | 9, 10, 11, 12 | 25 % |

The tasks below describe what each group covers. They are not a complete list of exam content, and the examples in brackets are not exhaustive. Each group has a cheat sheet that summarizes its material.

## Group 1: Text and Web Retrieval

Chapters: [1 Classical Text Retrieval](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-1/), [3 Advanced Text Processing](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-3/), [8 Web Search](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-8/). Cheat sheet: [cheat-sheet-text.md](cheat-sheet-text.md)

### Task 1.1: Design the text pipeline from documents to index terms

Knowledge of:

- retrieval units and granularity; retrieval with and without ranking, and facets
- offline indexing and online query processing with the same analysis
- tokenization, normalization (case, accents), stop words, stemming, lemmatization, character n-grams
- Zipf's law, term discrimination and inverse document frequency

Skills in:

- choosing the analysis steps for a collection and stating what each gains in recall and costs in precision
- diagnosing a missed or wrong match caused by the pipeline
- explaining why exact token matching remains valuable next to embeddings

### Task 1.2: Rank documents with classical retrieval models

Knowledge of:

- set-of-words, bag-of-words and tf·idf representations; sparse vectors; positional information
- standard and extended Boolean retrieval
- the vector space model (inner product, cosine) and its shortcomings
- the binary independence model and relevance feedback
- BM25 (saturation, length normalization) and BM25F for fields

Skills in:

- predicting which document a model ranks higher, and why
- explaining which problem of an earlier model a later model solves
- deciding which failures a ranking model can and cannot fix

### Task 1.3: Understand and rewrite queries

Knowledge of:

- phrases at index time and proximity at query time
- vocabulary mismatch (synonyms, broader and narrower terms, compounds, misspellings, name variants); query expansion versus query rewriting
- ambiguity (homonyms) and how context resolves it
- query understanding: language detection, part-of-speech tags, named entities; intent routing to backends

Skills in:

- choosing a means against a mismatch or an ambiguity, with its effect on recall and precision
- following a query through the processing steps to a structured backend query

### Task 1.4: Rank results on the web

Knowledge of:

- how web search differs from classical retrieval; query types; head and tail queries
- signals beyond the page text: anchor text, location, language, quality and trust
- link analysis: PageRank, personalized PageRank, HITS
- the ranking stack, learning to rank, labels from raters and clicks, position bias, online tests
- pointwise, pairwise and listwise training goals

Skills in:

- choosing and applying signals for a query, and making them robust against manipulation
- choosing between static and query-specific link analysis
- placing a signal in the right stage of the ranking stack, and judging click evidence

## Group 2: Evaluation

Chapter: [2 Performance Evaluation](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-2/). Cheat sheet: [cheat-sheet-eval.md](cheat-sheet-eval.md)

### Task 2.1: Measure the quality of retrieval results

Knowledge of:

- benchmark design: collection, information needs, relevance judgments, measurement procedure; pooling
- precision, recall, fallout and the F-measure
- ranked measures: P@k, reciprocal rank and MRR, average precision and MAP, the precision-recall curve, R-precision
- graded relevance: cumulative gain, DCG, nDCG
- macro and micro averaging

Skills in:

- choosing the measure that matches an information need
- computing simple measures from counts, and recognizing what the given data cannot determine

### Task 2.2: Evaluate classifiers and choose thresholds

Knowledge of:

- the confusion matrix and its rates; asymmetric error costs; multiclass evaluation
- the effect of prevalence on accuracy and precision
- score distributions, thresholds, ROC curves, AUC and operating points

Skills in:

- deciding which error an application must limit, and which rate to report
- judging a reported accuracy or precision against the class balance
- choosing an operating point from the costs of an application

### Task 2.3: Read evaluation results critically

Knowledge of:

- incomplete and noisy ground truth
- overfitting to a fixed test set, leakage, and the spread behind a mean score
- efficiency (latency, throughput, cost) as a second axis next to effectiveness

Skills in:

- judging whether a reported gain supports the conclusion drawn from it
- setting up a fair comparison of two systems

## Group 3: Indexing and Vector Search

Chapters: [4 Index for Text Retrieval](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-4/), [6 Vector Search](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-6/), and from [5 Semantic Search](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-5/) the quantization of embeddings and rank fusion. Cheat sheet: [cheat-sheet-index.md](cheat-sheet-index.md)

### Task 3.1: Build and query an inverted index

Knowledge of:

- inverted files: vocabulary, postings, document table; Boolean operators as merges of sorted postings
- immutable segments, deletions and updates
- ranked retrieval over postings: term-at-a-time and document-at-a-time
- filtering on metadata and faceted search
- parallel segments, shards, replicas and regions

Skills in:

- explaining what a query costs, and why some queries are expensive
- choosing a filtering strategy and a scaling lever for a requirement

### Task 3.2: Compare vectors and find nearest neighbors

Knowledge of:

- distance and similarity measures; normalization of vectors and dimensions
- exact search structures in low dimensions and why they fail in high dimensions (distance concentration)
- approximate search: fewer dimensions, fewer bits per vector (quantization), fewer vectors read (inverted file, graphs); rescoring
- the trade-off between recall and speed, and the role of a reranker

Skills in:

- choosing a similarity measure and the normalization a setup needs
- choosing an approximate method and its settings for a constraint (memory, size, updates)
- choosing an operating point from end-to-end quality

### Task 3.3: Run vector search in practice

Knowledge of:

- filtering in vector search: before, after and during the search
- hybrid search: vector search, keyword search and filters in one request; reciprocal rank fusion; learned combination

Skills in:

- diagnosing short or poor results under a filter, and fixing them
- combining rankings from different retrievers

## Group 4: Semantic Search and RAG

Chapters: [5 Semantic Search](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-5/), [7 Retrieval-Augmented Generation](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-7/). Cheat sheet: [cheat-sheet-semantic.md](cheat-sheet-semantic.md)

### Task 4.1: Compare texts by meaning

Knowledge of:

- latent semantic indexing and static word embeddings, and their limits
- why pooling word vectors fails for longer texts
- bi-encoders, cross-encoders and late interaction; two-stage retrieval
- pipeline architectures from BM25 with a reranker to hybrid and multi-stage retrieval; model selection criteria

Skills in:

- choosing an architecture for a stage of a pipeline
- choosing a pipeline and a model size for a collection and its queries

### Task 4.2: Build retrieval-augmented generation

Knowledge of:

- why language models need retrieval, and when RAG is the wrong tool
- the stages of a RAG pipeline and what each repairs
- chunking strategies and their trade-offs
- result generation: reader, generator, synthesizer; citations
- agentic RAG: the agent loop, tools and guardrails

Skills in:

- choosing between RAG and its alternatives for a requirement
- choosing pipeline stages, a chunking strategy and an answer form for a scenario

### Task 4.3: Evaluate a RAG system

Knowledge of:

- context recall and context precision; correctness, faithfulness, relevance and completeness of answers
- latency and cost per answer; LLM judges; paired comparisons of configurations

Skills in:

- computing simple context and answer measures from counts
- locating the failing stage from retrieval and answer scores

## Group 5: Multimedia Content and Features

Chapters: [9 Content Analysis & Metadata](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-9/), [10 Image Features](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-10/), [11 Audio Features](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-11/), [12 Video Features](https://roger-weber.github.io/mmir-unibasel-hs26/book/index-12/). Cheat sheet: [cheat-sheet-media.md](cheat-sheet-media.md)

### Task 5.1: Bridge the semantic gap

Knowledge of:

- the semantic gap and the abstraction ladder of features
- metadata: sources, standards, costs and trustworthiness
- the two transformer paths: describe then search text, or a shared embedding space; hallucinated metadata

Skills in:

- deciding which part of the gap a query hits, and which path closes it
- choosing a source of metadata or a transformer path for a collection

### Task 5.2: Describe and search images

Knowledge of:

- color spaces, color histograms and color moments; edges, texture and shape; histograms of oriented gradients; local keypoint features
- convolutional networks, embeddings from their layers, transfer learning, face embeddings
- vision transformers, CLIP and DINOv2; where hand-crafted features still win

Skills in:

- choosing a feature or a model for a query type, and stating what it ignores

### Task 5.3: Analyze and search audio

Knowledge of:

- sampling, spectrum, spectrogram, frames and segments; the time-frequency trade-off
- perceptual features (energy, zero crossings, spectral features, mel scale, MFCCs) and musical features (chroma, pitch, tempo)
- audio classification: decision trees, learned classifiers, zero-shot labels with CLAP
- speech transcription: the classical pipeline, end-to-end models; searching transcripts
- audio fingerprinting

Skills in:

- choosing a feature, a classifier or a specialized method for a task
- diagnosing why spoken names or labels are not found, and choosing a remedy

### Task 5.4: Analyze and search video

Knowledge of:

- shot boundary detection: change measures, choosing the threshold, gradual transitions, keyframes
- motion detection with a static camera; optical flow and its assumptions
- video search: units of video, image and motion models, aligning speech, text and audio on the time line, moment retrieval

Skills in:

- diagnosing detection errors and choosing a remedy
- designing an index of moments for a query type

## Out of scope

The following is **not** tested:

- **Chapter 0** (overview) and the **appendices A to G**. The appendices are background for readers who want it.
- **Sections not assessed:** 5.3 *Transformer Embeddings*, and 6.3 *The Curse of Dimensionality* beyond the recap at the start of section 6.4. The book marks both as background.
- **Optional readings** marked as such in the book.
- **Formulas and parameters:** derivations, formulas to reproduce, parameter names and values (for example the parameters of BM25 or the damping factor of PageRank). Simple computations from counts are in scope.
- **Algorithm internals:** for example the SIFT algorithm, graph layers of HNSW, product-quantization codebooks, HMM training and decoding, the matrices of Lucas-Kanade, the steps of MFCC. The principle of a method and how it differs from its neighbors are in scope.
- **Topics left out of the exam:** text extraction, crawling, result-page composition, feeds, the Naive Bayes algorithm, collocation measures (PMI, likelihood ratio), postings compression, full-text search in databases.
- **Products and frameworks:** APIs and configuration of Lucene, OpenSearch, FAISS, pgvector or RAG frameworks; prompt and tool-call formats. Products appear only as examples of a mechanism.
- **Model and benchmark facts:** model names and sizes, leaderboard scores, benchmark numbers.
- **Demos and exercises:** their code, libraries and outputs. Use them to understand the methods.

## How to prepare

- **Cheat sheets:** one per group, about 2 to 3 hours of reading in total. They summarize the material along the topics of each group, add formulas and algorithms where these help understanding, and point to the sections and demos worth a deeper look.
  - [Text and Web Retrieval](cheat-sheet-text.md)
  - [Evaluation](cheat-sheet-eval.md)
  - [Indexing and Vector Search](cheat-sheet-index.md)
  - [Semantic Search and RAG](cheat-sheet-semantic.md)
  - [Multimedia Content and Features](cheat-sheet-media.md)
- **Exam preparation app:** practice exams in the format of the written exam. *Link follows.*
- **Book:** the [course book](https://roger-weber.github.io/mmir-unibasel-hs26/book/) is the reference for every topic.
- **Quiz app:** the [chapter quizzes](https://roger-weber.github.io/mmir-unibasel-hs26/quiz/) check the basics of each chapter.
