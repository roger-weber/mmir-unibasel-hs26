---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Opening
chapter: Why Retrieval Matters
section: From Ctrl+F to AI Answers
order: "0.1"
---

# From Ctrl+F to AI Answers

## The Problem Retrieval Solves

Consider three tasks you might perform today:

1. You type "how to fix a leaky faucet" into a search engine. The best tutorial uses the phrase "repair a dripping tap". A keyword-only system would miss it entirely.
2. You upload a 50-page contract to an AI assistant and ask "what are the termination clauses?" The system must find the relevant paragraphs, extract the answer, and present it in natural language.
3. You hum a melody into your phone. Somewhere in a catalogue of 100 million tracks, the system identifies yours from a few seconds of imperfect audio.

Each of these requires a different kind of retrieval. The first bridges a vocabulary gap between how people ask and how documents are written. The second combines retrieval with generation. The third matches across entirely different representations of the same content: a hummed melody against a studio recording.

The gap between what a user means and how information is stored is called the *semantic gap*. Closing it is the central challenge of retrieval, and every technique in this course addresses some part of it.

## The Retrieval Spectrum

Retrieval systems have evolved through four distinct stages. Each stage did not replace the previous one; rather, it added a new capability on top.

```{list-table} The retrieval spectrum: from keyword match to generation.
:header-rows: 1
:name: tbl-retrieval-spectrum
:widths: 20 30 25 25

* - Stage
  - What it does
  - Core technique
  - Covered in
* - Keyword match
  - Find documents containing the exact query terms
  - Boolean retrieval, inverted index
  - Chapters 1, 4
* - Statistical ranking
  - Rank documents by term importance and frequency
  - TF-IDF, BM25
  - Chapters 1, 4
* - Semantic understanding
  - Match by meaning, not surface words
  - Embeddings, transformers
  - Chapters 5, 6
* - Retrieval-augmented generation
  - Retrieve context, then generate an answer
  - RAG pipelines
  - Chapter 7
```

Two forces made this evolution possible. First, data scale: the world will produce over 400 zettabytes of data by 2028, far beyond what any system could index with brute force. Retrieval must be efficient. Second, compute: GPUs and specialized hardware made neural retrieval models practical at production scale, turning what was research in 2018 into standard infrastructure by 2024.

## What You Will Learn

This course is organized into three parts that build on each other:

```{list-table} Course structure: foundations, search systems, and advanced topics.
:header-rows: 1
:name: tbl-course-roadmap
:widths: 20 40 40

* - Part
  - Chapters
  - What you can do after
* - Foundations (Ch 1-3)
  - Classical text retrieval, performance evaluation, advanced text processing
  - Build a working text search engine and measure whether it actually returns good results
* - Search Systems (Ch 4-7)
  - Indexing, semantic search, vector search, RAG pipelines
  - Scale search to millions of documents, search by meaning, and build systems that generate answers from retrieved evidence
* - Advanced Topics (Ch 8-12)
  - Web search, content analysis, visual features, audio features, video features
  - Extend retrieval beyond text to images, audio, and video
```

By the end of this course, you will be able to:

- Build a text retrieval system from scratch using an inverted index
- Evaluate retrieval quality with precision, recall, and ranking metrics
- Process raw text into features that improve search quality
- Design a semantic search system using dense embeddings
- Construct a RAG pipeline that answers questions from a document collection
- Apply retrieval techniques to non-text media: images, audio, and video
- Choose the right retrieval architecture for a given problem

The path from Chapter 1 to Chapter 12 is cumulative: each chapter introduces one retrieval capability and the techniques to implement it. We begin with the simplest useful system (keyword search over a small collection) and end with systems that search across media types and generate natural-language answers.
