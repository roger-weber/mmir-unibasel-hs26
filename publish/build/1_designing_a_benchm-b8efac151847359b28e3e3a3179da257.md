---
author: Roger Weber
edition: HS26
status: updated
part: Foundations
chapter: Performance Evaluation
section: Designing a Retrieval Benchmark
order: "2.1"
---

(performance-evaluation-benchmark-design)=
# Designing a Retrieval Benchmark

A university library holds 50 books. A student submits one information need: which books give a beginning master's student solid computer science foundations? Fifteen of the 50 books are useful for that need. Two retrieval systems answer the same request, and their first five results look like this.

| Rank | System A result | Useful? | System B result | Useful? |
|---|---|---|---|---|
| 1 | Computer Organization and Design | yes | Database Systems: The Complete Book | yes |
| 2 | Operating System Concepts | yes | Pattern Recognition and Machine Learning | yes |
| 3 | The Handmaid's Tale | no | Introduction to Algorithms | yes |
| 4 | Narrative of the Life of Frederick Douglass | no | On the Origin of Species | no |
| 5 | Towards a New Architecture | no | Introduction to the Theory of Computation | yes |

System A does not stop at rank 5. It returns 25 books in total, half the library, and finds 12 of the 15 useful ones. System B returns only 8 books, of which 6 are useful. So System A surfaces more of what exists, and System B wastes less of the reader's attention.

Which system is better? The question cannot be answered yet, and that is the point of this section. Nothing is missing about the two result lists: we know every title and every judgment. What is missing is a statement of who is searching, what a good outcome means for that person, and a procedure that produces the same verdict when someone else repeats the comparison.

## What Counts as a Better Result

Relevance is not a property of a book. It is a judgment about whether a book serves a particular person's need at a particular moment. "Introduction to Algorithms" is useful to the master's student and useless to someone looking for a novel to read on a train. The same document therefore changes status when the need changes, which means every measurement in this chapter rests on a human decision made before any system was run.

That subjectivity is unavoidable, but it is also containable. Once the need is stated and the judgments are recorded, the counting becomes mechanical: any two people who apply the same procedure to the same lists arrive at the same numbers. Evaluation does not remove the subjective step. It isolates it, writes it down, and then reasons objectively on top of it.

What a good outcome means still varies enormously across needs. Consider three positions on a spectrum.

At one end are needs where only the first few results matter. Someone checking a single fact, looking up which year BM25 was published, or navigating to a known page reads two or three results and stops. Anything below rank 5 might as well not exist. Here System B wins: its early results are dense with useful material, and the nine relevant books it never returned cost the user nothing.

At the other end are needs where missing anything is the real failure. A patent lawyer searching for prior art, or a legal team running discovery before a trial, cannot afford to overlook one document. A long result list with many irrelevant entries is an acceptable price, because a human will read through all of it and the cost of a miss is a lost case. Here System A wins, and a system that returned only eight books would be unusable no matter how precise those eight were.

Between the ends are needs that require a good and broad-enough set, but not everything. A student writing the related-work chapter of a master's thesis wants a representative overview of a field: enough sources to describe the state of the art, drawn widely enough not to miss a major line of work, but small enough to actually read. Neither extreme serves this student. System A's noise wastes reading time, and System B's eight books do not support a literature overview.

The same tension appears in every domain. Web search optimizes the top of the list because users rarely look further. Biomedical literature search balances coverage against reading effort over a larger result set. The lesson is not that one behaviour is correct, but that "better" is meaningless until the intended use is named.

## Why Anecdotes Are Not Evidence

Product advertising is full of comparative claims with no shared basis. A vendor states that a search engine is forty percent more accurate, without naming the collection it was tested on, the queries that were used, who judged the results, or what "accurate" counts. Such a claim cannot be checked, and it is usually easy to construct: pick the queries where your system happens to win, and the number follows.

Retrieval systems invite exactly this failure, because single-example demonstrations are trivially available. For almost any pair of systems, someone can find one query where the first beats the second and another query where the reverse holds. The two result lists at the start of this section are a demonstration, not evidence. They describe one need in a 50-book collection, so they support no general statement about either system.

A scientific comparison replaces the anecdote with a fixed experimental setup. The collection is frozen, the information needs are written down in advance, the relevance judgments are made independently of the systems being compared, and the measurement procedure is stated before any results are seen. Others can then rerun the same experiment and obtain the same numbers. This is what a benchmark provides: not a guarantee that the winner is better for every purpose, but a claim that is reproducible and open to challenge.

Comparable setups also make progress cumulative. When many research groups report results on the same benchmark, a new method can be placed against a decade of prior work instead of against whatever the authors chose to reimplement.

```{admonition} From Cranfield to BEIR: how the field learned to compare systems (optional reading)
:class: note dropdown

Shared evaluation is older than most of the methods it evaluates, and the paradigm has been revised roughly once per decade.

The Cranfield experiments, run by Cyril Cleverdon at the College of Aeronautics in Cranfield from the late 1950s into the 1960s, established the idea that retrieval quality can be measured offline. Cleverdon assembled a fixed set of documents, a fixed set of queries, and relevance judgments made by subject experts, then compared indexing methods against that fixed material. The abstraction was radical: rather than observing real users at work, the experiment freezes their needs into reusable test data. Every benchmark since has inherited both the power and the weakness of that move.

The Text REtrieval Conference, launched by the US National Institute of Standards and Technology in 1992, scaled the paradigm to collections far too large for exhaustive judging. TREC introduced pooling: participants submit their runs, the organizers merge the top-ranked documents from all submissions into a pool, and only the pool is judged. It also introduced tracks, so that ad-hoc search, question answering, legal discovery, and web search could each be evaluated with material suited to that task rather than by one universal test.

Comparable efforts followed for other language communities. CLEF began in 2000 in Europe, growing out of TREC's cross-language work, and NTCIR started in Japan in the late 1990s with a focus on Japanese and other Asian languages. Both extended the paradigm to multilingual and cross-lingual retrieval, where a query in one language must find documents in another.

MS MARCO, released by Microsoft in 2016, changed the economics again. Built from real Bing queries, it was large enough to train neural ranking models rather than merely to test them, which is why it became the standard proving ground for learned retrieval. The price was sparse judgments: with hundreds of thousands of queries, only a small number of documents per query could be labeled.

BEIR, introduced in 2021, responded to a new failure mode. Models tuned on MS MARCO performed impressively on MS MARCO and often disappointed elsewhere. BEIR bundles eighteen heterogeneous collections and evaluates models without task-specific training, measuring whether a method generalizes rather than whether it has learned one benchmark well.
```

## The Four Components of a Benchmark

A benchmark is built from four parts: a document collection, a set of information needs, relevance judgments, and a measurement procedure. Each part answers one question, and a weakness in any of them undermines everything measured on top.

The **document collection** defines the universe the system may search. It can hold news articles, scientific papers, legal filings, product descriptions, or, as in our running example, library records. Two properties matter most. The collection must be stable, because results are only comparable over time if the searched material does not change underneath them. It must also be representative, reflecting the document lengths, content types, and topic mix that the target users really encounter. Well-known examples include the TREC collections, which range from newspaper archives to biomedical abstracts, MS MARCO for web-style passage retrieval, and domain corpora such as PubMed Central and arXiv for scientific literature.

The **information needs** state what the users actually want. It is worth separating the need from the query string: the need is "which books give a beginning master's student solid computer science foundations", while the query is whatever short text the user types into the search box. Assessors judge against the need, not the keywords, which is why the need must be written out clearly enough that two people would interpret it the same way. Needs are best derived from real user behaviour such as query logs. When no log exists, large language models can draft candidate needs from samples of the collection, which expands coverage cheaply, though the drafts still require review to stay realistic.

The **relevance judgments** record, for each need, which documents satisfy it. They are the ground truth, so their quality bounds the quality of every conclusion. Human assessors work from written guidelines, receive training on example cases, and are checked for agreement, because vague criteria produce inconsistent labels that no metric can repair. Large language models can pre-screen documents and propose judgments for clear-cut cases, leaving human effort for the borderline ones. Judgments must then be frozen: revising them later silently invalidates every result measured earlier.

The **measurement procedure** turns a result list into a number. It fixes which measure is computed, at which cutoff, and how values are combined across needs. The next three sections develop these measures, starting with results treated as an unordered set, then as a ranking, then as a ranking with graded relevance. What matters here is that the procedure is chosen before the experiment, not after the results are known.

The following table collects the practical rules for each component.

| Component | What it provides | Design rule | Common failure |
|---|---|---|---|
| Document collection | The universe that may be searched | Freeze it for the benchmark's lifetime, make it representative of the target domain, preprocess consistently, and give every document a stable identifier | Built around one feature, so it flatters the methods that exploit that feature |
| Information needs | What the target users want | Derive needs from real user behaviour, state each one unambiguously, and provide 25 to 50 for a first experiment or 100 and more for a serious comparison | Wording so vague that assessors disagree and the ground truth becomes noisy |
| Relevance judgments | The ground truth for scoring results | Write assessor guidelines, train assessors, check their agreement, and freeze the labels once published | Judgments revised mid-project, which invalidates all earlier measurements |
| Measurement procedure | The rule that turns results into numbers | State the measure, the cutoff, and the averaging method before running any experiment | Measure chosen after seeing the results, which turns evaluation into advocacy |

Building all four parts is expensive, which is why established benchmarks are reused heavily and why a small, carefully judged collection is often more valuable than a large, carelessly labeled one.

## Judging at Scale

One practical obstacle remains. Judging every document for every need is impossible at realistic scale: fifty books can be assessed exhaustively, fifty million cannot. Pooling is the standard response. The organizers collect the documents that participating systems actually retrieved, judge that pool, and treat everything unjudged as not relevant. [Figure %s](#fig-pooling-dense) shows the situation this assumes. Relevant documents that no system retrieved stay unjudged, so scores come out slightly pessimistic, but since no participant found them, the comparison between participants still holds.

```{figure} images/figure_2_1.png
:name: fig-pooling-dense
:width: 50%

Pooled assessment when participating systems retrieve most relevant documents.
```

This assumption is sound while the pool covers most of the relevant material, and it weakens as the judged fraction shrinks. Pooling is one of several places where a benchmark can mislead rather than inform, and [](#performance-evaluation-limits) returns to all of them once we have the measures needed to describe what gets distorted.

```{note}
Sections 5 and 6 evaluate text classifiers rather than retrieval results, and the ground truth there is a test set rather than a benchmark of the kind described above. Three differences matter. The unit of evaluation is a single text item and its label, not a need-document pair. The labels are normally complete, so pooling does not apply, but class balance and label quality become the central concerns. And a classification dataset is split into training, validation, and test parts, with any decision threshold tuned on the validation part and never on the test part. Section 5 develops these points where they are needed.
```

With a benchmark in place, the remaining question is mechanical: how do we turn a result list into a number that reflects how well the need was served? The next section answers it for results treated as an unordered set, using the same library collection and the same two systems.
