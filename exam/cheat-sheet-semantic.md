# Cheat sheet: Semantic Search and RAG

Semantic search (chapter 5): dense representations from LSI and word embeddings to bi-encoders,
cross-encoders and ColBERT, and the pipelines built from them. Retrieval-augmented generation
(chapter 7): when to retrieve, the RAG pipeline, chunking, result generation, agentic RAG and
evaluation.

## Chapter 5: Semantic Search

### LSI and static word embeddings

- Term matching assumes term independence, so "car repair" misses "automobile maintenance". Dense
  representations close this vocabulary mismatch.
- LSI reduces the term-document matrix (truncated SVD) to a few latent topics. Terms that occur in
  similar documents move close together, so a document can rank high without any query term.
  Costs: dense vectors (no inverted index, every document is compared), topics specific to the
  corpus (no transfer to another domain), a fixed vocabulary.
- Word embeddings follow the distributional hypothesis (a word is known by the company it keeps):
  they learn from local context windows on large raw corpora, need no labels and transfer across
  collections. fastText builds a word from character n-grams, so unknown and misspelled words
  still get vectors. Related words cluster; simple analogies work by vector arithmetic.
- The limit of both: one vector per word, whatever the context ("bank" as money or river).

### Bi-encoders, cross-encoders and ColBERT

- Pooling word vectors into one text vector fails: in a long text each token counts little, stop
  words pull the average toward generic directions, and word order is lost ("dog bites man").
- **Bi-encoder** (Sentence-BERT): encodes each text on its own into one vector, trained on pairs so
  that similar texts lie close (contrastive losses, in-batch and hard negatives). Documents are
  encoded once at indexing; a query costs one encoding plus one dot product per document.
- One vector per passage compresses, more so for long inputs, so a longer context window does not
  guarantee a better embedding: chunking a long document often beats one vector for all of it.
- **Cross-encoder**: reads query and passage together, so every token attends to every other. Best
  quality (it sees that "who" asks for a person), but one full model pass per pair at query time,
  so it reranks only a shortlist.
- **Two-stage pipeline**: a bi-encoder or BM25 retrieves the top 100 to 1000, a cross-encoder
  reranks them: near cross-encoder quality at near bi-encoder speed.
- **Late interaction (ColBERT)**: encodes query and document separately but keeps one contextual
  vector per token and matches each query token with its best document token. Documents stay
  precomputed and nothing is squeezed into one vector, so quality comes close to the cross-encoder;
  the cost is many vectors per document to store and compare.

| | Bi-encoder | ColBERT | Cross-encoder |
|---|---|---|---|
| Stored per document | one vector | one vector per token | nothing |
| Query-time cost | one dot product per document | query tokens times document tokens | full model pass per pair |
| Quality | good | very good | excellent |
| Typical stage | first-stage retrieval | retrieval, moderate collections | reranking the top-k |

**Math** (*for understanding; formulas are not asked in the exam*). Bi-encoder score with
unit-length vectors (dot product equals cosine), and ColBERT's MaxSim sum over query tokens:

$$
\text{score}(q, d) = \mathbf{q}^\top \mathbf{d}
\qquad\qquad
\text{score}_{\text{ColBERT}}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{q}_i^\top \mathbf{d}_j
$$

**Go deeper:** [Semantic similarity with transformers](https://roger-weber.github.io/mmir-unibasel-hs26/book/semantic-similarity-with-transformers/), [sentence embeddings demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch05-04-sentence-embeddings.ipynb)

### Designing a semantic search pipeline

- Dense retrieval raises recall but costs three things classical search did not: search without an
  inverted index, lower precision from a wider semantic net, and information lost when a long text
  becomes one vector.
- Every design decision trades compute cost, latency and retrieval quality. BM25 is orders of
  magnitude faster than dense search; cross-encoder reranking is orders of magnitude slower than
  retrieval.

| Pipeline | Adds | Limit |
|---|---|---|
| A: BM25 only | fast baseline | queries and documents use different words |
| B: BM25 + reranker | better order of BM25 candidates | misses documents sharing no query term |
| C: dense (+ reranker) | semantic matches, paraphrases | exact codes, names, product IDs |
| D: hybrid, fused, reranked | exact and semantic matches | robust for most collections; two retrievers |
| E: multi-stage | cheap first stage for huge collections | needs rescoring and an ANN index |

- A reranker only reorders what the first stage found; it cannot repair its recall.
- Fusion in D (reciprocal rank fusion) and the cheap first stage of E (truncated Matryoshka
  embeddings, scalar or binary quantization): see the [index sheet](cheat-sheet-index.md#the-three-levers-of-approximate-search).
- Starting points: below 10,000 documents dense retrieval plus cross-encoder (brute force is fast
  enough); up to about a million hybrid plus cross-encoder; beyond that multi-stage. Product search
  needs hybrid (product IDs need BM25); patent search favours recall over latency.
- Model choice weighs dimension, model size, context window, languages and instruction support
  against encoding time and storage. Leaderboard scores change quickly; a small model plus BM25
  can nearly match a far more expensive large one.

**Go deeper:** [Building semantic search](https://roger-weber.github.io/mmir-unibasel-hs26/book/building-semantic-search/), [pipeline benchmark demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch05-05-building-semantic-search.ipynb)

## Chapter 7: Retrieval-Augmented Generation

### Why and when to retrieve

- Parametric knowledge has three limits: time (weights frozen at training), scope (private and
  specialized documents never seen) and trust (the model writes fluently whether or not it knows,
  so it hallucinates).
- RAG leaves the model fixed and changes what it reads. Fine-tuning on new facts is costly, slow
  and soon stale; RAG is cheaper and fresher, pulls the answer toward evidence and can cite. It
  reduces invention but does not guarantee truth: a noisy or outdated corpus gives a confidently
  wrong answer.
- RAG is the wrong tool when the knowledge is stable and general, when the whole corpus fits into
  the context window, when the goal is a different style or format (fine-tuning changes how the
  model writes), or when the question maps onto structured data (a database query).
- Ongoing costs: an index to keep fresh, latency and tokens for every answer, and a quality
  ceiling set by the corpus. A RAG answer must be clearly better than a ranked list, not merely
  different.
- An agent decides per query: memory for a stable fact, a database tool for a count, retrieval for
  a policy question.

### The RAG pipeline

- Two-step RAG inherits each retriever's weakness: BM25 misses "won" against "was awarded", dense
  retrieval blurs details such as "who" (see [bi-encoders](#bi-encoders-cross-encoders-and-colbert)).
  The advanced pipeline chains stages, each repairing the one before:
  1. **Query expansion**: spelling, synonyms, sub-questions (classical techniques: see
     [text sheet](cheat-sheet-text.md#vocabulary-mismatch)); with a language model HyDE (search with a written
     answer) and multi-query (several rewrites, merged).
  2. **Sparse and dense retrieval**: exact terms and meaning.
  3. **Hybrid retrieval** fuses both, after bringing them to a common unit: dense hits on small
     chunks are resolved to their parent chunk and duplicates collapsed.
  4. **Reranking**: a cross-encoder cuts the candidates to a handful.
  5. **Result generation**, with **guardrails** on what enters and leaves and a **multi-hop**
     loop that sends a follow-up query back to the start.
- New for RAG are two-level chunking and the generation stage, since the consumer is now a model
  rather than a person.
- Every stage is optional and trades speed for quality: a simple lookup skips dense retrieval and
  reranking, a hard domain question takes the full chain. Users accept seconds for a better
  answer, which makes a multi-stage pipeline affordable.
- A framework pipeline is a fixed graph encoding one intent; a new intent needs new code.

### Chunking

- Large chunks give the generator context to reason over; small chunks give sharp embeddings with
  one clear meaning.

| Method | Gains | Costs |
|---|---|---|
| Fixed size, overlap | fast, simple | cuts across topics |
| Sentence boundaries | no broken sentences | blind to topic shifts |
| Structure | follows the author's units | very uneven sizes; merge small paragraphs |
| Semantic | groups sentences by embedding similarity; no markup needed | embeds every sentence |
| Hierarchical | small chunks retrieve, parent section feeds the generator | two levels to store |
| Query-aware | context sized after seeing the query | intent read at query time |

- Hierarchical chunking suits RAG best; query-aware chunking builds on it (one chunk for a fact,
  the whole section for a summary).
- No strategy wins everywhere. Vary only the chunking, keep the pipeline fixed, and measure answer
  recall at $k$ (share of questions for which a top-$k$ chunk contains the answer; an answer split
  across a boundary is in no single chunk) and context precision. A size histogram cannot show
  whether boundaries line up with answers.
- Set chunk sizes in the model's tokens, not in characters.
- BM25 tolerates large chunks (surrounding text does not change the scores of matching terms); a
  bi-encoder needs short ones (the whole chunk becomes one vector). A hybrid system chunks for the
  bi-encoder (chapter 5).

**Go deeper:** [Chunking strategies](https://roger-weber.github.io/mmir-unibasel-hs26/book/chunking-strategies/), [chunking demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch07-01-chunking-strategies.ipynb)

### Result generation: reader, generator, synthesizer

- The three forms trade faithfulness for flexibility.
- **Reader**: reads question and passage together (like a cross-encoder), marks the answer span and
  returns it verbatim. Grounded by construction, but it cannot combine or rephrase.
- **Generator**: writes a new answer from the passages; fluent and able to merge facts, but free to
  hallucinate. The prompt grounds it: answer only from the passages, say so when the answer is
  missing. A fluent answer still needs its claims checked against the passages.
- **Synthesizer**: a generator plus iteration over several retrieval rounds, for questions no
  single passage answers. Most powerful, hardest to keep honest; each hop adds latency.
- The form is chosen per query by context size and quality, question complexity, user intent and
  risk (a reader where an invented detail is dangerous).
- **Citations** are kept through the pipeline as passage metadata, so users can verify, trust
  grows, and developers can trace errors.

### Agentic RAG and guardrails

- A fixed pipeline encodes one intent; a multi-part question (the winners, then each winner's
  university) needs a chain nobody coded.
- The loop: **reason** about what is missing, **act** by calling a tool, **observe** the result.
  It stops when the evidence suffices or an iteration budget runs out.
- Steps and tools follow the query: multi-hop flows are assembled ad hoc, a failing source
  triggers a fallback such as web search, recurring flows can be given as skills.
- Tools are declared by name, description and parameters; the model reads the descriptions to
  choose.
- Intent detection, once a separate classifier (see [text sheet](cheat-sheet-text.md#intent-routing)), becomes the
  loop's first thought.
- Guardrails sit at both edges and run at every hop. Input: reject unsafe or out-of-scope requests,
  and check each tool call in plain code; a blocked call returns its reason as an observation.
  Output: the answer still serves the intent and every claim is supported, otherwise fall back
  to uncertainty.

### Evaluating RAG

- Four axes: retrieved context, answer quality, latency and cost.
- Retrieval sets the ceiling. **Context recall**: was everything needed retrieved? **Context
  precision**: how much of the context is relevant? Irrelevant passages invite wrong combinations.
- Latency as p95 (95% of requests complete within it), cost per answer, and for an agent the
  trajectory (how many calls, the right tools).
- Answer quality: **correctness** (against a reference), **faithfulness** (every claim supported by
  the context), **relevance**, **completeness**.
- Faithfulness measures agreement with the context, not the world: a true claim from memory lowers
  it, and outdated documents make a faithful answer wrong.
- Reference-based evaluation needs written answers; reference-free evaluation scales to
  production. An LLM judge is checked against human judgments; deterministic checks decide
  wherever a rule can.

| Retrieval | Answers | Diagnosis |
|---|---|---|
| low | low | fix retrieval |
| high | low | fix generation |
| low | high | suspicious: answering from memory |
| high | high | working; save latency and cost |

- Comparing two configurations is a paired experiment on the same questions, one component
  changed; only the disagreements count (see [eval sheet](cheat-sheet-eval.md#what-a-reported-score-supports)). Fix acceptable
  latency and cost beforehand.

**Math** (*for understanding; formulas are not asked in the exam, but these ratios may be computed
from small counts*):

$$
\text{context recall} = \frac{\text{required pieces of evidence retrieved}}{\text{required pieces of evidence}}
\qquad
\text{context precision} = \frac{\text{relevant retrieved passages}}{\text{retrieved passages}}
$$

$$
\text{faithfulness} = \frac{\text{claims in the answer supported by the context}}{\text{claims in the answer}}
$$

Book example: four pieces of evidence needed, five chunks retrieved, two relevant: context recall
$2/4$, context precision $2/5$.

**Go deeper:** [Evaluating RAG](https://roger-weber.github.io/mmir-unibasel-hs26/book/evaluation/), [RAG evaluation demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch07-05-rag-evaluation.ipynb)
