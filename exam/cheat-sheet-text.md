# Cheat sheet: Text and Web Retrieval

Classical text retrieval from the document to the ranked list: the text pipeline that turns documents and queries into terms, the
retrieval models from Boolean retrieval to BM25, query processing against vocabulary mismatch and ambiguity (chapters 1 and 3),
and how web search ranks with signals beyond the page text, link analysis and learning to rank (chapter 8).

## The text pipeline

### Retrieval system architecture

- Every retrieval system answers three connected questions: which unit is returned, what work is done before a query arrives, and whether matches are filtered or ranked.
- **The retrieval unit** is what gets indexed, matched and returned: a document ID, metadata attributes (author, date, language) for filtering and facets, not necessarily matched as full text, and the content attributes indexed for search and ranking.
- **Granularity is a trade-off.** Finer units (sections, pages) give more specific results but more entries to manage; coarser units (a whole book) keep the index small but leave the user to find the passage. Splitting is decided once at indexing time: by document boundary (file, email, web page), by structure (chapter, section, legal article), by line or log entry, or at fixed page or paragraph counts when there is no usable structure. Chunking for RAG is covered in the [semantic sheet](cheat-sheet-semantic.md#chunking).
- **Three ways to deliver results:**

| Pattern | What it returns | Order |
|---|---|---|
| Retriever only | the set of documents that satisfy the query | none; matches can be returned as they are found |
| Retriever with facets | the same set, filtered and sorted by metadata | by metadata; relevance is unchanged |
| Retriever plus ranker | candidates scored by a ranking model | by estimated degree of relevance |

- In the retriever–ranker pipeline, the ranker scores only the candidates of the fast retriever, not the whole collection.
- **Offline and online.** Scanning every file at query time works for a laptop, not for millions of articles. The offline phase ingests and splits documents, identifies metadata, extracts and normalizes features and stores them in an index. The online phase runs the query through the same tokenization, optionally broadens it (spelling correction, synonyms), searches the index and ranks.
- **Symmetry.** Queries and documents must live in the same feature space, so the query passes the same analyzer (tokenizer, filters, stemmer) as the documents. A changed analyzer means rebuilding the index, which makes mistakes at ingestion time expensive.

### Tokens and exact matching

- Tokenization turns the character stream into tokens; a **term** is the normalized form of a token. The distinct terms form the vocabulary, one dimension per term.
- Lexical matching works by identity: each distinct token gets its own ID, and two tokens are either identical or unrelated. "cat" and "cats", or "Zürich" and "Zurich", are independent terms, and "cat" does not match "feline".
- Recall therefore depends on tokenization. A naive split at whitespace and punctuation breaks abbreviations ("U.K." into "U" and "K"), numbers ("$1.4 billion" into "1", "4", "billion") and possessives; trained tokenizers with abbreviation and exception lists keep "U.K." and "1.4" whole.
- **Normalization** merges surface forms into one term to raise recall: case folding, Unicode normalization (the ligature "ﬁ" becomes "fi") and accent folding ("ü" to "u", "ß" to "ss"). [Stemming](#stemming-and-its-trade-off) and [lemmatization](#lemmatization) go further; misspellings belong to [Vocabulary mismatch](#vocabulary-mismatch).
- Every merge can cost precision: accent folding makes German "schon" (already) and "schön" (beautiful) one token. Unicode normalization has no downside and is the default; case folding fits case-insensitive search; accent folding fits when users cannot easily type the accents (cross-language web search usually folds, a single-language legal index does not).
- Exactness is also a strength. Product IDs, error codes and rare names match exactly as tokens, where dense embeddings may return similar but wrong results. Hybrid search therefore keeps BM25 next to dense retrieval ([semantic sheet](cheat-sheet-semantic.md#designing-a-semantic-search-pipeline)).

### Character n-grams

- Overlapping character n-grams can be the index terms. Trigrams with a `#` word-start marker turn "This course teaches" into `#thi his #cou our urs rse #tea eac ach che hes`. For text without spaces, the markers are left out and the whole character stream is cut into trigrams. Queries are encoded the same way.
- Unlike stemming and lemmatization, n-grams need no language rules and no dictionary.
- Matches are partial, so they survive inflection and typos: for the query "teach multtimedia", 10 of the 12 query trigrams occur in "teaches multimedia retrieval".
- The same principle (match parts of a word when the whole word fails) is behind the fixed-size, learned sub-word vocabularies of transformers (BPE, WordPiece), which build unknown words from known pieces (chapter 5).

### Stop words

- A stop list holds the frequent function words of a language (100 to 200 for English); they appear in most documents and cannot discriminate.
- Removal has a cost when the stop word carries the content: Stephen King's "It"; "IT security", which leaves only "security"; "To be or not to be", which disappears; "The Who".
- Phrase queries are the most fragile: removal turns "The Who" into an empty query and "not guilty" into "guilty". The filter must respect the phrase context or skip those tokens.
- POS-aware filtering keeps "IT" as a noun in "the IT department" and drops "it" as a pronoun in "it is easy" (see [Query understanding](#query-understanding)).
- BM25 makes removal less necessary. IDF pushes the weight of terms that occur everywhere toward zero ([Term discrimination and IDF](#term-discrimination-and-idf)), and saturation keeps a repeated term from dominating ([BM25](#bm25)).
- The modern default: keep stop words in the index (storage is cheap), and skip them only when building phrase vocabularies, computing term-based similarities or building compact features for a classifier.

### Stemming and its trade-off

- A stemmer strips suffixes with ordered rules until it reaches a shared pseudo-stem. The stem need not be a word; it only has to be the same for all forms ("computing", "computation", "computer" become "comput"; "carrying" and "carried" become "carri"). Stemming is fast and needs no dictionary.
- It runs on documents and queries alike, so the query "computation" (stemmed to "comput") matches documents that contain "computer".
- Merging forms raises recall, and every wrong merge costs precision:
  - **Over-stemming** merges unrelated words ("universal" and "university" become "univers") and lowers precision.
  - **Under-stemming** keeps related forms apart ("alumnus" and "alumni"; "be", "was", "been"; "go" and "went" share no suffix) and lowers recall.
- **Porter** is the classical English stemmer (about 60 rules). **Lancaster** is more aggressive, repeating its rules until none fires: fast, but shorter stems and more collisions ("one" and "only" become "on"), acceptable when recall matters in a large collection, a problem for short documents, name search and precise queries. **Snowball** is Porter's framework for about 20 European languages; its German stemmer still keeps "Reparatur" and "reparieren" apart.
- The rules are written for one language. An English stemmer on French text strips letters ("les" becomes "l") without merging the right forms, so the language has to be known first.
- Stemming is the choice when speed matters or no dictionary is available.

### Lemmatization

- A **lemma** is the dictionary base form ("went" to "go", "better" to "good", "mice" to "mouse"). A **stem** is a pseudo-form that only has to be the same for all forms of a word ("comput"). The two are different things.
- A lemmatizer combines rules for regular inflections, an exception list for irregular forms and a dictionary of base forms, all for one language. Words it cannot resolve (names, misspellings, loanwords) come back unchanged.
- It needs the part of speech. "meeting" as a noun stays "meeting"; as a verb it becomes "meet". The wrong tag gives the wrong lemma.
- It merges only the forms of one lemma. Unrelated words such as "universal" and "university" stay apart, so there are fewer wrong merges and less loss of precision. Derived words such as "computer" and "computation" also stay apart, so recall is lower than with a stemmer.
- It is strongest for inflected languages and strong verbs: French "avons" becomes "avoir", where Snowball gives "avon"; German "Häuschen" stays a lemma, where Snowball gives the spurious stem "hausch".
- The cost is a POS tagger plus a dictionary lookup per token, against a few string operations for stemming. Lemmatization is preferred when accuracy on inflected languages or strong verbs matters more than throughput.
- Dense embeddings learn these normalizations on their own, but hybrid search keeps stemming or lemmatization on the BM25 side, which needs it for recall.

**Go deeper:** [Stemming and lemmatization](https://roger-weber.github.io/mmir-unibasel-hs26/book/stemming-and-lemmatization/)

### Term discrimination and IDF

- **Zipf's law:** if terms are ranked by frequency, the probability of the term at rank $r$ is roughly $c/r$. A few terms dominate the collection and most terms are rare.
- This gives three zones. The most frequent terms appear everywhere and cannot discriminate. The rarest terms are very specific but seldom occur in queries. The most useful terms for separating relevant from non-relevant documents lie in between.
- The intuition: remove a term from every document. If the documents become harder to tell apart, the term was discriminating; removing "the" changes nothing.
- **IDF** weights a term by how few documents contain it, using the **document frequency** df(t): the number of documents that contain the term at least once, not its occurrences within a document. IDF closely tracks the measured discrimination power of terms.
- **Probabilistic IDF.** BM25's IDF is the BIR term weight without relevance information ($r_j = 0.5$, $n_j$ from df), a weight for how well a term separates relevant from non-relevant documents ([BIR and relevance feedback](#bir-and-relevance-feedback)). It turns negative when a term occurs in more than half of the collection, which would let a matching term lower a score. The Lucene variant stays positive and is almost identical to the classical IDF. The variants share the intuition, but their values are not interchangeable.

**Math** (*for understanding; formulas are not asked in the exam*): the smoothed classical IDF, the original BM25 IDF and the positive Lucene variant.

$$\text{idf}(t) = \log\frac{N+1}{\text{df}(t)+1} \qquad \text{idf}_{\text{BM25}}(t) = \log\frac{N-\text{df}(t)+0.5}{\text{df}(t)+0.5} \qquad \text{idf}_{+}(t) = \log\frac{N+1}{\text{df}(t)+0.5}$$

## Retrieval models

### Document representation

- A document becomes a vector with one component per vocabulary term. The representations differ in what each component records:
  - **Set-of-words:** presence (1) or absence (0). Three occurrences of "cat" look the same as one.
  - **Bag-of-words:** the count tf(D, t) of each term.
  - **tf·idf:** the count times the term's rarity in the collection. Terms that are frequent in this document but rare elsewhere dominate.
- **BM25's term-frequency component** keeps the count but bends it:
  - **Saturation:** the first occurrence counts strongly, each further one adds a little less, and the total stays below a fixed limit. Keyword stuffing therefore gains little.
  - **Length normalization:** one occurrence weighs more in a short document than in a long one, because it covers a larger share of the content. A longer document needs proportionally more occurrences for the same weight.
- **Sparsity.** A vocabulary can hold millions of terms, but a document uses only a few hundred or thousand, so nearly all components are zero. The inverted file stores only the non-zero entries, and a query touches only documents that share a query term. Why dense embeddings cannot use this is covered in the [semantic sheet](cheat-sheet-semantic.md#lsi-and-static-word-embeddings).
- All of these representations ignore order: "New York" and "York New" give the same vector. A **positional index** keeps a **hit list** of every position of a term in a document, which proximity scoring needs ([Phrases and proximity](#phrases-and-proximity)).

**Math** (*for understanding; formulas are not asked in the exam*): the tf·idf weight of term $t_j$ in document $D_i$.

$$d_{i,j} = \text{tf}(D_i, t_j) \cdot \text{idf}(t_j)$$

### Boolean and extended Boolean retrieval

- **Standard Boolean.** The query is a predicate over the set-of-words representation, built from "term present", "term absent", AND and OR. Every query can be rewritten as an OR of ANDs (disjunctive normal form).
- The result is an unordered set: every match has the same status, however often or wherever the terms occur. Evaluation uses set operations on the documents of each term (AND is an intersection, OR a union), which the inverted file stores.
- **All or nothing.** In the book's 12-document example, `cat AND dog AND trees` returns nothing, although several documents are plausible partial matches; `cat OR dog OR trees` returns 11 of the 12, including one about binary search trees.
- Strengths: precise, explainable semantics, fast set operations and a natural fit with metadata filters (`language = English`). The model suits exact filtering better than ranking.
- **Extended Boolean** keeps the query language and its tree and changes the evaluation. Documents are represented by weighted terms (normalized tf·idf), each atomic predicate gets a score between 0 and 1, and soft AND and OR operators (such as the p-norm) combine the scores along the tree. The result is a ranked list in which a document that misses a term gets a lower score instead of being dropped.
- The combination is a heuristic choice: Boolean algebra defines AND and OR only for true and false, so another operator, parameter or normalization can change the order. In the book's example, a document with four "dog" and no "cat" narrowly outranks one with both terms for `cat AND dog`. Users still have to write Boolean expressions.
- Today's form: strict Boolean filters combined with optional scoring clauses. Boolean OR as the candidate stage over postings is covered in the [index sheet](cheat-sheet-index.md#ranked-retrieval-over-postings-taat-and-daat).

### Vector space model

- The query is treated as a short document in the same tf·idf space. Free-text queries need no operators, a document can score with a single shared term, and the output is ranked.
- **Inner product:** the sum of query weight times document weight over the shared terms. It grows linearly with term frequency and has no upper bound (a retrieval status value, not a similarity in [0, 1]). Long documents and repeated terms win, and authors can push a page up by repeating terms.
- **Cosine:** the inner product divided by both vector lengths. It measures direction, not length, and lies between 0 and 1 for tf·idf vectors. This removes the length effect, but every document term enters the norm: the non-query terms of a long document lower its score, although they add nothing negative to the numerator.
- Running example for `cat dog forest`: the short "Cat Dog Forest." gets the maximum cosine of 1. A longer document that repeats all three terms twice ranks first by inner product but gets a cosine of only 0.342 because of its many other terms.
- Neither measure has saturation or a robust length normalization; both are heuristics, and terms stay independent (no "feline" for "cat"). BM25 replaced the model as the primary lexical ranker.
- **Fast evaluation.** Only query terms contribute. The system reads just their postings, accumulates a score per candidate, divides by the document norms stored at indexing time (not computed per query) and sorts. Geometrically, the inner product projects each document onto the few query dimensions. Cosine as an inner product of normalized vectors is covered in the [index sheet](cheat-sheet-index.md#similarity-measures).
- **Where it lives on:** tf·idf with cosine remains a transparent baseline for document similarity, clustering and near-duplicate detection, and inner product and cosine became the similarity measures of dense embeddings and nearest-neighbour search ([index sheet](cheat-sheet-index.md#similarity-measures)).

**Math** (*for understanding; formulas are not asked in the exam*): the inner product and the cosine of query vector $\mathbf{q}$ and document vector $\mathbf{d}_i$.

$$\text{sim}_{\text{dot}}(Q, D_i) = \sum_{j=1}^{M} q_j \, d_{i,j} \qquad \text{sim}_{\cos}(Q, D_i) = \frac{\sum_{j=1}^{M} q_j \, d_{i,j}}{\sqrt{\sum_{j=1}^{M} q_j^2} \cdot \sqrt{\sum_{j=1}^{M} d_{i,j}^2}}$$

### BIR and relevance feedback

- The **Binary Independence Model** (BIR) ranks by the odds of relevance. It assumes binary term presence, independent terms, and non-query terms that occur equally often in relevant and non-relevant documents, so only the query terms present in a document count, as a sum of term weights.
- $r_j$ is the probability that term $t_j$ occurs in a relevant document, $n_j$ that it occurs in a non-relevant one. The weight is positive when the term is more typical of relevant documents.
- **Without feedback**, $r_j = 0.5$ and $n_j$ comes from the document frequency. This gives BM25's IDF, which therefore needs no relevance judgments.
- **Relevance feedback.** The user marks results as relevant or not; the system re-estimates $r_j$ and $n_j$ by counting in both samples (with pseudo-counts, so that no estimate is exactly 0 or 1) and recomputes the weights, also for terms that were never in the query:
  - A strongly positive weight marks an expansion term ("woodland" in the book's example).
  - A weight near zero means there is no evidence either way, so the term is dropped.
  - A strongly negative weight can penalize documents (a term that occurs only in non-relevant results).
- Explicit feedback is rarely applied directly. A user who judges ten results gives a sample far too small for reliable estimates, and no user will rate hundreds.
- The idea returns indirectly. **Pseudo-relevance feedback** treats the top results as relevant and picks expansion terms with the same weights (see [Vocabulary mismatch](#vocabulary-mismatch)). Web engines turn clicks into implicit labels for learning to rank (chapter 8).

**Math** (*for understanding; formulas are not asked in the exam*): the BIR score with its term weight, and the estimates without feedback.

$$\text{sim}_{\text{BIR}}(Q, D_i) = \sum_{t_j \in Q \cap D_i} c_j \qquad c_j = \log\frac{r_j (1 - n_j)}{n_j (1 - r_j)} \qquad r_j = 0.5, \quad n_j = \frac{\text{df}(t_j) + 0.5}{N + 1}$$

### BM25

- BM25 keeps BIR's additive sum and its probabilistic IDF, and answers the problems of vector-space scoring:
  - **Saturation:** the term-frequency part has a fixed upper bound, so repetition no longer pays without limit.
  - **Length normalization:** saturation depends on the document's length relative to the collection average. A long document needs proportionally more occurrences and cannot win by its length alone.
  - **No vector norms:** the score is a plain sum over the query terms present, so non-query terms cannot lower it, as they do with cosine. A matching query term can, however: with the original BM25 IDF, a term that occurs in more than half of the collection gets a negative weight. Many implementations therefore use the always-positive Lucene variant ([Term discrimination and IDF](#term-discrimination-and-idf)).
- In the running example, the short exact match "Cat Dog Forest." ranks first, and the document with four "dog" and no other query term falls behind documents with several query terms.
- BM25 stays lexical: no "feline" for "cat", one "forest" for both senses, and tuning its parameters does not change that. Synonyms and homonyms need [query processing](#vocabulary-mismatch) or embeddings. A BM25 score is a ranking value, not a calibrated probability of relevance.
- **Where it stays:** BM25 is the default lexical ranker of Lucene-based engines and the first stage of hybrid search and RAG. Exact matches on IDs, names, brands and rare terms are where it beats embeddings; product search needs hybrid retrieval because product IDs need BM25 ([semantic sheet](cheat-sheet-semantic.md#designing-a-semantic-search-pipeline)).
- **BM25F** extends BM25 to fields (title, headings, body, anchor texts): weight the counts per field, add them to one term frequency, and saturate once. If each field saturated separately, a page could collect the full score once per field by repeating a keyword everywhere. This matters on the web, where keyword stuffing is a standard SEO trick: in the book's example, nine well-placed occurrences of "university" score almost as high as forty stuffed ones.

**Math** (*for understanding; formulas are not asked in the exam*): BM25 with the positive Lucene IDF $\text{idf}_{+}$; $k_1$ controls how fast the term frequency saturates, $b$ how strongly the length $|D_i|$ relative to the average length avgdl counts. BM25F replaces tf by the field-weighted sum.

$$\text{sim}_{\text{BM25}}(Q, D_i) = \sum_{t \in Q \cap D_i} \text{idf}_{+}(t) \cdot \frac{\text{tf}(D_i, t) \cdot (k_1 + 1)}{\text{tf}(D_i, t) + k_1 \cdot \left(1 - b + b \cdot \frac{|D_i|}{\text{avgdl}}\right)}$$

$$\widetilde{\text{tf}}(t, D) = \sum_{f} w_f \cdot \text{tf}_f(t, D)$$

**Go deeper:** [Probabilistic ranking and BM25](https://roger-weber.github.io/mmir-unibasel-hs26/book/probabilistic-ranking-bm25/), [retrieval models demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch01-02-retrieval-models.ipynb)

## Query processing

### Phrases and proximity

- Bag-of-words matches "new" and "york" anywhere ("a new book from York University"), so incidental co-occurrences rank like the real phrase. Every "New York" document is still found: the problem is precision and ranking, not recall.
- **Index time: phrase tokens.** A fixed vocabulary of selected bi-grams is emitted as extra tokens (`new_york`) in one pass over each document and query. The single tokens stay, so "Holmes" still matches a document that only says "Sherlock Holmes".
- Indexing every bi-gram is unaffordable (a 100k-term vocabulary would become 10 billion pairs, mostly noise such as "of the"), so the bi-grams are selected offline from corpus statistics. A missed bi-gram costs ranking quality, not recall. Systems stop at bi-grams and leave longer phrases to proximity.
- **Query time: proximity.** Merge the hit lists of the query terms by position, pair neighbouring hits of *different* terms, count the pairs per distance, and reward close pairs more. Pairs far apart do not count, so more pairs help only when they are close.
- Proximity works for any combination of query terms, including the new names of every news event, and needs no list of known phrases. Its costs: the positional index stores every position and is considerably larger, and mentions by surname alone form no pairs. The same positions also answer phrase queries.

### Vocabulary mismatch

- Query and document use different words for the same concept, and lexical matching misses the document: a **recall** problem. Stemming and lemmatization do not help, because "buy" and "purchase" reduce to unrelated forms.

| Form | Example | Classical means |
|---|---|---|
| Synonyms, domain shorthand | "buy" and "purchase"; "F1" and "Grand Prix" | synonym lists, query expansion |
| Broader and narrower words | "dog" and "labrador" | narrower terms added at lower weight; category trees |
| Compounds | "Bücherregal" and "Regal"; "QueryParser" and "parser" | index the compound and its parts |
| Misspellings, name variants | "Goehte" for "Goethe"; "Britny" for "Britney" | spell correction; variant expansion |

- **Synonyms.** WordNet covers linguistic synonyms. Domain shorthand, brand aliases and category synonyms ("laptop" and "notebook") need lists maintained by the search team, which work for the frequent, anticipated terms but cannot keep up with the long tail.
- **Hypernyms and hyponyms** form a hierarchy. Shops turn it into category trees for faceted navigation (drill down, or broaden to the parent). Narrower terms enter a query at a lower weight than the original ("cat" adds "kitten", "siamese").
- **Compounds** hide several words in one token (German, Finnish, Dutch; code identifiers in code search). The index keeps the compound together with the parts of its plausible splits, which are generated by hyphenation rules, filtered by a dictionary and ranked by part frequency. **Endocentric** compounds ("sunglasses" are glasses) split safely; **exocentric** ones ("skyscraper" is not a scraper) add tokens with the wrong meaning and cost precision. News search gains from splitting, legal search often does not.
- **Misspellings.** The index keeps the original token and adds the corrected form; at query time the system runs the corrected query or asks "did you mean". Edit distance and phonetic codes find candidates. For names, one person's typo is another person's correct name, so the original spelling stays and the query is expanded to the common variants instead of rewritten to one form.
- **Expansion or rewriting.** Expansion adds terms and keeps the user's, usually with a lower weight for the added ones so that exact matches still rank first. Rewriting replaces or restructures the user's terms (spell correction, alias substitution such as "F1" to "Formula 1", date normalization). A rewrite that keeps the original terms is an expansion in disguise.
- **Semantic search** learns synonyms from context and covers the long tail without lists, learns hypernyms less reliably, and its sub-word tokenizers often split compounds into incoherent fragments. Hybrid stacks therefore keep lists for the frequent terms and the decompounder on the BM25 side, and leave the tail to embeddings ([semantic sheet](cheat-sheet-semantic.md#designing-a-semantic-search-pipeline)).
- Every merge can cost precision: a wrong synonym, an exocentric split, two people merged into one name.

**Go deeper:** [Query understanding](https://roger-weber.github.io/mmir-unibasel-hs26/book/query-understanding/), [Phrases and compounds](https://roger-weber.github.io/mmir-unibasel-hs26/book/phrases-and-compounds/)

### Ambiguity

- One surface form, several concepts. Homonyms such as "bank", "apple", "jaguar", "lead" and the "forest" of "random forest" make the result list mix senses: a **precision** problem, whereas synonyms hurt recall. Lemmatization keeps one lemma for "bank" whichever sense is meant.
- Merging surfaces creates the same problem: over-stemming, accent folding ("schon" and "schön"), an exocentric compound split.
- **Classical means read the context:** surrounding words ("money", "loan" or "river", "shore"), POS tags ("lead" as the metal or the verb), and phrase tokens.
- **Web engines** rank first the interpretation most users mean, learned from earlier clicks (the Apple company page for "apple"), and place rarer senses a few positions further down the first page (result diversification).
- **Contextual embeddings** (sentence encoders, transformers) give a word a different vector in each context and resolve the sense without rules ([semantic sheet](cheat-sheet-semantic.md#bi-encoders-cross-encoders-and-colbert)).
- All of these means need context. A one-word query offers none, so the system can only show several senses or ask ("did you mean").

### Query understanding

- Query understanding turns a raw query into a structured query (language, entities, intent, corrected tokens) that the system can act on.
- **Language detection comes first**, because the stop list, stemmer and lemmatizer depend on it. For long documents, rules suffice: the script, distinctive characters ("ß", "ñ"), counts of stop words and frequent words. Short queries give the rules too little evidence ("pain" is French and English) and need a classifier over character n-grams, with a fallback language when the top confidence is close to the second.
- **POS tagging** assigns word classes from context ("run" as a noun or a verb). It drives selective stop-word filtering ("IT department"), gives the lemmatizer its input and reveals the question form: a WH-word, a copula and a proper noun ("Who is Albert Einstein?") point to a person lookup.
- **Named entity recognition** labels spans (person, location, organization, product, date, money), and each type triggers an action: a person leads to person-oriented sources, a date to a recency filter, a location to a hard filter and a map, a brand to shopping and review pages.
- The order follows these dependencies: language before every language-specific step, POS tags before selective stop-word filtering, entities before intent classification (entity types are its features). WH-words are kept as classification features rather than removed as stop words.

**Go deeper:** [Intent routing and classification](https://roger-weber.github.io/mmir-unibasel-hs26/book/intent-routing-and-classification/) (end-to-end walkthrough)

### Intent routing

- The router classifies a query into the backends the system supports: book, web, image, product, people, news and map search, plus one-shot answers (weather, calculator, definition, translation).
- **Features:** the token bag after stemming and stop-word removal, POS counts, entity labels (has-PERSON, has-LOCATION, has-DATE), question markers and the detected language. The training data are query logs labelled with the backend the user clicked into.
- **The prior.** The logs reflect today's backend distribution, not the ideal one: with an empirical prior, queries with weak evidence go to the dominant backend regardless of their features. Intent routing therefore uses a uniform prior, or balances the training data first.
- Naive Bayes is the classical baseline (the same model as for language detection). Its independence assumption cannot express conjunctions such as "has-PERSON and WH-word", so production systems use gradient-boosted trees or small transformers.
- The router does not retrieve: it picks a backend and passes on a structured query with filters (for a library catalogue, language and author).
- **LLMs** can run the whole pipeline in one prompt, at the price of latency (100 to 1000 ms against under a millisecond), a measurable cost per query, and confidence: a classifier's posterior can be thresholded and calibrated, while the confidence an LLM writes into its answer is generated text, not a probability. Production stacks run the classical pipeline first and escalate low-confidence queries to an LLM.

## Web search

### How web search differs from classical retrieval

- Classical retrieval assumes that every document deserves to be found and that its text describes it honestly. On the web, neither holds:
  - no quality control (outdated pages, errors, scams, spam);
  - mixed formats with little reliable structure beyond title, URL and text;
  - queries of two or three words without context;
  - millions of matches that look equally similar to the query;
  - many copies of the same content, which term matching scores alike.
- Ranking is adversarial, because visibility is worth money. SEO has a legitimate side (descriptive titles, clean HTML, good content that others link to) and a manipulative one (keyword stuffing, hidden text, link farms, cloaking). The page text is exactly what a site controls.
- Query types by the user's goal (Broder):
  - **navigational**: one specific site, usually one correct result, expected at rank 1;
  - **transactional**: buy, book, download;
  - **informational**: knowledge about a topic.

  A classical system serves the narrow intents of its domain (artists, albums and songs in a music database). These three types are a coarse version of the routing intents of chapter 3.
- Query frequencies follow Zipf's law. A **fat head** of repeated, mostly navigational queries can be served from a table of query-target pairs. A **long tail** of rare and new queries has little or no click history, and this is where engines compete.
- As a result, ranking draws on evidence from outside the page text: links and anchors written by others, clicks, the location and language of user and page, and site quality and trust. With hundreds of signals, the combination is learned (learning to rank), not set by hand.

### Ranking signals beyond the page text

- Besides proximity and BM25F field weighting (see [BM25](#bm25)), four signals come from outside the term counts:

| signal | obtained from | applied as |
|---|---|---|
| anchor text | links on other pages, stored under the target URL at crawl time | anchor index with its own score, usually weighted above the page text |
| location | page: directory entry, address, country domain, server IP; user: GPS, IP, a place in the query | boost as strong as the query's local intent |
| language | page: detected at indexing; user: query, interface, browser settings | hard filter, or a learned feature |
| quality and trust | rater judgments and classifiers trained on them; site ownership, age, reputation | penalties, removal of whole sites, trust feature |

- **Anchor text** is written by others, so the page owner cannot control it, and it also describes pages with little text or pages never fetched. Its count saturates gently, for example $\log(1 + n)$, because many independent votes are the evidence wanted. **One vote per domain** defuses Google bombs and link campaigns: three spam domains earn three votes, however many anchors they place.
- **Location:** "best pizza" depends strongly on it, "Albert Einstein" hardly. A place named in the query takes precedence over the user's position ("best pizza in Paris" typed in Basel).
- **Language:** a filter is all or nothing and also removes an English page that would be the best result when no good German page exists. As a learned feature, the ranker learns how strongly to prefer the query language: German pages first for a recipe, English pages still high for an error message of a programming framework.
- **Trust** favours established sites. This is a cost for an excellent new site, which freshness and diversity must offset.

**Go deeper:** [web ranking signals demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch08-01-web-ranking-signals.ipynb)

### Link analysis: PageRank and HITS

- A link is an implicit recommendation. Counting incoming links (in-degree) treats every link as an equal vote and is easy to inflate with a link farm.
- **PageRank** is the share of time a random surfer spends on a page. The surfer follows a random outgoing link or now and then jumps to a random page.
  - A link counts with the PageRank of its source, split among the source's outgoing links. In the book's seven-page example, a page with two links from the central page outranks one with four links from weak pages.
  - A link farm that nobody links to collects only the small jump shares of its pages.
  - The random jump keeps closed groups of pages (rank sinks) from absorbing all importance and guarantees a unique solution.
- PageRank is **static and query-independent**: computed offline for the whole web and used as one feature next to content signals. It measures prominence, not relevance; weighted too strongly, it puts the same famous pages on top of every query. New pages have few incoming links, which freshness signals must offset.
- **Personalized PageRank** jumps back to a chosen set of pages instead of a random page. This gives importance from the point of view of one topic or one user ("who to follow").
- **HITS** is **query-specific**. It grows a small graph from the top text results and the pages linked to or from them. On this graph, **hubs** link to good authorities and **authorities** are linked by good hubs. HITS finds the best sources for one topic, but it is costly at query time and fragile, since a few strongly linked off-topic pages can take over.

**Math** (*for understanding; formulas are not asked in the exam*), with $L(q)$ the number of outgoing links of $q$ and $N$ the number of pages:

$$
\text{PR}(p) = \frac{1 - \alpha}{N} + \alpha \cdot \sum_{q \rightarrow p} \frac{\text{PR}(q)}{L(q)}
$$

HITS updates, each followed by normalization:

$$
a(p) = \sum_{q \rightarrow p} h(q) \qquad h(p) = \sum_{p \rightarrow q} a(q)
$$

**Go deeper:** [Link Analysis: PageRank and HITS](https://roger-weber.github.io/mmir-unibasel-hs26/book/link-analysis/)

### The ranking stack and learning to rank

- Ranking runs in stages, and each stage applies a more expensive model to fewer candidates:
  1. the retriever takes about ten thousand candidates from the inverted index with a cheap score (BM25F);
  2. a learned ranker orders them by a feature vector of hundreds of signals;
  3. a cross-encoder rescores the top hundred.
- **Learned, not hand-tuned:** in the book's "uni basel" example, three quite different weight vectors rank perfectly, and one query cannot tell them apart. The weights must work over thousands of queries and hundreds of signals and be chosen again whenever these change, so they are learned from judged examples.
- A weighted sum adds each signal regardless of the others. Tree ensembles also capture interactions, for example that PageRank should count only when the page matches the query.
- Features are query-dependent (BM25F, proximity, anchor score), query-independent (PageRank, site quality, URL depth) or describe the query alone (length, intent), which lets the model weight the other features per kind of query.
- **Human raters** give precise but expensive labels, used mainly for test sets (offline nDCG, see the [evaluation sheet](cheat-sheet-eval.md#graded-relevance-dcg-and-ndcg)) and hard queries.
- **Clicks** are plentiful but biased:
  - **Position bias:** users click top results largely because they are on top. A click at rank 5, after reading past four results, is stronger evidence than a click at rank 1.
  - Clicks are reweighted for position before training: a click at rank $k$ counts $1/p_k$, where $p_k$, the probability that users look at rank $k$, is estimated with small randomized swaps.
  - A click shows that a result looked promising, not that it satisfied. Dwell time and quick returns refine it.
- The ranker is retrained continuously and checked online before release (A/B tests, interleaving). The loop has a blind spot: a page that is never shown collects no clicks.

**Go deeper:** [Learning to Rank](https://roger-weber.github.io/mmir-unibasel-hs26/book/learning-to-rank/)

### Pointwise, pairwise and listwise training

- The three are training goals, not model types: any scoring model can be trained with any of them.
- **Pointwise** predicts each page's grade on its own. It spends effort on errors that leave the order unchanged, misses small errors that swap two pages, and treats grades as comparable across queries.
- **Pairwise** orders each pair with different grades. This is closer to the goal, but every pair counts the same: a swap at ranks 1 and 2 costs as much as one at ranks 40 and 41.
- **Listwise** weights each error by its effect on a ranking measure such as nDCG, so the top of the list counts most. nDCG changes in jumps when two scores cross, so the push on each pair is scaled by how much nDCG would change if the two pages swapped places.

**Math** (*for understanding; formulas are not asked in the exam*), with scores $s$, grades $y$ and the logistic function $\sigma$:

$$
\text{pointwise:} \quad \mathcal{L} = \sum_{D} (s_D - y_D)^2
$$

$$
\text{pairwise:} \quad \mathcal{L} = \sum_{(i,j) : \, y_i > y_j} -\log \sigma(s_i - s_j)
$$

$$
\text{listwise (pair weight):} \quad \lambda_{ij} = |\Delta \text{nDCG}_{ij}| \cdot \sigma\big(-(s_i - s_j)\big)
$$
