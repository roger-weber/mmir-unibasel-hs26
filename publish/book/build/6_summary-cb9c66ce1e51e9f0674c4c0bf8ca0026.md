---
author: Roger Weber
edition: HS26
status: updated
book_part: Foundations
chapter: Advanced Text Processing
section: Summary
order: "3.6"
---

# Summary

## Method Comparison

Stemmers and lemmatizers side by side.

| Method | Approach | Correctness | Speed | Multi-lingual | Needs dictionary |
|--------|----------|-------------|-------|---------------|------------------|
| Porter | Rule-based suffix stripping (English) | pseudo-stem | fast | English only | no |
| Lancaster | Aggressive rule-based (English) | pseudo-stem, over-stems short words | fastest | English only | no |
| Snowball | Rule-based framework | pseudo-stem | fast | 20+ languages | no |
| WordNet | Dictionary + POS lemmatizer | linguistic lemma | slow | English only (multilingual variants weaker) | yes |
| spaCy | Neural POS tagger + dictionary lemmatizer | linguistic lemma | medium | many languages | yes |

Use a rule-based stemmer when performance dominates or when no dictionary is available for the language. Use a dictionary-based lemmatizer when accuracy on strongly-inflected languages matters (German case-inflected nouns, English strong verbs like "went -> go") or when the downstream task needs linguistic base forms rather than arbitrary reductions. Snowball is the go-to compromise for multi-language corpora when only rule-based options are practical.

## Key Takeaways

1. Classical retrieval fails in two directions: one concept has many surface forms on the document side (recall gap), and free-text queries carry structure that the retriever ignores (query understanding gap). Advanced text processing addresses both.
2. Tokenization is the first stage and the largest source of quality issues. A naive regex tokenizer breaks on possessives, numbers, abbreviations, and punctuation. Modern nltk and spaCy tokenizers handle these but still need retrieval-oriented cleanup.
3. Normalization decides which surface forms collapse to the same token. Case folding is nearly always applied. Unicode normalization is applied silently. Accent folding trades recall for precision and depends on the scenario.
4. Sentence segmentation is a small but essential step. Every RAG chunking strategy, every POS tagger, and every query-analysis pipeline depends on being able to find sentence boundaries reliably.
5. Stop-word removal is a size-versus-recall trade-off. Modern BM25-based systems handle stop words gracefully through IDF and can afford to keep them in the index. Legacy vector-space systems remove them aggressively. Phrase queries and titles like Stephen King's "It" break under aggressive removal.
6. Stemming is a fast pseudo-linguistic reduction; lemmatization is a slower dictionary-based one that produces the true linguistic base form. Choose based on speed budget, language complexity, and whether downstream steps need real words.
7. Phrases and compounds sit on opposite sides of the word-boundary problem. Bi-grams like "New York" pack multiple tokens into one concept and need explicit phrase indexing; compounds like "Wolkenkratzer" hide multiple concepts in one token and need compound splitting. PMI, LHR, and log-frequency scoring are the classical selectors.
8. Query understanding turns free text into structured features: language, POS tags, named entities, corrected spelling. Together they let the retriever route the query to the right backend and generate structured queries rather than doing bag-of-words matching.
9. Naive Bayes over engineered features is enough to build both a language detector and an intent classifier. Modern production stacks use neural classifiers for accuracy but keep Naive Bayes as the baseline and as the fast path for cost-sensitive deployments.
10. Every classical technique in this chapter still runs in production 2025 search stacks. Dense retrieval subsumes some of them (synonym expansion) implicitly, but hybrid retrieval combines both branches and needs the classical stack on the lexical side.

## Key Formulas

$$\text{pmi}(t_1, t_2) = \log_2 \frac{N \cdot \text{tf}(t_1, t_2)}{\text{tf}(t_1) \cdot \text{tf}(t_2)}$$

Pointwise Mutual Information: high when two tokens almost always appear together and rarely apart. Biased toward rare-word bi-grams; requires a minimum-frequency filter.

$$\log \lambda = \log \frac{L(H_1)}{L(H_2)}$$

Log Likelihood Ratio: log-ratio of the probability under independence to the probability under dependence. Robust on sparse data; ranks by $-2 \log \lambda$.

$$\text{score}(S) = \frac{1}{|S|} \sum_{p_i \in S} \log \frac{\text{tf}(p_i)}{N}$$

Compound Split Score: average log-frequency of the parts in split $S$. Choose the split with the highest score.

$$\hat{C} = \arg\max_{k} \left( \log P(C_k) + \sum_{j=1}^{M} \log P(x_j \mid C_k) \right)$$

Naive Bayes Maximum A Posteriori: pick the class that maximizes the log-prior plus the sum of log-likelihoods. Used for language detection over character n-grams and for intent classification over feature vectors combining tokens, POS tags, and NER labels.

```{admonition} Exam focus
:class: attention
- The two failure directions classical retrieval hits, and which chapter technique addresses each
- Why possessives, abbreviations, and multi-token numbers break the naive regex tokenizer
- The difference between stemming (pseudo-stem, rule-based) and lemmatization (real lemma, dictionary-based), and when to choose each
- Why aggressive stop-word removal breaks queries like "The Who" and "IT security", and how BM25's IDF weighting reduces the need for aggressive removal
- How PMI and LHR rank the same bi-grams differently and why both are useful
- Why endocentric compounds split usefully and exocentric ones do not
- The three retrieval-side uses of POS tagging (stop-word filtering, lemmatizer disambiguation, question-form analysis)
- How language detection and intent classification are the same Naive Bayes machinery with different features and classes
- Where in modern hybrid retrieval each classical step still runs, and where dense embeddings subsume it
```

## Self-Check Questions

1. (Understand) The naive tokenizer produces `['Watson', 's']` from "Watson's". Under what retrieval scenario is this desirable, and under what scenario is it a problem? What information is permanently lost?
2. (Understand) Why does BM25 tolerate stop words that appear in the index better than a raw vector-space TF-IDF model does?
3. (Analyze) Given a Sherlock Holmes corpus where "said" occurs 207 times, "Holmes" 94 times, "Sherlock" 51 times, "Sherlock Holmes" 48 times, and "said Holmes" 12 times, compute PMI for the two bi-grams. Explain why the ranking matches (or does not match) what a human would call "the more meaningful phrase". Assume a corpus of $N = 43{,}000$ tokens.
4. (Analyze) Explain why aggressive accent folding hurts a legal-document retrieval system for German but helps a general-purpose web search engine over the same language.
5. (Analyze) A retrieval system indexes "Wolkenkratzer" only as itself (no compound splitting). A user queries "Wolke". A different user queries "Kratzer". Which query, if any, retrieves documents about skyscrapers, and why? Would splitting help both users equally? Would you split this specific compound?
6. (Evaluate) A production search team proposes replacing the entire classical query pipeline (tokenization, stemming, POS, NER, intent classification) with a single call to a large language model that outputs a structured query directly. State two reasons the team might keep the classical pipeline anyway, and one query type where the LLM call is likely to win.
7. (Apply) A user queries "Livres de Molière". Walk through the end-to-end pipeline from section 5, listing the output of each step and the final routing decision. Assume the search stack supports French.

```{admonition} Test Your Knowledge
:class: hint
[Take the Chapter 3 Quiz ->](link-to-quiz-app)
```

## Further Reading

- Porter, M. F. (1980). **An Algorithm for Suffix Stripping**. *Program*, 14(3), 130-137. [Author copy](https://tartarus.org/martin/PorterStemmer/def.txt). The original Porter Stemmer paper, still the reference for how to build a rule-based English stemmer and the ancestor of every rule-based stemmer that followed.

- Paice, C. D. (1990). **Another Stemmer**. *ACM SIGIR Forum*, 24(3), 56-61. [ACM copy](https://dl.acm.org/doi/pdf/10.1145/101306.101310). Introduced the Lancaster stemmer and the aggressive suffix-stripping strategy that trades precision for speed.

- Manning, C. D., & Schütze, H. (1999). **Foundations of Statistical Natural Language Processing**, Chapter 5: Collocations. *MIT Press*. The standard reference for PMI, likelihood-ratio testing, and other statistical collocation-scoring methods; source of the LHR formulation used in this chapter.

- Kiss, T., & Strunk, J. (2006). **Unsupervised Multilingual Sentence Boundary Detection**. *Computational Linguistics*, 32(4), 485-525. [Free PDF](https://aclanthology.org/J06-4003.pdf). The Punkt algorithm, still the sentence-segmentation default in NLTK almost two decades later.

- Koehn, P., & Knight, K. (2003). **Empirical Methods for Compound Splitting**. *Proceedings of the 10th Conference of the European Chapter of the ACL*, 187-193. [Free PDF](https://arxiv.org/ftp/cs/papers/0302/0302032.pdf). The frequency-based compound-splitting method this chapter uses, evaluated for machine translation but applicable directly to retrieval.

- Minixhofer, B., Pfeiffer, J., & Vulić, I. (2023). **CompoundPiece: Evaluating and Improving Decompounding Performance of Language Models**. [arXiv:2305.14214](https://arxiv.org/abs/2305.14214). Shows that modern subword tokenizers (SentencePiece, BPE) still decompound German poorly, motivating why explicit decompounders remain in hybrid retrieval stacks.

**Implementations and tools**

- [NLTK](https://www.nltk.org/) — Python toolkit for classical NLP; ships all the stemmers, tokenizers, POS taggers, and collocation finders used in this chapter.
- [spaCy](https://spacy.io/) — Industrial-strength NLP library with neural taggers and lemmatizers for many languages.
- [WordNet](https://wordnet.princeton.edu/) — Lexical database for English synonyms, hypernyms, and hyponyms.
- [Snowball](https://snowballstem.org/) — Multi-language stemmer framework, source of the German and French stemmers used in section 2.
- [lingua-language-detector](https://github.com/pemistahl/lingua-py) — Character-n-gram Naive Bayes language detector covering 70+ languages.
- [Elasticsearch decompounder documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-dict-decomp-tokenfilter.html) — Production compound-splitter configuration for German, Dutch, and other compounding languages.

````{admonition} From ELIZA to Transformers: a short history of NLP (optional reading)
:class: note dropdown

The classical techniques in this chapter descend from a much longer research programme that runs from 1950s symbolic AI through statistical corpus-based methods to today's neural language models. Not exam-relevant, but useful for placing the tools in context and understanding where the field went next.

**Symbolic era (1950s-1980s)**. Rule-based systems dominated NLP. Joseph Weizenbaum's ELIZA (1966) simulated a Rogerian psychotherapist by matching keyword patterns and substituting into templates; users routinely believed it understood them. Terry Winograd's SHRDLU (1971) parsed natural-language commands in a blocks-world simulator, resolving pronouns ("it") and detecting ambiguity ("which pyramid?") explicitly. Both systems worked spectacularly in their narrow domains and failed the moment inputs strayed outside those domains. The lesson was that natural language cannot be fully formalized with hand-written rules.

```{figure} images/figure_3_1.png
:name: fig-eliza-chatbot-terminal-session
:width: 70%

A terminal session with ELIZA (Landsteiner's 2005 reimplementation of Weizenbaum's 1966 program). Simple pattern matching, no world model, and users still developed emotional attachments.
```

```{figure} images/figure_3_2.png
:name: fig-shrdlu-dialogue-session
:width: 70%

A SHRDLU session (Winograd, 1971), tracking pronoun references across turns and reporting ambiguity explicitly when a referring expression is underspecified.
```

**Statistical era (1980s-2010s)**. Hidden Markov Models drove advances in POS tagging, speech recognition, and named entity recognition. N-gram language models powered early machine translation and text generation. Annotated corpora such as Brown and the Penn Treebank provided the labelled data these methods required. The statistical bet paid off: data-driven models beat hand-crafted rules on almost every task. The limits were the size of the labelled datasets, heavy feature engineering, and no ability to capture long-range meaning.

**Neural era (2010s-present)**. Bengio's 2003 neural language model represented words as continuous vectors. Word2Vec (Mikolov et al., 2013) made word embeddings practical at scale. RNNs, then LSTMs and GRUs, handled sequence dependencies. In 2017 the Transformer replaced recurrence with self-attention, enabling parallel processing of full sequences and much longer contexts. The techniques in this chapter still power the lexical-retrieval side of modern hybrid systems; the neural side is covered in the semantic-search chapter.
````

```{bibliography}
:filter: False
```
