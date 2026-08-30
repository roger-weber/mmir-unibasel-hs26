---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Foundations
chapter: Advanced Text Processing
section: Query Understanding
order: "3.4"
---

# Query Understanding

<!--
TODO (rewrite): re-order as:
  1. Homonyms and synonyms (short; WordNet code snippet)
  2. Hypernyms and hyponyms; faceted search
  3. Part-of-speech tagging (compact overview only)
     - What POS is + one figure (figure_3_3 constituency tree)
     - Why it matters for retrieval: stop-word substitution, stemmer
       disambiguation, question-form analysis
     - One spaCy code snippet
     - One sentence on HMM / neural taggers; pointer to
       `appendix_ml_foundations/app_5_hidden_markov_models.md`
  4. Named Entity Recognition
     - What NER is; four routing examples (person / date / location / product)
     - Chunking (rule-based grammar) as an aside
  5. Spell correction and "did you mean?"

Drop from this section (do not carry over):
  - Full HMM transition/emission math (moved to appendix reference)
  - Transformation-based POS tagging with training-loop derivation
  - Deep-learning POS extended explanation (one sentence is enough)
  - Framework tour depth (nltk / spaCy / transformers all in parallel);
    replace with one comparison table + one code snippet per library
-->

## Homonyms and synonyms

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, homonyms/synonyms block -->

When analyzing text, we encounter homonyms and synonyms. A **homonym** is a word spelled the same as another word but with a different meaning and sometimes a different pronunciation. For example, "lead" can mean to guide, or a metal. A **synonym** is a different word with a similar or nearly identical meaning, often used to avoid repetition, for example "big" and "large".

- **Synonyms** are commonly used to add variety to written text. However, this affects the retrieval engine's ability to match query terms with those in the document. If the document contains "purchase", a query with "buy" or "acquire" cannot match it due to the different token forms. There are two main alternatives to address this. First, synonym expansion involves tokenizing the document and/or the query and expanding tokens using predefined synonym lists. Second, as discussed later in this book, word embeddings can be used to map terms into a dense space that captures relationships between words.

- **Homonyms** require analyzing the context to clarify the intended meaning. In straightforward cases, part-of-speech tags can distinguish between verb and noun forms (e.g., "lead" as a guide or as a metal). More advanced solutions use machine-learning models to determine the context accurately or analyze grammatical structures for context. When a query contains a homonym, we can either select the most common meaning or present the user with individual results for each potential interpretation. For example, the word "bank" has several meanings (sloping land by water, financial institution). We can seek user feedback for the correct interpretation or offer two result options with synonym expansion for both possible meanings.

<!-- TODO (rewrite): the raw source has a floating [MATH_ERROR] placeholder here
     with no visible math context. Investigate the original PPTX slide to
     determine whether it was a formula, a diagram legend, or slide artefact.
     If it was a formula for synonym overlap or homonym disambiguation, restore
     it; otherwise remove. -->

[MATH_ERROR: origin unclear. Reconstruct from PPTX or remove.]

## Hypernyms and hyponyms

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, hypernyms/hyponyms block -->

Another common word relationship is between **hypernyms** and **hyponyms**. A hypernym has a broader, more general meaning and is often seen as the higher-level category among words. In contrast, a hyponym has a narrower, more specific meaning and is typically viewed as the lower-level category. For instance, "animal" is a hypernym related to the hyponyms "cat" and "dog", which represent more specific types of animals. Words can be hypernyms and hyponyms at the same time: a "mammal" is a hypernym for "cat" but a hyponym for "animal".

There are three ways to make use of these relationships in retrieval:

- **Faceted search** lets users explore search results by expanding or narrowing categories using hypernym/hyponym relationships. In an image search for "animals", users can drill down to more specific types like "cats" and "dogs". Conversely, if a query is too specific and yields few results, users can quickly broaden the search using presented hypernym hierarchies.

- **Query expansion** automatically expands the user's query with hypernyms and hyponyms to broaden the search. We can assign weights to the original term, its hypernyms (with less weight), and its hyponyms (with the same or less weight) to incorporate term relationships into the search process.

- **Relevance ranking** considers hypernym/hyponym relationships to evaluate document relevance, even when the query term is absent. This is similar to query expansion, but the distinction lies in where the expansion occurs. In query expansion, we submit a longer query with weighted hypernyms and hyponyms. In relevance ranking, we retain the user's original query and adjust scoring functions to account for hypernyms and hyponyms.

The WordNet website offers an online demo for in-depth exploration of synonyms, homonyms, hypernyms, and hyponyms in English. WordNet data is accessible in Python via `nltk.corpus.wordnet.synsets(word)`, which returns synsets providing functions to access synonyms, homonyms, hypernyms, meronyms, and various other relationships. See <http://wordnetweb.princeton.edu/perl/webwn> for the WordNet online demo.

## Part-of-speech tagging

<!-- source: 4_4_part_of_speech.md, opening blocks. Full HMM math and taggers
     tour is dropped per plan; keep only the retrieval-relevant overview. -->

Sentences consist of words belonging to various grammatical classes, known as **part-of-speech (POS) categories**, which share similar grammatical characteristics. In English, common parts of speech include noun, verb, adjective, adverb, pronoun, conjunction, interjection, numeral, article, and determiner. POS tagging assigns the appropriate class to a word based on its position and function within a sentence. Note that the mapping from a word to its POS tag is not always deterministic; for example, the word "run" can function as both a noun and a verb.

In information retrieval, POS tags support three practical uses:

- **Selective stop-word filtering**: keep "IT" if it is a noun (information technology, the pronoun-of-a-title case), remove it if it is a pronoun.
- **Stemmer disambiguation**: the lemma of "run" depends on whether it is a noun or a verb; passing the POS to the lemmatizer improves the result.
- **Question-form analysis**: in query processing we analyze structure, especially in questions. This helps extract details and query directly against structured metadata instead of relying solely on keyword-based search. The question "Who is Albert Einstein?" splits into a query word "who", a verb "is", and a name "Albert Einstein". Using POS tagging we can infer that the user seeks a person named "Albert Einstein" and query a "people" database rather than doing a full-text search.

A **treebank** is a text corpus where sentences are parsed and annotated to depict their syntactic structure. These trees also convey information about word-to-word grammatical relationships and hierarchical sentence composition. Treebanks offer labeled data to aid algorithms in learning grammatical structures and associating POS tags with words in sentences.

<!-- insert {figure} images/figure_3_3.png as fig-constituency-parse-tree here to
     illustrate what a POS-tagged parse tree looks like -->

### How POS taggers work (brief)

<!-- source: 4_4_part_of_speech.md, condensed per plan. Full HMM math is dropped;
     transformation-based derivation is dropped; deep learning gets one sentence. -->

Three families of taggers have been used historically:

- **Rule-based**: predefined rules based on suffixes and context assign POS tags to words. Simple and fast, but each language requires a fresh set of rules.
- **Statistical (HMM)**: a Hidden Markov Model where the hidden states are POS tags, the observations are terms, and transition and emission probabilities are learned from a tagged corpus. Decoded with the Viterbi algorithm.
- **Neural (transformer-based)**: modern taggers use a transformer that maps a token sequence to a POS-tag sequence, trained on POS-tagged corpora.

All three families achieve high accuracy on English (often above 99%). Neural taggers dominate today because they generalize better across genres and are easier to extend to new languages given enough tagged data.

```{seealso}
The Hidden Markov Model formulation used by statistical POS taggers is covered
in [](#appendix-hidden-markov-models). This chapter does not repeat the
transition, emission, or Viterbi derivations.
```

<!-- Note: figure_3_8.png (HMM transition/emission tables) belongs in the
     appendix rather than here. See _figures.md "Reserved / unused". -->

### POS tagging in Python

<!-- source: 4_4_part_of_speech.md, code blocks, converted from bullet-list format -->

```python
import nltk

tokens        = nltk.word_tokenize(text_en)
tagged_tokens = nltk.pos_tag(tokens, tagset=None)         # or tagset='universal'
ner_chunks    = [c for c in nltk.ne_chunk(tagged_tokens) if hasattr(c, 'label')]
```

`nltk` supports different tag sets. The standard tag set is more detailed; the universal tag set focuses on a few main categories. The standard set is often used for deep NLP tasks to construct parse trees which allow the extraction of context and the transformation of sentences.

WH-words are: where, what, which, when, ...

With NLTK, use `nltk.help.upenn_tagset()` to inspect the tag set. Proper nouns are specific people, places, and things.

```python
import spacy

nlp_spacy      = spacy.load('en_core_web_sm')
tokens         = nlp_spacy(text)
tagged_tokens  = [(t.text, t.pos_)   for t in tokens]
ner_entities   = [(e.text, e.label_) for e in tokens.ents]
```

spaCy uses a neural network to predict POS and NER tags, however with different tag names than `nltk`. spaCy also offers support for various languages; refer to their documentation to choose the suitable model.

The `transformers` library offers two pipelines for extracting POS and NER tags using trained neural networks. It also supports fine-tuning of NER models to adapt to specific document collections and scenarios. Since transformer models are continuously advancing, we present the general code structure and recommend visiting the Hugging Face website for the latest models:

```python
from transformers import pipeline

nlp_bert      = pipeline("token-classification",
                         model="vblagoje/bert-english-uncased-finetuned-pos",
                         aggregation_strategy="max")
tokens        = nlp_bert(text)
tagged_tokens = [(token['word'], token['entity_group']) for token in tokens]

nlp_ner       = pipeline("ner",
                         model="dslim/bert-base-NER",
                         aggregation_strategy="max")
ner_tokens    = nlp_ner(text_ner)
ner_entities  = [(t['word'], t['entity_group']) for t in ner_tokens]
```

The aggregation strategy combines sub-word tokens to reconstruct words or n-grams, especially for names. Without an aggregation strategy the model assigns entity values to individual model tokens, potentially splitting words into smaller tokens without grouping them into entities. Consult the model description to determine the specific POS and NER tags used, as these vary between different models.

## Named Entity Recognition

<!-- source: 4_4_part_of_speech.md, NER block -->

A closely related task is **Named Entity Recognition (NER)**, illustrated by "Albert Einstein" in the earlier example. Typical NER categories include names, locations, organizations, and currencies. NER terms are typically not found in dictionaries, and their frequencies and occurrences vary over time. Identifying a term (or n-gram) as an NER helps us infer the user's intent. In some scenarios NER searches are common, such as online shops, where proper indexing is crucial.

Instead of learning valuable bi-grams to add to the dictionary, we can generate them using simple rules based on NER tags. For instance, if we encounter two names (two consecutive NER-people tags), we can index them both individually and as a bi-gram (or tri-gram if three consecutive names appear). POS tags provide additional insights into the roles of names, as seen in sentences like "How to drive from Basel to Luzern", "I am in Basel and want to drive to Luzern", or "How to drive to Luzern from Basel".

NER tags enable the extraction of valuable contextual information, such as person names, locations, or product names. When applied to a query in question form, this helps understand the user's intent and optimize search results:

- **Person**: "Who is Albert Einstein?" -> prioritize web pages like Wikipedia, IMDb, Musicbrainz, and sports sites that users commonly visit to gather information about prominent individuals.
- **Time/date**: "Who won the F1 race last weekend?" -> enhance the visibility of news articles or use the extracted date to conduct a temporal range query.
- **Location**: "What to do in Basel?" -> prioritize regional content and provide a map of the named location to assist users with navigation.
- **Product brand**: "Where is the latest iPhone available?" -> boost advertisements and shopping sites, run a product search to offer a best-price view, or provide recommendations.

## Chunking with rule-based grammars

<!-- source: 4_4_part_of_speech.md, chunking block -->

**Chunking** creates non-overlapping phrases using a defined grammar. For example, the grammar `NP: {<DT>?<JJ>*<NN>}` combines articles, adjectives, and nouns into a single group, which supports understanding of term relationships for more effective searching. For instance, "a red car" forms a parse tree that links the adjective "red" with the noun "car". More intricate grammars enable the dissection of sentences into smaller components, allowing reasoning about context and sentence meaning through additional dependency information between terms.

A good online demo with deep NLP capabilities is available at <https://corenlp.run>.

To analyze sentence structure, we require a grammar similar to that used in programming languages. Unlike programming languages, natural-language grammar is imperfect and riddled with ambiguities, making it challenging for both humans and machines to grasp context. Grammar alone cannot resolve these ambiguities; context plays a crucial role.

<!-- insert {figure} images/figure_3_4.png as fig-dependency-parse-tree-loves
     here to illustrate a dependency parse -->

### Example (raw source, to be reformatted as a table)

<!-- source: 4_4_part_of_speech.md, closing example paragraph -->

- **English POS example**: the source has a table (blue) showing POS tags for the methods discussed on an English sentence, with punctuation and repeating words removed.
- **NER example**: "Jack Higgins, wearing Nike shoes, deposits £50,000 with BestBank in London at Jermyn Street close to Piccadilly Circus." Expected NER tags include person (Jack Higgins), product (Nike shoes), currency amount (£50,000), organization (BestBank), city (London), street (Jermyn Street), landmark (Piccadilly Circus).
- **POS frequency in a novel**: the source has a table (orange) listing POS tags and their frequency in an English novel.

<!-- TODO (rewrite): regenerate the three tables from a demo notebook so the
     numbers match a specific corpus, then paste them here as proper markdown
     tables. -->

## Spell correction and "did you mean?"

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, closing spelling block -->

The last part of query understanding considers spelling mistakes and how to treat them. Typically we employ a spellchecker to replace words not found in the dictionary.

- **During indexing**, we retain the original misspelled version in the document representation and add the auto-corrected version(s) to the index.
- **During query time**, we can expand the query with auto-corrected versions or suggest alternative queries if the misspelled query yields insufficient results ("did you mean?").

Spelling mistakes, especially in names, are common but can be challenging to differentiate from intentional variations. The name "Britney" has various alternative forms such as "Britni", "Brittney", "Britnee", "Britneigh", "Britnie", and many more. If we only used auto-corrected versions we may not find these alternative forms.
