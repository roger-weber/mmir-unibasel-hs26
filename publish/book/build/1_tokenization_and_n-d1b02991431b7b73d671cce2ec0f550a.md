---
author: Roger Weber
edition: HS26
status: updated
book_part: Foundations
chapter: Advanced Text Processing
section: Tokenization and Normalization
order: "3.1"
---

(advanced-text-processing-tokenization)=
# Tokenization and Normalization

The chapter opener showed a bookshop returning nothing for "car repair" because "automobile repair" is stored under a different token. That failure begins at the very first stage of the retrieval pipeline: how raw text is split into tokens and normalized before any matching happens. This section walks through that stage, starting with the simple regex from [](#classical-text-retrieval) and ending with a rule-based language detector that decides which downstream pipeline to invoke.

## The naive tokenizer and where it breaks

Chapter 1 introduced a one-line tokenizer:

```python
def word_tokenize(text: str) -> list[str]:
    text = re.sub(r'[^\w\-]+', ' ', text)
    return [t for t in text.split(' ') if t]
```

It substitutes any non-word character with a space, then splits on whitespace. It works on well-behaved English prose. Give it a sentence that exercises real-world text and it fails in several distinct ways at once:

> I buy my parents' 10% of U.K. startup for $1.4 billion. Dr. Watson's cat called Mrs. Hersley and it was w.r.o.n.g., more to come ...

Running the naive tokenizer produces:

```
['I', 'buy', 'my', 'parents', '10', 'of', 'U', 'K', 'startup',
 'for', '1', '4', 'billion', 'Dr', 'Watson', 's', 'cat', 'called',
 'Mrs', 'Hersley', 'and', 'it', 'was', 'w', 'r', 'o', 'n', 'g',
 'more', 'to', 'come']
```

Four systematic problems appear:

- **Possessives are handled inconsistently by shape**. `parents'` (trailing apostrophe only) becomes just `parents`: the apostrophe and the following space collapse into one space and the apostrophe leaves no trace. `Watson's` (apostrophe followed by `s`) becomes two tokens, `Watson` and a stray `s`, because the apostrophe becomes a space that splits the two letters apart. For retrieval, folding "Watson's" into "Watson" is often desirable (users search for the name, not the possessive), so the stray `s` is dropped in a subsequent cleanup step and both possessive forms end up equivalent. For syntactic analysis this is harmful, because the link between "Watson" and "cat" is broken and the shape-dependent behaviour is easy to overlook.
- **Numbers, currencies, and percentages fall apart**. `10%` becomes `10`, `$1.4 billion` becomes `1`, `4`, `billion`. Small integers survive; anything realistic does not.
- **Abbreviations disintegrate**. `U.K.`, `Dr.`, `Mrs.`, and the artificial `w.r.o.n.g.` all break at their internal periods. Searching for `U.K.` is impossible against this index.
- **Punctuation vanishes**. Fine for retrieval, but downstream sentence analysis loses its cues.

For a retrieval-only pipeline over English, this behaviour is not fatal: users rarely search for "U.K." with the period, and possessive stripping helps recall. But the distinctions are gone for good, and any task that needs sentence structure or numeric precision needs a better tokenizer up front.

## Modern tokenization with nltk and spaCy

Both `nltk` and `spaCy` ship word tokenizers built from rules combined with abbreviation and exception lists learned from text corpora. Running each on the demonstration sentence:

```python
import nltk
import spacy

sentence = ("I buy my parents' 10% of U.K. startup for $1.4 billion. "
            "Dr. Watson's cat called Mrs. Hersley and it was w.r.o.n.g., more to come ...")

nltk.word_tokenize(sentence)
# ['I', 'buy', 'my', 'parents', "'", '10', '%', 'of', 'U.K.', 'startup', 'for',
#  '$', '1.4', 'billion', '.', 'Dr.', 'Watson', "'s", 'cat', 'called', 'Mrs.',
#  'Hersley', 'and', 'it', 'was', 'w.r.o.n.g.', ',', 'more', 'to', 'come', '...']

[t.text for t in spacy.load('en_core_web_sm')(sentence)]
# ['I', 'buy', 'my', 'parents', "'", '10', '%', 'of', 'U.K.', 'startup', 'for',
#  '$', '1.4', 'billion', '.', 'Dr.', 'Watson', "'s", 'cat', 'called', 'Mrs.',
#  'Hersley', 'and', 'it', 'was', 'w.r.o.n.g', '.', ',', 'more', 'to', 'come', '...']
```

The two outputs agree on every token except one: nltk treats `w.r.o.n.g.` as a single abbreviation, spaCy splits the trailing period as sentence-terminating punctuation. On real text the two libraries usually agree; artificial or novel abbreviations are where they diverge.

Compared with the naive regex, the trained tokenizers fix several of the earlier failures but not all of them, and the two possessive shapes still behave differently:

- **Possessives**. `parents'` produces `parents` and a bare `'`. `Watson's` produces `Watson` and `'s`. Both tokenizers keep the apostrophe visible; only the `'s` form carries a marker that distinguishes it from any other apostrophe. Retrieval pipelines usually drop both apostrophe tokens after this step; syntactic pipelines keep them because they carry grammatical information.
- **Numbers**. Decimals stay together (`1.4`), which fixes the naive regex's worst behaviour. But the currency and percent symbols split off as their own tokens: `$1.4` becomes `['$', '1.4']` and `10%` becomes `['10', '%']`. A downstream step that wants "$1.4 billion" as one money entity needs to recombine, which is the job of the named entity recognizer covered later in the chapter.
- **Abbreviations**. `U.K.`, `Dr.`, and `Mrs.` are kept whole with their trailing periods, because both libraries carry curated abbreviation lists. `w.r.o.n.g.` is the divergent case: nltk keeps it whole, spaCy splits its final period. Real abbreviations rarely differ between the two libraries.
- **Punctuation**. `.`, `,`, and `...` are preserved as their own tokens. Retrieval pipelines filter them out; syntactic pipelines keep them.

For retrieval, we still want a leaner token list than the tokenizer alone provides. The standard cleanup after tokenization is:

1. Drop single-letter tokens and pure punctuation tokens.
2. Drop numbers and currency tokens unless the collection is numeric (a scientific-paper index would keep them).
3. Drop possessive tokens (`"'s"`) once the possessive information has been used.

The chapter's demo notebook composes these three cleanup steps on top of a regex tokenizer and compares the intermediate outputs side by side.

## Case, Unicode, and accents

A tokenizer separates words. Normalization decides which surface forms of the same word collapse to the same token. Consider three variants of one city name:

> `Zurich`, `Zürich`, `ZURICH`

They refer to the same city. To match a query for `Zurich` against a document that spells the city `Zürich`, the index has to normalize both spellings to a shared form before storing them. Three transforms handle almost all cases in European text:

- **Case folding** applies `text.lower()`, or the locale-aware equivalent for Turkish `İ`/`ı`. The three variants collapse to `zurich`, `zürich`, `zurich`.
- **Unicode normalization** applies `unicodedata.normalize('NFKC', text)` to canonicalise characters that have multiple valid encodings, such as the ligature `ﬁ` (one codepoint) unfolding to `fi` (two ASCII letters).
- **Accent folding** strips diacritics after normalization, mapping `ü -> u`, `é -> e`, `ß -> ss`. The library `unidecode` does this for arbitrary Unicode; `nltk`'s stemmers apply narrower per-language rules.

After all three, the three variants above become `zurich`. Users can type any spelling and match any document.

```{admonition} Accent folding is not free
:class: warning

Aggressive accent folding trades recall for precision. In German, "schon" (already) and "schön" (beautiful) have different meanings; folding both to "schon" makes them collide in the index. In French, "où" (where) and "ou" (or) become the same token. Whether to fold depends on the retrieval scenario: web search over cross-language content typically folds, while a legal-document index over a single language does not.
```

In practice, modern text-processing stacks apply Unicode normalization by default because it has no downside, case-fold when the retrieval scenario is case-insensitive (most web search), and fold accents only when the user population cannot easily type them.

## Sentence segmentation

Some tasks need whole sentences as input, not tokens. Part-of-speech taggers work sentence by sentence because word ambiguity resolves in context. RAG systems (see the retrieval-augmented generation chapter) build larger chunks by grouping consecutive sentences until they hit a size limit. Both need a reliable answer to the question: where does one sentence end and the next begin?

Splitting on `.` is the naive approach and fails on the same abbreviations we saw earlier. In "Dr. Watson's cat called Mrs. Hersley.", the periods after `Dr` and `Mrs` are not sentence boundaries; only the trailing period is. Multi-line quoted dialogue, ellipses, and decimal numbers add more edge cases.

Two libraries handle this well:

- **Punkt** (NLTK) is an unsupervised sentence segmenter that learns abbreviations and sentence-start patterns from a training corpus. NLTK ships pre-trained models for English and several other languages. It is fast and accurate on newspaper-style prose; it struggles on narrative dialogue where punctuation appears both inside and outside quotation marks.
- **spaCy** segments sentences as a side effect of its full linguistic pipeline. It is heavier than Punkt but handles complex sentence structures more consistently and supports many languages out of the box.

```python
from nltk.tokenize import sent_tokenize
sent_tokenize("Dr. Watson's cat called Mrs. Hersley. She was Egyptian.")
# ["Dr. Watson's cat called Mrs. Hersley.", "She was Egyptian."]
```

Sentence segmentation is one of the few pieces of a classical pipeline that stayed useful into the LLM era. Every RAG system built on top of embeddings still needs to decide where its chunks begin and end, and sentence boundaries are the most reliable low-cost signal for that decision.

## Word boundaries are not always spaces

Not every writing system uses spaces to separate words, and not every "word" in a modern document is meant to be a single token. Both cases break the "split on whitespace" assumption at the root.

**Scriptio continua** languages, including Chinese, Japanese, Thai, and classical Greek, do not use spaces at all. A single sentence is a continuous string of characters:

```
莎拉波娃现在居住在美国东南部的佛罗里达。

莎拉波娃    现在    居住   在    美国    东南部       的      佛罗里达
Sharapova  now     lives  in   US     southeast    in      Florida
```

**Code identifiers** pack multiple words into a single token by convention: `QueryParser`, `assertEquals`, `word_tokenize`. A code-search index that stores only whole identifiers cannot answer a query for `parser`.

**Spoken-language transcripts** come from a continuous phoneme stream. Speakers do not pause between words; the pauses that exist mark breath, hesitation, or emphasis, and rarely align with word boundaries. Even the pause at a sentence end is unreliable in fluent speech. The segmenter has to infer word boundaries from the phoneme sequence itself, aided by a pronunciation dictionary and a language model, and non-native pronunciation or unclear articulation shifts where those boundaries land.

There are two ways to break these continuous streams into useful tokens:

- **Dictionary lookup with a probabilistic tie-breaker**. The dictionary lists every candidate word starting at the current position, and a trained model (historically an HMM, today usually a neural network) picks the most likely segmentation. This works well when the vocabulary is closed and the domain is stable. It breaks on names, brand terms, and loanwords.
- **Sub-word tokens used directly for retrieval**. Instead of reconstructing words, index overlapping character n-grams. A common choice is trigrams with a `#` prefix marking word-start positions:

    ```
    This course teaches multimedia retrieval.
    #Thi his #cou our urs rse #tea eac ach che hes #mul ult lti tim ime med edi dia ...
    ```

    Encode queries the same way:

    ```
    Q = "teach multtimedia"
        #tea eac ach #mul ult ltt tti tim ime med edi dia
    ```

    Even with a different inflected form ("teaches" versus "teach") and a typo ("multtimedia"), 10 of the 12 query trigrams match trigrams from the document. A retrieval model that supports partial matching with a proximity bonus finds the passage without any explicit stemming or spell-check.

This sub-word idea reappears in an industrial form in Byte Pair Encoding (BPE) and WordPiece, both of which power large language models by training the token vocabulary on a corpus rather than fixing it to trigrams. Those methods are covered in the semantic-search chapter; here we note only that "match parts of words when whole-word matching fails" is not a workaround but the underlying idea of every modern sub-word tokenizer.

## Language detection

Everything above depends on knowing what language the input is in. A German document benefits from Snowball's German rules and the German stop-word list; running it through the English pipeline strips accents but leaves compounds and inflections untouched. A query like "Bücher von Goethe" from the chapter opener needs to be recognized as German before the retrieval system can restrict results to German-language books or apply German-specific normalization.

For long documents, a small number of hand-crafted rules is enough:

- **Alphabet diversity** looks at which scripts appear. Latin, Cyrillic, Greek, Arabic, Devanagari, Thai, and CJK are easy to tell apart even from a single character.
- **Character diversity** narrows down within a script. `ä`, `ö`, `ü` are strong signals for German (and Turkish, Estonian, Finnish); `ç` for French, Portuguese, or Turkish; `ñ` for Spanish; `ß` almost exclusively for German.
- **Stop-word counts** track how many of the top hundred function words of each candidate language appear. English "the", "of", "and"; German "der", "die", "und", "ist"; French "le", "la", "de", "et". Whichever language's stop-word list matches most terms wins.
- **Vocabulary counts** extend the stop-word idea to a longer list of frequent content words.

For document-side indexing this is usually enough: a paragraph or a page provides many opportunities for the rules to fire, and the winning language emerges quickly. Once the language is known, the pipeline branches: a German document goes to the German Snowball stemmer, an English document to the English WordNet lemmatizer, and so on.

Short queries break the rules, though. "Mein computer" mixes German with an English loanword; "pain" is a French word for bread and an English word for suffering. Neither has enough evidence for the rules to fire reliably. That regime needs a statistical classifier over character n-grams, developed with Naive Bayes in [](#advanced-text-processing-intent-routing).

## Where this sits in modern retrieval

Tokenization and normalization are the least glamorous part of the retrieval stack and still the largest source of quality issues in production. Modern lexical retrievers like Lucene, Solr, and Elasticsearch expose the whole pipeline as a chain of configurable analyzers: a tokenizer, a lowercase filter, a Unicode-normalization filter, a stop-word filter, a stemmer, sometimes a compound splitter. Dense retrieval sidesteps most of these choices by learning them implicitly in the embedding model, but hybrid stacks (BM25 combined with dense retrieval) still need the lexical side, and the lexical side still starts with these steps. Getting them wrong at ingestion time is expensive to fix later because the index has to be rebuilt.

```{admonition} Hands-on: Tokenization and Normalization
:class: hint

Run the naive regex tokenizer, the nltk word tokenizer, and the spaCy tokenizer on the "Dr. Watson" sentence. Normalize the results with case folding and accent stripping, and observe the effect on a small BM25 index.

[Open notebook -&gt;](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch03-01-tokenization.ipynb)

*Includes pre-run results; you can read through or download and experiment.*
```

The next section takes tokens as given and looks at the second half of the recall gap: collapsing "car", "cars", "carrying", and "carried" so a document mentioning any of them matches a query for any other.

```{admonition} From ELIZA to Transformers: a short history of NLP (optional reading)
:class: note dropdown

The tokenization techniques in this section descend from a much longer research programme that runs from 1950s symbolic AI through statistical corpus-based methods to today's neural language models. Not exam-relevant, but useful for placing the tools in context.

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
```
