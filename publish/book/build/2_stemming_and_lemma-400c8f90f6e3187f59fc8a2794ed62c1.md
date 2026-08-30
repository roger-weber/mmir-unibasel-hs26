---
author: Roger Weber
edition: HS26
status: updated
book_part: Foundations
chapter: Advanced Text Processing
section: Stemming and Lemmatization
order: "3.2"
---

(advanced-text-processing-stemming)=
# Stemming and Lemmatization

The bookshop catalog stores documents word by word. A user searches "car repair" and misses titles like "cars", "repairing cars", and "repaired vehicles" because each surface form is a distinct token in the index. Tokenization from the previous section fixed splitting; it did not fix the fact that one root word has many written shapes. This section closes that gap. First it decides which tokens are worth indexing at all, then it collapses inflected surface forms of the same word onto a common stem or lemma.

## Stop words: what to keep out

Half a dozen very frequent words dominate any English text. "The", "of", "and", "to", "is" together account for roughly a quarter of tokens in typical prose. Storing every occurrence of these in the index wastes space and inflates term frequencies without adding any content signal: two documents both use "the" hundreds of times, and that shared usage tells us nothing about their similarity. A **stop-word list** is the standard shortcut: a fixed set of high-frequency function words to discard before indexing and querying. English stop-word lists usually contain 100-200 words; a compact 63-word list looks like this:

```python
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', ...}

def remove_stopwords(tokens: list[str], stopwords: set = None) -> list[str]:
    sw = stopwords if stopwords is not None else STOPWORDS
    return [t for t in tokens if t not in sw]
```

Every major language has its own list. `nltk.corpus.stopwords.words('german')` returns 232 German function words; French, Spanish, Italian, and the other main European languages are provided by both `nltk` and `spaCy`.

Dropping stop words works well most of the time and fails visibly the rest of the time. The cost is a small class of queries where the stop word is content-bearing:

- A search for **Stephen King's "It"** finds nothing if the tokenizer drops "it" as a stop word. The novel's title is the stop word.
- A search for **"IT security"** loses "IT" (information technology) for the same reason. The remaining "security" retrieves every document about locks, guards, and stock exchanges.
- **"To be or not to be"** consists entirely of stop words. Every word disappears.
- **"The Who"** (the band) has both tokens on typical stop lists.

BM25 already handles high-frequency words gracefully through inverse document frequency: a term that appears in most documents receives a near-zero weight even if we index it. This makes aggressive stop-word removal less necessary than it was in earlier bag-of-words models. The modern default is to keep stop words in the index (storage is cheap) but skip them when generating phrase collocations, computing term-based similarities, or building compact features for a classifier.

```{admonition} Never strip stop words from a phrase query
:class: warning

Dropping stop words before phrase matching turns "The Who" into an empty query and "not guilty" into "guilty". If the retrieval scenario supports phrase or entity queries, the stop-word filter must be aware of the phrase context or must not run on those tokens at all.
```

## Stemming: rule-based reduction

Once stop words are out, we still have "car", "cars", "carrying", "carried", "carrier". They all point to the same concept, but a bag-of-words index sees five distinct tokens. **Stemming** applies a set of ordered rules that strip suffixes off inflected forms so that variants of one root end up as the same token.

The three widely-used English rule-based stemmers are Porter, Lancaster, and Snowball. All three take a token, apply an ordered sequence of suffix-stripping rules, and return a stem. The stem does not need to be a real English word; it only needs to be the same for all variants of the same lemma.

### The Porter algorithm

The Porter algorithm is the classical English stemmer, published by Martin Porter in 1980. Chapter 1 introduced it as a black box; here we look at the internal structure.

Porter defines a **vocal** (vowel) as A, E, I, O, U, or Y when the preceding character is not a vocal (so "Y" is a vowel in "rhythm" but a consonant in "yellow"). All other characters are consonants. Let `C` be a sequence of consonants and `V` a sequence of vocals. Every English word can be written in the form:

$$[C](VC)^{m}[V]$$

where the exponent $m$ is the **measure** of the word. `TREE` has $m = 0$ (one consonant, one vocal, no `VC` pair), `TROUBLE` has $m = 1$, and `TROUBLES` has $m = 2$. Porter uses $m$ to prevent over-stemming: rules that would strip "ATE" from "OPERATE" are gated behind $m \geq 1$ so they do not also strip the "ATE" from "MATE".

The algorithm applies five main steps in order, each with several sub-steps. Every sub-step lists a small number of ordered rules that match suffix patterns. The first rule that matches fires, and the algorithm moves on to the next sub-step. Step 1 strips plurals and past tenses. Step 2 collapses common derivational suffixes onto a shorter form (`-ational -> -ate`). Step 3 continues in the same direction (`-icate -> -ic`). Steps 4 and 5 remove residual suffixes and clean up trailing letters.

The full rule set covers around 60 patterns and stems on the order of 95% of English text correctly. Porter's original paper (Porter, 1980) lists every rule; the `nltk.PorterStemmer` implementation is faithful to that specification.

```python
import nltk
porter = nltk.PorterStemmer()

for w in ['car', 'cars', 'carrying', 'carried', 'carrier']:
    print(w, '->', porter.stem(w))

# car       -> car
# cars      -> car
# carrying  -> carri
# carried   -> carri
# carrier   -> carrier
```

Note that "carrying" and "carried" collapse to `carri` (not to `carry`), and "carrier" is left intact. The stems are not linguistically pretty, but they map three variants of the verb "carry" onto the same token, which is what the retrieval index needs.

### Lancaster and Snowball

**Lancaster** (Paice, 1990) is a more aggressive stemmer for English. It applies iterative rule-based suffix removal until no rule fires, producing shorter stems than Porter. Speed is its main advantage: on average it does fewer rule scans per word. Its main risk is collisions: unrelated words can reduce to the same short stem.

```python
lancaster = nltk.LancasterStemmer()
for w in ['one', 'only', 'organization', 'organize']:
    print(w, '->', lancaster.stem(w))

# one          -> on
# only         -> on
# organization -> org
# organize     -> org
```

"one" and "only" both stem to "on", so a document about "the only book on Rome" collides with one about "book number one". For a scenario that prizes recall the collisions are usually acceptable; for a name search or a precise query they are a problem.

```{admonition} Lancaster over-stems short words
:class: warning

Lancaster reduces short words to two- or three-letter stems that collide across unrelated meanings ("one" and "only" both become "on"). Use Lancaster when speed dominates and the collection is large enough that a few token collisions do not affect ranking. Prefer Porter or Snowball on short-document collections where each collision matters.
```

**Snowball** is Martin Porter's own multi-language framework, published later, that generalizes the Porter design. It supplies stemmers for around 20 European languages including English, German, French, Spanish, Italian, Dutch, and Russian. On English, Snowball is essentially a revised Porter with a handful of small corrections; on other languages, it is often the only rule-based option available.

```python
snowball_en = nltk.SnowballStemmer("english")
snowball_de = nltk.SnowballStemmer("german")

snowball_de.stem("Wagen"), snowball_de.stem("Wägen")   # ('wag', 'wag')
snowball_de.stem("Reparatur"), snowball_de.stem("reparieren")  # ('reparatur', 'repar')
```

The German stemmer folds umlauts (`Wägen -> Wagen -> wag`) and reduces "reparieren" to `repar`, which does not match `reparatur`. Rule-based stemmers approximate the linguistic stem but do not compute it: for German, the noun "Reparatur" and the verb "reparieren" share a root but have different stems under any pure suffix-stripping rule set.

## Lemmatization: dictionary-based reduction

Rule-based stemming is fast and needs no external data, but it produces linguistically incorrect stems and cannot handle strong inflections. English "go" and "went" share no letters at all, so no suffix rule can map them together. **Lemmatization** takes a different approach: look the word up in a dictionary and return its recorded base form.

The three components of a dictionary-based lemmatizer are:

1. A **rule set** for regular inflections (`-s`, `-ed`, `-ing`, plurals with `-ies`).
2. An **exception list** for irregular forms: `went -> go`, `ate -> eat`, `mice -> mouse`, `children -> child`.
3. A **dictionary** of all base forms in the language, used to check whether a candidate form is valid.

The algorithm proceeds in order:

1. Take the current token and its part-of-speech tag (verb, noun, adjective, ...). The POS tag is needed because "run" as a noun and "run" as a verb have the same base form here, but "meeting" as a noun (`meeting`) and "meeting" as a verb (`meet`) do not.
2. If the token is in the dictionary as-is, it is already a base form. Return it.
3. If the token is in the exception list for its POS, return the base form listed there.
4. Apply the regular-inflection rules for its POS, checking the dictionary after each rule. Return the first match.
5. If no rule produces a dictionary word, return the token unchanged. This handles names, misspellings, and loanwords.

`WordNetLemmatizer` in `nltk` and the built-in lemmatizer in `spaCy` are the two standard implementations. Their exception lists cover several thousand irregular forms each.

```python
wn = nltk.WordNetLemmatizer()
wn.lemmatize('carried', 'v')    # 'carry'
wn.lemmatize('carrier', 'n')    # 'carrier'
wn.lemmatize('mice', 'n')       # 'mouse'
wn.lemmatize('went', 'v')       # 'go'
wn.lemmatize('better', 'a')     # 'good'
```

Note the last case: "better" lemmatizes to "good" because the exception list encodes the adjective's comparative form. No rule-based stemmer can do this.

`spaCy` runs a full linguistic pipeline that assigns POS tags automatically, so the lemmatizer is applied without the caller having to pass a tag:

```python
nlp = spacy.load('en_core_web_sm')
for token in nlp("She carried the mice yesterday and went home"):
    print(token.text, '->', token.lemma_)

# She      -> she
# carried  -> carry
# the      -> the
# mice     -> mouse
# yesterday-> yesterday
# and      -> and
# went     -> go
# home     -> home
```

For retrieval, the practical difference between rule-based stemming and dictionary lemmatization is that a lemmatizer gives you the linguistic base form and a stemmer gives you an arbitrary reduction that just happens to be consistent within the class. Both work for matching a query to a document. Lemmatization is heavier because it needs POS tagging plus a dictionary lookup; stemming is a handful of string operations.

## Multi-lingual comparison

Every stemmer and lemmatizer described above assumes one language. Running an English stemmer on French text strips accents and collapses "les" to "l" without doing anything useful. The right choice depends on the language of the input, which is what the language detector from the previous section is for.

Snowball is the go-to rule-based option outside English because it covers 20 languages under a single framework. spaCy is the go-to lemmatizer for the same set. On the same French sentence, the two give different stems:

```
Input:      "Les chats ont aperçu les oiseaux"

Snowball:   ['le', 'chat', 'ont', 'apercu', 'le', 'oisea']
spaCy:      ['le', 'chat', 'avoir', 'apercevoir', 'le', 'oiseau']
```

Snowball preserves accents by default in French (it strips them in German), but it does not know that "ont" is a form of "avoir" or that "aperçu" is a past participle of "apercevoir". spaCy's lemmatizer handles both because it has the French verb tables.

In German, the difference is even sharper because German inflects both nouns (for case, gender, and number) and verbs (for person, tense, and mood). Snowball reduces "beschloss" and "beschließen" to different stems; spaCy maps both to "beschließen". For queries against German text, spaCy's dictionary lemmatizer noticeably improves recall on strongly-inflected content.

Neither approach handles compound splitting, which is the concern of the next section: "Wolkenkratzer" (skyscraper) stays as a single token under both stemmers.

## Where this sits in modern retrieval

Rule-based stemming and dictionary lemmatization are still the default text-processing configuration for lexical retrievers. Lucene, Solr, and Elasticsearch ship analyzers for every major language that combine tokenization, stop-word filtering, and stemming or lemmatization in the same order this section presented them. Stemming is the option chosen when performance matters or when no dictionary is available; lemmatization is preferred when accuracy on strongly-inflected languages or on strong-verb English matters more than throughput.

Dense retrieval learns these normalizations implicitly during embedding training. A well-trained embedding model puts "carried", "carrying", and "carrier" near each other in vector space without an explicit stemming step. This does not make the lexical stack obsolete: hybrid retrieval combines BM25 (which needs stemming or lemmatization to keep recall high) with dense retrieval (which does not), and the two together consistently outperform either alone. The classical stemmers of this section are still on the query path of every serious production search stack in 2025.

```{admonition} Hands-on: Stemming and Lemmatization
:class: hint

Compare Porter, Lancaster, Snowball, WordNet, and spaCy on English, German, and French text. See how each stemmer treats strong verbs, plurals, and compounds.

[Open notebook -&gt;](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch03-02-stemming.ipynb)

*Includes pre-run results; you can read through or download and experiment.*
```

The next section takes stemmed and lemmatized tokens as given and looks at what happens when meaning crosses word boundaries: "New York" is not "New" plus "York", and "Wolkenkratzer" is not "Wolke" plus "kratzer".
