---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Foundations
chapter: Advanced Text Processing
section: Stemming and Lemmatization
order: "3.2"
---

# Stemming and Lemmatization

<!--
TODO (rewrite): re-order as:
  1. Stop words (new proper subsection)
  2. Rule-based stemming: Porter (with full mechanics from _porter_details_from_ch01.md)
  3. Rule-based stemming: Lancaster (warning admonition on over-stemming)
  4. Rule-based stemming: Snowball (multi-lingual)
  5. Dictionary-based lemmatization: WordNet, spaCy (with the suffix-rule table and exception lists)
  6. Multi-lingual comparison (German, French tables)

Insert a Model Comparison table at the end (moves to 6_summary.md when done).
-->

## Stop words

<!--
TODO (rewrite, new subsection): write from scratch based on the plan agreed with
Roger.

Content to cover:
- What are stop words: the ~100-200 most frequent function words in a language
  (English: "the", "of", "and", "to", ...)
- Pros of removing them: smaller index, faster BM25 scoring, fewer noise matches
  in bag-of-words models.
- Cons of removing them: false negatives on titles and phrases where the stop
  word is content-bearing. Trigger examples:
    * Stephen King's "It" (novel and film) - the query "It" is filtered out
      entirely if we drop stop words.
    * Books about "IT" (information technology) - same problem.
    * "To be or not to be" - all stop words; the phrase disappears.
    * "The Who" (band name) - both tokens dropped.
- BM25 nuance: IDF already down-weights stop words, so aggressive removal is
  often unnecessary in probabilistic ranking. In vector-space TF-IDF, stop-word
  removal has a larger effect.
- Modern retrieval choice: keep stop words in the index (storage is cheap),
  but skip them in phrase-query and collocation extraction.
- Language-specific stop-word lists via nltk (`stopwords.words('english')`,
  'german', 'french', ...) and spaCy.
- Warning admonition: name searches and quoted phrases where stop-word removal
  destroys the query.
-->

## Stemming and lemmatization overview

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, opening block -->

Stemming reduces words in the input to their root, ensuring variants match during search. For example, "houses" in a document can match a query with "house". Stemming is language-dependent, but typically removing prefixes and suffixes is effective for most languages. Words that can undergo significant inflections, like "go" and "went", present more challenges. We distinguish different types of stemming algorithms:

- **Rule-based stemmers** use rules to transform words to their stems, which may not always be linguistically correct but are designed to match variants of the same root. In text retrieval, displaying these stems to users is not necessary; they are only used for quick lookup with inverted files.

- **Dictionary-based stemmers** use a small set of rules for regular inflections and rely on a dictionary and irregular-inflection list to find the correct linguistic stem. In text retrieval, this improves the success for matching word variants, especially in cases with strong inflections like "go" and "went".

## Rule-based stemmers: Porter, Lancaster, Snowball

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, rule-based stemmer block -->

In a previous chapter, we introduced the Porter algorithm, a basic English stemmer that creates pseudo-stems to unify word variations. The Lancaster stemmer is another rule-based stemmer for English. It aggressively cuts off word endings (suffixes), which can lead to very short stems. It is faster than other algorithms and suitable for general English text processing.

In various retrieval situations, handling diverse languages is common. Applying Porter or Lancaster stemmers to non-English text does not work. As a solution, Martin Porter introduced the Snowball framework to create rule-based stemmers for multiple languages similar to Porter and Lancaster. This framework features its own rule-definition language and can generate code for different programming languages. The result is still a pseudo-stem, and in languages with strong inflections the stem can vary due to gender, tense, or case changes. These algorithms are highly efficient and can operate in any environment without requiring a large dictionary.

### Porter Algorithm: full mechanics

<!-- source: _porter_details_from_ch01.md, integrated here per plan.
     The overview was already introduced in ch01 §2 (feature extraction pipeline).
     This section provides the full rule structure. -->

For English, the Porter Algorithm finds a near-stem of words. This stem is not linguistically correct but it often reduces words with the same linguistic stem to the same near-stem. The algorithm is highly efficient, and various extensions have been proposed over the years. Porter's original version from 1980 is built on the following notation:

- Porter defines `v` as a "vocal" (vowel) if:
    - it is an A, E, I, O, U
    - it is a Y and the preceding character is not a "vocal" (e.g. RY, BY)
- All other characters are consonants (`c`).
- Let `C` be a sequence of consonants and let `V` be a sequence of vocals.
- Each word follows the pattern: `[C](VC)m[V]`, where `m` is the *measure* of the word.

Additional conditions used by the rules:

- `*o`: stem ends with `cvc`; the second consonant must not be W, X, or Y (matches -WIL, -HOP)
- `*d`: stem with double consonant (-TT, -SS)
- `*v*`: stem contains a vocal

The rules establish mappings for words using the forms mentioned above. The variable `m` is used to prevent over-stemming of short words.

There are 5 main steps with several sub-steps within each. Each (sub-)step includes a list of ordered rules to match the endings of terms. Only the first rule that matches is applied, and the algorithm proceeds to the next (sub-)step. Most sub-steps have only a few rules (fewer than 10) and no more than 20 rules total.

For the complete set of rules, refer to Porter's original paper: **An Algorithm for Suffix Stripping**, *Program*, Vol. 14, No. 3, 1980.

## Dictionary-based lemmatization: WordNet and spaCy

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md -->

An improvement over the rule-based stemmers are dictionary-based stemmers such as those provided by WordNet and spaCy. They consist of three parts:

- a simple rule-based stemmer for regular inflections (e.g., "-ing", "-ed")
- an exception list for irregular inflections
- a dictionary of all possible stems of the language

The dictionary-based algorithm works as follows:

1. Retrieve a part-of-speech (POS) tag for the current word. This is typically done during tokenization and considers the broader context to determine the correct tag (e.g., noun, verb, adjective, punctuation). For example, whether "run" is a noun or a verb depends on its context.
2. Search for the word in the dictionary; if it is found, then the word is not inflected, and we return it as its own stem.
3. Search for the word in the exception list for its POS tag; if it is found, return the stem as given in the list.
4. Apply rules based on the POS tag to shorten regularly inflected forms using their suffixes. Use each applicable rule and check the dictionary; if the word is found, return the form from the dictionary.
5. If no dictionary entry is found, return the word as its own stem. This can occur with names, misspelled words, or loanwords (words from another language, like English words in German).

<!--
TODO (rewrite): the source has a suffix-rules table (NOUN/VERB/ADJ endings) and
three exception-list examples (adj.exc ~1500 entries, noun.exc ~2000, verb.exc
~2400). Convert them into proper markdown tables during the rewrite.
-->

### Suffix rules (raw source, to be converted to a table)

```
Type    Suffix    Ending
NOUN    s
NOUN    ses       s
NOUN    xes       x
NOUN    zes       z
NOUN    ches      ch
NOUN    shes      sh
NOUN    men       man
NOUN    ies       y

VERB    s
VERB    ies       y
VERB    es        e
VERB    es
VERB    ed        e
VERB    ed
VERB    ing       e
VERB    ing

ADJ     er
ADJ     est
ADJ     er        e
ADJ     est       e
```

### Exception-list samples (raw source, to be trimmed to representative rows)

```
adj.exc (1500 entries):
    stagiest      stagy
    stalkier      stalky
    stalkiest     stalky
    stapler       stapler
    starchier     starchy
    starchiest    starchy
    starrier      starry
    starriest     starry
    statelier     stately
    stateliest    stately

noun.exc (2000 entries):
    neuromata     neuroma
    neuroptera    neuropteron
    neuroses      neurosis
    nevi          nevus
    nibelungen    nibelung
    nidi          nidus
    nielli        niello
    nilgai        nilgai
    nimbi         nimbus
    nimbostrati   nimbostratus
    noctilucae    noctiluca

verb.exc (2400 entries):
    ate           eat
    atrophied     atrophy
    averred       aver
    averring      aver
    awoke         awake
    awoken        awake
    babied        baby
    baby-sat      baby-sit
    baby-sitting  baby-sit
    back-pedalled back-pedal
    back-pedalling back-pedal
    backbit       backbite
```

## Using the stemmers in Python

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, code block -->

`nltk` and `spaCy` provide multiple stemmers for text processing in different languages:

```python
import nltk
from nltk.corpus import wordnet
import spacy

# building the stemmers
porter    = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
snowball  = nltk.SnowballStemmer("english")
wordnet   = nltk.WordNetLemmatizer()
spacy_nlp = spacy.load('en_core_web_sm')

# applying them
porter.stem('discovered')
lancaster.stem('discovered')
snowball.stem('discovered')
wordnet.lemmatize('discovered', 'v')

# spaCy processes the full text sequence, not just one word
for token in spacy_nlp('I have discovered it'):
    print(token.text, token.lemma_)
```

## Comparing stemmers across languages

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, English/German/French comparison prose -->

### English

The Snowball and Porter algorithms yield very similar results as they mostly rely on the same rules; Snowball is a slightly revised version of Porter. Lancaster is more aggressive in removing suffixes, often resulting in overly short stems that may collide with unrelated words, especially when they are short themselves (e.g., "one" and "only" both reduced to "on"). WordNet and spaCy produce similar results to each other, but their stems differ from those of the rule-based algorithms. Notably, all WordNet and spaCy stems are linguistically correct. In text retrieval, stem correctness matters less than ensuring variants map to the same stem and thus the same token ID in the index. This is evident in examples like "had" and "have", which the rule-based algorithms map to different stems, while the dictionary-based algorithms map them to the same base form "have". This enhances the search engine's ability to match query variants with those found in documents.

### German

Snowball and spaCy are good options for German stemming, allowing us to compare Snowball's rule-based approach with spaCy's dictionary-based approach. We observe similar differences between rule-based (Snowball) and dictionary-based (spaCy) stemming as in English. Additionally, Snowball maps special characters to a base character set and converts text to lowercase. It also handles cases like "ae" -> "a" if the text does not use "ä" correctly. spaCy corrects casing only if a word starts a sentence and would normally be in lowercase. Snowball's results are acceptable for text retrieval, but spaCy performs significantly better in identifying the true linguistic stem and matching strongly inflected variants (consider "beschloß" and "beschließen").

### French

For French stemming, Snowball and spaCy are again good options. Unlike in German, French Snowball retains accented characters but still converts words to lowercase, while spaCy preserves casing for names. We observe similar differences between the rule-based (Snowball) and dictionary-based (spaCy) approaches as in the German example. The ability to map various inflected forms to the same stem is even more noticeable in French, as Snowball often assigns different stems to different inflected forms of the same root (e.g., "aperçu" and "aperçut", "avait" and "avaient").

<!--
TODO (rewrite): produce three real markdown tables, one per language, listing
the input word, its stem under each of Porter / Lancaster / Snowball / WordNet
/ spaCy, and its POS tag where relevant. The source PPTX contains the original
tables as figures; regenerate them as text.
-->
