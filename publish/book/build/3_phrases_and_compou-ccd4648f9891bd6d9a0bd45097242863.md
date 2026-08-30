---
author: Roger Weber
edition: HS26
status: updated
book_part: Foundations
chapter: Advanced Text Processing
section: Phrases and Compounds
order: "3.3"
---

(advanced-text-processing-phrases)=
# Phrases and Compounds

Tokenization decides what a token is; stemming and lemmatization decide which tokens are equivalent. Both operate at the level of one word. This section handles the two cases where a one-word view of meaning breaks down. Phrases pack multiple words into one concept: "New York" is not "New" plus "York", and a query for "New" retrieves the wrong documents. Compounds do the opposite: "Bücherregal" (bookcase) is one German token built from "Bücher" (books) and "Regal" (shelf). A shopper on an e-commerce site who searches for "Regal" will not see the bookcase listings until we split the compound into its parts. Both cases need explicit handling before the retrieval model sees the tokens.

## Bi-grams, tri-grams, phrases

The naive bag-of-words model treats a document as a multiset of independent tokens. "New York City", "Salt Lake City", and "prime minister" become bags of three or two unrelated tokens. Searching for "New York" as two separate terms still retrieves every document that mentions the city, so recall is not the problem. Precision is: the query also matches documents where "new" and "York" occur unrelated ("a new book from York University"), and a plain bag-of-words index has no way to tell those apart from documents where the two words sit side by side. Without a proximity or phrase mechanism, the genuine "New York" documents cannot be ranked above the incidental co-occurrences. Indexing "New York" as a single token `new_york` restores precision and gives the ranker a clean signal.

This generalises to n-grams: a bi-gram joins two adjacent words, a tri-gram three, and so on. Indexing the bi-grams and tri-grams of "the city of New York City" produces

```
the, city, of, new_york_city, new_york, york_city, city
```

with `_` marking a phrase token. We keep the original single-word tokens and emit every overlapping n-gram rather than only the longest one. This is deliberate: a query for "New York" must still match even though this document only ever contains the longer "New York City", and a query for the single word "city" must still match as well. Keeping all of them costs some index space but avoids missing any of these queries. A phrase query for "New York" now hits `new_york` directly, without proximity constraints. The question that follows is which n-grams are worth indexing. Adding every bi-gram in the corpus is prohibitive (it turns a 100k-token vocabulary into a 10-billion-pair vocabulary) and most of them are noise: "he is", "of the", "and to". We need a scoring function that selects the small subset of bi-grams that mean something the individual tokens do not.

## Naive frequency and the stop-word problem

The obvious approach is to count how often each bi-gram appears in the corpus and keep the most frequent ones. On any English text this immediately breaks:

```
of the      12,847
in the       9,203
to the       7,451
and the      6,988
for the      5,102
...
```

The top of the list is dominated by pairs of stop words. Filtering these out helps but does not solve the problem: even after stripping stop-word pairs, high-frequency co-occurrences of common words ("said Holmes", "young man", "could see") still bubble to the top and squeeze out the phrases we actually want. Frequency alone cannot tell us whether two words appear together because they mean something together or because they are both common.

Two better scoring functions capture that distinction: pointwise mutual information (PMI) and the likelihood ratio (LHR).

## Pointwise mutual information

The intuition behind **pointwise mutual information** is straightforward. Two words that mean something together should co-occur more often than their individual frequencies would suggest under independence. In a Sherlock Holmes novel of $N$ tokens, "said" appears 207 times and "Holmes" appears 94 times. Under independence, the expected number of "said Holmes" bi-grams is $207 \cdot 94 / N$, which for a typical novel comes out to about 2 or 3 occurrences. The bi-gram actually occurs 12 times. That is more than expected but only mildly so: "said" is paired with dozens of other names throughout the book.

Compare with "Sherlock Holmes". "Sherlock" appears 51 times, of which 48 are followed by "Holmes". Almost every occurrence of "Sherlock" is part of the phrase. Under independence, the expected co-occurrence count would be $51 \cdot 94 / N \approx 0.5$; the observed count of 48 is nearly a hundred times higher.

```{admonition} Key Formula: Pointwise Mutual Information
:class: important

$$\text{pmi}(t_1, t_2) = \log_2 \frac{p(t_1, t_2)}{p(t_1) \cdot p(t_2)} = \log_2 \frac{N \cdot \text{tf}(t_1, t_2)}{\text{tf}(t_1) \cdot \text{tf}(t_2)}$$

PMI is the log-ratio of the observed joint probability to the joint probability that independence would predict. It is high when two tokens almost always appear together and rarely apart.
```

Because $\log_2(N)$ is the same constant for every bi-gram in the corpus, ranking by PMI is equivalent to ranking by $\log_2 \text{tf}(t_1, t_2) - \log_2 \text{tf}(t_1) - \log_2 \text{tf}(t_2)$. A minimum-frequency filter is still essential: the formula gives its highest values to bi-grams where all three counts equal 1 (a rare word next to another rare word), which is not what we want.

```{admonition} Example: PMI on the Sherlock Holmes corpus
:class: example

Ranked by PMI with a minimum bi-gram frequency of 3, the top phrases from the Sherlock Holmes corpus (roughly 43,000 tokens) are:

| Bi-gram | tf($t_1$, $t_2$) | tf($t_1$) | tf($t_2$) | PMI |
|---|---|---|---|---|
| Baker Street | 15 | 15 | 20 | 12.20 |
| Scotland Yard | 8 | 8 | 12 | 12.13 |
| Sherlock Holmes | 48 | 51 | 94 | 9.75 |
| my dear | 34 | 350 | 45 | 6.06 |
| said Holmes | 12 | 207 | 94 | 3.13 |

"Baker Street" and "Scotland Yard" score highest because both components are rare and both components almost always appear as part of the phrase. "Sherlock Holmes" is next: "Sherlock" is not rare, but 48 of 51 occurrences pair it with "Holmes". "Said Holmes" is a real bi-gram in a Holmes novel but not a distinctive phrase: "said" pairs with hundreds of other words, so its PMI is low.
```

PMI has one systematic bias: it prefers bi-grams built from rare words. Because $\log_2 N$ dominates the score, a bi-gram made of two words that each appear once and always together achieves the maximum possible PMI. This is why the minimum-frequency filter is not optional.

## Likelihood ratio

The likelihood ratio test attacks the same problem from a hypothesis-testing angle rather than an information-theoretic one. It asks: how much better is the "these words are dependent" hypothesis than the "these words are independent" hypothesis, given the observed counts?

Let $\text{tf}_1 = \text{tf}(t_1)$, $\text{tf}_2 = \text{tf}(t_2)$, and $\text{tf}_{12} = \text{tf}(t_1, t_2)$. Two hypotheses:

- $H_1$ (independence): the probability that $t_2$ follows any word is a single value $p$, whether the previous word is $t_1$ or not. The maximum-likelihood estimate is $p = \text{tf}_2 / N$.
- $H_2$ (dependence): the probability that $t_2$ follows $t_1$ is $p_1 = \text{tf}_{12} / \text{tf}_1$, and the probability that $t_2$ follows any word other than $t_1$ is $p_2 = (\text{tf}_2 - \text{tf}_{12}) / (N - \text{tf}_1)$, with $p_1 \neq p_2$.

Assuming a binomial distribution for the "next word is $t_2$" events, the likelihood of each hypothesis is:

$$L(H_1) = b(\text{tf}_{12}; \text{tf}_1, p) \cdot b(\text{tf}_2 - \text{tf}_{12}; N - \text{tf}_1, p)$$

$$L(H_2) = b(\text{tf}_{12}; \text{tf}_1, p_1) \cdot b(\text{tf}_2 - \text{tf}_{12}; N - \text{tf}_1, p_2)$$

with $b(k; n, x) = \binom{n}{k} x^k (1-x)^{n-k}$.

```{admonition} Key Formula: Log Likelihood Ratio
:class: important

$$\log \lambda = \log \frac{L(H_1)}{L(H_2)}$$

The log-ratio of the likelihood under independence to the likelihood under dependence. Values near zero mean the two hypotheses fit the data equally well; strongly negative values mean dependence fits much better and the bi-gram is genuine. Bi-grams are usually ranked by $-2 \log \lambda$, which under $H_1$ follows a chi-squared distribution with one degree of freedom.
```

LHR is more robust than PMI on sparse data because it directly compares two probabilistic models rather than comparing observed to expected counts. It does not have PMI's rare-word bias, so infrequent bi-grams score less aggressively. The trade-off is that LHR does include some frequent stop-word patterns, because those patterns really are non-independent even if they are not interesting phrases.

```{admonition} Example: LHR on the Sherlock Holmes corpus
:class: example

Ranked by $-2 \log \lambda$ on the same corpus, with no minimum-frequency filter:

| Bi-gram | tf($t_1$, $t_2$) | tf($t_1$) | tf($t_2$) | $-2 \log \lambda$ |
|---|---|---|---|---|
| I am | 39 | 890 | 41 | 452 |
| Sherlock Holmes | 48 | 51 | 94 | 385 |
| said Holmes | 12 | 207 | 94 | 68 |
| the door | 40 | 3,205 | 91 | 61 |
| my dear | 34 | 350 | 45 | 59 |

"I am" tops the list not because it is a useful phrase but because 39 of 41 occurrences of "am" follow "I": the dependence is extreme. To surface real phrases, filter out bi-grams containing any stop word before ranking. After that filter, "Sherlock Holmes" moves to the top, followed by other named entities and characteristic collocations.
```

The two measures rank the same corpus differently. PMI favours bi-grams where the components are rare and locked together. LHR favours bi-grams where the components co-occur far more often than chance predicts, whether or not the components are rare. In practice, both are useful. Many systems compute both and take the union of their top-$k$ lists.

## Choosing a threshold and indexing strategy

Once a scoring function is chosen, the retrieval engineer picks a threshold and adds every bi-gram above it to the vocabulary. The threshold is not a precise cut-off; it trades index size against phrase coverage. Missing a bi-gram is not fatal because proximity search (see the classical text retrieval chapter) can find it approximately.

The other choice is whether to index the individual tokens **in addition to** the phrase. Indexing both means a query for "Holmes" still matches documents containing "Sherlock Holmes", at the cost of a small storage overhead. In practice this is almost always the right choice: users query at whichever granularity they choose, and the index should support both.

The same construction extends to tri-grams and quad-grams. Beyond quad-grams, sparsity becomes a problem: any four-word sequence is rare enough that scoring functions lose statistical power. Most production systems stop at bi-grams and rely on proximity search for longer patterns.

```python
from nltk.collocations import (
    BigramCollocationFinder, BigramAssocMeasures,
)
from nltk.corpus import stopwords

# Build the finder from a token stream
finder = BigramCollocationFinder.from_words(tokens)

# Require at least 3 co-occurrences
finder.apply_freq_filter(3)

# Drop bi-grams with a stop word or a very short token
ignored = set(stopwords.words('english'))
finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored)

# Rank by PMI (or likelihood_ratio, or raw_freq)
for (t1, t2), score in finder.score_ngrams(BigramAssocMeasures.pmi)[:20]:
    print(f'{t1} {t2}: pmi={score:.2f}')
```

`nltk` also provides `TrigramCollocationFinder` and `QuadgramCollocationFinder` with matching `TrigramAssocMeasures` and `QuadgramAssocMeasures`.

## Compounds

A phrase glues two tokens together. A compound goes the other way: it hides several words inside a single token. German makes this productive: "Wolkenkratzer" (skyscraper) combines "Wolke" (cloud) and "Kratzer" (scratcher). "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz" (a Mecklenburg-Vorpommern law from 1999-2013 delegating the supervision of beef labelling) chains together seven concepts as one token. Finnish and Dutch have the same construction. Programming languages have their own version: `word_tokenize`, `assertEquals`, and `QueryParser` each pack a whole phrase into one identifier.

Compounds break retrieval in an asymmetric way. A user who types "Etikettierung Gesetz" cannot match a document indexed under the full compound. The reverse also fails: a query for "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz" only finds documents that spelled it exactly the same way, which is optimistic for a 63-letter word. The fix is to index both the compound and its parts, giving the retriever a chance to match either.

```{admonition} Endocentric vs exocentric compounds
:class: warning

Not every compound splits usefully. **Endocentric** compounds derive their meaning from their parts: "sunglasses" are a kind of glasses, "Abfalleimer" is a kind of Eimer (bucket), and splitting into parts is safe. **Exocentric** compounds do not: a "skyscraper" is not a kind of "scraper", and the German "Wolkenkratzer" is not a kind of "Kratzer" (scratcher). Splitting an exocentric compound into parts adds tokens with the wrong semantics to the document, hurting precision. Whether the recall gain outweighs the precision loss depends on the collection: news search benefits from splitting, legal search often does not.
```

## Splitting compounds

Compound splitting proceeds in two stages: generate candidate splits, then score them.

For **generating candidates**, we use language-specific rules. In English, hyphens and hyphenation syllables give the splits: "must-have" splits into "must" and "have", "skyscraper" into "sky", "scrap", "er". In German, syllable boundaries following German hyphenation rules produce candidates: "Wolkenkratzer" becomes ("wol", "ken", "krat", "zer"), and every way of grouping consecutive syllables into words yields a candidate split: (wolken, kratzer), (wolken, krat, zer), (wol, kenkratzer), and so on. German also uses binding letters like "s" between compound parts ("Schifffahrtskapitän" needs to be tested both as "Schifffahrt-Kapitän" and "Schifffahrts-Kapitän"), so the candidate generator must try both variants.

For **scoring candidates**, we discard splits whose parts are not in the language's vocabulary or dictionary, then rank the remaining splits by how frequent their parts are.

```{admonition} Key Formula: Compound Split Score
:class: important

$$\text{score}(S) = \frac{1}{|S|} \sum_{p_i \in S} \log \frac{\text{tf}(p_i)}{N}$$

The average log-frequency of the parts in split $S$, computed against a reference corpus of $N$ tokens. Higher scores indicate splits into common parts, which are more likely to be the intended decomposition.
```

The intuition: splitting "Wolkenkratzer" into ("Wolke", "Kratzer") produces two moderately common German words. Splitting into ("Wol", "Kenkratzer") produces one non-word and one very rare word, giving a much lower score. The average is used rather than the sum so that splits with different numbers of parts are comparable.

```{admonition} Example: Splitting Abfalleimer
:class: example

The candidate splits of "Abfalleimer" (garbage can) and their scores against a German reference corpus:

| Split | Parts | Sum of log-frequencies | Average |
|---|---|---|---|
| (Abfall, Eimer) | 2 common German nouns | high | winner |
| (Abfalle, Imer) | "Imer" not in dictionary | discarded | - |
| (Ab, falleimer) | "falleimer" not in dictionary | discarded | - |

The valid split (Abfall, Eimer) wins. Indexing "Abfalleimer" as its full form plus the two parts means queries for "Abfall", "Eimer", or "Abfalleimer" all match documents about garbage cans.
```

## Where this sits in modern retrieval

Phrase detection and compound splitting are both still active production techniques. Lucene ships two compound-word filters, `DictionaryCompoundWordTokenFilter` and `HyphenationCompoundWordTokenFilter`, and Elasticsearch's default recommendation for German, Dutch, and other compounding languages is the hyphenation decompounder. Solr exposes the same filters through the Redlink extension. Actively maintained German-language dictionaries such as `uschindler/german-decompounder` provide the compound-split vocabularies that these filters consume.

Dense retrieval learns the compound structure implicitly through sub-word tokenizers like BPE and WordPiece, but the learning is imperfect. A 2023 study (arXiv:2305.14214) showed that SentencePiece, the tokenizer most widely used with multilingual language models, still splits German compounds into semantically incoherent fragments a significant fraction of the time. Hybrid retrieval stacks that combine BM25 with dense embeddings therefore keep the explicit decompounder on the BM25 branch, letting the neural side learn what it can while the classical side handles the compounds the classical way.

Phrase detection has an analogous story. Bi-gram indexing appears in modern lexical retrievers wherever named entities matter: product-search stacks index brand-name bi-grams, patent search indexes technical compounds, and academic search indexes conference names. PMI and LHR remain the standard offline scoring functions for building those bi-gram vocabularies.

```{admonition} Hands-on: Phrases and Compounds
:class: hint

Extract PMI and LHR bi-grams from a text corpus, apply frequency and stop-word filters, and split a set of German compounds into their parts.

[Open notebook -&gt;](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch03-03-phrases-and-compounds.ipynb)

*Includes pre-run results; you can read through or download and experiment.*
```

The next section leaves the document-recall problem behind and turns to the query side of the two-way failure from the chapter opener. Given a natural-language query, how do we extract enough structure from it that the retriever can do more than bag-of-words matching?
