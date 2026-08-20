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

with `_` marking a phrase token. We keep the original single-word tokens and emit every overlapping n-gram rather than only the longest one. This is deliberate: a query for "New York" must still match even though this document only ever contains the longer "New York City", and a query for the single word "city" must still match as well. Keeping all of them costs some index space but avoids missing any of these queries. A phrase query for "New York" now hits `new_york` directly, without proximity constraints. The question that follows is which n-grams are worth indexing. Adding every bi-gram in the corpus is prohibitive (it turns a 100k-token vocabulary into a 10-billion-pair vocabulary) and most of them are noise: "it is", "of the", "and to". We need a scoring function that selects the small subset of bi-grams that mean something the individual tokens do not.

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

The intuition behind **pointwise mutual information** is straightforward. Two words that mean something together should co-occur more often than their individual frequencies would suggest under independence. In *A Study in Scarlet*, a corpus of about 43,200 tokens, "said" appears 207 times and "Holmes" 98 times. If the two were independent, we would expect the bi-gram "said Holmes" to occur about $207 \cdot 98 / N \approx 0.5$ times. It actually occurs 12 times, roughly 25 times more than chance. That is a real signal, but a modest one: "said" precedes dozens of different names throughout the book.

Compare "Sherlock Holmes". "Sherlock" appears 52 times, and every one of those 52 occurrences is followed by "Holmes". Independence would predict about $52 \cdot 98 / N \approx 0.1$ co-occurrences; the observed count of 52 is more than 400 times higher. "Sherlock" is effectively bound to "Holmes".

```{admonition} Key Formula: Pointwise Mutual Information
:class: important

$$\text{pmi}(t_1, t_2) = \log_2 \frac{p(t_1, t_2)}{p(t_1) \cdot p(t_2)} = \log_2 \frac{N \cdot \text{tf}(t_1, t_2)}{\text{tf}(t_1) \cdot \text{tf}(t_2)}$$

Here $\text{tf}(t)$ is the number of times term $t$ occurs in the corpus, and $\text{tf}(t_1, t_2)$ is the co-occurrence count: how often the bi-gram appears, that is, how often $t_2$ immediately follows $t_1$. Dividing each count by the corpus size $N$ turns it into a probability: $p(t) = \text{tf}(t)/N$ for a single term and $p(t_1, t_2) = \text{tf}(t_1, t_2)/N$ for the pair. PMI is then the log-ratio of the observed joint probability to the joint probability that independence would predict; it is high when two tokens almost always appear together and rarely apart.
```

Two things help read the formula. First, $\log_2(N)$ is the same constant for every bi-gram, so ranking by PMI is the same as ranking by $\log_2 \text{tf}(t_1, t_2) - \log_2 \text{tf}(t_1) - \log_2 \text{tf}(t_2)$. Second, look at what maximizes the score. PMI is largest when all three counts equal 1: a word that occurs once, sitting next to another word that occurs once, giving $\text{pmi} = \log_2 N$. Even when the two words and their pair all occur $n$ times together and never apart, the score is $\log_2(N/n) = \log_2 N - \log_2 n$, which only falls as $n$ grows. So PMI rewards rarity, and this is the mirror image of the stop-word problem: instead of frequent function words dominating, two rare words that happen to land side by side by chance float to the top. In *A Study in Scarlet*, 203 different bi-grams reach the maximum score of $\log_2 N \approx 15.4$, every one of them a pair of words that each occur exactly once and only next to each other ("aqua tofana", "admired treated", "ambitious title"). All of them outrank "Sherlock Holmes". A single chance adjacency is no evidence of a real collocation; there is simply too little frequency to be confident the two words prefer each other. The cure is a minimum-frequency filter: require a bi-gram to occur at least, say, three times before scoring it at all, which drops the low-evidence rare pairs.

```{admonition} Example: PMI on the Sherlock Holmes corpus
:class: example

Tokens are lowercased and reduced to alphabetic-only words, as in the normalization step of the previous section. Ranked by PMI with a minimum bi-gram frequency of 3 and a stop-word filter, the top of the list for *A Study in Scarlet* (about 43,200 tokens, 115 bi-grams surviving the filter) looks like this:

| Rank | Bi-gram | tf($t_1$, $t_2$) | tf($t_1$) | tf($t_2$) | PMI |
|---|---|---|---|---|---|
| 1 | audley court | 3 | 3 | 4 | 13.40 |
| 2 | torquay terrace | 3 | 3 | 4 | 13.40 |
| 3 | avenging angels | 4 | 4 | 4 | 13.40 |
| 4 | lauriston gardens | 6 | 6 | 6 | 12.81 |
| 8 | salt lake | 10 | 11 | 10 | 11.94 |
| 11 | scotland yard | 6 | 8 | 9 | 11.81 |
| ... | ... | | | | |
| 45 | sherlock holmes | 52 | 52 | 98 | 8.78 |

Even after the frequency filter, PMI's rare-word bias persists. The top three are tied at 13.40 because each reduces to $\log_2(N/4)$: "audley court" and "torquay terrace" pair a word occurring 3 times with one occurring 4 times, always together, and "avenging angels" pairs two words occurring 4 times each. All are rare pairs whose parts appear almost only together. "Sherlock Holmes", the defining phrase of the book, ranks only 45th of 115: because "Holmes" occurs 98 times, the larger denominator drags its PMI down to 8.78. The filter removed the tf=1 noise but did not fix the underlying preference for rarity.
```

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

LHR is more robust than PMI on sparse data because it directly compares two probabilistic models rather than comparing observed to expected counts. It does not have PMI's rare-word bias, so infrequent bi-grams score less aggressively. The trade-off is that LHR ranks frequent grammatical pairs highly, because those pairs really are non-independent even though they are useless as phrases. The bi-gram "I am" is the clearest example: "am" occurs 41 times and 39 of them follow "I", so the dependence is overwhelming, yet "I am" is worthless as an index phrase. This is why LHR still needs a stop-word filter: without one, "of the", "to be", "had been", and "I am" crowd the top of the ranking alongside the genuine names.

```{admonition} Example: LHR on the Sherlock Holmes corpus
:class: example

Ranked by $-2 \log \lambda$ on the same corpus, with the same minimum-frequency and stop-word filter as the PMI table:

| Rank | Bi-gram | tf($t_1$, $t_2$) | tf($t_1$) | tf($t_2$) | $-2 \log \lambda$ |
|---|---|---|---|---|---|
| 1 | sherlock holmes | 52 | 52 | 98 | 668 |
| 2 | jefferson hope | 34 | 37 | 56 | 458 |
| 3 | john ferrier | 29 | 39 | 62 | 352 |
| 4 | brixton road | 13 | 15 | 27 | 188 |
| 5 | salt lake | 10 | 11 | 10 | 181 |
| 9 | scotland yard | 6 | 8 | 9 | 98 |
| 11 | baker street | 6 | 6 | 29 | 89 |

With the stop-word filter in place, LHR puts the main characters and places of the novel on top: "Sherlock Holmes", "Jefferson Hope", "John Ferrier", "Brixton Road". Without the filter, the grammatical pairs discussed above ("of the", "to be", "I am") would sit among them. The filter removes that whole class in one step.
```

## PMI or LHR: which to use

The two measures rank the same corpus differently because they measure different things. PMI measures the *strength* of association: how much more often two words appear together than chance predicts, regardless of how often that is. LHR measures the *evidence* for association: how confidently the data rules out independence, which grows with frequency. "Sherlock Holmes" makes the contrast concrete: it ranks 45th of 115 under PMI, held down because "Holmes" is common, but first under LHR, which rewards the sheer weight of 52 co-occurrences.

Because they optimise different quantities, the two fail in opposite directions, and a genuine phrase can score well on one and poorly on the other:

- **Rare but tight** phrases occur a handful of times, always as a unit. They score high on PMI and low on LHR. In our results these are PMI's top rows: "Lauriston Gardens" (the address of the murder), "Audley Court", "chemical laboratory". LHR buries them below its frequency-driven head, so PMI is what surfaces them.
- **Frequent but loose** pairs such as "young man" or "could see" occur often but are not real phrases. They score moderately on LHR but low on PMI, which correctly discounts them because the individual words are common.
- **Frequent and tight** phrases such as "Sherlock Holmes" or "Jefferson Hope" score high on both and need no adjudication.

No single measure is uniformly best. Some practical guidance:

- **For a phrase vocabulary that maximises recall** (finding as many genuine phrases as possible), take the top-$k$ from both measures and union them: LHR contributes the well-attested head, PMI the rare-but-specific tail of named entities and technical terms that users actually search for. A union only adds candidates, so plan a downstream prune, whether a human review pass or a secondary filter, to drop the frequent-but-loose pairs that LHR lets through.
- **For a compact, high-precision list**, use LHR alone with the stop-word filter and a frequency floor. Its preference for well-attested pairs is an asset here, and you accept missing some rare phrases.
- **The filters matter more than the choice of measure.** A minimum-frequency floor removes PMI's chance pairs and a stop-word filter removes LHR's grammatical pairs; without both, neither measure produces a usable list. Scale the frequency floor to the corpus: 3 works for a single novel, but a large collection needs a higher floor to keep the vocabulary manageable.
- **An alternative to the union** is to use one measure to rank and the other to filter, for example rank by LHR but drop any pair whose PMI falls below a threshold. This keeps LHR's ordering while using PMI to veto the frequent-but-loose pairs, trading some recall for precision.

## Choosing a threshold and indexing strategy

Once a scoring function is chosen, we pick a threshold and add every bi-gram above it to the vocabulary. The threshold is not a precise cut-off; it trades index size against phrase coverage. Missing a bi-gram is not fatal: the document is still retrievable through its individual terms, so recall is unaffected. What we lose is the precision and ranking boost the phrase token would have given, which lets more incidental co-occurrences through as noise. For many use cases, such as fact-checking or broad information search, that extra noise is acceptable.

````{admonition} Building the bi-gram set with nltk (optional reading)
:class: note dropdown

`nltk` provides collocation finders that do the ranking and filtering directly. The result feeds the `bigrams` set used above.

```python
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import stopwords

finder = BigramCollocationFinder.from_words(tokens)
finder.apply_freq_filter(3)                             # require at least 3 co-occurrences

ignored = set(stopwords.words('english'))
finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored)

# take the top-k scoring pairs and keep them as tuples
top = finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)[:200]
bigrams = {pair for pair, score in top}
```

`nltk` also provides `TrigramCollocationFinder` and `QuadgramCollocationFinder` with matching `TrigramAssocMeasures` and `QuadgramAssocMeasures`.
````


Applying the vocabulary changes how we tokenize, without changing the single scan through the text. Once the accepted bi-grams are known, we tokenize each document, and every query the same way, in one left-to-right pass. At each position we emit the single token, and we do one extra vocabulary lookup to check whether the current token together with the next one forms an accepted bi-gram; if it does, we emit the phrase token as well and move on to the next position, so overlapping phrases are still caught. Keeping the single tokens alongside the phrase is deliberate, as the chapter opener showed: a query for "Holmes" must still match a document that only contains "Sherlock Holmes".

A compact implementation keeps two vocabularies: the ordinary single tokens, and a set of accepted token pairs stored as Python tuples.

```python
tokens = {"a", ..., "city",... "new", ..., "york", ...}  # single token vocabulary
bigrams = {("new", "york"), ("york", "city")}   # accepted pairs from PMI/LHR

def apply_phrases(tokens: list[str], bigrams: set[tuple[str, str]]) -> list[str]:
    result = []
    for i, tok in enumerate(tokens):
        result.append(tok)                              # always keep the single token
        if i + 1 < len(tokens) and (tok, tokens[i + 1]) in bigrams:
            result.append(f"{tok}_{tokens[i + 1]}")     # add the phrase token
    return result

apply_phrases("the city of new york city".split(), bigrams)
# ['the', 'city', 'of', 'new', 'new_york', 'york', 'york_city', 'city']
```

The two overlapping phrases "new york" and "york city" are both emitted, next to the single tokens, exactly the mix the chapter opener described. The accepted-pair set itself comes from the collocation finder: rank with PMI or LHR, apply the frequency and stop-word filters, take the top-$k$, and store each surviving bi-gram as a tuple.


The construction extends to tri-grams and quad-grams: keep a set of triples, check a window of three tokens, and emit a three-word phrase when it matches. Returns diminish quickly, though. Any four-word sequence is rare enough that the scoring functions lose statistical power, and the extra vocabulary rarely earns its storage. Most production systems stop at bi-grams and handle the occasional longer phrase with proximity search at query time.

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
