---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Foundations
chapter: Advanced Text Processing
section: Phrases and Compounds
order: "3.3"
---

# Phrases and Compounds

<!--
TODO (rewrite): re-order as:
  1. Motivation: "New York" != "New" + "York"; the phrase-detection idea
  2. Naive bi-gram frequency and its stop-word problem
  3. Key Formula: PMI (with worked example)
  4. Key Formula: LHR (fix the Unicode-italic mess in the raw LaTeX below)
  5. Threshold discussion; indexing both parts and phrase
  6. Extension to tri-grams / quad-grams (short)
  7. nltk collocations code snippet
  8. Compounds: motivation (German law, code identifiers), endocentric/exocentric
  9. Key Formula: log-frequency split score (with one worked example)
  10. Modern relevance paragraph (Lucene/Solr/Elasticsearch decompounders,
      CompoundPiece research 2023, dense retrieval nuance)
-->

## Phrases: bi-grams and multi-word expressions

<!-- source: 4_2_tokenization_revisited.md, N-grams block -->

Rather than making tokens smaller, we can create larger tokens by combining multiple words into a single token known as **n-grams**. This approach is particularly valuable in languages where words form phrases with distinct or more specific meanings. Examples include:

- Idioms and fixed expressions: *mother tongue*, *red handed*, *butterfly effect*, *black box*, *cold shoulder*, *silver bullet*, *piece of cake*.
- Domain phrases: *thai food*, *prime minister*, *middle management*, *crystal clear*, *chief of staff*, *speed dial*, *multimedia retrieval*.
- Named entities: *New York City*, *Salt Lake City*, *Albert Einstein*, *Amazon Web Services*, *Ford Mustang*, *University of Basel*.

In all these examples, it makes more sense to use phrases rather than the individual terms. To enrich a vocabulary with phrases, we can create them manually or form them automatically from a text corpus.

## Naive bi-gram frequency

A naive approach first constructs all possible bi-grams in a corpus and then counts their occurrences. The top-*n* most frequent bi-grams are added to the vocabulary. However, this method has a clear limitation: "of the" is the most frequent bi-gram simply because it comprises two frequently-used stop words in the language.

A first enhancement excludes stop words when generating bi-grams and considers only consecutive pairs of non-stop words. Ensure that you do not merely remove stop words from the stream but eliminate pairs containing a stop word. Otherwise you create pairs that originally had a stop word in between.

With this filter, the result appears more favorable, with names from the novel forming new terms in the vocabulary. This streamlines the search for names since we only need to search for the bi-gram, eliminating the need to search for individual parts and apply a proximity constraint.

Nonetheless, a few issues remain. Phrases like "said Holmes", "could see", and "young man" are common pairs, but they do not contribute significantly to describing the context they appear in. In the case of "said Holmes", we observe that these two terms co-occur relatively rarely but are more often associated with other words (e.g., "said" is not exclusively used with "Holmes").

## Pointwise Mutual Information (PMI)

<!-- source: 4_2_tokenization_revisited.md, PMI block -->

The **Pointwise Mutual Information (PMI)** measures word associations by comparing their actual co-occurrence frequency to what would be expected if they were independent. In our previous example, the bi-gram "said Holmes" occurred 12 times together. However, "said" appeared 207 times, and "Holmes" 94 times individually. In essence, "said" and "Holmes" rarely co-occur (12 out of a maximum of 94 times), and "said" pairs with many other words. Although they occur together more frequently than other bi-grams, this observation suggests they are not a distinctive-enough bi-gram for our vocabulary. We are more interested in word pairs like the names, which predominantly appear as bi-grams (even though first and last names can also occur independently).

To formalize this measure, let $t_{1}$ represent the first term in the bi-gram and $t_{2}$ the second term. We count the occurrences of the individual terms as $\text{tf}(t_{1})$ and $\text{tf}(t_{2})$, and of the bi-gram as $\text{tf}(t_{1}, t_{2})$. PMI compares the likelihood of terms occurring together to the expected probability if they were independent of each other:

$$\text{pmi}(t_{1}, t_{2}) = \log_{2} \frac{p(t_{1}, t_{2})}{p(t_{1}) \cdot p(t_{2})} = \log_{2} p(t_{1}, t_{2}) - \log_{2} p(t_{1}) - \log_{2} p(t_{2})$$

If the corpus comprises $N$ terms, the probabilities are determined by the ratio of the term frequency to $N$:

$$\text{pmi}(t_{1}, t_{2}) = \log_{2} \frac{\frac{\text{tf}(t_{1}, t_{2})}{N}}{\frac{\text{tf}(t_{1})}{N} \cdot \frac{\text{tf}(t_{2})}{N}} = \log_{2} \frac{N \cdot \text{tf}(t_{1}, t_{2})}{\text{tf}(t_{1}) \cdot \text{tf}(t_{2})} \propto \log_{2} \frac{\text{tf}(t_{1}, t_{2})}{\text{tf}(t_{1}) \cdot \text{tf}(t_{2})}$$

In the last part of the formula, we eliminated the constant multiplier $\log_{2}(N)$ that applies to all bi-grams. The rightmost formula now determines the significance of bi-grams with the PMI measure. While it is possible to remove the $\log_{2}(\cdot)$ as well, keeping it in place helps maintain values within more manageable ranges for humans.

Note: the PMI value is maximized when $\text{tf}(t_{1}) = \text{tf}(t_{2}) = \text{tf}(t_{1}, t_{2})$, meaning all occurrences of the two terms exist exclusively within the bi-gram. If a term appears outside the bi-gram, the denominator becomes larger, resulting in a smaller PMI value.

Stop words that appear in bi-grams are naturally given lower weights because they are highly frequent outside of the bi-gram context. There is therefore no longer a necessity to employ a stop-word filter (although it can still be used for efficiency when computing PMI).

<!--
TODO (rewrite): promote this to a Key Formula admonition:
```{admonition} Key Formula: PMI
:class: important
...
```
Intuition: PMI is high when two terms almost always occur together and rarely apart.
-->

### PMI worked example

Using PMI on the same novel corpus, the highest-scoring bi-grams are dominated by rare pairs. "Army Medical" appears only once, and the terms within the bi-gram also occur only once within that bi-gram, which is why it receives the highest score.

We already established that the PMI score is highest when $\text{tf}(t_{1}) = \text{tf}(t_{2}) = \text{tf}(t_{1}, t_{2})$. Let such a bi-gram occur $n$ times. The PMI score is then:

$$\text{pmi}(t_{1}, t_{2}) = \log_{2} \frac{N \cdot \text{tf}(t_{1}, t_{2})}{\text{tf}(t_{1}) \cdot \text{tf}(t_{2})} = \log_{2} \frac{N \cdot n}{n \cdot n} = \log_{2}(N) - \log_{2}(n)$$

In other words, for bi-grams where the terms only occur together in that bi-gram, the PMI is high when the count $n$ is low. The optimal value is achieved when $n = 1$, as demonstrated in the result table ($\log_{2}(N)$ is $15.39$ for this example).

To improve the quality of returned bi-grams, we can apply a minimum-frequency filter. This eliminates both bi-grams containing stop words and bi-grams with infrequent terms. The bi-gram result improves, revealing many names from the novel and capturing meaningful pairs like "never mind", "old farmer", or "two detectives".

## Likelihood Ratio (LHR)

<!-- source: 4_2_tokenization_revisited.md, LHR block -->

**Likelihood Ratios (LHR)** are another form of hypothesis testing, similar to the chi-squared test but more robust when dealing with sparse data. Moreover, the resulting number is easier to interpret, indicating how much more likely one hypothesis is compared to another. In the context of bi-grams, the initial hypothesis $H_{1}$ assumes independence between terms $t_{1}$ and $t_{2}$ in the bi-gram:

$$H_{1}: \; P(t_{2} \mid t_{1}) = P(t_{2} \mid \neg t_{1}) = p$$

The first probability represents the conditional likelihood of $t_{2}$ following $t_{1}$; the second is the conditional probability of $t_{2}$ not following $t_{1}$. Let $\text{tf}_{1} = \text{tf}(t_{1})$, $\text{tf}_{2} = \text{tf}(t_{2})$, and $\text{tf}_{12} = \text{tf}(t_{1}, t_{2})$. For hypothesis $H_{1}$, we can use the maximum-likelihood estimate for $p = \text{tf}_{2} / N$, where $p$ is the probability of $t_{2}$ following any term, whether it is $t_{1}$ or not (independence). Assuming a binomial distribution, we can calculate the likelihood of observing these counts as:

$$L(H_{1}) = b(\text{tf}_{12}; \text{tf}_{1}, p) \cdot b(\text{tf}_{2} - \text{tf}_{12}; N - \text{tf}_{1}, p)$$

with $b(k; n, x) = \binom{n}{k} x^{k} (1 - x)^{n-k}$.

The first binomial distribution calculates the likelihood of observing $\text{tf}_{12}$ instances of $t_{2}$ following $t_{1}$ out of $\text{tf}_{1}$ occurrences, considering the probability $p$ that the term $t_{2}$ appears at any position. The second hypothesis $H_{2}$ assumes that $t_{2}$ depends on $t_{1}$ and hence the conditional probabilities differ:

$$H_{2}: \; p_{1} = P(t_{2} \mid t_{1}), \quad p_{2} = P(t_{2} \mid \neg t_{1}), \quad p_{1} \neq p_{2}$$

As before, we can use maximum-likelihood estimates $p_{1} = \text{tf}_{12} / \text{tf}_{1}$ and $p_{2} = (\text{tf}_{2} - \text{tf}_{12}) / (N - \text{tf}_{1})$ from the observed counts. Assuming a binomial distribution, we compute the likelihood of the second hypothesis:

$$L(H_{2}) = b(\text{tf}_{12}; \text{tf}_{1}, p_{1}) \cdot b(\text{tf}_{2} - \text{tf}_{12}; N - \text{tf}_{1}, p_{2})$$

Finally, the log likelihood ratio is:

<!-- TODO (rewrite): the source rendered this equation with Unicode italic
     characters (mathematical alphanumeric symbols block), not proper LaTeX.
     Restore the correct LaTeX formulation below and remove this placeholder. -->

[MATH_ERROR: rewrite the log-lambda formula as proper LaTeX.

Intended formula:

$$\log \lambda = \log \frac{L(H_1)}{L(H_2)} = \log \frac{L(\text{tf}_{12}; \text{tf}_1, p) \cdot L(\text{tf}_2 - \text{tf}_{12}; N - \text{tf}_1, p)}{L(\text{tf}_{12}; \text{tf}_1, p_1) \cdot L(\text{tf}_2 - \text{tf}_{12}; N - \text{tf}_1, p_2)}$$

with $L(k; n, x) = x^{k} (1 - x)^{n-k}$.

Verify against Manning and Schütze, *Foundations of Statistical NLP*, Chapter 5.]

<!--
TODO (rewrite): promote to a Key Formula admonition with a one-line intuition:
"LHR compares dependence to independence: high values mean t2 strongly favours
appearing after t1 (or strongly avoids it)."
-->

### LHR worked example

Applying the LHR measure to the same example text, we sort bi-grams by $-2 \cdot \log \lambda$. The top results interestingly show stop words reappearing. Unlike the naive frequency approach where stop words appear due to their high frequency, LHR compares the hypothesis of independence versus dependence. Consider the bi-gram "I am": it is not a significant bi-gram for the vocabulary, yet "am" is strongly dependent on "I" and follows the term "I" in 39 out of 41 instances.

We can enhance the quality of bi-grams obtained with LHR by excluding those containing a stop word (do not filter stop words before forming bi-grams). Unlike the PMI ranking, the more frequent names now occupy the top positions. Notably, "Sherlock Holmes" appears 48 times as a bi-gram and attains the highest LHR value, while it held only the 14th place in the PMI ranking, due to the PMI's preference for lower numbers of occurrences.

## Choosing a threshold and indexing strategy

<!-- source: 4_2_tokenization_revisited.md, closing paragraphs of the n-gram block -->

With all the bi-gram scoring methods discussed, we need to establish a threshold. All bi-grams with scores exceeding this threshold are included in the vocabulary; the rest are excluded. There is no need to be accurate in setting the threshold; rather, we should take additional search and storage overhead into account. If we missed a bi-gram, we can still find it with proximity measures.

When creating bi-grams, we can also choose to index both individual terms and the bi-gram. This allows us, for example, to search for "Holmes" which would otherwise not match with occurrences of the bi-gram "Sherlock Holmes".

We can expand this concept to tri-grams or even quad-grams and expand our vocabulary accordingly (typically selecting several hundreds to thousands in a large corpus).

## nltk collocations in code

<!-- source: 4_2_tokenization_revisited.md, code block, converted from bullet-list to real Python -->

`nltk` offers convenient functions for handling collocations:

```python
from nltk.collocations import (
    BigramCollocationFinder,  TrigramCollocationFinder,  QuadgramCollocationFinder,
    BigramAssocMeasures,      TrigramAssocMeasures,      QuadgramAssocMeasures,
)
from nltk.corpus import stopwords

# choose bi-grams, tri-grams, or quad-grams
finder = BigramCollocationFinder.from_words(tokens)
# finder = TrigramCollocationFinder.from_words(tokens)
# finder = QuadgramCollocationFinder.from_words(tokens)

# choose a measure (must match the finder; here for bi-grams)
measure = BigramAssocMeasures.raw_freq
# measure = BigramAssocMeasures.pmi
# measure = BigramAssocMeasures.likelihood_ratio

# apply a frequency filter
finder.apply_freq_filter(3)

# apply a stop-word filter
ignored_words   = stopwords.words('english')
stopword_filter = lambda w: len(w) < 3 or w.lower() in ignored_words
finder.apply_word_filter(stopword_filter)

# obtain top-k results
k      = 20
scores = finder.score_ngrams(measure)[:k]

# output: term 1, term 2, freq(t1), freq(t2), freq(bigram), score
for ((t1, t2), score) in scores:
    print(f'{t1} {t2} {finder.word_fd[t1]} {finder.word_fd[t2]} {finder.ngram_fd[(t1, t2)]} {score}')
```

## Compounds

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, compounds block -->

In linguistics, **compounds** are words created by combining two or more base words, occasionally using binding syllables (e.g., "Liebeslied") or characters (e.g., "must-have"). While most languages support basic compound formation to create new words (e.g., "smalltalk"), languages such as German and Finnish permit the formation of arbitrarily long compounds. A few examples:

- **Finnish**:
    - *kolmivaihekilowattituntimittari* -> electricity meter
    - *atomiydinenergiareaktorigeneraattorilauhduttajaturbiiniratasvaihde* -> atomic nuclear energy reactor generator condenser turbine cogwheel stage
    - *rautatieasema* -> railway station

- **German**:
    - *Wolkenkratzer* -> skyscraper
    - *Rinderkennzeichnungs- und Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz* (a real law in Mecklenburg-Vorpommern, 1999-2013) -> cattle marking and beef labeling supervision duties delegation law
    - *Stacheldraht* -> barbed wire

- **Dutch**:
    - *arbeidsongeschiktheidsverzekering* -> disability insurance
    - *rioolwaterzuiveringsinstallatie* -> sewage treatment plant
    - *doorgroeimogelijkheden* -> possibilities for advancement

### Endocentric versus exocentric compounds

We can classify compounds as either endocentric or exocentric:

- **Endocentric compounds** derive their meaning from their constituent parts. They have a "head" that imparts both semantic and syntactic attributes to the compound, while the other elements modify and refine its meaning. In "sunglasses", "glasses" serves as the head and "sun" acts as the modifier.

- **Exocentric compounds** do not derive their meaning from their constituent parts and may even ignore the lexical class of their individual elements (e.g., "must-have" is a noun, not a verb). In "skyscraper", neither "sky" nor "scraper" acts as the head, and the term names an entirely different object.

### Why compounds matter for retrieval

Consider a compound word like "Rindfleischetikettierungsueberwachungsaufgabenuebertragungsgesetz". The first problem is spelling it correctly, which makes it hard to find in document titles when users make spelling mistakes in their query. Another issue is that a user must list all the constituent parts to find the document. It would be more user-friendly to allow a partial query such as "Rindfleisch Etikettierung Gesetz". Unfortunately, the retrieval models we have discussed so far do not support partial term queries against the vocabulary.

The recommended approach is to split compounds into their parts and include both the parts and the full compound as tokens in the document. For example, the German word "Abfalleimer" becomes "Abfall", "Eimer", and "Abfalleimer". This helps match a wider range of queries and works well for endocentric compounds, where the parts reflect the compound's meaning. It is less effective for exocentric compounds such as "skyscraper" or the German "Wolkenkratzer". Splitting "skyscraper" into "sky" and "scraper", or "Wolkenkratzer" into "Wolke" and "Kratzer", adds incorrect semantics to the document. Whether the benefit from splitting endocentric compounds outweighs the harm from creating wrong meanings for exocentric ones depends on the retrieval scenario.

### Compound splitting

<!-- source: 4_3_lemmatization_and_linguistic_transformation.md, compound-splitting block -->

We present two methods for automatically splitting compounds. Both use rule-based or morphological analysis to find possible splits of a term. The details vary by language:

- In English, we can split compounds using hyphens and syllables in accordance with English hyphenation rules. For example, "must-have" becomes "must" and "have"; and "skyscraper" becomes "sky", "scrap", and "er".

- In German, we split on syllables following German hyphenation rules. For example, "Wolkenkratzer" becomes "wol", "ken", "krat", "zer"; and "Schifffahrtskapitän" becomes "Schiff", ("fahrt", "fahrts"), "ka", "pi", "tän". In the last example, "s" is a binding letter for compound generation, so we have to test with both pieces "fahrt" and "fahrts".

As a next step, we produce all possible combinations of such splits:

```
skyscraper           (sky, scrap, er), (skyscrap, er), (sky, scraper)
wolkenkratzer        (wol, ken, krat, zer), (wolken, krat, zer), (wol, kenkrat, zer),
                     (wol, ken, kratzer), (wolken, kratzer), (wolkenkrat, zer),
                     (wol, kenkratzer)
Schifffahrtskapitän  (Schiff, fahrt, ka, pi, tän), (Schiff, fahrts, ka, pi, tän),
                     (Schifffahrt, ka, pi, tän), (Schifffahrts, ka, pi, tän),
                     ..., (Schifffahrt, kapitän), (Schifffahrts, kapitän)
```

To find valid splits, we start by discarding any splits that contain components not found in our vocabulary or dictionary. When multiple options remain, we determine the best split based on the frequency of the components. Let $\mathbb{S}$ represent the set of all possible splits, and let $S = \{p_{i}\}$ represent all the individual components of split option $S \in \mathbb{S}$. We compute $\text{tf}(p_{i})$ as the number of times piece $p_{i}$ appears in the corpus (or is provided by the dictionary), and $N$ as the total number of tokens.

<!-- TODO (rewrite): insert the log-frequency score formula from the source
     (currently missing an explicit LaTeX rendering) and promote it to a Key
     Formula admonition. Suggested form:

$$\text{score}(S) = \frac{1}{|S|} \sum_{p_i \in S} \log \frac{\text{tf}(p_i)}{N}$$

Intuition: pick the split whose average log-frequency of parts is highest;
this is the most probable decomposition. -->

In simpler terms, we choose the split with the highest average log-frequency values for its components. This indicates the most probable way to combine the parts into a compound.

### Modern relevance of compound splitting

<!--
TODO (rewrite, new paragraph): fold the web-search findings into 1-2 paragraphs.

Points to cover:
- Lucene ships `DictionaryCompoundWordTokenFilter` and
  `HyphenationCompoundWordTokenFilter` in `org.apache.lucene.analysis.compound`.
  Solr and Elasticsearch expose both.
- Elastic recommends the hyphenation decompounder as the default for German,
  Dutch, and other compounding languages (Elastic search-labs, Sept 2025).
- Actively maintained dictionaries: `uschindler/german-decompounder`,
  `redlink-gmbh/solr-compound-word-filter`.
- Dense retrieval sidesteps compounds via BPE / WordPiece subword tokenizers,
  but a 2023 arXiv study (2305.14214, CompoundPiece) shows even modern
  subword tokenizers do a mediocre job of decompounding German.
- Hybrid search stacks (BM25 + dense) still benefit from explicit decompounding
  on the BM25 side. So this classical technique remains a real production
  choice in 2025.

Cite Elastic docs and the arXiv paper in Further Reading.
-->
