---
author: Roger Weber
edition: HS26
status: updated
book_part: Foundations
chapter: Advanced Text Processing
section: Intent Routing and Classification
order: "3.5"
---

(advanced-text-processing-intent-routing)=
# Intent Routing and Classification

Recall the second query from the chapter opener: "Bücher von Goethe". By this point in the chapter, the pipeline has enough machinery to see it clearly. The tokenizer splits it into three tokens. The rule-based language detector recognizes German characters and function words. POS tagging identifies "Bücher" as a plural noun, "von" as a preposition, and "Goethe" as a proper noun. Named entity recognition promotes "Goethe" to a person entity. What has not been decided yet is what to do with all this structure. Should the query go to a book catalogue, a web index, an author database, or a general-purpose LLM? This section turns the extracted features into that routing decision.

## The end-to-end query pipeline

A query enters the retrieval system as a raw string. Before any matching happens, it flows through the classical pipeline this chapter has been building:

```
raw query
    -> tokenize                       (section 1)
    -> normalize (case, Unicode)      (section 1)
    -> segment sentences              (section 1, if multi-sentence)
    -> detect language                (section 1 rules + this section's classifier)
    -> stop-word filter               (section 2, language-specific)
    -> stem or lemmatize              (section 2, language-specific)
    -> split compounds                (section 3, if applicable)
    -> POS-tag                        (section 4)
    -> extract named entities         (section 4)
    -> spell-correct                  (section 4)
    -> classify intent                (this section)
    -> route to backend               (this section)
```

The order matters. Language detection has to come before any language-specific step; POS tagging has to come before selective stop-word filtering; NER has to come before intent classification because the entity types are input features to the classifier. What comes out at the end is a structured representation of the query that a downstream system can act on: a language, a set of entities, an intent label, and a corrected token list.

Two of these classification steps use the same underlying machinery. Language detection classifies a query into one of the languages the system knows; intent classification classifies a query into one of the backends the system supports. Both are text classification problems and both can be solved with the same Naive Bayes model. This section develops that model once, then applies it to both problems.

## Naive Bayes for text classification

Naive Bayes chooses the most probable class given the observed features by applying Bayes' theorem and assuming that features are conditionally independent given the class. The derivation lives in the ML foundations appendix; here we state only the decision rule.

For a feature vector $\mathbf{x} = (x_1, \ldots, x_M)$ and classes $C_1, \ldots, C_K$:

```{admonition} Key Formula: Naive Bayes Decision Rule
:class: important

$$\hat{C} = \arg\max_{k} \; P(C_k) \cdot \prod_{j=1}^{M} P(x_j \mid C_k)$$

In log form, which is what implementations actually compute:

$$\hat{C} = \arg\max_{k} \; \log P(C_k) + \sum_{j=1}^{M} \log P(x_j \mid C_k)$$

The prior $P(C_k)$ captures the base rate of each class, and the likelihood $P(x_j \mid C_k)$ captures how well feature $x_j$ fits class $C_k$. The full derivation from Bayes' theorem is in the appendix.
```

Three things need to be learned from a labelled training corpus:

- **The prior $P(C_k)$**: fraction of training documents in class $C_k$. When the training data reflects real-world class frequencies, use it directly; when it is artificially balanced (equal examples per class) or when class balance is unknown, set a uniform prior $P(C_k) = 1/K$ and the priors drop out of the argmax.
- **The likelihood $P(x_j \mid C_k)$**: fraction of feature occurrences in class $C_k$ that fall on token or feature $x_j$. This is a maximum-likelihood estimate over the training data.
- **A smoothing constant**. If $x_j$ never appeared in the training data for class $C_k$, the maximum-likelihood estimate is zero, and one zero collapses the whole product to zero. Add-one (Laplace) smoothing avoids this by adding a small constant to every count.

The product in the decision rule is a direct consequence of the independence assumption: taking the features to be conditionally independent given the class turns the joint likelihood $P(\mathbf{x} \mid C_k)$ into the product $\prod_j P(x_j \mid C_k)$ of per-feature terms. This is what makes the estimation possible at all. In classical text classification the feature space has tens or hundreds of thousands of dimensions, so any particular full vector $\mathbf{x}$ almost never recurs in the training data, and $P(\mathbf{x} \mid C_k)$ could never be estimated directly. Each individual dimension $x_j$, by contrast, is observed many times across the corpus, so the per-feature factors $P(x_j \mid C_k)$ can be counted reliably. Independence trades a joint probability we cannot measure for a product of marginals we can.

The assumption is not literally true, since tokens in real text are not independent, but Naive Bayes still performs well on text because the errors it introduces tend to affect all classes similarly and rarely change which class wins the argmax. Its speed and small memory footprint are the reasons it stays in production stacks two decades after neural classifiers became viable.

## Language detection as classification

Language detection uses character-based n-grams as features. For $n$ from 1 to 5, count how many times each n-gram appears in the input text. This produces a bag-of-n-grams representation exactly like the bag-of-words representation used for topic classification, except that the tokens are short character strings instead of whole words.

Character n-grams work well as features because each language has characteristic short sequences that rarely appear in other languages:

- German trigrams: `sch`, `cht`, `hen`, `und`, `der`, `die`
- English trigrams: `the`, `ing`, `and`, `ion`, `ent`
- French trigrams: `les`, `des`, `que`, `ent`, `oui`
- Italian trigrams: `che`, `non`, `per`, `ent`

Some trigrams like `ent` appear in multiple languages but with different frequencies, which is enough for a Naive Bayes classifier to disambiguate.

Setting up the classifier:

- **Classes**: the target languages. `lingua` covers 70 languages; a domain-specific detector might cover only 5 or 10.
- **Features**: character n-grams from a per-language frequency profile built during training. Most of the discriminating signal sits in a modest set of frequent n-grams, so a frugal detector can store only those significant likelihoods and stay tiny. `lingua` in high accuracy mode keeps the full observed profile across five n-gram orders (unigram through fivegram): the English model holds about 380,000 n-grams (~3.3 MB), German about 470,000 (~4.2 MB). It trades size for robustness on short and ambiguous inputs rather than pruning to the discriminating core.
- **Prior**: uniform when the incoming text has unknown language distribution; empirical when the retrieval system knows its user base is 60% English, 30% German, 10% French.
- **Likelihood**: for the multinomial variant, $P(t_j \mid C_k) = n_{k,j} / \sum_l n_{k,l}$ where $n_{k,j}$ is the count of n-gram $j$ in the training text for language $k$. With +1 smoothing, $P(t_j \mid C_k) = (n_{k,j} + 1) / (\sum_l n_{k,l} + V)$ where $V$ is the vocabulary size.

The decision rule then takes the standard log-form:

$$\hat{C} = \arg\max_{k} \left( \log P(C_k) + \sum_{j} c_j \log P(t_j \mid C_k) \right)$$

where $c_j$ is the observed count of n-gram $j$ in the input.

````{admonition} Example: Detecting German for "Bücher von Goethe"
:class: example

Extract the character trigrams, padding each word boundary with `_` so that word-initial and word-final patterns become features in their own right (the classic Cavnar-Trenkle text-categorization convention):

```
_bü, büc, üch, che, her, er_, _vo, von, on_, _go, goe, oet, eth, the, he_
```

For each candidate language $k \in \{\text{DE}, \text{EN}, \text{FR}, \text{IT}\}$, compute $\log P(C_k) + \sum \log P(t_j \mid C_k)$ over the 15 trigrams. German wins by a large margin because:

- `_bü`, `büc`, `üch` are common in German and essentially absent from English, French, and Italian (`ü` is a diacritic that only German and a few other Latin-script languages use).
- `_vo`, `von`, `on_` are common in German ("von" is the fourth most frequent German preposition).
- `_go`, `goe`, `oet` appear in German names (Goethe, Görlitz, Göttingen) at rates much higher than in English.

The boundary padding matters because languages differ in how words *begin* and *end*, not only in their interior sequences: `_bü` and `_vo` capture a distinctly German way of starting a word. `lingua` reaches the same verdict from unpadded n-grams taken within each word (its stored profiles carry no boundary marker), returning `Language.GERMAN` with a confidence above 0.95.
````

For very short queries with only one or two trigrams that discriminate between languages, confidence drops. `lingua` reports its confidence as a normalized posterior, and the retrieval system can fall back to a default language (English, or the user's browser locale) when the top confidence is too close to the second-best.

```python
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()
detector.detect_language_of("Bücher von Goethe")
# Language.GERMAN

# Restrict to expected languages and get confidence scores
euro = [Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN]
d2 = LanguageDetectorBuilder.from_languages(*euro).build()
d2.compute_language_confidence_values("Bücher von Goethe")
# GERMAN: 0.96, ENGLISH: 0.02, FRENCH: 0.01, ITALIAN: 0.01
```

`langdetect` is a lighter alternative: rule- and n-gram-based, 55 languages, ISO-code output.

```python
from langdetect import detect
detect("Bücher von Goethe")   # 'de'
detect("Livres de Goethe")    # 'fr'
detect("Books by Goethe")     # 'en'
```

## Intent classification as classification

The same machinery routes the query to a backend. Once the language is known, an **intent classifier** decides which of the search system's supported intents the query expresses:

- `book_search`: query looks for books (title, author, ISBN)
- `web_search`: general-purpose keyword search
- `image_search`: query looks for images
- `product_search`: query looks for shoppable products
- `people_search`: query looks for information about a person
- `news_search`: query has a temporal component or a current-events subject
- `map_search`: query looks for a location or directions
- `weather`, `calculator`, `definition`, `translation`: one-shot vertical intents

The training data is query logs annotated with the backend the user actually clicked into. Features are richer than for language detection because more of the pipeline's output can be used:

- The token bag after stemming and stop-word removal
- POS-tag counts (how many nouns, verbs, WH-words)
- NER labels present in the query (has-PERSON, has-LOCATION, has-DATE, has-MONEY)
- Question-form markers (starts with a WH-word, ends with a question mark)
- Detected language (as a categorical feature)

With these features, "Bücher von Goethe" scores highly on `book_search` because it contains a book-domain plural noun ("Bücher") and a PERSON entity ("Goethe") in an author-attribution construction. "What to do in Basel?" scores highly on `map_search` and `web_search` because it contains a WH-word, a LOCATION entity, and no product or person entity. "Who is Albert Einstein?" scores highly on `people_search` because of the WH-word combined with a PERSON entity.

The classifier itself is exactly the same Naive Bayes model as for language detection, just with different features and different classes. In practice, a modern production stack fine-tunes a small neural classifier on top of the same features because it captures conjunctions ("has-PERSON AND WH-word AND is-question") that Naive Bayes cannot express under the independence assumption. Chapter 12 develops the neural variants; this chapter stays with Naive Bayes because it is enough to make the routing decision explicit.

```{admonition} Naive Bayes prefers uniform priors for query classification
:class: warning

Query-log training data reflects the current backend distribution, not the ideal one. If 80% of past queries went to web search, an empirical prior will route new queries to web search 80% of the time regardless of their features, because the prior term overwhelms weak features. For intent routing, use a uniform prior and let the features drive the decision, or use empirical priors only after balancing the training data across intents.
```

## End-to-end walkthrough

Putting all five sections of the chapter together on the running query:

```
Input:              "Bücher von Goethe"

Tokenize:           ['Bücher', 'von', 'Goethe']
Normalize (case):   ['bücher', 'von', 'goethe']
Detect language:    German (confidence 0.96)
Stop words:         drop 'von'  ->  ['bücher', 'goethe']
Stem (Snowball DE): ['buch', 'goeth']
POS tag:            [(Bücher, NOUN), (von, ADP), (Goethe, PROPN)]
NER:                [(Goethe, PERSON)]
Spell-check:        no correction needed
Intent classify:    book_search  (features: language=DE, PERSON present,
                                  plural NOUN of book-domain lemma)
Route:              library catalog with filters
                        language = "de"
                        author   = "Goethe"
                        content  = *
```

The library catalog answers a structured query with all Goethe titles it holds. No general-purpose keyword match against "Bücher von Goethe" was ever needed.

A second walkthrough on the other chapter-opener query. This one combined a WH-question form, tokens that would not appear in the answer, and a domain-vocabulary gap.

```
Input:                       "Who won the F1 race on the weekend?"

Tokenize:                    ['Who', 'won', 'the', 'F1', 'race', 'on',
                              'the', 'weekend', '?']
Normalize (case):            ['who', 'won', 'the', 'f1', 'race', 'on',
                              'the', 'weekend']
Detect language:             English
POS tag:                     [(who, WH-PRON), (won, VERB), (the, DET),
                              (F1, PROPN), (race, NOUN), (on, ADP),
                              (the, DET), (weekend, NOUN)]
NER:                         [(F1, EVENT), (the weekend, DATE)]
Selective stop-word filter:  drop 'the', 'on'
                             keep 'who' as a classification feature
                                  (WH-word signals question form)
Stem/lemma (English):        ['win', 'F1', 'race', 'weekend']
Synonym expansion (domain):  F1     -> {Formula 1, Grand Prix}
                             race   -> {Grand Prix, GP}
Temporal normalization:      the weekend -> [Sat, Sun of the past weekend]
                                          -> ['2026-08-08', '2026-08-09']
Intent classify:             news_search
                                 features: WH-word + EVENT entity +
                                           DATE entity + motorsport terms
                                 features against 'question' subclass:
                                           WH-word + won + who
Route:                       news index with
                                 query = "Formula 1" OR "Grand Prix" OR "GP"
                                 date  = [2026-08-08, 2026-08-09]
                                 sort  = date desc
```

The classical pipeline turned three tokens of question form and one date reference into a scoped keyword query the news index can answer: a Boolean OR over motorsport aliases, a date filter for the past weekend, and a sort by date. Each of the three gaps named in the chapter opener has a specific step above that addresses it. The missing "who" is absorbed into the classification features rather than searched as a keyword. The missing "weekend" becomes a temporal range that a race-results page dated in that range will match. The missing "race" is replaced by the domain-synonym expansion.

That is as far as this chapter takes the F1 example. What the news backend returns is a ranked list of race-result pages, not the two-word answer "Verstappen won" that the user actually requested. Two later chapters carry the example the rest of the way:

- The **semantic search** chapter shows how a dense-retrieval index closes the domain-vocabulary gap without a hand-maintained synonym list, catching the driver names, sponsor terms, and circuit aliases the classical stack never saw.
- The **retrieval-augmented generation** chapter runs a language model over the top-ranked page and returns the extracted name as the answer, turning a list of results into a fact.

The classical pipeline of this chapter is still there in both cases. It prepares the query and prunes the search space so the more expensive downstream steps only see the pages that matter. The router itself does not do the retrieval; it picks a backend and passes the structured query to it. Everything upstream in this chapter was building the structure that the router now dispatches on.

## Where this sits in modern retrieval

Classical intent routing runs in every large search stack in 2025. Google Search, Amazon Search, and enterprise search products all have an intent-classification stage before the retriever. The classifier is rarely still Naive Bayes: production systems have moved to gradient-boosted trees or small transformers because the feature interactions matter and the classifier is on the query-latency critical path only in a few-millisecond budget. Naive Bayes remains the pedagogical baseline and a common first pass in cost-sensitive deployments (mobile, embedded).

Large language models can perform the whole pipeline in one prompt: extract entities, detect the language, classify the intent, and generate the structured backend query. This is a live area of production adoption, especially for open-ended assistant systems. The trade-offs against classical pipelines are latency (an LLM call is 100-1000 milliseconds versus sub-millisecond for the classical stack), cost (per-query LLM cost is measurable, per-query classical cost is not), and confidence quantification (Naive Bayes gives calibrated posteriors, LLMs do not). Production stacks in 2025 usually run the classical pipeline first, escalate to LLMs for the queries where the classical stack's confidence is low, and reserve LLMs for high-value verticals like agentic search and complex reformulation.

Two forward references pick up where this section leaves off:

- Chapter 12 covers deep-learning text classification with TextCNN and transformer-based classifiers, using sentiment analysis as its running example. The Naive Bayes classifier of this section is the baseline those methods are compared against.
- The semantic-search chapter covers embedding-based intent classification: instead of extracting features from the query and running Naive Bayes, embed the query into a vector space and compare it to prototype vectors for each intent. Modern hybrid systems combine both approaches.

```{admonition} Hands-on: Intent Routing
:class: hint

Train a Naive Bayes language detector on a small multi-language corpus, then extend it into an intent classifier over a set of example queries. Compare the classical pipeline against a single-prompt LLM call.

[Open notebook -&gt;](https://github.com/mmir-unibasel/mmir-unibasel-hs26/blob/main/ch03-05-intent-routing.ipynb)

*Includes pre-run results; you can read through or download and experiment.*
```

The next page summarizes the chapter and points to the material that follows it in the book.
