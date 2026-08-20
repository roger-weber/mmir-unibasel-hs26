---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Foundations
chapter: Advanced Text Processing
section: Intent Routing and Classification
order: "3.5"
---

# Intent Routing and Classification

<!--
TODO (rewrite): this is a new section built around the query-understanding
payoff. Write it from scratch, using the raw NB material moved from ch12 as
the classifier core.

Section outline:
  1. The end-to-end query analysis pipeline
       raw query -> tokenize -> normalize -> language-detect -> POS/NER
                 -> classify intent -> route
  2. "Bücher von Goethe" worked example
       - tokenize -> ["Bücher", "von", "Goethe"]
       - language detect -> German
       - POS/NER -> ("Bücher"=noun, "Goethe"=person)
       - intent classify -> {book_search: 0.9, general_search: 0.1}
       - route -> library catalog with author filter="Goethe"
  3. Naive Bayes classifier (reference appendix for the math)
  4. Language detection as the concrete NB example (moved from ch12 §14.2.1)
  5. Intent classification as the second NB example
  6. Modern angle: LLM prompt-based intent extraction (one paragraph)
  7. Bridge forward: pointer to ch12 for deep classification (TextCNN,
     transformer-based) and to semantic-search chapter for embedding-based
     intent classification
-->

## The end-to-end query analysis pipeline

<!--
TODO (rewrite): write a compact end-to-end walkthrough that composes the
techniques from sections 1-4 into one pipeline:

1. Tokenize (section 1)
2. Normalize: case, Unicode, accents (section 1)
3. Detect language (section 1 rules + section 5 classifier)
4. Sentence segmentation, if needed (section 1)
5. POS tagging (section 4)
6. NER (section 4)
7. Compound splitting for compounding languages (section 3)
8. Stemming or lemmatization (section 2)
9. Intent classification (this section)
10. Route to appropriate backend

Then walk through "Bücher von Goethe" as the running example, showing what each
stage produces and how the final routing decision is made.
-->

## Naive Bayes for classification of short texts

<!-- source: moved from book/chapters/ch12_video_structural_features/14_2_text_features.md §14.2.1
     Purpose here: provide the classifier machinery that underlies both
     language detection and query intent classification. -->

Naive Bayes employs a conditional probability model based on Bayes' theorem:

$$P(C_{k} \mid \mathbf{x}) = \frac{P(C_{k}) \cdot P(\mathbf{x} \mid C_{k})}{P(\mathbf{x})}, \qquad \text{posterior} = \frac{\text{likelihood} \cdot \text{prior}}{\text{evidence}}$$

In this equation, $\mathbf{x}$ represents a feature vector and $C_{k}$ is the class or target. $P(C_{k})$ is the prior, that is knowledge about the distribution of classes. $P(\mathbf{x} \mid C_{k})$ is the likelihood of observing feature $\mathbf{x}$ for a specific class $C_{k}$, and $P(\mathbf{x})$ is the overall evidence of observing $\mathbf{x}$, regardless of class. $P(C_{k} \mid \mathbf{x})$ represents the posterior, which is the knowledge gained when observing feature $\mathbf{x}$ and lets us infer its association with class $C_{k}$.

Consider $\mathbf{x}$ as a high-dimensional vector, often derived from a vast term space used in documents. Given the high dimensionality and the restricted training data, accurately modeling the probability distribution function in this sparse space is challenging. To simplify, naive Bayes assumes conditional independence among features:

$$P(\mathbf{x} \mid C_{k}) = P(x_{1}, \ldots, x_{M} \mid C_{k}) = \prod_{j=1}^{M} P(x_{j} \mid C_{k})$$

Using the probability model, we choose the most probable hypothesis, that is the class $C_{k^{*}}$ that maximizes the posterior. This selection principle is commonly referred to as **maximum a posteriori (MAP)**:

<!-- TODO (rewrite): the source has a [MATH_ERROR] here where the MAP decision
     rule should be written out. Suggested form:

$$C_{k^{*}} = \arg\max_{k} P(C_{k}) \cdot \prod_{j=1}^{M} P(x_{j} \mid C_{k})$$

Note: $P(\mathbf{x})$ is a constant across classes and does not affect the
argmax, so it drops out. -->

[MATH_ERROR: write the MAP decision rule as proper LaTeX (see TODO above).]

That is it. The equation describes the decision rule of naive Bayes. The only thing left are the estimates for the probabilities on the right-hand side.

```{seealso}
The full derivation of naive Bayes, including smoothing and multivariate
Bernoulli vs multinomial variants, is in [](#appendix-naive-bayes). This
section restates only the results needed for language detection and intent
classification.
```

## Language detection with character n-grams

<!-- source: moved from ch12/14_2_text_features.md §14.2.1, second half.
     This is the classifier-based counterpart to the rule-based sketch in
     section 1 of this chapter. -->

In the language-detection scenario, we use character-based n-grams of varying lengths (e.g., $n$ from 1 to 5). We count how often these n-grams appear in the text, resulting in a bag-of-words representation that forms a multinomial distribution. The feature vector $\mathbf{x}$ represents these counts over a defined vocabulary for each language.

The priors $P(C_{k})$ depend on the scenario. We can use a maximum-likelihood estimator based on observations in the training set. Let $N_{k}$ be the number of texts for the language denoted by class $C_{k}$, and $N$ the total number of texts:

$$P(C_{k}) = \frac{N_{k}}{N}$$

If we lack knowledge of the language distribution or wish to avoid training bias, we can select a constant prior for all classes:

$$P(C_{k}) = \frac{1}{K}$$

The constant prior can then be omitted from subsequent calculations since it only scales posteriors for all classes.

To estimate the likelihoods $P(x_{j} \mid C_{k})$ from texts in a language represented by class $C_{k}$, we count the n-gram occurrences in the training data for that language (multinomial distribution). For each language, we first establish an appropriate vocabulary using methods similar to WordPiece or BPE to control vocabulary size, prioritizing the most frequent n-grams since they have the most influence on the posterior. Let $n_{k,j}$ denote the total occurrences of n-gram $t_{j}$ in all training texts for the language $C_{k}$:

$$p_{k,j} = \frac{n_{k,j}}{\sum_{l} n_{k,l}} \qquad \text{or smoothed:} \qquad p_{k,j} = \frac{n_{k,j} + 1}{\sum_{l} n_{k,l} + M}$$

As we choose the vocabulary tailored to the target language and exclude infrequent or absent n-grams from the test set, we do not usually require "+1" smoothing here. In other text classification tasks, however, smoothing prevents $p_{k,j}$ from reaching 0 for rare tokens during predictions (which would collapse the posterior to 0).

Finally, we can predict the language based on posteriors. Instead of multiplying probabilities we use sums over log-probabilities:

<!-- TODO (rewrite): the source has a [MATH_ERROR] where the log-posterior
     decision rule for language detection should appear. Suggested form:

$$\hat{C} = \arg\max_{k} \left( \log P(C_{k}) + \sum_{j} n_{j} \log p_{k,j} \right)$$

where $n_{j}$ is the observed count of n-gram $j$ in the test text.

Follow with softmax over the log-scores across target languages. -->

[MATH_ERROR: log-posterior sum + softmax expression for language detection.]

We can obtain scores with a softmax over the target languages and select the language with the highest score.

## Language detection in code

<!-- source: ch12 code snippets, cleaned up -->

The `lingua-language-detector` is a highly efficient language detector with over 99% accuracy for more than 70 languages:

```python
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()

detector.detect_language_of("This is an example sentence")  # Language.ENGLISH
detector.detect_language_of("Je suis un exemple de phrase")  # Language.FRENCH
```

We can also inquire about the likelihood of a phrase belonging to a particular set of languages:

```python
languages = [Language.ENGLISH, Language.FRENCH, Language.ITALIAN]
detector  = LanguageDetectorBuilder.from_languages(*languages).build()

detector.compute_language_confidence_values("Je suis à New York")
# -> FRENCH: 0.45, ENGLISH: 0.37, ITALIAN: 0.18
```

The detector also predicts languages from fragments, showing it operates at sub-word level:

```python
detector.compute_language_confidence_values("hau mei")
# -> GERMAN: 0.82, ENGLISH: 0.10, ITALIAN: 0.07
```

The 3-grams "hau" and "mei" are more common in German texts than in English and Italian, resulting in higher confidence scores.

Another Python library is `langdetect`, also a rule- and n-gram-based language detector for 55 languages that returns ISO codes:

```python
from langdetect import detect, detect_langs

detect("This is an example sentence")      # 'en'
detect("je suis un exemple de phrase")     # 'fr'
detect("Este es un ejemplo de frase")      # 'es'
detect("Dies ist ein Beispieltext")        # 'de'
detect("Questo è un esempio di frase")     # 'it'

detect_langs("Je suis à New York")         # [fr:0.86, en:0.14]
```

## Intent classification

<!--
TODO (rewrite, new subsection): apply the same NB machinery to intent
classification.

Points to cover:
- Feature set: tokens after normalization, POS tags, NER labels, question-form
  markers (WH-words, "?" presence), presence of currencies, dates, product
  names.
- Class set: example intent labels for a general search backend
    {web_search, image_search, video_search, product_search, people_search,
     news_search, map_search, calculator, weather, definition, ...}
- Training data: query logs annotated with clicked-backend labels.
- Inference: MAP over classes, then route.
- Practical concern: query ambiguity ("Jaguar" -> car or animal) -> present
  multiple options or use user history as an extra prior.
- Where classical NB wins: fast, cheap, explainable, no GPU. Great for
  first-line filtering before more expensive re-ranking or LLM routing.

Then the "Bücher von Goethe" walkthrough referenced at the top of the section.
-->

## Modern angle: LLM-based intent extraction

<!--
TODO (rewrite, new paragraph): compare the classical NB router to the modern
LLM prompt approach.

Points to cover:
- LLM can extract intent, entities, and routing decisions from a single prompt.
- Trade-offs: LLM is more flexible and handles edge cases, but is slower,
  costs more per query, and can hallucinate slots.
- Practical pattern: use classical NB as a fast first pass, escalate to LLM
  for low-confidence queries. Points to ch07 RAG and the ch12 deep
  classification methods for the neural side.
-->

## Bridge forward

<!--
TODO (rewrite): close the section with pointers.

- To [](#ch12-text-features) for deep-learning text classification (TextCNN,
  transformer-based classifiers).
- To the future semantic-search chapter for embedding-based intent classifiers.
- To [](#appendix-naive-bayes) for the full NB derivation.
-->
