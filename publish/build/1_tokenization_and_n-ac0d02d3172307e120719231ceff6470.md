---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Foundations
chapter: Advanced Text Processing
section: Tokenization and Normalization
order: "3.1"
---

# Tokenization and Normalization

<!--
TODO (rewrite): re-order subsections into a clean progression:
  1. Naive tokenizer weaknesses
  2. Modern tokenization (nltk / spaCy)
  3. Retrieval-oriented cleanup (short-token filter, non-word filter)
  4. Case, Unicode, and accent normalization
  5. Sentence segmentation
  6. Scriptio continua and code identifiers (motivates sub-word)
  7. Character n-gram sub-word tokens with `#` word-boundary prefix
  8. Rule-based language detection sketch (with pointer forward to section 5 for the classifier)

Add figures 3-1 (ELIZA) and 3-2 (SHRDLU) inside the optional NLP-history dropdown.
-->

## Naive tokenizer and its weaknesses

<!-- source: 4_2_tokenization_revisited.md, opening block -->

In the previous chapter, we divided the text into parts and employed a straightforward tokenization method. A token is separated by non-word characters. Here is a straightforward Python implementation:

```python
def word_tokenize(text: str) -> list[str]:
    text = re.sub(r'[^\w\-]+', ' ', text)
    return [token for token in text.split(' ') if token]
```

In this basic scenario, any string of characters that is not a Unicode letter, number, underscore, or hyphen gets substituted with a single space. Afterwards, we break the text at spaces to create a token list. This method usually functions effectively but has certain limitations. Take a look at the sentence below and the resulting split:

> I buy my parents' 10% of U.K. startup for $1.4 billion. Dr. Watson's cat called Mrs. Hersley and it was w.r.o.n.g., more to come ...

While it is an artificial and nonsensical sentence, it highlights some of the weaknesses:

- The possessive "'s" (also: parents "'") is omitted, resulting in a single "s" token. This outcome can be advantageous or disadvantageous, depending on the task. It is beneficial for retrieval since it enables the merging of "Watson" and "Watson's", allowing users to find the name without testing alternative written forms. However, it becomes problematic for sentence analysis as it breaks the link between "Watson" and "cat". The typical retrieval approach involves removing possessive forms and single-letter as well as non-alphabetic terms.

- Numbers function well when they are small, positive integers, but tokenization struggles with percentages, currencies, and floating-point numbers, among other cases not covered here. In the context of retrieval and NLP, numbers are often disregarded or entirely removed. However, in generative AI, the language model may need to generate an answer with the accurate dollar amount from this sentence.

- Abbreviations like "U.K.", "Dr.", "Mrs.", and the artificial "w.r.o.n.g." are not accurately identified. Abbreviations with multiple dots are treated as separate terms, with all tokens lacking the final dot that signifies an abbreviation. Consequently, searching for "U.K." is not feasible.

- Interpunctuation is absent. This is beneficial for retrieval but restricts sentence analysis for context and word relationships.

## Modern tokenization with nltk and spaCy

<!-- source: 4_2_tokenization_revisited.md -->

Modern word-based tokenizers are available in the `nltk` and `spaCy` libraries. The outcome for the same sentence closely matches between the two libraries, with the exception of the artificial abbreviation "w.r.o.n.g." which spaCy's neural model finds challenging. Compared with the previous page's basic method:

- "'s" and "'" possessives are treated as terms. In retrieval tasks, they can be filtered out; in NLP tasks, they aid sentence structure analysis.
- Floating-point numbers are now recognized correctly, including negative numbers (not shown). Percentages and currency symbols are split into individual tokens, preserving this information compared with the previous method.
- Abbreviations are accurately identified and represented as single tokens. Both `nltk` and `spaCy` employ machine learning to detect common abbreviations.
- Interpunctuation is fully retained. In retrieval tasks it can be filtered out; in NLP tasks it helps analyze sentence structure.
- Both packages offer support for language-specific peculiarities, such as French abbreviations. Refer to their documentation for details on enabling multi-lingual tokenization.

## Retrieval-oriented cleanup

<!-- source: 4_2_tokenization_revisited.md, "Tokenization for Retrieval" block -->

The tokens produced by modern tokenizers are well-suited for NLP tasks such as part-of-speech tagging. For retrieval tasks, however, many of these tokens are unnecessary as they do not provide additional information. To create a token list for retrieval scenarios, we typically apply these cleanup actions:

- Remove short tokens, like single-letter ones, as they lack specific content description.
- Exclude non-word tokens, such as numbers and special characters, except for words with hyphens and abbreviations with dots. This also removes tokens from possessive forms.
- Optionally, convert Unicode characters (e.g., accents) to their closest ASCII equivalents, e.g., "Zürich" to "Zurich". This can reduce vocabulary size and simplify matching between queries and documents, especially when users lack easy access to specific Unicode letters (e.g., "słychać" with characters not found on the keyboard).
- Optionally, convert tokens to lowercase or apply case conversion to their standardized form. This is useful for scenarios like sentence beginnings with capitalized words, title case usage, or dealing with misspellings.

<!--
TODO (rewrite): promote "case, Unicode, accents" to its own subsection with a
worked example. Reference `demos/shared/text.py` for the helper functions used
by chapter demos (`tokenize`, `remove_stopwords`, `pipeline`).
-->

## Sentence segmentation

<!-- source: moved from book/chapters/ch07_retrieval_augmented_generation/8_2_chunking_strategies.md
     Purpose here: (a) required by POS taggers and query analysis, (b) natural
     extension of the abbreviation issues in the naive tokenizer, (c) foundation
     that ch07 chunking builds on. Ch07 keeps chunk composition and references
     this section for the mechanics. -->

Sentence boundary detection may sound trivial, but it is more complex than simply splitting at every period. Abbreviations like "Dr." and "U.K." from the earlier example each contain a period that does not end a sentence. Multi-line dialogue and quoted material create further edge cases. To handle these accurately, libraries provide robust models for sentence tokenization:

- **NLTK's Punkt Sentence Tokenizer**: Punkt uses an unsupervised algorithm that learns how sentences start and end from a training corpus in the target language. It builds internal models of abbreviations, common sentence starters, and frequent boundary patterns. NLTK includes a pre-trained English Punkt model, and you can train custom models for other languages. It is fast and usually very accurate, but it can struggle with complex narrative dialogue where punctuation appears both inside and outside quotation marks.
- **spaCy Sentence Segmentation**: spaCy uses machine learning and deep linguistic parsing to detect sentence boundaries. It supports many languages built in and handles complex sentence structures more consistently than rule-based systems. It is heavier than Punkt, requiring more CPU and GPU resources, but it is well suited for production systems where accuracy and extensibility matter.

Sentence segmentation is a prerequisite for POS tagging (which needs full sentences as input) and for the chunking strategies used in RAG (see [](#chunking-strategies) in the retrieval-augmented generation chapter).

## Word-boundary ambiguity and sub-word tokens

<!-- source: 4_2_tokenization_revisited.md, "There are scenarios where..." block -->

There are scenarios where it is not obvious where a word starts and ends:

- **Scriptio continua** is a writing style without spaces or word separators, often lacking punctuation and sentence boundaries. Prominent examples include Chinese, Japanese, Thai, as well as classical Greek and Latin. Here is an example in Chinese:

    ```
    莎拉波娃现在居住在美国东南部的佛罗里达。

    莎拉波娃  现在   居住  在    美国   东南部     的    佛罗里达
    Sharapova now     lives in       US       southeastern     Florida
    ```

- **Programming identifiers** are a modern variation, where literals cannot contain spaces. Developers use conventions like `QueryParser`, `assertEquals`, `word_tokenize`, and `preserve_line`. Coding assistants can break these literals into meaningful tokens to grasp the developer's intent. Retrieval over source-code corpora needs the same capability.

- **Transcribed spoken language** produces a phoneme stream and then determines word boundaries. In speech, words are not separated; they are joined into a continuous stream of phonemes:

    ```
    ðɪskɔːsˈtiːʧɪzˌmʌltɪˈmiːdiərɪˈtriːvᵊl.

    ðɪs   kɔːs     ˈtiːʧɪz     ˌmʌltɪˈmiːdiə  rɪˈtriːvᵊl.
    This  course   teaches     multimedia     retrieval.
    ```

Two approaches break continuous streams into tokens:

- **Dictionary lookup with a probabilistic model**: The dictionary determines which character sequences can form words and lists all options at the current text position. For example, in "h e s i t a t e" we could extract "hesitate" or "he, sit, ate" or "he's, it, ate". A trained hidden Markov model (or a neural network) resolves the ambiguity and selects the most likely sequence. Language-specific models use rules such as maximum matching (longest sequence in the dictionary) plus language-specific character usage. Common challenges include names of people and brands as well as loan words from other languages, for example English computing terms in German or Thai.

- **Sub-word tokens used directly for retrieval**: The phoneme (or character) stream is divided into overlapping sequences of fixed length, typically three symbols. Using the earlier phoneme example, the stream

    ```
    ðɪskɔːsˈtiːʧɪzˌmʌltɪˈmiːdiərɪˈtriːvᵊl.
    ```

    becomes

    ```
    ðɪs  ɪsk  skɔ  kɔːs  ɔːsˈt  ...  ˌmʌl  ʌlt  ltɪ  tɪˈm  ɪˈmi  ˈmiːd  iːdi  diə  iər  ...
    ```

    Stress symbols in the phoneme stream are combined with the following phoneme, enlarging the symbol vocabulary. To match this with a query, consider a user searching for "multimedia". This query is translated into a phoneme stream and then segmented into sequences of three phonemes:

    ```
    multimedia    ˌmʌltɪˈmiːdiə    ˌmʌl  ʌlt  ltɪ  tɪˈm  ɪˈmi  ˈmiːd  iːdi  diə
    ```

    This creates an 8-token query, and we can employ a standard retrieval method that may consider token proximity. An intriguing outcome of this method is that we do not need to match all sub-sequences to locate relevant spoken text passages. If a non-native speaker mispronounces words, or someone articulates unclearly, the phoneme stream from the spoken text may differ from the one generated by the query. As long as there are sufficient overlaps between the sequences, we can still locate the passage.

    We can apply this method also in situations where word boundaries are identifiable:

    ```
    This course teaches multimedia retrieval.
        thi  his  cou  our  urs  rse  tea  eac  ach  che  hes  mul  ult  lti  tim  ime  med  edi  dia  ret  etr  tri  ...
    ```

    In this scenario, we create sub-sequences only within words, avoiding sub-word tokens spanning two words. An extension of this approach differentiates between sub-sequences at the beginning of a word and those within, by prefixing sub-sequences at the start of words with `#` (or any unused symbol):

    ```
    This course teaches multimedia retrieval.
        #Thi  his  #cou  our  urs  rse  #tea  eac  ach  che  hes  #mul  ult  lti  tim  ime  med  edi  dia  ...
    ```

    We can transform queries the same way. An example to illustrate the advantages:

    ```
    Q = "teach multtimedia"       #tea  eac  ach  #mul  ult  ltt  tti  tim  ime  med  edi  dia
    ```

    Despite a different inflected form for "teach" and a misspelling (double "tt"), 10 out of 12 sub-sequences match those from the sentence above. A retrieval model with partial matching and optional token proximity can locate the relevant passage without stemming or spelling corrections. Modern large language models use a similar approach, discussed in a later chapter.

```{seealso}
Byte Pair Encoding (BPE) and WordPiece tokenization, which industrialise the
sub-word idea for large language models, are covered in the semantic search
chapter. See `_parked_for_semantic_search.md` for the parked material.
```

## Language detection (rule-based sketch)

<!-- source: moved from book/chapters/ch12_video_structural_features/14_2_text_features.md §14.2.1
     Rule-based portion kept here as a normalization prerequisite; the
     Naive Bayes classifier version lives in section 5. -->

Many normalization decisions depend on the language of the input. Which stemmer to apply, which stop-word list to use, which accent rules to normalize with, and later which POS tagger to invoke all require the language to be known. **Language detection** identifies the language of a text or document. This task is straightforward for long documents but becomes challenging for short texts or when many languages must be distinguished automatically.

For longer texts, a small number of rules is often sufficient:

- **Alphabet diversity**: Each language uses a characteristic script. Examples include Latin, Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, Tamil, Bengali, as well as Chinese, Hiragana, and Katakana. Languages often borrow words, leading to a mix of scripts; we can filter out rarely-used alphabets when identifying the primary language.
- **Character diversity**: Some languages have special characters within their alphabet that are typical of their linguistic uniqueness. Diacritical marks and accent symbols in Latin-based scripts, or tonal markers in some Asian languages, add distinctive features to characters. Only a few Latin-based languages use ä, ö, and ü.
- **Stop-word counts**: Evaluating stop-word frequencies can reveal a language. English and French often employ frequent stop words, while others such as Mandarin Chinese rely less on them. Managed stop-word lists let us guess a language by counting how often its stop words occur.
- **Vocabulary counts**: Examining unique words or vocabulary can also help identify the language. Different languages have distinct vocabularies, and comparing word frequencies and diversity supports accurate identification.

For longer texts, these rules quickly identify the language. The method does not easily scale to large numbers of languages: the alphabet and character rules stay simple, but the stop-word lists and vocabularies require large amounts of data per language. For shorter texts or brief phrases, the rules become less effective unless comprehensive vocabularies for all languages are available. A single word or short phrase can exist in multiple languages, and phrases containing loanwords such as "mein computer" (a German phrase with an English loanword) make the problem harder still.

Modern language detectors go beyond these rules and use character-based n-grams together with a Naive Bayes classifier. That formulation is covered in the intent-routing section of this chapter (see [](#intent-routing-and-classification)), because it is the same machinery we use for query intent classification.

<!--
TODO (rewrite): add a compact code snippet with `lingua` or `langdetect` here,
showing the "Bücher von Goethe" case that resolves to German. Keep the deeper
NB math to section 5.
-->

## NLP history (context)

<!--
TODO (rewrite): move the NLP-history block into an optional-reading dropdown of
the form:

```{admonition} From ELIZA to Transformers: a short history of NLP (optional reading)
:class: note dropdown
[three-era narrative with figures 3-1 (ELIZA) and 3-2 (SHRDLU)]
```

Below is the raw source material for that dropdown. Not exam-relevant.
-->

<!-- source: 4_1_introduction.md, three-era history and NLP paradigm shifts -->

The evolution of NLP can be seen in three major paradigm shifts:

- **Symbolic Era (1950s-1980s)**: Rule-based systems using manually crafted linguistic rules.
- **Statistical Era (1980s-2010s)**: Data-driven approaches using probabilistic models and machine learning.
- **Neural Era (2010s-present)**: Deep learning models with attention mechanisms and large-scale pre-training.

### Symbolic Era (1950s-1980s)

The symbolic approach was based on the belief that human-like language understanding could be achieved by applying formal rules and structures. Researchers tried to encode knowledge about the world in a form computers could process using symbolic methods. The approach relied heavily on pattern matching and substitution techniques to process natural language input.

- **ELIZA (1964)**: Joseph Weizenbaum's ELIZA became one of the best-known early NLP programs. It imitated human conversation by answering typed input in everyday language. The program was based on Rogerian psychotherapy, in which the therapist helps patients reflect by turning their words into questions. Despite relying on simple pattern matching, many users developed surprisingly strong emotional attachments to ELIZA. <!-- insert {figure} images/figure_3_1.png as fig-eliza-chatbot-terminal-session here -->

- **SHRDLU (1968)**: Terry Winograd's SHRDLU was a major advance in rule-based natural language processing. SHRDLU could understand and carry out natural language commands in a simple "blocks world" environment, showing that computers can process complex instructions. <!-- insert {figure} images/figure_3_2.png as fig-shrdlu-dialogue-session here; the raw dialogue transcript is inside the figure, no need to duplicate in prose -->

However, the limits of rule-based systems became clear during this period. Natural language contains pervasive ambiguity at many levels. For example, the sentence "I saw the man with the telescope" has several valid interpretations, and rule-based systems had no principled way to choose between them.

Human language cannot be fully formalized. Idioms, metaphors, novel constructions, and the steady arrival of new words mean no fixed set of rules can cover everything. Systems can fail badly when they encounter inputs outside their expected range.

### Statistical Era (1980s-2010s)

The Statistical Era marked a major shift from hand-crafted, rule-based systems to data-driven probabilistic models. Rather than writing linguistic rules by hand, researchers used large text collections and statistical algorithms to automatically learn language patterns. The idea was that simple models trained on vast amounts of data would outperform complex, linguistically informed systems.

Key technologies were Hidden Markov Models (HMMs) and n-gram language models. HMMs drove major advances in part-of-speech tagging, speech recognition, and named entity recognition. N-gram models captured local word patterns and powered early machine translation and text generation. Annotated corpora like the Brown Corpus and the Penn Treebank were essential because they provided standard data to train and test statistical methods.

Probabilistic Context-Free Grammars (PCFGs) improved syntactic parsing by assigning probabilities to grammar rules, letting systems rank and choose among possible parse trees. Support Vector Machines (SVMs) changed text classification by performing well with high-dimensional data, helping tasks such as spam detection and sentiment analysis.

Even with its successes, the statistical approach had clear limits. Models needed large labeled datasets, a lot of feature engineering, and they struggled to grasp meaning and handle long-range dependencies. When progress stalled in the early 2000s, these problems paved the way for the neural revolution.

The Statistical Era built the foundation for modern NLP: a data-driven mindset, probabilistic thinking, and evaluation-focused methods. These principles enabled deep learning and today's powerful language models.

Sample output from a tri-gram model trained on a single book:

> From time to time I heard a cry of fire, it will be the man who is in a few minutes of his own delicate and finely adjusted temperament was to be a man who is ...

### Neural Era (2010s-present)

The Neural Era built on decades of research in symbolic and statistical methods but achieved unprecedented progress by enabling models to learn language directly from data.

The shift began with Yoshua Bengio's (2003) neural probabilistic language model. It introduced the idea of representing words as continuous vectors. This broke the limits of rigid symbolic models and showed that neural networks can learn word context.

A major advance came in 2013 with Word2Vec, developed by Tomas Mikolov and his team at Google. Word2Vec efficiently produced word embeddings, dense vector representations that capture both semantic and syntactic similarity. Soon after, GloVe (2014) from Stanford offered a complementary approach, making vector representations a core part of NLP systems.

The next wave of progress focused on sequence modeling, with recurrent neural networks (RNNs) and their improved versions, LSTMs and GRUs, better at handling sequential dependencies in language. These models powered major advances in translation, speech recognition, and text generation during the 2010s.

In 2017 the Transformer architecture, introduced by Vaswani and colleagues in "Attention is All You Need", created a real shift in the field. By replacing recurrence with self-attention, Transformers enable parallel processing of whole sequences and better capture long-range dependencies.

This era marks a turning point in NLP's evolution, as deep learning models overcome earlier limits and bring machines closer than ever to understanding and producing human language with nuance, coherence, and creativity.
