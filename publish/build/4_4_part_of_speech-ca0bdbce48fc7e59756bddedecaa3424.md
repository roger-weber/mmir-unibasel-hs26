# Part of Speech

Sentences consist of words belonging to various grammatical classes, known as part-of-speech (POS) categories, which share similar grammatical characteristics. In English, common parts of speech include noun, verb, adjective, adverb, pronoun, conjunction, interjection, numeral, article, and determiner. POS tagging is a method for assigning the appropriate class to a word based on its position and function within a sentence. Note that the mapping from a word to its POS tag is not always deterministic; for example, the word “run” can function as both a noun and a verb.

In information retrieval, POS tags can substitute words for stop word filtering. For example, we keep “IT” if it is a noun but remove it if it is a pronoun. In query processing, we analyze a query's structure, especially in questions. This analysis helps to extract details, and to query directly against structured metadata instead of relying solely on keyword–based search. For instance, the question “Who is Albert Einstein?” can be split into a query word “who”, a verb “is”, and a name “Albert Einstein”. Using POS tagging, we can infer that the user seeks a person named “Albert Einstein”. Rather than a full-text search, we query a ‘people’ database for structured information.

A treebank is a text corpus where sentences are parsed and annotated to depict their syntactic structure. These trees also convey information about word-to-word grammatical relationships and hierarchical sentence composition. Treebanks offer labeled data to aid algorithms in learning grammatical structures and associating POS tags with words in sentences.

A closely related issue is Named Entity Recognition (NER), as seen with “Albert Einstein” in the previous example. Typical categories for NER classification include names, locations, organizations, and currencies. Typically, NER terms are not found in dictionaries, and their frequencies and occurrences vary over time. Identifying a term (or n-gram) as an NER helps us to infer the user's intent. In some cases, NER searches are common, such as in online shops, where proper indexing is crucial. Instead of learning valuable bi-grams to add to the dictionary, we can generate them using simple rules based on NER tags. For instance, if we encounter two names (two consecutive NER-people tags), we can index them both individually and as a bi-gram (or tri-gram if three consecutive names appear). POS tags provide additional insights into the roles of names, as seen in sentences like “How to drive from Basel to Luzern”, “I am in Basel and want to drive to Luzern”, or “How to drive to Luzern from Basel”.

Let’s look at a few common implementation for POS and NER tagging, and the algorithms behind them.

Rule-based POS tagging: These algorithms use predefined rules based on context to assign POS tags to words. In machine learning models, simplified versions of these rules are often integrated as fallback or pre-filtering mechanisms to enhance accuracy for common patterns. For instance, in English:

    - If a word ends with the suffix “ing” and the stem (without “ing”) is a recognized verb, assign the tag VERB.

  - This rule effectively handles many regular gerund forms like “hearing” and “walking”, correctly ignores “thing” and “king”, but can not handle “running” or “swimming” correctly (though it is straightforward to add a secondary rule for such cases). While simple and fast in processing, each language requires a fresh set of rules.

Stochastic POS Tagging: A hidden Markov model (HMM), is a probabilistic technique applied in multiple domains. It is a graphical representation comprising states, their probabilistic transitions, and emitted symbols at each state. It consists of four essential components:

  - Hidden States: These represent unobservable internal states that capture the system's behavior. They form a Markov chain, where transitions between states depend solely on the current state and are independent of previous states.

  - Observations: These are symbols or events linked to hidden states, emitted during transitions between one hidden state and another. However, uncertainty often exists regarding which symbol is emitted because each hidden state can produce multiple outcomes (according to a probability distribution), and an observable symbol or event can be emitted by multiple hidden states.

  - Transition Probabilities: Usually represented as a transition matrix, these probabilities indicate the likelihood of moving from one hidden state to another. They are learned from training data, often using maximum likelihood estimation. In the transition matrix, rows correspond to the current state, and columns correspond to the next state.

  - Emission Probabilities: These are probabilities associated with each hidden state, indicating the likelihood of generating specific observations or emissions when in that state. These probabilities help define how likely it is for a hidden state to produce particular observable outcomes. Emission probabilities are often represented as an emission matrix or emission probability distribution, with rows corresponding to the state, and columns representing the observations.

  - For simplicity, let's consider three POS tags: nouns, verbs, and others. The structure of a very simple Hidden Markov Model (HMM) for POS tagging is as follows:

    - Each POS tag is a hidden state, including a start state that marks the beginning of sentence processing. The transition matrix specifies the probability of transitioning between hidden states. These probabilities can be learned by using maximum likelihood estimation based on state transitions in the training data. For a transition probability from state $s_{i}$ to $s_{i+1}$, denoted as $P(s_{i+1}|s_{i})$, and the count of such state transitions in the training data as $C(s_{i}, s_{i+1})$, the maximum likelihood estimate for $P(s_{i+1}|s_{i}),  ∀1\leq i<m$ is given by:

    - Where $m$ represents the total number of states. The smoothing variant serves to avoid zero-values in the transition matrix, which can cause numerical problems when performing calculations with logarithms in the Viterbi algorithm. Additionally, it enables the model to handle transitions that were not observed in the training data but are encountered when processing new sentences. Similarly, we compute the maximum likelihood estimates for the probability $P(t_{k}|s_{i})$ of term $t_{k}$ being emitted at state $s_{i}$ based on counts denoted as $C(s_{i}, t_{k})$:

noun

verb

other

<start>

term 1

term 1

term 1

$P(s_{i+1}|s_{i})=\frac{C(s_{i},s_{i+1})}{\sum_{j=1}^{m}C(s_{i},s_{j})}                                                                         P(s_{i+1}|s_{i})=\frac{C\left(s_{i},s_{i+1}\right)+\varepsilon }{\sum_{j=1}^{m}C\left(s_{i},s_{j}\right)+m∙\varepsilon }$

with smoothing (small $\varepsilon $):

$P(t_{k}|s_{i})=\frac{C\left(s_{i},t_{k}\right)}{\sum_{k=1}^{n}C\left(s_{i},t_{k}\right)}                                                                         P(t_{k}|s_{i})=\frac{C\left(s_{i},t_{k}\right)+\varepsilon }{\sum_{k=1}^{n}C\left(s_{i},s_{j}\right)+m∙\varepsilon }$

with smoothing (small $\varepsilon $):

  - Using the trained Hidden Markov Model, we can apply the Viterbi Algorithm to determine the most probable state transitions for a given sentence's observed term sequence. These states can then be linked to the terms in the sequence to assign them the corresponding POS tags.

Transformation-based POS tagging builds upon rule-based POS tagging by iteratively correcting errors. It starts with an initial, simple, and hand-crafted rule-based tagging, which is compared to training data to find errors. Transformation rules, either learned from training data or manually created based on observed patterns, are then used to correct these errors. This error identification and rule application process is repeated until a maximum number of iterations is reached or no more errors are found. During sentence analysis, the initial tagging and all transformation rules are consistently applied to generate the final POS tagging for the sequence.

  - Example: After applying the “ing” rule for verb detection, we notice that in some cases, the gerund form of a verb can also function as a noun. For example, in the sentence “The swimming was nice”, “swimming” is a noun but was initially tagged as a verb. To address this, we introduce a straightforward transformation rule: verbs that follow articles or adjectives should be reclassified as nouns.

Deep learning POS tagging, on the other hand, employs neural networks to automatically learn and predict POS tags for words in a text. Instead of relying on hand-crafted rules or transformations, deep learning models leverage large datasets to capture complex patterns and relationships between words and their corresponding POS tags. These models use layers of neurons and sophisticated architectures to process sequential data, making them particularly effective for tasks like POS tagging, where the order of words in a sentence is crucial. Deep learning approaches have achieved remarkable accuracy in various natural language processing tasks, including POS tagging, and continue to be a cornerstone of modern NLP research and applications.

  - Modern taggers utilize a transformer-based architecture. In this architecture, the input sequence represents the sentence and is transformed into embeddings and including positional encoding. The transformer architecture then maps this sequence to a POS (Part-of-Speech) sequence. Training these models with POS-tagged sentences enables the neural network to learn its parameters. These specialized models are optimized for POS-tagging and are not suitable for other tasks.

All these approaches have a common limitation: they usually support only one language. While multi-language POS-tagging is possible, it is more advisable to use a language-optimized model when the language of the sentence is known. Most of these approaches achieve high accuracy, often exceeding 99%, across a wide range of sentences.

NLTK employs a transformation-based POS tagger. The process starts with sentence tokenization, followed by a dedicated classifier that predicts the POS tags for the tokens. It is crucial to input the entire sentence into the POS tagger because words that can assume various grammatical roles might be misclassified otherwise (e.g., "running" as a gerund form and "running" as a noun). By considering the complete sentence context, NLTK can provide more accurate POS tags. Here is the Python code to obtain and list the tokens:

    - tokens = nltk.word_tokenize(text_en)

    - # tagset = None for standard, or tagset = 'universal'

    - tagged_tokens = nltk.pos_tag(tokens, tagset=tagset)

    - ner_chunks = [chunk for chunk in nltk.ne_chunk(tagged_tokens) if hasattr(chunk,'label')]

  - nltk supports different tag sets. The standard tag set is more detailed and depicted on the left side. The universal tag set focuses on a few main categories as shown on the right side. The standard set is often used for deep NLP tasks to construct parse trees which allows the extraction of context and the transformation of sentences.

WH-words are: where, what, which, when, …

with NLTK, use nltk.help.upenn_tagset()

Proper nouns are specific people, places, things.

  - Named Entity Recognition (NER) is a transformation based on POS tagging. It involves collapsing individual tokens or groups of tokens into a single named entity using an entity database. These entities can be names of people, product brands, companies, non-governmental organizations, locations, currencies, and more. In NLTK, the POS tagged tokens are processed with the ne_chunk function to obtain NER tags. These NER tags enable the extraction of valuable contextual information, such as person names, locations, or product names. When applied to a query in question form, it helps to better understand the user's intent and to optimize search results:

    - When searching for a name, such as “Who is Albert Einstein?”, prioritize web pages like Wikipedia, IMDb, Musicbrainz, and sports sites that users commonly visit to gather information about prominent individuals.

    - When a query includes time, date, or age information, like “Who won the F1 race last weekend?”, enhance the visibility of news articles or utilize the extracted time/date to conduct a temporal range query.

    - When a query mentions a location, such as “What to do in Basel?”, prioritize regional content and provide a map of the named location to assist users with navigation.

    - When a query involves product brands, like “Where is the latest iPhone available?”, boost advertisements and shopping sites, conduct a product search to offer a "best price" view, or provide recommendations for buyers.

  - Chunking is a versatile technique that involves creating non-overlapping phrases using a defined grammar. For example, the grammar NP: {<DT>?<JJ>*<NN>} combines articles, adjectives, and nouns into a single group, facilitating the understanding of term relationships for more effective searching. For instance, “a red car” would form a parse tree that links the adjective “red” with the noun “car”. More intricate grammars enable the dissection of sentences into smaller components, allowing for reasoning about context and sentence meaning through additional dependency information between terms.

    - A good online demo with deep NLP capabilities is available here: https://corenlp.run

  - To analyze sentence structure, we require a grammar similar to that used in programming languages. Unlike programming languages, natural language grammar is imperfect and riddled with ambiguities, making it challenging for both humans and machines to grasp context. Grammar alone cannot resolve these ambiguities; context plays a crucial role in their resolution.

spaCy uses a neural network to predict POS and NER tags, however with different tag names than nltk. The left table below shows the POS tags, and right table the NER tags. The code is also simple:

  - nlp_spacy = spacy.load('en_core_web_sm’)

  - tokens = nlp_spacy(text)

  - tagged_tokens = [(t.text, t.pos_) for t in tokens]

  - ner_entities = [(e.text, e.label_) for e in tokens.ents]

  - spaCy also offers support for various languages. Refer to their documentation to choose the suitable model.

Finally, the transformers library offers two pipelines for extracting POS and NER tags using trained neural networks. It also supports fine-tuning of NER tags to adapt to specific document collections and scenarios. Since transformers models are continuously advancing, we present the general code structure here and recommend visiting the Hugging Face website to access the latest models for these pipelines:

    - nlp_bert = pipeline("token-classification", 		     model="vblagoje/bert-english-uncased-finetuned-pos",		     aggregation_strategy="max")

    - tokens = nlp_bert(text)

    - tagged_tokens =  [(token['word'], token['entity_group']) for token in tokens]

    - nlp_bert = pipeline("ner", 		     model="dslim/bert-base-NER", 		     aggregation_strategy="max")

    - tokens = nlp_bert(text_ner)

    - ner_entities = [(t['word'], t['entity_group']) for t in tokens]

  - The aggregation strategy combines tokens, typically sub-words, to reconstruct words or n-grams, especially for names. Without an aggregation strategy, the model assigns entity values to individual model tokens, potentially splitting words into smaller tokens without grouping them into entities.

  - Please consult the model description to determine the specific POS and NER tags used, as these may vary between different models. In addition to the English version used here, there are models available for other languages as well.

Example: The table on the left (blue) shows the POS tags for the methods discussed on an English sentence (punctuation and repeating words were removed). On the right side (green), we see the NER tags for the sentence: “Jack Higgins, wearing Nike shoes, deposits £50,000 with BestBank in London at Jermyn Street close to Piccadilly Circus”. And finally, the lower, right tables (orange) list the POS tags and their frequency in an English novel.
