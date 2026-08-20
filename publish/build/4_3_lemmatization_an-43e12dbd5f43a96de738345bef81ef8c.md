# Lemmatization and Linguistic Transformation

Stemming reduces words in the input to their root, ensuring variants match during search. For example, ‘houses’ in a document can match a query with ‘house’. Stemming is language-dependent, but typically, removing prefixes and suffixes is effective for most languages. Words that can undergo significant inflections, like 'go' and 'went,' present more challenges. We distinguish different types of stemming algorithms:

  - Rule-based stemmers use rules to transform words to their stems, which may not always be linguistically correct but are designed to match variants of the same root. In text retrieval, displaying these stems to users is not necessary; they are only used for quick lookup with inverted files.

  - Dictionary-based stemmers use a small set of rules for regular inflections and rely on a dictionary and irregular inflection list to find the correct linguistic stem. In text retrieval, this improves the success for matching word variants, especially in cases with strong inflections like 'go' and 'went’.

In a previous chapter, we learned about the Porter algorithm, a basic English stemmer that creates pseudo-stems to unify word variations. The Lancaster stemmer is another rule-based stemmer for English. It aggressively cuts off word endings (suffixes), which can lead to very short stems. It is faster than other algorithms and suitable for general English text processing.

In various retrieval situations, handling diverse languages is common. Applying Porter or Lancaster stemmers to non-English text does not work. As a solution, Martin Porter introduced the Snowball framework to create rule-based stemmers for multiple languages similar to Porter and Lancaster. This framework features its own rule definition language, and can generate code for different programming languages. The result is still a pseudo-stem, and in languages with strong inflections, the stem can vary due to gender, tense, or case changes. These algorithms are highly efficient and can operate in any environment without requiring a large dictionary.

An improvement over the rule based stemmers are dictionary based stemmers such as provided by WordNet and spaCy. They consist of three parts:

  - a simple rule-based stemmer for regular inflections (e.g., ‘-ing’, ‘-ed’)

  - an exception list for irregular inflections

  - a dictionary of all possible stems of the language

The dictionary based algorithms works as follows:

  - Retrieve a part-of-speech (POS) tag for the current word. This is typically done during tokenization and considers the broader context to determine the correct tag (e.g., noun, verb, adjective, punctuation). For example, whether ‘run’ is a noun or a verb depends on its context.

  - Search for the word in the dictionary; if it is found, then the word is not inflected, we can return it as its own stem.

  - Search for the word in the exception list for its POS tag (see tables below for examples); if it is found, we can return the stem as given in the list.

  - Apply rules based on the POS tag to shorten regularly inflected forms using their suffixes. The table on the right shows some English examples. Use each applicable rule and check the dictionary; if the word is found, return the form from the dictionary.

  - If no dictionary entry is found, return the word as its own stem. This can occur with names, misspelled words, or loanwords (words from another language, like English words in German).

Type 	Suffix 	Ending

NOUN 	s

NOUN	ses	s

NOUN	xes	x

NOUN	zes	z

NOUN	ches	ch

NOUN	shes	sh

NOUN	men	man

NOUN	ies	y

VERB	s

VERB	ies	y

VERB	es	e

VERB	es

VERB	ed	e

VERB	ed

VERB	ing	e

VERB	ing

ADJ	er

ADJ	est

ADJ	er	e

ADJ	est	e

adj.exc (1500):

  - ...

  - stagiest 	stagy

  - stalkier 	stalky

  - stalkiest 	stalky

  - stapler 	stapler

  - starchier 	starchy

  - starchiest 	starchy

  - starer 	starer

  - starest 	starest

  - starrier 	starry

  - starriest 	starry

  - statelier 	stately

  - stateliest 	stately

  - ...

noun.exc (2000):

  - ...

  - neuromata 	neuroma

  - neuroptera	neuropteron

  - neuroses 	neurosis

  - nevi 	nevus

  - nibelungen 	nibelung

  - nidi 	nidus

  - nielli 	niello

  - nilgai 	nilgai

  - nimbi 	nimbus

  - nimbostrati 	nimbostratus

  - noctilucae 	noctiluca

  - ...

verb.exc (2400):

  - ...

  - ate 	eat

  - atrophied 	atrophy

  - averred 	aver

  - averring 	aver

  - awoke 	awake

  - awoken 	awake

  - babied 	baby

  - baby-sat 	baby-sit

  - baby-sitting 	baby-sit

  - back-pedalled 	back-pedal

  - back-pedalling 	back-pedal

  - backbit 	backbite

  - ...

nltk and spaCy provide multiple stemmers for text processing in different languages. The code on the right demonstrates how to begin using these stemmers.

  - English Example: The table below displays stemming outcomes for Porter, Lancaster, Snowball, WordNet, and spaCy when applied to English text. The table excludes terms that yield the same stem across all stemmers. It also include the part-of-speech tag (pos) that was used for WordNet and spaCy.

  - import nltk

  - from nltk.corpus import wordnet

  - import spacy

  - # building a stemmer

  - porter = nltk.PorterStemmer()

  - lancaster = nltk.LancasterStemmer()

  - snowball = nltk.SnowballStemmer("english")

  - wordnet = nltk.WordNetLemmatizer()

  - spacy = spacy.load('en_core_web_sm')

  - # applying it

  - porter.stem('discovered')

  - lancaster.stem('discovered')

  - snowball.stem('discovered')

  - wordnet.lemmatize('discovered', 'v')

  - # spacy processes full text sequence, # not just one word

for token in spacy(‘I have discovered it'):

print(token.text, token.lemma_)

  - English Example (cont’d): The Snowball and Porter algorithm yield very similar results as they mostly rely on the same rules, with Snowball being a slightly revised version of Porter. In contrast, Lancaster is more aggressive in removing suffixes, often resulting in overly short stems that may collide with unrelated words, especially when they are short themselves (e.g., ‘one’ and ‘only’ both reduced to ‘on’). WordNet and spaCy produce similar results, but their stems differ from those of the other algorithms. Notably, all WordNet and spaCy stems are linguistically correct (‘bottle’ vs. ‘bottle’). In text retrieval, stem correctness matters less than ensuring variants map to the same stem and thus the same token ID in the index. This is evident in examples like ‘had’ and ‘have’ which the rule-based algorithms map to different stems, while the dictionary-based algorithms map them to the same base form ‘have’. This enhances the search engine's ability to match query variants with those found in documents.

  - German Example: Snowball and spaCy are good options for German stemming, allowing us to compare Snowball's rule-based approach with spaCy's dictionary-based approach. The results are displayed on the right. We observe similar differences between rule-based (Snowball) and dictionary-based (spaCy) stemming. Additionally, Snowball maps special characters to a base character set and converts text to lowercase. It also handles cases like ‘ae’ → ‘a’ if the text doesn't use ‘ä’ correctly. spaCy corrects casing only if a word starts a sentence and would normally be in lowercase. Snowball's results are acceptable for text retrieval, but spaCy performs significantly better in identifying the true linguistic stem and matching strongly inflected variants (consider ‘beschloß’ and ‘beschließen’).

  - French Example: For French stemming, Snowball and spaCy are good options, and the table on the right compares their performance with a French text. Unlike in German, French Snowball retains accented characters but still converts words to lowercase, while spaCy preserves casing for names. We observe similar differences between the rule-based (Snowball) and dictionary-based (spaCy) approaches as seen in the German example. In this French example, the ability to map various inflected forms to the same stem is even more noticeable, as Snowball often assigns different stems to different inflected forms of the same root (e.g., ‘aperçu’ and ‘aperçut’, ‘avait’ and ‘avaient’).

In linguistics, compounds are words created by combining two or more base words, occasionally using binding syllables (e.g., ‘Liebeslied’) or characters (e.g., ‘must-have’). While most languages support basic compound formation to create new words (e.g., ‘smalltalk’), languages such as German and Finnish permit the formation of arbitrary long compounds. Let's examine a few examples:

  - Finnish:

      - kolmivaihe­kilowattitunti­mittari	en: electricity meter

      - atomiydinenergiareaktorigeneraattorilauhduttajaturbiiniratasvaihde	en: atomic nuclear energy reactor generator condenser turbine cogwheel stage

      - rautatieasema 	en: railway station

  - German:

      - Wolkenkratzer	en: skyscraper

      - Rinderkennzeichnungs- und Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz (German (law in Mecklenburg-Vorpommern, 1999-2013)	en: cattle marking and beef labeling supervision duties delegation law

      - Stacheldraht	en: barbed wire

  - Dutch

      - arbeids­ongeschiktheids­verzekering 	en: disability insurance

      - rioolwater­zuiverings­installatie	en: sewage treatment plant

      - doorgroei­mogelijkheden	en: possibilities for advancement

We can classify compounds as either endocentric or exocentric.

  - Endocentric compounds derive their meaning from their constituent parts. They have a ‘head’ that imparts both semantic and syntactic attributes to the compound, while the other elements modify and refine its meaning. For instance, in ‘sunglasses’, ‘glasses’ serves as the head, and ‘sun’ acts as the modifier.

  - Exocentric compounds do not derive their meaning from their constituent parts and may even ignore the lexical class of their individual elements (e.g., ‘must-have’ is a noun, not a verb). In the word ‘skyscraper’, neither ‘sky’ nor ‘scraper’ acts as the head, and the term names an entirely different object (in this case, a type of building).

Consider a compound word like 'Rindfleischetikettierungsberwachungsaufgabenbertragungsgesetz'. The first problem is spelling it correctly, which makes it hard to find in document titles when users make spelling mistakes in their query. That may be why the law was abandoned. Another issue is that we must list all the constituent parts to find the document. It would be more user friendly to allow a partial query such as 'Rindfleisch Etikettierung Gesetz'. Unfortunately, the retrieval models we have discussed so far do not support partial term queries against the vocabulary or using such matches for actual searches.

The recommended approach is to split compounds into their parts and include both the parts and the full compound as tokens in the document. For example, the German word 'Abfalleimer' becomes 'Abfall',' Eimer', and 'Abfalleimer'. This helps match a wider range of queries and works well for endocentric compounds, where the parts reflect the compound's meaning. It is less effective for exocentric compounds such as 'skyscraper' or the German 'Wolkenkratzer'. Splitting 'skyscraper' into 'sky' and 'scraper', or 'Wolkenkratzer' into 'Wolke' and 'Kratzer', adds incorrect semantics to the document. Whether the benefit from splitting endocentric compounds outweighs the harm from creating wrong meanings for exocentric ones depends on the retrieval scenario.

We present two methods for automatically splitting compounds. Both use rule based or morphological analysis to find possible splits of a term. The details vary by language. Let us look at examples:

  - In English, we can split compounds using hyphens and syllables in accordance with English hyphenation rules. For example: ‘must-have’ becomes ‘must’ and ‘have’; and ‘skyscraper’ becomes ‘sky’, ‘scrap’, and ‘er’.

  - In German, we split on syllables following German hyphenation rules. For example, ‘Wolkenkratzer’ becomes ‘wol’, ‘ken’, ‘krat’, ’zer’; and ‘Schifffahrtskapitän’ becomes ‘Schiff’, (‘fahrt’, ‘fahrts’), ‘ka’, ‘pi’, ‘tän’. Note that in that last example ‘s’ is a binding letter for compound generation and we have to test with both pieces ‘fahrt’ and ‘fahrts’.

  - As a next step, we produce all possible combinations of such splits:

    - skyscraper   		 (sky, scrap, er), (skyscrap, er), (sky, scraper)

    - wolkenkratzer		 (wol, ken, krat, zer), (wolken, krat, zer), (wol, kenkrat, zer), (wol, ken, kratzer),			     (wolken, kratzer), (wolkenkrat, zer), (wol, kenkratzer)

    - Schifffahrtskapitän	 (Schiff, fahrt, ka, pi, tän), (Schiff, fahrts, ka, pi, tän), (Schifffahrt, ka, pi, tän),			      (Schifffharts, ka, pi, tän), (Schiff, fahrtka, pi, tän), (Schiff, fahrtska, pi, tän),			      …  (Schifffhart, kapitän), (Schifffharts, kapitän)

To find valid splits, we start by discarding any splits that contain components not found in our vocabulary or dictionary. When multiple options remain, we determine the best split based on the frequency of the components. Let $𝕊$ represent the set of all possible splits, and let $S=\left\{p_{i}\right\}$ represent all the individual components of split option $S\in 𝕊$. We calculate $tf(p_{i})$ as the number of times piece $p_{i}$ appears in the corpus (or is provided by the dictionary), and $N$ represents the total number of tokens in the corpus (or as given by the dictionary):

  - In simpler terms, we choose the split with the highest average log-frequency values for its components. This indicates the most probable way to combine the parts into a compound.

When analyzing text, we encounter homonyms and synonyms. A homonym is a word spelled the same as another word but with a different meaning and sometimes a different pronunciation. For example, 'lead' can mean to guide or a metal. A synonym is a different word with a similar or nearly identical meaning, often used to avoid repetition, for example 'big' and 'large'. Next we will examine how they affect text retrieval and ways to handle them.

  - Synonyms are commonly used to add variety to written text. However, this can affect the retrieval engine's ability to match query terms with those in the document. For instance, if the document contains ‘purchase’, a query with ‘buy’ or ‘acquire’ can not match it due to the different token forms. There are two main alternatives to address this. First, synonym expansion involves tokenizing the document (and/or the query) and expanding tokens using predefined synonym lists. Second, as discussed later in this course, word embeddings can be used to map terms into a high-dimensional, sparse space, considering relationships between words.

  - Handling homonyms involves analyzing the context to clarify the intended meaning. In straightforward cases, part-of-speech tags can distinguish between verb and noun forms (e.g., ‘lead’ as a guide or as a metal). More advanced solutions use machine learning models to determine the context accurately or analyze grammatical structures for context. When a query contains a homonym, we can either select the most common meaning or present the user with individual results for each potential interpretations. For example, the word ‘bank’ has several meanings (sloping land by water, financial institution). We can seek user feedback for the correct interpretation or offer two result options with synonym expansion for both possible meanings.

[MATH_ERROR]

Another common word relationship are hypernyms and hyponyms. A hypernym has a broader, more general meaning and is often seen as the higher-level category among words. In contrast, a hyponym has a narrower, more specific meaning and is typically viewed as the lower-level category. For instance, ‘animal’ is a hypernym (a more general class) related to the hyponyms ‘cat’ and ‘dog’ which represent more specific types of animals. Words can be hypernyms and hyponyms at the same time. A mammal is hypernym for cat, but a hyponym for animal.

  - Faceted search enables users to explore search results by expanding or narrowing categories using hypernym/hyponym relationships. For example, in an image search for ‘animals’, users can drill down to more specific types like ‘cats’ and ‘dogs’. Conversely, if a query is too specific and yields few results, users can quickly broaden the search using presented hypernym hierarchies.

  - Automatically expand queries with hypernyms and hyponyms to broaden the search. We can assign weights to the original term, its hypernyms (with less weight), and its hyponyms (with the same or less weight) to incorporate term relationships into the search process.

  - Relevance ranking considers hypernym/hyponym relationships to evaluate document relevance, even when the query term is absent. This is similar to query expansion, but the distinction lies in where the expansion occurs. In query expansion, we submit a longer query with weighted hypernyms and hyponyms. In relevance ranking, we retain the user's original query and adjust scoring functions to account for hypernyms and hyponyms.

The WordNet website offers an online demo for in-depth exploration of synonyms, homonyms, hypernyms, and hyponyms in English. You can access WordNet data through nltk.corpus.wordnet.synsets(word), which returns synsets providing functions to access synonyms, homonyms, hypernyms, meronyms, and various other relationships. Visit http://wordnetweb.princeton.edu/perl/webwn  for the WordNet online demo.

The last discussion in this section on tokenization considers spelling mistakes and how to treat them.  Typically, we employ a spellchecker to replace words not found in the dictionary. During document indexing, we retain the original misspelled version, and add the auto-corrected version(s) to the index.  In queries, we can expand the query with auto-corrected version(s) or suggest alternative queries if the misspelled query yields insufficient results (“did you mean?”). Spelling mistakes, especially in names, are common but can be challenging to differentiate from intentional variations. For instance, the name “Britney” has various alternative forms such as “Britni”, “Brittney”, “Britnee”, “Britneigh”, “Britnie” and many more. If we would only use auto-corrected versions, we may not find these alternative forms.
