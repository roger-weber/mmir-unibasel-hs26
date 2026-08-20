# Tokenization Revisited

In the previous chapter, we divided the text into parts and employed a straightforward tokenization method. A token is separated by non-word characters. Here is a straightforward Python implementation:

    - def word_tokenize(text: str) -> list[str]:

    - text = re.sub(r'[^\w\-]+', ' ', text)

    - return [token for token in text.split(' ') if token]

  - In this basic scenario, any string of characters that isn't a Unicode letter, number, underscore, or hyphen gets substituted with a single space. Afterward, we break the text at spaces to create a token list. This method usually functions effectively but has certain limitations. Take a look at the sentence below and the resulting split on the right:

  - I buy my parents' 10% of U.K. startup for $1.4 billion. Dr. Watson's cat called Mrs. Hersley and it was w.r.o.n.g., more to come ...

  - While it is an artificial and nonsensical sentence, it highlights some of the weaknesses:

  - The possessive “’s” (also: parents “’”) is omitted, resulting in a single “s” token. This outcome can be advantageous or disadvantageous, depending on the task. It's beneficial for retrieval since it enables the merging of “Watson” and “Watson’s” allowing users to find the name without testing alternative written forms. However, it becomes problematic for sentence analysis as it breaks the link between “Watson” and “cat”. The typical retrieval approach involves removing possessive forms and single-letter as well as non-alphabetic terms.

  - Numbers function well when they are small, positive integers, but tokenization struggles with percentages, currencies, and floating-point numbers, among other cases not covered here. In the context of retrieval and NLP, numbers are often disregarded or entirely removed. However, in generative AI, the language model may need to generate an answer with the accurate dollar amount from this sentence.

  - Abbreviations like "U.K.," "Dr.," "Mrs.," and the artificial "w.r.o.n.g." are not accurately identified. Abbreviations with multiple dots are treated as separate terms, with all tokens lacking the final dot that signifies an abbreviation. Consequently, searching for "U.K." is not feasible.

  - Interpunctuation is absent. This is beneficial for retrieval but restricts sentence analysis for context and word relationships.

Modern word-based tokenizers are available in the nltk and spaCy libraries. The outcome for the same sentence is displayed on the right. They closely match each other, with the exception of the artificial abbreviation "w.r.o.n.g." which spaCy's neural model finds challenging. A short comparison to the previous page's basic method:

  - “’s” and “’” possessives are treated as terms. In retrieval tasks, they can be filtered out, while in NLP tasks, they aid sentence structure analysis.

  - Floating-point numbers are now recognized correctly, including negative numbers (not shown). Percentages and currency symbols are split into individual tokens, preserving this information compared to the previous method.

  - Abbreviations are accurately identified and represented as single tokens. Both nltk and spaCy employ machine learning to detect common abbreviations.

  - Interpunctuation is fully retained. In retrieval tasks, it can be filtered out, whereas in NLP tasks, it helps to analyze sentence structure.

  - Both packages offer support for language-specific peculiarities, such as French abbreviations. Refer to their documentation for details on enabling multi-lingual tokenization.

Tokenization for Retrieval: The tokens displayed on the right are well-suited for NLP and will use them later for part-of-speech tagging. However, for retrieval tasks, many of these tokens are unnecessary as they do not provide additional information. To create a token list for retrieval scenarios, we can undertake the following cleanup actions:

  - Remove short tokens, like single-letter ones, as they lack specific content description.

  - Exclude non-word tokens, such as numbers and special characters, except for words with hyphens and abbreviations with dots. This also removes tokens from possessive forms.

  - Optionally, convert Unicode characters (e.g., accents) to their closest ASCII equivalents, e.g., “Zürich” to “Zurich”. This can reduce vocabulary size (from 16/32 bits to 8 bits) and simplify matching between queries and documents, especially when users lack easy access to specific Unicode letters (e.g., "słychać" with characters not found on the keyboard).

  - Optionally, convert tokens to lowercase or apply case conversion to their standardized form. This is useful for scenarios like sentence beginnings with capitalized words, title case usage, or dealing with misspellings.

There are scenarios where it is not obvious where a word starts and ends:

  - Scriptio continua is a writing style without spaces or word separators, often lacking punctuation and sentence boundaries. Prominent examples include Chinese, Japanese, Thai, as well as classical Greek and Latin. Here is an example in Chinese:

        - 莎拉波娃现在居住在美国东南部的佛罗里达。

        - 莎拉波娃  现在   居住  在    美国   东南部     的    佛罗里达

        - Sharapova now     lives in       US       southeastern     Florida

  - A modern variation is found in programming, where literals cannot contain spaces. Depending on coding style, developers employ different methods to create meaningful names, like QueryParser, assertEquals, word_tokenize, preserve_line, and more. Coding assistants can break these literals into meaningful tokens to grasp the developer's intent.

  - Transcribing spoken language into written form initially creates a phoneme stream and then determines word boundaries. However, in speech, words are not separated; instead, they are joined into a continuous stream of phonemes. For the transcriber who identifies phonemes in the first step, it appears as follows:

      - ðɪskɔːsˈtiːʧɪzˌmʌltɪˈmiːdiərɪˈtriːvᵊl.

      - ðɪs   kɔːs     ˈtiːʧɪz     ˌmʌltɪˈmiːdiə  rɪˈtriːvᵊl.

      - This course teaches multimedia    retrieval.

There are two different approaches to break continuous streams into tokens:

  - We can combine a dictionary-based approach with a hidden Markov model (or a neural network). The dictionary helps determine if a character sequence can form a word and provides all options for the current text position. For instance, consider the English character sequence “h e s i t a t e“. We could extract the single token “hesitate” or the series of tokens “he“, "sit“, “ate“; or “he’s”, “it“, “ate“. Vocabulary lookup can introduce ambiguity which we can resolve using a trained hidden Markov model (or a neural network). This model evaluates alternatives and selects the most likely sequence such as “hesitate” for our example. Language-specific models can utilize rules like maximum matching (finding the longest sequence in the dictionary) and language-specific character usage to identify word boundaries. A general challenges, for instance also in transcription of spoken text, are names of people or brands as well as loan words from other language (e.g., English computer terms in German or Thai).

  - Another approach involves sub-words that are used directly for retrieval. Using the previous example for spoken text retrieval, the phoneme stream is divided into overlapping sequences of three phonemes:

      - ðɪskɔːsˈtiːʧɪzˌmʌltɪˈmiːdiərɪˈtriːvᵊl.    ðɪs  ɪsk  skɔ  kɔːs  ɔːsˈt  … ˌmʌl  ʌlt  ltɪ  tɪˈm  ɪˈmi  ˈmiːd  iːdi  diə  iər  …

      - Stress symbols in the phoneme stream are combined with the following phoneme, enlarging the symbol vocabulary. To match this with a query, let's consider the user is searching for “multimedia”. This query is initially translated into a phoneme stream and then segmented into sequences of three phonemes:

      - multimedia    ˌmʌltɪˈmiːdiə    ˌmʌl  ʌlt  ltɪ  tɪˈm  ɪˈmi  ˈmiːd  iːdi  diə

      - This creates an 8-token query, and we can employ a standard retrieval method that may consider token proximity. An intriguing outcome of this method is that we do not need to match all sub-sequences to locate relevant spoken text passages. For instance, if a non-native speaker mispronounces words or someone has unclear articulation, the phoneme stream from the spoken text may differ from the one generated by the query. However, as long as there are sufficient overlaps between the sequences, we can still locate the passage.

      - We can apply this method also in situations where word boundaries are identifiable:

      - This course teaches multimedia retrieval.	thi  his  cou  our  urs  rse  tea  eac  ach  che  hes  mul  ult  lti  tim  ime  med  edi  dia  ret  etr  tri  …

      - In this scenario, we create sub-sequences only within words, avoiding sub-word tokens spanning across two words unlike the phoneme example above. An extension of this approach is to differentiate between sub-sequences at the beginning of a word and those within. This distinction can be made by treating them as separate tokens and prefixing sub-sequences at the start of words with "#" (or any unused symbol).

      - This course teaches multimedia retrieval.	#Thi  his  #cou  our  urs  rse  #tea  eac  ach  che  hes  #mul  ult  lti  tim  ime  med  edi  dia  …

      - We can transform queries the same way. Let’s use an example to illustrate some of the advantages:

      - Q = “teach multtimedia”      #tea  eac  ach  #mul  ult  ltt  tti  tim  ime  med  edi  dia

      - Despite having a different flexed form for “teach” and a misspelling (double “tt”), 10 out of 12 sub-sequences match those from the sentence above. A retrieval model with partial matching and optional token proximity consideration can locate the relevant passage without requiring stemming or spelling corrections. Modern large language models use a similar approach as we will discuss later in this course.

N-grams: Rather than making tokens smaller, we can create larger tokens by combining multiple words into a single token known as n-grams. This approach is particularly valuable in languages where words form phrases with distinct or more specific meanings. Examples include:

  - - mother tongue, red handed, butterfly effect, black box, cold shoulder, silver bullet, piece of cake

  - - thai food, prime minister, middle management, crystal clear, chief of staff, speed dial, multimedia retrieval

  - - New York City, Salt Lake City, Albert Einstein, Amazon Web Services, Ford Mustang, University of Basel

  - In all these examples, it makes more sense to use phrases rather than the individual terms. In order to enrich a vocabulary with phrases, we can create them manually or form them automatically from a text corpus.

  - A naïve approach first constructs all possible bi-grams in a corpus and then counts their occurrences. The top-n most frequent bi-grams are added to the vocabulary. However, a limitation of this method is evident in the upper table on the right side:

    - "of the" is the most frequent bi-gram simply because it comprises two frequently used stop words in the language.

  - A first enhancement excludes stop words when generating bi-grams and considers only consecutive pairs of non-stop words. Ensure that you do not merely remove stop words from the stream but eliminate pairs containing a stop word. Otherwise, you create pairs that originally had a stop word in between. The lower table on the right illustrates the outcomes:

    - The result appears more favorable than previously, with names from the novel forming new terms in the vocabulary. This streamlines the search for names since we only need to search for the bi-gram, eliminating the need to search for individual parts and apply a proximity constraint.

    - Nonetheless, a few issues remain. Phrases like “said Holmes”, “could see”, and “young man” are common pairs, but they do not contribute significantly to describing the context they appear in. Notably, in the case of “said Holmes”, we observe that these two terms less frequently occur together but are more often associated with other words (e.g., “said” is not exclusively used with “Holmes”).

  - The Pointwise Mutual Information (PMI) measures word associations by comparing their actual co-occurrence frequency to what would be expected if they were independent. In our previous example, we noted the bi-gram “said Holmes” occurred 12 times together. However, “said” appeared 207 times, and “Holmes” 94 times individually. In essence, “said” and “Holmes” rarely co-occur (12 out of a maximum of 94 times), and “said” pairs with many other words. Although they occur together more frequently than other bi-grams, this observation suggests they are not a distinctive enough bi-gram for our vocabulary. We are more interested in word pairs like the names which predominantly appear as bi-grams (even though first and last names can also occur independently).

    - To formalize this measure, let $t_{1}$ represent the first term in the bi-gram and $t_{2}$ the second term. We count the occurrences of the individual terms as $tf(t_{1})$ and $tf(t_{2})$, and of the bi-gram as $tf(t_{1}, t_{2})$. PMI compares the likelihood of terms occurring together to the expected probability if they were independent of each other:

    - If the corpus comprises $N$ terms, the probabilities are determined by the ratio of the term frequency to $N$:

    - In the last part of the formula above, we eliminated the constant multiplier $log_{2}(N)$ that applies to all bi-grams. The right-most formula now determines the significance of bi-grams with the PMI measure. While it is possible to remove the $log_{2}()$ as well, keeping it in place helps maintain values within more manageable ranges for humans.

    - Note: the PMI value is maximized when $tf(t_{1}) = tf(t_{2}) = tf(t_{1}, t_{2})$, meaning that all occurrences of the two terms exist exclusively within the bi-gram. If a term appears outside the bi-gram, the denominator becomes larger, resulting in a smaller PMI value as a consequence.

    - As a result of the previous statement, stop words that appear in bi-grams are naturally given lower weights because they are highly frequent outside of the bi-gram context. Consequently, there is no longer a necessity to employ a stop word filter (although it can still be used for efficiency when computing PMI).

$pmi\left(t_{1},t_{2}\right)=log_{2}\frac{p\left(t_{1},t_{2}\right)}{p\left(t_{1}\right)∙p\left(t_{2}\right)}=log_{2}p\left(t_{1},t_{2}\right)−log_{2}p\left(t_{1}\right)−log_{2}p\left(t_{2}\right)$

$pmi\left(t_{1},t_{2}\right)=log_{2}\frac{p\left(t_{1},t_{2}\right)}{p\left(t_{1}\right)∙p\left(t_{2}\right)}=log_{2}\frac{\frac{tf(t_{1}, t_{2})}{N}}{\frac{tf(t_{1})}{N}∙\frac{tf(t_{2})}{N}}=log_{2}\frac{N∙tf(t_{1},t_{2})}{tf(t_{1})∙tf\left(t_{2}\right)}    \~   log_{2}\frac{tf(t_{1},t_{2})}{tf(t_{1})∙tf\left(t_{2}\right)}$

    - Now, let’s use the PMI measure to find the most significant bi-grams in the same example text. The upper table on the right side displays the outcomes. Notably, all stop words have been excluded, but at the top, we see bi-grams consisting of rare terms. For instance, “Army Medical” appears only once, and the terms within the bi-gram also occur only once within that bi-gram. This is why it receives the highest score.

    - We already established that the PMI score is the highest if $tf(t_{1}) = tf(t_{2}) = tf(t_{1}, t_{2})$. Let’s say such a bi-gram occurs $n$ times. The PMI score is then given by:

    - In other words, for bi-grams where the terms only occur together in that bi-gram, the PMI is high when the count $n$, denoting the number of bi-gram occurrences, is low. The optimal value is achieved when $n=1$, as demonstrated in the result table ($log_{2}(N)$ is $15.39$ for this example).

    - To improve the quality of returned bigrams, we can apply a minimum frequency filter as applied for the lower table on the right side. This now eliminates all bi-grams with stop-words and all bi-grams with infrequent terms.

    - The bi-gram result is improved, revealing numerous names from the novel and capturing meaningful pairs like “never mind”, “old farmer”, or “two detectives”.

$pmi\left(t_{1},t_{2}\right) =log_{2}\frac{N∙tf(t_{1},t_{2})}{tf(t_{1})∙tf\left(t_{2}\right)}    =  log_{2}\frac{N∙n}{n∙n} $

$=  log_{2}\left(N\right)−log_{2}(n)   $

  - Likelihood Ratios (LHR) are another form of hypothesis testing, similar to the chi-squared test but more robust when dealing with sparse data. Moreover, the resulting number is easier to interpret, indicating how much more likely one hypothesis is compared to another. In the context of bi-grams, the initial hypothesis $H_{1}$ assumes independence between terms $t_{1}$ and $t_{2}$ in the bi-gram. This hypothesis can be represented as:

    - The first probability represents the conditional likelihood of $t_{2}$ following $t_{1}$, while the second one is the conditional probability of $t_{2}$ not following $t_{1}$. Let $tf_{1}=tf(t_{1})$ represent the occurrences of $t_{1}$, $tf_{2}=tf(t_{2})$ for $t_{2}$, and $tf_{12}=tf(t_{1},t_{2})$ for the bi-gram. For hypothesis $H_{1}$, we can use the maximum likelihood estimate for $p=tf_{2}/N$, where $p$ is the probability of $t_{2}$ following any term, whether it is $t_{1}$ or not (independence). Assuming a binomial distribution, we can calculate the likelihood of observing these counts as follows:

    - The first binomial distribution calculates the likelihood of observing $tf_{12}$ instances of $t_{2}$ following $t_{1}$ out of $tf_{1}$ occurrences, considering the probability $p$ that the term $t_{2}$ appears at any position. The second hypothesis $H_{2}$ assumes that $t_{2}$ depends on $t_{1}$ and hence the conditional probabilities differ:

    - As previously, we can employ maximum likelihood estimates for $p_{1}=tf_{12}/tf_{1}$ and $p_{2}=(tf_{2}−tf_{12})/(N−tf_{1})$ using the observed counts. Assuming a binomial distribution, we can then compute the likelihood of the second hypothesis, which is similar to the first, considering both scenarios: $t_{2}$ following $t_{1}$ and $t_{2}$ not following $t_{1}$.

    - Lastly, the likelihood ratio log lambda is given as

$H_{1}:  P\left(t_{2}\right|t_{1})=P\left(t_{2}\right|¬t_{1})=p$

$L\left(H_{1}\right)=b\left(tf_{12};tf_{1}, p\right)∙b\left(tf_{2}−tf_{12};N−tf_{1}, p\right)$            with     $b(k; n, x) = \left(\begin{matrix}n\\k\end{matrix}\right)x^{k}\left(1−x\right)^{n−k}$

$H_{2}:  p_{1}=P\left(t_{2}\right|t_{1})$       $p_{2}=P\left(t_{2}\right|¬t_{1})$          $p_{1}\ne p_{2}$

$L\left(H_{2}\right)=b\left(tf_{12};tf_{1}, p_{1}\right)∙b\left(tf_{2}−tf_{12};N−tf_{1}, p_{2}\right)$         with     $b(k; n, x) = \left(\begin{matrix}n\\k\end{matrix}\right)x^{k}\left(1−x\right)^{n−k}$

$log 𝜆 = log 𝐿 ( 𝐻 1 ) 𝐿 ( 𝐻 2 ) = log 𝐿 𝑡 𝑓 12 ; 𝑡 𝑓 1 , 𝑝 ∙ 𝐿 𝑡 𝑓 2 − 𝑡 𝑓 12 ; 𝑁 − 𝑡 𝑓 1 , 𝑝 𝐿 𝑡 𝑓 12 ; 𝑡 𝑓 1 , 𝑝 1 ∙ 𝐿 𝑡 𝑓 2 − 𝑡 𝑓 12 ; 𝑁 − 𝑡 𝑓 1 , 𝑝 2$        with     $L(k; n, x) = x^{k}\left(1−x\right)^{n−k}$

    - Now, let's apply the LHR measure to identify the most significant bi-grams in the same example text. For that purpose, we sort the bi-grams by $−2∙log\lambda $. The upper table on the right shows the results, and interestingly, the stop-words have reappeared. Unlike the naive approach where stop words appeared due to their high frequency, we now have a different scenario as LHR compares the hypothesis of independence versus dependence. Take, for instance, the bi-gram “I am” which is not a significant bi-gram for the vocabulary. Nevertheless, it is evident that “am” is strongly dependent on “I” and follows the term “I” in 39 out of 41 instances.

    - We can enhance the quality of bi-grams obtained with LHR by excluding those containing a stop word (do not filter stop words before forming bi-grams). The final outcome is displayed in the lower table on the right. Unlike the PMI ranking, the more frequent names now occupy the top positions. Notably, “Sherlock Holmes” appears 48 times as a bi-gram and attains the highest LHR value, while it held only the 14th place in the PMI ranking, due to the preference of the PMI for lower numbers of occurrences.

With all the bi-gram scoring methods that we discussed so far, we need to establish a threshold. All bi-grams with scores exceeding this threshold are included in the vocabulary, while the rest are excluded. There is no need to be accurate in setting the threshold, rather, we should take additional search and storage overhead into account. If we missed a bi-gram, we can still find it with proximity measures.

When creating bi-grams, we can also choose to index both individual terms and the bi-gram. This allows us, for example, to search for “Holmes” which would otherwise not match with occurrences of the bi-gram “Sherlock Holmes”.

We can expand this concept to tri-grams or even quad-grams. The tables on the right display the top n-grams in the corpus, and we can expand our vocabulary accordingly (typically selecting several hundreds to thousands in a large corpus).

Finally, the code below demonstrates how to compute the n-gram tables discussed in this section. nltk offers a variety of convenient functions for handling collocations. For more details, refer to the documentation.

from nltk.collocations import (

BigramCollocationFinder,  TrigramCollocationFinder,   QuadgramCollocationFinder,

BigramAssocMeasures,      TrigramAssocMeasures,       QuadgramAssocMeasures

)

from nltk.corpus import stopwords

# choose bi-grams, tri-grams, quad-grams

finder = QuadgramCollocationFinder.from_words(tokens)

finder = TrigramCollocationFinder.from_words(tokens)

finder = BigramCollocationFinder.from_words(tokens)

# choose a measure (must match with the finder, here for bi-grams)

measure = BigramAssocMeasures.raw_freq

measure = BigramAssocMeasures.pmi

measure = BigramAssocMeasures.likelihood_ratio

# apply frequency filter

finder.apply_freq_filter(3)

#apply stop word filter

ignored_words = stopwords.words('english')

stopword_filter = lambda w: len(w) < 3 or w.lower() in ignored_words

finder.apply_word_filter(stopword_filter)

# obtain results (top-k)

k = 20

scores = finder.score_ngrams(measure)[:k]

# output term 1, term 2, freq of term 1, freq of term 2, freq of bigram, score

for ((t1,t2),score) in scores:

print(f'{t1} {t2} {finder.word_fd[t1]} {finder.word_fd[t2]} {finder.ngram_fd[(t1,t2)]} {score}')

Tokenization has regained significance alongside the rise of large language models. Later in this course, we will explore these models and their applications in retrieval scenarios. When dealing with text in machine learning, a central challenge is how to input text into the model:

  - Many machine learning models typically process continuous input values, with exceptions like decision trees and naive Bayes. In neural networks, a fully connected layer multiplies input values by weights, adds a bias, and applies an activation function. This process results in an outcome represented by a function $f$ applied to the input values. Therefore, we must find a way to map the generated tokens (assuming words for now) to meaningful input values.

  - An initial idea is to assign a unique ID to each token as we add them to the vocabulary. To illustrate this, consider a simple example: after tokenizing the sentence “the cat and the dog”, we have four tokens:

      - the  1     cat  2     and  3     dog  4

    - These IDs enable us to represent the original sentence as a sequence of numbers, such as [1, 2, 3, 1, 4]. However, we cannot directly input these numbers into a machine learning model: the tokens in the example sentence have weak semantic relationships among themselves. Yet, assigning them numerical values like 1, 2, 3, and 4 implies a strong relationship between them. For instance, if “cat” is mapped to 2 and “dog” is mapped to 4, does this imply that 2 cats make up a dog? Additionally, the mapping of “and” to 3, positioned between “cat” and “dog” may suggest that “and” is also an animal because of its proximity to “cat” and “dog”.

  - To prevent such misinterpretations, the standard practice in data science and machine learning is to convert discrete values (such as categories or token IDs) into one-hot vectors. These vectors have a dimensionality equal to the number of tokens (or categories), and each token is represented by a vector containing all zeros except for one component, determined by the token ID, which is set to 1. The following illustrates this approach in an example of sentiment analysis, where the model predicts whether a sentence is positive, negative, or neutral:

the

cat

and

the

dog

1

2

3

1

4

1 0 0 0

0 1 0 0

0 0 1 0

1 0 0 0

0 0 0 1

machine learning model

positive

neutral

negative

one-hot vector

tokenID

token

  - One-hot vectors work well for a few hundred categories but become impractical for vocabularies with millions of entries because they require huge input layers in machine learning models. Expanding the vocabulary after a model is trained is also difficult. For example, adding new terms like names or brands usually means retraining the model. To avoid this, models such as BERT (Google, 2018) with a 30,000-token vocabulary, GPT-3 (OpenAI, 2020) with 50,000 tokens, and Google Bard (2023) with 137,000 tokens use stable, compact vocabularies and tokenization methods like Word Piece and Byte-Pair Encoding.

  - Before we go deeper into these tokenization methods, note that none of the previously mentioned language models utilize one-hot input vectors due to performance and storage reasons. Instead, they incorporate an extra layer for embedding and positional encoding before the data proceeds to the transformer blocks of the model. While we will explore embeddings later in this course, it is opportune to discuss here shortly the complete process from text to transformer layers, as depicted at the bottom of this page:

    - Embedding layers transform one-hot vectors into lower-dimensional, dense vectors. The concept is to map tokens with semantic similarity closer together in this lower-dimensional space. For example, “cat” and “cats” are semantically related, but one-hot vectors treat them as distinct representations. Embedding layers, however, assign “cat” and “cats” similar vectors, enabling the model to grasp their relationship more effectively.

    - The embedding layer is a basic fully connected network that maps the one-hot vector into a lower-dimensional representation. This transformation is learned alongside the rest of the model and does not include bias or an activation function. Given that the input is a one-hot vector, the embedding layer essentially looks up the corresponding column in the weight matrix, avoiding the need for expensive matrix-vector multiplications.

    - The positional encoding is a sinusoidal signal function for the model to understand the order of tokens in the input sequence as transformers lack the inherent sense of order found in recurrent neural networks (RNN).

■■

■■

■■

■■

■■

block 1

positive

neutral

negative

embedding layer

■■

■■

■■

■■

■■

■■

■■

■■

■■

■■

positionalencoding

1

2

3

1

4

tokenID

the

cat

and

the

dog

token

multi layer

block …

block n

input fortransformer

softmax

transformer architecture

task specificnetwork

embeddings

  - Byte Pair Encoding (BPE) tokenization: initially proposed for text compression, it was employed by OpenAI to reduce the vocabulary sizes. There are many variants but they all share the same idea:

    - We begin by normalizing the input sequence. Older models converted Unicode characters to ASCII and to lowercase. More recent models operate at the byte level, treating Unicode characters as sequences of bytes. This enables newer models to better handle languages with special characters. While punctuation is often minimized, it still holds significance in ensuring the model can generate coherent sentences. Let's illustrate BPE tokenization with a simple example and the normalization step:

        - This course is about this topic.         this course is about this topic

    - The initial vocabulary is established with all the characters of all words in the corpus. With our simple example, the initial vocabulary is:

        - a, b, c, e, h, i, o, p, r, s, t, u

    - Next, we expand the vocabulary by including the most common bi-gram found within all words of the corpus. We start by creating a bag-of-words representation and break down each word into character sequences. In our simple example, most words appear only once, but in practice, they would have varying frequencies (2nd column):

        - this	2	t, h, i, s

        - course	1	c, o, u, r, s, e

        - is	1	i, s

        - about	1	a, b, o, u, t

        - topic	1	t, o, p, i, c

    - For each word, we generate all possible bi-grams. For instance, with the word this we form the bi-grams th, hi, and is. Subsequently, we count the occurrences of these bi-grams, factoring in the frequency of the words:

        - is (3), th (2), hi (2), ou (2), cu (1), ur (1), rs (1), se (1), ab (1), bo (1), …

    - is is the most frequent bi-gram. So we create a new vocabulary item for it:

        - a, b, c, e, h, i, o, p, r, s, t, u, is

    - Using this updated vocabulary, we can now modify the word representations by substituting consecutive i and s with the new vocabulary item is.

        - this	2	t, h, is

        - course	1	c, o, u, r, s, e

        - is	1	is

        - about	1	a, b, o, u, t

        - topic	1	t, o, p, i, c

    - Now, we can repeat this process by generating again possible bi-grams (where is counts as one item, not two):

        - th (2), his (2), ou (2), cu (1), ur (1), rs (1), se (1), ab (1), bo (1), ut (1), …

    - This time, we encounter a tie, so we can randomly select one of the best pairs. Let's choose th and incorporate it into the vocabulary. As previously, we update our word representation:

        - this	2	th, is

        - course	1	c, o, u, r, s, e

        - is	1	is

        - about	1	a, b, o, u, t

        - topic	1	t, o, p, i, c

    - The process continues until we have arrived at a specified vocabulary size. For the example, we stop at 20:

        - vocabulary:  a b c cou cour cours course e h i is o ou p r s t th this u

    - Using this vocabulary, we can now encode our original sentence as follows:

        - this  course  is  a b ou t  this  t o p i c

    - Even if some words are no longer in the vocabulary, we can still represent them as a sequence of smaller tokens. In a large corpus, this allows the model to handle all words, including misspelled ones, and accept previously unseen words. Newer models use byte-level encoding, avoiding the need to reserve the entire Unicode alphabet in the vocabulary. Instead, they start with 256 initial entries and the BPE algorithm will automatically compose 2 bytes to represent common Unicode characters in the corpus.

  - The transformers library offers efficient tools for training custom tokenizers. The code on the right demonstrates how to utilize the library to train custom BPE tokenizers (steps 1-4) and how to reuse an existing BPE tokenizer (step 5 with GPT-2).

    - Training a language model requires special tokens that signal particular conditions to the model. For example, the "unknown" token is used for any input sequence that does not match a token in the vocabulary.

    - The trainer runs the BPE algorithm. It lets us set the vocabulary size, choose a minimum frequency for pairs to be added to the vocabulary, and define prefixes and suffixes for pairs that occur inside a word or at its end. This ensures the same pair, such as "is", is treated as different tokens when it appears at the beginning, middle, or end of a word.

    - Specifies how input text is split into words, how punctuation is handled, and what normalization is applied before tokenization, for example converting text to lowercase. Keep in mind that normalization helps tasks like classification but should be avoided for language models that generate text. For example, converting all letters to lowercase can prevent a model from producing properly capitalized English sentences.

    - Trains the BPE tokenizer using a set of input files, and then apply it to encode some text.

    - Obtains a GPT-2 pre-trained tokenizer and uses it to encode some text.

  - WordPiece Tokenization follows BPE's general approach but extends it in two key ways:

    - It distinguishes between characters at the word's beginning and those in the middle. The original BPE version, initially stemming from a compression algorithm, did not account for character positions. However, later extensions of BPE, including the transforms library demonstrated on the previous page, incorporated this concept. In our example sentence, "this course is about this topic“, the starting vocabulary is altered: '##' serves as a special annotation for word pieces that start within the word:

        - vocabulary: 	##b, ##c, ##e, ##h, ##i, ##o, ##p, ##r, ##s, ##t, ##u, a, c, i, t

        - this	2	t, ##h, ##i, ##s

        - course	1	c, ##o, ##u, ##r, ##s, ##e

        - is	1	i, ##s

        - about	1	a, ##b, ##o, ##u, ##t

        - topic	1	t, ##o, ##p, ##i, ##c

        - While this approach doubles the base vocabulary, it enhances our ability to capture prefixes, which frequently convey shared semantics across words. The same principle applies to suffixes, which often group inflected forms based on gender, numbers, tense, and case.

    - It constructs pairs in the same manner as BPE but prioritizes pairs with their components occurring more frequently together than with other pieces. This results in the same scoring criteria that we introduced for selecting bi-grams with PMI. If $(a, b)$ represents a potential candidate pair, we denote the frequencies of the individual elements of the pair as $tf(a)$ and $tf(b)$, and the frequency of the pair itself as $tf(a, b)$. The best pair is determined as follows:

        - As discussed with PMI, we need to apply a minimum frequency filter as the formula above prefers infrequent pairs such as the ones with $tf(a,b)=tf(a)=tf(b)=1$.

    - The BPW and WordPiece algorithms share the same process: they begin with an initial vocabulary, create pairs, count frequencies, expand the vocabulary with the best pair, merge pairs for all words, and continue this cycle of pair creation, vocabulary expansion, and merging until a specified vocabulary size is achieved.

[MATH_ERROR]

  - We can use the transformers library's efficient WordPiece trainers. The code on the right shows how to train custom WordPiece tokenizers (steps 1-4) and how to reuse an existing WordPiece tokenizer (step 5 uses BERT uncased).

    - As before, we define special tokens based on the task we want the model to perform. For example, the "unknown" token is used for any input sequence that does not match a token in the vocabulary.

    - The trainer runs the WordPiece algorithm. It lets us set the vocabulary size, choose a minimum frequency for pairs to be included, and define prefixes and suffixes for pairs inside words or at their ends.

    - Define how input text is split into words, how punctuation is handled, and what normalization is applied before tokenization, for example converting text to lowercase. Choose normalization that fits the scenario and the model you are using. For example, converting text to lowercase can prevent the model from producing grammatically correct output.

    - Train the WordPiece tokenizer on a set of input files, then use it to encode text.

    - Obtain a BERT (uncased) pre-trained tokenizer and use it to encode some text.
