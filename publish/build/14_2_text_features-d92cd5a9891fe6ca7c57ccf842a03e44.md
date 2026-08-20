# Text Features

Topic modeling and clustering differs from text classification. The distinction lies in the approach: classification relies on supervised learning with predefined classes, learning how features align with these classes. In contrast, topic modeling and clustering is unsupervised, aiming to detect clusters or co-occurrences of terms within text documents, and assigning them topic labels (as illustrated in the figure below). LSI also employs unsupervised techniques to learn topics through singular value decomposition and thereby reduces the rank of the document-term matrix. While this process shares similarities with topic modeling methods like Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF), LSI's primary objective is not to extract and explain topics found within the collection. Instead, it leverages these abstract topics for semantic retrieval.

While deep learning can handle various tasks, we should also explore cost-effective methods that provide satisfactory solutions. Training complex language models can be resource-intensive, whereas simpler techniques can yield comparable results, particularly when term occurrences is the dominating factor for class assignments.

In the following, we review efficient methods that continue to deliver excellent results. Many remain in use to offer state-of-the-art structural features for particular use cases.

TermsEmbeddingsPOS Tags

Classification

Topic Modelling

supervised

unsupervised


## 14.2.1 Language Detection


Language detection is the problem of determining the language in a text or document. This task is rather simple for long documents, but can become quite challenging for short texts or when a large number of languages have to be detected automatically. A related problem: detecting programming languages.

Let’s start with the simple method of detecting languages in longer texts. The most efficient approach is to apply a number of rules to detect the language:

  - Alphabet Diversity: Each language has unique characters found in only a few related languages. Examples include Latin, Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, Tamil, Bengali scripts, as well as Chinese, Hiragana, and Katakana characters. Languages often borrow words from others, leading to a mix of alphabets. To handle this, we can filter out rarely used alphabets.

  - Character Diversity: Some languages have special characters within an alphabet that are typical of their linguistic uniqueness. For instance, diacritical marks and accent symbols in Latin-based scripts, or tonal markers in certain Asian languages, add distinctive features to characters. Only a few Latin based languages use ä, ö, and ü.

  - Stop word Counts: Evaluating stop word frequencies in text can reveal a language. For instance, languages like English and French often employ frequent stop words, while others, like Mandarin Chinese, rely less on them. Using managed stop word lists allows us to guess a language simply by counting how often its stop words occurs.

  - Vocabulary Counts: Examining the unique words or vocabulary in a text can also help identify the language. Different languages have distinct vocabularies, and by comparing word frequencies and diversity, it becomes possible to make language determinations with a degree of accuracy.

For longer texts, these rules quickly lead to the identification of the language. However, the method does not easily scale to large numbers of languages. The alphabet and character rules are simple, but the stop word lists and vocabularies (most frequent words) require large amount of data to perform language detection.

For shorter texts or brief phrases, these methods are less effective unless we have comprehensive vocabularies for all languages. In some cases, it may be challenging to identify the correct language, as a single word or short phrase can exist in multiple languages. Even more complex are phrases containing loanwords from other languages, such as IT terms in a German phrase like "mein computer".

Modern language detectors operate at the sub-word level and incorporate rules like those mentioned earlier, such as Alphabet and Character rules. Additionally, they introduce new rules based on character-based n-grams that are specific to certain languages. The key distinction of these new methods, however, lies in utilizing the frequencies of character-based n-grams for language detection, and this is achieved using a Naïve Bayes learning model.

Naïve Bayes employs a conditional probability model based on Bayes' theorem.

  - In this equation, $𝒙$ represents a feature vector, and $C_{k}$ is the class or target. $P\left(C_{k}\right)$ is the prior, that is knowledge about the distribution (probability) of classes $C_{k}$. $P\left(C_{k}\right)$ is the likelihood of observing feature $𝒙$ for a specific class $C_{k}$, and $P\left(𝒙\right)$ is the overall evidence of observing $𝒙$, regardless of class. $P\left(𝒙\right)$ represents the posterior which is the knowledge gained or predicted when observing feature $𝒙$, allowing us to infer its association with class $C_{k}$.

Consider $𝒙$ as a high-dimensional vector, often derived from a vast term space used in documents. Given the high dimensionality and the restricted training data, accurately modeling the probability distribution function in this sparse space is challenging. To simplify, naïve Bayes assumes conditional independence among features, resulting in the following simplification:

Using the probability model, we choose the most probable hypothesis, that is class $C_{k^{∗}}$ that maximizes the probability function. This selection principle is commonly referred to as maximum a posteriori (MAP):

Now, we need to estimate the probabilities $P(C_{k})$ and $P\left(C_{k}\right)$ based on observations from the training set.

$P\left(𝒙\right)=\frac{P\left(C_{k}\right)∙P\left(C_{k}\right)}{P\left(𝒙\right)}$

$posterior=\frac{likelihood ∙prior}{evidence}$

$P\left(𝒙\right)=P\left(x_{1},…,x_{M}\right)=\frac{1}{P\left(𝒙\right)}∙P(C_{k})∙\prod_{j=1}^{M}P\left(C_{k}\right)$

Note that $P\left(𝒙\right)$  is a constant across classes $C_{k}$ and only scales the probabilities. For our purposes, we do not require its value

[MATH_ERROR]

That is it! The equation describes the decision rule of Naïve Bayes. The only thing left are the estimates for the probabilities on the right hand side

In our language detection scenario, we use character-based n-grams of varying lengths (e.g., n from 1 to 5). We count how often these n-grams appear in the text, resulting in a bag-of-words representation that forms a multinomial distribution. The feature vector $𝒙$ represents these counts for a defined vocabulary for each language.

The priors $P\left(C_{k}\right)$ depend on the scenario: we can use a maximum likelihood estimator based on observations in the training set. Let $N_{k}$ be the number of texts for the language denoted by class $C_{k}$, and $N$ be the total number of texts:

  - If we lack knowledge of the language distribution or wish to avoid training bias, we can select a constant prior for all classes, which can then be omitted from subsequent calculations since it only scales posteriors for all classes.

To estimate the likelihoods $P\left(C_{k}\right)$ from texts in a language represented by class $C_{k}$, we count the n-gram occurrences in the training data for that language (multinomial distribution). For each language, we establish first an appropriate vocabulary, using methods similar to word-pieces or BPE, to control vocabulary size. We prioritize the most frequent n-grams since they have the most influence on the posterior and impact language determination the most. Let $n_{k,j}$ denote the total occurrences of n-gram $t_{j}$ in all training texts for the language represented by class $C_{k}$:

  - As we choose the vocabulary tailored to the target language and exclude infrequent or absent n-grams from the test set, we do not require the “+1” smoothing on the right-hand side. However, in other text classification tasks, smoothing prevents $p_{k,j}$ from reaching 0 for rare tokens during predictions (which leads to a posterior of value 0).

Finally, we can predict the language based on posteriors. Instead of multiplying probabilities as shown on the previous page, we rather use sums over log-probabilities:

  - We can obtain scores with a softmax classifier over the target languages, and select the language with highest score.

$P\left(C_{k}\right)=\frac{N_{k}}{N}$

$P\left(C_{k}\right)=\frac{1}{K}$

or if $N_{k}$ is not known:

$p_{k,j}=\frac{n_{k,j}}{\sum_{l}^{}n_{k,l}}$

$p_{k,j}=\frac{n_{k,j}+1}{\sum_{l}^{}n_{k,l}+M}$

or smoothed:

[MATH_ERROR]

Examples: The lingua-language-detector is a highly efficient language detector with over 99% accuracy for more than 70 languages. Let's explore its functionality through examples.

  - We can also inquire about the likelihood of a phrase belonging to a particular set of languages:

  - The detector also are able to predict the languages out of fragments:

  - This also demonstrates that the detector operates at sub-word levels. The 3-grams “hau” and “mei” are more common in German texts than in English and Italian, resulting in higher confidence scores.

Another Python library is langdetect, which is also a rule and n-grams based language detector for 55 languages. It provides ISO-codes for the detected languages:

from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()

detector.detect_language_of("This is an example sentence")	#  Language.ENGLISH

detector.detect_language_of("Je suis un exemple de phrase")	#  Language.FRENCH

detector.detect_language_of("                                  ")		#  Language.THAI

languages = [Language.ENGLISH, Language.FRENCH, Language.ITALIAN]

detector = LanguageDetectorBuilder.from_languages(*languages).build()

detector.compute_language_confidence_values("Je suis à New York")

⮡  FRENCH: 0.45 ENGLISH: 0.37 ITALIAN: 0.18

confidence_values = detector.compute_language_confidence_values("hau mei")

⮡  GERMAN: 0.82 ENGLISH: 0.10 ITALIAN: 0.07

from langdetect import detect, detect_langsdetect("This is an example sentence")	#  en

detect("je suis un exemple de phrase")	#  fr

detect("Este es un ejemplo de frase")	#  es

detect("Dies ist ein Beispieltext")	#  de

detect("Questo è un esempio di frase")	#  it

detect("             ")		#  th

detect_langs("Je suis à New York")	#  [fr:0.86, en: 0.14]

## 14.2.2 Sentiment Analysis


Sentiment analysis deciphers human language to understand emotions and opinions. It's widely used to assess customer sentiment from reviews, social media, and support cases, aiding data-driven decisions for product improvement and customer satisfaction. It also can help to filter and moderate user-generated content to safeguard brand reputation and enforce community guidelines.

In sentiment analysis, a fundamental task is to classify text polarity, identifying if it is positive, negative, or neutral. Advanced tasks can recognize emotions like anger, fear, disgust, joy, and surprise. Here, we focus on basic sentiment prediction with positive, negative, and neutral categories. Let’s start with a look at some example, and the challenges machine learning models may encompass:

  - Many statements are straightforward and sentiment is often driven by a few key words:

      - “I like this product”  				 positive

      - “I was going to the town”  			 neutral

      - “The food was really bad”			 negative

  - However, we can express ourselves in many, sometimes confusing ways that are difficult to analyze:

      - “I can’t say I liked it”				 negation handling

      - “Drinking wine is not my thing”			 negative or neutral?

      - “What a fine artist you've become!”		 potentially sarcastic

      - “I haven't ever owed anything to anyone”		 lots of negation, but actually positive

We consider 3 different approaches:

  - Naïve Bayes with the example of sentiment analysis in Twitter (now called X)

  - TextCNN, a convolutional network on embeddings to predict classes

  - Transformer based classification models

Naïve Bayes is popular for its simplicity, speed, and accuracy. We used it before for language detection with a multinomial distribution, considering term presence and counts. In Twitter sentiment analysis, short texts mean terms usually occur only once, except for stop words. We use a set-of-word representation and assume a multivariate Bernoulli distribution for likelihood estimation. This examples uses two classes: positive and negative

  - The priors $P\left(C_{k}\right)$ measure how likely a messages fall into one of the two classes (positive, negative). We can use a maximum likelihood estimator based on observations in the training set. Let $N_{k}$ be the number tweets for class $C_{k}$, and $N$ be the total number of texts (with $k =$ ‘positive’ or ‘negative’):

    - If we do not know the sentiment distribution or wish to avoid training bias, we can select a constant prior for all classes, which can then be omitted from subsequent calculations since it only scales posteriors for all classes.

  - Assuming a multivariate Bernoulli distribution for the set-of-word representations, we can estimate the  likelihoods $P\left(C_{k}\right)$ as follows. Let $N_{k}(x_{j}=1)$ denote the number of messages from $C_{k}$ that contain a term $t_{j}$:

    - We can use either smoothing to prevent $p_{k,j}=0$ if a term $t_{j}$ does not occur in the messages of class $C_{k}$, or we simply ignore terms that were not present in the training data of class $C_{k}$ during predictions.

  - Finally, we can predict the sentiment (‘positive’ or ‘negative’ class) based on posteriors. Instead of multiplying probabilities, we again use sums over log-probabilities (and ignore terms with $p_{k,j}=0$):

  - Instead of using the entire vocabulary, we can reduce features by selecting the most informative terms.

$P\left(C_{k}\right)=\frac{N_{k}}{N}$

$P\left(C_{k}\right)=\frac{1}{K}$

or if $N_{k}$ is not known:

$p_{k,j}=\frac{N_{k}(x_{j}=1)}{N_{k}}$

$𝑝 𝑘 , 𝑗 = min 𝑁 𝑘 −1, max 1, 𝑁 𝑘 ( 𝑥 𝑗 =1) 𝑁 𝑘$

or smoothed:

[MATH_ERROR]

The code on the right hand side shows the sentiment analysis implementation:

We utilize the Twitter samples data from the nltk corpus, consisting of 5,000 positive and 5,000 negative labeled tweets. These tweets are read and labeled accordingly. Additionally, we create a list of stop words, create a Snowball stemmer, and utilize a tweet tokenizer that recognizes Twitter-specific tokens like hashtags, user tags, and emoticons.

In the process of cleaning the tweets, we eliminate HTTP links and user tags, as they are not relevant for sentiment analysis. We employ the Twitter tokenizer and remove single-letter tokens, numbers, and stop words. However, we retain emoticons like “:-)” since they can carry sentiment information.

We divide the training and test data into an 80:20 ratio while ensuring an even distribution of positive and negative tweets in both the training and test subsets through stratification.

We obtain a classifier from the nltk Naïve Bayes training and then assess its training accuracy (0.999) and test accuracy (0.995). The Naïve Bayes classifier makes only 16 incorrect predictions out of 10,000 samples. Some of the most informative features for this classifier are “:)” and “:(“, among others.

In this scenario, Naïve Bayes is not only highly accurate but also remarkably fast, with training and classification taking less than a second. None of the deep learning methods can compete with this speed.


## 14.2.3 Text Classification with Deep Learning


Sentiment analysis falls under text classification, and Naïve Bayes methods can be expanded to handle broader classification tasks. In this section, we explore the application of deep learning techniques to tackle more complex classification challenges. It is essential to begin by discussing the fundamental differences beforehand:

  - Naïve Bayes is a straightforward, yet highly effective and efficient method capable of real-time classification with low resource demands. Training and re-training are quick and straightforward, and model parameters occupy minimal storage. Storage consumption and performance can be further enhanced by selecting a subset of the most informative features. Consequently, Naïve Bayes, along with other simple classifiers like XGBoost or SVM, serves as an excellent initial choice. More advanced methods should only be considered when they can substantiate increased resource requirements with significantly higher accuracy.

  - Consider the sentiment analysis results from previous sections. Naïve Bayes achieves high accuracy, scoring 0.995 with only 16 false predictions out of 10,000 samples. While theoretically, we could opt for a deep learning approach like a transformer-based sentiment analyzer, such a model would not classify tweets as quickly as Naïve Bayes. In fact, a basic RoBERTa model takes seconds to minutes for classifying 10,000 samples (dependent on available hardware). Even if it achieved perfect accuracy (100%), the enhanced quality would not justify the significantly greater resource requirements.

  - Simpler models like Naïve Bayes rely on the independence assumption. In many complex scenarios, this assumption does not hold, leading to a rapid decline in the performance of simple models. While we can employ lemmatization techniques to enhance quality, these models cannot capture dependencies. On the other hand, deep learning models can adapt to complex scenarios and are versatile enough to handle various classification tasks without requiring substantial architectural changes, or additional training for new labels.

In this section, we examine the architecture of TextCNN and transformer-based classification architectures which offer distinct approaches to text classification. TextCNN utilizes convolutional layers to extract features from text embeddings, making it effective for capturing local patterns in data. In contrast, transformers excel in handling long-range dependencies through self-attention mechanisms, making them ideal for tasks requiring a broader context understanding. While textCNN is computationally efficient and interpretable, transformers are highly flexible and excel in tasks demanding nuanced contextual understanding. The choice between the two depends on the specific requirements of the classification problem, with textCNN being suitable for simpler tasks, and transformers shining in more complex, context-sensitive scenarios.

TextCNN-Architecture

This

is

an

example

for

a

deep

learning

model

d-dimensionalembeddings

feature maps for sequencesof 2, 3, 4, and 5 tokens(3 maps for each)

$⊛$

max pooling

concat

fullyconnected

softmax

class 1

class 2

class 3

class 4

class 5

1D-convolution

TextCNN Architecture

  - Tokens are converted into d-dimensional embedding vectors and fed into the network. Unlike the transformers architecture, the sequence length is treated as an input dimension, not an architectural parameter. This allows us to handle sequences of arbitrary length and apply convolutions to both short and long sentences without the need for padding. We can select any method and dimensionality for the embeddings.

  - We can utilize a set of feature maps to perform 1D convolutions on the sequence of embeddings. A feature map consists of weights of size $n×d×m$, where $n$ represents the number of consecutive embeddings in the sequence window, $d$ is the embedding dimensionality, and $m$ denotes the number of output values from the convolution. The feature map traverses the sequence, applying 1D convolution to the next $n$ embeddings, adding a bias, and applying an activation function for an output value. With $m$ feature maps, we compute $m$ output values for each position. With a sequence length of $s$, this results in $s−n+1$ values for each of the $m$ feature maps.

  - As the input sequence can vary in length, the next step employs max pooling to condense the $s−n+1$ values from each feature map into a single value. These resulting values are then concatenated into a vector of fixed length. In this example, we utilized feature maps of dimensions $2×d×3$, $3×d×3$, $4×d×3$, and $5×d×3$, resulting in a concatenated feature vector of dimensions $4∗3=12$ as the output of the convolutional layer.

  - A fully connected network translates this 12-dimensional vector into $k$ logits (in our example, $k=5$) and then applies a softmax function to predict the text's associated class.

Transformer-based classification leverages pre-trained transformer models like BERT or GPT as the core, extending them with extra layers to make class predictions. Typically, in transformer-based classification, we initiate a sequence with a model-specific token (e.g., [CLS] for BERT) and utilize the corresponding encoder output vector. This vector is then passed through a deep classification layer, which computes the logits for the $k $classes related to the task. A softmax function is applied to determine the class to which the input text belongs.

There are two ways to train the model for a given classification task:

  - The base transformer model (also called foundation model) is frozen and we only train the parameters of the additional classification layers. The foundation model can be shared across various classification tasks.

[CLS]Question [SEP]

BERT

classification layer

softmax

  - Both the base transformer model and the classification layer are trained together. This leads to a fine-tuned foundation model optimized for the classification task, but requires separate models for each classification task.

Modern large language models can easily extract classification information from text documents through prompt engineering. By providing a text document, the prompt asks the language model to extract the desired classes in a specific output format, like JSON.

class 1

class 2

class 3

class 4

class 5

In the year 1878 I took my degree of Doctor of Medicine of the University of London, and proceeded to Netley to go through the course prescribed for surgeons in the army. Having completed my studies there, I was duly attached to the Fifth Northumberland Fusiliers as Assistant Surgeon. The regiment was stationed in India at the time, and before I could join it, the second Afghan war had broken out. On landing at Bombay, I learned that my corps …

{

"news": 0.1,

"novel": 0.1,

"social media post": 0.1,

"product review": 0.1,

"lyrics": 0.7

}

  - The analysis suggests that the given text is most likely song lyrics, with a moderately high confidence. The key factors that led to this classification are:

  - - The repetitive, rhythmic structure of the text, with short, poetic lines, is characteristic …

  - <prompt>

  - You are an advanced natural language processing (NLP) model trained to classify a given text document into predefined categories and output the results in a JSON format.

  - Input: You will be given a text document. Your task is to analyze the content of the document and determine the classification(s) for it.

  - Output: Your output should be a JSON object with the following structure: …

