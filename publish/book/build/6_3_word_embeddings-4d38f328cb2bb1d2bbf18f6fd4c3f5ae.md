# Word Embeddings

Word embeddings, like latent semantic indexing, map words (or sub-words) to a $d$-dimensional space, where $d$ is much smaller than the vocabulary size, usually ranging from 100 to 1,000 dimensions. The key distinction from LSI is as follows:

  - LSI maps documents to vectors, while embeddings map vocabularies to vectors. Modern transformer-based embeddings, however, can generate embeddings for entire sequences, capturing richer semantics than earlier models that aggregated vectors along the sequence.

  - LSI examines semantic relationships between terms globally, without considering the distance or context of term occurrences. Earlier embedding models used defined context windows around words to establish these relationships. Transformer-based models address the challenge of sequence-to-sequence transformations by employing self-attention mechanisms to acquire contextualized word embeddings.

  - LSI reduces dimensionality and minimizes the loss on the original document-term matrix with lower-dimensional representations. However, SVD can be computationally expensive and does not scale efficiently (although it can handle millions of documents). Embeddings employ neural network-based learning techniques to optimize the loss between predictions and targets, offering a more efficient approach.

  - LSI employs a static vocabulary and necessitates retraining to accommodate new terms (such as names or brands). Earlier embedding models are vocabulary-based as well, but some operate at a sub-word level, enabling them to handle unknown or misspelled terms. Transformer-based models utilize pre-trained sub-words like BPE and word pieces to generate embeddings for a smaller vocabulary. They can address unseen words by breaking them into sub-parts and providing embeddings for these smaller tokens.

  - LSI mappings can theoretically be applied to other collections, but this often results in suboptimal performance because latent topics are shaped by both documents and terms specific to a collection. Therefore, an LSI model tailored for IT might not excel with biology articles. Conversely, embeddings create mappings based on terms and context specific to the language, enabling reuse in different but similar collections. However, optimal performance with embeddings requires collection-specific optimization. For instance, embeddings from one language do not transfer well to another language. Transformer-based models may also face issues with suboptimal tokens (e.g., BPE, word pieces) when transferred to different languages, affecting embedding quality.


## 6.3.1 Word2vec


Word2Vec, introduced in 2013 by Mikolov et al., maps words to a $d$-dimensional vector space. The mapping uses context windows around each word and is trained in a self-supervised way, so no labeled data is needed. Word2Vec has two variants:

  - The Skip-Gram Model operates within a context window of size $2m+1$ around a center word. It learns word representations by predicting context words from the center word. For example, if the center word is “apple”, the model predicts which words are likely to appear in the context window, like “juice”, “tree”, “red”, or “eat”. Words like “complex”, “retrieval”, “planet”, and “learn” are less likely to be found near “apple”.

  - The Continuous Bag of Words (CBOW) Model employs a similar method with a context window of size $2m+1$. However, it learns representations to predict the center word from all the context words. For instance, in the sentence “the apple is [blank] and tastes delicious”, CBOW would aim to predict the center word "[blank]" based on the surrounding words “the”, “apple”, “is”, “and”, “tastes”, and “delicious”. A word like “ripe” is a better match than the word “car”.

Both models yield a mapping from the vocabulary to a $𝒅$-dimensional vector which can be used for different tasks:

  - Semantic Word Analysis: Vector representations establish semantic relationships between terms, enabling the use of similarity measures (e.g., cosine, Euclidean, dot-product) to identify closely related terms. These relationships are learned and adapted for the collection, eliminating the need for manual dictionary curation.

  - Token Classification: Word embeddings can enhance part-of-speech and named entity recognition, replacing the need for manual or rule-based methods.

  - Machine Translation: When translating between languages, word embeddings assist in selecting the optimal words and arranging them in the target language by considering the broader context. This helps to resolve ambiguities and to improve translation quality.

  - Text Classification: By representing entire documents as sequences of vectors using embeddings, we can enhance machine learning approaches. These semantically rich representations can lead to improved quality even with simple models. Similarly, embeddings help to discover latent topics in collections.

  - Text Retrieval: Word embeddings' semantic relationships improve query-document matching in retrieval tasks, which we will explore in greater detail later in this section.

The Skip-Gram Model: Let's take the phrase “the dog chases a cat” with its center word “chases” and a context window of size $2m+1=5$ ($m=2$ words before and $m=2$ words after the center word). In the skip-gram model, we assess the conditional probability of the center word generating the surrounding words, assuming independence among the surrounding words:

  - We can illustrate this relationship graphically as follows:

  - In the skip-gram model, a word $w_{i}$ is represented by two $d$-dimensional vectors $𝒗_{i}\in ℝ^{d}$ and $𝒖_{i}\in ℝ^{d}$ when employed as a center word ($𝒗_{i}$) or as a surrounding word ($𝒖_{i}$). We can employ a softmax operation to model the conditional probability of generating the surrounding word $w_{s}$ from the center word $w_{c}$:

    - where $𝕋$ represents the set of words in a corpus of documents. Assuming the corpus consists of a sequence of $n$ words $w_{1}…w_{n}$, the likelihood function for the skip-gram model is expressed as:

    - with a context window of size $2m+1$. The objective is to find vectors $𝒗_{i}\in ℝ^{d}$ and $𝒖_{i}\in ℝ^{d}$ that maximize the likelihood function, and we then can use the mapping $w_{i}\rightarrow v_{i}$ to translate words from the vocabulary to a $d$-dimensional vector.

$P\left(the dog the cat | chases\right)=P\left(the|chases\right)∙P\left(dog|chases\right)∙P\left(a|chases\right)∙P\left(cat|chases\right)$

chases

the

dog

a

cat

$P\left(w_{s} | w_{c}\right)=\frac{e^{𝒖_{s}^{T}𝒗_{c}}}{\sum_{i\in 𝕋 }^{}e^{𝒖_{i}^{T}𝒗_{c}}}$

$\prod_{i=m+1}^{n−m}\prod_{j=i−m, j\ne i}^{i+m}P\left(w_{j}|w_{i}\right)$

  - To optimize the model's likelihood function, we can minimize the following loss function instead:

    - We cannot directly find a solution for the above optimization problem. Instead, we employ a gradient descent method during a training phase to minimize the loss. An alternative approach is the modeling of the optimization problem with a single hidden layer network, as shown at the bottom of this page. The model takes a one-hot vector representing the center word as input. It then passes through a fully connected network without bias and activation function. The columns in the corresponding weight matrix represent vectors $𝒗_{i}$. The hidden layer comprises $d$ neurons and is followed by another fully connected network, again without bias and activation function. The columns in this matrix correspond to the vectors $𝒖_{i}$. A softmax classifier on the output layer is compared with the target one-hot vector of surrounding words.

0

0

0

1

0

0

0

0

0

0

0

0

one-hotvector

center wordrepresentation($\left|𝕋\right|$ dimensional)

fullyconnected $𝒗_{i}$

$d$ hidden neurons(linear activation)

0

0

0

0

0

0

0

0

1

0

0

0

one-hotvector

each surrounding wordis a training target($\left|𝕋\right|$ dimensional)

output layersoftmax classifier($\left|𝕋\right|$ dimensional)

error

function

$− 𝑖 = 𝑚 +1 𝑛 − 𝑚 𝑗 = 𝑖 − 𝑚 , 𝑗 ≠ 𝑖 𝑖 + 𝑚 log 𝑃 𝑤 𝑗 | 𝑤 𝑖$

fullyconnected $𝒖_{i}$

no bias and activation function

  - We can employ a self-supervised approach to train the model, which means we can utilize supervised learning techniques without relying on external or human-provided labels. To train word2vec models using a large text corpus, we create the training, test, and validation datasets from the corpus as follows:

    - We enumerate all windows of size $2m+1$ in the corpus. To avoid incorrect associations across sentence boundaries, we first split texts into sentences and ensure that windows are confined within sentences.

    - For every window, we generate pairs of center word and surrounding word. In each window, we produce $2m$ data samples. As an example, consider the window “the red apple tastes fine”. We create four pairs: (apple, the), (apple, red), (apple, tastes), and (apple, fine). Each of these pairs contributes to training the model.

    - Optionally, we can sub-sample or exclude pairs with common terms (stop words). Sub-sampling reduces the numbers of pairs often to as low as 1%. This enhances accuracy and accelerates the training.

    - Optionally, we can lemmatize or tokenize words (e.g., stemming) to decrease vocabulary size, but this may restrict the applicability of vectors for some contexts.

  - During training, we utilize pairs of center and surrounding word to compute the loss function and make adjustments to the model weights, that is the vectors $𝒖_{i}$ and $𝒗_{i}$. This process is iterated until the loss function reaches a sufficiently low value. The outcome is a vocabulary-to-vector mapping, represented as $𝒗_{i}$.

  - To efficiently handle word pairs without using memory-intensive one-hot vectors, a PyTorch implementation can use the Embedding layer and the CrossEntropyLoss function, directly working with token IDs instead of vectors. The code below defines the model on the left side, and outlines the training process on the right side.

class SkipGramModel(nn.Module):

def __init__(self, vocab_size: int, embedd_size: int):

super().__init__()

self.embeddings = nn.Embedding(

num_embeddings=vocab_size,

embedding_dim=embedd_size)

self.linear = nn.Linear(

in_features=embedd_size,

out_features=vocab_size,

)

def forward(self, inputs):

x = self.embeddings(inputs)

x = self.linear(x)

return x

# sketch of training process

model = SkipGramModel(vocab_size=len(vocab), embedd_size=100)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

for inputs, labels in batch_loader_skipgram(text_corpus):

optimizer.zero_grad()

outputs = model(inputs)

loss = loss_fn(outputs, labels)

loss.backward()

optimizer.step()

Word2vec also defines an alternative model known as the Continuous Bag of Words (CBOW) model. It operates similarly to the skip-gram model but focuses on the likelihood that the surrounding words generate the center word. To illustrate, let's revisit the phrase “the dog chases a cat” with the center word “chases” and a context window of size $2m+1=5$. We evaluate the conditional probability of the surrounding words generating the center word, with an assumption of independence among the surrounding words:

  - We can illustrate this relationship graphically as follows:

  - In the CBOW model, a word $w_{i}$ is represented by two $d$-dimensional vectors $𝒗_{i}\in ℝ^{d}$ and $𝒖_{i}\in ℝ^{d}$ when employed as a center word ($𝒗_{i}$) or as a surrounding word ($𝒖_{i}$). We can employ a softmax operation to model the conditional probability of generating the center word $w_{c}$ from its surrounding words $w_{c−m}, …,w_{c−1},w_{c+1},…,w_{c+m}$:

    - where $𝕋$ represents the set of words in a corpus of documents. Assuming the corpus consists of a sequence of $n$ words $w_{1}…w_{n}$, the likelihood function for the skip-gram model is expressed as:

    - with a context window of size $2m+1$. The objective is to find vectors $𝒗_{i}\in ℝ^{d}$ and $𝒖_{i}\in ℝ^{d}$ that maximize the likelihood function, and we then can use the mapping $w_{i}\rightarrow v_{i}$ to translate words from the vocabulary to a $d$-dimensional vector.

$P\left(chases | the dog the cat\right)$

chases

the

dog

a

cat

$P\left(w_{c} | w_{c−m},…,w_{c−1},w_{c+1},…,w_{c+m}\right)=\frac{e^{\frac{1}{2m}𝒖_{c}^{T}\left(𝒗_{c−m}+…+𝒗_{c−1}+𝒗_{c+1}+…+𝒗_{c+m}\right)}}{\sum_{i\in 𝕋 }^{}e^{𝒖_{i}^{T}\left(𝒗_{i−m}+…+𝒗_{i−1}+𝒗_{i+1}+…+𝒗_{i+m}\right)}}$

$\prod_{i=m+1}^{n−m}P\left(w_{c} | w_{c−m},…,w_{c−1},w_{c+1},…,w_{c+m}\right)$

no bias and activation function

  - To optimize the model's likelihood function, we can minimize the following loss function instead:

    - We cannot directly find a solution for the above optimization problem. Instead, we employ a gradient descent method during a training phase to minimize the loss. An alternative approach is the modeling of the optimization problem with a single hidden layer network, as shown at the bottom of this page. Contrary to the skip-gram-model, the input is now an $2m$-hot vector with each surrounding word setting a component to $1/2m$. It then follows the same structure as with the skip-gram model, passing through a fully connected network without bias and activation function. The columns in the corresponding weight matrix represent vectors $𝒗_{i}$. The hidden layer comprises $d$ neurons and is followed by another fully connected network, again without bias and activation function. The columns in this matrix correspond to the vectors $𝒖_{i}$. A softmax classifier on the output layer is compared with the target one-hot vector of the center word.

0

0

0

1/4

0

0

1/4

0

1/4

0

0

1/4

0

all words in surroundingaverage vectors

averaged wordsin surrouding($\left|𝕋\right|$ dimensional)

fullyconnected $𝒗_{i}$

$d$ hidden neurons(linear activation)

0

0

0

0

0

0

0

0

1

0

0

0

one-hotvector

center word is thetraining target($\left|𝕋\right|$ dimensional)

output layersoftmax classifier($\left|𝕋\right|$ dimensional)

error

function

$− 𝑖 = 𝑚 +1 𝑛 − 𝑚 log 𝑃 𝑤 𝑐 | 𝑤 𝑐 − 𝑚 ,…, 𝑤 𝑐 −1 , 𝑤 𝑐 +1 ,…, 𝑤 𝑐 + 𝑚$

fullyconnected $𝒖_{i}$

  - We can employ a similar self-supervised approach to train the model as with the skip-gram model. To train CBOW models using a large text corpus, we create the training, test, and validation datasets as follows:

    - We enumerate all windows of size $2m+1$ in the corpus as with the skip-gram model.

    - For every window, however, we generate only one pair of surrounding words and center word. As an example, consider the window “the red apple tastes fine”. The data sample is given now as ([the, red, tastes, fine], apple).

    - Optionally, we can sub-sample or exclude pairs with common terms (stop words), and lemmatize or tokenize words (e.g., stemming) to decrease vocabulary size similar as discussed with the skip-gram model.

  - During training, we utilize pairs of surrounding words and center word to compute the loss function and make adjustments to the model weights, that is the vectors $𝒖_{i}$ and $𝒗_{i}$. This process is iterated until the loss function reaches a sufficiently low value. The outcome is a vocabulary-to-vector mapping, represented as $𝒗_{i}$.

  - To efficiently handle word pairs without using memory-intensive one-hot vectors, a PyTorch implementation can use the Embedding layer and the CrossEntropyLoss function, directly working with token IDs instead of vectors. The key difference in the model code, shown on the left, is that inputs are now vectors of token IDs (representing the surrounding words), rather than a scalar (representing one surrounding word). The model first maps all surrounding words to their embeddings and then averages them to incorporate the $1/2m$ input encoding from the previous page. The training process, depicted on the right side, is similar to the skip-gram model, except for the process to generate data batches

class CBOWModel(nn.Module):

def __init__(self, vocab_size: int, embedd_size: int):

super().__init__()

self.embeddings = nn.Embedding(

num_embeddings=vocab_size,

embedding_dim=embedd_size

)

self.linear = nn.Linear(

in_features=embedd_size,

out_features=vocab_size,

)

def forward(self, inputs):

x = self.embeddings(inputs)

x = x.mean(axis=1)

x = self.linear(x)

return x

# sketch of training process

model = CBOWModel(vocab_size=len(vocab), embedd_size=100)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

for inputs, labels in batch_loader_cbow(text_corpus):

optimizer.zero_grad()

outputs = model(inputs)

loss = loss_fn(outputs, labels)

loss.backward()

optimizer.step()


## 6.3.2 GloVe and fastText


GloVe (Global Vectors for Word Representation) is a self-supervised embedding algorithm developed at Stanford in 2014. Like other methods, it considers word co-occurrences within context windows. However, it differs in its loss function, which relies on co-occurrence probability ratios:

  - Let $X_{ij}$ represent the number of occurrences of word $w_{j}$ within the context of word $w_{i}$, defined by a window around $w_{i}$. Then, $X_{i}=\sum_{k}^{}X_{ik}$ denotes the total number of co-occurrences for word $w_{i}$. Define $P_{ij}=P(j|i)=X_{ij}/X_{i}$ as the probability of word $w_{j}$ appearing in the context of word $w_{i}$.

  - The initial idea is to examine the relationship between two words, $w_{i}$ and $w_{j}$, by analyzing the ratio $P_{ik}$/$P_{jk}$ in relation to a third word, $w_{k}$. If $w_{i}$ and $w_{k}$ are related while $w_{j}$, and $w_{k}$ are not, the ratio $P_{ik}$/$P_{jk}$ will be large. Conversely, if $w_{j}$, and $w_{k}$ are related but $w_{i}$ and $w_{k}$ are not, the ratio $P_{ik}$/$P_{jk}$ will be small. When both $w_{i}$ and $w_{j}$, are either related or unrelated to $w_{k}$, the ratio should be approximately 1. Here is the example for $w_{i}$=“ice” and $w_{j}$=“steam” from the original paper:

  - We introduce vectors $u_{i}$ to represent words in a $d$-dimensional space and vectors $v_{k}$ to represent context words in a $d$-dimensional space (adopting notation similar to word2vec, deviating from the original paper's notation). Additionally, we require biases $b_{i}$ for words and $c_{j}$ for context words. Using the relationships described earlier, we can build a cost function employing weighted least squares:

    - The function $f(x)=min⁡((x/x_{max})\^\alpha , 1)$ assigns weights to co-occurrences, giving higher weights to more frequent ones while avoiding excessive emphasis on very frequent co-occurrences. Similar to word2vec, we create pairs from the corpus and train the parameters to optimize the cost function. The resulting vectors $u_{i}$ represent word vectors. A detailed training algorithm discussion is omitted here due to its similarity to word2vec.

$𝐽 = 𝑖 , 𝑗 =1 𝑛 𝑓 𝑋 𝑖𝑗 ∙ 𝑢 𝑖 𝑇 ∙ 𝑣 𝑗 + 𝑏 𝑖 + 𝑐 𝑗 − log 𝑋 𝑖𝑗 2$

fastText was developed by Facebook's AI Research (FAIR) team and first published in 2017. It improves word2vec algorithms in two ways:

  - fastText employs sub-word representations for center words and constructs word vectors by summing the vectors of their sub-words. In the context of a phrase like “the dog chases a cat” with “chases” as the center word, the context words (“the”, “dog”, “a”, and “cat”) maintain their full-word vectors $u_{j}$ and are not split into sub-words. However, the center word “chases” is broken into sub-words, for instance, of length 3, such as “<ch”, “cha”, “has”, “ase”, “ses”, and “es>”. The special characters “<“ and “>” mark the start and the end of the word allowing for the differentiation of prefixes, suffixes, and sub-words in the middle of a word. fastText considers sub-words ranging from length 3 to 6 within the annotated word (“<chases>”). Let's denote the set of sub-words for “chases” as $ℤ$, and assign vector representations $z_{g}$ to each sub-word $g\in ℤ$. The vector representation $v_{i}$ for “chases” is then calculated as the sum of its sub-word representations:

    - We then integrate this representation of the center word into the skip-gram model, as illustrated on the right side, and proceed to optimize the loss function as described in the word2vec part. This process yields vectors $z_{g}$ for the sub-words, enabling us to build of word embeddings for any words, including those not present in the corpus.

  - fastText expedites training by subsampling frequent co-occurrences and utilizing a negative sampling technique. Instead of comparing the softmax result to the one-hot target, both the center (sub-)word and the surrounding word are mapped to their respective $d$-dimensional representations, and their vectors are compared using dot products. In addition to the positive sample, $𝒎$ negative samples (words not found in the center word's surroundings) are included to expedite training iterations and enhance weight matrix updates. Detailed information can be found in the relevant research papers. As result, fastText can reduce the computational complexity and produce embeddings faster than the original word2vec approach

  - The sub-word approach can be customized from fixed sizes (3 to 6) to variable lengths using methods like byte pair encoding (BPE) or word pieces algorithms. This results in more concise vocabularies and enables the encoding of diverse Unicode words, as previously discussed in the tokenization section. For instance, English has approximately 3 x 10^8 possible 6-grams, and it may not be efficient to store representations for all of them. By utilizing BPE and word pieces, we can establish an upper limit on the storage required for embeddings.

$v_{i}=\sum_{g\in ℤ}^{}z_{g}$

$P\left(w_{s} | w_{c}\right)=\frac{e^{𝒖_{s}^{T}𝒗_{c}}}{\sum_{i\in 𝕋 }^{}e^{𝒖_{i}^{T}𝒗_{c}}}$

skip-gram


## 6.3.3 Practical Applications


The code on the right side illustrates how to use the gensim package for learning and utilizing embeddings:

Gensim can train models with word2vec or fastText, as the code shows. For better embeddings, use large collections of sentences. The movie reviews example may not produce reliable embeddings.

The gensim package includes several pre-trained embeddings, such as GloVe vectors, which provide general-purpose embeddings mainly for English. Because these models were trained on diverse topics, they may not give the best results for a specific collection.

Embeddings let us study word relationships. We create vector representations for words, for example "cat." We then compare these vectors to find the most similar words. Similarity comes from shared contexts, so "dog" is often very similar to "cat." We can also do arithmetic with vectors, for example "Paris + Germany - France = Berlin." Another useful task is finding the odd word out in a list, the word that does not belong with the others.

spaCy's models have vector embeddings (token.vector) in their pipelines. In the base models, spaCy employs floret, a variation of fastText that generates more space-efficient vector tables. Models like "en_core_web_trf" utilizes transformers (BERT based) for embeddings.

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import gensim.downloader

# get the glove-wiki-gigaword-100 word vectors

word_vectors = gensim.downloader.load('glove-wiki-gigaword-100')

# get vectors for terms

words = "animal bird dog cat horse fish bee … dinner lunch".split()

vectors = word_vectors[words]

# apply a PCA to map to 2 dimensions

pca = PCA(n_components=2)

result = pca.fit_transform(vectors)

# create a scatter plot of the projection

plt.figure(figsize=(14,10))

plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):

plt.annotate(word, xy=(result[i, 0]+0.05, result[i, 1]-0.05), 	    fontsize=14)

plt.show()

An effective demonstration of embeddings is provided in the figure below, along with the code for generating it. We employ a pre-trained GloVe set with 100 dimensions, trained on Wikipedia text. Utilizing 25 words, we map them to the GloVe vector space. To enhance visualization, we project these vectors into a 2-dimensional space using PCA:

  - In the previous example, we employed the doesnt_match function on the words “bird dog cat town”. In the visualization, it is now evident that the word “town” is distinctly separated from the animal-related words. This demonstrates how we can identify words that don't belong to a specific group.

  - We can also identify clusters of words, such as those related to animals, city names, or words associated with people. Instead of manually constructing relationships from dictionaries, we can now learn these associations between words automatically. It is important to note that these associations do not necessarily represent semantic relationships but indicate whether two words tend to appear in similar contexts. For example, in our previous example, “dog” was the most similar word to “cat”. This does not mean that cats are similar to dogs, but rather that they are frequently discussed in similar contexts, such as when people talk about their pets.

A larger online visualization for embedding is here: https://projector.tensorflow.org
