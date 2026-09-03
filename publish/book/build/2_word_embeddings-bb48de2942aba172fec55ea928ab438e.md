---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Semantic Search
section: Word Embeddings
order: "5.2"
---

(word-embeddings)=
# Word Embeddings

Word embeddings map each word in a vocabulary to a dense vector of $d$ dimensions (typically 100-300), where words appearing in similar contexts receive similar vectors. Unlike LSI, which derives representations from a global term-document matrix, word embeddings learn from local context windows and scale efficiently to billions of training tokens. This section covers the foundational models: Word2Vec, GloVe, and fastText.

## From LSI to Embeddings

While both LSI and word embeddings produce dense vector representations, they differ in fundamental ways:

| Aspect | LSI | Word Embeddings |
|--------|-----|-----------------|
| Unit of representation | Documents (and terms as by-product) | Words (documents via aggregation) |
| Context signal | Global co-occurrence across entire documents | Local context windows (typically 5-10 words) |
| Training | SVD on term-document matrix | Neural network with gradient descent |
| Vocabulary | Fixed; retraining needed for new terms | Sub-word variants possible (fastText) |
| Transferability | Corpus-specific; poor cross-domain | Language-level; transfers across collections |
| Scalability | $O(mnk)$ for SVD; expensive for large corpora | $O(n \cdot d \cdot w)$; scales to web corpora |

The key conceptual shift is the **distributional hypothesis**: a word's meaning is defined by the company it keeps (Firth, 1957). Word2Vec operationalizes this by training a neural network to predict either context from a center word (Skip-Gram) or a center word from its context (CBOW).

## Word2Vec

Word2Vec, introduced by Mikolov et al. in 2013, maps words to a $d$-dimensional vector space using self-supervised learning on raw text. No labeled data is required. It has two architectures:

### Skip-Gram

The Skip-Gram model operates within a context window of size $2m+1$ around a center word and learns to predict context words from the center word. For the phrase "the dog chases a cat" with center word "chases" and $m=2$:

$$
P(\text{the, dog, a, cat} \mid \text{chases}) = P(\text{the} \mid \text{chases}) \cdot P(\text{dog} \mid \text{chases}) \cdot P(\text{a} \mid \text{chases}) \cdot P(\text{cat} \mid \text{chases})
$$

```{figure} images/figure_5_10.png
:name: fig-skipgram-context
:width: 50%

Dependency tree for "the dog chases a cat". In the Skip-Gram model, the center word "chases" predicts each surrounding word independently.
```

Each word $w_i$ has two $d$-dimensional vectors: $\mathbf{v}_i$ when used as a center word, and $\mathbf{u}_i$ when used as a context word. The conditional probability uses a softmax over the entire vocabulary $\mathbb{T}$:

$$
P(w_s \mid w_c) = \frac{\exp(\mathbf{u}_s^\top \mathbf{v}_c)}{\sum_{i \in \mathbb{T}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}
$$

```{admonition} Key Formula: Skip-Gram Loss
:class: important

$$
\mathcal{L} = -\sum_{i=m+1}^{n-m} \sum_{\substack{j=i-m \\ j \neq i}}^{i+m} \log P(w_j \mid w_i)
$$

The loss sums the negative log-probability of each context word given its center word, over all windows in the corpus. Minimizing this loss via gradient descent produces the word vectors $\mathbf{v}_i$.
```

```{figure} images/figure_5_11.png
:name: fig-skipgram-architecture
:width: 80%

Skip-Gram as a neural network: a one-hot input (center word) is projected through a weight matrix (yielding $\mathbf{v}_i$), passed through $d$ hidden neurons with linear activation, then projected to vocabulary size (yielding $\mathbf{u}_i$), and compared to the target context word via softmax and cross-entropy loss.
```

### CBOW

The Continuous Bag of Words (CBOW) model reverses the prediction direction: it predicts the center word from the average of its context word vectors:

$$
P(w_c \mid w_{c-m}, \ldots, w_{c+m}) = \frac{\exp\!\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{c-m} + \cdots + \mathbf{v}_{c+m})\right)}{\sum_{i \in \mathbb{T}} \exp\!\left(\mathbf{u}_i^\top (\mathbf{v}_{i-m} + \cdots + \mathbf{v}_{i+m})\right)}
$$

```{figure} images/figure_5_12.png
:name: fig-cbow-context
:width: 50%

In CBOW, all context words point to the center word "chases", which serves as the prediction target.
```

```{figure} images/figure_5_13.png
:name: fig-cbow-architecture
:width: 80%

CBOW architecture: context word vectors are averaged into a single $|\mathbb{T}|$-dimensional input, passed through a hidden layer of $d$ neurons, then a softmax classifier predicts the center word.
```

CBOW produces one training sample per window (context → center word), while Skip-Gram produces $2m$ samples (center word → each context word). In practice, Skip-Gram performs better on rare words because it generates more training signal per occurrence, while CBOW trains faster and works well for frequent words.

### Training

A PyTorch implementation uses the `Embedding` layer to avoid explicit one-hot vectors:

```python
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.linear(x)
        return x
```

Training creates pairs from context windows across the corpus. Sub-sampling frequent words (reducing pairs with stop words to ~1% of their count) improves quality and speed.

## GloVe and Subword Models

### GloVe

GloVe (Pennington et al., 2014) also learns word vectors from co-occurrence, but uses a global co-occurrence matrix rather than local windows. Let $X_{ij}$ count how often word $w_j$ appears in the context of $w_i$. GloVe's insight is that the *ratio* of co-occurrence probabilities $P_{ik}/P_{jk}$ distinguishes related from unrelated word pairs better than raw counts.

The model optimizes a weighted least-squares objective:

```{admonition} Key Formula: GloVe Cost Function
:class: important

$$
J = \sum_{i,j=1}^{n} f(X_{ij}) \cdot \left(\mathbf{u}_i^\top \mathbf{v}_j + b_i + c_j - \log X_{ij}\right)^2
$$

The weighting function $f(x) = \min((x / x_{\max})^\alpha, 1)$ prevents very frequent pairs from dominating, while the log co-occurrence target captures the ratio relationships.
```

### fastText

fastText (Bojanowski et al., 2017) improves Word2Vec by representing each word as the sum of its sub-word (character n-gram) vectors. For the center word "chases" with 3-grams:

$$
\mathbf{v}_{\text{chases}} = \sum_{g \in \mathcal{Z}} \mathbf{z}_g = \mathbf{z}_{\text{<ch}} + \mathbf{z}_{\text{cha}} + \mathbf{z}_{\text{has}} + \mathbf{z}_{\text{ase}} + \mathbf{z}_{\text{ses}} + \mathbf{z}_{\text{es>}}
$$

This sub-word approach has two advantages:

1. **Morphological generalization**: "chasing", "chased", and "chases" share sub-words, so they receive related vectors without explicit stemming.
2. **Handling unknown words**: any new word can be represented by combining its sub-word vectors, even if the word itself was never seen during training.

fastText also introduces **negative sampling** to replace the expensive softmax over the full vocabulary. Instead of normalizing across all $|\mathbb{T}|$ terms, it contrasts the true context pair against $m$ randomly sampled negative words, dramatically reducing computation.

Sub-word lengths can be fixed (3-6 characters) or variable using BPE or WordPiece algorithms, trading vocabulary size against representation granularity.

## Emergent Properties and Limitations

Word embeddings trained on large corpora exhibit remarkable emergent structure:

- **Semantic clustering**: words with related meanings form clusters in the embedding space.
- **Analogies**: vector arithmetic captures relational patterns. The classic example: $\mathbf{v}_{\text{Paris}} - \mathbf{v}_{\text{France}} + \mathbf{v}_{\text{Germany}} \approx \mathbf{v}_{\text{Berlin}}$.
- **Odd-one-out detection**: the word least similar to a group can be identified by distance from the group centroid.

```{figure} images/figure_5_14.png
:name: fig-embedding-clusters
:width: 75%

PCA projection of GloVe embeddings for 25 words. Semantic clusters emerge: animals (bottom-left), meals (top), cities (right), and people (bottom-center) occupy distinct regions despite no explicit category supervision.
```

```{admonition} Example
:class: example

Using pre-trained GloVe vectors (100 dimensions, trained on Wikipedia):
```python
import gensim.downloader
wv = gensim.downloader.load('glove-wiki-gigaword-100')

# Most similar to "cat"
wv.most_similar("cat", topn=3)
# [('dog', 0.92), ('cats', 0.85), ('pet', 0.80)]

# Analogy: Paris - France + Germany = ?
wv.most_similar(positive=["paris", "germany"], negative=["france"], topn=1)
# [('berlin', 0.89)]

# Odd one out
wv.doesnt_match(["bird", "dog", "cat", "town"])
# 'town'
```
```

Despite these strengths, static embeddings have a fundamental limitation: **each word receives a single vector regardless of context**. The word "bank" gets the same embedding whether it means a financial institution or a riverbank. In the sentence "I went to the bank on Park Street; I sat and watched the boats go by", humans resolve the ambiguity instantly, but Word2Vec cannot.

This polysemy problem, combined with the lack of word-order sensitivity, motivated the development of contextual embeddings using transformer architectures, covered in the next section.
