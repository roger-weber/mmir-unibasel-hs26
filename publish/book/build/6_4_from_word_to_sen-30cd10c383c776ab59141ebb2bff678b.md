# From Word to Sentence Embeddings

Word embeddings give rich semantic representations of words, but they are not directly suitable for text retrieval because a text is represented by hundreds or even thousands of separate vectors. To make retrieval possible, we must summarize these word-level embeddings into a single compact vector, similar to how TFIDF summarizes term importance in a document. This summarization, often called pooling, produces a fixed-length vector that captures the overall meaning of a sentence, paragraph, or document.

Two common pooling methods are average pooling and max pooling:

  - Average pooling (also called mean pooling) computes the element-wise average across all token embeddings in the sequence. Given $m$ token embeddings $𝑿=\{𝒙_{1}, 𝒙_{2},…, 𝒙_{m}\}$ where each $𝒙_{i}\in ℝ^{d}$ represents a token in a $d$-dimensional space, average pooling produces:

    - This method provides a balanced, holistic representation by ensuring every token contributes equally to the final embedding. It captures the overall semantic tone and general context of the text, making it particularly effective for tasks like semantic similarity and document retrieval where comprehensive understanding matters.

  - Max pooling selects the maximum value across each dimension of the token embeddings:

    - Where the maximum is taken for each of the $d$ dimensions separately. This highlights the most salient features in the text, capturing dominant signals that often correspond to keywords or other critical information. Max pooling is useful when particular tokens, such as "amazing" in sentiment analysis or domain-specific technical terms, carry disproportionate semantic weight.

$pool_{avg}\left(𝑿\right)=\frac{1}{m}\sum_{i=1}^{m}𝒙_{i}$

$pool_{max}\left(𝑿\right)=\left[\begin{matrix}\max_{1\leq i\leq m}x_{i,1}\\\max_{1\leq i\leq m}x_{i,2}\\\vdots \\\max_{1\leq i\leq m}x_{i,d}\end{matrix}\right]$

After documents and queries are converted into single vectors, we need a function to measure semantic fit. Cosine similarity is the common choice. It measures the angle between vectors, so it ignores vector length and captures only directional alignment. This works well for comparing word embeddings because those vectors are often not normalized; a raw dot product can be dominated by one unusually large value in a single dimension.

In practice, we normalize pooled embedding vectors when indexing documents and queries to convert cosine similarity into a dot product and greatly reduce computation during search. Let $\hat{𝒅}$ denote the normalized pooled document embedding and $\hat{𝒒}$ the normalized pooled query embedding. Then:

This framework enables efficient text retrieval: documents are embedded, pooled and normalized once during indexing, queries are embedded, pooled, and normalized at search time, and the dot-product ranks documents by semantic relevance (implementing a cosine measure on the original pooled embeddings).

To understand pooling mechanisms, consider how they work in the latent space of embeddings.

  - Average pooling computes a centroid in the embedding space. Each token embedding is a point in $d$-dimensional space, and averaging gives the geometric center of those points. When comparing a query to a document, we measure the angle between the query centroid and the document centroid using cosine similarity. Let $\hat{𝒒}$ and $\hat{𝒅}$ be the normalized centroids of the word embeddings for query $Q$ and document $D$. They are obtained as follows:

    - Using cosine similarity, we obtain the score between $Q$ and $D$ as follows (regrouping the sums):

$\hat{𝒅}=\frac{𝒅}{\left‖𝒅\right‖}$

$\hat{𝒒}=\frac{𝒒}{\left‖𝒒\right‖}$

$sim_{cos}\left(Q,D\right)=\frac{𝒒∙𝒅}{\left‖𝒒\right‖∙\left‖𝒅\right‖}   =\hat{𝒒}∙\hat{𝒅}$

$\hat{𝒒}=\frac{\overbar{𝒒}}{\left‖\overbar{𝒒}\right‖}=\frac{\frac{𝟏}{𝑵}\sum_{i=1}^{N}𝒒_{i}}{\left‖\overbar{𝒒}\right‖}$

$\hat{𝒅}=\frac{\overbar{𝒅}}{\left‖\overbar{𝒅}\right‖}=\frac{\frac{𝟏}{𝑴}\sum_{j=1}^{M}𝒅_{j}}{\left‖\overbar{𝒅}\right‖}$

$sim\left(Q,D\right)=\hat{𝒒}∙\hat{𝒅}=\frac{1}{\left‖\overbar{𝒒}\right‖∙\left‖\overbar{𝒅}\right‖}∙\frac{1}{N∙M}\sum_{i=1}^{N}\sum_{j=1}^{M}𝒒_{i}∙𝒅_{j}   $

  - Average pooling (continued)

    - From the last equation, average pooling compares every token in the query with every token in the document using the dot product, then normalizes and averages the results. Pairs of semantically similar tokens increase the sum, while dissimilar pairs decrease it. Summing over hundreds or thousands of tokens can dilute important information. For example, averaging 500 token embeddings gives each token only 0.2% of the final score, which can wash out distinctive features that distinguish documents. This dilution grows with document length, so the aggregated embedding tends to reflect generic, averaged meaning rather than nuanced content.

    - What about stop words? With average pooling, stop words can dominate the pooled vector because they appear often and pull the average toward directions in the embedding space that carry little meaning. Stop words typically account for 30% to 50% of words in natural language text. This large share reflects their role as grammatical building blocks, even though they add little semantic content. For example, in typical search queries about 12.9% of tokens are stop words, while in question queries about 39% are stop words. If a document has 40% stop words and a query has 30%, then content tokens make up 60% of the document and 70% of the query. Only $0.6∗0.7=0.42$, so just 42 percent of token pairs are between content tokens; the rest include at least one stop word. These stop words dilute the result and make documents harder to distinguish. For that reason, the standard approach is to remove stop words before applying the pooling function.

  - With max pooling, each embedding dimension takes the maximum value across all token embeddings. After the final normalization step, dominant dimensions show large values while less important dimensions are near zero. Max pooling has different challenges than average pooling. Although it preserves strong features, it discards contextual relationships between tokens by keeping only the maximum value for each dimension. For a document with 1,000 tokens, max pooling may retain information from only a small fraction of tokens (for example, 10 to 50), and it can miss important semantic nuances that come from token interactions, phrase structures, or syntactic dependencies. This makes it less sensitive to the broader context and more focused on highly salient individual token features.

    - Common words that occur in many contexts, like the stop words "the," "of," and "and," tend to have embeddings with small magnitudes near zero. Because they appear in diverse settings, their vectors are pulled in many directions during training and end up shorter. With max pooling, stop words rarely produce strong activations in any latent dimension, so their impact is small. They still add computational cost, so it is generally more efficient to ignore stop words when using max pooling, particularly in long documents, without significantly affecting the quality of the resulting embeddings.

Limitations of Statistical Pooling

  - Average and max pooling are simple, effective ways to compress token-level embeddings into a single vector. They are statistical rather than semantic methods. Average pooling treats every token the same, which can wash out distinctions between important and less relevant words. Max pooling preserves dominant features but discards contextual relationships between tokens, for example the link between adjectives and nouns like "red car" versus "white car". Both methods fail to encode word order or syntactic relationships and cannot capture long-range dependencies across sentences.

  - For example, in the sentence "The cat that chased the mouse was black," average pooling treats each token equally and ignores the connection between "cat" and "chased the mouse", while max pooling focuses on a few strong dimensions and can miss important interactions.

These limitations motivated the development of contextual embeddings, where each token’s vector is influenced by surrounding words. This allows embeddings to represent not just the word itself, but its role and meaning in the specific context of the sentence or document.

  - The paper Attention Is All You Need (2017) introduced the Transformer, a new neural network that relies entirely on self-attention instead of recurrence or convolution for sequence modeling. The Transformer uses multi-head attention to capture relationships between words and positional encoding to preserve word order. This design allows parallel processing, which greatly improves training speed and scalability. The model achieved state-of-the-art results in machine translation and became the foundation for many later models, including BERT and GPT, transforming natural language processing.

  - Both GPT and BERT use the Transformer architecture but differ in design and training goals. GPT (Generative Pre-trained Transformer) uses only the decoder and is trained with a unidirectional left-to-right language modeling objective, making it well suited for text generation. BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder and is trained with masked language modeling and next sentence prediction, which lets it capture context from both directions. GPT focuses on natural language generation, while BERT focuses on natural language understanding.

  - The initial approach using BERT and the encoded [CLS] token did not yield the expected results. However, the introduction of SBERT (Sentence-BERT), with its bi-encoder and cross-encoder architectures, marked a significant breakthrough, producing far better results than pooled word embeddings. Bi-encoders generate contextualized embeddings for sequences, enabling effective retrieval of semantically similar passages. Cross-encoders, on the other hand, compare pairs of sequences directly and yield a score, making them ideal for reranking results.

To create more semantically rich embeddings, the first idea was to use BERT to generate contextualized embeddings:

  - BERT uses a WordPiece tokenizer to split words into subword units, for example "playing" becomes "play" + "##ing", so the model can handle rare or unknown words. Each token is then converted into a 768-dimensional embedding for BERT-Base (1024 dimensions for BERT-Large). Positional encoding record each token's position in the sequence to preserve order, while segment encoding mark whether a token belongs to sentence A or sentence B (using [SEP] as a seperator). They are added to the token embeddings to form the input vectors for the encoder.

  - BERT-Base has 12 Transformer encoder blocks with 12 attention heads per block. BERT-Large has 24 blocks with 16 heads per block. Inside each block, bidirectional self-attention lets each token attend to all other tokens at once, so the model captures context from both left and right. The encoder produces a 768-dimensional contextual vector for each token (1024 dimensions for BERT-Large), used for tasks like classification, question answering, and named entity recognition.

  - BERT accepts up to 512 tokens per input, including the special tokens [CLS] (classification) and [SEP] (separator).Truncating, padding and masking enable support for any sequence length. The [CLS] token is always at the start of the sequence. Its final hidden state, the first output vector, is usually used as a contextual embedding for the entire sequence. Alternatively, we can pool over all encoded outputs, for example with average pooling or max pooling.

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

□□□□(masked)

positionalencoding

1

2

3

1

4

2

5

6

7

0*(masked)

tokenID

[CLS]

the

cat

that

chased

the

mouse

was

black

[PAD]*(503 times)

token

Encoder(12 or 24 layers)

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

□□□□(masked)

tokenembeddings

input forencoder

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

□□□□(masked)

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

□□□□(masked)

segmentencoding

encoded

token

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

■■■■

□□□□(masked)

Option 1:

[CLS] encoding as the  contextualized embedding for entire sequence

Option 2:

average or max pooling over all (non-masked) encodings as the  contextualized embedding for entire sequence

Sentence-BERT (SBERT): Siamese Architecture for Effective Sentence Embeddings

  - Sentence-BERT (SBERT) improves the creation of meaningful sentence embeddings. It uses a so-called bi-encoder architecture: two sentences are passed separately through the same BERT-based encoder to produce fixed-size embeddings. The encoder output can either be the [CLS] token encoding or the mean of all token encodings. SBERT is fine-tuned on datasets, where it learns to label sentence pairs as entailing, contradicting, or neutral. Training uses similarity-based losses, typically contrastive, triplet, or multiple negatives ranking loss, with cosine similarity so that semantically similar sentences have closer embeddings and dissimilar ones are farther apart.

  - The advantage of the bi-encoder architecture is that we can preprocess every passage into an embedding vector that captures the deep contextual meaning learned during training. A query is encoded the same way. Sentence Transformers usually output normalized embeddings, so cosine similarity can be replaced by the simpler dot product. This also prevents implementations from choosing a different similarity measure that would conflict with the training setup. In geometric terms, normalization maps all vectors onto a hypersphere so that the angle between vectors becomes the only factor determining their similarity, effectively removing the influence of vector magnitude and ensuring that comparisons are based purely on semantic alignment rather than scale differences.

Qwen Embedding Models: State-of-the-Art Multilingual Text Representation

  - Qwen3 embedding models represents the latest advancement in text embedding technology, specifically designed for comprehensive text embedding and ranking tasks. Built upon the dense foundational models of the Qwen3 series, these models provide various sizes (0.6B, 4B, and 8B parameters) with exceptional multilingual capabilities.​

  - Qwen3 embedding models use a dual encoder design built on the Qwen3 foundation model. The base model is a decoder-based transformer that processes each text segment and produces dense semantic embeddings. These embeddings are taken from the hidden state of the final [EOS] token, which acts as a summary of the input text. This in contrast to the encoder based BERT embedding models that start with the [CLS] token.

  - Qwen3 embeddings use LoRA (Low-Rank Adaptation) to efficiently fine-tune the model for embedding tasks. LoRA adds small, trainable low-rank matrices to the model's weight matrices, letting it adapt to new tasks with very few extra parameters. This preserves the base model's pretrained knowledge while improving its ability to capture fine-grained semantic relationships in text. The dual encoder is trained with similarity-based objectives similar to SEBRT (contrastive loss, cosine similarity).

  - The 8B Qwen3 embedding model achieved No.1 ranking on the MTEB multilingual leaderboard (June 2025). The models support over 100 languages and provide flexible vector definitions across all dimensions. It supports a context window of up to 32k tokens which typically includes instructions for downstream tasks.

A cross-encoder is a neural model used in text retrieval and ranking systems. It scores how relevant a candidate text, such as a passage or document, is to a query. Instead of relying on precomputed representations, a cross-encoder processes the query and the document together in the same transformer network. This lets the model learn and use fine-grained relationships between query and text at the token level, producing very accurate relevance judgments.

  - To see why cross-encoders matter, compare them with the more efficient bi-encoder (or dual-encoder). In a bi-encoder, the query and each document are turned separately into dense vectors. Relevance is then measured with a simple similarity function such as cosine similarity or dot product. Because document embeddings can be precomputed and indexed, bi-encoders are very fast and scalable, making them ideal for searching huge text collections. However, encoding the query and document independently limits the model's ability to capture subtle, context dependent relationships between specific words or phrases.

  - The cross-encoder avoids that limitation by joining the query and document into one input sequence, often separated by a special token like [SEP], and sending the combined text through a transformer model such as BERT. The model attends to all tokens from both the query and the document at once, allowing it to capture precise semantic interactions. The final representation, usually taken from the [CLS] token, goes through a classifier or regression layer to produce a relevance score. Because each query-document pair is processed together, cross-encoders require much more computation, especially when ranking thousands or millions of documents.

  - Therefore, cross-encoders are used for re-ranking rather than direct retrieval. A common workflow is to first use a bi-encoder or BM25 to quickly retrieve a small set of candidate documents, and then apply a cross-encoder to re-evaluate and reorder those candidates with greater accuracy. This hybrid method balances speed and precision.

The Qwen3-Reranker models from Alibaba are cross-encoders built for precise text re-ranking. Built on the Qwen3 foundation model, they score relevance by taking a query and candidate text together. Unlike Qwen3-Embedding models, which use a dual-encoder design for fast large-scale retrieval, the re-rankers output a direct relevance score. They support task-optimized instructions and very long context windows up to 32,000 tokens. Batch inference lets them score multiple documents at once. A typical input sequence looks as follows:

  - [instruction] {query} [candidate document]

They are built on the Qwen3 transformer architecture and support long contexts and more than 100 languages, so they can handle multilingual and long-document ranking. The models are trained on supervised pairwise relevance data and are optimized for precise reranking rather than general embedding.

Summary: Text retrieval and semantic similarity models can use different architectures, each balancing efficiency and accuracy in its own way. The figure at the bottom high-lights each architecture (here with BERT):

  - Pooling of word embeddings represents a sentence by averaging or max-pooling its word embeddings, for example from word2vec or GloVe. This creates a single fixed-length vector that roughly captures the sentence meaning. It is fast and easy to implement, but it discards word order and detailed context, so it performs poorly on complex language tasks.

  - Pooling of [CLS] token from encoder uses the final hidden state of the special [CLS] token with a transformer based encoder such as BERT. Alternatively, we can pool the final hidden states over all tokens. Both approaches capture sentence-level meaning but still encode each text independently, producing contextualized vectors for fast retrieval. Because this architecture has not been fine-tuned for semantic similarity tasks, it is more often used for later classification tasks than for semantic search.

  - A bi-encoder /dual-encoder encodes the query and the document separately and learns from labeled examples how to judge relevance between them. This produces contextualized embeddings that can be used for semantic search with dot-product. Retrieval is fast because document embeddings can be precomputed and stored in a vector index. However, since the query and the document do not attend to each other during encoding, some semantic nuances are lost.

  - Cross-encoder combines the query and document into one input sequence, allowing the model to jointly attend to all tokens. This enables precise, token-level interaction modeling and leads to superior ranking accuracy. The trade-off is efficiency: it must process every query–document pair, making it much slower and typically reserved for reranking a small set of top candidates retrieved by a bi-encoder.

The cat and the dog.

pooling

[CLS]The cat and the dog.

BERT

pooling

[CLS]Sent A

[CLS]Sent B

BERT

BERT

pooling

pooling

softmax

score 0…1

[CLS]Sent A [SEP]Sent B

BERT

score 0…1

classifier

1) Pooling of token      embeddings

2) Pooling of encoder

3) Bi-encoder / dual-encoder

4) Cross-encoder
