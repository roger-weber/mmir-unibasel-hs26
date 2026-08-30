# How Retrieval Systems Find Answers

A retrieval system addresses the following fundamental problem:

First, what do we mean with "relevant for query Q in the context of the query originator"

  - Relevancy is the degree to which retrieved information matches a query and meets the query originator's needs or intent. For example, if someone asks "Where can I get a pizza tonight?", the user's location is crucial to providing a relevant result. A restaurant that makes excellent pizza will not be relevant if it is too far away. Social media often presents information without an explicit query; users still expect content that fits their interests rather than random posts. Relevance is also shaped by the user's objective: when searching for a product, it matters whether the user wants to evaluate the item or find the best place to buy it.

  - Objective relevance means factual correctness. If someone searches for the capital of France, Paris is objectively relevant because it is the capital. Subjective relevance depends on individual preferences. When a user asks for "movies to watch," what counts as relevant will vary with taste and with personal definitions of a good movie.  Search engines use feedback from users to improve relevance. Clicks and longer time spent on a result signal that it was useful, and ranking algorithms learn from those signals to show results that better satisfy users.

  - Query-less search is an approach where users discover content without entering an explicit query. Platforms such as Instagram Reels and TikTok analyze user preferences, behavior, and trends to recommend videos that match inferred interests, enabling discovery without typed or spoken queries. Relevance is defined implicitly by how well the recommended content matches the user's inferred interests and intent, even though the user never specifies a query. Because there is no explicit question to measure against, the system estimates relevance from behavioral signals and contextual data. Content is considered relevant when it aligns with these inferred preferences and keeps the user engaged, for example, when the user consistently watches similar videos to completion or interacts positively with them. In other words, relevance is not judged by textual similarity to a query but by predicted satisfaction and engagement based on the user's past behavior and the patterns the system detects.

Given a set of N documents $D_{0}$ to $D_{N−1}$ and a query Q, find a set of documents $D_{i_{j}}$ with $0\leq j< k$ that are relevant for the query Q in the context of query originator. Rank the documents such that $D_{i_{0}}$ is the most relevant document and $D_{i_{k−1}}$ is the least relevant document for query Q in the context of the query originator.

  - Offline processing: concerned with analyzing documents in advance, extracting features, and organizing these features into indexes that allow for fast retrieval. Some systems also adjust their (objective) relevance ranking during this phase (see IDF or Google's PageRank later in the course)

Because scanning billions of documents at query time is infeasible given the user's expectation for near-instant responses, retrieval is split into offline processing and online query answering:

data sources

document analysis

metadata

feature extraction

metadata

features

ranking & scoring

scores

retrieval index

optimization & scaling

query: best song by "The Beatles"

query parsing & analysis

retrieval & ranking

result presentation

query refinement & iteration

feature (meta-)data

scores & doc ids

meta data & docs




(opening-semantic-gap)=
Searching images, audio, and video is harder than searching text because of the so-called semantic gap. Users typically enter queries as keywords. Text search can match those keywords directly because queries and documents use the same representation. By contrast, images cannot be matched to keywords at the pixel level. For example, there is no clear, fixed relation between the keyword "cat" and the many ways a cat can appear in an image.

When queries and media use different forms, the retrieval system must translate between them. This mismatch is called the semantic gap:

    - The semantic gap refers to the disparity between low-level features extracted from multimedia data and the high-level semantics that humans associate with that data.

10010011001010101010110101101010101010101010101101

picture of a "cat"

machine interpretation

human interpretation
Traditional, mostly hand-crafted pipelines for feature extraction (illustrations on the following pages)

  - Since the start of information retrieval, feature extraction has been a carefully designed, hand-crafted process for describing media content precisely and concisely so data can be retrieved quickly and with high relevance. In this course, we study many of these approaches, which remain useful despite the rise of improved methods based on large language models.

  - For text, common features include term vectors, bag-of-words, and n-grams. Natural language processing steps include stemming, handling synonyms, and resolving homonyms. Systems also extract embeddings and measure term discriminative power with methods such as inverse document frequency. Web and social retrieval add weights from metadata and link analytics like PageRank. Image retrieval once used simple features such as color, texture, and shape and now depends largely on neural network features from classification and representation models. Audio retrieval uses frequency and amplitude domain features, musical features such as pitch and tempo, and speech recognition. Video retrieval adds shot detection and motion analysis.

General-purpose feature extractors via prompting and large language models  (illustrations on the following pages)

  - The rise of generative AI has changed how people create content and how they extract useful information from it. Traditional analysis extracted features specific to each media type. Methods were often one-off and required new engineering for each domain and often for each use case scenario. For example, analyzing medical images required tailored techniques different from those used to detect faces in images.

  - Generative AI and multi-modal transformers shift the emphasis from manual feature engineering to prompt engineering. A single multi-modal model can be prompted with requests such as "Describe the key visual elements in this image", and it will generate a summary of the most important parts of the image. The same approach can be adapted to different tasks and domains by changing prompts, and light fine-tuning methods such as LoRa (Low Rank Adaptation) can quickly adapt the model to domain-specific needs without retraining the entire model.

  - This universal approach has cut the cost of content analysis by about tenfold. Because the same core infrastructure can support many media types, teams can experiment faster: they can start with generic prompts and models to validate a use case and invest in fine-tuning only after the application proves valuable.

  - Prompts become the primary engineering artifact, which introduces new risks. Prompt execution is not strictly deterministic. The internal workings of large language models and multimodal transformers are not transparent, so one cannot fully explain how they generate outputs. Updates to underlying models can change behavior in subtle ways, affecting the applicability and stability of deployed solutions even when prompts do not change.

Traditional, mostly hand-crafted pipelines for feature extraction

keywordextraction

NLP

bag-of-word

[0.1,0.2,0.3,...]

text

hand-crafted

hand-crafted

hand-crafted

[0.1,0.2,0.3,...]

[0.1,0.2,0.3,...]

[0.1,0.2,0.3,...]

low levelfeatures

HOG / SIFT SURF

SVM

car

image

hand-crafted

hand-crafted

supervised

shot detection

optical flow

combine streams

car movingfast

video

supervised

hand-crafted

supervised

spectral features

tempo features

classifier

hip-hop

music

hand-crafted

hand-crafted

supervised

MFCC

Mixture of Gaussians

classifier

/ˈkɑr/

speech

hand-crafted

unsupervised

supervised

General-purpose feature extractors via prompting and large language models

The main subject of this image is a majestic, snow-covered mountain peak, which appears to be the iconic Matterhorn in the Swiss Alps. The mountain dominates the frame, its distinctive pyramid shape rising dramatically against a clear blue sky. The setting is a high-altitude alpine environment, with the peak surrounded by other snow-capped mountains and glaciers visible in the lower portions of the image. The background is primarily composed of a vivid blue sky with a few wispy clouds. The colors in the image are striking, with the brilliant white of the snow contrasting sharply against the deep blue of the sky. The lighting appears to be natural sunlight, creating a play of light and shadow across the mountain's face that accentuates its rugged features and crevices. There are no visible people, animals, or man-made objects in the image. The focus is entirely on the natural grandeur of the mountain. The overall mood of the image is one of awe-inspiring beauty and serene majesty. There's a sense of isolation and pristine wilderness that the mountain embodies. Notable details include the jagged ridgelines of the mountain, the smooth snow fields on its flanks, and the wisps of cloud that cling to its lower slopes, suggesting high winds at the peak. The composition of the image is well- balanced, with the mountain placed slightly off-center, allowing the eye to follow its slopes from base to peak. The surrounding mountains and glaciers ...

traditional feature engineering

universal prompt engineering



Online query answering evolved rapidly in the past years

  - Early systems provided simple retrieval without relevance ranking, which was enough for file searches or basic catalog queries. As users asked for finer control and higher precision, systems added filtering and ranking, trading speed for accuracy. With the web's growth, large-scale retriever-ranker architectures became standard, offering high accuracy and low latency but higher infrastructure costs.

  - Demand for question answering drove the next advance. Systems began combining retrieval with reading or generation to give direct answers instead of lists of documents. These hybrid models improved the user experience but required more computation and raised operating costs. For example, a web search for a factual question, such as "Who won the Formula 1 race this weekend", now returns a direct answer instead of a list of links that the user must click and read. Often, finding the answer on the linked pages took more time than the search itself.

  - The field is now entering a period of rapid change after a long steady phase. Retrieval-augmented generation and agentic extensions combine planning, tool use, and iterative reasoning, improving accuracy and flexibility while challenging traditional cost and latency trade-offs. This faster research cycle reflects a wider range of use cases, from everyday search to complex multi-step problem solving, and points to rapid innovation in retrieval systems.

  - Throughout this course, we will study models ranging from classical text retrieval approaches to modern methods that use agentic AI. In what follows, we describe the different retrieval types and explain what distinguishes them.

Retriever-only systems use a retriever component to identify documents that match a query and present them to the user without an explicit relevance ranking. This basic search functionality is widely available in file search tools and simple web applications. Without ranking, filtering, and sorting, this approach works only for small data sets where queries usually narrow results to a few items which users can quickly assess for relevance.

  -  Example: https://www.goodreads.com/search?q=agatha+christie&search_type=books

Retriever

query

doc 1

doc 2

doc 3

...

index

Retriever-Filter systems are similar to retriever-only approaches but add a filtering and sorting stage to refine results before presentation. Filters let users narrow results by parameters such as year or rating, and sorting can be driven by attributes like popularity or price. Relevance may affect ordering, but other criteria often dominate. This search feature is common in e-commerce applications and is often enhanced with faceted search.

  -  Example: https://www.galaxus.ch/en/search?q=clothes+iron

Retriever-Ranker systems first select a pool of candidate documents using the retriever and then apply a ranker to assign a relevance score to each candidate, returning documents by score. This is a common architecture in both classical and modern retrieval systems and is frequently enhanced with semantic search and context-sensitive ranking such as user location, objective importance, and subjective importance. Web search engines typically use this model, combining text retrieval with web-specific ranking signals.

  -  Example: https://www.google.com/search?q=multimedia+retrieval+lecture       (change your location with a VPN client and submit again)

Retriever

query

doc 1

doc 2

doc 3

...

index

Filter & Sort

meta-data

criteria

Retriever

query

doc 1

doc 2

doc 3

...

index

(Filter &) Ranker

rank model

Retriever-Reader systems are designed for question-style queries that ask for a specific answer. The retriever fetches relevant documents and the reader identifies one or more passages within those documents that answer the question, returning the passages rather than a list of documents. Readers often rely on language models to locate concise answers in the result documents.

  -  Example: Google's 'featured snippet from the web' (this feature is now often replaced with a retriever-generator answer)		https://www.google.com/search?q=what+is+the+main+ingredient+in+tylenol		https://www.google.com/search?q=What+did+Albert+Einstein+win+the+Nobel+Prize+for%3F		(this later query may provide an answer from the knowledge database; try "People also ask" for retriever-reader answers)

Retriever-Generator systems, also known as Retrieval-Augmented Generation or RAG, combine the retriever with a generative language model. The retriever selects relevant documents or passages and those snippets are combined with the user query into a prompt template for a large language model. The model then generates a comprehensive answer rather than extracting a single passage.

  -  Example: Bing copilot search ("What did Albert Einstein win the Nobel Prize for?")		https://www.bing.com/copilotsearch?q=What+did+Albert+Einstein+win+the+Nobel+Prize+for%3f

Retriever

query

doc 1

doc 2

doc 3

...

index

(Filter &) Ranker

rank model

Retriever

query

index

Generator

LLM

documents
Retriever-Synthesizer systems fetch a set of relevant documents and then instruct a language model to synthesize a condensed summary rather than extracting a direct answer. This is especially useful for exploratory or conceptual queries, where users benefit from an overview of multiple sources rather than a single factoid. Unlike Retriever-Reader systems that target pinpoint answers, the summarizer model must integrate information, reconcile conflicting statements, and produce a coherent narrative.

  -  Example: perplexity.ai with "How does vector space retrieval work?" (normal search)		https://www.perplexity.ai/search/how-does-vector-space-retrieva-DuutsrURRgeeYmrJw_0bZg

Agentic RAG systems extend the concept of Retriever-Generator and Retriever-Synthesizer with agentic capabilities. In this setup, the agent receives the query and actively decides how to fetch information, which sources to query, and whether to iterate or reformulate queries. It can plan a sequence of retrieval actions based on intermediate results, not just a single retrieval pass. Once the agent has gathered and processed the necessary information, a generative language model synthesizes a comprehensive final answer.

 Example: perplexity.ai with "How does vector space retrieval work?" (pro version with multi-step reasoning)		https://www.perplexity.ai/search/how-does-vector-space-retrieva-01jcYjFZQWmwzpaCVGdfRQ

Retriever

query

index

Synthesizer

languagemodel

documents

Agent

query

Synthesizer

languagemodel

scratchpad

scratch  pad

Tools

knowledge base

web search

tool / API

user context

code execution

reasoning

retriever tools

  - Generator-only systems use only generative models and have no explicit retriever. They produce answers from knowledge stored in their training data. General-purpose models can handle many tasks with careful prompting, but fine-tuned models are needed for specialized or business queries. A major risk is hallucination, when the model gives plausible but incorrect information. This occurs because the model is trained to always answer and to satisfy the user, even when it lacks facts. Ambiguous questions, gaps in the training data, and the lack of retrieval grounding increase this risk. In short, generator-only systems are flexible but trade reliability for broad usefulness, making them vulnerable to mistakes in specialized or high-stakes situations.

  -  Example: ChatGPT with "What did Albert Einstein win the Nobel Prize for?"		https://chatgpt.com/share/68c57b5f-d174-8011-a6c1-14a0eb5078fa

query

Generator

LLM

domainknowledge

fine-tuning

prompt template
