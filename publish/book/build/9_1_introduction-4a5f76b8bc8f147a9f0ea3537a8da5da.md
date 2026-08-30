# Introduction

The initial web search engines relied on traditional retrieval techniques, which were suitable for the early web stages. As web interest grew, it became clear that these methods were inadequate. Google was the first search engine to recognize web search as a distinct challenge and added improved ways to rank pages. It used a retriever-ranker structure with a basic Boolean search for finding web pages, similar to other engines. However, it ranked pages using newly developed features tailored to web users' unique interests and the rapidly evolving web structure.

The web search experience has evolved over the years. Initially, users received a list of pages with ads in between. They had to refine queries and find answers within the documents. For example, a query like "Who is the US president?" would lead to the official US government site, but users had to extract the information themselves. This led them away from the search portal, which couldn't generate revenue through ads. To keep users on the site longer, search providers have enhanced the query experience in the following ways:

  - The initial enhancement introduced factual knowledge cards alongside or above search results. For fact-based queries, users received curated answers. The previous query would display a data card with the current US president, providing a convenient response for users. Likewise, product and brand queries showed ads and price summaries for common products. However, this was only possible for facts and products stored in the search provider's knowledge base.

  - The second improvement began with the introduction of neural network-based reader models as discussed in the previous chapter. A reader is a machine comprehension model that can pinpoint the most relevant passage on a page when given a natural language query. This enabled search engines to offer answers to nearly any natural language question, provided a search result page contained the relevant information. For example, when querying the US president, the search results now include a highlighted passage from one of them that contains the answer.

  - The third advancement came with the rise of transformer models and the power of large language models. As of the time of writing this, search engine providers are exploring retrieval augmented generation (RAG) along with interactive chat elements. As discussed in the previous chapter, RAG involves taking a few passages from search results, creating a prompt with context and the user query, and using a language model to generate a direct answer to the question. Users can also engage in chat to get more information and receive links to confirm the accuracy of the generated responses.

An overview of the different stages of web retrieval over the course of time:

  - In the 1990s, early search engines like AltaVista and Yahoo operated with smaller data sets compared to today's vast web. They primarily focused on building catalogs (Yahoo) and basic keyword-based searching (AltaVista). Due to the limited number of search results in that era, classical ranking functions were prevalent, with some tweaks, such as domain boosts, to score documents.

  - In 1998, Google originated from a Stanford research project and rapidly gained widespread usage with its innovative approach to web search. Being among the pioneers, Google understood the specific requirements of web users and optimized its ranking function for optimal results in common query types. It also introduced a distributed architecture, enabling more extensive crawling and simultaneous query handling compared to other search engines of the time. Most notably, it introduced PageRank a novel approach to re-rank results.

Retriever

query

index

Ranker

rank model

Retriever

query

index

  - In 2012, Google launched the Knowledge Graph, a curated knowledge database containing facts from reliable sources. At its debut, it held 18 billion facts about 800 million entities. By 2023, it has expanded to include 800 billion facts on 8 billion entities. When a query aligns with an entity, a knowledge card appears alongside search results, providing users with verified information about that entity. This greatly benefited users who would otherwise need to sift through search results and linked documents to find their desired answers.


  - Google has employed snippets since its beginning to provide contextual information from web pages in search results. Snippet generation is automated, but web authors can influence it using meta-information. In 2016, Rich Snippets were introduced initially for movies and recipes, later expanding to a wider array of queries. In 2020, Passage Ranking was introduced and went live in 2021. Unlike snippets, it uses a language model (BERT) to identify the most relevant passage in the top results for fact-based queries. This retriever-reader model aids users in quickly finding relevant content in lengthy documents. To enhance the web experience, most browsers now support the "scroll to text" feature, allowing users to jump to specific passages on a page from search results.

  - In 2023, Microsoft, in a deepened partnership with OpenAI, introduced the next generation of web search on Bing. This innovation combines retrieval augmented generation (RAG) and interactive chat features. RAG utilizes a large language model to provide direct answers by generating responses from top search results and the user's query. The chat assistant offers source links for verification and allows users to ask follow-up questions in the same context. Google also launched Bard in 2023, a conversational AI chatbot based on Google's language model PaLM. Both providers face the challenge of the high cost of generating answers, and balancing the need for users to obtain good search results.

Retriever

query

index

Reader

languagemodel

documents

Retriever:BM25 or embeddings

query

Generator: Prompt template + query + context
Before we dive into the content, it is worth to review the differences between text retrieval and web search:

  - Collection: In classical retrieval settings, content is controlled and quality-assured. For example, online shop product catalogs undergo rigorous checks to maintain high quality and a positive user experience, ensuring data accuracy. Conversely, the web lacks such quality controls. While we may trust some sources more than others, the sheer scale of the web makes implementing quality controls or assessments impractical. Search engines must contend with a wide spectrum of quality, including poor grammar, (intentionally) incorrect information, as well as aggressive spamming and search result manipulations.

  - Documents: Classical retrieval, as seen in systems like Lucene, relies on a shared document structure for easy information extraction and structured or faceted search to refine results. The web lacks such a uniform structure. While many documents use common formats like PDF and HTML, extracting structural details beyond title, URI, and content proves challenging. Modern user experience methods, such as single-page applications, render large portions of the web inaccessible to search engines (referred to as the deep web), despite the information being publicly available. Some content can be accessed through deep links, but often requires the web crawler to locate a link to that page from another source.

  - Queries: Classical retrieval relies on specific query context. In a music database, queries typically relate to distinct object types like artists, producers, songwriters, albums, songs, or lyrics. In such narrow contexts, classical methods excel in providing relevant information. In contrast, a web search engine handles a wide array of query types, some comprising just 2-3 terms without any further context. Without structured data, search providers must not only locate relevant pages but also infer the implicit context of the query.

  - Results: Classical retrieval functions well on smaller scales but can also handle millions or billions of entries. It is easier to control result size, provide users with filters, facets, and sorting options for query refinement due to structured data and contextual boundaries. In contrast, a typical web search with 2-3 terms can yield billions of candidate documents, all containing the query terms, making it challenging to rank pages (as they all appear equally similar) or to provided faceted search. Thus, additional features are necessary to improve page ranking.

  - Relationships: The web lacks structure within documents but is a highly interconnected network with pages as nodes and links as edges. These links convey relationships between pages and identify influential sites ("go-to" destinations). This enables a web search engine to rank pages not just based on text but also using network topology. In contrast, classical retrieval contexts may have relationships among data items, but their structure is usually curated and provides minimal additional information. For example, relationship information (artist/song links for instance) in a music database may not reveal the popularity of songs.

What are the most common web query types? We can classify search queries into three categories:

  - Navigational search queries: users aim to reach a specific website or its particular section. Examples: "youtube," "amazon prime," "wikipedia," or "facebook login“. While bookmarks were designed to address this specific need, many users rather employ a search engine to find the entry point for their favoured web application.

  - Transactional search queries: Users search for brands, products, hotels, restaurants, etc., with the intent to make a purchase or order goods and services. This covers a broad range of queries, including "new coffee machine" or "trip to Hawaii" where the transactional intent may not be immediately clear.

  - Informational search queries: Users seek news, events, sports, celebrities, or general information. Some queries are posed as direct questions: "who is the US president?“, "how to make mac & cheese?“, "where to find a lake for swimming“, or "what is the weather tomorrow?“.

The following statistics, sourced from backlinko.com, ahrefs.com, and statista.com, have been published since 2020. While the numbers may have evolved, the fundamental distribution remains largely consistent. The initial statistics relate to query length, encompassing the number of terms and the character count (backlinko.com):

  - These are average values across different queries. Yet, as we will explore on the next page, queries are often repeated multiple times, and most frequently used queries in terms of search volume tend to be shorter in length.

ahrefs.com examined query repetition frequency. As expected, the distribution adheres to Zipf's law, as displayed on the right: a small set of queries, mainly navigational ones, form the "Fat Head“, recurring millions of times. In the “Chunky Middle”, you find common queries appearing in thousands of searches, while the “Long Tail" includes all other queries that appear in fewer than 100 searches.

According to backlinko.com, 92% of unique query strings fall within the long tail. However, these long-tail queries account for a relatively small portion of the total search volume (3.3%). The majority of search demand is concentrated in a small percentage of high-volume terms. To illustrate, the top 500 most popular search terms constitute 8% of all search volume, and the top 2000 keywords are responsible for 12% of searches on both Google and Bing. The median search volume for a query is only 10 searches per month, meaning that 50% of queries are nearly unique for users, with minimal repetition over a month. These unique queries significantly influence users' choice of a search engine.

Google handles trillions of searches annually, yet 15% of these queries (not search volume) are entirely new to Google. In other words, a significant portion of queries arises from news events, new products, or general interest in sports and people.

According to backlinko.com, 14% of queries are in question form. The most common question word is "how“, followed by "what“, "where“, and others (as shown on the right). However, this statistic does not encompass the intent of queries lacking a question word. For example, a name search could represent either a "who" (for people) or "what" (for products and services) question.

