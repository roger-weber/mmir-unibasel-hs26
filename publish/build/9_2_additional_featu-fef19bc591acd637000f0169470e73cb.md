# Additional Features

In this section, we outline a few essential text related features for web retrieval. The following sections will delve into link information, expanding beyond web search to encompass a broader array of social media-related searches.

Today, web search engines must handle over 40 billion pages, 60+ trillion unique URIs, and an index size exceeding 100 PB. A standard query yields millions, or even billions, of results, yet users expect the top page or first link to be the most relevant one for their search context. However, for single or double-term queries and millions of candidate pages containing the terms, finding the most relevant ones based solely on term occurrences is not possible.

  - Example query=“ford”: does the user mean the car, the president, or a ford to cross the river?

  - Example query =“uni basel”: which department or faculty is the user looking for? are they looking for courses, research, or maybe for a job?

  - Example query=“best pizza”: is the user asking for the best pizza type (funghi)? a pizza delivery service of that name? or a nearby restaurant with good pizzas?

  - Example query=“roger federer”: is the user looking for documents that contain “roger” and “federer”?

A drawback of classical retrieval is its emphasis on term occurrences. Consider the query "roger federer“ which many retrieval methods interpret as a search for documents containing the terms "roger" and "federer“. While this approach may be suitable for classical retrieval, it falls short on the web, where pages may include these terms but not in close proximity (e.g., a marathon results page with numerous names).

As an extension of the classical document descriptors, web retrieval considers:

  - proximity of query terms in documents (the closer the better)

  - term occurrence weighting based on tag

  - term boosting if it occurs in an anchor text of a link to the page

  - language and location awareness

  - penalties for low quality pages or block for entire sites (click baits, malicious content, link farms)

Proximity of query terms in documents: In classical retrieval, we employed n-grams to expand our vocabulary with common term sequences aiding in expressing the closeness of query terms. However, n-grams are limited to a small set of sequences and can't adapt swiftly to new sequences arising from news or events. Given the importance of name and celebrity searches in web search, we require a method to factor in proximity when scoring at query time.

  - We enhance the postings in the index with position information, including the term’s original position in the document's sequence. The posting then becomes a hit-list, as illustrated in this example:

    - The hit-list replaces term frequency and lists all positions where the term appeared.

  - Let's consider a query with "white house". At query time, we create pairs of hits. Theoretically, we could generate the cross-product of all possible pairs, but it is adequate to only consider close matches (which may involve the same term in different pairs). Additionally, we aim to preserve the order of terms in the query. For the example mentioned, we might choose the following pairs:

  - Next, we calculate the distance for each pair and create a histogram with n bins. Each bin corresponds to a given distance range. For example, the first bin includes pairs with a distance of 1 (adjacent). The second bin includes distances of 2 and 3, and so forth. We then count how often pair distances fall within each bin's range:

  - Finally, we apply a scoring function to the bin distribution to enhance the document's ranking. A straightforward function involves a weighted sum over the bins, with higher weights assigned to closer proximity:

  - We can assume that more sophisticated functions are used to measure proximity, but the concept is the same.

hitlist['white'] = [1, 13, 81, 109, 156, 195]

hitlist['house'] = [2, 82, 112, 157, 189, 226]

pairs = [(1,2), (81,82), (109, 112), (156, 157), (189,195)]

pbins = [3,0,1,0,0,1,0,0,0,0]

weights = [89,55,34,21,13,8,5,3,2,1]

score_proximity = $\sum_{i}^{} $pbins[i] * weights[i]

Term occurrence weighting based on tag: In classical retrieval, all term occurrences were considered equal. However, with HTML, we can distinguish the significance of an occurrence by its position in the document and the surrounding tags. For instance, a term in the title should carry more weight than one in a paragraph, as the author chose it to describe the page's content. Similarly, encounters of terms in <h1> or <h2> tags indicate the term's high importance.

  - Let’s consider a page from the university of Basel for the term “university”:

    - The list on the right displays all instances of "university" along with the surrounding tags. In classical retrieval, we count 9 occurrences of the term, and with BM25, we utilize a saturation function on term frequencies to avoid excessive emphasis on frequent occurrences. In the web context, this saturation function is even more critical to safeguard against spammers attempting to manipulate scoring and rankings for specific target keywords.

  - In addition, we count each occurrence in tags separately. For the example above, we obtain:

  - The final term frequency is a weighted sum of each element in this list, employing a saturation function sat_func(n) on term occurrences to mitigate the impact of keyword spamming:

…<title> … university …</title>

…<h1> … university …</h1>

…<b> … university …</b>

…<p> … university …</p>

…<p> … university …</p>

…<p> … university …</p>

…<a> … university …</a>

…<b> … university …</b>

…<h1> … university …</h1>

terms = [(university,<title>,1),	 (university,<h1>,2),   (university,<b>,2),          (university,<p>,3),	 (university,<a>, 1)]

weights[tag  weight] = [<title>  13, <h1>  5, <p>  1, <a>  0.5]

tf[university] = $\sum_{terms\left[i,1\right]=university}^{} $sat_func(terms[i,3]) * weights[terms[i,2]]
Term boosting if it occurs in an anchor text of a link to the page: Anchor texts in a document typically provide precise descriptions of the linked page, although occasional generic terms like "click here" or "can be found here" may appear. With many thousands of links to a page, we will encounter numerous valuable keywords. While optimizing search results with term occurrences within a document is relatively straightforward, creating thousands or even millions of links (from various domains) to boost a term for a target page is much more challenging. At this stage, we are focusing on the terms used to describe the remote page, without considering the link as a relationship.

  - Let’s consider a page from the university of Basel including two pages that refer to that page:

    - First, how do we locate all links to a page for describing it with anchor keywords? During the crawling process, we extract link and anchor text, storing them in a dedicated database. Upon completion of the crawl process, we establish a separate index for all pages based on the anchor texts. We apply a saturation function to term occurrences, but one that accommodates larger numbers and may even grow indefinitely, such as a logarithmic function. This allows large numbers of pages to "vote" for a page and a keyword.

  - In the scoring process, we take this secondary index into account and derive a boost value for the document. As these occurrences are more challenging to manipulate than the terms within a document, we assign a greater weight to them. This does not imply blind trust in all anchor texts being accurate, but rather, we assume that there are sufficient reliable descriptions, often necessitating thousands or millions of anchor occurrences before a page receives a substantial boost.

  - A "Google bomb" is an attempt to manipulate this secondary score by orchestrating the inclusion of the same keyword in millions of links. In response, most search engines impose restrictions on the number of anchors per domain, with each domain having a maximum vote on how well a term aligns with a target page.


Language and location awareness: In 2014, Google launched the Pigeon update, which modified its scoring function to prioritize local search results. It considered the user's location and the distance to businesses. Initially, the goal was to enhance results for queries like "where to eat pizzas?" as users sought nearby restaurants based on their current location, especially on mobile devices. Over time, this concept expanded to include products, places, and general searches, as long as the query didn't specify a particular location.

  - First, the search engine needs to collect (estimated) geolocation data for each page or website. For businesses like restaurants, owners often manage this information themselves to ensure accuracy on platforms like Google Maps. Alternatively, the crawler can assign an approximate location using the IP address and a managed database.

  - Second, the search engine considers either the exact geolocation from the user's device or an estimated location based on their IP address. However, this relies on users willingly sharing their current location and not using third-party tools to alter it, like VPNs. If the query explicitly mentions a location, such as the name of a city or country, that information takes precedence. For instance, a query like "best pizza in Paris?" produces different results than "best pizza?". The query analyzer identifies geolocation details through named entities and maps each entity to an accurate location. This applies not only to city names but also to landmarks, as seen in queries like "best pizza Niagara Falls?" or "best pizza Golden Gate Bridge?"

  - Finally, the search engine must determine the relevance of geolocation data for each query, considering the query's context and type. For instance, queries like "best pizza" receive a higher emphasis on location information, while queries for "Albert Einstein" prioritize global results but may still include local sites that host exhibitions related to Albert Einstein's work with a boosted local ranking.

Penalties for low quality pages or block for entire sites: Search engine providers also monitor the internet to identify potentially harmful, illegal, or malicious content. They employ various techniques to detect such sites and impose penalties (e.g., for thin content with excessive ads or keyword spamming) or may even block sites entirely from search results (e.g., link farms, doorway pages, and illegal content). More recently, this includes maintaining global information about site trustworthiness. Trustworthiness can be determined from factors like grammatical accuracy, keyword usage, and known ownership of the page. Registered corporations, for example, receive higher trust scores compared to anonymous blogs. This practice has impacted nearly 12% of Google queries. However, it can make it challenging for newer websites to compete with established ones for top rankings.
