# Inverted Files

In all traditional retrieval models discussed so far, we have seen that their scoring functions depend only on the query terms. This choice cannot capture deeper semantic relationships between terms, but it is a practical trade off because it allows much faster query processing. Let us explore why this is the case:

  - Assume a collection of $N$ documents and a vocabulary of $M$ terms. On average a document has $K$ distinct terms (much smaller than $M$), and a typical query contains about $L=5$ terms (much smaller than $K)$. Each document is represented as a sparse $M$-dimensional vector, using either the bag-of-words model or the set-of-words model.

  - A simple storage scheme reserves an entry for every term in every document, $N∙M$ entries in total. In the set-of-words model, each entry is a single bit indicating whether the term appears. The bag-of-words model needs more space, storing term frequencies using 4, 8, 16, or 32 bits per entry. For example, models such as BM25 use term saturation allowing for term frequency compression to 4 bits without noticeably hurting retrieval accuracy.

  - Scanning all $N∙M$ entries for each query is inefficient. Most entries are zero because each document contains only a small fraction, $K/M$, of all terms. A simple improvement is to store documents in a sparse format, keeping only the $K$ non-zero entries per document. Each entry is either a term ID (for a set-of-words) or a term ID plus its frequency or weight (for a bag-of-words). This reduces storage from $N∙M$ to $N∙K$ entries, offset by slightly larger entries.

  - Although this sparse representation is more compact, we still must scan all documents at query time. Only a small fraction of that data, specifically the entries for the query terms, affects the final relevance score. Most of what we read never contributes to the result.

To address this inefficiency, we flip the storage layout. Instead of saving for each document the list of terms it contains, we save for each term the list of documents that contain it. This is the main idea of the inverted index, also called the inverted file. In other words, switch from terms-per-document storage to documents-per-term storage.

  - This change does not alter the total number of stored entries. There are still $N$ times $K$ term-document pairs, but query processing improves dramatically. When a query arrives, we do not need to read all data. We only need to look up the rows, or posting lists, for the query terms.

  - If each term appears, on average, in $N∙K/M$ documents, and the query contains $L$ terms, then we only need to read about $N∙K∙L/M$ entries. Compared to scanning all $N∙K$ entries, this yields a reduction by a factor of$ L/M$. For example, if a query has $L = 5$ terms and the vocabulary size $M=1,000,000$, we read only about five-millionths of the data, a huge efficiency gain.

An inverted index organizes data by terms rather than by documents. This lets retrieval systems examine only the small part of the data relevant to each query, providing compact storage and fast searches.

  - Keeping this fundamental concept in mind, let's start with the Boolean retrieval model. The inverted index consists of the vocabulary ($M$ terms), and for each term, a list of postings contains all documents that include the term. For the set-of-words model, term frequencies are not necessary, and the Boolean model does not require document frequencies or $idf$-values. The inverted index further contains a document table with additional metadata:

  - As we add new documents to the table, we continue including the document ID in the postings of terms found in the document. If documents are added sequentially, the postings are arranged based on the order of document insertion, which, in our simple example, corresponds to increasing document IDs. For certain implementations, preserving this order is crucial for faster retrieval.

The basic implementation stores postings as sets of document IDs within a vocabulary using terms as keys. For instance, index['cat'] contains the set of IDs of documents that contain the term ‘cat’ at least once.

  - For query evaluation, we adhere to three rules:

    - expr1 AND expr2: translates to an intersection of the sets from sub-expressions expr1 and expr2

    - expr1 OR expr2: translates to a union of the sets from sub-expressions expr1 and expr2

    - expr1 AND NOT(expr2): translates to a sub-traction of the set of expr2 from the set of expr1

    - Generalization to AND/OR over multiple sub-expressions are straightforward

  - However, we cannot evaluate OR-queries when one sub-expression is of the form NOT(expr). While it's technically possible to construct NOT(expr) by using all documents except those returned by expr, this approach becomes inefficient for large collections

  - In AND-queries, NOT(expr)-parts need to be re-ordered to the end to apply set subtraction. Additionally, at least one element of the AND-query must not be in the form NOT(expr)

  - Indeed, while these limitations may be viewed as constraints in our implementation, they have minimal impact on practical scenarios. Queries like "cat OR NOT(dog)" do not align with typical search intentions as they essentially select all documents except those with dog but not cat, i.e., it can be rephrased as "NOT(dog AND NOT cat)".

The set-based evaluation from before does not scale well with the number of documents. In cases with millions to billions of postings for a term, we want to fetch data from an external storage device (which is also a good idea for persistence). But instead of reading all postings into main memory, we read them as streams sorted by the document IDs. Take the postings of cat and dog as an example:

  - To evaluate a query like "cat AND dog", retrieve the first entry for each term: document 1 for cat and document 3 for dog. If they match, that document satisfies the query. If not, read the next entry for the term with the smaller document ID. In this example, advance the cat entry to 4. It does not match, so advance the dog entry, which now has the smaller ID. The next dog entry is 4, which matches the cat entry. The first matching document is 4, so we return it as a result for our query.

    - For the next result, we fetch the next postings for both terms and repeat the process. Eventually we identify 10 as the second answer. Then we fetch the following posting for both terms. Because the cat's postings are exhausted, we stop the evaluation and end the iteration. Even though dog still has postings, the lack of cat postings means no remaining document can match. The table below shows this approach step-by-step:

  - The OR-operator is implemented similarly; however, the iteration returns each time the smallest entry of sub-expressions. In the provided example, the OR-operator would start by returning 1, then advance cat and return 3, progress dog and return 4, move both cat and dog and return 8, advance cat and return 10, move again both cat and dog, and finally return 12. The evaluation concludes once all postings are consumed.

  - The evaluation of "cat AND NOT(dog)" evaluation follows the same pattern as the AND flow, but the outcomes differ (matching occurs when cat posting is not equal to dog posting):

  - Generalizing to multiple operands is simple. However, the same limitations as in set-based implementations apply, and here it becomes clearer why supporting queries like "cat OR NOT(dog)" is not ideal. In our implementation, for the NOT(dog) operand, we would need to list all documents except those in dog's postings. Since document frequencies of terms can be low, enumerating NOT(dog) could involve millions or billions of document IDs, substantially slowing retrieval. On the other side, queries like "cat OR NOT(dog)" are not intuitive.

  - We can use the same method for any mix of AND and OR operators nested within one another, as each evaluation method mentioned above produces sorted document IDs. Similar to single term searches, we can handle NOT operators when they are within an AND expression that contains at least one sub-expression without a NOT at the highest level (a nested NOT further down in the sub-expression is not an issue).

We omit here a detailed discussion for the Extended Boolean Retrieval model. The approach is similar with the models to follow, that is, we first fetch all candidate documents (union of postings over all query terms) and then evaluate for each document the overall score using one of the score combining functions.


## 5.2.1 Inverted Files for the BIR model


The Binary Independence Retrieval (BIR) model, Vector Space retrieval, Extended Boolean retrieval, and BM25 models exhibit several similarities when evaluated using inverted indexes. Conceptually, they adopt a retriever-ranker approach as previously explained:

  - By utilizing inverted files, the retriever component retrieves the union of postings for the query terms. This yields a candidate list for the filter & ranker, which then employs the model's designated scoring function for each candidate document to generate the ranked list.

Implementations frequently combine retriever/filter/ranker components for enhanced performance. We initially study the fundamental versions: document-at-a-time and term-at-a-time using the BIR model, owing to its uncomplicated scoring function (sum of $c_{i}$). Subsequently, we expand this to the vector space and BM25 models. The Extended Boolean model is omitted due to its diminished relevance in today’s search contexts.

  - The document-at-a-time method retrieves documents consecutively through streaming like for the Boolean OR-operand approach. At each step, we obtain the document with the smallest doc ID from the sorted postings of each query term, and pass it along with its query terms to the scoring function. The ranker maintains a list of the best k documents encountered and maintains this list upon processing all candidates. The "top-k" mechanism minimizes storage needs, but still enables users to browse through several pages if k is chosen sufficiently large (e.g., k=1000)

  - The term-at-a-time approach processes query terms one by one. For each term it updates the candidate document list and adjusts document scores using the scoring function based on that term's presence. After all terms are processed, documents are sorted by their final scores to produce the ranked list. Unlike document-at-a-time, this method cannot maintain a top-k list during processing to reduce memory use. It can also produce very long candidate lists when the query includes common terms with long posting lists. One optimization is to skip frequent terms that are unlikely to change the ranking significantly.

Retriever

query

doc 1

doc 2

doc 3

…

index

(Filter &) Ranker

rank model

The Python code on the right shows a simplified version for the document-at-a-time retrieval technique for the BIR model. The search_DAAT function takes a query string, a desired number of results (k), and feedback data collected on documents.

  - We start by turning the query string into a set of words using a provided analyzer

  - Using feedback, we compute $c_{j}$-weights and trim terms. For instance, we might keep only the top-n weights from a larger set of query terms

  - The primary loop resembles the Or-implementation of the Boolean model. We sort the postings of each query term by document IDs. We iterate through the postings (index[term]) in a stream based manner (iters), selecting the smallest ID across the next elements (nexts) in the stream as a new candidate document id

  - If we have user feedback, we can skip 'non-relevant' documents. Otherwise, if the document is relevant or there's no feedback, we calculate the score by summing $c_{j}$-values (term_weights[j][1]), pairing it with the document's smallest ID, and adding it to the topk object. This object uses a heap to maintain (doc_id, score) tuples, ordered by score for efficient access to top-k results (no need for repeated sorting after each iteration)

  - In the main loop's final step, we fetch the subsequent postings for each term where the smallest ID was at the stream's front (nexts)

Now, let's explore the term-at-a-time approach for the BIR model on the right side. The search_TAAT function takes a query string, a desired number of results (k), and feedback data collected on documents.

  - We start by turning the query string into a set of words using a provided analyzer.

  - Using feedback, we compute $c_{j}$-weights and trim terms. For instance, we might keep only the top-n weights from a larger set of query terms

  - The main loop runs through each query term (sorted by their weights in query_weights) and all postings (index[term]). It keeps track of a score for each seen document (dictionary scores)

  - If we have user feedback, we can skip 'non-relevant' documents. Otherwise, if the document is relevant or there's no feedback, we add the $c_{j}$-value of the current term (weight) to the scores dictionary. The update line also establishes new entries for previously unseen documents

  - Once the main loop concludes, the scores dictionary contains a value for each document that has at least one query term. Instead of directly sorting scores, we take a similar approach as with DAAT. We utilize the TopKList and include all document IDs and their corresponding scores to reduce computational complexity for sorted access of the result lit.

Discussion: DAAT vs. TAAT

  - Both methods have similar complexity in terms of the number of read postings. They both focus on documents that have at least one query term and a non-zero score

  - Both approaches can efficiently filter out previously marked non-relevant documents to prevent their reappearance in future results

  - The TAAT implementation is shorter and more concise but has a drawback with the scores dictionary. If query term postings are lengthy, this dictionary can become sizable

  - On the other hand, the DAAT approach computes scores in a single step for documents and adds them to a heap within the TopKList object. This heap not only provides efficient access in sorted order but can also be pruned occasionally if it becomes too large (by excluding candidates that are provably not in top-k).

Including Predicates in Evaluation: We can expand both methods to search for documents with predicates like "star wars" and "year < 2000". The assessment of these queries depends on how we can evaluate the condition:

  - A priori filtering: Store the document attributes (metadata) used in predicates in an external index with an efficient evaluation plan, such as a database. For example, the predicate "year < 2000" can be resolved by an index lookup that returns the document IDs that satisfy it. The index can be a B tree or an inverted list.

  - The best approach for combining text retrieval and predicate filtering is to first retrieve all document IDs that meet the predicate, then pass that set as an inverted list into the search function.

  - Inside the search function, filter out any candidates not in that predicate set. In code, this corresponds to the place where non relevant documents are removed during feedback.

  - Other than evaluating the predicate, the search algorithm does not become more complex.

  - A posteriori filtering: If there is no index for the condition, or if evaluation requires scanning all document data, we check the predicate for each candidate when we return results (in Python yield) from the TopKList object.

  - TopKList's heap produces a stream sorted by decreasing score. Before handing an object to the caller, we evaluate the document's predicate by accessing its metadata at random. If the predicate fails, we skip that document and take the next one from the heap.

  - In the best case, with a less selective predicate, we only evaluate the predicate for the documents we return and for a few that are skipped. In the worst case, with a highly selective predicate, we must evaluate the predicate for a large portion of the documents in the heap. This is still better than evaluating the predicate across all documents.

  - Inline filtering: store metadata in a separate file aligned with the posting lists and ordered by document ID. Add this metadata file to the stream list in the code and advance it in the same way as the other streams

      - In the code, extend the if clause for relevance feedback checks to include a check for the predicate.

      - Continue to use the smallest document ID from the postings for the next iteration, and advance the metadata stream to that same ID.

      - Reading all metadata for every query with predicates can be costly. One optimization is to maintain a stream per metadata attribute and evaluate predicates to produce a stream of only the documents that satisfy the predicate, which is similar to a priori filtering.


## 5.2.2 Inverted Files for the Vector Space model (and BM25)


In terms of the algorithms, both the BIR model and the Vector Space model are conceptually the same. The DAAT and TAAT implementations work similarly with these modifications:

  - Postings now comprise tuples with document IDs and term frequencies, sorted by document ID

  - Queries change into a bag-of-words model, including terms and their frequencies for the query

  - We need access to a vocabulary containing document frequencies. As an optimization, we can save required idf-weights alongside postings in the inverted files (to avoid random vocabulary accesses)

  - A similarity function that calculates scores based on the query vector and a document vector subset including query terms and their frequencies.

  - For cosine similarity, we additionally require the document vector's length (=$\left‖𝒅\right‖$)

  - For BM25, we also need the document length (number of term occurrences $|D|$), an average document length ($adl$), and parameters $k$ and $b$ for the calculation

The inner vector product can compute all scores using the data in the inverted files (index in the implementation), but both the cosine measure and the BM25 similarity function need an extra lookup for document-related data (document length, norm of document vector). This can notably raise retrieval costs, demanding extra optimizations for consistent performance. To prevent such lookups, we can normalize document vectors at index build time.

  - If the normalization parameters ($idf$, $k$, $b$, $\left|D\right|$, $adl$) changes then we need to rebuild the index. Setting $q_{j}=idf(t_{j})$ for the BM25, all three measures reduce to a dot-product between normalized document and query vector.Alternatively, we can store the extra document information with each posting. This may increase storage and scanning costs but gives more flexibility to adjust scoring parameters.

$sim_{cosine}\left(Q,D\right)=\sum_{j=1}^{M}\hat{d_{j}} ∙\hat{q_{j}}$              with    $\hat{d_{j}}=\frac{idf\left(t_{j}\right)∙tf\left(D,t_{j}\right)}{\left‖𝒅\right‖}$                    and    $\hat{q}_{j}=\frac{idf\left(t_{j}\right)∙tf\left(Q,t_{j}\right)}{\left‖𝒒\right‖}$

$sim_{BM25}\left(Q,D\right)=\sum_{j=1}^{M}idf\left(t_{j}\right)∙\hat{d_{j}}=$     with    $\hat{d_{j}}=\frac{tf\left(D,t_{j}\right)∙\left(k+1\right)}{tf\left(D,t_{j}\right)+k∙\left(1−b+b\frac{\left|D\right|}{adl}\right)}$         and    $idf ( t j ) = log 𝑁 − 𝑑𝑓 𝑡 𝑗 +0.5 𝑑𝑓 𝑡 𝑗 +0.5$


## 5.2.3 Inverted Files Implementation with SQL


We can build traditional text retrieval using a database with inverted lists, created through database index structures. The code on the right outlines the steps for carrying out Boolean and vector space retrieval.

We generate tables for documents, vocabulary, and postings, along with a temporary table for the query of a search. The last setup creates an index over the posting table and terms. This builds a B-tree whose leaf nodes hold document IDs and term frequencies, implementing an inverted index inside the database.

Before re-building the collection, we delete all data from all tables

Next, we go through each document in the collection. For each document, we add an entry to the document table, form a bag-of-words representation of the document, and insert tuples (term, docId, tf) into the posting table.

We count the number of documents for the calculation of idf-weights. In the code on the right, we employ the standard formula, although we could choose any variant that fits the search scenario (for Boolean searches, idf and tf values are not used). Lastly, we count the document frequency and calculate idf-weights for each term by grouping the posting table by terms and inserting the outcomes into the vocabulary table.

For Boolean AND-searches with 2 terms, we join the posting table with itself and pick postings for search terms (:term1, :term2) sharing the same docId. Since we created an index over posting(term), the database will execute two B-tree lookups to retrieve lists of (docId, tf) from leaf nodes and matching them with the WHERE-clause. Finally, we join the results with the document table to provide document details. A Boolean OR-search does not require a self-join of the posting table, yet the query still involves 2 B-tree lookups, matching the WHERE-clause, and merging with the document table to return results. While OR-queries might seem simpler (fewer joins), their evaluation complexity is the same.

To handle any number of query terms, we utilize a temporary query table and populate it with query terms (using tf=1 following the set-of-words model). For AND-queries, we link the posting and query table. The database executes a B-tree lookup for each query term, grouping them by docId. When a docId-group contains as many entries as there are query terms, it satisfies the AND-condition. We then combine these results with the document table to create the response. For OR-queries, we apply the same process, but we omit the HAVING-clause since we return all documents having at least one matching query term. Query evaluation complexity grows linearly with the number of query terms.

Using the temporary query table, we can implement various vector space models. In the code on the right, we provide an example using the dot-product measure. Similar to before, we insert the query terms into the query table and then join the query table with the posting table. This triggers B-tree lookups for the posting table for each query term, and we group the postings by docId. Since vector space models function like an OR-Boolean query for candidate selection, a HAVING-clause is not required. However, we need to join the results with both the document and vocabulary tables to calculate the scores. The final ORDER BY clause arranges the documents by decreasing scores.

Consider how we added predicates to the Python retrieval code using a priori, a posteriori, and inline filtering. Each method has advantages and limitations. In a database, we can evaluate predicates and retrieve text in the same query and let the database engine choose the best execution plan. The example on the right shows vector space retrieval with the predicate "year > 1990".


## 5.2.4 Compression and Metadata


Index compression is a key optimization for modern information retrieval systems. Effective compression can cut storage by a factor of four or more and improve query performance because more index data fits into memory and reading data from disk is faster.

The main idea behind index compression comes from posting lists in inverted indexes. Posting lists store document identifiers in sorted order, so the differences between consecutive identifiers, called delta gaps or d-gaps, are usually small integers.

  - Variable-Byte (VByte) encoding is the most widely used byte-aligned compression method for posting lists. It uses the most significant bit of each byte as a continuation flag, while the other seven bits store data. If a number needs multiple bytes, every byte except the last has its MSB set to 1. The final byte has MSB set to 0 to mark the end of the encoded integer.

  - Consider the example on the right with document id 31415. Its 32-bit binary representation is shown in the second row. We split that binary number into 7-bit groups and discard any leading zero groups. For each remaining group we add a 1 as the high bit, except for the final group. The resulting hexadecimal encoding is: 81 F5 37.

  - To decode, read bytes while their high bit is set. Then combine the bytes' lower seven bits into a 32-bit value, padding the high bits with zeros. Interpreting that value in decimal gives 31415.

delta: 	31415  (difference between consecutive IDs)

chunks:	[1000 0001]   [1111 0101]   [0011 0111]

binary:	0000 0000 0000 0000 0111 1010 1011 0111

encoding:	81 F5 37

decoding:	10000001   11110101    00110111

stop if most significant bit is 0

merge:	0000001  11110101  0110111

32-bit:	0000 0000 0000 0000 0111 1010 1011 0111

delta: 	31415

prefix with 1s except for last chunk

postings: 	[... 95673 127088 ...]

Other compression approaches:

  - PForDelta (Patched Frame-of-Reference Delta) is an advanced compression method that handles integers in fixed-size blocks, usually 128 consecutive posting deltas. For each block it finds the smallest number of bits $b$ that can represent about 90% of the values; those are called regular values. The remaining roughly 10% are stored as exceptions using full 32-bit values. The key idea is to use the unused $b$-bit slots to build linked lists that point to the exception values. This yields better compression than Variable-Byte encoding.

  - Byte-aligned compression puts length markers in the first two bits of each encoded value to show how many bytes the value uses. A common format uses these two-bit markers: 00 for one byte (values 0 to 63), 01 for two bytes (64 to 16,383), 10 for three bytes, and 11 for four bytes. This keeps data aligned to bytes for faster processing and produces compressed indexes about 15 to 20 percent the size of the uncompressed index.

Adding Metadata

  - Modern inverted indexes store extra metadata with document identifiers to support advanced queries and ranking. Term frequency metadata records how often a term appears in a document, enabling TF-IDF scoring and other relevance measures. This data is usually stored as small integers after each document ID in the posting list and compressed with the same methods used for document identifiers.

  - Document length normalization data stores precomputed values used by ranking functions such as BM25. This includes document lengths and field-specific statistics. Instead of computing these values during query processing, they are precomputed and kept in compressed structures. This uses more storage but reduces work at query time.

Support for Faceted Search

  - Faceted search needs categorical metadata so users can filter results by attributes like date ranges, geographic locations, or content types. An inverted index for faceted search stores a separate posting list for each facet value, so the system can quickly intersect those lists when processing queries.

  - Clustered indexing group similar documents by shared facet values, allowing more efficient compression by using redundancy between related posting lists. If documents in a cluster share many terms, their posting lists can be encoded relative to a reference list, yielding significant space savings.

  - Hierarchical facet compression uses the relationships among facet values to reduce storage. For example, geographic facets form a hierarchy like Country > State > City, so compressed representations store only the differences between levels instead of the full path for each item.
