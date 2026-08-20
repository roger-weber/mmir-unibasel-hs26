# Setting Up a Benchmark

In information retrieval, comparing methods is essential to see how well systems meet users' needs. A new ranking algorithm or neural model may look promising on its own, but without a shared test against existing approaches, claims of improvement stay anecdotal. Benchmarks give that common ground: curated collections of documents, queries, and relevance judgments that let us run the same experiments and use the same metrics. This lets us compare different approaches or fine tune hyperparameters to get better results.

There is no single best benchmark or evaluation method. Retrieval goals vary widely, from answering fact-checking questions in milliseconds to surfacing long, nuanced reports for expert analysts. A collection built for web search will not capture the challenges of legal discovery, medical literature review, or cross-lingual retrieval. Metrics that reward precision in the top ten results may be irrelevant when recall or diversity matter most. Effective evaluation depends on the intended use case: the domain, the users, and the trade-offs they care about.

An effective benchmark consists of four closely linked parts: a document collection, a set of queries or topics, a method for judging relevance, and clear performance goals. Each part helps create a fair testing environment for comparing retrieval methods.

  - Document collection. This is the foundation, the universe of data a search system must explore. It can include news articles, academic papers, legal filings, social media posts, or product descriptions, depending on the application. A good collection is stable so results remain reproducible over time, and it is representative so it reflects the real domain. Classic examples include MS MARCO, with more than half a million Bing queries and millions of passages for web style search, and the TREC collections, which cover sources from newspaper archives to biomedical abstracts. Domain specific corpora such as PubMed Central for medical literature and arXiv for scientific papers show how specialized benchmarks address particular needs.

  - Queries or topics. They express real users' information needs. They may be short keyword searches (“pizza”) or detailed natural-language questions (“For which research did Albert Einstein get the noble prize?”). A good query set covers simple factual lookups and complex exploratory tasks. For example, the TREC Web Track uses both brief navigational queries and multi-sentence investigative topics to show how systems handle different levels of complexity within the given use case scenario.

Relevance Judgments. To measure retrieval accuracy, we mark each query-document pair as relevant or not relevant. Because relevance is subjective, assessors usually follow written guidelines to keep judgments consistent. Two main approaches are used in practice:

  - Dense Assessments: In traditional competitions, each contestant runs every query. Organizers pool the documents retrieved by all participants and then judge them. This produces a relatively dense set of relevance assessments that covers all documents any system retrieved. Although not every document is judged, this approach generally preserves the relative ranking of competing algorithms. For example, an unjudged relevant document may slightly inflate an algorithm's recall estimate, but including its assessment does not change the methods' comparative performance.

  - Sparse Assessments: Large scale benchmarks like MS MARCO cover thousands of queries, so judging every retrieved document is impractical. Organizers therefore evaluate only a subset of documents per query, leaving relevance labels sparse. Many retrieved documents remain unjudged, and some relevant items may never be evaluated. As a result, a system can retrieve genuinely relevant material but receive no credit if those documents were not judged. When interpreting results, it is important to note that sparse judgments primarily affect the evaluation of top-ranked precision, while metrics measuring full-recall performance may be less reliable.

all documents

truerelevant

assessed &relevant

retrieved by any contestant

all documents

truerelevant

retrieved by any contestant

assessed & relevant

missing assessmentsimpact relative ranking

missing assessmentsbut no impact on ranking

Dense Assessments

Sparse Assessments

  - Performance goals. A benchmark also defines what success looks like by listing the metrics and criteria for the use case. For example, web search often values high precision in the top ten results and low latency. Legal discovery tends to prioritize high recall. Recommendation systems usually aim for a balance of diversity and novelty. Metrics such as Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG), or task specific cost models tie these goals to measurable outcomes. Examples (we define the metrics later in more details):

    - A web search engine should give instant answers to millions of users worldwide. Its main goal is to return highly relevant results, with top-ten precision above 85% and NDCG@10 above 0.90. Users expect near-instantaneous responses, so average query latency must stay below 200 milliseconds. The system must handle peak loads of over 10,000 queries per second and keep cost per query extremely low, ideally less than a fraction of a cent. The core challenge is balancing speed and relevance to deliver a smooth user experience.

    - In legal document discovery, the goal is to find every document relevant to a lawsuit or regulatory review, since missing even one relevant item can have serious consequences. Systems are judged mainly on high overall recall, meaning they must retrieve nearly all relevant documents rather than stopping at an arbitrary cutoff such as the top 100 results. Query latency is less important; queries that take minutes are acceptable, and throughput needs are moderate, often just a few queries per day. Precision matters too to avoid overwhelming reviewers with irrelevant material, but recall remains the primary success measure.

    - An e-commerce recommendation system delivers personalized product suggestions in real time. Success is measured by relevance, for example a high click-through rate typically above 5 percent, while keeping recommendations diverse. To keep users engaged, latency must be low and recommendations should appear within 50 milliseconds. Throughput must scale to millions of recommendations per hour to serve all active users. The system must also control costs by using efficient batch inference and caching, balancing relevance and novelty with computational efficiency.

    - For a biomedical research portal, the goal is to help scientists locate a substantial set of relevant scientific papers or clinical studies. Success requires a careful balance between recall and precision: researchers want to find as many relevant documents as possible, but high precision is also critical to avoid wasting time reading irrelevant papers. Performance is evaluated over a larger number of retrieved results (beyond just the top 20), with metrics like MAP or recall@100–200 used to reflect this balance. Latency should remain under one second to maintain a smooth search experience, and throughput is moderate, typically hundreds to thousands of queries per day. The emphasis is on supporting high-quality, evidence-based research by delivering comprehensive and relevant document sets efficiently.


## 3.1.1 Practical Aspects for Defining a Robust Benchmark


Document collection

  - Purpose and scope. The collection must match the domain or use case being evaluated, such as general web search, scientific literature, or legal documents. Results from one domain do not automatically transfer to another. When choosing the best method for an application, the benchmark must match the data and scope of that use case.

  - Reproducibility. Keep the collection fixed during experiments. To support fine-tuning and improvements over the application lifetime, keep the benchmark stable and avoid frequent expansions. Each change requires costly reassessments of relevance and re-evaluations of retrieval results. Avoid benchmarks that target a single feature, as they tend to favor that feature. Instead, design benchmarks from the user perspective and without assuming any specific search method.

  - Representativeness. Make sure the collection reflects realistic document distributions, content types, and topic diversity for the target users. Preprocess documents consistently: extract text, remove duplicates, assign metadata, and index each document with a unique identifier so retrieval and assessment are accurate.

Queries or topics

  - Purpose and realism. Queries express the target users' information needs. They should come from real user queries so they match those needs as closely as possible. For new document collections or when user data is scarce, large language models (LLMs) can generate extra queries based on user behavior patterns or on samples from the collection. This expands coverage, keeps queries realistic, and reduces manual work.

  - Clarity and consistency. Each query should state a clear information need to avoid ambiguity for assessors. Ambiguous queries reduce the consistency of relevance judgments and weaken evaluation validity. When possible, check queries with pilot tests or expert review to ensure they produce meaningful result distributions and match likely user expectations.

  - Quantity. Provide enough queries for statistically robust evaluation. Small collections may need 25 to 50 queries for basic experiments, while larger benchmarks should include 100 or more to capture topic diversity and enable meaningful comparisons across retrieval systems.

  - Maintenance and stability. Queries, like the document collection, should stay fixed to ensure reproducible results. Changing queries requires new relevance assessments and can invalidate earlier evaluations. Record all query sources, how queries were created, and how they were validated to keep experiments transparent and repeatable.

Relevance Judgements

  - Purpose and grounding. Relevance judgments are the ground truth for evaluating retrieval systems. They indicate which documents satisfy each query's information need, and they are essential for reliably measuring system performance and making meaningful comparisons. They should not change over the lifetime of a benchmark, because any change would render previous results useless.

  - Consistency and clarity. Human assessors must follow clear guidelines so their judgments are consistent. If instructions are unclear or relevance criteria are vague, assessors may disagree and the benchmark may lose validity. Training, practice examples, and regular quality checks help align assessors' judgments.

  - Scale and efficiency. Assessing every document for every query is rarely possible for large collections. Methods like pooling, which means judging only the top results from competing retrieval systems, or statistical sampling reduce the workload while keeping evaluations reliable.

  - LLM-assisted assessment. Large language models (LLMs) can create initial relevance judgments by screening documents or suggesting judgments for clear cases. Human assessors then review uncertain or borderline items. This hybrid approach improves efficiency, lowers costs, and speeds up benchmark creation, especially for new or large document collections.

Performance goals.

  - Purpose and definition. Performance goals define success for a retrieval system on a benchmark. Different domains require different performance aspects, and a metric useful in one case may be less useful in another. Precision and recall are the most common metrics, but scenario-specific definitions are often needed for meaningful results.

  - Relevance and ranking metrics. Metrics like MAP, NDCG, and precision at k measure how well a system finds relevant documents, but the importance of ranking varies by domain. In legal search, ranking matters less. High recall and good precision across all relevant documents are critical. In web search, users focus on the top results, so ranking is most important. Biomedical literature search needs a balance of recall and precision over a larger set to provide comprehensive but manageable results.

  - Operational metrics. Beyond relevance, practical performance factors include latency, throughput, and cost. Latency is the system's response time, usually measured in milliseconds or seconds, and it affects user satisfaction. Throughput is how many queries or recommendations the system can handle per unit of time, which matters in high-traffic settings. Cost covers the computing resources needed for retrieval.
