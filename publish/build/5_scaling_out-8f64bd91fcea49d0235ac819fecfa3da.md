---
author: Roger Weber
edition: HS26
status: in-progress
book_part: Search Systems
chapter: Index for Text Retrieval
section: Scaling Out
order: "4.5"
---

(indexing-scaling-out)=
# Scaling Out

A single Lucene index scales a long way. It handles up to about 2.1 billion documents, and its segment structure lets one query run in parallel across segments. But a single node has limits. Once the index grows past roughly 20 to 40 GB, query latency is bound by memory and disk throughput, and a single machine cannot serve hundreds of concurrent queries. Beyond that point we distribute the index across many machines. This section works up from parallelism inside one node to a globally distributed deployment, using the three engines built on Lucene: Apache Solr, Elasticsearch, and OpenSearch.

## The distributed engines

All three engines wrap Lucene and add clustering, horizontal scaling, and near-real-time updates. **Solr** offers a schema-driven, configuration-centric model with strong support for faceting and multilingual text, and is a common choice for enterprise and site search. **Elasticsearch** exposes a schema-flexible REST API and is heavily used for log and security analytics, where together with Logstash and Kibana it forms the ELK stack, though it is equally a capable document search engine. **OpenSearch** began in 2021 as an Apache 2.0 fork of Elasticsearch, created after Elastic changed the Elasticsearch license, and is led by a community with Amazon's backing while staying largely compatible with Elasticsearch.

Elasticsearch and OpenSearch matter beyond keyword search: both have added dense vector (approximate nearest-neighbor) search, so the same cluster can serve lexical and semantic retrieval together. We return to that hybrid setup when we cover vector search in a later chapter.

## Parallelism within a node

The first level of scaling needs no extra machines. Lucene can split an index into several segments, and a merge policy controls how many there are. [](#fig-segment-scatter-gather) shows the pattern: suppose we keep 15 equal segments on a server with 16 virtual CPUs. A coordinator thread parses the query and hands each segment to a searcher thread on one of the remaining 15 CPUs. Each thread searches its segment, and the coordinator merges the partial results before replying. This is a scatter-gather, or map-reduce, pattern: map the query across segments, reduce the partial results into one ranked list.

```{figure} images/figure_4_7.png
:name: fig-segment-scatter-gather
:width: 90%

Segment-level parallelism: a coordinator thread scatters a query across per-segment searcher threads and gathers their partial results.
```

The speedup is real but bounded. If 99% of the work is the segment search, Amdahl's law gives roughly a 13-fold speedup over a single thread. But latency still grows with the index size, only one query runs at a time at full width, and pushing to ever-larger machines produces a monolithic server with no high availability. To go further we need more nodes, not more cores.

## Sharding across nodes

Sharding distributes the collection across many smaller worker nodes. Each **shard** holds a distinct subset of the documents as its own Lucene index, with its own segments and term statistics. Search now uses two levels of map-reduce, shown in [](#fig-shard-segment-mapreduce): the query is scattered across shards, and within each shard across its segments, then reduced back up.

```{figure} images/figure_4_8.png
:name: fig-shard-segment-mapreduce
:width: 95%

Two-level map-reduce: a coordinator fans the query out across shards, and each shard fans it out across its segments before results merge bottom-up.
```

Because each shard keeps its own term statistics, the same document could score slightly differently depending on which shard holds it. Search is not an exact operation, so small score differences across shards are acceptable as long as they do not distort the overall ranking. When a document is added, a coordinator node assigns it to a shard, either by a policy such as round-robin or by hashing a chosen prefix of the document identifier so that related documents land on the same shard. The shard count is usually fixed, because redistributing documents is expensive. Sharding removes the single-node document limit and lets us run many small worker nodes, for example as containers in a Kubernetes cluster, rather than one large server.

## Replication and availability

Sharding spreads data out but does not by itself protect against node failure. For that we replicate each shard and place the replicas in different availability zones, avoiding the same server, rack, or data center. [](#fig-shard-replication-az) shows a deployment of four shards, each with one leader and one replica, arranged so that a shard's leader and replica sit in different zones. Within a shard, the leader node indexes new documents and propagates the changes to its replicas; the engines replicate at the storage level so every replica returns identical results.

```{figure} images/figure_4_9.png
:name: fig-shard-replication-az
:width: 90%

Leader and replica shards placed across two availability zones: new documents are indexed by the leader and propagated to the replica, and each shard's replica sits in a different zone from its leader.
```

Replicas do more than protect against failure. Each leader can also act as a coordinator, routing each shard's part of a query to any replica or leader that holds it. Adding replicas therefore adds concurrent-query capacity roughly in proportion to their number, up to the capacity of the coordinating nodes. Scaling concurrency this way needs more physical servers, but they can be modest ones hosting small container nodes.

## Distributing across regions

The highest level of distribution places whole clusters in several geographic regions near the users they serve. [](#fig-geo-distributed-clusters) shows two regions, one for Europe and one for Asia. Writes go to a primary region and are replicated across regions; a DNS service with a geoproximity policy resolves the application hostname to the entry point of the nearest region, with the other region as a fallback. A client is routed to its closest cluster, cutting the round-trip latency that a distant single region would impose.

```{figure} images/figure_4_10.png
:name: fig-geo-distributed-clusters
:width: 90%

Geo-distributed clusters: writes replicate from a primary region to another, and DNS geoproximity routing sends each client to its nearest cluster.
```

Regional distribution does not speed up a single query. With inter-region latencies of 100 to 200 ms, splitting one search across regions would cost more than it saves, so each query runs within one region. What regional distribution buys is more total concurrent capacity, higher availability, and lower latency for clients near a region. Without it, clients far from a single cluster would see every query pay the long round trip.
