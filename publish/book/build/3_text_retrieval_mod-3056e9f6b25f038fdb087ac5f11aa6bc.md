---
author: Roger Weber
edition: HS26
status: not-reviewed
part: Foundations
chapter: Classical Text Retrieval
section: Text Retrieval Models
order: "1.3"
---

(classical-text-boolean)=
# Text Retrieval Models

In the upcoming sections, we explore various retrieval models, examining their pros and cons. While we focus on the key methods, it's important to note that there are numerous extensions in literature.

## Standard Boolean Model

The original Boolean models treated documents and queries as sets of words, aiming to find documents containing all query terms. Later, Boolean expressions enhanced queries and allowed for more complex search scenarios. A key advantage was the ability to decide for each document whether it is relevant and in the result, independently of the rest of the collection. As such, the Standard Boolean Model functions as a filtering predicate selecting relevant items rather than assessing their relevance. Initially, Boolean retrieval focused on data retrieval, lacking the capacity to rank documents by importance. We labeled these as "Retrieval-only" engines.

Boolean expressions consist of two atomic predicates and two methods for merging them into expressions. The atomic predicates are: 1) presence of a term ('must be present') and 2) absence of a term ('must not be present'). These atomic predicates are then combined using the AND and OR operators to create the query expression:

$$Q = t \quad \text{(term } t \text{ must be present)}$$

$$Q = \lnot t \quad \text{(term } t \text{ must not be present)}$$

$$Q = Q_1 \lor Q_2 \quad \text{(sub-query } Q_1 \text{ or } Q_2 \text{ fulfilled)}$$

$$Q = Q_1 \land Q_2 \quad \text{(both } Q_1 \text{ and } Q_2 \text{ fulfilled)}$$

Following the rules for Boolean expressions, we can transform the query expression into a disjunctive normal form (DNF):

```{admonition} Key Formula: Boolean Query in DNF
:class: important

$$Q = \bigvee_{l=1}^{L}\left(\bigwedge_{k=1}^{K_l} \tau_{l,k}\right)$$

with $\tau_{l,k} = t_{j(l,k)}$ or $\tau_{l,k} = \lnot t_{j(l,k)}$, where $j(l,k)$ maps to the index of the term used in the query.

Any Boolean query can be rewritten as a disjunction (OR) of conjunctions (AND). This normal form enables systematic evaluation using set operations.
```

Query evaluation can be approached in two ways: 1) individually assess the predicate for every document, and 2) employ set operations to derive the result set from the entire collection:

- For each document being examined, calculate the values for all $\tau_{l,k}$ based on the presence or absence of query terms in the document, considering whether the term 'must be present' or 'must not be present'. If the evaluation of the disjunctive normal form results in a true value, the document is marked as relevant.

- To enhance query evaluation speed, we only need to focus on documents that either contain the query term ('must be present') or don't contain it ('must not be present'). Consequently, for each atomic predicate, we can create sets $\mathbb{S}_{l,k}$ that include precisely the documents that satisfy the atomic predicate:

$$\mathbb{S}_{l,k} = \begin{cases} \{D_i \mid \text{tf}(D_i, t_{j(l,k)}) \geq 1\} & \text{if } \tau_{l,k} = t_{j(l,k)} \\ \{D_i \mid \text{tf}(D_i, t_{j(l,k)}) = 0\} & \text{if } \tau_{l,k} = \lnot t_{j(l,k)} \end{cases}$$

Following the same structure of the disjunctive normal-form of the query, we use set intersections and unions to compute the final set of relevant documents:

$$\mathbb{Q} = \bigcup_{l=1}^{L} \bigcap_{k=1}^{K_l} \mathbb{S}_{l,k}$$

Later in this chapter, we will introduce the inverted file method, which applies this evaluation scheme to provide fast response times.

```{admonition} Example
:class: example
Consider a query $Q = (\text{cat} \land \text{dog}) \lor \text{pet}$ over a collection of 5 documents:

- $D_1$: "A cat walked down the street."
- $D_2$: "The dog chased the cat."
- $D_3$: "She adopted a pet rabbit."
- $D_4$: "The cat played with the dog."
- $D_5$: "A pet dog ran in the park."

**Step 1**: Identify sets for atomic predicates:
- $\mathbb{S}_{\text{cat}} = \{D_1, D_2, D_4\}$
- $\mathbb{S}_{\text{dog}} = \{D_2, D_4, D_5\}$
- $\mathbb{S}_{\text{pet}} = \{D_3, D_5\}$

**Step 2**: Apply Boolean operations:
- $\mathbb{S}_{\text{cat}} \cap \mathbb{S}_{\text{dog}} = \{D_2, D_4\}$
- $\mathbb{Q} = \{D_2, D_4\} \cup \{D_3, D_5\} = \{D_2, D_3, D_4, D_5\}$

The result set contains 4 documents. Note that $D_1$ is excluded despite containing "cat" because it lacks "dog" and "pet".
```

**Advantages**: Simple model with clear query semantics. Easy to implement and user-friendly. Fast evaluation with sets enables quick searches, even for large data sets. Boolean expressions offer precise control for including or excluding documents, influencing result size. This model can explain why a document was considered relevant. Easy to extend with other filtering criteria over metadata of documents (e.g., language = 'English').

**Disadvantages**: Limited control over result size — users may get too few or too many results. Larger result sets lack ranking, requiring manual browsing. If the set of relevant documents is small, the method does not show 'partial matches', i.e., documents that fulfill some of the atomic predicates but not all. Although the query language is simple, users may find it hard to express a complex information need as a combination of ANDs and ORs. All terms have the same weight, hence, stop words contribute equally to the result as the more significant terms. The Boolean model resembles data retrieval more than information retrieval.

(classical-text-extended-boolean)=
## Extended Boolean Model

The method above works well when searches focus mostly on metadata, as in shop or library queries. A major drawback is that sorting does not consider how well an object matches the query. Consider the query "cat AND dog" and these three documents:

- "A cat walked down the street."
- "The dog chased the cat."
- "The cat played with the dog when another cat and dog approached them."

Documents 2 and 3 satisfy the condition "cat AND dog", but document 1 is rejected by Boolean logic even though it is partially relevant to the query. Moreover, document 3 contains the query terms more often and appears to be a better match, yet the Boolean expression treats documents 2 and 3 as equivalent.

In 1983, Salton et al. extended the Boolean model to overcome these drawbacks:

- Introduce scores for ranking, considering weights for terms and term occurrences for atomic predicates.
- Support partial matches, i.e., positive scores for documents that do not fulfill all atomic predicates.

The Extended Boolean Model adopts a bag-of-words approach, assigning normalized vectors ($\mathbf{d}_{i}$) to documents using term occurrences and inverse document frequency ($\text{idf}$). Normalization ensures values within the vector components range between 0 and 1:

$$d_{i,j} = \min\left(1,\; \frac{\text{tf}(D_i, t_j) \cdot \text{idf}(t_j)}{\alpha}\right) \quad \forall j: 1 \leq j \leq M$$

with $\alpha = \max(\text{tf}(D_i, t_j) \cdot \text{idf}(t_j))$ (or some other normalization value).

However, the query remains a Boolean expression as in the standard model. Rather than 'true' and 'false', atomic predicates yield a similarity score between 0 and 1, determined by the vector component and the 'must be present' or 'must not be present' predicate:

$$\text{sim}(\tau_{l,k}, D_i) = \begin{cases} d_{i,j(l,k)} & \text{if } \tau_{l,k} = t_{j(l,k)} \\ 1 - d_{i,j(l,k)} & \text{if } \tau_{l,k} = \lnot t_{j(l,k)} \end{cases}$$

Using the similarity scores for atomic predicates, we can establish how scores are merged for the AND and OR operators in Boolean expressions. Several common methods exist:

**Fuzzy Algebraic** (only works for two operands):

$$\text{sim}(Q_1 \land Q_2, D_i) = \text{sim}(Q_1, D_i) \cdot \text{sim}(Q_2, D_i)$$

$$\text{sim}(Q_1 \lor Q_2, D_i) = \text{sim}(Q_1, D_i) + \text{sim}(Q_2, D_i) - \text{sim}(Q_1, D_i) \cdot \text{sim}(Q_2, D_i)$$

**Fuzzy Set** (generalizes to $K$ sub-queries):

$$\text{sim}(Q_1 \land Q_2, D_i) = \min\{\text{sim}(Q_1, D_i),\; \text{sim}(Q_2, D_i)\}$$

$$\text{sim}(Q_1 \lor Q_2, D_i) = \max\{\text{sim}(Q_1, D_i),\; \text{sim}(Q_2, D_i)\}$$

**Soft Boolean Operator** (generalizes to $K$ sub-queries):

$$\text{sim}(Q_1 \land Q_2, D_i) = (1-\alpha) \cdot \min\{\text{sim}(Q_1, D_i), \text{sim}(Q_2, D_i)\} + \alpha \cdot \max\{\text{sim}(Q_1, D_i), \text{sim}(Q_2, D_i)\}$$

with $0 \leq \alpha \leq 0.5$

$$\text{sim}(Q_1 \lor Q_2, D_i) = (1-\beta) \cdot \min\{\text{sim}(Q_1, D_i), \text{sim}(Q_2, D_i)\} + \beta \cdot \max\{\text{sim}(Q_1, D_i), \text{sim}(Q_2, D_i)\}$$

with $0.5 \leq \beta \leq 1$

```{admonition} Key Formula: P-Norm Model
:class: important

$$\text{sim}\left(\bigwedge_{k=1}^{K} Q_k, D_i\right) = 1 - \sqrt[p]{\frac{\sum_{k=1}^{K}(1 - \text{sim}(Q_k, D_i))^p}{K}} \quad \text{with } 1 \leq p < \infty$$

$$\text{sim}\left(\bigvee_{k=1}^{K} Q_k, D_i\right) = \sqrt[p]{\frac{\sum_{k=1}^{K} \text{sim}(Q_k, D_i)^p}{K}}$$

The p-norm model uses distances in the query sub-vector space. When $p=1$ it behaves like averaging; as $p \to \infty$ it approaches the min/max behavior of the fuzzy set model.
```

**Advantages**: Simple model with clear query semantics as with standard Boolean model. User-friendly and easy to implement. While query evaluation is heuristic, it offers solid performance. With the inverted file method, similarity values can be efficiently computed. Unlike the standard Boolean model, it provides ranked lists and partial matches, allowing control over result size. Terms are treated differently based on term occurrence and discrimination power.

**Disadvantages**: Heuristic similarity scores lack clear theoretical explanation. Users might struggle to express complex information needs using the simple query language. Retrieval quality is decent, but other methods with similar computational complexity yield better outcomes.
