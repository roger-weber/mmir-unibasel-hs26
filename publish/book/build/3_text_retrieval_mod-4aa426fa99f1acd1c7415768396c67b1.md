---
author: Roger Weber
edition: HS26
status: not-reviewed
book_part: Foundations
chapter: Classical Text Retrieval
section: Boolean Retrieval
order: "1.3"
---

(classical-text-boolean)=
# Boolean Retrieval

The previous section transformed raw text into tokens and term-frequency vectors. A retrieval model takes these representations and decides which documents match a query and, for ranked models, how strongly each document matches. We use the following small fictional library catalogue as a running example. Its wording is deliberately controlled so that the effects of preprocessing, term weighting, document length, and lexical ambiguity remain visible.

| ID | Catalogue entry |
|---|---|
| $D_1$ | **The Cat and Dog in the Forest.** A cat and a dog begin a woodland adventure. |
| $D_2$ | **Cats of the Woodland.** Wild cats begin an adventure beneath ancient trees. |
| $D_3$ | **The Forest Hound.** A loyal dog follows a woodland trail through ancient trees. |
| $D_4$ | **Feline Detective.** A clever cat solves mysteries with a canine companion. |
| $D_5$ | **Random Forest for Pet Detection.** A random forest model classifies cats and dogs in photographs. |
| $D_6$ | **Forests of Search Trees.** Algorithms explore a forest of binary trees and graph paths. |
| $D_7$ | **Dog Dog Dog!** A dog chases a ball, finds a bone, and wakes the neighbors. |
| $D_8$ | **Forest Forest Forest!** A forest guide names trees, flowers, rivers, birds, and hidden ruins. |
| $D_9$ | **Cat Dog Forest.** |
| $D_{10}$ | **Cat Dog Forest.** A cat meets a dog in a forest beside rivers, mountains, castles, villages, bridges, and caves. |
| $D_{11}$ | **Woodland Companions.** A kitten and a puppy share a moonlit adventure among old trees. |
| $D_{12}$ | **The Pet Bakery.** A cat, a dog, and a baker make cakes, bread, biscuits, pies, and coffee. |

Following [](#classical-text-feature-extraction), we must define the preprocessing pipeline before applying a retrieval model. For this example, we lowercase the text, split it at punctuation and whitespace, and remove common stop words. We deliberately apply neither stemming nor lemmatization, and we do not expand synonyms. For example:

$D_1 \mapsto [\text{cat},\text{dog},\text{forest},\text{cat},\text{dog},\text{begin},\text{woodland},\text{adventure}]$

$D_2 \mapsto [\text{cats},\text{woodland},\text{wild},\text{cats},\text{begin},\text{adventure},\text{beneath},\text{ancient},\text{trees}]$

These choices expose an important property of lexical matching: terms match by identity, not by visual or semantic similarity. During vocabulary construction, each distinct token receives its own identifier. For example, the vocabulary might map "cat" to token ID 70 and "cats" to token ID 83. A retrieval model therefore treats them as two unrelated dimensions, even though a reader immediately recognizes their relationship. For the same reason, "cat" does not match "feline" or "kitten", and "dog" does not match "hound", "canine", or "puppy". The reverse problem occurs with "forest": the same token ID represents both a natural environment and the technical concept in "random forest", although the meanings differ.

## Standard Boolean Model

Consider the query `cat AND dog`. For each document, we test two conditions: the token "cat" is present, and the token "dog" is present. The complete predicate evaluates to true only if both conditions are met. The result set is therefore $\{D_1,D_9,D_{10},D_{12}\}$. For `cat OR dog`, the predicate evaluates to true if at least one condition is met, producing the larger result set $\{D_1,D_3,D_4,D_7,D_9,D_{10},D_{12}\}$.

This is the central idea of the Standard Boolean Model: a query is a predicate that maps each document to either true or false. Documents for which the predicate is true enter the result set; all others are excluded. The model filters rather than ranks, so every returned document has the same status regardless of how often or where the terms occur. This makes Boolean retrieval simple to understand, easy to calculate, and straightforward to explain.

### Query Language

The query grammar defines two atomic predicates: a term is present, or a term is absent. It then defines two composition rules: AND and OR combine existing predicates into more complex expressions. Formally:

$$Q = t \quad \text{(term } t \text{ must be present)}$$

$$Q = \lnot t \quad \text{(term } t \text{ must be absent)}$$

$$Q = Q_1 \lor Q_2 \quad \text{(at least one sub-query must be true)}$$

$$Q = Q_1 \land Q_2 \quad \text{(both sub-queries must be true)}$$

For example, $(\text{cat} \land \text{dog}) \lor \text{hound}$ returns a document if it contains both "cat" and "dog", or if it contains "hound". Boolean algebra allows any such expression to be rewritten in disjunctive normal form (DNF):

```{admonition} Key Formula: Boolean Query in DNF
:class: important

$$Q = \bigvee_{l=1}^{L}\left(\bigwedge_{k=1}^{K_l} \tau_{l,k}\right)$$

with $\tau_{l,k} = t_{j(l,k)}$ or $\tau_{l,k} = \lnot t_{j(l,k)}$, where $j(l,k)$ maps to the index of the term used in the query.

Any Boolean query can be rewritten as a disjunction (OR) of conjunctions (AND). This normal form enables systematic evaluation using set operations.
```

### Evaluating a Query

The direct evaluation method follows the initial intuition. For each document $D_i$, the system checks whether every atomic predicate $\tau_{l,k}$ is true or false and then evaluates the complete expression. For `cat AND dog`, $D_1$ produces $\text{true} \land \text{true} = \text{true}$, whereas $D_4$ produces $\text{true} \land \text{false} = \text{false}$.

A second method starts from all documents that satisfy each atomic predicate. Let $\mathbb{S}_t$ denote the set of documents containing term $t$. For the running collection:

$\mathbb{S}_{\text{cat}} = \{D_1,D_4,D_9,D_{10},D_{12}\}$

$\mathbb{S}_{\text{dog}} = \{D_1,D_3,D_7,D_9,D_{10},D_{12}\}$

The documents satisfying `cat AND dog` must belong to both sets. The documents satisfying `cat OR dog` may belong to either set:

$\mathbb{S}_{\text{cat}} \cap \mathbb{S}_{\text{dog}} = \{D_1,D_9,D_{10},D_{12}\}$

$\mathbb{S}_{\text{cat}} \cup \mathbb{S}_{\text{dog}} = \{D_1,D_3,D_4,D_7,D_9,D_{10},D_{12}\}$

More generally, each atomic predicate defines a set of matching documents:

$\mathbb{S}_{l,k} = \begin{cases} \{D_i \mid \text{tf}(D_i, t_{j(l,k)}) \geq 1\} & \text{if } \tau_{l,k} = t_{j(l,k)} \\ \{D_i \mid \text{tf}(D_i, t_{j(l,k)}) = 0\} & \text{if } \tau_{l,k} = \lnot t_{j(l,k)}. \end{cases}$

The DNF structure translates directly into intersections for AND and unions for OR:

$\mathbb{Q} = \bigcup_{l=1}^{L} \bigcap_{k=1}^{K_l} \mathbb{S}_{l,k}$

An inverted file stores these term-document sets so that the system can combine them without scanning every document. Chapter 4 develops this index structure and its efficient query-processing algorithms.

### Limitations of Boolean Retrieval

Suppose the information need is a story involving a cat, a dog, and trees. The strict query `cat AND dog AND trees` returns no documents. Several entries are plausible partial matches, but none contains all three exact tokens. For example, $D_3$ contains "dog" and "trees", while $D_1$ contains "cat", "dog", "forest", and "woodland". Boolean evaluation excludes both because a missing term makes the complete AND predicate false.

Relaxing the query to `cat OR dog OR trees` creates the opposite problem: it returns 11 of the 12 documents. The result includes technical entries such as $D_6$ because it contains "trees", even though these are binary search trees rather than trees in an animal story. Small changes to a Boolean expression can therefore produce abrupt changes in result-set size, from no results to almost the entire collection.

The model also provides no basis for ordering the matches. For `cat AND dog`, $D_1$, $D_9$, $D_{10}$, and $D_{12}$ are equivalent Boolean matches. The model ignores that $D_1$ and $D_{10}$ repeat both query terms, that $D_9$ consists only of the three topical terms "cat", "dog", and "forest", and that much of $D_{12}$ concerns baking. Term frequency, term importance, document length, and partial evidence do not affect the result.

These limitations establish the questions for the models that follow. The Extended Boolean Model introduces graded matching and ranking while retaining Boolean query structure. Probabilistic and vector-space models provide alternative foundations for term weighting and ranking, while BM25 later combines probabilistic term importance with term-frequency saturation and document-length normalization. We will return to the same collection to see which limitation each model addresses.

**Advantages**: The model has simple and precise query semantics. Its decisions are easy to calculate and explain, and set operations support fast evaluation over large collections. Boolean predicates also combine naturally with structured metadata filters such as `language = English`.

**Disadvantages**: Users must express their information need as a Boolean expression and may obtain either too few or too many results. The model neither ranks matches nor represents partial matches. All predicates have equal influence, regardless of term frequency or discriminating power. These properties make the model better suited to exact filtering than to ranking documents by relevance.

Boolean retrieval nevertheless remains an important component of modern text and web search systems. A fast first stage often combines query terms with OR to retrieve a broad candidate set. A second stage then assigns relevance scores and ranks these candidates, allowing the user to inspect the strongest matches first. The next model takes an initial step in this direction by replacing Boolean decisions with graded scores.

(classical-text-extended-boolean)=
## Extended Boolean Model

The query `cat AND dog` gave $D_1$, $D_9$, $D_{10}$, and $D_{12}$ the same Boolean status. Yet their evidence differs: $D_1$ and $D_{10}$ contain both terms twice, while $D_9$ and $D_{12}$ contain them once. Documents such as $D_3$ and $D_4$ contain only one query term and are excluded completely. The Extended Boolean Model replaces this hard boundary with a graded score. It can rank exact matches and retain partial matches with lower scores.

### What Remains Boolean, and What Changes?

The query language remains the same. Users still express requirements with term predicates, AND, OR, and NOT, and the query retains its tree structure. What changes is the evaluation of that tree. An atomic predicate no longer produces only true or false. It produces a score between 0 and 1 that reflects the strength of the term evidence. Soft versions of AND and OR then combine these scores.

This extension directly addresses two limitations of the Standard Boolean Model. Term frequency and inverse document frequency can distinguish stronger from weaker evidence, while graded operators allow a document to receive a positive score even if it does not satisfy every predicate. The output is therefore a ranked list rather than an unordered set.

### Formalizing Graded Evidence

Each document is represented by normalized TF-IDF weights. For term $t_j$ in document $D_i$, define

$d_{i,j} = \min\left(1,\; \frac{\text{tf}(D_i,t_j)\,\text{idf}(t_j)}{\alpha}\right), \qquad 0 \leq d_{i,j} \leq 1,$

where $\alpha$ is a fixed normalization value shared by the collection. A positive term predicate uses this weight directly; a negative predicate reverses it:

$\text{sim}(\tau,D_i) = \begin{cases} d_{i,j} & \text{if } \tau=t_j, \\ 1-d_{i,j} & \text{if } \tau=\lnot t_j. \end{cases}$

The remaining question is how to combine the atomic scores. The p-norm model, introduced by Salton, Fox, and Wu; see [**Extended Boolean Information Retrieval**](https://ecommons.cornell.edu/handle/1813/6351) in the Further Reading section of the chapter summary, interprets soft AND as closeness to the ideal point $(1,\ldots,1)$ and soft OR as distance from $(0,\ldots,0)$.

```{admonition} Key Formula: P-Norm Model
:class: important

$\text{sim}\left(\bigwedge_{k=1}^{K} Q_k,D_i\right) = 1-\sqrt[p]{\frac{\sum_{k=1}^{K}\left(1-\text{sim}(Q_k,D_i)\right)^p}{K}}$

$\text{sim}\left(\bigvee_{k=1}^{K} Q_k,D_i\right) = \sqrt[p]{\frac{\sum_{k=1}^{K}\text{sim}(Q_k,D_i)^p}{K}}, \qquad 1 \leq p < \infty$

When $p=1$, both operators average their evidence. As $p$ increases, AND becomes more sensitive to the weakest predicate and OR becomes more sensitive to the strongest predicate. In the limit $p\to\infty$, they approach minimum and maximum.
```

Other soft Boolean operators are possible. They differ in how strongly one high or low atomic score influences the combined result.

```{admonition} Alternative soft Boolean operators (optional reading)
:class: note dropdown

The fuzzy set model uses the weakest score for AND and the strongest score for OR:

$\text{sim}(Q_1\land Q_2,D_i)=\min\{\text{sim}(Q_1,D_i),\text{sim}(Q_2,D_i)\}$

$\text{sim}(Q_1\lor Q_2,D_i)=\max\{\text{sim}(Q_1,D_i),\text{sim}(Q_2,D_i)\}$

This preserves the strict interpretation of AND and OR most directly, but ignores all scores except the extreme one.

The fuzzy algebraic model combines two operands using a product for AND and a probabilistic sum for OR:

$\text{sim}(Q_1\land Q_2,D_i)=\text{sim}(Q_1,D_i)\,\text{sim}(Q_2,D_i)$

$\text{sim}(Q_1\lor Q_2,D_i)=\text{sim}(Q_1,D_i)+\text{sim}(Q_2,D_i)-\text{sim}(Q_1,D_i)\,\text{sim}(Q_2,D_i)$

A further family interpolates between the minimum and maximum. For AND, a parameter closer to the minimum emphasizes the weakest condition; for OR, a parameter closer to the maximum emphasizes the strongest condition. These alternatives illustrate that softening Boolean logic requires a modelling choice: Boolean algebra alone does not determine the graded operator.
```

### Evaluating the Running Example

Consider `cat AND dog` with $p=2$. In the 12-document collection, $\text{df}(\text{cat})=5$ and $\text{df}(\text{dog})=6$, so

$\text{idf}(\text{cat})=\ln(12/5)\approx0.875, \qquad \text{idf}(\text{dog})=\ln(12/6)\approx0.693.$

For this example, choose $\alpha=4\ln 2\approx2.773$, the largest TF-IDF weight among these query terms. We compute the two atomic scores for each candidate and combine them with the p-norm AND formula.

| Document | $d_{i,\text{cat}}$ | $d_{i,\text{dog}}$ | p-norm AND score |
|---|---:|---:|---:|
| $D_1$ | 0.632 | 0.500 | 0.561 |
| $D_{10}$ | 0.632 | 0.500 | 0.561 |
| $D_7$ | 0.000 | 1.000 | 0.293 |
| $D_9$ | 0.316 | 0.250 | 0.282 |
| $D_{12}$ | 0.316 | 0.250 | 0.282 |
| $D_4$ | 0.316 | 0.000 | 0.143 |
| $D_3$ | 0.000 | 0.250 | 0.116 |

The model now distinguishes repeated evidence: $D_1$ and $D_{10}$ rank above $D_9$ and $D_{12}$. It also assigns positive scores to the partial matches $D_3$, $D_4$, and $D_7$. Evaluation can still use an inverted file: the union of the query-term postings supplies candidates, after which the system computes and sorts their scores.

### Limitations of the Extended Boolean Model

The ranking also exposes the heuristic nature of the method. $D_7$ contains "dog" four times but no "cat", yet it narrowly outranks $D_9$, which contains both requested terms. A different value of $p$, another normalization constant, or another soft operator can change this order. The query grammar states what AND and OR mean for true and false values, but it does not uniquely determine how graded evidence should be combined.

The model also inherits the lexical limitations of its representation. It still does not connect "cat" with "cats" or "feline", and it cannot distinguish the meanings of "forest". Finally, users must still translate an information need into an explicit Boolean expression. Graded evaluation softens the consequences of that expression but does not remove the burden of formulating it.

**Advantages**: The model retains the familiar Boolean query structure while adding partial matches and ranked output. TF-IDF weights distinguish terms by occurrence and collection frequency, and inverted files still support efficient candidate retrieval.

**Disadvantages**: Scores depend on heuristic choices for normalization, operators, and parameters, which can produce counter-intuitive rankings. The model retains lexical matching and explicit Boolean query formulation.

The classical p-norm model is rarely used as the primary ranker in modern search engines. Its central idea remains relevant, however: contemporary systems often combine strict Boolean filters with optional scoring clauses. This preserves explicit query constraints while allowing partial matches to receive different relevance scores. The Vector Space Model in the next section takes the next conceptual step: it replaces explicit Boolean expressions with free-text queries and ranks documents directly from their weighted term vectors.