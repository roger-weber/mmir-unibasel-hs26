---
author: Roger Weber
edition: HS26
status: updated
book_part: Foundations
chapter: Performance Evaluation
section: When Evaluation Misleads
order: "2.7"
---

(performance-evaluation-limits)=
# When Evaluation Misleads

Two results arrive on the same afternoon. A retrieval group reports that their new ranker raises nDCG@10 on a public benchmark from 0.41 to 0.43. A colleague reports that their spam filter reaches 99.2% accuracy on a held-out test set. Both numbers were computed correctly with the measures developed in the previous sections. Both might still be worthless.

Now that the measures are in hand, we can look at what they do not tell us. Every number in this chapter rests on a chain of decisions: which documents were collected, which needs were written down, who judged them, how the labels were split, and which measure was reported. A weakness anywhere in that chain reaches the final score, and the score itself never reveals it. Reading evaluation results critically means knowing where the chain tends to break.

## Incomplete Ground Truth

Recall the pooling arrangement from [](#performance-evaluation-benchmark-design): at realistic scale, only the documents that participating systems actually retrieved get judged, and everything unjudged counts as not relevant. That convention is what makes large benchmarks affordable, and it is also the first place scores go wrong.

In the **dense case** (TREC-style pooling), the benchmark assessed all documents that participating systems submitted in a given year. The judgments are thorough for those systems. But when a new algorithm is evaluated against the same reused collection, it may retrieve relevant documents that no original participant found. Those documents were never assessed and count as non-relevant. The new system is penalized for being original: it found material the pool missed, and the benchmark cannot reward it. A careless evaluator might even conclude the new system is worse, when in fact it discovered answers nobody had seen before.

In the **sparse case** (MS MARCO-style benchmarks with millions of queries), each query has only one or a few assessed relevant documents. [Figure %s](#fig-pooling-sparse) shows this situation. Any document that equally qualifies as a correct answer but was never selected for assessment penalizes rather than rewards the system that retrieves it. The bias is systematic: retrieval systems trained on such benchmarks tend to replicate what was selected for assessment rather than what is relevant in a broader sense.

```{figure} images/figure_2_2.png
:name: fig-pooling-sparse
:width: 50%

Sparse assessment: only a fraction of genuinely relevant documents were ever judged. Systems retrieving unjudged relevant material are penalized.
```

Recall suffers most, and in a direction that depends on who ran the experiment. Precision at rank 10 needs only the top ten documents judged, which pooling usually delivers. Recall needs the total amount of relevant material in the collection, and that is exactly what incomplete judging never establishes. For the systems that built the pool, recall comes out too high, because the divisor counts only the relevant documents somebody found. For a system evaluated later against the reused collection, the error can run the other way, since its own relevant results may be unjudged and scored as misses.

```{admonition} Unjudged is not the same as not relevant
:class: warning

Every metric in this chapter treats an unjudged document as non-relevant, because the arithmetic needs a label and there is none. That is a convention forced by cost, not a finding about the document. When comparing two systems on a sparsely judged benchmark, a lower score can mean "retrieved different documents" rather than "retrieved worse documents".
```

## Noisy Ground Truth

Even judged labels are opinions. Assessors work from written guidelines and still disagree on borderline cases, so a benchmark encodes one panel's reading of each need. This is tolerable for comparison, since every system is scored against the same labels, but it puts a floor under how small a difference we can meaningfully interpret. A gap of one point in a measure whose ground truth two qualified assessors would not fully reproduce is not a result.

Classification test sets have the same problem in a different shape. Their labels are usually complete, so pooling bias does not apply, but completeness is not correctness. Whoever labelled the messages had to decide whether an aggressive marketing newsletter counts as spam, and different labellers draw that line differently. A classifier that learned one labeller's line will look better than one that learned another's, independently of which is more useful.

Rare classes make this worse because small counts are unstable. In an intent router with four routes, a test set may contain hundreds of knowledge-base queries and a dozen calculator queries. Per-class recall for the calculator route then moves by eight percentage points when a single example changes label, and a macro-averaged score built from such classes inherits that noise. Before trusting a per-class number, check how many test items it was computed from.

## Optimizing Against the Test

A benchmark that stays fixed for years stops being a test and becomes a target. Recall how a TREC-style evaluation runs: participants receive the collection and the topics, run their systems on their own machines, and submit ranked lists that the organizers judge afterwards. Holding the topics is what makes the risk possible. A research group can read the topics, notice that several hinge on ambiguous words, add a hand-built list of expansion terms, adjust the stop word list, add exceptions to the stemmer, and keep every change that raises the score. Each step on its own looks like sound engineering. The result is a system that has absorbed the properties of this topic set and this collection: which words happen to be ambiguous here, and which documents happen to be relevant. The score rises and the method does not transfer. No dishonesty is required, because the selection pressure alone is enough.

Issuing a new topic set for each cycle limits the damage while a benchmark is live, and TREC does this. The cost is comparability, since results measured on this year's topics cannot be placed directly against last year's. The concentrated risk appears after a cycle closes, when the collection, its topics, and its judgments become a reusable test collection that papers measure against for years. Armstrong, Moffat, Webber, and Zobel examined a decade of published ad-hoc retrieval results obtained that way in [**Improvements That Don't Add Up**](https://people.eng.unimelb.edu.au/jzobel/fulltext/cikm09.pdf). Dozens of papers reported gains and many claimed statistical significance, yet the improvements did not accumulate into visible progress, and the baselines were often weaker than the median system from the original TREC cycle. Reported improvement and real improvement had come apart.

Classification has a sharper and more mechanical version of the same failure. A decision threshold tuned on the test set produces a score that cannot be trusted, because the threshold was chosen with knowledge of the answers. This is why the validation split exists, and why the threshold selection in [](#performance-evaluation-thresholds) must happen on validation data. The extreme case is contamination: if the test items and their labels were part of a model's training data, the model can score well by recalling an answer rather than by solving the task. Large language models make this problem acute, because they are trained on web-scale corpora that include publicly available exam questions. When GPT-4 is evaluated on the SAT or the MMLU benchmark, the questions and their correct answers were likely present somewhere in the training data. Researchers have shown that GPT-4 can guess missing answer options in MMLU questions with a 57% exact-match rate, far above chance, which strongly suggests memorization rather than reasoning ([Investigating Data Contamination in Modern Benchmarks for Large Language Models](https://arxiv.org/abs/2311.09783), 2023). A high score on a contaminated benchmark tells us the model saw the answers before, not that it can generalize to unseen problems.

Withholding the labels is a partial defence. MS MARCO keeps the labels of its evaluation set private and scores submissions centrally, so participants cannot tune against them directly. The leaderboard's own designers describe the limit in [**Fostering Coopetition While Plugging Leaks**](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/04/sigir2022-msmarco-leaderboard.pdf): every submission returns a score, and every score leaks a little information about the hidden set. Enough attempts turn a blind test into a slowly revealed one, which is why the official metric carries a deliberate artifact meant to frustrate reverse-engineering.

A genuinely blind evaluation reverses the direction in which data travels. Instead of sending the topics to the participant, the participant sends the system to the evaluator: code or a container executed against topics it has never seen, with judgments made only after all results are collected. This is known as evaluation as a service, or software submission. [**TIRA**](https://arxiv.org/abs/2305.18932) implements it by keeping the test data and ground truth out of public reach and running each submitted system in a sandbox that prevents data leaks. The protocol costs the organizers considerably more effort, which is why it remains the exception. One caveat survives even then: a blind topic set shows that a system never saw these needs, but for a model pretrained on web text it cannot show that the documents themselves were absent from training.

```{admonition} A benchmark score is not a claim about generalization
:class: warning

A score describes how a system performs on one collection, one topic set, and one panel's judgments, usually under a protocol that let the system's authors read the topics first. Whether the method transfers elsewhere is a separate claim needing separate evidence: a different collection, a topic set the authors never inspected, or a blind evaluation.
```

## The Test Set Is Not Production

Benchmarks are frozen so that results stay reproducible, and freezing is what makes them drift away from the world they were drawn from. Vocabulary shifts, topics appear that the collection predates, document formats change. The stability that makes a benchmark useful for comparison steadily reduces how much it says about current traffic.

For classification the mismatch is measurable, and prevalence is the clearest case. Suppose a spam filter detects 90% of spam and correctly passes 98% of legitimate mail. On a balanced test set of 500 spam and 500 legitimate messages, it produces 450 true positives and 10 false positives, so precision is $450/460 \approx 97.8\%$. Deploy the same filter, unchanged, on a mailbox where only 5% of messages are spam. Out of 500 spam and 9,500 legitimate messages it still catches 450 and still misclassifies 2% of legitimate mail, but 2% of 9,500 is 190 false positives, so precision falls to $450/640 \approx 70.3\%$.

Nothing about the classifier changed. Recall and specificity are identical in both settings. Only the class balance moved, and precision, the measure the user actually experiences, dropped by 27 percentage points. A test set whose prevalence does not match deployment therefore reports a precision that deployment will not reproduce.

## One Number Is Not a Result

A single score on a single collection is weak evidence, for a reason that has nothing to do with retrieval. Per-need scores vary enormously: the same system can reach average precision above 0.8 on one need and near zero on another. An average over 50 needs conceals that spread, so two systems differing by a point in the mean may be indistinguishable once the variation between needs is taken into account. Reporting the spread, and testing whether an observed difference exceeds it, matters as much as the measure itself.

The problem compounds across a community. When hundreds of groups tune against one public benchmark, the leading entry is partly the winner of a large search over method variants, and some of its margin is the luck of that search rather than a property of the method. This is the same multiple-comparisons effect that makes a single significance test misleading when many hypotheses were tried, and it is one mechanism behind the missing accumulation that Armstrong and colleagues documented.

## Effectiveness Is Not the Only Axis

Finally, relevance is one dimension of quality among several. A production system must also answer quickly, sustain many concurrent requests, and stay affordable, and these goals pull against each other. Later chapters on vector search make the trade-off explicit: approximate nearest-neighbour methods deliberately accept slightly worse results in exchange for large gains in speed and cost, and that exchange is often the right decision. A benchmark reporting only effectiveness cannot express it, which is why operational targets such as latency, throughput, and cost per query belong in the performance goals next to the retrieval measures.

None of this makes evaluation futile. It makes evaluation a claim with a scope. The following questions establish that scope for any reported score:

- How complete are the judgments, and could a better system have been penalized for retrieving unjudged documents?
- Who produced the labels, and how much would a second panel agree?
- Did the authors see the test data before reporting, and was any threshold or hyperparameter chosen using it?
- How closely do the collection and the class balance match the setting where the system will run?
- How large is the difference relative to the variation across needs, and how many alternatives were tried before this one was reported?
- What does the score cost in latency and compute?

A result that survives these questions is worth acting on.
