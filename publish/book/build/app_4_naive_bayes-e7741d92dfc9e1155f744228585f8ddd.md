# Naive Bayes

Naive Bayes comes from a line of probabilistic thinking that begins with Thomas Bayes, the eighteenth century minister and mathematician whose work on inverse probability led to what is now Bayes theorem. Statisticians turned his idea into a general method for updating beliefs when new evidence appears. In the twentieth century this method became important in pattern recognition, where researchers needed models that handle uncertain observations yet stay computationally simple. The Naive Bayes classifier is a practical compromise: it uses Bayes rule but assumes the features describing an instance are conditionally independent given the class label. That assumption is rarely exactly true, but it simplifies the joint probability so the model is fast, flexible and easy to train.

The model computes how likely the features are under each class and chooses the class with the highest posterior probability. Its appeal is that each feature provides a small piece of evidence, so the classifier can still work when data are sparse. In the early years of information retrieval research, this robustness drew attention. Text collections are high dimensional and dominated by rare terms. Many models struggle in that setting, but Naive Bayes handles it well because it breaks the likelihood into independent contributions from each term frequency or binary occurrence. Learning becomes counting how often terms appear in documents of each class and normalizing those counts. This estimation scales to large corpora and updates naturally when new documents arrive.

In retrieval tasks that require classification, Naive Bayes sits between indexing and ranking. When a system must decide a document's category, the classifier can use term statistics the index already stores. Its probability-based structure also fits the retrieval tradition of treating scores as estimates of relevance. Some studies have linked Naive Bayes to probability-based retrieval models, letting its simple probability model guide ranking and relevance feedback. Even though modern systems often use neural methods, Naive Bayes remains a useful baseline because it is easy to interpret and its assumptions are clear. It also performs well in low resource situations, for example when only a few labeled examples exist for each class.

Although it is simple, the model is elegant. By treating features as independent, it turns a potentially intractable problem into one that is solvable and practical. Its long use in classification and retrieval shows a lasting lesson in applied machine learning: a model does not have to mirror the worlds complexity to be useful. It must instead provide stable, scalable estimates that support decisions across domains. Naive Bayes meets that standard, so it remains in toolkits even after more advanced methods have appeared.

Naïve Bayes uses a conditional probability model based on Bayes theorem:

  - where $𝒙$ is a feature vector and $C_{k}$ the class (=target). $P\left(C_{k}\right)$ is the so-called “prior”, i.e., the knowledge (here a probability) about the distribution of classes $C_{k}$. $P\left(C_{k}\right)$ is the likelihood to observe the feature $𝒙$ for a given class $C_{k}$, and $P\left(𝒙\right)$ is the evidence to observe $𝒙$ (for any class). $P\left(𝒙\right)$ is then the so-called “posterior”, i.e., the knowledge we gain (or better: predict) given the observation of feature $𝒙$ to infer that it belongs to class $C_{k}$.

Let x be a high-dimensional vector, for instance, from a huge term space for documents. Due to the high-dimensionality and the limited set of training data, it is difficult to accurately describe the probability distribution function in such a sparse space. To simplify matters, naïve Bayes assumes conditional independence of features. This immediately leads to the following simplification:

Given the probability model, we pick the hypothesis (here: class $C_{k^{∗}}$) which is most probable. This selection rule is also known as the maximum a posteriori (MAP):

To obtain the prior $P\left(C_{k}\right)$ and the likelihood $P\left(C_{k}\right)$, we need to estimates the probability distributions based on the training set. And we need to address a number of practical issues such as numerical underflow due to the multiplication of many (small) probabilities, smoothing to address missing features, and feature selection.

$P\left(𝒙\right)=\frac{P\left(C_{k}\right)∙P\left(C_{k}\right)}{P\left(𝒙\right)}$

$posterior=\frac{likelihood ∙prior}{evidence}$

$P\left(𝒙\right)=P\left(x_{1},…,x_{M}\right)=\frac{1}{P\left(𝒙\right)}∙P(C_{k})∙\prod_{j=1}^{M}P\left(C_{k}\right)$

Note that $P\left(𝒙\right)$ is a constant over classes $c_{k}$ and scales the probabilities. For our purposes, we do not need to know it.

[MATH_ERROR]

That's it! The equation describes the decision rule of Naïve Bayes. The only thing left are the estimates for the probabilities on the right hand side

Learning process

  - Estimating $P\left(C_{k}\right)$ is the easy part: let $N_{k}$ bet the number of training items with label $C_{k}$ and let $N$ be the total number of training items. Then:

    - If the exact numbers are not clear (for instance, spam classifier: what is the ratio between spam and normal email?), the probabilities can be approximated with $P\left(C_{k}\right)=1/K$ with $K$ denoting the number of classes, i.e., equiprobable classes. This is not accurate but works well in practice.

  - To find the probability distribution $P\left(C_{k}\right)$ we first need to model the underlying distribution of values for $x_{j}$, and then learn the model parameters from the training set. The typical approach to learn estimators from training data is the maximum likelihood estimation (MLE), i.e., choosing model parameters that maximize the likelihood of making the observations given the parameters.

  - Let $x_{j}$ be discrete with values from $𝕍_{j}$. Let $N_{k}\left(x_{j}=v\right)$ with $v\in 𝕍_{j}$ be the number of training items with label $C_{k}$ that have $x_{j}=v$. In other words, it denotes how often $x_{j}=v$ is observed in the training set for items belonging to the class $C_{k}$. Naturally, we obtain

$P\left(C_{k}\right)=\frac{N_{k}}{N}$

$P\left(x_{j}=v | C_{k}\right)=\frac{N_{k}\left(x_{j}=v\right)}{N_{k}}$

  - What if a value $v$ is never seen for $x_{j}$ over a class $C_{k}$. Obviously, $P\left(x_{j}=v | C_{k}\right)=0$ and with that:

    - In other words, if $v$ was never observed for a class $C_{k}$, its presence in a new data item eliminates $C_{k}$ as a prediction regardless how well the other features support $C_{k}$. To prevent 0-probabilities, we need to smooth the probability distribution, commonly using Laplace smoothing (add-1). The idea is that we “steal” probability mass and distribute it to the values with 0-probabilities:

    - Note: the sum of $P\left(x_{j}=v | C_{k}\right)$ over all values $v\in 𝕍_{j}$ is still 1. But we got rid of 0-probabilities.

    - Red indicates “stolen” probability mass and green denotes added probability mass.

$P\left(𝒙\right)=P\left(x_{1},…,x_{j}=v,…,x_{M}\right)=0$

$P\left(x_{j}=v | C_{k}\right)=\frac{N_{k}\left(x_{j}=v\right)+1}{N_{k}+\left|𝕍_{j}\right|}$

stolen

added

  - A special case is a discrete Boolean value $x_{j}\in \{0,1\}$ denoting the presence ($x_{j}=1$) or absence ($x_{j}=0$) of a feature in the training data. In this case, the distribution follows a Bernoulli event model (or a multivariate Bernoulli event model if several values are Boolean). As the probabilities sum up to 1, only one parameter is required:

    - with $p_{k,j}$ representing the probability that the feature is present, i.e., how often $x_{j}=1$ is observed in the training set for objects with label $C_{k}$. Hence:

    - Note that smoothing is done with stealing 1 only in the extreme case that all observations are the same (either all $x_{j}=1$ or all $x_{j}=0$).

  - A final case for discrete values is the multinomial event model which is given by a feature vector $𝒙=\left(x_{1},…,x_{M}\right)$ representing a histogram with $x_{j}$ counting the number of times a feature or event $j$ was observed in the training set. An example from text classification is $x_{j}$ denoting the number of occurrences of a term $t_{j}$ in a document. The probability distribution is given by:

    - Let $n_{k,j}$ be the total number of occurrences of feature j in all training items with label $C_{k}$. Then:

$P\left(x_{j} | C_{k}\right)=\left(p_{k,j}\right)^{x_{j}}∙\left(1−p_{k,j}\right)^{\left(1−x_{j}\right)}$

$p_{k,j}=\frac{N_{k}\left(x_{j}=1\right)}{N_{k}}$

$𝑝 𝑘 , 𝑗 = min 𝑁 𝑘 − 1 , max 1 , 𝑁 𝑘 𝑥 𝑗 = 1 𝑁 𝑘$

or smoothed:

$P\left(𝒙 | C_{k}\right)=\frac{\left(\sum_{j}^{}x_{j}\right)!}{\prod_{j}^{}x_{j}!}∙\prod_{j}^{}\left(p_{k,j}\right)^{x_{j}}$

Note that the factor to the left of the product symbol is a constant when looking for the best class $C_{k}$ and hence drops in the argmax equation

$p_{k,j}=\frac{n_{k,j}}{\sum_{l}^{}n_{k,l}}$

$p_{k,j}=\frac{n_{k,j}+1}{\sum_{l}^{}n_{k,l}+M}$

or smoothed:

  - If feature values $x_{i}$ are continuous, we need to choose a model for the probability distribution $p\left(C_{k}\right)$ and then learn the parameters of the model using the training set. A common approach is assuming a Gaussian distribution with the two parameters $\m _{k,i}$ denoting the mean value, and $\sigma _{k,i}^{2}$ being the variance. The probability distribution is defined as:

    - To estimate the two parameters, we need to use the unbiased estimators based on the observations from the training set. Let $N_{k}=\left|C_{k}\right|$ be the number of training items with label $C_{k}$:

  - Using a Gaussian mixture model, we can adopt to arbitrarily shaped distribution function. We overlay $L$ normal distributions $𝒩\left(\m _{k,i,l}, \sigma _{k,i,l}^{2}\right)$ with weights $w_{l}$:

    - To learn the parameters of the normal distributions, we can use the Expectation Maximization approach (see clustering methods). In addition, we should use a validation set to adjust the hyper-parameter $L$, i.e., if $L$ is large, we may fit the probability distribution for the training set well, but cannot generalize to the validation set due to overfitting. Using least mean squared errors over the validation set provides an instrument to control $L$.

$p\left(C_{k}\right)=\frac{1}{\sqrt{2\pi \sigma _{k,i}^{2}}}∙e^{−\frac{\left(x_{i}−\m _{k,i}\right)^{2}}{2\sigma _{k,i}^{2}}}$

$\m _{k,i}=\frac{1}{N_{k}}\sum_{𝒙\in C_{k}}^{}x_{i}$

$\sigma _{k,i}=\frac{1}{N_{k}−1}\sum_{𝒙\in C_{k}}^{}\left(x_{i}−\m _{k,i}\right)^{2}$

When estimating variance from samples, we must account for the error in the estimated mean value, that is, we underestimate the variance because differences between values and the estimated mean are too small.

$p\left(C_{k}\right)=\sum_{l=1}^{L}w_{l}∙𝒩\left(\m _{k,i,l}, \sigma _{k,i,l}^{2}\right)$

Prediction

  - To predict the class $C_{k^{∗}} $ to which a new data item with features $𝒙$ belongs to, we apply the maximum a posteriori (MAP) selection:

    - With moderate to large numbers for M, we run into practical issues due to the multiplications of small probabilities (numerical underflow). To provide a stable calculation of the probabilities, naïve Bayes algorithms compute log-probabilities as the logarithm does not impact the ordering:

  - To reduce the noise of a large number of features, we can focus on a few features only that are sufficient to classify data items. In general terms, we want to identify features whose presence or absence is correlated with the data item having or not having a label. This leads to 4 tests for each of the combinations of {“feature present”, “feature not present”} and {“item in class”, “item not in class”}. If there is a strong correlation for any combination of events, then the feature is discriminative for classification. Literature provides several approaches with Chi-square and mutual information being the most prominent ones. A much simpler approach is to select the most discriminative features, much like we have seen in classical text retrieval.

[MATH_ERROR]

[MATH_ERROR]
