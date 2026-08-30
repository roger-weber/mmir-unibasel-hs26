# Hidden Markov Models

Sequential data appears whenever observations unfold over time, from acoustic signals in speech recognition to DNA sequences in bioinformatics. Early work on random processes introduced Markov chains to capture dependencies between successive events. These models assume the future depends only on the present state. Many real systems hide their true condition. Speech is produced by articulatory states that cannot be measured directly, and biological processes often involve hidden functional states that give rise to observable residues or signals. To describe these invisible dynamics while modeling their visible outcomes, researchers developed Hidden Markov Models (HMMs). HMMs extend Markov chains by pairing each hidden state with a probability distribution over possible observations. As speech recognition grew more complex in the late twentieth century, HMMs became a natural and mathematically grounded tool for modeling the temporal structure of spoken language.

The classical Markov assumption says that for a sequence of states $X_{1},X_{2},…,X_{T}$, the probability of the next state depends only on the current state. Formally, $P\left(X_{t}∣X_{t−1} …X_{1}\right)=P\left(X_{t}∣X_{t−1}\right)$. Such models are adequate when the states themselves can be observed. For example, a simple weather model with states sunny, cloudy, and rainy describes how weather changes if we directly see those states. Many systems provide only indirect evidence. A speech recognizer measures acoustic features rather than phonetic labels. This motivates hidden state models. In these models a hidden state sequence $X_{t}$ evolves as a Markov chain and emits observable symbols $O_{t}$ according to emission probabilities. The observations are noisy reflections of the hidden states. For example, sensors may produce signals influenced by the true weather while the true weather remains hidden.

In speech recognition, Markov models treat hidden states as linguistic units like phonemes, which represent distinct speech sounds. Phonemes are not directly observable; the system captures acoustic features from the audio waveform, and those features serve as the observations. Hidden states follow a Markov process, so the probability of the current phoneme depends only on the previous one. The observed acoustic signals are produced in a probabilistic way by each hidden phoneme, because the sound of a phoneme can vary across speakers, background noise, and other factors.

This setup lets a speech recognizer represent how speech sounds change over time as a sequence of hidden phoneme states, using the recorded sound features as indirect evidence. The Markov assumption simplifies the model by assuming each phoneme depends mainly on nearby phonemes, and the probabilistic link between hidden states and observed sounds accounts for the variability and uncertainty in real audio recordings.

An HMM consists of

  - a finite set of hidden states $x$

  - a set of observable symbols $o$

  - a matrix of transition probabilities $a_{ij}=P(X_{t}=j∣X_{t−1}=i)$

  - a set of emission probabilities $b_{j}(o)=P(O_{t}=o∣X_{t}=j)$

  - and an initial distribution $\pi _{i}=P(X_{1}=i)$

Given a state sequence $x_{1},…,x_{T}$ and an observation sequence $o_{1},…,o_{T}$​, the joint probability factors into the product of initial, transition and emission terms

This factorization shows that hidden states follow the Markov property and that observations are independent given those states. In speech recognition, the hidden states can represent phonemes, and each state generates acoustic feature vectors from a probability distribution, often modeled by a mixture of Gaussians or other parametric models.

Working with an HMM involves three main tasks:

  - First, compute the probability of an observation sequence to measure how well the model explains the data.

  - Second, find the most likely sequence of hidden states that could have produced the observations. This is important for applications such as phoneme recognition, where the hidden states are the desired output.

  - Third, estimate the model parameters from data, whether the hidden states are known or must be inferred.

  - These three tasks form the analytical core of HMMs. Without efficient algorithms for likelihood calculation, decoding and parameter estimation, HMMs would be impractical for real-world sequences.

$P\left(x_{1},…,x_{T},o_{1},…,o_{T}\right)=\pi _{x_{1}}⋅b_{x_{1}}\left(o_{1}\right)⋅\prod_{t=2}^{T}a_{x_{t−1}}⋅b_{x_{1}}(o_{t})$

Before introducing decoding, review the ideas behind dynamic programming. Many optimization problems have optimal substructure: the best solution to a large problem can be built from best solutions to smaller subproblems. These subproblems also often overlap. Instead of listing all possible state sequences, dynamic programming stores intermediate results and reuses them, saving a lot of computation. For HMMs, finding the most probable path in a state space of size $N$ for a sequence of length $T$ would otherwise require evaluating $N^{T}$ possibilities. Dynamic programming reduces this to a manageable time by using the recursive structure implied by the Markov property.

The decoding problem is to find the hidden state sequence $x_{1}, …, x_{T}$ that maximizes the joint probability with the given observations. The Viterbi algorithm defines variables $\theta _{t}(j)$ that give the probability of the best state sequence ending in state $j$ at time $t$. The Viterbi method is an efficient and intuitive dynamic programming procedure:

  - Initialization is

  - For each time $t > 1$, the recursion is

    - which picks the best previous state $i$. At the same time, backpointers $\psi _{t}(i)$

    - record which previous state gave that maximum.

  - After reaching time $T$, the algorithm picks the state with the largest $\theta _{T}(j)$ and backtracks along the $\psi _{t}$ pointers to recover the full optimal sequence.

$\theta _{t}\left(j\right)= b_{j}\left(o_{t}\right)⋅ \max_{i}\left(\theta _{t−1}\left(i\right)⋅ a_{ij}\right)$

$\theta _{1}\left(j\right)=\pi _{j}⋅ b_{j}(o_{1})$

[MATH_ERROR]

How well you can train a hidden Markov model depends on the data you have. With labeled data, where each observation sequence comes with its true hidden state sequence, estimating parameters is simple: count state transitions and emissions, then normalize to get transition and emission probabilities. In the unsupervised case only the observations are available. The Baum-Welch algorithm uses expectation maximization (EM): it computes expected state occupancies and transitions and then updates parameters to increase the data likelihood. Viterbi training is an alternative. It finds a single best state sequence with the Viterbi algorithm and updates parameters from that hard alignment. Its updates are simpler but the method can get stuck in local optima. In speech recognition both approaches have been used, although EM remains the standard for robust estimation.

Using HMMs for long sequences creates numerical problems. Repeated multiplications of small probabilities cause underflow, so implementations use log probabilities to keep values stable. Initialization is also important in unsupervised training because poor starting parameters can make the algorithm settle in a suboptimal region of the parameter space. Despite these issues, HMMs work well in many fields. In speech tagging they map acoustic signals to phoneme or word labels. In gene prediction they represent hidden biological states that produce characteristic nucleotide patterns. For temporal segmentation they model transitions between phases of an event, for example activity cycles in sensor data. The Viterbi algorithm is the standard method for decoding in all of these cases.

Classical HMMs assume first order Markov dynamics and independent emissions. Several extensions relax these assumptions. Higher order models capture dependencies that span multiple time steps. Hierarchical HMMs add structure at different levels of abstraction to reflect complex temporal patterns. Discriminative versions change the training goal to focus on classification accuracy rather than fitting a generative model. Contemporary sequence models, such as neural networks, often outperform HMMs on large scale tasks. Still, the clarity and mathematical elegance of HMMs make them useful for applications that require interpretability. The path from Markov chains to hidden state models shows how a simple idea becomes a versatile tool for understanding sequential data.
