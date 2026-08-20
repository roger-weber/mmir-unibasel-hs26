# Neural Networks

Artificial neural networks are machine learning models inspired by the brain. Brain research has often influenced new approaches, such as using connections between non-adjacent neuron layers (multi-layer approach). Neural networks are commonly used to model the brain and its learning algorithms.

The initial phase of neural network research began in the late 1950s, primarily centered on single perceptrons in hardware. Multiple perceptrons could work in parallel but were limited to input and output connections. The well-known issue with perceptrons was their inability to learn the basic XOR function. While it was demonstrated that a two-layer network could encode XOR, its limitations were evident, marking the onset of the first AI winter.

The second wave began in the 1960s when hidden layers were introduced. Various researchers explored similar ideas, but major credit is often attributed to Rumelhart, Hinton, and Williams for their 1986 paper on backpropagation, which remains a foundational reference in textbooks. This revival led to convolutional networks, recurrent networks, belief networks, and many concepts seen in modern deep learning. Yet, the field grappled with calculation problems (vanishing and exploding gradients) and computational constraints in the 1980s and 1990s.

In the early 2000s, research and funding for the field were scarce. However, a small research team led by Hinton, funded by the Canadian government, rebranded it as "Deep Learning" and published a groundbreaking paper in 2006, introducing a fast learning algorithm for deep belief nets. Simultaneously, computing power greatly expanded. Inspired by the Canadian team, the field resurged, realizing that GPUs were up to 100 times faster than CPUs. This enabled rapid training of deep networks in hours and days instead of weeks and months. In 2011, Google initiated the Google Brain project, connecting thousands of CPUs for a network with 1 billion weights.

Over the past five years, transformers have seen significant advancements, including the growth of massive models, increased efficiency, progress in few-shot learning, and widespread applications across various domains. The launch of ChatGPT created a new “euphoria” around AI, especially, generative AI. Hinton and Hopfield received the nobel prize for Physics in 2024 for their foundational work on neural networks.

source: https://www.researchgate.net/figure/An-overview-of-recent-object-detection-performance-we-can-observe-a-significant_fig3_336934637

source: https://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part1.html

source: https://www.researchgate.net/figure/Theoretical-Nvidia-GPUs-GFLOPS-per-Watt-Data-in-Table-8-in-the-appendix_fig3_354573934


## 99.2.1 The Perceptron


Let's start with the original perceptron concept: it's essentially a binary classifier that maps a real-valued input vector $𝒙\in ℝ^{K}$ to a binary output value $f\left(𝒙\right)$:

  - Here, $𝒘\in ℝ^{K}$ represents the weights, and $b$ is the bias. Based on this definition, the perceptron divides space with a hyperplane described by $𝒘^{T}𝒙+b$. In a broader context, $L$ perceptrons with weights $𝒘_{l}$ and bias $b_{l}$ connect to $K$ input values $i_{k}$ and generate $L$ binary output values $o_{l}$. We can illustrate this general configuration as follows:

  - The learning algorithm is then as follows:                  (demo: https://codepen.io/bagrounds/full/wdqypY)

  - Convergence occurs only when the dataset is linearly separable. Otherwise, the algorithm might fail. Various variants have been developed to tackle this challenge.

$f\left(𝒙\right)=\left\{\begin{matrix}1&if  𝒘^{⊤}𝒙+b>0\\0&otherwise           \end{matrix}\right.$

Initialize the weights $w_{k,l}^{(0)}$ and the biases $b_{L}^{\left(0\right)}$ with small random values. Set a learning rate $0\leq \alpha \leq 1$

For each example $𝒙\in 𝕋$, apply it to the perceptron, i.e., let $𝒊=𝒙$

  - Calculate that actual output:	$o_{l}=f\left(\sum_{k=1}^{K}i_{k}∙w_{k,l}+b_{l}\right)$

  - Update the weights: 	$w_{k,l}^{(t+1)}=w_{k,l}^{(t)}+\alpha \left(t_{l}−o_{l}\right)∙i_{k,l}$	(i.e., only adjust if target$\ne $output)

  - Update the bias:	$b_{l}^{(t+1)}=b_{l}^{(t)}+\alpha \left(t_{l}−o_{l}\right)$	(i.e., only adjust if target$\ne $output)

$i_{1}$

$i_{2}$

$i_{K}$

.

.

.

$b_{1}$

$b_{2}$

$b_{L}$

.

.

.

$x_{1}$

$x_{2}$

$x_{K}$

$w_{k,l}$

$o_{1}$

$o_{2}$

$o_{L}$

$t_{1}$

$t_{2}$

$t_{K}$

input

sample

output

weights

bias

target

$∀1\leq l\leq L:  o_{l}=f\left(\sum_{k=1}^{K}i_{k}∙w_{k}+b\right)$

with the binary step function

$f\left(z\right)=\left\{\begin{matrix}1&z>0          \\0&otherwise \end{matrix}\right.$

In simple terms, the perceptron learning algorithm adjusts weights (and bias) only when the target differs from the output. If the output is 0 and the target is 1, weights and bias increase; otherwise, they decrease (assuming $x_{l}\geq 0$). It's important to note that the algorithm doesn't aim to optimize any objective function; it's a heuristic method for weight learning. When the data is separable, it converges to a binary space partition using a hyperplane (one of many possible partitions).

In contrast, the support vector machine (SVM) finds the best hyperplane that separates the sets while maximizing the margin (distance from marginal points to the hyperplane). SVM can also handle non-separable data by minimizing the partitioning error. The details of SVM computation are not discussed here.

A binary classifier can be adapted for learning multiclass outputs. The "one-vs-all" approach trains a binary classifier for each of the $L$ classes to distinguish class $C_{l}$ from the others. This means using $L$ perceptrons, with the binary target vector $𝒕$ having $t_{l}=1$ and all other components set to 0. During prediction, the class with the highest output value is considered the “winner”. Alternatively, the "one-vs-one" strategy employs $L(L−1)/2$ perceptrons to differentiate pairs of classes, training these perceptrons individually. Again, the class with the highest output value during prediction is regarded as the “winner”.

SVM's linear classification might seem restrictive. However, SVM employs the "kernel trick“, where data is mapped to a higher-dimensional space, enhancing separability. This mapping is typically nonlinear. The "kernel trick" means we don't explicitly compute the high-dimensional mapping; instead, we calculate the inner product required for SVM using kernels like $K\left(𝒙,𝒚\right)=\left(1+𝒙^{⊤}𝒚\right)^{2}$ in a 6-dimensional space for $𝒙,𝒚\in ℝ^{2}$. A Gaussian kernel, $𝐾 𝒙 , 𝒚 = exp − 𝛾 𝒙 − 𝒚 2$, yields an infinite-dimensional mapping $\phi $. The "kernel trick" is seen as human intervention into machine learning. SVM classification is efficient but requires an appropriate kernel function design for the problem.

Perceptron

SVM

possiblesolutions

exactly one optimal solution


## 99.2.2 Multilayer networks


Multilayer networks represent a significant evolution from the original perceptron. They introduce several key changes to the architecture: instead of a single layer, these networks feature multiple "hidden" layers situated between the input and output layers. These hidden layers allow for more complex computations. Additionally, the activation functions are not restricted to binary outputs, enabling more varied responses from the individual neurons. To optimize the performance, objective functions are used to define the optimal state for all the network parameters. This helps the network learn and adapt more effectively. Finally, a new learning algorithm, known as backpropagation, is employed to adjust the weights of the network, facilitating the training and fine-tuning.

We begin by examining the basics using a straightforward two-layer network as a concrete example. Afterward, we extend these concepts to networks of arbitrary shapes.

The network comprises two input neurons $i_{1},i_{2}$, two hidden neurons $h_{1},h_{2}$, and two output neurons $o_{1},o_{2}$. Shared biases include $b_{1}$ for the hidden neurons, and $b_{2}$ for the output neurons, with the bias modeled as a weight from a neuron always in the state 1. The connections are defined by weights $w_{1}$ to $w_{8}$, linking one layer to the next, without inter-layer connections or cycles, simplifying the topology. Additional nodes $J_{1}$ and $J_{2}$ are introduced to measure the training error, with $J$ representing the overall training error.

$i_{1}$

$i_{2}$

$1$

$h_{1}$

$h_{2}$

sample

$1$

$o_{1}$

$o_{2}$

$J_{1}$

$J_{2}$

$J$

$𝒙\in 𝕋$

$t_{1}\left(𝒙\right)$

$t_{2}\left(𝒙\right)$

input

output

hidden

error

$b_{1}$

$b_{2}$

$w_{1}$

$w_{2}$

$w_{3}$

$w_{4}$

$w_{5}$

$w_{6}$

$w_{7}$

$w_{8}$

Feed-Forward: When provided with a data sample $𝒙$ from the training set $𝕋$, the network calculates the state of each neuron using a straightforward model.

We denote the summation result as $s$ and apply the logistic activation function $\phi $, also known as the soft step function. Using this, we can determine the state of each neuron based on the input $𝒙\in 𝕋$.

The calculations are straightforward. The term feed-forward denotes that we “feed” the data sample first into the input layer, and then forward the results from one layer to the next one. Each layer can be computed concurrently. Later on, we will see different activation functions and also different approaches to connectivity and sharing of weights between subsequent layers. The principle model for neurons remains the same for most deep networks. We will also encounter special dropout neurons, that set input elements to zero with a certain probability to prevent overfitting of the network.

$b$

$w_{k}$

$1$

$a_{k}$

$s=\sum_{k}^{}a_{k}∙w_{k}+b$

$y=\phi \left(s\right)=\frac{1}{1+e^{−s}}$

$Σ$

$\phi $

$y$

input

summation

output

activation

$i_{1}=x_{1}$   and    $i_{2}=x_{2}$

$h_{1}=\phi \left(s_{h_{1}}\right)=\phi \left(w_{1}∙x_{1}+w_{2}∙x_{2}+b_{1}\right)$     and    $h_{2}=\phi \left(s_{h_{2}}\right)=\phi \left(w_{3}∙x_{1}+w_{4}∙x_{2}+b_{1}\right)$

$o_{1}=\phi \left(s_{o_{1}}\right)=\phi \left(w_{5}∙h_{1}+w_{6}∙h_{2}+b_{2}\right)=\phi \left(w_{5}∙\phi \left(w_{1}∙x_{1}+w_{2}∙x_{2}+b_{1}\right)+w_{6}∙\phi \left(w_{3}∙x_{1}+w_{4}∙x_{2}+b_{1}\right)+b_{2}\right)$

$o_{2}=\phi \left(s_{o_{2}}\right)=\phi \left(w_{7}∙h_{1}+w_{8}∙h_{2}+b_{2}\right)=\phi \left(w_{7}∙\phi \left(w_{1}∙x_{1}+w_{2}∙x_{2}+b_{1}\right)+w_{8}∙\phi \left(w_{3}∙x_{1}+w_{4}∙x_{2}+b_{1}\right)+b_{2}\right)$

$\phi \left(s\right)=\frac{1}{1+e^{−s}}$

$\phi \left(s\right)$

weightsbias
Error Function: We aim to assess the network's ability to predict targets for all data samples in the training set $𝕋$. To begin, we employ the mean square error (MSE).

  - Here, $𝜽$ represents the network's parameters. In our example, $𝜽=(w_{1},…,w_{8},b_{1},b_{2})$. The process of learning the network involves finding the parameters $\theta ^{∗}$ that minimize the error function:

  - Because of the network's size and the volume of data, solving the equation directly is often impractical. Instead, we employ the gradient descent method to iteratively find a (local) optimum. The gradient descent method relies on the gradient $𝛁J(𝜽)$ for the network's parameters $\theta $, defining the network's learning strategy:

  - Gradient descent can be slow near the (local) minimum and may exhibit zigzag behavior, especially for poorly conditioned convex functions. Moreover, with large-scale datasets and networks, gradient descent demands significant computational resources and storage capacity to compute the gradient, which we can derive directly for the network as we will explore later.

  - Improved gradient descent methods, such as SGD, Adagrad, Adam, natural gradient descent, and Nesterov Accelerated Gradient, offer solutions to standard gradient descent's limitations. They enhance convergence speed, handle complex optimization landscapes, and adapt learning rates based on parameter-specific characteristics. These techniques are vital for efficiently training machine learning models and neural networks.

$J\left(𝜽\right)=\frac{1}{\left|𝕋\right|}\sum_{𝒙\in 𝕋}^{}J\left(𝒙;𝜽\right)=\frac{1}{2∙\left|𝕋\right|} \sum_{𝒙\in 𝕋}^{}\left‖t\left(𝒙\right)−o\left(𝒙;𝜽\right)\right‖_{2}^{2}$

[MATH_ERROR]

Choose an initial random vector for $𝜽^{(0)}$ and a learning rate $0\leq \eta \leq 1$

Repeat until $\left‖𝜽^{\left(t+1\right)}−𝜽^{\left(t\right)}\right‖_{2}^{2}\leq \epsilon $   or    $t>t_{max}$

  - Compute gradient:	$∆^{(t)}=\eta ∙𝛁J\left(𝜽^{\left(t\right)}\right)$

  - Adjust parameters:	$𝜽^{\left(t+1\right)}=𝜽^{\left(t\right)}−∆^{(t)}$

  - Neural network algorithms commonly employ stochastic gradient descent (SGD) along with momentum to mitigate the zigzag problem. Instead of computing the gradient over all data samples, SGD approximates it using a   sub-set of the data (so-called mini-batch), which minimizes storage requirements. However, SGD can exhibit slow convergence in later iterations. Momentum, on the other hand, accelerates descent by retaining the gradient from the previous iteration and applying a fraction $\gamma $ of it in the current descent.

    - The momentum $\gamma $ determines how long a past gradient remains influential. Typically, we begin with $\gamma =0.5$ and gradually increase it to $\gamma =0.9$ or even higher once the initial learning stabilizes.

  - The above algorithm outlines the learning strategy. In each epoch (step 2), the entire training set is processed, adjusting the network's weights and biases for each mini-batch. The remaining task is to calculate the gradient $𝛁J(𝒙;𝜽)$ for the current data sample and the current network parameters.

Choose an initial random vector for $𝜽^{(0)}$, a learning rate $0\leq \eta \leq 1$, and a momentum $0\leq \gamma \leq 1$.

Repeat until $\left‖𝜽^{\left(t+1\right)}−𝜽^{\left(t\right)}\right‖_{2}^{2}\leq \epsilon $   or    $t>t_{max}$

  - Randomly shuffle the training set $𝕋$ into $K$ subsets $𝕋_{k}$

  - For each $𝕋_{k}$:

    - Compute gradient:	$∆=\gamma ∙∆+\eta /|𝕋_{k}|∙\sum_{𝒙\in 𝕋_{k}}^{}𝛁J(𝒙;𝜽^{(t)})$

    - Adjust parameters:	$𝜽^{\left(t+1\right)}=𝜽^{\left(t\right)}−∆$

  - Increase $\gamma $

$J\left(𝒙;𝜽\right)=\frac{1}{2}\left‖t\left(𝒙\right)−o\left(𝒙;𝜽\right)\right‖_{2}^{2}$

$𝛁J\left(𝒙;𝜽\right)=?$

Gradient Calculation: Before delving into the backpropagation algorithm, let's revisit our initial example network with two input nodes, two hidden nodes, and two output nodes. To perform stochastic gradient descent, we must compute the gradient. In our example, we have $𝜽=(w_{1},…,w_{8},b_{1},b_{2})$. The gradient is represented by the partial derivatives with respect to $J\left(𝒙;𝜽\right)$:

  - With the specified targets $t_{1}$ and $t_{2}$ for data sample $𝒙$, and using the previously defined functions $o_{1}$ and $o_{2}$ that depend on $𝒙$, as well as the weights $w_{1},…,w_{8}$ and biases $b_{1}$ and $b_{2}$.

  - Let’s start with $w_{5}$. It exclusively influences $o_{1}$, not $o_{2}$. Therefore, the partial derivative is as follows:

  - Let us start simple: consider $w_{5}$. It only occurs in $o_{1}$ but not in $o_{2}$. Thus the partial derivative is:

$𝛁J\left(𝒙;𝜽\right)=\left(\frac{\partial J}{\partial w_{1}},…,\frac{\partial J}{\partial w_{8}},\frac{\partial J}{\partial b_{1}},\frac{\partial J}{\partial b_{2}}\right)$

$J\left(𝒙;𝜽\right)=J_{1}\left(𝒙;𝜽\right)+J_{2}\left(𝒙;𝜽\right)=\frac{1}{2}∙\left(t_{1}−o_{1}\right)^{2}+\frac{1}{2}∙\left(t_{2}−o_{2}\right)^{2}$

$o_{1}=\phi \left(s_{o_{1}}\right)=\phi \left(w_{5}∙h_{1}+w_{6}∙h_{2}+b_{2}\right)                            o_{2}=\phi \left(s_{o_{2}}\right)=\phi \left(w_{7}∙h_{1}+w_{8}∙h_{2}+b_{2}\right)$

$\frac{\partial J}{\partial w_{5}}=\frac{\partial }{\partial w_{5}}\left(\frac{1}{2}∙\left(t_{1}−o_{1}\right)^{2}+\frac{1}{2}∙\left(t_{2}−o_{2}\right)^{2}\right)=\frac{\partial }{\partial w_{5}}\left(\frac{1}{2}∙\left(t_{1}−o_{1}\right)^{2}\right)=\left(t_{1}−o_{1}\right)∙\frac{\partial o_{1}}{\partial w_{5}}$

$\frac{\partial o_{1}}{\partial w_{5}}=\frac{\partial }{\partial w_{5}}\left(\phi \left(s_{o_{1}}\right)\right)=\phi \left(s_{o_{1}}\right)∙\left(1−\phi \left(s_{o_{1}}\right)\right)∙\frac{\partial s_{o}_{1} }{\partial w_{5}}=o_{1}∙\left(1−o_{1}\right)∙\frac{\partial s_{o}_{1} }{\partial w_{5}}$

$\frac{\partial s_{o}_{1} }{\partial w_{5}}=\frac{\partial }{\partial w_{5}}\left(w_{5}∙h_{1}+w_{6}∙h_{2}+b_{2}\right)=h_{1}$

all together:

$\frac{\partial J}{\partial w_{5}}=\left(t_{1}−o_{1}\right)∙o_{1}\left(1−o_{1}\right)∙h_{1}$

$\phi \left(s\right)=\frac{1}{1+e^{−s}}$$\phi ^{′}=\phi ∙\left(1−\phi \right)$

  - Similarly, we obtain the other partial derivatives $\frac{\partial J}{\partial w_{6}}$, $\frac{\partial J}{\partial w_{7}}$, $\frac{\partial J}{\partial w_{8}}$, and $\frac{\partial J}{\partial b_{2}}$. Altogether, we have:

    - We can observe recurring patterns in the calculations: the error function derivatives are multiplied by the activation function derivatives and then multiplied by the summation derivatives. To calculate the gradients, we need the results (states) from the feed-forward step, allowing us to efficiently compute the gradients using the backpropagation method.

  - Now, let's consider the remaining partial derivatives (refer to the next page for details on deriving them for $w_{1}$).

$\frac{\partial J}{\partial w_{5}}=\left(t_{1}−o_{1}\right)∙o_{1}\left(1−o_{1}\right)∙h_{1}$		 $\frac{\partial J}{\partial w_{6}}=\left(t_{1}−o_{1}\right)∙o_{1}\left(1−o_{1}\right)∙h_{2}$

$\frac{\partial J}{\partial w_{7}}=\left(t_{2}−o_{2}\right)∙o_{2}\left(1−o_{2}\right)∙h_{1}$ 		$\frac{\partial J}{\partial w_{8}}=\left(t_{2}−o_{2}\right)∙o_{2}\left(1−o_{2}\right)∙h_{2}$

$\frac{\partial J}{\partial b_{2}}=\left(t_{1}−o_{1}\right)∙o_{1}\left(1−o_{1}\right)+\left(t_{2}−o_{2}\right)∙o_{2}\left(1−o_{2}\right) $

$\frac{\partial J}{\partial w_{1}}=h_{1}∙\left(1−h_{1}\right)∙x_{1}∙\left(\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{5}+\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{7}\right)$

$\frac{\partial J}{\partial w_{2}}=h_{1}∙\left(1−h_{1}\right)∙x_{2}∙\left(\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{5}+\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{7}\right)$

$\frac{\partial J}{\partial w_{3}}=h_{2}∙\left(1−h_{2}\right)∙x_{1}∙\left(\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{6}+\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{8}\right)$

$\frac{\partial J}{\partial w_{4}}=h_{2}∙\left(1−h_{2}\right)∙x_{2}∙\left(\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{6}+\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{8}\right)$

$\frac{\partial J}{\partial b_{1}}=h_{1}∙\left(1−h_{1}\right)∙\left(\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{5}+\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{7}\right)+$

$            h_{2}∙\left(1−h_{2}\right)∙\left(\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{6}+\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{8}\right)$

  - Now, let's focus on $w_{1}$. It's worth noting that $w_{1}$ is present in $h_{1}$, which, in turn, contributes to both $o_{1}$ and $o_{2}$.

$o_{1}=\phi \left(s_{o_{1}}\right)=\phi \left(w_{5}∙h_{1}+w_{6}∙h_{2}+b_{2}\right)                            o_{2}=\phi \left(s_{o_{2}}\right)=\phi \left(w_{7}∙h_{1}+w_{8}∙h_{2}+b_{2}\right)$

$h_{1}=\phi \left(s_{h_{1}}\right)=\phi \left(w_{1}∙x_{1}+w_{2}∙x_{2}+b_{1}\right)                            h_{2}=\phi \left(s_{h_{2}}\right)=\phi \left(w_{3}∙x_{1}+w_{4}∙x_{2}+b_{1}\right)$

$\frac{\partial J}{\partial w_{1}}=\frac{\partial }{\partial w_{1}}\left(\frac{1}{2}∙\left(t_{1}−o_{1}\right)^{2}+\frac{1}{2}∙\left(t_{2}−o_{2}\right)^{2}\right)=\left(t_{1}−o_{1}\right)∙\frac{\partial o_{1}}{\partial w_{1}}+\left(t_{2}−o_{2}\right)∙\frac{\partial o_{2}}{\partial w_{1}}$

$\frac{\partial o_{1}}{\partial w_{1}}=\frac{\partial }{\partial w_{1}}\left(\phi \left(s_{o_{1}}\right)\right)=\phi \left(s_{o_{1}}\right)∙\left(1−\phi \left(s_{o_{1}}\right)\right)∙\frac{\partial s_{o}_{1} }{\partial w_{1}}=o_{1}∙\left(1−o_{1}\right)∙\frac{\partial s_{o}_{1} }{\partial w_{1}}$

$\frac{\partial s_{o}_{1} }{\partial w_{1}}=\frac{\partial }{\partial w_{1}}\left(w_{5}∙h_{1}+w_{6}∙h_{2}+b_{2}\right)=w_{5}∙\frac{\partial h_{1} }{\partial w_{1}}$

$\frac{\partial h_{1} }{\partial w_{1}}=\frac{\partial }{\partial w_{1}}\left(\phi \left(s_{h_{1}}\right)\right)=\phi \left(s_{h_{1}}\right)∙\left(1−\phi \left(s_{h_{1}}\right)\right)∙\frac{\partial s_{h}_{1} }{\partial w_{1}}=h_{1}∙\left(1−h_{1}\right)∙\frac{\partial s_{h}_{1} }{\partial w_{1}}$

$\frac{\partial s_{h}_{1} }{\partial w_{1}}=\frac{\partial }{\partial w_{1}}\left(w_{1}∙x_{1}+w_{2}∙x_{2}+b_{1}\right)=x_{1}$

all together:

$\frac{\partial J}{\partial w_{1}}=\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{5}∙h_{1}∙\left(1−h_{1}\right)∙x_{1}+$             $\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{7}∙h_{1}∙\left(1−h_{1}\right)∙x_{1}$

$\phi \left(s\right)=\frac{1}{1+e^{−s}}$$\phi ^{′}=\phi ∙\left(1−\phi \right)$

$\frac{\partial o_{2}}{\partial w_{1}}=o_{2}∙\left(1−o_{2}\right)∙\frac{\partial s_{o}_{2} }{\partial w_{1}}$

$\frac{\partial s_{o}_{2} }{\partial w_{1}}=w_{7}∙\frac{\partial h_{1} }{\partial w_{1}}$


## 99.2.3 Backpropagation


It is indeed feasible to compute all partial derivatives for the gradient, but this approach appears laborious and error-prone. Is there a simpler way? Yes, there is. Backpropagation offers a remarkably straightforward method to calculate the gradient, beginning at the error node and working backward toward the input nodes. Although it doesn't yield closed-form derivatives, it efficiently computes the gradient without redundant calculations.

  - Let's revisit the chain rule from calculus:

    - In graphical notation, we obtain the forward path to compute the composite function:

    - To compute the derivative $\frac{dz}{dx}$  for $x$, we work backward. We start by calculating $f′(y)$ and then multiply it by $g′(x)$. This requires keeping track of intermediate results to use them on the backward path for derivative calculation

$F\left(x\right)=f∘g=f\left(g\left(x\right)\right)$                      $F’\left(x\right)=f^{′}\left(g\left(x\right)\right)∙g′(x)$

or in Leibniz notation with  $z=f(y)$ and $y=g(x)$:     $\frac{dz}{dx}=\frac{dz}{dy}∙\frac{dy}{dx}=f′(y)∙g′(x)$

$g$

$f$

$x$

$z$

$y=g(x)$

$x$

$z=f(y)$

forward

$g$

$f$

$x$

$z$

$y=g(x)$

$x$

$z=f(y)$

$g′$

$f′$

$\frac{dz}{dx}=\frac{dz}{dy}∙\frac{dy}{dx}$

$1$

$\frac{dz}{dy}=f′(y)$

$\frac{dz}{dx}=\frac{dz}{dy}∙g′(x)$

$1$

$x$

$y$

forward

backward

  - Likewise, we can examine multivariable chain rules.

    - In graphical notation, we establish the forward path for function computation.

    - To calculate the derivative $\frac{dz}{dx}$ for $x$, we move backward in a manner similar to what we've done before.

$F\left(x\right)=f\left(g\left(x\right),h(x)\right)$                      $F’\left(x\right)=f^{′}\left(g\left(x\right),h(x)\right)∙g^{′}(x)+f^{′}\left(g\left(x\right),h\left(x\right)\right)∙h^{′}\left(x\right)$

or in Leibniz notation with  $z=f\left(y\right)$, $y=g(x)$ and $w=h(x)$

$\frac{dz}{dx}=\frac{dz}{dy}∙\frac{dy}{dx}+\frac{dz}{dw}∙\frac{dw}{dx}=f^{′}\left(y,w\right)∙g^{′}\left(x\right)+f^{′}\left(y,w\right)∙h′\left(x\right)$

$g$

$f$

$x$

$z$

$y=g(x)$

$x$

$z=f(y,w)$

$h$

$x$

$w=h(x)$

forward

$\frac{dz}{dy}=f′(y,w)$

$y, w$

$g$

$f$

$x$

$z$

$y=g(x)$

$x$

$z=f(y,w)$

$h$

$x$

$w=h(x)$

$g′$

$f′$

$1$

$1$

$h′$

$\frac{dz}{dy}=f′(y,w)$

$x$

$x$

$\frac{dz}{dx}=\frac{dz}{dy}∙g′(x)$

$\frac{dz}{dx}=\frac{dz}{dy}∙h′(x)$

$\frac{dz}{dx}=\frac{dz}{dy}∙\frac{dy}{dx}+\frac{dz}{dw}∙\frac{dw}{dx}$

$+$

backward

forward

Now, let's employ the chain rule within our neural network. We begin with the output neurons. To streamline the structure, we introduce a node $a_{0}$, which is always in the state 1, and the weight $w_{0}=b$, representing the bias. This adjustment simplifies the formulas. The visual representation of the forward and backward paths is illustrated below.

  - Each layer produces $\theta $-values, which are propagated back to the inputs to adjust the parameters in every layer. In the previous illustration, we employed individual biases $b_{l}$ for each node. However, if we were to use a shared bias across the layer, as in the example, we would simply sum up the deltas for nodes using the same bias, i.e.,:

$1$

$Σ_{1}$

$Σ_{k}$

$Σ_{K}$

$b$

$Σ′_{1}$

$\theta _{1}$

$\frac{\partial J}{\partial b}=\sum_{k}^{}\theta _{k}$

$\theta _{k}$

$\theta _{K}$

$b^{new}=b−∆_{b}$

$∆_{b}=\gamma ∙∆_{b}+\eta ∙\frac{\partial J}{\partial b}$

$b$

$b$

$Σ′_{k}$

$Σ′_{K}$

$1$

$1$

$1$

$Σ$

$\phi $

$s_{l}=\sum_{k}^{}a_{k}∙w_{k,l}$

$a_{k}$

$J_{l}$

$J$

$o_{l}=\phi \left(s_{l}\right)$

$J_{l}=\frac{1}{2}\left(t_{l}−o_{l}\right)^{2}$

$t$

$t_{l}$

$J=\sum_{l}^{}J_{l}$

$w_{k,l}, w_{0,l}=b_{l}$

$Σ′$

$\phi ′$

$\theta _{l}=o_{l}∙\left(1−o_{l}\right)∙\left(t_{l}−o_{l}\right)$

$J_{l}′$

$J′$

$(t_{l}−o_{l})$

$1$

$1$

$t_{l},o_{l}$

$J_{l}$

$a_{k}$

$\frac{\partial J}{\partial w_{k,l}}=a_{k}∙\theta _{l}$

$\theta _{l}$

$a_{0}=1$

$w_{k,l}^{new}=w_{k,l}−∆_{k,l}$

$∆_{k,l}=\gamma ∙∆_{k,l}+\eta ∙\frac{\partial J}{\partial w_{k,l}}$

$\phi \left(s\right)=\frac{1}{1+e^{−s}}$$\phi ^{′}=\phi ∙\left(1−\phi \right)$

$a_{k}$

$1$

backward

forward

$o_{l}=\phi \left(s_{l}\right)$

  - Hidden layers are computed in a similar manner, but during backpropagation, there are $L$ incoming edges from the subsequent layer. The visual representation of the forward and backward paths is as follows:

  - In summary, the backpropagation algorithm is a crucial component of stochastic gradient descent, where we seek the optimal parameters (weights, biases, etc.) for the network. To compute the gradient of these parameters with respect to an error function $J$, we first use the network in a forward pass to predict the output with the current parameters. Simultaneously, we track intermediate values needed for the backward pass. We then calculate the error for a single sample and propagate the partial derivatives backward through the previous layers. At each layer, we compute $∆$-values for the weights to update them. It's essential to note that the previous weights are still required for the preceding layer to compute its partial derivative (as seen in the figure above, the (+)-node relies on weights $v_{l},m$ from the subsequent layer).

$Σ′$

$\phi ′$

$\theta _{l}=o_{l}∙\left(1−o_{l}\right)∙\sum_{m}^{}v_{l,m}∙\theta _{m}$

$\sum_{m}^{}v_{l,m}∙\theta _{m}$

$o_{l}$

$a_{k}$

$\frac{\partial J}{\partial w_{k,l}}=a_{k}∙\theta _{l}$

$\theta _{l}$

$w_{k,l}^{new}=w_{k,l}−∆_{k,l}$

$∆_{k,l}=\gamma ∙∆_{k,l}+\eta ∙\frac{\partial J}{\partial w_{k,l}}$

$\phi \left(s\right)=\frac{1}{1+e^{−s}}$$\phi ^{′}=\phi ∙\left(1−\phi \right)$

$Σ$

$\phi $

$s_{l}=\sum_{k}^{}a_{k}∙w_{k,l}$

$a_{k}$

$o_{l}=\phi \left(s_{l}\right)$

$w_{k,l}, w_{0,l}=b_{l}$

$a_{0}=1$

$a_{k}$

$1$

$Σ$

$v_{l,m}$

$Σ′$

$+$

$\theta _{m}$

$o_{l}$

backward

forward

Generic implementation of multilayer networks: let us model a dense multilayer network. We assume $N$ layers $L_{i}$ and we denote $L_{0}$ to be the input layer and $L_{N}$ to be the output layer. Each layer has $M_{i}$ neurons with states $o_{i,k}$ with $0\leq i\leq N$ and $0\leq k\leq M_{i}$ whereby $o_{i,0}=1$ (used for the bias). Further we use weights $w_{i,k,l}$ with $1\leq i\leq N$, $0\leq k\leq M_{i}$ and $1\leq l\leq M_{i−1}$ to connect the $l$-th node of layer $L_{i−1}$ with the $k$-th node of layer $L_{i}$. In addition, we keep track of the increments $∆_{i,k,l}$ for the computation of the gradients $\frac{ \partial J}{\partial w_{i,k,l}}$.

  - Example with 3 layers:

  - The Feed Forward process is defined as follows:

    - So far we have used the logistic activation function $\phi \left(s\right)=\frac{1}{1+e^{−z}}$  and the mean square error (MSE) with $J\left(\theta \right)=\frac{1}{2∙\left|𝕋\right|} \sum_{𝒙\in 𝕋}^{}\left‖t\left(𝒙\right)−o\left(𝒙;𝜽\right)\right‖_{2}^{2}$ such that $E_{k}\left(o_{N,k};t_{k}\right)=\frac{1}{2}\left(t_{k}−o_{N,k}\right)^{2}$. We will see further activation functions and error (or loss) functions in the deep learning section.

$o_{0,0}=1$

$o_{0,1}$

$o_{0,M_{0}}$

…

$o_{1,0}=1$

$o_{1,1}$

$o_{1,M_{1}}$

…

$o_{2,0}=1$

$o_{2,1}$

$o_{2,M_{2}}$

…

$o_{3,1}$

$o_{3,M_{3}}$

…

$J_{1}$

$J_{M_{3}}$

…

$J$

$w_{1,k,l}$

$w_{2,k,l}$

$w_{3,k,l}$

Initialize $o_{o,k}=x_{k}$ from the current data sample $𝒙\in 𝕋⊂ℝ^{M_{0}}$ with target $𝒕\in ℝ^{M_{N}}$

For each layer $L_{i}$ with $i$ iterating from $1$ to $N$:

  - Compute $o_{i,k}=\phi (\sum_{l}^{}w_{i,k,l}∙o_{i−1,l})$ with a selected activation function $\phi $ for all $1\leq k\leq M_{i} $

Compute $J_{k}=E_{k}(o_{N,k}; t_{k})$ with a selected error function $E$ for all $1\leq k\leq M_{N}$

Compute training error $J\left(x;\theta \right)=\sum_{k}^{}J_{k}=E(o_{N,k};t_{k})$ for current sample

  - Finally, backpropagation (e.g., with logistic activation function and mean square error) is implemented as follows:

    - Note: While it may be tempting to update the weights within the inner loop (step 4), we must retain the old weights for the preceding layer in the next iteration of step 4 to compute $\theta _{i,k}$.

Modern deep learning models still use fully connected multilayer networks. However, the original approaches from the 1980s and 1990s faced several challenges, which we will address in the deep learning section. The primary issues revolved around numerical problems during gradient computation (such as vanishing and exploding values) and the substantial computational power required for training moderate to large networks. Conversely, smaller networks did not perform effectively in typical classification tasks, and alternatives emerged such as SVM with kernel functions. Eventually, SVM replaced neural networks, resulting in a temporary decline in research on neural networks after the 1990s.

Given target $𝒕$ and assume output $𝒐_{N}$ from feed forward step; assume learning rate $\eta $ and momentum $\gamma $

Initialize $∆_{i,k,l}=0$

Compute $\theta _{N,k}=\phi ^{′}\left(o_{N,k}\right)∙E_{k}^{′}\left(o_{N,k};t_{k}\right)=$ $o_{N,k}∙\left(1−o_{N,k}\right)∙\left(t_{k}−o_{N,k}\right)$ for all $1\leq k\leq M_{N}$

For each layer $L_{i}$ with $i$ iterating from $N−1$ down to $1$:

  - Compute  $\theta _{i,k}=\phi ^{′}\left(o_{i,k}\right)∙\sum_{l}^{}w_{i+1,l,k}∙\theta _{i+1,l}$ for all $1\leq k\leq M_{i}$

  - Compute $∆_{i,k,l}=\gamma ∙∆_{i,k,l}+\eta ∙o_{i−1,l}∙\theta _{i,k}$ for all $1\leq k\leq M_{i}$

Update weights $w_{i,k,l}=w_{i,k,l}−∆_{i,k,l}$


## 99.2.4 Vanishing and exploding Gradients


The second wave of neural network research quickly dwindled due to fundamental issues in the learning algorithm. Despite the theoretical capacity of neural networks to learn any function, this often didn't translate into practical success. Adding more hidden layers didn't necessarily improve results, and larger networks became increasingly unstable. The challenges of vanishing and exploding gradients and the competition from support vector machines (SVM) with sophisticated kernels led the field into a deadlock. Only the Canadian government continued to fund neural network research, with Geoff Hinton and his team publishing a breakthrough paper in 2006 on deep belief networks that addressed early backpropagation issues. Simultaneously, the availability of large labeled datasets and the parallel processing power of GPUs significantly accelerated the success of what is now known as deep learning.

First, let's address the vanishing gradient problem. In the network from the previous section, we had an input layer, a hidden layer, and an output layer. We optimized the network's parameters by minimizing a quadratic cost function. The backpropagation algorithm computes gradients and updates a weight on the first layer with the following formula:

  - The gradient involves two multiplicative terms, both having factors of the form $x∙(1−x)$ due to the use of the sigmoid activation function. Here, $x$ represents the output of a neuron after the activation function, specifically $x=\phi \left(s\right)=\frac{1}{1+e^{−s}}$. Additionally, these multiplications include the weights of the last layer. When we introduce more hidden layers to the network, we encounter more factors of the form $x∙(1−x)$ and more weights from subsequent layers in the gradients of the first layer's weights and biases.

$\frac{\partial J}{\partial w_{1}}=\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{5}∙h_{1}∙\left(1−h_{1}\right)∙x_{1}+\left(t_{2}−o_{2}\right)∙o_{2}∙\left(1−o_{2}\right)∙w_{7}∙h_{1}∙\left(1−h_{1}\right)∙x_{1}$

  - The derivative of the sigmoid function $\phi \left(s\right)=\frac{1}{1+e^{−s}}$ is depicted on the right-hand side. Notably, its maximum value is 1/4, and values rapidly diminish on both sides. When we initialize weights between 0 and 1, the gradient computation becomes a sequence of small value multiplications, resulting in very minor updates to weights and biases, even if they are significantly off target. This necessitates a large number of iterations to converge to optimal values, making the learning process slow and resource-intensive.

$\phi ^{′}=\frac{1}{1+e^{−s}}∙\left(1−\frac{1}{1+e^{−s}}\right)$

$\left(t_{1}−o_{1}\right)∙o_{1}∙\left(1−o_{1}\right)∙w_{5}∙h_{1}∙\left(1−h_{1}\right)∙x_{1}\leq 1/16$

$\leq 1/4$

$\leq 1/4$

$\leq 1$

    - This results in a quarter reduction of gradients for each layer during backpropagation, causing training of networks with many layers (such as GoogLeNet with around 20 layers) to become exceedingly slow.

  - Conversely, when we extend the weights and input values beyond the usual range of -1 to 1, the gradients start to explode because we are now multiplying several values larger than 1. With just a few layers, gradients grow exponentially as they propagate backward, causing the weights and biases to increase in absolute values. This, in turn, leads to potentially even larger gradients in the subsequent iterations, ultimately resulting in unstable gradient computations and causing numerous attempts at deeper networks to fail.

The breakthrough moment for deep learning was a result of several key factors and developments. It began with the availability of large labeled datasets like ImageNet, which allowed deep learning models to learn from extensive data. This was further empowered by the increased computational power, particularly the use of GPUs, which made training large neural networks efficient. Advanced architectures, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), greatly improved model performance, while innovative activation functions like ReLU helped mitigate the vanishing gradient problem. Regularization techniques, including dropout and L1/L2 regularization, enhanced model generalization, and optimization algorithms like Adam and RMSprop made training more efficient. Transfer learning, where pre-trained models are fine-tuned for specific tasks, accelerated model development. Pioneering research, industry investment, and remarkable success in diverse applications contributed to the resurgence of neural networks, marking a significant breakthrough in artificial intelligence.
  - Therefore, the gradients of the activation function don’t accelerate the vanishing and exploding gradients problem. ReLU has become the prevailing activation function in deep learning, despite some associated challenges:

  - With output values no longer confined to the range [0, 1], a challenge arises when mapping the last layer's output to class labels. To address this, the softmax function is employed to transform output values into class probabilities. It is frequently used alongside the cross-entropy loss function to streamline gradient computations, as shown in the following equations. Let $o_{k}$ represent the $k$-th output value, and $y_{k}$ denote the target label. Then:

  - Batch normalization (as discussed in the advanced image retrieval chapter) addresses the value range issues by normalizing the activations of each layer during training. This helps in mitigating internal covariate shift, making training more stable and efficient.

The rectified linear unit (ReLU) is a fundamental activation function, replacing the previous sigmoid function. Although various activation functions exist, ReLU played a pivotal role in ensuring more reliable gradient calculations. It is defined as:

  - The function is depicted on the right. What makes it unique? Firstly, it closely resembles the behavior of biological neurons, in contrast to the sigmoid function and hyperbolic tangent, which draw inspiration from probability theory. Secondly, its gradient is either 0 or 1:

$𝜑 𝑠 = max 0, 𝑠$

$\phi ′\left(s\right)=\left\{\begin{array}{c}0,  s<0\\\&1,  s\geq 0\end{array}\right.$

$p_{k}=\frac{e^{o_{k}}}{\sum_{j}^{}e^{o_{j}}}$

$\frac{\partial J}{\partial o_{k}}=p_{k}−y_{k}$

$𝐽 𝜽 = − 𝑘 𝑦 𝑘 ∙ log 𝑝 𝑘$

that is simple!

$𝑱$ is defined as the cross-entropy loss function. $𝜽$ contains all parameters of the network, i.e., weights and biases.
  - The derivative of ReLU can be 0, stopping backpropagation at that unit, limiting weight adjustments in early layers. While some view this as a network regularization method, resembling the sparse connections in biological neurons, others raise concerns about randomly closed paths due to initial weight and bias selections, hindering the network's learning. An alternative activation function is the leaky ReLU, defined as follows (including its derivative, plotted on the right side):

$\phi \left(s\right)=\left\{ \begin{array}{c}0.01∙s,  s<0\\\&            s,  s\geq 0\end{array}\right.$

$\phi ′\left(s\right)=\left\{ \begin{array}{c}0.01,  s<0\\\&      1,  s\geq 0\end{array}\right.$

    - The advantage is that the derivative is never becoming 0; it is small for negative values allowing a network to recover a closed path

    - The parametric ReLU is the generalization of a leaky ReLU with a learnable parameter $\alpha $ instead of the constant value used for the leaky ReLU above:

    - During training, the model learns the parameter for each activation function, which helps it adjust the slope for negative values to better fit the situation.

  - ReLU and leaky ReLU have non-smooth derivatives that jump from 0 to 1. In contrast, the Exponential Linear Unit (ELU) is a smooth, continuous function with a derivative that allows for updates to negative values:

  - Look at the right side for a graph of the function and its derivative. The derivative isn't smooth, but it doesn't suddenly change at 0. Instead, the function values keep decreasing gradually as the argument value becomes negative and smaller.

$\phi \left(s\right)=\left\{ \begin{array}{c}\alpha \left(e^{s}−1\right),  s<0\\\&                 s,  s\geq 0\end{array}\right.$

$\phi ′\left(s\right)=\left\{ \begin{array}{c}\alpha ⋅e^{s},  s<0\\\&        1,  s\geq 0\end{array}\right.$

$\phi \left(s\right)=\left\{ \begin{array}{c}\alpha ∙s,  s<0\\\&      s,  s\geq 0\end{array}\right.$

$\phi ′\left(s\right)=\left\{ \begin{array}{c}\alpha ,  s<0\\\&1,  s\geq 0\end{array}\right.$


## 99.2.5 Deep Learning Architecture Improvements


Deep learning has witnessed remarkable architectural enhancements aimed at tackling the vanishing and exploding gradient issues. Rather than relying on fully connected layers, modern deep networks have adopted a range of sophisticated techniques. These include convolutional layers, which employ only a few weights and biases to connect to numerous output neurons. This design allows for the aggregation of thousands of updates during backpropagation, significantly reducing the overall number of parameters. Additionally, regularization techniques, such as drop-out, have been employed to limit the number of active connections, thus improving training efficiency and curbing the risk of overfitting.

In the case of recurrent neural networks (RNNs), which initially had issues with the vanishing gradient problem, innovative solutions like gated recurrent units (GRUs) and long short-term memory (LSTM) cells have been instrumental. These innovations have empowered RNNs to capture long-range dependencies within sequential data. Furthermore, the integration of bidirectional RNNs and attention mechanisms has significantly enhanced their contextual understanding.

Concurrently, transformers have revolutionized the domain of natural language processing. They introduced self-attention mechanisms, allowing models to weigh the importance of different input elements dynamically. Exemplified by models like BERT and GPT-3, transformers have demonstrated their superior capabilities in pre-training and fine-tuning tasks, enabling them to better comprehend and generate human-like text. To make these advancements more accessible and versatile, both RNNs and transformers have embraced efficiency, parallelization, and scalability. These steps have ensured their practicality in a broad spectrum of real-world applications, from machine translation to image captioning. The ongoing evolution of deep learning architectures signifies substantial progress in the field, ushering in breakthroughs across multiple domains.

We omit here a detailed description of all architectural improvements and refer to the chapter on advanced text retrieval and advanced image retrieval for selected examples of architectural improvements.

Changing the network structure by cutting down on parameters may not always be the best option. It can make it harder for the network to handle complicated tasks, and smaller networks have not always worked well. But we can tweak the network structure to get more weight updates and add extra paths for updating the first layers.

    - Deep learning adds a new layer called the convolutional layer. This layer connects a small spatial neighborhood (see figure above on the right, 5x5 input neurons) to a hidden neuron. This happens for every location in the matrix, resulting in a hidden layer of the same size (padding is used at the boundaries). The output of the neuron takes the form of a 2-dimensional convolution with a trainable kernel:

    - One interesting aspect is that the weights and bias are shared by all the neurons in the new layer. This means we can train any kernel that provides useful features for the classification task, and we also get a lot of updates on weights and biases through backpropagation. For example, in a 1024x1024 layer, we get more than 1 million updates for the weights and the bias, one for each pixel. The convolution takes all input channels into account and allows for any number of output features. So, the output at the hidden neuron is not just a single value, but a multi-dimensional vector that can be used as the input for the next layer. Another interesting point is that we can remove the human interaction mentioned earlier and create a more general structure that can adjust to various tasks. We will look at some sample structures later in the chapter.

  - So far, we have been looking at layers that are fully connected to the previous layer. Each connection had its own weight, and neurons had their own bias or a shared bias. However, when it comes to understanding images, our eyes use receptive fields to pick out details from nearby areas. These fields work the same way regardless of what we are looking at.

    - In the past, images were prepared for learning using various methods (like Gaussian, Sobel, HOG) that provide receptive field inputs into the network. However, this approach also placed limits on how much a network could learn and required human intervention that may not transfer from one classification task to the next.

$o_{i,j}(𝒙)=\phi \left(b+\sum_{k,l}^{}w_{k,l}∙x_{i+k,j+l} \right)$
    - The convolutional layer typically takes a 2-dimensional input vector with $M$ dimensions and produces a 2-dimensional output vector with $N$ dimensions. For example, in image classification, we might begin with 3 channels ($M=3$) and generate 20 features per pixel ($N=20$). The convolution function is a mapping from an $M$-dimensional input vector to an $N$-dimensional output vector. At a pixel location $(x, y)$, we get:

    - Let's say we have a 5x5 convolution on $M=3$ input channels, and we want $N=20$ output features. The formula above has shared biases $b_{n}$ for each output feature, and shared weights $w_{k,l,m,n}$ for each position in the 5x5 window, for each channel, and each output feature. So, we end up with 20 biases and 1500 weights. This is true no matter the size of the input images. On the other hand, a fully connected layer would need separate biases $b_{k,l,n}$ and weights $w_{k,l,n,i,j,m}$ for each output field, feature, each input field, and channel. With a 256x256 image, this would require a huge number of biases and weights, making it impractical for even small images. Convolution layers, on the other hand, can work with any image size without increasing the number of parameters (although the cost of running the program goes up as the number of pixels increases). This reduces the number of parameters on the one hand, and provides a general structure that can learn the best features for the classification task.

$o_{i,j,n}\left(𝒙\right)=\phi \left(b_{n}+\sum_{k,l,m}^{}w_{k,l,m,n}∙x_{i+k,j+l,m}\right)$

    - Strides: Convolution uses a sliding window to calculate an output value at each location. It's possible to specify how far apart two consecutive windows should be. A stride of (2,2) means that only every other value in both dimensions is used as the starting location of the window. This results in only half as many rows and columns in the output. Strides can be used to decrease the initial size of the network. A (2,2) stride will result in 4 times fewer output neurons. This can be useful for scaling down the size of images and computing features at different scales.

    - Padding: Convolutional layers usually use odd numbers for the height and width of the kernel. By using 0s for the values outside the input matrix, we can keep the same dimensions for the output layer as the input layer.

    - Sizing: Convolution can handle images of any size, but the network architecture needs the final 2D layer to connect to a group of output neurons that generate the classification result (softmax). Strides and pooling layers gradually decrease the dimensionality until it's possible to use a fully connected network to map from a 2D layer to a 1D classification layer without the need of a large number of parameters.

    - The 1x1 convolution is used to decrease the size of the feature values and the number of parameters in the next layers. For example, if we want to learn a 5x5 convolution with 20 output features and 20 input features, we would need to learn 10000 weights and 20 biases (a total of 10020 parameters). Using a 1x1 convolution can reduce the number of parameters to learn.

      - To start, we can use a 1x1 convolution to create 3 output features from the 20 input features. This layer needs 60 weights and 3 biases, for a total of 63 parameters.

      - Next, we input the 3 features from the 1x1 convolution into a 5x5 convolution with 20 output features. This requires 1500 weights and 20 biases, for a total of 1520 parameters.

      - The new network structure has 1583 parameters, while the old one had 10020 with a simple mapping.

  - Convolution layers are often followed by Pooling Layers. Pooling summarizes the values of a neighborhood and reduces the number of neurons for the next layer.

    - Other types of summarization functions can be used alongside max pooling. Some common examples are average pooling and 2-Norm pooling. Pooling layers are crucial for controlling the size and the number of parameters in the network model. This significantly cuts down on computation and the risk of overfitting. It's important to note that pooling only reduces spatial dimensions if the stride is larger than 1. However, it doesn't reduce the number of features. To do that, a 1x1 convolution is needed, as mentioned earlier.

    - Global pooling simplifies a whole feature map to one value, eliminating the need for a fully connected layer.

  - A flattening layer is used to change the final 2D layer (usually the output of a pooling layer) into a single 1D vector for each feature value. To reduce the feature dimensions to 1, an extra 1x1 convolution is needed.

  - In image classification, we use convolution, pooling, and flattening layers together. This allows us to handle large images with fewer parameters. The basic structure is as follows:

$o_{i,j,n}\left(𝒙\right)=\max_{l,k} x_{i+k,j+l,n}$

    - For instance, look at the picture on the right side. A 2x2 max-pooling layer maps the highest value in the 2x2 window to its output neuron. If we also use a stride of (2,2), this makes each dimension 4 times smaller. If the input has multiple channels, the pooling operator is used on each channel separately. In this case, we don't use an activation function.

convolution

pooling

1x1 convolution

flattening
## 99.2.6 Regularization


Regularization is a crucial aspect of deep learning to mitigate overfitting to the training data, especially when using millions or billions of parameters:

  - As discussed in the first section of this chapter, overfitting arises when the model contains an excessive number of parameters, allowing it to memorize data rather than deduce general patterns from it. To address this issue and identify overfitting problems, we need strategies to prevent the network from merely memorizing the input-to-target mapping.

  - Overfitting is a failure to generalize which becomes apparent when we apply a trained model to new data not included in the training process. We can utilize a validation set as follows to detect common signs of overfitting:

    - Nearly perfect accuracy on the training set during training.

    - A considerably lower accuracy on the validation set at the end of training.

    - A widening gap between training accuracy and validation accuracy as training progresses.

    - Sometimes: phase of decreasing validation accuracy after a phase of progress.

epochs / iterations

accuracy

100%

gap is growing over time; significant difference

training set

validation set

epochs / iterations

accuracy

100%

still a gap but validation accuracy much closer following progress of training set

training set

validation set

Overfitting

Regularization

We can choose from various regularization methods.

  - Modifying the network structure by reducing the parameters is not always a viable choice, as it limits the ability to learn complex tasks, and small networks have shown limited success.

  - Increase the training set by augmenting existing data, such as applying small rotations, adjusting brightness, adding noise, using Gaussian filters, and more. These modifications can significantly expand the training data, increasing the dataset without requiring additional labeling.

  - Revised learning strategies with enhanced learning algorithms, weight adjustment decay, and early stopping have yielded promising outcomes for large-scale networks.

  - Modify the cost function to favor simpler models. An effective approach is to introduce a penalty into the cost function for utilizing large weights. Smaller weights, ideally approaching zero, reduce model complexity. This allows us to strike a balance between training overfitting and penalizing more intricate models. Our cost function now appears as follows (L2 regularization):

    - With $\left|𝕋\right|$ representing the number of training samples and $\lambda >0$ as the regularization parameter, it's important to note that we apply penalties to the weights, not the biases. This results in a modified update for $w_{i}$ during backpropagation. Let $∆_{i}$ denote the update for $w_{i}$ without regularization, then:

    - Regularization introduces a weight decay factor of $\left(1−\frac{\eta \lambda }{\left|𝕋\right|}\right)$ for each weight, causing them to decrease over time unless the gradient offsets this effect by increasing the weights during learning. This technique has proven effective in significantly reducing the risk of overfitting.

$J_{reg}\left(𝜽\right)=J\left(𝜽\right)+\frac{\lambda }{2∙\left|𝕋\right|}\sum_{i}^{}w_{i}^{2}$

$w_{i}^{(t+1)}=\left(1−\frac{\eta \lambda }{\left|𝕋\right|}\right)∙w_{i}^{\left(t\right)}−∆_{i}$

Increase the training set:

  - In general, bigger models with more parameters need more data to avoid overfitting and to help the model learning generalization rules. For example, a large language model with 1 billion parameters (the biggest models are now trying out 1 trillion parameters) needs 5 to 10 times more data points than parameters, which is 5 to 10 billion labeled data points. To get this much data, model designers introduced self-supervised training: the model is trained with masked sequences from a text corpus and has to predict the masked terms. Another option is to predict the next term in a sequence of terms, which can also come from a large text corpus with self-supervision.

  - To classify images, we can make changes like cropping, rotating slightly, adjusting brightness, adding noise, using Gaussian filters, and more. These changes can greatly increase the training data, expanding the dataset without needing more labeling and make it more robust (reduce variance) against small changes in the images.

  - In NLP, the foundational models are trained on a general next / masked token task in a self-supervised manner. We can then use these models and apply fine-tuning with extra layers and task-specific data. Because the foundational model has a general understanding of language, the fine-tuning requires much less data to optimize.

  - In 2016, Ian Goodfellow introduced the idea of Generative Adversarial Networks (GAN). When creating new data like images, you have two models competing with each other: 1) a generator that creates an image from noise, and 2) a discriminator that distinguishes between real and fake (i.e., generated) images. As the two models compete with each other, they become better at generating images and at telling fake from real ones over time. The discriminator tries to maximize the chance of giving the right label to both real and generated samples, while the generator tries to minimize the chance of the discriminator's right answer.

Generator

Discriminator

Predicted Label

noise



Modify the cost function: L1/L2 Regularization

  - An effective approach is to introduce a penalty into the cost function for utilizing many weights and large weights. Smaller weights, ideally approaching zero, reduce model complexity. This allows us to strike a balance between training overfitting and penalizing more intricate models. Our cost function now appears as follows (L2 Regularization):

    - With $\left|𝕋\right|$ representing the number of training samples and $\lambda >0$ as the regularization parameter, it's important to note that we apply penalties to the weights, not the biases. This results in a modified update for $w_{i}$ during backpropagation. Let $∆_{i}$ denote the update for $w_{i}$ without regularization, and $\eta $ be the learning rate, then:

    - Regularization introduces a weight decay factor of $\left(1−\frac{\eta \lambda }{\left|𝕋\right|}\right)$ for each weight, causing them to decrease over time unless the gradient offsets this effect by increasing the weights during learning. This technique has proven effective in significantly reducing the risk of overfitting. The hyperparameter $\lambda $ stays the same during training but can be changed through hyperparameter optimization. On the other hand, $\eta $ can change over the epochs.

    - Fixed Schedule: learning rate $\eta $ remains the same for all epochs

    - Learning Rate Decay: Gradually decrease the learning rate. You can do this by following a set schedule (for example, every few epochs) or by considering specific conditions (such as when your performance levels off).

    - Adaptive Methods: Utilize adaptive learning rate techniques like Adam, Adagrad, RMSprop, etc., to adjust the learning rates for each parameter based on past gradients or other factors.

  - L1 Regularization adds up the absolute weights, while L2 Regularization uses squared weights in the loss function.

$J_{reg}\left(𝜽\right)=J\left(𝜽\right)+\frac{\lambda }{2∙\left|𝕋\right|}\sum_{i}^{}w_{i}^{2}$

$w_{i}^{(t+1)}=\left(1−\frac{\eta \lambda }{\left|𝕋\right|}\right)∙w_{i}^{\left(t\right)}−∆_{i}$

Advanced Learning Strategies:

  - New learning techniques improve Stochastic Gradient Descent (SGD) to make training neural networks more efficient and effective. Momentum, RMSprop, and Adam are methods that address issues with traditional SGD. Momentum uses past gradients to speed up convergence and handle shallow or noisy gradients. RMSprop adjusts learning rates for each parameter separately, giving better results in changing environments. Adam combines momentum and RMSprop, using adaptive moment estimates and per-parameter learning rates.

  - Batch Normalization (BN) and Layer Normalization (LN) are techniques used in deep learning to improve the training stability and accelerate the convergence of neural networks. They normalize the input to a layer, mitigating issues like vanishing or exploding gradients, and allowing for more stable and faster training.

    - The sigmoid activation function limits outputs to a range of 0 to 1, which then become inputs for the next layer. However, using ReLU allows for values to become very large or very small without any limits. This can cause values to explode and disrupt the training process. To address this issue, batch and layer normalization apply a Gaussian normalization before the values enter the next layer:  $\overbar{x}=(x−mean\left(x_{i}\right))/\sqrt{var\left(x_{i}\right)+\varepsilon }$. This normalization changes the input values to a normal distribution with a mean of 0 and a standard deviation of 1.

    - Batch normalization (BN) uses the values of the mini-batch samples in the current training iteration to estimate the distribution and applies normalization to each input value separately. After training, the mean and variance values are kept as model parameters for inference. To accurately estimate the population mean and variance, larger batch sizes are needed. This can make it more challenging to train networks for tasks like object detection and semantic segmentation, which typically involve high input resolution.

    - Layer normalization (LN) uses the values from the entire layer and the mini-batch samples in the current training iteration to estimate the distribution and apply the same normalization for all input values. The mean and variance values are then kept as model parameters for inference after training. Group Normalization is similar to Layer Normalization in that it is applied along the feature direction. However, it divides the features into specific groups and normalizes each group separately. In practice, Group Normalization has been found to perform better than Layer Normalization, and its parameter "num_groups" is adjusted as a hyperparameter.

    - Weight Standardization (WS) is transforming the weights of any layer to have zero mean and unit variance. Often, this method is combined with either Batch or Layer normalization to achieve stable training conditions.

    - Generally, batch normalization works better in image classification with convolutions, while layer normalization is often used in recurrent neural networks (RNN) and transformer architectures.

  - The Dropout technique involves heuristic adjustments to the network structure during learning. At any given time, only a portion of the network is active, with nodes randomly selected for activation. This selection can vary throughout the learning process.

    - During each training step, nodes are dropped out with a probability of $1−p$. This results in different sets of active nodes learning from the training examples over time.

    - Feed forward: If a node is dropped out, its output value is set to 0, but weights and biases are retained, as the node may become active again in subsequent training steps.

    - Back propagation: When a node is dropped out, it no longer propagates changes, and the weights of connections to/from such a node do not receive updates.

    - The final prediction model utilizes all nodes but adjusts their weights by $(1−p)$. The dropout technique can be seen as training multiple networks concurrently. These individual networks are then combined into a larger network. This approach helps mitigate overfitting, as each subset of the network adapts differently to the training data. By "averaging" the networks for prediction, the impact of overfitting in one subset is balanced by the other subsets, which may have overfitted other aspects of the training data.
