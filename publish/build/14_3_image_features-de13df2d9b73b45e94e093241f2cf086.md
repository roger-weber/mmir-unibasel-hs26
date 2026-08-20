# Image Features

Rowley, Baluja, and Kanade (1998) from Carnegie Mellon University developed an elaborated method for finding faces of different sizes and angles. They kept the neural network small by first teaching it to recognize standard faces, and then searching through images for faces. The detection network uses a 20x20 input network (preprocessed image window). In the first layer, they created three types of receptive fields:

  - 4 times 10x10 areas,

  - 16 times 5x5 areas, and

  - 6 times overlapping 20x5 areas

  - Each area is fully connected to a hidden unit, which is then fully connected to an output. An output of 1 means a face is present, and an output of -1 means no face is present.

A second network (router network) was trained to predict the direction of a face within a window. The 20x20 input network (preprocessed image window) is connected to hidden units, which are then connected to 36 output values representing an angle. This angle can be used to adjust the face before using the detection network.

During inference, the system runs multiple times using a sliding window technique and different scales to detect faces that are larger or smaller than the standard 20x20 faces used for training.

After training, we can locate faces in an image by following these steps: first, we create a pyramid of images by making them smaller and smaller. This helps us find faces of various sizes. Then, a 20x20 window moves across the image, and for each position, the network checks if the window has a face. Because we use normalized faces, the algorithm can identify the location and orientation of faces, and estimate the position of the eyes.
While the method performs well, the need for human intervention, such as normalizing, selecting window size, rotating, equalizing histograms, and using a sliding window approach, makes it challenging to use the network for other tasks, like identifying cats or dogs. Early neural network classifiers were often only optimized for one specific task (in this case, recognizing faces) and couldn't adapt to new situations without more human involvement.

The second wave of neural network research quickly dwindled due to fundamental issues in the learning algorithm. Despite the theoretical capacity of neural networks to learn any function, this often didn't translate into practical success. Adding more hidden layers didn't necessarily improve results, and larger networks became increasingly unstable. The challenges of vanishing and exploding gradients and the competition from support vector machines (SVM) with sophisticated kernels led the field into a deadlock. Only the Canadian government continued to fund neural network research, with Geoff Hinton and his team publishing a breakthrough paper in 2006 on deep belief networks that addressed early backpropagation issues. Simultaneously, the availability of large labeled datasets and the parallel processing power of GPUs significantly accelerated the success of what is now known as deep learning.

The breakthrough moment for deep learning was a result of several key factors and developments. It began with the availability of large labeled datasets like ImageNet, which allowed deep learning models to learn from extensive data. This was further empowered by the increased computational power, particularly the use of GPUs, which made training large neural networks efficient. Advanced architectures, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), greatly improved model performance, while innovative activation functions like ReLU helped mitigate the vanishing gradient problem. Regularization techniques, including dropout and L1/L2 regularization, enhanced model generalization, and optimization algorithms like Adam and RMSprop made training more efficient. Transfer learning, where pre-trained models are fine-tuned for specific tasks, accelerated model development. Pioneering research, industry investment, and remarkable success in diverse applications contributed to the resurgence of neural networks, marking a significant breakthrough in artificial intelligence.

See the online neural network playground at http://playground.tensorflow.org/ to experiment with the limitation of early multi-layer networks.


## 14.3.1 Deep Learning Architecture


Even though CNNs became well-known in the computer vision and machine learning communities after LeNet was introduced in 1995, they didn't immediately become the dominant method. While LeNet showed good results on small early datasets, it wasn't clear if CNNs could perform well on larger, more realistic datasets. In fact, from the early 1990s until the breakthrough results of 2010s, neural networks were often outperformed by other machine learning methods like support vector machines.

Now, let's take a look at LeNet (1995). One of the first challenges in image classification is the recognition of handwritten digits in the MNIST database, which contains tens of thousands of 28x28 samples. Each digit is normalized in the 28x28 bounding box and anti-aliased, which introduced grayscale levels.

The LeNet structure uses the architecture shown below with the sigmoid activation function. It achieved 98.7% accuracy on the MNIST database without human intervention. Newer models now achieve 99.9% accuracy with deep learning improvements. Let's examine how the architecture has evolved over time.

source: https://d2l.ai/chapter_convolutional-neural-networks/lenet.html#lenet

Visit https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html  for an online visualization of the network learning for MNIST

AlexNet (2012), which used an 8-layer CNN, won the 2012 ImageNet Large Scale Visual Recognition Challenge. It was the first time, that a neural network demonstrated that learned features can surpass manually-designed features, changing the previous approach in computer vision. The architectures of AlexNet and LeNet are very similar, as shown in the figure below.

  - AlexNet is deeper than LeNet with 5 convolutional layers, 2 fully connected hidden layers, and one fully connected output layer. And, AlexNet used ReLU instead of sigmoid as its activation function.

  - The input images for AlexNet are of much higher resolution (224x224 vs. 28x28). The convolutional layers produce a vast amount of features (up to 384 features).

  - After the final convolutional layer, there are two huge fully connected layers with 4096 outputs. These layers require nearly 1GB model parameters, which was a challenge at the time of its development.

VGG-11

AlexNet

54x54x96

26x26x256

26x26x96

12x12x256

12x12x384

12x12x384

12x12x256

5x5x256

  - Finally, AlexNet can classify 1000 categories, a huge improvement from the 10 classes in early digit recognition tasks.

The Visual Geometry Group (VGG) at Oxford University created the concept of network blocks that can be used again. The VGG block is made up of a series of 3x3-convolutions followed by a max-pooling layer with a stride of 2.

  - The AlexNet architecture is updated with a sequence of VGG-blocks followed by the fully connected layers that lead into the final classification layer.

  - The original VGG network had five blocks. The first two blocks each have one convolutional layer, while the last three blocks each have two convolutional layers. The first block has 64 output channels, and each following block doubles the number of output channels until it reaches 512. This network is often referred to as VGG-11 because it has eight convolutional layers and three fully connected layers.

source: https://d2l.ai/chapter_convolutional-modern/alexnet.html

GoogleLeNet won the ILSVRC 2014 Classification Challenge, which involved 500,000 images labeled with 200 different objects. It used a three-part architecture: the stem (data input), the body (convolutions), and the head (classification). The original design included intermediate loss functions to speed up training of earlier layers, but these are no longer necessary with better learning methods.

  - The stem uses 224x224x3 images as input and applies initial convolutions and pooling to prepare the images for extracting low-level features.

  - The building blocks are called inception modules. They have different convolution types on multiple paths, which are combined at the end. In more details, the inception module provides four paths: 1) a 1x1 convolution that reduces the dimensionality of the incoming feature, 2) a 3x3 convolution preceded by a 1x1 convolution to adjust dimensionality, 3) a 5x5 convolution preceded by a 1x1 convolution to adjust dimensionality, and 4) a 3x3 max pooling layer followed by a 1x1 convolution to adjust dimensionality.

  - If we use this module again, the network model can generate paths for features that come from different sequences of these basic convolutions. This reduces the need for human involvement in designing the network and makes it more versatile and able to handle a variety of classification tasks.

  - The head of the model is using a global average pooling (7x7 maps) to transform the 2D representation into a 1D feature vector. The last fully connected layer gives the input for the softmax classification output. The model can classify into 1000 different classes.
The full architecture of GoogleLeNet:

source: https://joelouismarino.github.io/images/blog_images/blog_googlenet_keras/googlenet_diagram.png

see note

see note

Note: these intermediate loss function helped to accelerate the training process. Meanwhile, they are no longer needed as newer models and learning algorithms emerged.

ResNet

ResNet introduced a residual block, which is shown on the lower left side.

  - The left version has two paths: 1) a path with convolutions and batch normalization (the so-called residual block), and 2) a direct (identity) path from the input to the output (the so-called residual connection). The two paths are added together (not concatenated) and then given the ReLU activation function. This means the convolution layers need to maintain the same dimensions as the input (no strides, and the same number of features)

  - The main concept is that the block doesn't learn the function $f(x)$, but instead learns the delta function $g(x) = f(x) − x$ that should be added to the input. This has a key advantage because during backpropagation, the gradients flow directly through the second path to previous layers, avoiding issues like vanishing gradients. As a result, the base layers in the model train faster and the learning process becomes more stable.

  - The right version has a similar design. The first path is the same as in the left version, but the second path uses a 1x1 convolution without changing the number of features.

DenseNet builds on this concept by adding a residual connection to each following layer and combining the outputs of the layers instead of just adding them. This creates faster paths for gradient backpropagation.

  - A dense block is made up of several convolution blocks, each with the same number of features. During forward propagation, we combine the input and output of each convolution block on the feature dimension.

  - When you add a lot of dense blocks, the model becomes too complex because it increases the number of features. A transition layer reduces the number of features by using a 1x1 convolution. It also cuts the height and width in half using average pooling with a stride of 2.

source: https://d2l.ai/chapter_convolutional-modern/resnet.html

The Vision Transformer (ViT) is an image classification model that leverages a transformer like structure. But instead of working on text tokens, we first patch the image using fixed-size patches, learn embeddings for these patches, add position embeddings, and then input the sequence into a standard Transformer encoder. To classify the image, an extra learnable classification token is added to the sequence.

  - The diagram on the bottom left shows the basic layout of vision transformers. It was the first successful attempt to train a transformer encoder on ImageNet, and it produced great results compared to traditional convolutional architectures. The attention mechanism in transformer blocks keeps improving the connections between image patches, just like we do with transformers for natural language processing. The architecture converts an image into a series of vector representations. To classify it, we can add a simple multi-layer block on top that gives a probability distribution over classes using softmax.

  - The figure on the bottom right shows a typical example how the vision transformer attends to image regions that are useful for classification purposes and allows the model to focus on relevant regions of the image.

8.4 Deep Learning Architecture

source: Google Research, 2021 (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)

Image-to-text transformers represent a pivotal advancement in computer vision and natural language processing (NLP). Their underlying framework combines the strengths of convolutional neural networks (CNNs) for visual feature extraction and transformer-based models for sequence generation.

The visual encoder typically uses a CNN or a Vision Transformer (ViT) to process raw image data. CNNs extract spatially hierarchical features, while ViTs capture global image information through self-attention mechanisms. The resulting visual features are transformed into embeddings suitable for input into the text generation component.

A transformer-based decoder generates textual descriptions based on the embeddings provided by the encoder. The decoder uses self-attention to manage dependencies in the text sequence and cross-attention to align the image features with the generated words.

The models are trained using large-scale datasets of image-caption pairs, leveraging loss functions like cross-entropy for text prediction and contrastive losses for ensuring

This architectural integration of visual encoding and transformer-based sequence generation has set a benchmark in the multimodal AI domain, showcasing the transformative potential of image-to-text transformers in diverse applications.
Contrastive Language–Image Pre-training (CLIP) learns to link images with text through large-scale contrastive training. It uses two encoders: one for text and one for images. The text encoder is a transformer that produces embeddings. The image encoder, which can be a convolutional network or a Vision Transformer, converts an image into a compact vector that captures its visual content.

During training, CLIP processes many image-caption pairs at once. Each caption goes through the text encoder and each image goes through the image encoder. The model compares every image embedding to every text embedding in the batch. The correct image and its caption form a positive example, and all other image-text combinations are negative. Contrastive learning adjusts the model parameters so matching embeddings move closer in the shared space and mismatched embeddings move farther apart.

Because CLIP is trained on a wide range of natural language descriptions from the web, it learns a broad vocabulary of visual concepts. After training, it does not need fine tuning to perform classification. Instead of using a fixed set of learned output categories, it makes zero shot predictions by comparing an image embedding to text label embeddings. To classify an image, write short natural language prompts for each category and pass them through the text encoder to get label embeddings. Encode the image and compare its embedding to each label embedding. The label with the highest similarity is chosen as the predicted class.

This works because CLIP has learned a flexible alignment between text and vision. Instead of memorizing fixed categories, it maps images and text into the same geometric space. Similar meanings sit close together, while different meanings are farther apart. The zero-shot mechanism uses this geometry, allowing the model to handle recognition tasks it was not directly trained on.

source: https://arxiv.org/pdf/2103.00020

