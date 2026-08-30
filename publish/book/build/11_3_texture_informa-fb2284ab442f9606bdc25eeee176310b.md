# Texture Information

Texture is key to image retrieval because it gives information about how colors, patterns, and shapes are arranged in an image. Color mainly refers to hue and saturation, while texture describes how those colors are distributed and organized. Texture also reveals surface details like roughness, smoothness, or grain, which help recognize and distinguish objects and scenes.

In image retrieval, systems use texture features together with color features to improve accuracy and robustness. Analyzing both color and texture gives more complete and reliable results. This combination helps systems better understand image content and makes it easier to find relevant images in large databases.

Texture describes the structure of a surface or a part of an image. It shows how color is spread in space, how that spread changes, and the direction and frequency of those changes. Texture analysis can be done in three ways:

  - Structural approach: Identify groups of basic elements called "texels" that form regular, repeating patterns, as shown in the examples below. Color information is usually ignored.

    - When the focus is specifically on identifying recurring spatial patterns or textures in satellite imagery to distinguish categories such as crops, water, or forest, it is often described as texture-based land cover classification or remote sensing image classification.

  - Statistical approach: Analyze pixel neighborhoods, measure how pixels are arranged, and produce statistical summaries like histograms and moments. We use edge detection and filters to extract texture features.

  - Fourier approach: Apply the Fourier transform to convert the image into the frequency domain, then extract information that indicates the presence of Gabor filters in that domain.

Texture analysis is often done on grayscale images. In those cases, we can use the $Y$ or $L$ component from the CIE color models. The original image must be converted from sRGB to linear RGB before converting to CIE XYZ or CIE Lab, as described in sRGB to linear RGB conversion. For this discussion, we focus on monochrome images with only a brightness or luminance channel. More advanced methods may also use chromaticity information for texture analysis.

Edge magnitude and direction (structural approach)

  - Edges in images result from various factors, as shown on the right. Detecting edges means finding large gradients or sudden changes between neighboring pixels, which mark boundaries between image regions. A common method is to smooth the image with a Gaussian and then convolve it with the Sobel operator. This produces the gradient values in the $x$ and $y$ directions, called $g_{x}$ and $g_{y}$. The kernel matrices used are shown below:A kernel is applied to an image by sliding it one pixel at a time and computing the weighted sum of the pixels it covers at each position. For the Sobel operator, the horizontal kernel highlights intensity changes from left to right and produces $g_{x}$, while the vertical kernel highlights changes from top to bottom and produces $g_{x}$.

$𝐆_{x}=\frac{1}{8}\left[\begin{matrix}+1&0&−1\\+2&0&−2\\+1&0&−1\end{matrix}\right]$

$𝐆_{y}=\frac{1}{8}\left[\begin{matrix}+1&+2&+1\\0&0&0\\−1&−2&−1\end{matrix}\right]$

$g_{mag}\left(x,y\right)=\sqrt{g_{x}\left(x,y\right)^{2}+g_{y}\left(x,y\right)^{2}}$

$g_{dir}\left(x,y\right)=\arctan(\left(g_{x}\left(x,y\right),g_{y}\left(x,y\right)\right))$

$arctan⁡(x,y)$ is the arc tangent of $y/x$ taking the quadrant of $(x,y)$ into account

Discontinuity of surface orientation (its normal vector changes)

Discontinuity of depth of vision (e.g. foreground vs. background)

Discontinuity of illumination (e.g., a shadow cast by an object)

Discontinuity of surface color of material properties

    - At each pixel, the kernel multiplies its weights by the local pixel intensities, sums the products, and writes that value to the output image. Repeating this over the whole image produces the full $g_{x}$ and $g_{y}$ gradient maps.  We can then calculate the gradient magnitude, denoted as $g_{mag}\left(x,y\right)$, and the gradient direction, represented as $g_{dir}\left(x,y\right)$, using the following formulas:

  - After the transformation above, we get two values for each image pixel. The first value gives the change magnitude (energy). The second value gives the change direction, from darker to lighter. A value of zero corresponds to a vertical edge, with the change direction perpendicular to the edge and the lighter pixel on the right.
  - We can now generate straightforward texture-based features.

    - Edge Density of Image: The percentage of the image with gradient magnitude $g_{mag}\left(x,y\right)\geq \ta $ for a specified threshold $\ta $. This indicates the presence of edges with sufficient energy. Continuous regions in the image, like the sky or a lake, yield lower values, while images with multiple objects or cityscapes have higher values.

    - Gradient Histograms: This method is analogous to the color histogram approach, but we now have only two values per pixel to quantify (direction and magnitude). Remember that differences in direction are computed as $min 𝛼 − 𝛽 ,2 𝜋 − 𝛼 − 𝛽$. We must normalize energy and direction ranges to calculate the distance $d_{i,j}$ between two reference gradients. This allows us to compute the matrix $𝐀$ for the quadratic distance measure. When given two histograms $𝒉$ and $𝒈$, and assuming $N$ reference gradients, we obtain distances as follows:

    - Gradient Moments: We compute moments for both magnitude and direction and compute a covariance between them. Let X denote either magnitude or direction. This produces nine feature values that describe the gradient distribution:

$f_{edge}=\frac{1}{N∙M}\sum_{x,y}^{}\left\{\begin{matrix}1&if  g_{mag}\left(x,y\right)\geq \ta \\0&otherwise              \end{matrix}\right.$

$\theta _{Euclidean}\left(𝒉,𝒈\right)=\sqrt{\sum_{i=1}^{N}\left(h_{i}−g_{i}\right)^{2}}$

$\theta _{quadratic}\left(𝒉,𝒈\right)=\left(𝒉−𝒈\right)^{⊤}𝐀(𝒉−𝒈)$

$𝐀: a_{i,j}=1−\frac{d_{i,j}}{\max_{k,l}d_{k,l}}$

Distance normalized by maximum distance for all pairs of reference gradients

$\theta _{Manhattan}\left(𝒉,𝒈\right)=\sum_{i=1}^{N}\left|h_{i}−g_{i}\right|$

$v_{c}=\frac{1}{N∙M}\sum_{x,y}^{}\left(g_{c}\left(x,y\right)−\m _{c}\right)^{2}$

$\m _{c}=\frac{1}{N∙M}\sum_{x,y}^{}g_{c}(x,y)$

$k_{c}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\frac{g_{c}\left(x,y\right)−\m _{c}}{\sqrt{v_{c}}}\right)^{4}$

$s_{c}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\frac{g_{c}\left(x,y\right)−\m _{c}}{\sqrt{v_{c}}}\right)^{3}$

$cov_{mag,dir}=\frac{1}{N∙M}\sum_{x,y}^{}\left(g_{mag}\left(x,y\right)−\m _{mag}\right)−\left(g_{dir}\left(x,y\right)−\m _{dir}\right)  $

Laws’ Texture Energy (structural approach)

  - Laws texture masks calculate 9 values for each pixel in the image to capture different texture features. These masks are derived from 4 prototype vectors.

  - From these base vectors, we can generate 16 matrices by multiplying pairs of prototype vectors. For example, for E5L5, we obtain the kernel matrix $𝐆_{E5L5}$ as follows:

  - Since E5L5 and L5E5 measure similar aspects, we combine them into a single kernel and calculate the average of both matrices. By performing these reductions, we obtain 9 kernel matrices:

  - Using these 9 kernel matrices, we perform convolution to calculate 9 texture energy values, denoted as $e_{i}(x,y)$ for each pixel, where $1\leq i\leq 9$. From this point, we can employ the same methods as discussed earlier:

    - Histograms: While technically possible, we encounter the challenge of dealing with 9 values per pixel. If we quantize them into 4 ranges, we end up with $4^{9}$ = 262,144 reference energy values. Too many for a meaningful feature. Opting for just 2 ranges results in $2^{9}$ = 512 reference energies, which is more manageable, although it introduces a notable quantification error.

    - Moments: For every energy value, we can compute 4 moments and covariance values for the 36 potential pairs. This results in a 72-dimensional feature vector.

$𝒗_{L5}=\left[\begin{matrix}1&4&6&4&1\end{matrix}\right]$

$𝒗_{E5}=\left[\begin{matrix}−1&−2&0&2&1\end{matrix}\right]$

$𝒗_{S5}=\left[\begin{matrix}−1&0&2&0&−1\end{matrix}\right]$

$𝒗_{R5}=\left[\begin{matrix}1&−4&6&−4&1\end{matrix}\right]$

Level: (Gaussian) center-weighted local average

Edge: (gradient) responds to step edges

Spot: (Laplace of Gaussian) detects a spot

Ripple: (Gabor) detects ripples

$𝐆_{E5L5}=𝒗_{E5}^{⊤}𝒗_{L5}=\left[\begin{matrix}−1\\−2\\0\\2\\1\end{matrix}\right]\left[\begin{matrix}1&4&6&4&1\end{matrix}\right]=\left[\begin{matrix}−1&−4&−6&−4&−1\\−2&−8&−12&−8&−2\\0&0&0&0&0\\2&8&12&8&2\\1&4&6&4&1\end{matrix}\right]$

$𝔾=\left\{\frac{𝐆_{E5L5}+𝐆_{L5E5}}{2},\frac{𝐆_{L5R5}+𝐆_{R5L5}}{2},\frac{𝐆_{E5S5}+𝐆_{S5E5}}{2},\frac{𝐆_{S5L5}+𝐆_{L5S5}}{2},\frac{𝐆_{E5R5}+𝐆_{R5E5}}{2},\frac{𝐆_{S5R5}+𝐆_{R5S5}}{2} , 𝐆_{S5S5},𝐆_{R5R5,}𝐆_{E5E5}\right\}$

Gabor Moments (Fourier approach)

  - The 2D Fourier transform converts a grayscale image into frequency space, producing real and imaginary matrices. To visualize the result, we take the logarithm of the sum of the squares of the real and imaginary parts to reduce large energy differences. The 2D Fast Fourier Transform is a faster version that greatly lowers computational cost. However, it is limited to image dimensions that are powers of two.

  - In Fourier space, we apply Gabor filters to capture different frequency bands and directions. Each Gabor filter is applied to the Fourier-transformed image, which is a complex matrix. The filtered result is converted back to image space by an inverse Fourier transform, for example a fast iFFT. This filtered image shows the presence of the selected frequencies and directions in the original image. With 5 orientations and 3 scales, the filter bank has 15 Gabor filters, producing 15 filtered images. We compute statistical moments for each filtered image to obtain a varied set of texture descriptors. The following pages show the filter banks and their use in Fourier space.

image

imaginary

component

real

component

FFT

absolutevalue

log ofenergy of

frequencies

log

Gabor

Filter

filteredimage

iFFT

x

image

imaginary

component

real

component

FFT

  - The Gabor filter is defined as a Gaussian kernel multiplied by a complex sinusoid. Neurophysiological experiments have demonstrated that Gabor filters, when configured with appropriate parameters, exhibit behavior similar to receptive fields in the primary visual cortex. Here's the formal definition:

  - Before applying the Gaussian and sinusoidal components, the coordinates are rotated by an angle 𝜃. Using this definition and by adjusting the parameters, we can create different filters that are responsive to various frequencies and orientations. When we map the filter bank into the Fourier space, it takes on the layout as shown below on the right side. We can, as in the illustration, apply filters in 6 directions and at 3 different scales, resulting in 18 filters. This is illustrated with the examples on the next pages.

  - We can apply Gabor filters both in the spatial and the Fourier space, and we will exploit this fact to accelerate the calculations of texture features. In the spatial space, we obtain Kernels as described at the top and we can use convolution to compute the energy for each direction and scale.

$g_{\lambda ,\theta ,\phi ,\sigma ,\gamma }\left(x,y\right)=e^{−\frac{\overbar{x}^{2}+\gamma ^{2}\overbar{y}^{2}}{2\sigma ^{2}}}∙e^{i\left(2\pi \frac{\overbar{x}}{\lambda }+\phi \right)}$

Complex sinusoid with phase $\phi $ and wavelength $\lambda $. $1/\lambda $ is the frequency of the sinusoid.

$\overbar{x}=x\cos(\theta +)y\sin(\theta ) $

$\overbar{y}=−x\sin(\theta +)y\cos(\theta ) $

Gaussian kernel with standard deviation $\sigma $ and the spatial aspect ration $\gamma $

Fourier space

Center of Fourier space

A Gabor filter at $2\theta $ and high frequency ($=1\lambda $)

Spatial space

Example (1): Mapping image into Fourier space (FFT) and apply Gabor filter

direction 2

scale 3

direction 2

scale 2

direction 2

scale 1

direction 4

scale 1

Gabor-Filter

resulting image

FFT

x

x

x

x

This spike corresponds to the edge of the mast of the sail. The spike is orthogonal to the mast.

Mast of the sail creates a high contrast to the white of the wave.

Example (2): Mapping image into Fourier space (FFT) and apply Gabor filter

direction 2

scale 3

direction 2

scale 2

direction 2

scale 1

direction 4

scale 1

Gabor-Filter

resulting image

FFT

x

x

x

x

  - We can apply a Gabor filter in the Fourier domain. However, Fourier transforms can be computationally expensive. The FFT, as an efficient implementation, requires image height and width to be powers of two. Images that do not meet this requirement must be resized or padded carefully to avoid changing orientation or introducing artifacts.

  - To avoid these computational challenges, it is more practical to apply Gabor filters by convolving them with the image. This maps each pixel to an energy value determined by the filter's orientation and scale. Because Gabor filters produce complex numbers, we use their absolute values. Many image processing libraries, such as OpenCV and scikit-image, provide built-in Gabor kernels.

  - After obtaining the filtered images (as shown in the right-hand columns on the previous pages with the image examples), we can consolidate the results using standard methods like histograms or moments. Usually, we opt for 3-7 directions ($0\leq \theta \leq \pi $) and 2-5 scales (or frequencies; $1/\lambda $ typically measured in pixels, ranging from 0.05 to 0.5). When working with a large number of filters, moments are preferred to reduce dimensionality and circumvent the complexity of quadratic distance functions.

    - When using moments, we treat the absolute values in the filtered image as the raw data points and calculate mean, variance, skewness, and kurtosis on these values. In addition, we can calculate covariances between filters, adding substantially more features to the description. Let $\tilde{f_{i}}(x,y)$ represent the filtered (complex) image representation after applying the $i$-th Gabor filter. We get:

    - Example: choosing 5 directions and 3 scales leads to

      - 15 mean, 15 variance, 15 skewness, and 15 kurtosis values

      - 105 covariance values

      - Total: 165 values

$v_{i}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\left|\tilde{f}_{i}(x,y)\right|−\m _{i}\right)^{2}$

$\m _{i}=\frac{1}{N∙M}\sum_{x,y}^{}\left|\tilde{f}_{i}(x,y)\right|$

$k_{i}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\frac{\left|\tilde{f}_{i}(x,y)\right|−\m _{i}}{\sqrt{v_{i}}}\right)^{4}$

$s_{i}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\frac{\left|\tilde{f}_{i}(x,y)\right|−\m _{i}}{\sqrt{v_{i}}}\right)^{3}$

$cov_{i,j}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\left|\tilde{f}_{i}(x,y)\right|−\m _{i}\right)−\left(\left|\tilde{f}_{j}(x,y)\right|−\m _{j}\right)  $
