# Color Information

Color perception is the eye's way of representing how energy is spread across electromagnetic wavelengths. It is only an approximation because reducing that distribution to three values loses much information, so two different spectra can look identical.

This approximation lets us reproduce human color perception using three additive components that emit wavelengths matching the sensitivity of the red, green, and blue cones. These components form the RGB family, which is designed for human vision and may not work for animals with different cone sensitivities. Besides RGB, many other color systems exist. The most common systems include:

  - CIE – Established by the International Commission on Illumination (CIE) to establish a link between the physical signal and the perception of a typical human observer.

  - RGB – The prevailing system since the introduction of sRGB by HP and Microsoft in 1996.

  - HSL/HSV – This converts Cartesian RGB coordinates to cylindrical coordinates for hue and saturation, while incorporating luminance/brightness as the third component.

  - YUV – Utilized in NTSC and PAL signals and serves as the foundation for numerous image and compression algorithms like JPEG and MPEG (using YCbCr), which are not discussed further.

  - CMYK – In printing, this method subtracts color from an initially white canvas. The ink absorbs light, and a blend of various inks generates the intended color, which is not covered in further detail.

Spectrum $f\left(\lambda \right)$ of the light of an observed point

When we observe the emitted or reflected light spectrum $f\left(\lambda \right)$, we obtain 3 (or 4) values for each cone type (and rod). To calculate intensity, we multiply the observed spectrum by the sensitivity filter of the cones (e.g., $c_{red}\left(\lambda \right)$), then integrate the result across all wavelengths. For example, for the red cone:

[MATH_ERROR]
source: https://en.wikipedia.org/wiki/Color_temperature

To extract perceptual color features and use distance measures such as Euclidean distance, we need a color representation that matches human perception. Consider the four colors in sRGB space shown below. Between adjacent boxes, only one channel changes and the color distance is 100 units. Even so, we perceive these shifts differently. The jump from green to yellow (first to second) looks large, while the jump from orange to red (third to fourth) looks much smaller. This happens because sRGB maps light nonlinearly.

(255,200,100)

(255,100,100)

(255,0,100)

(155,200,100)

100 unit change

100 unit change

100 unit change

Human vision uses three types of cones, so three components are needed to represent the full range of colors. Color can be divided into:

  - Brightness is the visual perception of emitted or reflected light. It depends on an object's luminance but is not directly proportional to it. Instead, the observer interprets it subjectively.

  - Chromaticity is the objective description of color without luminance. It has two independent components: hue and saturation. Chromaticity diagrams show the visible color range, called the color gamut (see right).

    - Hue indicates how closely a color matches pure colors like red, green, or blue. Hue values lie along the edge of the chromaticity diagram and are measured in degrees around the white point, for example D65.

    - Saturation / Chroma / Colorfulness quantifies how much light is spread across the visual spectrum. Pure or highly saturated colors are concentrated near a single wavelength with high intensity. In the chromaticity diagram, saturation is the distance from the white point relative to the maximum possible distance in that direction. Note that green lies much farther from white than red or blue.

D65
## 11.2.1 Color Spaces


source: https://en.wikipedia.org/wiki/CIE_1931_color_space

The CIE defined color spaces to faithfully represent how people see color. The mathematical relationships behind these spaces are important for advanced color management. In 1931 the CIE introduced the XYZ color space to model human color perception. Their experiments showed that people see green as brighter than red or blue when all have the same physical intensity. At night, rods take over from cones, producing mostly monochrome vision with finer sensitivity to brightness changes.

  - The definitions of $X$, $Y$, and $Z$ differ from the usual additive or subtractive primary color model.  $Y$ denotes luminance. $X$ and $Z$ indicate chromaticity without brightness. $Y$ corresponds to M-cone sensitivity (green), $Z$ to S-cone sensitivity (blue), and $X$ is a combination of cone responses.

  - To calculate $X$, $Y$, and $Z$ from spectral data, a standard colorimetric observer was defined by experiments. This observer represents the average chromatic response of a typical person within a 2 degree field in the fovea, the central region where most cones are concentrated. The color matching functions $\overbar{x}\left(\lambda \right), \overbar{y}\left(\lambda \right)$ and $\overbar{z}\left(\lambda \right)$ define the spectral weights for the measured spectral radiance or reflectance $f(\lambda )$. We compute $X$, $Y$, and $Z$ by integrating over wavelengths from 380 nm to 780 nm:

  - One key strength of CIE XYZ is its completeness. It covers every color visible to the human eye, making it useful for specifying and analyzing color. However, many colors in the CIE XYZ space fall outside the gamut of standard RGB spaces, because RGB systems cannot reproduce the entire visible spectrum.

  - CIE XYZ, widely used in colorimetry, color management, and image processing, forms the basis of color spaces such as CIE Lab, CIE Luv, and CIE LCH. These spaces are used for color communication and calibration in printing, photography, and display technology.

[MATH_ERROR]

[MATH_ERROR]

[MATH_ERROR]
source: https://en.wikipedia.org/wiki/CIE_1931_color_space

The CIE xyY space, introduced in 1931, was the first attempt to separate chromaticity from luminance. In the CIE XYZ system, the Y value was designed to represent the standard observer's perceived luminance. The $x$, $y$, and $z$ components are obtained by normalizing the XYZ values.

  - The resulting color space comprises $x$, $y$, and $Y$. The $x$ and y values define the chromaticity diagram, shown in the lower right section of the page, and represent color without luminance. CIE xyY is a widely used color specification that covers all visible colors for the standard observer. Note that the chromaticity diagrams here are shown in the sRGB space, so they do not display the full CIE xyY color gamut. To reverse the transformation from $x$, $y$, and $Y$ values, follow these steps:

$x=\frac{X}{X+Y+Z}$

$y=\frac{Y}{X+Y+Z}$

$z=\frac{Z}{X+Y+Z}=1−x−y$

$X=\frac{Y}{y}x$

$Z=\frac{Y}{y}(1−x−y)$

Chromaticity diagram in the CIE xyY color space. Please be aware that this representation is in sRGB, and colors beyond the sRGB triangle may not be accurately displayed.

  - The outer curve of the chromaticity diagram, called the spectral locus, corresponds to wavelengths measured in nanometers. The figure shows those wavelengths in micrometers. The CIE xyY space describes color as seen by the standard observer. It does not represent an object's inherent color because perceived color changes with the lighting and the light source's color temperature. In low light the human eye cannot see color and is limited to shades of gray.

  - This color space is versatile and essential for accurate, consistent color across different devices and applications. By including all colors seen by the standard observer, it links human perception to digital color systems and ensures precise, uniform color reproduction and interpretation.
source: https://en.wikipedia.org/wiki/CIELAB_color_space

The CIE xyY space covers the entire human visible color range, but it is not perceptually uniform. Two colors the same distance apart can look very different depending on where they are in the space. The CIE Lab color space (1976) was created to provide a more perceptually uniform scale. It also exceeds the gamut of many other color spaces and is device independent, so it is commonly used to map colors between different systems.

  - The $L$ component represents lightness and is based on luminance $Y$, adjusted for perception to produce a uniform scale so that a one-unit change corresponds to an equal perceived change in lightness. $L$ typically ranges from 0 to 100, with $L=0$ corresponding to black and $L=100$ corresponding to white.

  - The component $a$ represents the red-green opponent axis. Negative values indicate green and positive values indicate red. These values usually range from -128 to 127. When the component equals 0, it represents a neutral gray.

  - The $b$ component shows the blue/yellow opponent axis. Negative values mean blue and positive values mean yellow. Values normally range from -128 to 127. When $b=0$, the color is neutral gray.

  - The transformation from $X$, $Y$, and $Z$ components under D65 and $0\leq Y\leq 255$ is:

$f(t)=\left\{\begin{matrix}\sqrt[3]{t}                 &if t>\left(\frac{6}{29}\right)^{3} \\\frac{841∙t}{108}+\frac{4}{29}&otherwise      \end{matrix}\right.$

$L^{∗}=116∙f\left(\frac{Y}{Y_{n}}\right)−16$

$a^{∗}=500∙\left(f\left(\frac{X}{X_{n}}\right)−f\left(\frac{Y}{Y_{n}}\right)\right)$

$b^{∗}=200∙\left(f\left(\frac{X}{X_{n}}\right)−f\left(\frac{Z}{Z_{n}}\right)\right)$

$X_{n}=242.364495$

$Y_{n}=255.0$

$Z_{n}=277.67358$
source: https://en.wikipedia.org/wiki/SRGB

RGB is the main color model used in computing. HP and Microsoft introduced sRGB as an additive model for monitors, printers, and the Internet. It is formalized as IEC 61966-2-1:1999 and is the default when no color model is specified.

  - sRGB uses the ITU-R BT.709 primaries (also called Rec. 709) to define its color gamut (the range of colors it can produce). It became widely used because it matched the standard CRT monitors of its time. The primaries are:

$c_{sRGB}=\left\{\begin{array}{c}\begin{matrix}12.92∙c_{linear}&              if c_{linear}\leq 0.0031308\end{matrix}\\\begin{matrix}1.055∙c_{linear}^{\frac{1}{2.4}}−0.05&otherwise                  \end{matrix}\end{array}\right.$

$c_{linear}=\left\{\begin{array}{c}\begin{matrix}\frac{c_{sRGB}}{12.92}&                  if c_{sRGB}\leq 0.04045\end{matrix}\\\begin{matrix}\left(\frac{c_{sRGB}+0.055}{1.055}\right)^{2.4}&otherwise          \end{matrix}\end{array}\right.$

  - In sRGB, colors lie inside the triangle in the figure on the right when only nonnegative values are used. This color gamut does not cover all chromaticities and misses a large part of the green and blue range.

  - The sRGB scales are nonlinear, with an approximate gamma of 2.2. To convert from linear RGB to sRGB, the specification gives functions that map each channel value. Let $c_{sRGB}$ represent a channel value (red, green, or blue) in sRGB, and $c_{linear}$ denote the corresponding value in linear RGB. Both range from 0 to 1. For quantized values, divide by or multiply by ($2^{bits}−1$).
source: https://en.wikipedia.org/wiki/Rec._2020

  - The conversion from CIE XYZ to linear RGB is as follows:

    - Note that the transformation above maps linear RGB to XYZ. To obtain sRGB values, apply an additional transformation as described on the previous page.

    - Note that the RGB space does not cover the entire XYZ space, which represents the colors humans can see. If a mapping produces values outside the 0 to 1 range, set them to the nearest limit: 0 for negative values and 1 for values greater than or equal to 1.

$\left[\begin{matrix}r_{linear}\\g_{linear}\\b_{linear}\end{matrix}\right]=\left[\begin{matrix}3.240479&−1.537150&−0.498535\\−0.969256&1.875992&0.041556\\0.055648&−0.204043&1.057311\end{matrix}\right]\left[\begin{matrix}X\\Y\\Z\end{matrix}\right]$

$\left[\begin{matrix}X\\Y\\Z\end{matrix}\right]=\left[\begin{matrix}0.412453&0.357580&0.180423\\0.212671&0.715160&0.072169\\0.019334&0.119193&0.950227\end{matrix}\right]\left[\begin{matrix}r_{linear}\\g_{linear}\\b_{linear}\end{matrix}\right]$

  - RGB values are often quantized into integer ranges. This uses simple multiplication and division by $2^{bits}−1$. For example, true color normally uses 8 bits per channel, so the multiplier is 255. Deep color uses 16 bits per channel, so the multiplier is 65535. In some cases, quantization is based on a set of reference colors, or a color palette. Then a color is represented by its nearest match in the palette.

  - Apart from sRGB and linear RGB, several other RGB models have been introduced. You can create an RGB space simply by choosing the primaries and the white point. These alternative models extend the original sRGB, which covers a limited color gamut, to include a wider range of colors. For example, Rec. 2020 is designed for ultra high definition television (UHDTV) and has a much wider gamut than HDTV based on Rec. 709. Some RGB models even go beyond the standard chromaticity chart to include more of the green and blue regions.
source: https://en.wikipedia.org/wiki/HSL_and_HSV

Artists typically begin with a relatively bright color and then apply

  - white to "tint" the color,

  - black to "shade" the color, or

  - both white and black (gray) to "tone" the color.

  - To support these techniques in computer graphics, the HSL and HSV color models provide alternate representations of RGB that make color manipulation easier. Both models use hue and chroma to define chromaticity. HSL adds lightness and places fully saturated colors at $L = 1/2$. It allows tinting ($L = 1$) and shading ($L = 0$) without changing saturation. HSV uses value and places fully saturated colors at $V = 1$. It allows shading ($V = 0$) without changing saturation, while tinting does change saturation.

$M=max⁡(R, G, B)$

$m=min⁡(R, G, B)$

$C=M−m$

$𝐻 ′= 0 if 𝐶 =0 𝐺 − 𝐵 𝐶 mod 6 if 𝑀 = 𝑅 𝐵 − 𝑅 𝐶 +2 if 𝑀 = 𝐺 𝑅 − 𝐺 𝐶 +4 if 𝑀 = 𝐵$

$H=60°∙H′$

$S_{HSV}=\left\{\begin{matrix}0&if V=0   \\\frac{C}{V}&otherwise\end{matrix}\right.$

$L=\frac{1}{2}(M+m)$

$V=M$

$S_{HSL}=\left\{\begin{matrix}0&if L=1   \\\frac{C}{1−\left|2L−1\right|}&otherwise\end{matrix}\right.$

$H=60°∙H′$


## 11.2.2 Color Features


Color Histograms: Histograms are a simple way to show a color distribution using a fixed set of reference colors that serve as the dataset's vocabulary. Each pixel is assigned to the nearest reference color, and the counts of each reference color in the image are tallied. To make the result scale invariant, divide the counts by the total number of pixels; the normalized values then give the probability of each reference color appearing.

$22°$

$45°$

$70°$

$155°$

$186°$

$278°$

$330°$

  - Selection of reference colors:

    - The simplest method is to quantize the $R$, $G$, $B$ values in linear RGB space, as shown on the right. For example, using 2 bits gives 4 uniform levels per channel, producing $4^{3}=64$ reference colors, often written $c_{i}$ with $1\leq i\leq 64$. You can pick any number of uniform ranges, for example 5, to get the number of reference colors you want.

    - For better perceptual color matching, a nonuniform distribution is preferred. In HSV space, the color hexagon can be split into regions of perceptually similar colors, as shown on the right. To reflect greater sensitivity to brightness, use more bins along the $V$ dimension. With 7 chromaticity values and 9 $V$ bins, we obtain 63 reference colors, denoted as $c_{i}$.

    - For uniform color spaces like CIE Lab, use uniform ranges. Give the $L$ axis more ranges than the $a$ and $b$ axes to account for differences in brightness sensitivity.

    - We quantify the similarity between two reference colors $c_{i}$ and $c_{j}$ by measuring the distance between their representations, for example $d_{i,j}$. In Cartesian coordinates this is the Euclidean distance between the centers that represent the colors. In cylindrical coordinates, as in HSV, compute the angle difference as $min 𝛼 − 𝛽 ,2 𝜋 − 𝛼 − 𝛽$ and combine it with the other channel differences using a Manhattan distance. In all cases normalize each channel to a common range, for example $[0,1]$, before computing distances.
  - Comparison of histogram (distance measure)

    - Let $h_{i}$ and $g_{i}$ be the normalized histograms of two images, arranged by $N$ reference colors  $c_{i}$ with $0\leq h_{i},g_{i}\leq 1$. Although color quantization uses a three dimensional color space, the histograms are one dimensional because they count the reference colors. We also compute distances $d_{i,j}=d_{j,i}$ between pairs of reference colors $c_{i}$ and $c_{j}$.

    - An initial simple approach is to calculate the Manhattan or Euclidean distance between histograms:

      - These distance formulas work effectively, but they do not consider the similarity between reference colors. A slight shift in lighting or color representation can result in significant distance variations.

    - To address cross-correlation between reference colors, we must apply a quadratic distance measure and utilize a matrix $𝐀$ based on the distances between reference colors:

    - When a user submits a sketch as a query or specifies the desired colors in an image, histogram intersections (equivalent to a partial match query) are more suitable. Let $g_{i}\ne 0$ represent the user-selected colors, and $g_{i}=0$ denote colors without user input.

$\theta _{Manhattan}\left(𝒉,𝒈\right)=\sum_{i=1}^{N}\left|h_{i}−g_{i}\right|$

$\theta _{Euclidean}\left(𝒉,𝒈\right)=\sqrt{\sum_{i=1}^{N}\left(h_{i}−g_{i}\right)^{2}}$

$𝛿 𝑖𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡𝑖𝑜𝑛 𝒉 , 𝒈 = 𝑖 =1 𝑁 min h 𝑖 , 𝑔 𝑖 min 𝒉 , 𝒈$

$\theta _{quadratic}\left(𝒉,𝒈\right)=\left(𝒉−𝒈\right)^{⊤}𝐀(𝒉−𝒈)$

$𝐀: a_{i,j}=1−\frac{d_{i,j}}{\max_{k,l}d_{k,l}}$

Distance normalized by maximum distance for all pairs of reference colors

  - Variants:

    - A simpler method uses luminance or brightness histograms and ignores color. First compute brightness, for example the $L^{∗}$ channel from CIE Lab. Then divide the luminance values into several uniform ranges. The next steps follow the methods described above, including computing quadratic distances to measure similarity between brightness values. The resulting features describe the image's brightness and are often used for shot detection in videos because changes in lighting often mark shot boundaries.

    - We can quantize chromaticity only and ignore brightness or luminance. Suitable color spaces for this are CIE Lab, CIE LCH, HSL, and HSV. These features describe how color is distributed and are robust to changes in lighting, as long as lighting does not greatly change perceived chromaticity. They also help find dominant colors, so users can search for blue horses by chroma instead of specifying an exact RGB value.

  - Discussion:

    - Histograms are simple but usually produce good results. They are robust to translation, rotation, noise, and scaling, and in some cases to changes in lighting.

    - Ignoring spatial relationships between pixels can produce unexpected results. For example, the blue lake at the bottom of the picture might match the blue sky at the top and a blue car in the middle. Because color similarity does not consider pixel locations, it is easy to create two images with the same histogram but different content.

    - The histogram intersection method helps a retrieval system find the color of the main objects. Users select a color, and the search is extended with a histogram subquery that uses the intersection technique.

    - Color histograms often have very high dimensionality. 64 dimensions may be a minimum for effective retrieval, and they can exceed a 1000. Searching in such high dimensional spaces uses a lot of resources and is inefficient. Dimensionality reduction methods, such as principal component analysis (PCA), can reduce the number of dimensions and address correlations among reference colors.

    - Best results come from quadratic distance functions, which account for similarities between reference colors. To avoid costly vector-matrix-vector multiplications during search, we can use the eigenvectors of matrix A to transform histograms into a space where Euclidean distance applies.

Color Moments: Statistical moments offer another way to describe color distributions in the chosen color space. Any of the earlier color spaces can be used, but perceptually uniform spaces such as CIE Lab are preferred for measuring distances and similarities. We typically use Lab rather than LCH to avoid dealing with angular differences.

  - Single-channel moments calculate statistical values for each channel ($L, a, b$). If $c$ represents a color channel, $N$ and $M$ denotes the number of rows and columns in the image, the first four moments are defined as follows:

    - Mean $\m _{c}$ and variance $v_{c}$ indicate the position and width of the distribution peak. Skewness $s_{c}$ reveals if the peak is skewed left or right. Kurtosis $k_{c}$ identifies the presence of outliers (values far from the mean). When considering three channels, this method yields 12 feature values describing the distribution in each channel.

  - We can calculate additional covariance values between pairs of channels. If we have three channels, we get three extra covariance values from the possible channel pairs:

  - We now can define a color moment as follows: Using the CIE Lab color space, we extract 12 moments and 3 covariances, for a total of 15 values. We arrange these values into a vector $𝒎$ (in a fixed order) and compare it to the feature vectors $𝒎_{i}$  and $𝒎_{j}$ from two images using either Euclidean or Manhattan distance.

$cov_{c_{1},c_{2}}=\frac{1}{N∙M}\sum_{x,y}^{}\left(c_{1}\left(x,y\right)−\m _{c_{1}}\right)∙\left(c_{2}\left(x,y\right)−\m _{c_{2}}\right)  $

$v_{c}=\frac{1}{N∙M}\sum_{x,y}^{}\left(c\left(x,y\right)−\m _{c}\right)^{2}$

$\m _{c}=\frac{1}{N∙M}\sum_{x,y}^{}c(x,y)$

$k_{c}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\frac{c\left(x,y\right)−\m _{c}}{\sqrt{v_{c}}}\right)^{4}$

$s_{c}=\frac{1}{N∙M}\sum_{x,y}^{}\left(\frac{c\left(x,y\right)−\m _{c}}{\sqrt{v_{c}}}\right)^{3}$

$\theta _{Manhattan}\left(𝒎_{i},𝒎_{j}\right)=\sum_{k=1}^{15}\left|𝒎_{i,k}−𝒎_{j,k}\right|$

$\theta _{Euclidean}\left(𝒎_{i},𝒎_{j}\right)=\sqrt{\sum_{k=1}^{15}\left(𝒎_{i,k}−𝒎_{j,k}\right)^{2}}$

  - When calculating moments, the formulas can be rewritten so one pass computes all values (where 𝑐 represents a color channel):

  - Variants:

    - Like histograms, we can compute moments for brightness (luminance) only. This removes the need for covariance and gives four brightness moments. Alternatively, we can compute moments for chromaticity only, ignoring brightness. That case yields eight moments plus one covariance value, forming a nine-dimensional feature.

  - Discussion:

    - The value ranges of moments can vary significantly. To apply a distance measure, we must first normalize the values into the same range. Scaling can be achieved by dividing with the standard deviation of the values along that dimension as outlined in the chapter on Vector Search.

    - Color moments, like histograms, are robust to translation, rotation, noise, and scaling. They can also tolerate some changes in lighting. However, like histograms, they ignore spatial relationships between pixels, which can lead to unexpected results.

    - Color moments are independent and do not need a cross-correlation matrix to compute quadratic distances. Their vectors are much shorter, typically about 15 values when all moments are used, while histograms can have up to 1000 bins. This compact representation speeds up searches without reducing retrieval quality.

$1\leq n\leq 4$

$a_{c,n}=\frac{1}{N∙M}\sum_{x,y}^{}c\left(x,y\right)^{n}$

$\m _{c}=a_{c,1}$

$v_{c}=a_{c,2}−a_{c,1}^{2}$

$s_{c}=\frac{a_{c,3}−3a_{c,2}∙a_{c,1}+2a_{c,1}^{3}}{v_{c}^{3/2}}$

$k_{c}=\frac{a_{c,4}−4a_{c,3}∙a_{c,1}+6a_{c,2}∙a_{c,1}^{2}−3a_{c,1}^{4} }{v_{c}^{2}}$

$cov_{c_{i},c_{j}}=b_{i,j}−\m _{c_{i}}∙\m _{c_{j}}$

$b_{i,j}=\frac{1}{N∙M}\sum_{x,y}^{}\left(c_{i}\left(x,y\right)∙c_{j}\left(x,y\right)\right)$

$1\leq i,j\leq 3$
