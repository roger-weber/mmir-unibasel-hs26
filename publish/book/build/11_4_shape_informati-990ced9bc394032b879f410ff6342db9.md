# Shape Information

Shape information is typically more important in object databases where objects are described by their shape, e.g., a 2D or 3D model. In images, we have different approaches to extract shape information:

  - Edge Detection: Edge detection algorithms like the Canny edge detector can locate the boundaries of objects in an image, which is a fundamental step in shape analysis.

  - Contour Detection: Contour detection algorithms, such as the OpenCV's findContours function, identify continuous curves that represent the shapes of objects within an image.

  - Region-based Segmentation: Techniques like region growing and region splitting can be used to group pixels with similar properties, which helps identify objects and their shapes.

Once shape information is available, we can use methods similar to those for texture analysis to characterize the contour in terms of edge directions or extracted shapes. Most techniques assume that we can describe the object's area or boundary in the image, allowing us to calculate the following shape descriptors:

  - Area: The percentage of pixels within the segment relative to the entire image.

  - Centroid: The average $x$ and $y$-values of the region (in the absence of mass values).

  - Axis of Least Inertia: This axis minimizes the squared distances to the region's boundary, useful for normalizing regions into a primary direction.

  - Eccentricity: The ratio of length to width of a bounding box in the principal direction.

  - Circularity Ratio: Indicates how closely the shape resembles a circle, often defined as the ratio of the area of the smallest containing circle to the region's area.

HOG (Histogram of Oriented Gradients) and SIFT (Scale-Invariant Feature Transform) are advanced computer vision techniques that capture and describe shape-related features in images. HOG quantifies gradient orientations in local image regions, making it suitable for object detection and recognizing shapes by analyzing the distribution of gradient directions. In contrast, SIFT identifies distinctive key points in an image, including corners and junctions, and describes local structures, including shapes. SIFT is valuable for shape recognition and matching due to its scale-invariance and robustness to variations in object size. Both methods are essential tools in computer vision for extracting and characterizing shape-related information, with HOG focusing on global shape analysis and object detection, while SIFT specializes in local shape descriptions and key point matching.

The Histogram of Oriented Gradients (HOG) technique was initially introduced in 1986 and experienced a resurgence in popularity when Dalal and Triggs utilized it in 2005 for pedestrian detection. Since then, it has been expanded and frequently serves as input data for neural networks.

  - Step 1 computes gradients, for example by applying Sobel operators to a grayscale image. HOG uses unsigned gradients, so directions range from 0 to $\pi $ (0 to 180 degrees). Angles between $\pi $ and $2\pi $ are reduced by subtracting $\pi $. Some HOG implementations let you choose signed or unsigned gradients, but Dalal and Triggs found the unsigned option worked best for pedestrian detection.

  - In Step 2 the image is split into cells, each covering an 8x8 pixel area. For each cell, HOG computes a histogram with 9 bins, a number found to work best for this case. The histogram is built from the gradient directions of the cell's 64 pixels, with each direction weighted by its gradient magnitude, as shown in the image below.

  - In Step 3, gradient magnitudes, which can vary with illumination, need to be normalized before comparing histograms. HOG uses a technique where it combines four neighboring cells and normalizes the concatenated histograms, resulting in a 36-bin histogram that sums up to 1. These four neighboring cells are organized in 2x2 cell groups, each covering an 8x8 pixel area, and they are shifted along the image at 8-pixel intervals. Each block produces a normalized histogram with 36 bins, and these blocks have partial overlaps.

  - In Step 4, you can either combine the histograms into global features or maintain a "bag" of local features for searching.

  - The HOG descriptors are used to train machine learning models, such as Support Vector Machines (SVMs) or neural networks, for object detection.

  - HOG features are widely used for object detection, human detection in surveillance, gesture recognition, face detection, biometric identification, and other tasks. Because they capture shape and structural information, they are an essential tool in many computer vision applications.
The Scale Invariant Feature Transform (SIFT) method robustly extracts features that can successfully match even when there are substantial changes in viewpoint. Take a look at the images in the upper right corner: SIFT is capable of matching key points, such as the mountain top, even in the presence of substantial alterations in scale, rotation, and viewpoint. The algorithms works in 4 steps:

  - Step 1: We generate an image pyramid using Gaussian filters at various standard deviations ($\sigma $) and scales, referred to as "octaves" in SIFT terminology (see right side for illustration). Each octave is downsampled to one-fourth of the previous one. Within each octave, the image is progressively blurred using Gaussian filters with increasing $\sigma $ values.

    - In every octave, we compute the difference of Gaussians (DOG) by subtracting neighboring images. These DOG images act as edge detectors for specific frequency bands.

    - The DOG image pyramid contains potential edges and points of interest, identified as the local minima and maxima within the DOG images. See illustration below.

Step 1


  - Step 2: We identify interesting points in the DOG pyramid using a one-pixel neighborhood. In the picture on the right, where "x" represents the current pixel, there are 8 neighbors in the same plane and 9 neighbors from the plane above and below. If the pixel is found to be a local minima or maxima within this neighborhood, it is marked accordingly. Otherwise, the pixel is discarded.

    - Starting with five Gaussian-blurred images in each octave, we create four DOG images, resulting in two extrema images at each octave. To reduce the number of keypoints, we eliminate all pixels with DOG values below a specified threshold, as these represent "flat" points. We also filter out edge points by considering their gradients; edges have a large gradient perpendicular to the edge and a small gradient along the edge. Our interest lies in corner points characterized by two substantial gradients.

  - Step 3: To create a rotation-invariant feature, we determine the principal orientation for each keypoint. SIFT builds a local histogram of gradient directions within the neighborhood of the keypoint. The size of this neighborhood window scales with the keypoint's scale. Each gradient direction is added to the histogram, weighted by its magnitude. The bin in the histogram with the highest value corresponds to the dominant direction. In the case of ties, all directions are considered.

    - SIFT uses the dominant orientation to normalize feature extraction, as shown in the next step. If it finds multiple orientations, SIFT creates features for each one. This normalization makes it easy to compare keypoints found from different viewpoints using a simple metric.

    - The primary direction of the keypoint may not align with its gradient direction.

  - Step 4: SIFT places a 4x4 grid oriented in the dominant direction around the keypoint, with the grid size scaling according to the keypoint's scale. Within each grid cell, a finer 4x4 mesh defines a neighborhood, and an 8-bin histogram captures gradient directions. For each point in this finer mesh, gradient orientation and magnitude are calculated, and the direction is added to the histogram with a Gaussian weight based on its distance from the keypoint. This process results in 8 values for each cell within the larger 4x4 grid, yielding a total of 128 feature values. See the bottom of the next page for an illustration of this step.

Step 2
  - SIFT features are invariant to scale, translation, and rotation. They are modeled on receptive fields in the primary visual cortex and capture local image patterns by orientation. SIFT descriptors are highly distinctive, and even small objects can produce many descriptors. Although they are complex, SIFT features can be computed nearly in real time. They are widely used for object recognition, motion detection, image alignment, and image stitching. Implementations exist in OpenCV, and scikit-image provides related methods such as daisy and harris. Like HOG, SIFT descriptors can be used as input to machine learning models.

Step 4





