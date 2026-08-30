# Shot Detection

A shot is a sequence of frames continuously recorded by a camera. A shot boundary can change the viewpoint in a scene or signal a new scene that shifts the time or place. Often there is a sudden change in the image, depending on the type of transition. The audio track often continues without a noticeable change, but it can also change abruptly at the shot boundary.

  - Hard cuts: There is no visual overlap between consecutive shots, and the last frame of one shot is clearly distinct from the first frame of the next. A hard cut happens when the image changes suddenly.

  - Soft cuts: The two shots blend together, switching from one to the other over several frames. Visual effects such as fades and swipes show the change between them. Unlike hard cuts, there is not a single frame that marks the end or the start. Instead, a series of frames overlap to create the transition.

  - Hard cuts are commonly used to switch the camera angle within a scene, such as when two people are talking and the viewpoint shifts from one person to the other. Soft cuts are used to signal the end of a scene and to guide the viewer's attention to a change in time or location.

You often can tell where one shot ends and another begins by looking at the video encoding. When the difference between frames is too big, the encoder uses an I-frame. These frames also help with fast navigation in the video and show up a lot. They're not great for finding shots, but they do make things a bit easier.

Shot Detection (hard cuts)

  - A hard cut is a sudden change in the video frames. To detect one, we compare two consecutive frames and measure how different they are. We can use a change threshold: if the distance between frames is above that threshold, we mark it as a shot boundary.

    - Pixel based comparison: One simple way to analyze the changes in an image over time is to calculate the difference in color for each pixel between two consecutive frames. This can be done by comparing the values of the red, green, and blue channels for each pixel in the two frames. Let $𝒇(x,y,t)$ be a vector function returning red, green, blue values for the pixel at position $(x,y)$ and at time $t$:

$d_{naive}\left(t\right)=\sum_{x,y}^{}\left|𝒇\left(x,y,t\right)−𝒇\left(x,y,t−1\right)\right|$

Shot Detection (hard cuts)

    - Histogram / Moments Comparison: to account for small changes in translation, rotation, and size, it's best to use moment and histogram features. We usually only look at the brightness values because shots typically have distinct brightness distributions that abruptly change from one shot to the next. The usual method is based on histograms of brightness values. Let's use $𝒉(t)$ to represent the histogram (or feature vector) for a frame at time $t$. Then we can get a more accurate distance measure with:

  - Previously, we introduced the ROC curve to find hyper parameters for classifiers. Let's use $f_{n}(x)$ to show the distribution of distances between frames in the same shot, and $f_{p}(x)$ to show the distribution of distances between frames in different shots. We define a threshold  as:

    - $d\left(t\right)<T$ indicates that the frame $t$ is part of the same shot (no shot change; negative scenario)

    - $d\left(t\right)\geq T$ denotes that frame $t$ belongs to a new shot (shot boundary; positive case)

    - We can calculate the rates of false positives, true positives, false negatives, and true negatives. The best threshold depends on our objective function, but usually we would select one that maximizes accuracy. The ROC table is a simple tool to compute the threshold $T$, as discussed in a separate chapter of this class.

$d_{M}\left(t\right)=\left|𝒉(t)−𝒉\left(t−1\right)\right|$

$d_{H}\left(t\right)=𝒉\left(t\right)^{⊤}⋅𝐀⋅𝒉(t−1)$

FPR

TNR

TPR

FNR
Shot Detection (soft cuts)

  - The method discussed before is effective for identifying abrupt cuts, but it has difficulties detecting slow visual transitions between shots. For example, a gradual fade-out and fade-in doesn't cause enough change in the brightness histograms to exceed the threshold. However, after some time, the image has changed noticeably.

  - One option is to model various transition effects and to recognize them in the video stream. A fade-out, fade-in is easy to see (the screen turns all black). Swipes (side to side or up and down) divide the image into two parts (one from the old shot, one for the new shot) and slowly change the ratio between them. However, it requires a lot of coding/learning to model all possible visual effects, and new effects cannot be identified.

  - Twin Thresholding is a method for finding slow, visual changes from one frame to another. It uses two thresholds: $T_{c}$ to detect hard cuts as discussed previously; and $T_{s}$, a lower and more sensitive threshold, to identify the potential start of a soft cut. When visual changes between subsequent frames exceed $T_{s}$, the current frame becomes the reference image, and we keep using it as the reference until either

      - the difference to the reference frame exceeds the hard cut threshold $T_{c}$, or

      - the difference falls below the threshold $T_{s}$ again.

    - In both cases, we release the reference frame and continue with the search for the next cut.

  - See the next page for an example of a fade-out/fade-in effect with the changes of the brightness histograms.

distances

time

$T_{c}$

$T_{s}$

a hard cut

potential start of a soft cut

no cut after all

a soft cut

potential start of a soft cut
