# Perceptual Features

Perceptual features try to match how listeners actually hear sound. In the time domain, one looks at energy fluctuations and envelope shapes to capture sensations of loudness and dynamics. In the frequency domain, one tracks brightness with measures such as spectral centroid, the distribution of energy across bands, and the amount of spectral change over time. Psychoacoustic scales like Mel convert frequency into a form that better reflects human perception. After computing these descriptors frame by frame, they are summarized at the segment level using statistics such as means, variances, and temporal trends. This stabilizes the descriptors and makes them closer to the perceptual units humans tend to recognize.

The first set of features describes audio files from an acoustic perspective across different domains.

  - Time Domain refers to analyzing the raw signal in terms of time and amplitude.

  - Frequency Domain transforms raw signal with STFT and analyzes frequencies and their energies at the given time point (see window technique)

  - Perceptual Domain is modelling the perceptual interpretation of the human ear

After this chapter we will study musical features and fingerprinting techniques that often rely on these perceptual features:

  - Musical features describe the structure listeners hear in melody, rhythm, and harmony. Pitch tracking follows the fundamental frequency and its changes, whether from a single voice or mixed with other sounds. Tempo estimation finds the periodic pattern that sets the pace of a piece, while beat tracking marks the regular points that guide musical flow. Depending on the application, one may also extract higher level information such as key, chord sequences, or the typical timbres of instruments. These descriptors let systems treat sound not just as energy patterns but as organized artistic material.

  - Fingerprinting uses a different method. Instead of describing how sound is heard or arranged musically, it creates a small code that uniquely identifies a recording. A common approach is to find clear features in the sound spectrum and encode the relationships between them. These codes must resist background noise, compression artifacts, and playback distortions so that short or imperfect queries can still be matched reliably to a large database. Fingerprinting is the core technology behind many commercial music recognition systems.


## 12.2.1 Features in the Time Domain


In the time domain we use the raw amplitude signal from a single frame (see segmentation). For example, with a sampling rate of 48 kHz and a frame length of 40 ms, the frame contains 1920 samples, and the hop between consecutive frames is 20 ms.

  - Short-Time Energy (STE) measures the total energy in a signal by adding up the squares of the values and then normalizing it by the length of the frame. In audio signals, power is typically measured in decibels, which is one-tenth of a bel, a unit used in the first telephone system. An increase of 10 dB represents a tenfold increase in power. The metric is logarithmic: $L_{P}=10log_{10}\left(P/P_{0}\right)$. With that, STE for an amplitude signal $x\left(t\right)$ within a frame $F_{i}$ (with samples from $1\leq t\leq N$) is defined as:

  - Zero-Crossing Rate (ZCR) measures how frequently the amplitude signal changes from positive to negative values within a frame.

  - Entropy of Energy (EoE) measures sudden changes in the energy of the audio signal within a frame $F_{i}$. To do this, the frame is divided into smaller sub-frames of the same length that cover the entire frame. For each sub-frame, the energy is measured and then adjusted to the total energy of the frame to get a sequence of values that add up to 1. The entropy of these values is the Entropy of Energy. Choose $L$ and $N_{sub}$ such that $N=L∙N_{sub}$:

$E_{STE}(i)=10log_{10}\left(\frac{1}{N}\sum_{t=1}^{N}x\left(t\right)^{2}\right)$

$ZCR(i)=\frac{1}{2N}\sum_{t=2}^{N}\left|sgn\left(x(t)\right)−sgn\left(x(t−1)\right)\right|$

$H_{EoE}\left(i\right)=−\sum_{l=1}^{L}e\left(i,l\right)∙log_{2}e(i,l)$

$e\left(i,l\right)=\frac{\sum_{t=l∙N_{sub}}^{\left(l+1\right)∙N_{sub}−1}x\left(t\right)^{2}}{\sum_{t=1}^{N}x\left(t\right)^{2}}$

Segment-level features in the time domain: These features provide statistics for a segment $S_{j}$ with $M$ frames. For example, if the segment is 4 seconds long, with frames of 40 milliseconds and a frame hop distance of 20 milliseconds, there will be 199/200 frames, depending on how the last frame is treated as it is partially inside and partially outside the segment.

  - Histograms are created by dividing the range of a feature's values (STE, ZCR, EOE) into sections and counting how many values fall into each section across the frames of a segment. The resulting numbers are normalized to create a histogram. This method is not commonly used because it produces larger features than moments.

  - Calculate the moments for STE, ZCR and EOE values within the segment $S_{j}$. These features describe the distribution of values within the segment.

  - Low Short-Time Energy Ratio (LSTER) measures the percentage of frames in the segment $S_{j}$ where the Short-Time Energy (STE) is less than one-third of the average STE across the segment. This is important because speech signals often have pauses between syllables, leading to greater variation.

  - The High Zero-Crossing Rate Ratio (HZCRR) measures the number of zero-crossings and counts how many  frames have much higher zero-crossings than on average within the segment:

The last two features are useful for distinguishing between speech and music signals. Speech has many frames with low energy, while music has consistently high energy throughout the segment. Additionally, speech has more frames with high zero-crossings because many frames have none due to energy pauses. On the other hand, music has a consistent rate of zero-crossings across frames, with only a few frames showing significantly more zero-crossings compared to the average frame in the segment.

$r_{LSTER}(j)=\frac{1}{M}\sum_{i=1}^{M}\left\{\begin{matrix}1&E_{STE}\left(i\right)<\frac{\m _{STE}(j)}{3}\\0&otherwise               \end{matrix}\right.$

$\m _{STE}(j)=\frac{1}{M}\sum_{i=1}^{M}E_{STE}(i)$

$r_{HZCRR}(j)=\frac{1}{M}\sum_{i=1}^{M}\left\{\begin{matrix}1&ZCR\left(i\right)\geq \frac{3\m _{ZCR}(j)}{2}\\0&otherwise               \end{matrix}\right.$

$\m _{ZCR}(j)=\frac{1}{M}\sum_{i=1}^{M}ZCR(i)$


## 12.2.2 Features in Frequency Domain


We first analyze the Fourier transformed signal in the frequency domain using a single frame $F_{i}$ (STFT). For example, with a sampling rate of $f_{s}=48$ kHz and a frame size of 40ms, there are $N=1920$ samples, and the hop distance between subsequent frames is 20ms. The Fourier transformed values$ X\left(i,\omega \right)$ denotes the frequency spectrum of frame $F_{i}$ with $0\leq \omega \leq f_{s}/2$ and with steps $∆\omega =f_{s}/N=25$ Hz. It's important to note that in the discrete notation of the Fourier transformed, only the first half of the values are needed as the second half is symmetrical ($X(i,k)$ with $0\leq k<N/2$). In the following, we often use the discrete form $X\left(i,k\right)=X(i,\omega (k))$ with $\omega \left(k\right)=k∙f_{s}/N$.

  - The Spectral Centroid (SC) is the center of gravity of the spectrum. It's the weighted average frequency in the spectrum of the frame, with the energy (magnitude) as weights. The energy is the absolute values of the complex value of $X(i,k)$. For convenience, let $K=N/2−1$. Therefore:

    - The centroid describes the “sharpness” of the signal is in the frame, a weighted average of frequency and energy.

  - Spectral Roll-off (SR, $\omega _{r}$) shows the frequency at which the total sum of energy with frequencies smaller than $\omega _{r}$ is $C=85$% of the overall sum of energy. We are looking for a value $0\leq r\leq K$.

    - Related to the spectral centroid, it measures how skewed the spectrum is towards higher frequencies which are dominant in speech.

  - Band-Level Energy (BLE) is the total energy within a specific frequency range. This range is determined using a weighting function in the Fourier domain with $0\leq k\leq K$. The feature value is measured in decibels to match how we perceive sound.

$SC\left(i\right)=\frac{\sum_{k=0}^{K}\omega \left(k\right)∙\left|X\left(i,k\right)\right|}{\sum_{k=0}^{K}\left|X\left(i,k\right)\right|}$

$\omega _{r}=\omega \left(r\right)$   with $r$ smallest value that fulfills:    $\sum_{k=0}^{r}\left|X\left(i,k\right)\right|\leq C∙\sum_{k=0}^{K}\left|X\left(i,k\right)\right|$

$BLE\left(i\right)=10log_{10}\left(\sum_{k=0}^{K}\left|X\left(i,k\right)\right|^{2}∙w\left(k\right)\right)$

  - Spectral Flux (SF) measures the squared differences in normalized magnitudes from the previous frame. It shows how the spectral content is changing locally. A high value means there is a sudden change in magnitudes, which can lead to a significant change in perception (only for $i>1$).

  - Spectral Bandwidth (SB) measures the distance of frequencies from the spectral centroid. It shows how much the frequencies deviate from the spectral centroid

Segment-level features in the frequency domain: to summarize a segment, we can use moments and histograms  over the frame values for the different features mentioned.

  - Histograms are created by dividing the range of a feature's values into sections and counting how many values fall into each section across the frames of a segment. The resulting numbers are then adjusted to create a histogram of the feature values. This method is not commonly used because it produces larger features than moments.

  - Moments can describe the distribution for the feature values within the segment $S_{j}$ with mean value, standard deviation, skewness, and kurtosis. For band level energy, we can use different weight functions and also calculate the covariance between different bands.

$SF\left(i\right)=\sum_{k=0}^{K}\left(\frac{\left|X\left(i,k\right)\right|}{\sum_{k=0}^{K}\left|X\left(i,k\right)\right|}−\frac{\left|X\left(i−1,k\right)\right|}{\sum_{k=0}^{K}\left|X\left(i−1,k\right)\right|}\right)^{2}$

$SB\left(i\right)=\sqrt{\frac{\sum_{k=0}^{K}\left|X\left(i,k\right)\right|∙\left(\omega \left(k\right)−SC\left(i\right)\right)^{2}}{\sum_{k=0}^{K}\left|X\left(i,k\right)\right|}}$


## 12.2.3 Psychoperceptual Features


Human hearing and the way people perceive sound differ from simple physical measurements. For example, loudness reflects the energy in a sound wave, but people hear frequencies differently, especially between 2 and 5 kHz, which are important for understanding speech. The following features are designed to match the perception of a normal human ear:

  - Loudness is how the sound pressure of the wave is perceived by a “normal” person. Consider the top figure on the right side. The red curves show how much energy is needed for an average listener to hear a pure tone as equally loud as the frequencies increase from left to right. The needed energy drops a lot between 2 and 5 kHz because the outer ear is amplifying signals in this frequency range.

  - The international standard IEC 61672:2003 has different weighting functions to model this perception, as shown in the bottom right figure. The A-weighting curve (blue) is used the most, even though it's only good for quiet sounds.

  - The human perception of sound averages loudness over a 600-1000ms time period. So, the loudness for the frame $F_{i}$ is the average over the previous 1000ms of the sound, not just the average of values in the current frame. Let 𝑂 be the number of frames over the last 1000ms. For instance, with a hop size of 20ms, 𝑂=50. Loudness is measured in decibels to approximately match the logarithmic perception of a person:

$L\left(i\right)=\frac{10}{O}\sum_{o=0}^{O−1}log_{10}\left(\frac{1}{K}\sum_{k=1}^{K}A\left(k\right)∙\left|X\left(i−o,k\right)\right|^{2}\right)$

  - Mel Frequency Cepstral Coefficients (MFCC) show the energy spectrum over Mel frequency bands, which are similar to the human auditory system. The method involves 4 steps:

    - Fourier Transform: Calculate the Fourier transform over the frame $F_{i}$. Unlike the STFT, we do not use a windowing function. Let $N$ be the number of samples in the frame $F_{i}$ and $f_{s}$ be the sampling rate (for example, $N=1920, f_{s}=48$ kHz).

    - Mel-Frequency Spectrum: We calculate the spectrum using Mel frequency bands. Usually, there are $B=26$ bands, and we use $f_{lower}=300 Hz$ and $f_{upper}=8000 Hz$ to represent the lowest and highest frequencies. The conversion between frequencies and mels is as follows:

      - The bands are triangular window functions in the frequency space. Three frequencies mark the beginning, middle, and end points. Two bands overlap: the start point of one band is the middle point of the previous band. The frequencies are calculated in the Mel space to align with human perception. Given $B$ bands, we need $B+2$ frequencies given by ($0\leq b\leq B+1$ )

$freq\left(m\right)=700∙\left(e^{\frac{m}{1125}}−1\right)$

$𝑚𝑒𝑙 𝑓 =1125∙ ln 1+ 𝑓 700$

$f_{c}\left(b\right)=freq\left(mel\left(f_{lower}\right)+b∙\frac{mel\left(f_{upper}\right)−mel\left(f_{lower}\right)}{B+1}\right)$

$X\left(t,k\right)=\frac{1}{N}\sum_{j=0}^{N−1}x\left(j\right)∙e^{−i2\pi \frac{jk}{N}}$

$\omega \left(k\right)=k∙\frac{f_{s}}{N}$
    - With the frequencies $f_{c}\left(b\right)$, we can define the windowing function $d(b,k)$ over the Fourier coefficients $X\left(t,k\right)$ for a given time point $t$. The shape has a triangle form:

    - This finally allows us to calculate the Mel-frequency spectrum by adding up the magnitude values of the Fourier coefficients for each of the bands. This results in $B$ values $M(t,b)$ for $1\leq b\leq B$:

    - Cepstral Coefficients: The cepstrum is like a spectrum of a spectrum. The updated version of MFCC calculates the coefficients using a discrete cosine transformation and uses the first half of the coefficients.  If we started with $B=26$, we obtain 13 cepstral values $c(t,b)$ with $1\leq b\leq B/2$:

$d\left(b,k\right)=\left\{\begin{matrix}0&if  \omega \left(k\right)<f_{c}\left(b−1\right)               \\\frac{\omega \left(k\right)−f_{c}\left(b−1\right)}{f_{c}\left(b\right)−f_{c}\left(b−1\right)}&if  f_{c}\left(b−1\right)\leq \omega \left(k\right)\leq f_{c}(b)\\\frac{\omega \left(k\right)−f_{c}\left(b+1\right)}{f_{c}\left(b\right)−f_{c}\left(b+1\right)}&if  f_{c}\left(b\right)\leq \omega \left(k\right)\leq f_{c}(b+1)\\0&if  w\left(k\right)\geq f_{c}\left(b+1\right)               \end{matrix}\right.$

$M\left(t,b\right)=\sum_{k=0}^{N/2−1}d\left(b,k\right)∙\left|X(t,k)\right|$

$c\left(t,b\right)=\sum_{j=1}^{B}M\left(t,j\right)∙\cos(\left(\frac{b\left(2j−1\right)\pi }{2B}\right))$

with $1\leq b\leq B/2$

    - Derivatives: The MFCC features are a mix of the cepstral $c\left(t,b\right)$ values and their first and second order derivatives. These derivatives show how the bands changes over time. By using 13 cepstral coefficients, we get 39 feature values.

Segment-level features in the psychoperceptual domain: We can calculate the moments or histograms of the perceptual features in a segment, just like we did before. For example, the standard deviation of the second MFCC coefficient $c(t,2)$ is very useful for telling the difference between speech and music.

  - MFCCs are commonly used in speech recognition. These feature values are employed in Hidden Markov Models or neural networks to recognize phonemes. Typically, cepstral coefficients from a large spoken text are clustered using a k-means clustering approach. These clusters are then used to quantize the vector and create states. The machine learning method then maps a series of state transitions to a phoneme. The phoneme stream is further processed to create words.

$∆c\left(t,b\right)=c\left(t+1,b\right)−c\left(t−1,b\right)$

$∆∆c\left(t,b\right)=∆c\left(t+1,b\right)−∆c\left(t−1,b\right)$

$MFCC(t)=[c\left(t,1\right),…,c\left(t,B/2\right),∆c\left(t,1\right),…,∆c\left(t,B/2\right),∆∆c\left(t,1\right),…,∆∆c\left(t,B/2\right)]$
