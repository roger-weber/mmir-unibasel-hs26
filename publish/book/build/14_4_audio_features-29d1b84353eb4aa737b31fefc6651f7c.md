# Audio Features

Audio analysis starts by asking how to turn a continuous stream of sound into information a machine can search, label, or interpret. The first step is to extract features that capture the signal's key qualities. These features reduce the raw waveform to measurements that match what listeners perceive, such as loudness, pitch, and timbre. Once extracted, they form the basis for organizing and understanding the audio.

A common first task is classifying an audio stream into broad categories such as noise, speech, or music. The system starts with perceptual features such as the spectral centroid, short term energy, and the zero crossing rate. These features are calculated in short time windows so the system can follow how the sound changes from moment to moment. Once the features have been extracted, a simple method such as a decision tree can be used to assign each segment to a class. A decision tree learns a sequence of tests that link feature values to labels. The structure of the tree allows the classification process to proceed in small, interpretable steps, and it can yield accurate predictions without requiring heavy computation.

Example: Automated Sport Summary (e.g., football match)

  - When audio features are used in sports video analysis, the same classification principles apply to event detection. The audio signal holds many clues about what is happening on the field, even when the visuals are complex. By tracking simple perceptual measures over time, a system can spot moments that matter to viewers and editors.

  - Imagine a recording of a football match. During ordinary play the crowd produces a steady, moderate energy level and the sound's frequency makeup stays relatively stable. When an exciting play happens the audio pattern changes sharply and clearly: energy rises quickly, the sound shifts toward higher frequencies as cheering becomes brighter, and the timing becomes more erratic. A decision tree that has learned to link these changes with strong audience reaction can flag the segment as a moment of interest.

  - Commentary gives a second way to spot important events. Commentators change their voice as the action speeds up. Their pitch becomes livelier, their speech rate rises, and their phrasing tightens during key plays. These changes show up in measures like pitch stability, formant movement, and short-term energy. A decision tree can combine these cues with crowd features to improve its estimate.

  - After the system finds these promising segments, a transcription model adds more structure. Speech recognition methods, from the state based Hidden Markov Model to the learned encodings of Whisper, let the system extract the commentator's words. Phrases such as goal, interception, or last minute attempt can confirm the moment's importance. They also let the system label the event type, creating a richer index for later retrieval.

Audio classification with decision trees:

  - Decision trees are straightforward and create effective classifiers that work well for many tasks. For instance, they can be used to classify audio signals as either speech or music.

  - During the learning phase, we have to prepare the audio signal, pull out features, collect statistical data about the features and how they relate to output categories (music, speech), and choose the top features for classification. In this case, we use XGBoost or C4.5 to pick features and create rules.

Example with a combined feature extraction and statistical computation [Castan, 2010]

AudioSignal

Framinglength=40ms

hop=20ms

Decision Tree Learning

RuleSet

features

targets

HZCRR

LSTER

AM Ratio

FFT

MFCC

VSF

$C_{0}$

$C_{1},…,C_{12}$

MET

VAR

Features (6 dimensions)

Feature Extraction

Segmentationlength=4s

hop=100ms

Statistical Computation

  - Framing and segmentation involve processing the audio signal in overlapping frames and segments. Each frame and segment has the same length, and the hop distance determines when the next frame/segment begins. Usually, features are extracted for each frame, and statistical measures are applied to the segment across its frame.

  - Castan (2010) focused on a small number of characteristic features:

    - HZCRR: The Zero-Crossing Rates (ZCR) measures how frequently the signal's amplitude passes the 0-value within a frame. The High Zero-Crossing Rate Ratio (HZCRR) measures the percentage of ZCR values in a segment that are 1.5 times higher than the average ZCR value of frames in the segment.

    - LSTER: The Short Time Energy (STE) is just the total of the squared amplitude of the signal within the frame (a measure of energy in the frame). The Low Short Time Energy Ratio measures the percentage of STE values of frames in the segment that are smaller than 50% of the average STE value of frames in the segment.

    - AMR: The Amplitude Modulation Ratio (AMR) calculates the low-pass energy of a frame by adding up the squared amplitude after using a low-pass filter with a cut-off at 25Hz. It then compares the highest energy to the lowest energy across all frames in the segment. Speech has a higher ratio than music because of the pauses between vowels and consonants.

    - VSF: The Spectral Flux (SF) is the distance between frames in their Fourier transformed signals (spectrum magnitudes). The Variation of Spectral Flux (VSF) measures the variance within the frames in the segment.

    - MET & VAR: We calculate 13 Mel-Frequency Cepstrum Coefficients (MFCC) for each frame, labeled $C_{0},…,C_{12}$. The Minimum-Energy Tracking (MET) measures how long $C_{0}$ is above a certain level. Short pauses in speech will lead to short frame lengths. VAR adds up the variance of all MFCC across the frames in the segment. Low VAR values suggest music.

  - During the prediction phase, we have to do the same pre-processing, windowing, feature extraction, and statistical calculations as in the learning phase. We also want to smooth out the results for the whole song (using a voting-based approach) or divide a continuous audio signal (like a radio broadcast) to find when the speech changes to music.

  - Smoothing involves adding up past predictions with decreasing weights to prevent rapid changes between targets. When there is sufficient evidence for a change, segmentation ends the current segment (different from the segments used for feature extraction) and assigns it the last class label. Then it starts a new segment.

  - Voting is easy. The file is classified based on the label that is predicted most often for its parts. Or, the classification can show the likelihood of different labels based on how often they are predicted for the file’s parts.

AudioSignal

Framinglength=40ms

hop=20ms

Voting

RuleSet

features

Feature Extraction

Segmentationlength=4s

hop=100ms

Statistical Computation

Smoothing

Segmentation & Classification

Classification

predictions

continuous stream

single file


## 14.4.1 Transcription


We won't delve too deeply into transcription. Instead, we'll provide a brief overview of the techniques and discuss retrieval aspects.

  - First, the audio signal must be pre-processed to remove any noise. This means that the pitch, tempo, and loudness of the speaker should not affect the result. The typical approach is to use the MFCC method discussed earlier.

  - The MFCC analysis produces a series of $p$-dimensional vectors. These vectors form the foundation for learning phonemes, which are the building blocks of words and texts. There are two methods for learning phonemes:

    - Use a Hidden Markov Model (HMM) to represent the phonemes using quantized vector data and model the temporal transitions. We can use a $k$-means algorithm to divide the $p$-dimensional vectors into a set of $k$ states.

    - Create a neural network to learn phonemes. This traditionally involved using recurrent networks, which can maintain a current state and pass it on to the next iteration of the network run. Modern approaches use transformer architectures with attention to perform transcription.

After identifying phonemes, we must then identify the words. Spoken text does not separate words with spaces, but instead comes as a continuous stream of phonemes. Recognizing words depends on the chosen language and also involves handling various dialects, imperfect pronunciations, intonations, and different ways of speaking words.

  - Understanding words requires separate HMM and NNs to learn how to predict words from sequences of phonemes. These methods often can only recognize words that were used during training. In the end, we have a stream of text and can use any text retrieval methods to search through spoken text.

  - To skip word recognition, we can search directly in the phoneme stream. The stream is captured using N-grams, which are overlapping sequences of N phonemes. If the query is not spoken text, it is translated into phonemes using a dictionary and the same N-gram extraction process occurs. Then, we search for the best passages in the spoken text library using the query N-grams.

Preprocessing

&

Windowing

FFT

MFCC

audio

signal

stream of

$𝒑$-dim

vectors

The Hidden Markov Model (HMM) creates a network of states using quantized MFCC data, with probabilities of moving from one state to another. Each phoneme has its own HMM, and a softmax across the phonetic units decides the recognized phoneme at a specific time. Making HMMs needs more human input or expertise but results in highly effective recognizers.

On the other hand, a Recurrent Neural Network (RNN) doesn't need as much specialized knowledge. "Recurrent" means the network keeps track of the current state and feeds it back into the network at each time step. You can think of a recurrent network as a series of connected networks, where the output of one network becomes the input for the next one.

s0

s1

s2

s3

a01

a12

a23

a11

a22

a33

b0,i

b1,i

b2,i

b3,i

A

A

A

A

A

h0

h1

h2

h3

h4

i0

i1

i2

i3

i4

Variations in speech tempo, how fast or slow someone talks, can compress, stretch, or omit parts of the audio signal, making it harder for models to align sounds with the correct phonetic units. To handle this, several strategies are employed:

  - Variable-length modeling: Rather than assuming each phoneme lasts a fixed amount of time, we let states last for different durations. A state can span several time frames, which lets the model handle slower or faster speech.

  - Use of Duration Models: Extensions include explicit duration modeling, which represents how long a phoneme or state is likely to last. This helps the system handle changes in tempo and avoid errors caused by unusually short or long durations.

  - Acoustic Feature Normalization: Features are sometimes normalized across time to lessen the effect of tempo changes. Methods such as cepstral mean normalization and vocal tract length normalization reduce the variability caused by different speaking rates.

  - Training with Varied Data: By training models on speech samples with diverse tempos and pronunciations, the system learns to generalize across tempo-induced variations.

Today's transcribers use transformer architectures to learn a sequence-to-sequence model from audio signals to text tokens. We have previously used transformers for natural language processing, where input and output were represented by token sequences. In image classification, the input sequence was a set of normal-sized image patches. When working with audio signals, we also need to perform some sort of normalization.

  - The speed at which people talk can differ a lot, but we don't have an easy way to make it consistent across a set of recordings. When we train, we mark the start and end of text with time stamps so the model can learn words spoken at different speeds.

  - The rate at which samples are taken can differ in recordings. When retrieving images, we encountered a similar problem with image sizes, which we reduced to fit the model's scale. When processing audio, reducing the sampling rate can cause unwanted auditory artifacts that make it difficult to distinguish between certain sounds (e.g., “s” and “f”). Conversely, higher sampling rates produce a lot of input data, requiring a large context window for the transformer (which is expensive). To lessen the amount of data going into the transformers, we use mel spectrograms as input data.

  - OpenAI's Whisper architecture is a multilingual transcriber with up to 1.5B parameters, as shown on the next page. It was trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using previous models. It currently supports about 100 languages with varying accuracy.

OpenAI/Whisper: https://github.com/openai/whisper
