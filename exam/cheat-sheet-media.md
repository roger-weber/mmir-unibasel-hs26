# Cheat sheet: Multimedia Content and Features

Retrieval beyond text: the semantic gap between signal features and concepts, metadata and the transformer paths that bridge it
(chapter 9), image features from color histograms to CLIP (chapter 10), audio features, classification, transcription and
fingerprinting (chapter 11), and shot detection, motion and video search (chapter 12).

## Chapter 9: Content Analysis & Metadata

### The semantic gap

- The semantic gap is the distance between the low-level features a system can compute from the signal and the high-level concepts people use to describe and search content. Text carries its meaning on the surface (query and document share one symbol system), so it needs matching. Images, audio and video hold pixel or amplitude values; their meaning must first be extracted (content analysis).
- The gap has two parts. Concepts visible in the content ("sky", "dome") can be closed by analysing the content. Concepts not in the signal ("UNESCO World Heritage Site", "where is Mumtaz Mahal buried?") need outside context: metadata, the web page around the item, or the world knowledge of a large model.
- The vocabulary mismatch of text ("car" versus "automobile") is a different problem: query and document use the same kind of symbols, only the words differ. A media item contains no symbols at all, so a description must exist before a vocabulary mismatch can arise. Embeddings closed most of the mismatch for text because they start from words.
- A new clip has no viewing history, and behaviour cannot tell whether it shows kittens or a lion, so platforms combine behavioural signals with content signals (captions, sound track, visual features).
- Content analysis climbs an abstraction ladder, each level more abstract than the one below:

| Level | Holds | Example |
|---|---|---|
| Raw | stored pixels or samples | JPEG, MP3 file |
| Perceptual (low) | how it looks or sounds | color histogram, texture |
| Structural (mid) | recognized patterns, categories | "wolf", scene type |
| Semantic (high) | descriptions in human concepts | caption, transcript |
| Contextual | information from outside | timestamp, location, tags |

- Raw values help little: two photos of one building taken a few steps apart share almost no identical pixels. Perceptual features assume that similar signal statistics mean similar content. This often holds, but not always: a red car and a sunset have similar color histograms. They suit query by example, near-duplicates and "more like this", not concept queries such as "pictures of cats".
- Good perceptual features stay stable under what perception ignores: translation, rotation, scale, lighting and loudness, noise.
- Three complementary paths bridge the gap: **keywords** (metadata searched as text), **features** (perceptual similarity, plus classifiers whose labels feed the keyword path), and **transformers** (generate a description for the text index, or embed media and text in one space).

### Metadata and its limits

- With metadata, media search becomes text search, and the whole text toolkit applies unchanged; the gap is bridged by whoever wrote the description.
- Sources: manual annotation; capture and harvesting at almost no cost (camera date and GPS in EXIF, caption and surrounding text of a web page); labels from classifiers and models. Values sit at a generic ("building"), specific ("Taj Mahal") or abstract ("devotion") level and are organized in facets, which map directly to filters.
- Standards (EXIF, IPTC, Dublin Core, ID3, schema.org) fix the fields, not the values: one annotator writes "Taj Mahal", another "religious building".
- Curated catalogues (IMDb, MusicBrainz) exist where commercial or community interest pays. Manual annotation fails on cost, consistency and depth. Harvesting must decide which text belongs to the image, judged on the rendered page; anchor texts are in the [text sheet](cheat-sheet-text.md#ranking-signals-beyond-the-page-text).
- Every label is a claim, not a fact: owners can mislead, while metadata written by others is harder to manipulate. Classifiers confuse similar classes (a coyote labelled "wolf"), and a confidence value expresses how strongly the model favours a label, not how often such labels are correct. Each source is evaluated against ground truth before it is trusted, with thresholds per label.

### Describe or embed: two transformer paths

- **Describe, then search text.** A prompted multimodal language model acts as a configurable metadata extractor and writes a description plus JSON fields: text fields for BM25 and text embeddings, entities and keywords for exact match and facets, boolean flags as filters ("outdoor photos without people"). A new field needs a new prompt, not new training.
- Gains: readable metadata an archivist can check, cheap mature text indexes, explainable hits. Costs: the index holds only what the model chose to write (an unmentioned red umbrella is never found), and every prompt change means a new pass of a large model over the collection.
- **Hallucination**: the model writes the most plausible continuation and states a guess with the same confidence as an observation, worst for names, places and dates. Remedies: mark uncertain values or leave fields empty; cross-check with independent metadata (GPS coordinates in Canada rule out "Matterhorn"); a controlled list of entities; human verification. Generated fields are evaluated like a classifier: sample items and measure precision and recall per field.
- **Shared embedding space.** A dual encoder such as CLIP has one encoder per modality, trained on image-caption pairs so that matching pairs get nearby vectors. One vector per item supports search in every direction and zero-shot labels from prompts such as "a photo of a mountain". A multimodal language model, in contrast, keeps an image as a sequence of tokens inside the model, not one indexable vector.

| Aspect | Describe, then search text | Shared embedding space |
|---|---|---|
| Coverage | what the description mentions | any visual concept learned |
| Explainability | high: matched words | low: a similarity score |
| Specific entities | good, if named correctly | weak for rare names, identifiers |
| Indexing cost | high: generative model per item | moderate: one encoder pass |
| Typical failure | hallucinated or missing details | plausible but wrong neighbours |

- The describe path excels at facts, filters and explainable results; the embedding path at visual details nobody wrote down ("a lonely bench in the fog"). A hybrid fuses both rankings (see the [index sheet](cheat-sheet-index.md#hybrid-search-and-rank-fusion)).

**Go deeper:** [Describe or embed (demo)](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch09-03-multimodal-transformers.ipynb)

## Chapter 10: Image Features

### Hand-crafted image features: color, edges, texture, shape

- **Color space.** Pixel values measure light, not perception: equal sRGB distances are not equal perceived differences. CIE Lab makes Euclidean distances approximate perceived differences. HSV is not perceptually uniform but separates hue from brightness, so a hue-only feature ignores shadows; gray and very dark pixels have no reliable hue. The task decides: Lab where perceived differences matter, hue where lighting should not.
- **Color histogram.** Each pixel goes to its nearest reference color; the counts are normalized. Size, shift, rotation and mirroring leave it unchanged. Bin-by-bin distances (L1, L2) punish a lighting change that crosses a bin edge as much as a change to red; a cross-bin (quadratic) distance credits similar colors at $N^2$ instead of $N$ operations; intersection answers partial queries such as a color palette.
- **Color moments** (mean, spread, skewness per channel, plus covariances) need about 15 values instead of hundreds of bins and have no bin edges, but describe a single peak (sky and forest average to a lightness few pixels have) and must be rescaled before comparison.
- Global color ignores layout (a lake counts like a sky); features per region win some back.
- **Edges** mark abrupt intensity changes (the gradient). Only surface-orientation and depth edges belong to the shape; shadow and material edges do not. Edge-direction histograms ignore where edges lie.
- **Texture versus shape.** Texture describes how intensity changes repeat over a region: a bank of Gabor filters at several orientations and scales, with responses summarized by moments, needs no segmentation. Shape describes the outline or area of one region, which must be found first, so it works on clean backgrounds (products on white, trademarks, cells).
- **HOG** keeps the layout of edge directions per cell and tolerates small shifts and lighting, but is not invariant to rotation or scale.
- **SIFT** describes keypoints relative to their own scale and orientation, so it finds the same object or scene despite scale, rotation and light changes, but not a category. Repetitive structures (windows, tree tops) match by chance, so the matches must agree on one geometric transformation; the count of consistent matches is the score. At scale, descriptors become visual words, weighted with tf-idf in an inverted file.

**Math** (*for understanding; formulas are not asked in the exam*). Intersection of image $h$ with a partial query $g$ (entries sum to less than 1), a similarity between 0 and 1:

$$\delta_{\cap}(h, g) = \frac{\sum_{i} \min(h_i, g_i)}{\min\left(\sum_{i} h_i, \sum_{i} g_i\right)}$$

Quadratic distance: matrix $A$ holds the similarity of reference colors ($a_{i,i} = 1$), so differences in similar bins partly cancel:

$$\delta_{\text{quad}}(h, g) = (h - g)^T A \, (h - g)$$

**Go deeper:** [Color Features](https://roger-weber.github.io/mmir-unibasel-hs26/book/color-features/), [Edges, Texture, and Shape](https://roger-weber.github.io/mmir-unibasel-hs26/book/edges-texture-shape/)

### Learned image features: CNNs

- Hand-crafted features work for a few classes with a clear difference (speech versus music), but nobody can design them for thousands of object classes. A convolutional network learns them from labelled photos; it is never told about edges.
- **Convolution** slides small learned filters over the whole image: one filter finds its pattern at every position with few parameters (shared weights give translation invariance). The first layer rediscovers edge, stripe and color filters.
- Each filter yields a **feature map**, bright where its pattern occurs; the next layer combines patterns. Every few layers the resolution is halved, so later layers see larger patterns, climbing from edges over textures and parts to objects.
- The output names trained classes only: the Taj Mahal becomes "mosque" with high confidence. A new class needs new labels and training, at least of a new head on the frozen network.
- **Embedding:** the second-to-last layer, averaged over positions. It was trained to separate classes, not to compare images, so distances within a class were never trained (another peak can outrank the same mountain). Metric learning on matching pairs fixes this, the image counterpart of the bi-encoder (see the [semantic sheet](cheat-sheet-semantic.md#bi-encoders-cross-encoders-and-colbert)). Early layers are poor retrieval features: every photo contains all kinds of edges and colors.
- **Transfer learning:** a frozen network with a small classifier on top is cheap and needs few labels. Fine-tuning helps for domains far from the training photos (X-rays, satellite images) but needs more data and can overwrite what was learned.
- **Face embeddings** encode identity via metric learning, so verification, identification and clustering become vector operations. As biometric data they need a legal basis, a clear purpose and an evaluation per demographic group.

**Math** (*for understanding; formulas are not asked in the exam*). One convolution step: multiply each filter weight by the pixel under it and sum; sliding over all positions yields the feature map:

$$y(x, y) = \sum_{i} \sum_{j} w_{i,j} \cdot p(x + i, y + j)$$

### Transformer embeddings for image search

- A Vision Transformer treats patches as tokens; attention links distant regions from the first layer.
- **CLIP** learns from image-caption pairs, without class labels: one space for text-to-image search and zero-shot labels. Strong on what captions mention (objects, scenes, landmarks), weaker on visual details captions rarely mention.
- **DINOv2** learns from images alone (different crops of an image must get the same description) and captures detailed visual similarity, but takes no text queries.

| Query | Model |
|---|---|
| look-alike images | DINOv2 |
| same concept, different look | CLIP image embedding |
| text query | CLIP |
| near-duplicates | DINOv2 separates copies most clearly |

- Color histograms fail on global edits: a grayscale copy lies farther from the original than an unrelated photo.
- Each model has its own score range; image-text cosines under CLIP typically lie between 0.1 and 0.4 even for good matches, and the differences between candidates carry the information. Thresholds are set per model on evaluation data, scores of different models are not comparable, and a new model means re-encoding the whole collection.
- Learned features replaced hand-crafted ones for concepts and categories. Hand-crafted features still win for exact color constraints (intersection for "this exact red"), the same object with evidence (consistent SIFT matches can be shown), explainability, and cost (no model, no GPU). Real systems combine both.

**Math** (*for understanding; formulas are not asked in the exam*). CLIP compares image embedding $v$ and text embedding $t$ by cosine; zero-shot picks the class prompt $t_c$ with the highest score:

$$\cos(v, t) = \frac{v \cdot t}{\|v\| \cdot \|t\|}$$

**Go deeper:** [Transformer Embeddings for Image Search](https://roger-weber.github.io/mmir-unibasel-hs26/book/transformer-image-search/), [Transformer embeddings and then vs. now (demo)](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch10-04-learned-features.ipynb)

## Chapter 11: Audio Features

### From waveform to spectrogram

- The sampling rate sets the highest representable frequency (half the rate), not the frequency resolution; speech needs a lower rate than music.
- One spectrum of a whole recording loses all time order. The short-time Fourier transform cuts the signal into short overlapping frames, one spectrum each: the spectrogram.
- A longer frame gives finer frequencies but blurs events in time; no frame gives both. Short frames for speech, long ones for pitch and chords. Overlap puts an event at a frame edge into the middle of the next frame.
- Audio has no natural boundaries: frames are grouped into overlapping segments of a few seconds, the unit a classifier labels and a search returns, like a text passage.
- The ear resolves low frequencies finely and high ones coarsely and hears loudness roughly logarithmically, the basis of the mel scale.

**Math** (*for understanding; formulas are not asked in the exam*). The resolution depends only on the frame duration $T$; a 40 ms frame resolves 25 Hz at any sampling rate:

$$
\Delta f = \frac{1}{T}
$$

### Perceptual and musical audio features

- Waveform: short-time energy (how strong a frame is) and zero-crossing rate (how often the sign changes; high for hissing, noise-like sounds).
- A single frame says little. Over a segment, speech alternates syllables and pauses: many low-energy frames and many high-crossing frames. Speech varies more from frame to frame, which segment statistics capture.
- Spectrum: the spectral centroid measures brightness, not pitch; the spectral flux measures how much the spectrum changes between frames (onsets of notes and syllables).
- Ear models: energy is not loudness, which needs frequency weighting and averaging over time. The mel scale spaces bands as the ear does; MFCCs keep the smooth spectral envelope (timbre) and drop the fine structure (pitch). MFCC values depend on the implementation, so index and queries must use the same pipeline.
- Chroma folds the energy into the 12 pitch classes regardless of octave: harmony and key, largely independent of the instrument, the classic feature for chords and cover versions. It is not the melody, which needs fundamental-frequency estimation (prone to half or double errors).
- Tempo comes from the periodicity of onsets, with octave errors that listeners share. Measured content can contradict metadata: a Brahms recording titled "in F♯ minor" plays in G minor.

**Go deeper:** [Perceptual Features](https://roger-weber.github.io/mmir-unibasel-hs26/book/perceptual-features/), [Musical Features](https://roger-weber.github.io/mmir-unibasel-hs26/book/musical-features/)

### Classifying audio

- A decision tree on segment statistics separates speech from music: fast, little training data, readable, small enough for a microcontroller. It fails on music that behaves like speech (drums, decaying vibraphone notes).
- Segment labels become stream labels by smoothing: stronger smoothing suppresses more isolated errors but reacts later to a real change. A file is labeled by majority vote.
- Learned classifiers (CNNs, audio transformers on the log-mel spectrogram) cover hundreds of sound classes with higher accuracy. Each class gets its own score, since several sounds can occur at once, so scores do not sum to one; the second-to-last layer is an embedding. Their labels have errors of their own.
- CLAP puts audio and text in one space: zero-shot labels via prompts and text queries, without training for a label. A softmax always picks one offered label (a whale song becomes "music"); a high score means "closer than the other prompts", so add a catch-all prompt or a threshold on the raw similarity. Prompt wording matters.
- A decision tree still fits two classes with a clear acoustic difference under tight hardware limits.

### Speech transcription

- Transcription is the [describe path](#describe-or-embed-two-transformer-paths) for audio: words go into the inverted index, timestamps point to the moment, speaker labels add a facet.
- The classical pipeline chains separately trained parts: MFCC vectors every few milliseconds; an acoustic model in which each phoneme is a hidden Markov model whose few states are passed left to right, so slow and fast speech fit; a pronunciation dictionary that assembles words; an n-gram language model that prefers likely sequences ("recognize speech" over "wreck a nice beach"); a search for the most probable transcript. Each part needs expert work, and a name outside the dictionary cannot be recognized.
- Recurrent networks replaced the acoustic model, then transformers the whole chain. Whisper is one network trained on weakly labeled web audio, with tasks for language detection, transcription, translation and timestamps. Its decoder is a language model, so on silence, music or noise it can hallucinate or repeat phrases: a transcript describes the audio, it is not the audio.
- Recognizers built on large language models add context knowledge and accept instructions (a language hint, a list of names).
- Many languages are not solved: quality falls along the long tail of training data, benchmark scores need not hold on one's own recordings, language detection fails on short or mixed clips, and dialects without a spelling are in effect translated (Swiss German into Standard German).
- Errors concentrate on the names users search for, and the word error rate weighs all words equally, so a low rate does not guarantee findable names. Remedies: hotword lists at indexing time (must be maintained) and phonetic matching at query time. Measure the recall of query terms on a sample.

### Audio fingerprinting

- Task: one exact recording among millions from a few seconds of distorted sound, within a second. Similarity features return similar songs; raw spectra fail under noise.
- Keep only the spectrogram peaks (the constellation map): noise rarely displaces the strongest peak, and compression or a phone microphone change amplitudes, not peak positions.
- Pair each anchor peak with peaks in a target zone; the code (two frequencies, time difference) is specific, independent of where the query starts, and redundant: a lost peak destroys only its own pairs. Codes and their postings (recording, anchor time) form an inverted file.
- Chance matches are many. The decision comes from time alignment: correct matches agree on one offset between query and recording, which also gives the position. The raw match count misleads, since a long recording collects many chance matches spread over many offsets.
- The work grows with the matched postings, not with the catalogue size.
- The price: it matches recordings, not songs, so live versions, covers and humming fail, and speed or pitch changes break it. A specialized method fits when the task asks for an exact item, queries are heavily distorted, or a huge catalogue must answer within a second.

**Math** (*for understanding; formulas are not asked in the exam*). With anchor times $t_q$ in the query and $t_s$ in recording $s$:

$$
\text{score}(s) = \max_{\delta} \; \text{number of matched codes with } t_s - t_q = \delta
$$

**Go deeper:** [fingerprinting demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch11-03-fingerprinting.ipynb)

## Chapter 12: Video Features

### Shot boundary detection

- A shot, one continuous camera recording, is the basic retrieval unit; its boundaries are the natural edges of a result. They must be found in the images, since sound runs on across cuts.
- A hard cut switches abruptly; a gradual transition (fade, dissolve, wipe) blends over many frames. Compression I-frames are cheap candidates, not boundaries.
- Change measures between consecutive frames: the pixel difference reacts to every motion (a pan, a person walking); the [histogram](#hand-crafted-image-features-color-edges-texture-shape) difference ignores where pixels lie, so it stays stable within a shot and jumps at a cut; a cross-bin distance also tolerates small shifts across bin edges. Similar shots hide a cut, a flash creates a false one.
- The cut threshold is a binary classification: pairs within a shot are negatives, pairs across a cut positives, and their distributions overlap. Procedure:
  1. Annotate the cuts in a sample of the target collection.
  2. Compute the change of every frame pair.
  3. Per threshold, compute the true-positive rate (share of cuts found) and the false-positive rate (share of within-shot pairs flagged).
  4. Plot the ROC curve and pick the operating point.
- A missed cut costs more than a false one: a false cut adds a redundant keyframe, a missed cut can drop the second shot from the index. So favour a high true-positive rate.
- One threshold does not fit all videos (genre, light, editing): tune on the target collection or use an adaptive threshold relative to the surrounding frames.
- In a gradual transition no single step crosses the cut threshold, and lowering it turns every larger motion into a cut; twin thresholding handles it (below). A slow pan or zoom then becomes a false soft cut unless camera motion is checked.
- A learned detector sees a window of frames and learns that a flash lasts one frame, at more compute; it still needs a threshold, and dissolves stay hard for both.
- Keyframes: the middle frame for a short shot; several (clustered or sampled) for a long shot that changes.

**Algorithm: twin thresholding** (*for understanding; formulas are not asked in the exam*). Two thresholds, an upper cut threshold $T_c$ and a lower start threshold $T_s$, applied to two distances: $d(t)$ between neighbouring frames $t-1$ and $t$, and the distance of a frame to a reference frame $r$.

1. $d(t) \geq T_c$: hard cut between $t-1$ and $t$.
2. $T_s \leq d(t) < T_c$: a gradual transition may start; frame $r = t-1$ becomes the reference.
3. Follow the frames while $d(t)$ stays at or above $T_s$: the content keeps changing.
4. When $d(t)$ falls below $T_s$, the change has stopped at frame $e$. Compare $e$ with the reference:

$$
d\big(h(e), h(r)\big) \geq T_c \;\Rightarrow\; \text{soft cut from } r \text{ to } e
$$

Each step of a dissolve is small, but the frames before and after it differ as much as the two sides of a hard cut. If $e$ is still close to $r$, it was a false alert (someone walked past): no cut.

**Go deeper:** [Shot Boundary Detection](https://roger-weber.github.io/mmir-unibasel-hs26/book/shot-boundary-detection/), [shot detection demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch12-01-shot-detection.ipynb)

### Motion detection and optical flow

- With a static camera and no cuts, motion is the retrieval unit. Motion shows as a difference between images; a threshold gives a mask of moving pixels.
- Background subtraction compares with a continuously updated background model: needs a static camera, leaves a ghost when a parked car leaves, needs a mixture model for waving leaves.
- Frame differencing compares with the previous frame: adapts to light at once, but shows mainly the leading and trailing edges, and a stopped object vanishes.
- A motion history image condenses recent masks into one image, a feature for the amount, direction and place of motion.
- A moving camera needs optical flow: the apparent motion of brightness patterns in the image, not the motion in the scene. Brightness constancy (a point keeps its brightness between two frames) gives one equation per pixel for two unknowns, so a pixel reveals only the motion across an edge: the aperture problem.
- Lucas-Kanade assumes one motion for a small window, so gradients in several directions fix the vector: reliable at corners and texture, undetermined on a straight edge or flat region. Real time on a few points; learned dense flow covers every pixel at far higher compute.
- All methods fail where brightness changes without motion (a light switched on).

**Math** (*for understanding; formulas are not asked in the exam*). The optical-flow constraint for the motion $(u, v)$ of a pixel, with spatial gradients and change over time:

$$
I_x \, u + I_y \, v = -I_t
$$

### Searching video

- Users want a moment, not a file. Units are frames, shots and scenes (shots sharing place, time or action; they need meaning and are harder to find than shots), and each match is returned with its position.
- The simplest engine embeds shot keyframes with an image model (which keyframes: see [Shot boundary detection](#shot-boundary-detection)). Sampling and pooling frames keeps brief content as a weaker trace; indexing every sampled frame finds it best, at a larger index.
- Image embeddings show what is visible, not what happens (opening versus closing a door). Models that see motion pay attention costs that grow quickly with the number of frames, so they see clips of seconds, not films.
- Long shots are split into overlapping windows of a few seconds, like text chunks.
- An index of moments aligns visual embeddings, transcripts, on-screen text and audio events on the time line. Each track is searched its own way: BM25 for spoken or shown words, vectors for visual concepts, filters for event types and time ranges. The rankings are fused, adjacent hits merge, and clip edges snap to shot boundaries. A spoken line is found in the subtitles, not in the frames.
- Aligned tracks need not talk about the same thing (background lyrics, a news ticker); a moment on which several tracks agree is more reliable.
- Cascades: cheap signals scan everything, expensive analysis runs on candidates. Football: crowd noise proposes, speech recognition and the scoreboard confirm; the crowd cheers after the goal, so the clip starts one shot before the reaction.
- For long videos, the index finds candidates and a video-language model inspects only those, like RAG.

**Go deeper:** [Video Understanding with Transformers](https://roger-weber.github.io/mmir-unibasel-hs26/book/video-understanding/), [text-to-video search demo](https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch12-03-text-to-video-search.ipynb)
