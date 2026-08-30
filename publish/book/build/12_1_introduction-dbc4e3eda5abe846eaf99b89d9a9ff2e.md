# Introduction

Sound has two complementary meanings. In physics it is a mechanical wave that travels through a medium. In perception it is the sensation those waves create when they reach a listener. Both perspectives describe the same phenomenon and are important for later audio analysis.

Physical Perspective: Sound begins when something vibrates and disturbs the surrounding medium. For example, a vibrating loudspeaker membrane pushes air molecules back and forth, creating alternating regions of compression and rarefaction. These variations travel outward as a wave with a specific wavelength, frequency, pressure, speed and direction. The wave moves through the medium, but the particles themselves only oscillate around their resting positions. Because sound depends on interactions among particles, it cannot travel through empty space. The relationship between wavelength and frequency is set by the wave speed, and in air under standard conditions that speed depends on temperature. Sound travels faster in liquids and faster still in solids, where dense molecular structures pass vibrations more efficiently. As waves move they spread, bounce off surfaces, bend at boundaries between different materials and gradually lose energy. Studying these behaviors lets one trace how sound travels through an environment and how it interacts with objects before reaching the listener.

  - The human ear can hear sounds with frequencies between 20Hz and 20kHz. This corresponds to sound waves that are 17m and 17mm long in air at standard conditions. The relationship between wavelength and frequency is determined by the speed of the wave: $\lambda ∙f=v$.

  - The speed of sound waves varies depending on the medium. In air at standard conditions, sound travels $v=331+0.6∙T$ m/s with $T$ the temperature in Celsius. In water, sound travels much faster at about $v=1482$ m/s. In solids, speeds are even higher, ranging from $v=4000$ m/s in wood to $v=12,000$ m/s in diamonds.

  - Sound pressure is the difference between the pressure in the medium and the pressure of the wave. It is usually measured in decibels using the formula: $L_{p}=20∙log_{10}(p/p_{ref})$, where $p$ is the sound pressure and $p_{ref}$ is the reference pressure (20 $\m $Pa in air). The factor of 20 is used because we are comparing the squares of pressures, and the logarithm adds an extra factor of 2. The logarithmic scale is necessary because of the wide range of perception. 0 dB is the threshold of hearing, and sounds above 120 dB can cause permanent hearing loss.

Human Perception adds another dimension to the story. Humans hear only part of the full range of sound in nature. We respond to frequencies between 20 hertz and 20 kilohertz, while other species extend that range. Cats hear much higher pitches, and bats go well into ultrasound, using those sounds to navigate and hunt. Within the human range, several perceptual attributes shape how we experience sound.

  - Pitch is how we hear the frequency of sound. It helps us tell if music is high or low. Pitch needs a steady and clear frequency to stand out from noise. It's connected to frequency but it's not the same thing.

  - Duration is how long a sound is heard, from when it's first noticed until it fades away. It's connected to the actual length of the wave signal, but it can make up for breaks in the signal. For example, even if a radio signal is interrupted, it can still be heard as a continuous message.

  - Loudness is how loud or soft a sound seems. Our ears react to sounds over short periods of time, so a quick sound might seem quieter than a longer sound that's actually the same loudness. How loud a sound seems can change depending on the mix of different frequencies.

  - Timbre is the range of frequencies we hear over time. Different sound sources, like a guitar, a rock falling, or the wind, have distinct timbres that help us tell them apart. Timbre is like a fingerprint for sound, describing how it changes over time.

  - Sonic Texture is how different sounds interact, such as in an orchestra or on a train. The sound of a quiet market is very different from a busy party.

  - Spatial Location refers to where a sound is coming from, not necessarily its actual source. This, combined with the sound's quality, helps us to focus on one specific source, like a friend at a party.

Sound is represented as an amplitude signal over time. To turn the continuous signal into a digital form, it is sampled at a fixed frequency $f_{s}$. The Nyquist-Shannon sampling theorem says that the sampling rate determines the highest frequency $f_{max}$ that can be accurately represented, which is half the sampling rate ($f_{max}=f_{s}/2$). Since human hearing ranges from 20Hz to 20kHz, CDs use a sampling rate of 44.1kHz and DVDs use 48kHz.
For many analysis tasks the time domain alone does not show how the spectrum changes. A waveform shows how pressure varies over time but not which frequency components produce that variation. When a sound contains several tones, harmonics, or noisy elements, those components blend into the amplitude curve and cannot be separated by sight. The frequency domain provides this missing view by breaking the signal into its component frequencies and showing how much energy each one carries.

A standard Fourier transform breaks a signal down into its constituent frequencies and provides one global spectrum that averages frequency content from beginning to end. Any changes that occur during the recording are lost in that average. This matters when sound evolves over time, as most natural and musical sounds do. A music instrument may begin with a bright attack and then settle into a softer tone. A single Fourier transform cannot capture these transitions because it merges every moment into one fixed description.

The Short-Time Fourier Transform (STFT) addresses this problem by applying a window function that isolates a small segment around a chosen time. Each windowed segment is transformed into a local frequency spectrum. By sliding the window along the signal, one obtains a record of how energy at different frequencies changes as the sound unfolds. The result is usually displayed as a spectrogram, where the squared magnitudes of the transform form a time frequency map. Window length controls the tradeoff between time and frequency resolution. A short window gives better timing but blurs frequency detail, while a long window gives better frequency detail but poorer timing.

  - The discrete frequency $\omega $ ranges from 0 to $f_{max}=f_{s}/2$ at steps of $f_{s}/N$ Hz, with a window size of $N$ samples. The absolute values of the complex value $X(t,\omega )$ indicate the magnitude of the frequency $\omega $ at time point $t$.

  - The image on the right shows the STFT with the red windowing function applied over time. The spectrogram then displays the squared magnitudes $\left|X\left(t,\omega \right)\right|^{2}$ over time. Different windowing functions can be used.

$X\left(t,\omega \right)=\sum_{n=−\infty }^{\infty }x\left(n\right)∙w\left(n−t\right)∙e^{−i\omega n}$
To build features, the signal is split again, not to change it but to organize the information (compare with chunking in text retrieval). The audio is first divided into overlapping frames, and each frame yields a feature vector. Overlap prevents abrupt changes at frame boundaries and keeps feature changes smooth. Frames are then grouped into longer segments that can span several seconds. These segments serve as the basic units for retrieval or classification. Frame-level features are combined into segment-level descriptors by computing statistics that summarize each segment's behavior. A recording thus becomes a sequence of segments, each reflecting a local portion of the sound.

Frame length and segment length depend on the task at hand.

  - Frame size: Assuming a sampling frequency of $f_{s}=48$ kHz, a frame size of 40ms gives us $N=1920$ samples. This means the frequency resolution of STFT is $\frac{f_{s}}{N}=20.83$ Hz. This resolution is not enough to distinguish between two musical pitches in the middle octave, but it's okay for the first and second octave. To improve frequency resolution, we can increase the window size, but this will blur the spectrum and we lose precision along the time axis. With STFT, we have to compromise either on frequency resolution or time resolution.

  - Segment size: The size of the segment depends on the specific task. For detecting timbre (like guitar, rock falling, wind), a shorter segment is suitable. For spoken text, different segmentation methods can be used. The 4s shown in the picture is a good starting point for analyzing audio in general.

With this basis we can extract perceptual features in both the time domain and the frequency domain. Time domain descriptors, like energy or envelope shape, estimate how loudness changes over time. Frequency domain descriptors, like spectral centroid, show the dominant frequencies and how they vary. To capture human perception, we must understand how the auditory system converts the physical amplitude signal and its spectrum into meaningful sensations.

AudioSignal

Framinglength=40ms

hop=20ms

features

Feature Extraction

Segmentationlength=4s

hop=100ms

Statistical Computation

The ear does not respond directly to raw pressure changes. It first filters sound with the outer ear, then amplifies and shapes it in the middle ear, and finally breaks the sound down by frequency along the basilar membrane in the inner ear. Different places on that membrane respond to different frequency ranges. The neural signals that result convey an organized version of the sound spectrum as we perceive it, not the raw spectrum.

  - The outer ear is the visible part of the ear. It reflects and amplifies sound and provides clues about where a sound comes from. Sounds enter the ear canal, which boosts frequencies near 3 kHz by as much as 100 times. This boost helps us recognize voices and hear differences like "s" versus "f". The amplified sound then reaches the eardrum.

  - Sound waves move from the eardrum into the air-filled middle ear and then through three tiny bones: the hammer (malleus), anvil (incus), and stirrup (stapes). These bones act like a lever to amplify the signal at the oval window, also called the vestibular window. This amplification is necessary because the cochlea is filled with fluid. A reflex in the middle ear helps protect against damage from very loud sounds.

Chittka L, Brockmann - Perception Space—The Final Frontier, A PLoS Biology Vol. 3, No. 4, e137 doi:10.1371/journal.pbio.0030137

outer ear

middle ear

inner ear

  - The inner ear consists of the cochlea and the vestibular system.

    - The vestibular system helps with balance and sensing body movement.

    - Inside the cochlea is the organ of Corti that does most of the ear's work of turning mechanical vibrations into nerve signals. It has two kinds of hair cells, each with its own role.

    - The outer hair cells act like tiny biological amplifiers. They change shape when vibrations arrive and, in doing so, sharpen the motion of the basilar membrane. This extra boost helps the ear tell apart frequencies that are very close together and makes it more sensitive to quiet sounds.

    - The inner hair cells convert vibration of the membrane into electrical signals for the auditory nerve. Their response is not proportional to motion: they respond strongly to small movements but cut back their output when motion becomes too large. This protects the inner ear and shapes how we perceive loudness. High frequencies are detected near the base, close to the middle ear, while low frequencies are detected toward the apex, creating a natural frequency map.

  - Outer hair cells in the cochlea actively amplify quiet sounds more than loud ones. This increases the ear's usable dynamic range. Chemical processes in the cochlea also let the ear adapt to steady or unchanging sounds. When the response to a constant tone fades slightly, new or changing sounds stand out more. Together, these mechanisms explain why the ear is both highly sensitive and highly selective, and why perceptual features must reflect these non-linear, adaptive properties.

  - Inner hair cells convert mechanical vibrations into electrochemical signals by releasing neurotransmitters, which are then picked up by the auditory nerve fibers. The cochlear nerve contains about thirty thousand of these fibers, and each one responds best to a particular frequency at a particular loudness level. Alongside it runs the vestibular nerve, which carries information about balance and motion. Sound information reaches the brain through two routes: the primary auditory pathway and the reticular pathway. The reticular pathway blends all sensory input and helps the brain decide which events need attention. The primary pathway follows a more specialized sequence.

    - The cochlear nuclear complex is the first brain region to analyze the frequency, intensity, and duration of the incoming signal.

    - The superior colliculus in the midbrain uses differences across frequency bands to estimate the direction from which a sound arrives.

    - The thalamus, specifically the medial geniculate body, prepares the information for actions such as speaking or reacting to a sudden noise.

    - Finally, the auditory cortex handles both basic and advanced aspects of hearing. Its neurons are arranged by frequency in detailed maps that help identify sound sources and reduce distortions caused by reflections. It also processes the fine timing of sounds, which is essential for understanding speech and recognizing complex auditory patterns such as music.
