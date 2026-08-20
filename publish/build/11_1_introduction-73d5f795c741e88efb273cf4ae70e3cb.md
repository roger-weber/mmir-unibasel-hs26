# Introduction

Information retrieval research often begins with text, where meaning is studied using language models, vector embeddings, and relevance feedback. Text already has built-in meaning: words and sentences link symbols to concepts. Even when ambiguity appears, the connection between representation and meaning is clear enough for retrieval systems to form more advanced interpretations using embeddings and retrieval augmented generation. Text supports reasoning about ideas, relationships, and discourse.

Images do not contain linguistic cues. Moving from text retrieval to visual retrieval therefore changes both the data and the gap that must be bridged. A document has words and grammar, while an image is an arrangement of light intensities and colors. The so-called semantic gap comes from the distance between that raw visual signal and the concepts people use to describe what they see. A retrieval system may be asked to find sunsets, happy faces, or industrial cities at night, yet none of those categories appears explicitly in the pixel grid.

Before addressing this gap with deep networks or multimodal language models, it helps to review how visual representation developed in earlier computer vision. The first steps did not try to recognize objects or interpret scenes. They aimed to describe images at the level of perception rather than meaning. These methods treated an image as a visual signal whose measurable attributes reveal structure, variation and distribution. Researchers examined the role of light, the behavior of reflectance, and the constraints of the human visual system. The transformation from illumination and material to retinal response guided how color spaces are defined, how edges appear, how textures emerge, and how these properties can be summarized for retrieval.

This approach fits into the course's broader arc. Text retrieval introduced the idea that documents can be converted to feature vectors, either by bag-of-words counts or by embeddings from large models. Moving to image retrieval keeps the same idea but at a lower level. Instead of vectors that capture meaning, we use vectors that record perceptual attributes. Color histograms, texture energies, and local gradients play the role of term frequencies, but they do not refer to high-level concepts.They are efforts to represent the visual world so it stays stable under common variations and gives enough information to support search.

A system that can describe an image directly from its content, even at a basic level, reduces the need for external labels. It also gives an initial view of how computer systems see the world. As the course continues, these perceptual methods form the basis for later techniques, including machine learning models for classification and clustering and the CLIP model that connects images and text.

Visual features are hand-crafted descriptors based on human visual processing and help close the semantic gap in image retrieval. They measure numeric properties of raw pixels to mimic how people perceive color, texture, shape, and layout. By modeling these perception steps, visual features enable content-based image retrieval, or CBIR, where queries are images, sketches, or color palettes rather than keywords.

These methods are inspired by the human visual system. Light reflected from surfaces reaches the retina, where cone cells compress rich spectral information into three signals tied to long, medium, and short wavelengths. This trichromatic encoding underlies color spaces such as RGB and CIE Lab. The mapping from physical wavelengths to perceived color is not uniform. Small changes in RGB can cause large shifts in appearance, while large changes in other areas of RGB may hardly be noticed. The raw signal records physical properties, but perception follows a different interpretation. Perceptual features therefore do not try to copy the physical world. They encode how the world looks to an observer by transforming pixel values into models that measure perceptual rather than physical distance.

Consider the two images on the right as a demonstration of how perception differs from raw measurement. The top image shows a board of alternating dark and light squares, with part of the pattern in shadow. Two squares labeled A and B look different in brightness. In the lower image a strip of the same color connects the two squares, making it clear they are the same shade. The brain interprets the scene using expectations about light and shadow rather than the raw signal. This illusion explains why what we perceive can differ from physical measurements and why image representations that use only pixel values do not match human judgments of similarity.

Designing features for perceptual images requires understanding not only which visual elements matter but also which variations do not matter. Human perception tolerates many changes in raw pixel values as long as the meaningful content stays the same. Effective feature design aims to give computational systems the same resilience:

  - Translation invariance is the idea that small shifts in an image rarely change how people interpret it. For example, a photograph of a cat is still recognizable whether the cat is centered or slightly to one side. Features should be built so that minor shifts do not change their representation. Convolutional filters, pooling operations, and overlapping receptive fields are common ways to achieve this, similar to the spatial stability seen in biological vision.

  - Rotation invariance is also important. An object rotated by a small angle should still be seen as the same object. A coffee mug turned slightly to the side does not lose its identity, yet a naive pixel based system might treat the rotated version as completely different. Using rotation aware filters or adding rotated examples to the training data helps keep recognition consistent.

  - Scale invariance deals with a related problem. Objects can appear at very different sizes depending on distance or camera quality. A tree far in the background of one picture and the same tree up close in another still look like the same object. People recognize this easily, but computer features must be explicitly designed to handle changing resolution. Multi-scale feature pyramids or representations that use relative rather than absolute measurements help preserve similarity across sizes.

  - Resolution invariance fits naturally into the broader goal of perceptual invariance and extends the idea of scale stability. When resolution changes, pixel-level detail can shift a lot while the basic structure of the scene usually stays the same. A high-resolution image shows fine textures, such as fabric patterns or wood grain, while a low-resolution version preserves only broad shapes and color regions. People still recognize the content easily, so feature systems should do the same.

  - Lighting invariance is essential because illumination often changes in unpredictable ways. A face lit from the side, under soft, diffuse light, or in harsh midday sun is still the same face. Features that depend on absolute brightness will fail in those situations. Methods that focus on edges, gradients, or reflectance rather than raw intensity are more robust to lighting changes.

  - Finally, a perceptual feature system must tolerate noise and imperfections. Real images often have sensor noise, compression artifacts, banding, or color quantization, especially in low-quality or heavily compressed photos. A system that treats every small distortion as meaningful cannot judge similarity reliably. Smoothing, robust statistical measures, and representations that capture global structure instead of local pixel irregularities help reduce sensitivity to these artifacts.

Physical basis of visual perception:

  - Every image comes from how illumination interacts with surface reflectance. Illumination $l\left(x,y,z\right)$ , is the amount of light at a point in space, measured in lux and weighted by the eye's sensitivity to different wavelengths. Reflectance $r(x,y,z)$, is the fraction of that light a surface reflects back toward the viewer.

  - The upper image on the right shows this interaction on a single surface fragment. Light from the source hits the material, and the material reflects different wavelengths in different directions. The viewer sees only the light that reaches their line of sight. Small changes in lighting direction, intensity, or spectral composition can change how the same surface looks, because what we see is a combination of illumination and reflectance.

  - The second image shows typical reflectance curves for natural materials. Snow reflects light across most wavelengths, vegetation shows strong peaks in the green and near-infrared regions, and water reflects very little. These curves show that the physical signal is rich and continuous, while human vision and camera sensors reduce it to just a few channels. Apparent color therefore depends as much on lighting as on the material's properties.

  - Typical illuminance and reflectance values show this variability. Sunlight can be tens of thousands of lux, indoor lighting a few hundred. Bright surfaces reflect most of that light, while dark surfaces reflect only a small part. This explains why pixel values do not directly match perceived color or brightness. Perceptual features must account for how people interpret the combined effects of lighting and material, not just the raw physical signal.

Chlorophyll has its reception peaks in the blue and red spectrum of light. Hence, we observe only the reflected green spectrum of light.

The eye captures light and converts different wavelengths into electrochemical signals the brain can interpret. Light enters through the cornea, passes the pupil, and then reaches the lens. These parts form an adjustable optical system that focuses images of objects at different distances. The system also shifts for different light levels, similar to a camera's aperture and focus. The lens projects an inverted image onto the retina at the back of the eye.

  - The retina has photoreceptors called rods and cones, each serving a different role in vision. Cones are concentrated in the central retina, especially in the macula and its center, the fovea, and they provide sharp, detailed, and color vision. There are three types of cones, each sensitive to a different range of wavelengths.

    - L cones detect long wavelengths and peak near 564 nanometers, which corresponds to red

    - M cones respond to medium wavelengths and peak near 534 nanometers, associated with green

    - S cones are sensitive to short wavelengths and peak around 420 nanometers, enabling blue perception

  - Rods are more numerous in the retina's peripheral regions and work best in low light. They are essential for night vision and peripheral awareness but cannot detect color. Rods and cones convert incoming photons into neural signals that the brain turns into images.

The human eye has about 6 million cones and 120 million rods.  Cones come in three types:

  - Roughly 1 percent S-cones that are sensitive to blue light,

  - 39 percent M-cones that detect green, and

  - 60 percent L-cones that detect red.

Near the center of vision around the fovea, the share of blue-sensitive cones can rise to as much as 7 percent. These proportions vary a lot between people, which affects color perception and can cause color blindness when some cones are missing or do not work.

Visual acuity measures how sharp and clear vision is. It reflects the eye's ability to see fine details. The standard test uses a Snellen chart. On that chart, 20/20 means the eye can clearly identify objects about 1.75 millimeters apart from 20 feet, which is a visual angle of roughly one arcminute (1/60 of a degree). By contrast, 20/40 means the person must be at 20 feet to see what someone with normal vision can see at 40 feet, showing reduced sharpness.

Standard Snellen Chart

1.4’ or less is required to drive a car


When comparing visual abilities across species, animals perceive the world very differently. For example, a cat's visual acuity is about 20/100, much lower than a human's. Cats have only two cone types, sensitive to blue (around 450 nm) and yellow (around 550 nm), so their color vision is limited. They see 6-8 times better than humans in low light because they have many rods and a reflective layer behind the retina called the tapetum lucidum. Their field of view is about 200 degrees versus the human average of 180 degrees, which gives them broader peripheral vision for hunting but a less detailed, somewhat blurred image compared to human vision.

Human

Cat

Dogs also have dichromatic vision with blue and yellow cones. Their visual acuity is about 20/75, a bit better than cats but still less sharp than human vision. Larger animals such as elephants have lower acuity, around 20/200. Small rodents see much less clearly, about 20/800. Insects like bees have acuity near 20/1200. Flies have the poorest acuity, roughly 20/10800.

On the other side, birds of prey such as eagles have extraordinary vision that is about 4-5 times sharper than average human vision. This clarity lets them spot prey from long distances. Many bird species are tetrachromatic; they have four types of cones and can see a wider range of colors, including ultraviolet light that humans cannot see. Similarly, goldfish and zebrafish have four distinct cone types. The extra cone is usually sensitive to ultraviolet wavelengths that peak near 370 nm, helping them detect UV light underwater.

Goldfish and zebrafish are tetrachromatic

Processing of visual information already starts in the retina, an approach that inspired the use of convolutions in deep learning. Rods and cones show a distinctive chemical behavior: they release glutamate in the dark and stop releasing it when exposed to light. This is unusual for a sensory system but releasing neurotransmitters in the dark helps to support regeneration during sleep.

  - Bipolar cells connect to either rods or cones, never both. On-bipolar cells become active and fire in light, while off-bipolar cells stop firing in bright conditions.

  - Next, ganglion cells form the first receptive fields by combining signals from several bipolar cells. These ganglion cells act like edge detectors, comparing the light level in a central region with that in the surrounding area. On-center ganglion cells fire when the center is bright and the surround is dark. Off-center ganglion cells fire when the center is dark and the surround is bright.

Additional retinal cell types, such as horizontal cells and amacrine cells, act as inhibitors to sharpen contrast. However, this enhanced contrast can sometimes cause dark and light boundaries to appear exaggerated, either under- or oversaturated. This effect, known as lateral inhibition, provides negative feedback to neighboring cells to amplify the difference between strong and weak signals. While this improves contrast perception, it can also create after-images, where visual impressions persist even after the original stimulus is gone.

Bipolar cells can connect to many Ganglion Cells

Different Ganglion Cells at work for their receptive field


The Lateral Geniculate Nucleus (LGN) performs receptive field functions similar to retinal ganglion cells and receives strong feedback from the visual cortex. It separates the two visual fields: the left visual field is processed by the right side of the brain and the right visual field by the left. The LGN also combines input from both eyes. Its first two layers emphasize rod-based signals for movement and contrast, while the next four layers emphasize cone-based signals for color and finer details of form.

The Primary Visual Cortex (V1), detects edges and orientations. Some V1 neurons respond only when a pattern appears in a specific position, while others respond to the same pattern regardless of position. Neurons in the visual cortex fire when particular patterns appear in their receptive fields. Lower levels of the system process simpler patterns. Higher levels handle more complex patterns, such as face detection. The information stream then follows two paths to those higher levels.

  - The Ventral Stream handles form recognition and object representation and is connected to long-term memory.

  - The Dorsal Stream processes motion and object location and also helps coordinate eye and arm movements, such as reaching for an object.

Cortical magnification means the brain uses more neurons to process signals from the center of the visual field, especially the fovea. Although the fovea is small on the retina, it takes up a much larger area in the visual cortex. This is because the fovea has a high density of cones, which provide detailed, color-rich vision. The brain magnifies that input so we can see fine detail, such as reading or recognizing faces.

The human visual system and deep learning models for image classification share similarities and differences. Both detect features, process information in layers, and learn from examples. The human system is biological, has limits on processing speed and scalability, and adapts well to varied visual input. Deep learning models are artificial, can process very large amounts of data quickly, and become highly efficient at specific classification tasks.

