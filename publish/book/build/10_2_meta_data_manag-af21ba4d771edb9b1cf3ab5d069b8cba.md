# Meta Data Management

We have previously discussed the concept of the “semantic gap” and its challenges. Let’s look at an example in the area of image retrieval and consider the picture below of the Tay Mahal as a running example:

  - The context of the picture is as follows:

    - The Taj Mahal, located in Agra, India, is a magnificent mausoleum built by Emperor Shah Jahan in memory of his beloved wife Mumtaz Mahal, who passed away in 1631. The Taj Mahal is one of the most iconic landmarks in the world and is recognized as a UNESCO World Heritage Site. Mumtaz Mahal's tomb is situated in the main chamber, alongside Shah Jahan's tomb.

  - When users search for pictures of the Taj Mahal, they may use various keywords, as depicted in the lower right box. The image retrieval system must then find matches to these search queries within its image database. However, it faces a challenge as it cannot directly compare the pixel information of images with the keywords provided by the users. Unlike text retrieval, this disparity requires other methods to bridge the semantic gap.

  - To address this gap, we need to map the distinct perspectives (pixels in images and user-provided keywords) into a comparable space where we can more effectively assess the relevance. In this introductory chapter, we focus on meta data to bridge the semantic gap.

This is what the machine ‘sees’ when trying to understand what Is depicted on the image.

This is what a user may enter to search for such pictures:

building, outdoor, sky, iconic

mausoleum, tomb, dome, minaret

UNESCO World Heritage Site

Taj Mahal, Indian architecture

where is Mumtaz Mahal buried?

semantic gap

Meta data refers to additional information associated with a source document, providing context, descriptions, or annotations. In our ongoing example, we can enrich the image of the Taj Mahal with various textual metadata elements, as presented on the following page. These textual pieces allow the retrieval system to find relevant images with text retrieval methods. For example, if the image’s description metadata includes the keywords “Taj” and “Mahal”, we can directly match it with user queries such as “Taj Mahal”.

However, it is not as easy as it seems. Firstly, we need to gather metadata for the images in the database. How?

  - Manual annotations: human workers inspect each image and provide context, descriptions, categories, tags and other metadata items for the image

  - Automated annotations: technical metadata, such as geo-location, can be captured at the time of taking the image. Additionally, AI workers can analyze an image and extract pre-learned annotations. If the image is embedded in a broader context, such as a web page, that context can yield more information about the image

  - Generative AI: the latest multi-modal transformer models can extract relevant information from a wide range of images, providing high-quality metadata and text descriptions.

Secondly, annotations obtained by two different workers, whether human or AI, can semantically differ from each other disabling a direct matching approach. Let’s consider an example:

  - Worker A adds the following keywords: Taj Mahal, India, iconic building, 17th century

  - Worker B adds the following keywords: religious building, great weather, few people outside, nice

  - Both workers have provided accurate annotations, but they differ in semantic levels. When comparing the phrases “Taj Mahal” with “religious building”, a retrieval system must consider the relationships between words. In this case, “Taj Mahal” is a more specific term used by workers familiar with the building, while workers who have never seen the Taj Mahal (or an AI not trained to recognize the building) or lack context may opt for the more generic phrase “religious building”. Similar relationships appear almost everywhere in natural language: horse ↔ mammal, Matterhorn ↔ mountain, Italy ↔ Europe. As we learned previously, we can address these relationships with natural language processing and the more recent development of embeddings

  - Worker B’s use of the keyword “nice” expresses a subjective and abstract concept. Obtaining and normalizing such abstract concepts, where two individuals would agree upon them, can be challenging. However, if we can match these abstract concepts with user preferences, we can provide more relevant examples for the user’s queries. For example, if a user plans a visit to India and searches for sites with “great” architecture, both “India” and “great” describe abstract concepts which are not present in the pixels alone, but are obtainable from metadata.

Let’s annotate our ongoing example, the picture of the Taj Mahal. In essence, we can consider three types of metadata: generic, specific, and abstract. Furthermore, we can annotate images across various facets as illustrated below. It’s worth noting that we can extract certain metadata like “outside”, “time/date taken”, “sky”, or “building” directly from the raw pixel information, regardless of whether a human or AI worker performs the task. However, other information such as “UNESCO”, “Mumtaz”, “1648” or even “Taj Mahal” requires a human or AI worker who possesses contextual awareness since these details cannot be derived from the pixels alone.
## 10.2.1 Manual Metadata Creation


The process of creating metadata using human workers has gained popularity on various machine learning platforms. In supervised learning, labeled data is required to train models, and these labels at the same time generate metadata for images. For instance, Amazon Mechanical Turk offers access to over 500,000 independent contractors who can perform well-defined tasks at specified prices, as depicted in the example below. Similarly, ChatGPT was trained with the help of thousands of workers to assess the quality of answers generated by the AI.

Annotation or labeling tasks typically cost around $1 and upwards, depending on the complexity of the task and the required domain knowledge. For basic tasks in machine learning, generic labels and descriptions often suffice. However, annotating stock images or categorizing items in a media archive demands more specific labels and extensive domain knowledge, leading to higher annotation costs. By leveraging a global workforce, annotation tasks can be scaled to millions of items at reasonable costs, yielding results within a reasonable time frame. We will see a few examples in one of the upcoming pages.

In the case of machine learning, the initial investment in training a model can subsequently produce automated labels, as we will explore further in this course.

The quality and substance of manually created labels can greatly vary depending on the domain expertise of the human workers. In the examples provided below, we can observe two distinct approaches to annotating a picture of the Taj Mahal. On the left side, we have the results of a more generic and concise labeling task, whereas the right side shows a comprehensive analysis and description that demonstrates deep domain knowledge.

This serves as a good example for the challenges when dealing with manually created metadata such as variations in level of detail, choice of keywords, and depth of domain knowledge. The annotations on the left side may not provide sufficient information to boost the image for queries of the Taj Mahal, while the annotations on the right side are so detailed that they are less likely to align with typical queries for the Taj Mahal.


Stock photo services and media company archives maintain concise keyword lists for each image. They also utilize "faceted navigation" which involves categorizing images based on various attributes with pre-defined values such as prominent individuals, locations, brands, or time periods like decades. For instance, sports event photos are examined to identify shots featuring known individuals. Only a limited number of selected shots from each event are annotated for faceted navigation to keep the overall number manageable. This allows users, like journalists, to easily browse through a curated list instead of scanning thousands of pictures when they need an image of a prominent person. However, one drawback of this approach is that acquiring pictures of individuals before they gain prominence is challenging and often relies on lucky discoveries or contributions from the individuals themselves or their entourage.

Roger Federer

John McEnroe

DarleneHard

need a picture of Darlene Hard for an obituary in the New York Times

not yet famous

nice shot of Roger Federer






In domains with a well curated set of items, such as songs or movies, metadata annotations with quality control and consistent structure are available. IMDb is an example of such a database that holds records for movies and episodes from various producers. Each item is annotated with predefined attributes and has relationships with other items. The database is curated by volunteers, actors, crews, and industry executives, and is accessible online in compiled formats. As such, this is an excellent illustration of “scaled-out” metadata gathering.

MusicBrainz is another good example for a community-maintained open-source encyclopedia of music information. It provides details about artists, albums, songs, and releases. When combined with a lyrics database, music search benefits from a wide range of textual features and factual data that are not obtainable from the raw audio alone.

With such curated databases, retrieval of information greatly benefits from high-quality annotations. Often, the metadata alone is sufficient to bridge the semantic gap, meaning that the audio data itself is only used in rare cases, such as with Shazam, to retrieve information about the currently playing song. Due to the commercial and community interest in these domains, the additional efforts involved in creating and maintaining the metadata are covered by increased revenues.

lyrics

## 10.2.2 Automated Metadata Extraction


Annotating arbitrary photos and videos raises challenges due to the absence of a curated reference database for readily obtaining metadata. The costs associated with annotating every single photo and video would far outweigh the added value of having metadata (unless you do it for your photos and videos as fun activity after vacations).

In such cases, automated annotations and AI-based metadata extraction provide valuable support for retrieval systems. The following examples illustrate the extraction of metadata at different semantic levels:

  - Perception level (left side, lower part): The signal information is processed to capture key aspects that enable comparisons between items based on how humans interpret the signals. This course will provide extensive examples covering various types of multimedia items.

  - Structural level (middle to right side): Machine learning methods analyze the signal information and its context to extract pre-trained metadata items that can be generic, specific, or abstract. Examples include generic object recognition (architecture, person, female, outdoors), specific object recognition (Brie Larson), or abstract concepts that a typical human would recognize (fun, age, happy).

perception

recognition



When documents are embedded on the web, there is a simple yet powerful approach to extracting context, relationship information, and textual metadata:

  - HTML tags such as <a> or <img> include special attributes that provide descriptions or short annotations for the referenced objects. These attributes can be extracted and used as metadata.

  - The surrounding area on the web page, including title information, text blocks, and captions, can serve as another valuable source of keywords that are likely to overlap with the context of the embedded object. While not always a perfect match, this source often provides sufficiently valuable information and is easily obtainable.

In the early days of the web, the "surrounding area" referred to the immediate vicinity within the HTML source code. By considering a window of a few dozen tokens before and after the embedded object, most of the relevant keywords could be captured. Additionally, the header sections (<h1>, <h2>) and title of the web page were useful sources. However, modern web applications utilize advanced scripts and CSS styles that dynamically change data and layout, making the direct neighborhood within the HTML less reliable for capturing relevant keywords. As shown in the illustration on the right-hand side, the distance between the image and the text paragraph can be large in terms of both text position and hierarchical position due to CSS instructions.

Taj Mahal

Advanced web-based metadata extraction considers the visual proximity between embedded objects and text blocks, even though it comes with higher extraction costs. Here's how it works:

  - The web page is rendered in a browser and we identify all objects and text elements of interest

  - Each DOM element has a bounding box, accessible through the getBoundingClientRect method which provides on-screen distances between objects

  - We can scan for visual, CSS, or textual cues to eliminate or weigh down text blocks that are not directly relevant such as sidebars or other articles

  - Distances and cues provide proximity weights for the keywords in text blocks that we can use to describe the context of the embedded object

Visual boundary between the two columns

Contains many of the keywords as we discussed earlier in this chapter

function getPositionAtCenter(element) {

const {top, left, width, height} =

element.getBoundingClientRect();

return {

x: left + width / 2,

y: top + height / 2

}

}

function getDistanceBetweenElements(a, b) {

const apos = getPositionAtCenter(a);

const bpos = getPositionAtCenter(b);

return Math.hypot(apos.x - bpos.x,

apos.y - bpos.y)

}

getDistanceBetweenElements(image, text)
Multi-modal transformer models can merge text and image data:

  - The encoder model can map text and images into the same conceptual space, making it possible to directly compare the two representations.

  - The decoder model can use encoded image data and a prompt to create text that matches the image.

  - By using various prompts, we can gather different information and adjust the meta-data we want to create and how we want it to look.

Example: Picture of the iconic Matterhorn

  - We start by using a prompt to create a detailed description of what can be seen in the image.

  - Then, we use a prompt to create a JSON document that includes various features extracted from the image content.

Note: You can create better prompts by using meta-prompts, which generate the prompt based on your basic specifications.

The main subject of this image is a majestic, snow-covered mountain peak, which appears to be the iconic Matterhorn in the Swiss Alps. The mountain dominates the frame, its distinctive pyramid shape rising dramatically against a clear blue sky. The setting is a high-altitude alpine environment, with the peak surrounded by other snow-capped mountains and glaciers visible in the lower portions of the image. The background is primarily composed of a vivid blue sky with a few wispy clouds. The colors in the image are striking, with the brilliant white of the snow contrasting sharply against the deep blue of the sky. The lighting appears to be natural sunlight, creating a play …

Prompt: Please provide a detailed description of the image you see. Include information about:- The main subject or focus of the image- The setting or background- Colors and lighting

- ...

{

"headline": "Majestic Matterhorn Pierces Blue Sky",

"keywords": ["mountain", "peak", "snow", … ],

"named_entities": ["Matterhorn"],

"dominant_colors": ["blue", "white", "gray"],

"dominant_forms_shapes": ["triangle", "pyramid"],

"extracted_text": "",

"people_present": false,

"is_selfie": false,

"is_outdoors": true,

"is_text_image": false

}

Prompt: Analyze the given image and provide a JSON output with the following information:

{...definition omitted here...}Please ensure that: 1) The "headline" is concise and accurately represents the image's main subject, 2) "keywords" include relevant terms describing the ...
The trustworthiness of metadata is a subject of concern, as highlighted by Cory Doctorow's "seven insurmountable obstacles" to achieving a meta-utopia. These obstacles include:

  - People lie: Unscrupulous content creators may publish misleading or dishonest metadata to redirect traffic

  - People are lazy: Many content publishers lack the motivation to thoroughly annotate their published content

  - People are stupid: Not all content publishers possess the necessary intelligence to effectively catalog their produced content

  - Mission impossible—know thyself: Inadvertently misleading metadata can be published by content creators

  - Schemas aren't neutral: Classification schemes are subjective and can introduce biases

  - Metrics influence results: Competing metadata standards bodies may never reach an agreement

  - More than one way to describe it: Resource description is subjective, and different perspectives exist.

With generative AI, we can add an 8th law that involves extracting keywords and meta-data.

  - Models hallucinate: Hallucinations happen when a language model predicts words that sound plausible but are not true. Because it selects the most likely next word from patterns rather than from facts, it can give incorrect information with confidence.

However, we should not disregard metadata entirely. Instead, it is important to exercise caution and carefully evaluate the information it provides. High-quality metadata, as seen in platforms like IMDb and MusicBrainz, can be exceptionally valuable. Observational metadata obtained through web crawling can also be beneficial, especially when the system is designed to resist manipulation. For example, the Google web search engine gives higher importance to anchor texts provided by others linking to a page rather than relying solely on the keywords provided by the content owner. However, even these advanced approaches can potentially be manipulated as was successfully demonstrated with the so-called Google-bomb.
