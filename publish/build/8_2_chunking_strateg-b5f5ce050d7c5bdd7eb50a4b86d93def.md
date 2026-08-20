# Chunking Strategies

Classical information retrieval used chunking to improve search. Long documents were split into smaller sections so search engines could find where a query matched and make sure the search terms were near each other. Instead of returning the whole document, retrieval systems showed the most relevant passage, giving the user both location and proximity context.

When Retrieval-Augmented Generation (RAG) first appeared, it used chunking for a practical reason: limits in both embedding models and language models. Early embedding models could handle only short pieces of text, often a few hundred tokens, before losing semantic clarity. At the same time, early language models had small context windows and could not accept long documents as input. For these reasons, splitting documents into manageable chunks was essential to enable retrieval and generation.

Today, chunking is no longer just a workaround for technical limits but a deliberate design choice that needs balance. Larger chunks give the language model more context and help it reason and produce accurate answers. But embeddings work best with shorter segments that preserve precise meaning and avoid blurring. Research and practice show that about 512-1024 tokens form the sweet spot: large enough for coherent ideas but small enough to yield high-quality embeddings.

The best chunk size depends on the task. For factual lookups or direct question answering, shorter chunks often work because the answer is usually in a small piece of text. For summarizing, extracting lists (for example, common retrieval models), or combining ideas across a document, you need longer context. If the system only retrieves short fragments, the model can miss important links between ideas.

Newer retrieval designs use hierarchical chunking. Documents are stored at multiple levels. Small, fine-grained chunks produce dense, accurate embeddings for retrieval. Larger sections or merged chunks give the language model enough context to reason and compose an answer. Additionally, newer methods use query-aware chunk selection. The system adjusts how much and what kind of context it sends to the language model based on the query. For a precise factual question, it sends only the most relevant small chunk. For a summary request or a question spanning several topics, it gathers and sends a much larger section or multiple related chunks.

Many frameworks include built-in chunking features, and LangChain is one of the most popular. LangChain lets developers choose simple fixed-size token or character chunking, recursive chunking that follows document structure, or overlap-based chunking to keep continuity between segments.

Method 1: Splitting the text into fixed-sized chunks

  - A simple chunking method divides a document into fixed-size pieces. First split the text into words, then group words until each chunk reaches the set size. To reduce cutting through sentences or paragraphs, add overlaps between chunks. If a sentence or paragraph is split, most of it will still appear in the next chunk, preserving context.

  - The langchain library has a convenient function for this kind of chunking:

      - from langchain_text_splitters import CharacterTextSplittertext_splitter = CharacterTextSplitter(

        - separator = " ",

        - chunk_size = 1000,

        - chunk_overlap  = 200,

        - add_start_index = True

      - )	text_splitter.split_text("…a very long text…")

  - In this example:

    - separator=" " tells the splitter to break the text along spaces (word boundaries).

    - chunk_size=1000 sets the maximum length of each chunk in characters.

    - chunk_overlap=200 defines an overlap of about 200 characters between subsequent chunks.

    - add_start_index=True stores the starting position of each chunk in the original document.

  - When this splitter is used on a sample document, the chunk length distribution shows most chunks are close to the 1,000-character limit. Small variations arise because the splitter avoids breaking words. The single short outlier is the final chunk, which contains the remaining text.

  - This method is practical and easy to implement. It works well when rough locality is enough. However, it has a major drawback: chunks do not line up with meaningful boundaries. A new topic, paragraph, or section can be split across two chunks or mixed with the end of an unrelated one. Retrieval systems may return passages that lack coherent context, and embeddings can merge unrelated concepts.

A Study in Scarlet
Method 2: Splitting at sentence boundaries

  - This approach is like fixed-size chunking, except that sentences, not raw character counts or word groups, are the smallest unit of text. By combining full sentences into chunks until a size limit is reached, this method avoids awkward breaks in the middle of sentences while still producing manageable chunks. Chunk sizes can vary slightly as a result, but the difference is usually negligible.

  - Sentence boundary detection may sound trivial, but it’s more complex than simply splitting at every period. To handle this accurately, libraries provide robust models for sentence tokenization:

    - NLTK’s Punkt Sentence Tokenizer: Punkt uses an unsupervised algorithm that learns how sentences start and end from a training corpus in the target language. It builds internal models of abbreviations, common sentence starters, and frequent boundary patterns. NLTK includes a pre-trained English Punkt model, and you can train custom models for other languages. It is fast and usually very accurate, but it can struggle with complex narrative dialogue where punctuation appears both inside and outside quotation marks.

    - spaCy Sentence Segmentation: spaCy uses machine learning and deep linguistic parsing to detect sentence boundaries. It supports many languages built in and handles complex sentence structures more consistently than rule-based systems. It is heavier than Punkt, requiring more CPU and GPU resources, but it is well suited for production systems where accuracy and extensibility matter.

  - LangChain supports sentence-aware text chunking using both NLTK and spaCy:

      - from langchain_text_splitters import SpacyTextSplitter, NLTKTextSplittertext_splitter = SpacyTextSplitter(    # or: NLTKTextSplitter

        - separator = " ",

        - chunk_size = 1000,

        - chunk_overlap  = 200,

        - add_start_index = True

      - )	text_splitter.split_text("…a very long text…")

  - Sentence-based chunking balances structure and simplicity. Compared with raw character splitting, it produces more coherent chunks that map naturally to meaning. However, sentence boundaries do not always match topic shifts or logical sections, so this method can still mix unrelated ideas within a chunk.

Method 3: splitting on structure

  - The concept is to divide text based on its structural elements. For instance, in books, we can split at pages, parts, chapters, sections, and paragraphs. Authors commonly use these structural elements to separate content, making them strong indicators of topic or aspect changes.

  - Detecting these structural elements relies on the document's format. In our running example, we utilized text from Project Gutenberg, where paragraphs, chapters, and parts are structured with increasing numbers of newlines before their start. For instance, 4 consecutive newlines indicate chapters, and 2 consecutive newlines indicate paragraph breaks. In HTML or Markdown formats, we can split by identifying headers (<h1>, <h2>, …), while treating the text within <p> and <div> tags as paragraphs.

      - from langchain_text_splitters import RecursiveCharacterTextSplittertext_splitter = RecursiveCharacterTextSplitter(

        - separators = ["\n\n"],    # use 4x\n for chapters, and 2x\n for paragraphs

        - chunk_size = 100,

        - chunk_overlap  = 20,

        - add_start_index = True

      - )	text_splitter.split_text("…a very long text…")

  - The plot on the right displays the distribution of chunk lengths obtained by applying the NLTK splitter to the sample document ("A Study in Scarlet"). The results for spaCy are very similar:

    - Most chunks are between 800 and 1000 characters long, with variations resulting from sentence lengths causing splits. Long sentences can result in much higher variance than the simple word splitting approach.

    - The left outlier represents the final chunk of the document, which is generally smaller.

    - The approach is a bit better than word-based splitting, however, we still have a misalignment between chunk size and semantic changes in the document.

A Study in Scarlet
  - To prevent adjacent paragraphs or chapters from merging, make the chunk size and overlap small enough that each chunk holds only one paragraph or one chapter. This splitting approach then produces separate chunks for every paragraph or chapter, depending on the separator you choose.

  - The figures on the right show chunk size distributions for chapters and for paragraphs, which contrast strongly with the word- and sentence-based approaches. Chunks now follow chapter or paragraph boundaries and give more accurate semantic coherence than the fixed-size splits used before.

  - Unlike earlier splitting methods, the chunks now vary in size. For chapters, small chunks can come from formatting at the start of the text or from parts that can be ignored. These coarse splits usually yield coherent pieces, but their sizes are hard to control and can be too large for embedding and language models, for example when several chapters are added to the RAG context.

  - Splitting by paragraphs (lower figure on the right) lets us control chunk size better. Most chunks stay below 1,500 characters, although this varies by author. But this method creates many very small chunks. This happens because the book is written as dialogue and different speakers are separated by blank lines. Those tiny chunks are not good for generating responses because they provide too little context and information to extract or synthesize an answer. So in this case, we want to merge the smaller chunks again.

A Study in Scarlet

A Study in Scarlet

  - To improve paragraph-based chunking, we increase chunk sizes and merge small paragraphs that are likely related. The example on the right shows this. Small chunks were removed, and most chunks are now between 800 and 1,000 characters, though some are longer because of the author's preference for long paragraphs. For excessively long paragraphs, we can split them further with the RecursiveCharacterTextSplitter. It accepts a list of separators and can break oversized chunks into smaller ones in stages.

  - Trying to control overlap of paragraph-based chunks by setting a fixed number of characters is difficult. Whole paragraphs can be longer than the overlap limit, so there may be no overlap. A better approach is to create paragraph chunks as described earlier and merge any small ones. For each paragraph, add the last sentence from the previous chunk and the first sentence from the next chunk. These added sentences often capture the meaning of nearby paragraphs and give better descriptions and context.

A Study in Scarlet

A Study in Scarlet

Method 4: semantic splitting

  - Splitting text by document structure (section headers, paragraphs, etc.) often produces coherent chunks with little effort. However, structural metadata is not always available or reliable. For example, scraped web pages may have inconsistent formatting, PDFs can have broken layouts, and scanned documents often lack clear headings. In such cases, we can use semantic splitting: group sentences by meaning rather than by formatting.

  - The general approach builds on sentence-based splitting and extends it with semantic similarity:

    - Define a similarity measure between sentences

    - Specify minimum and maximum chunk sizes

    - Split the text into sentences using NLTK or spaCy

    - Merge neighboring sentences or chunks if they are semantically similar not exceeding the maximum chunk size

  - Sentence Similarity: A naive approach uses term-based vector space models. Sentences are converted into word-frequency vectors and compared with measures such as cosine similarity. A major flaw is that two sentences that express the same idea with different words can receive a misleadingly low similarity score. To fix this, we use a bi-encoder that encodes sentences into dense vectors and measures similarity with cosine similarity. A more advanced approach uses models that detect whether the next sentence belongs in the current chunk.

  - Merging Chunks Based on Similarity: Once sentence embeddings are generated, we can begin merging:

    - Start with each sentence (or sentence group) as its own chunk.

    - Compute similarities between adjacent chunks

    - Merge the top-k most similar neighboring chunk pairs as long as maximum chunk size is not exceeded

    - Use a relatively small k, often a fraction of the current number of chunks

    - Repeat the process iteratively. Over multiple passes, this encourages the formation of longer, semantically coherent chunks that reflect topic continuity

    - This iterative merging process clusters related sentences while preserving boundaries between distinct topics. It also allows the flexibility to control how granular or expansive the final chunks should be.

  - Semantic splitting yields higher-quality chunks than fixed-size or structural methods, especially in narrative or technical documents where topic flow matters more than surface formatting. It is more computationally expensive, but produces chunks that align with meaning and improve embeddings, retrieval accuracy, and downstream responses in RAG systems.

Method 5: hierarchical chunking (for RAG use cases)

  - Hierarchical or hybrid chunking uses multiple splitting methods so the strengths of some methods offset the weaknesses of others. Instead of one chunk size or rule, the document is split at several levels of detail, for example into sections, then paragraphs, then sentences or token-based chunks. This approach is especially useful for retrieval-augmented generation (RAG), because it gives high quality embeddings for retrieval and larger context windows for synthesis.

  - The core idea is simple:

    - Split the document using large structural units such as chapter titles, headings (H1, H2), or sections.

    - Within each section, apply a finer splitter such as sentence-based, fixed-token, or semantic splitting.

    - Store both levels of chunks:

      - Small chunks  optimized for embeddings and similarity search.

      - Large chunks or merged segments  provide richer context to LLMs during answer generation.

  - Here is an example of how LangChain supports this:

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

header_splitter = MarkdownHeaderTextSplitter(["#", "##", "###"])

sections = header_splitter.split_text(long_markdown_text)

token_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

final_chunks = []

for sec in sections:

sub_chunks = token_splitter.split_text(sec.page_content)

for idx, sc in enumerate(sub_chunks):

final_chunks.append({

"content": sc,

"metadata": {**sec.metadata, "sub_chunk": idx}

})

Method 6: Query-Aware or Adaptive Chunking

  - Most chunking strategies split documents before any query is made. This is called static chunking. Not every user query needs the same context. For example, "When was Albert Einstein born?" needs a short, fact-centered chunk, while "Summarize how vector search works" needs much more context. Query-aware chunking changes chunk size and content to match the query type and intent. Instead of returning the same chunk size for all queries, the system dynamically selects, expands, or combines chunks depending on how much context is needed.

  - How It Works:

    - Preprocess the document using small, embedding-friendly chunks (e.g., 512 tokens).

    - Interpret the query intent (factual, analytical, summarization, list generation, multi-hop reasoning).

    - Retrieve top-k relevant chunks using vector search or sparse retrieval.

    - Adapt chunk assembly before passing to the LLM:

      - Factual queries: return only the highest-scoring chunk or two.

      - Summarization/explanation queries: Combine several neighboring chunks or return entire sections.

      - Multi-hop questions: Collect related chunks from different sections and stitch them together with metadata or reasoning prompts.

  - This approach reduces noise for simple questions and gives enough context for complex ones. It also improves efficiency by sending fewer tokens to the LLM when they are not needed, and it increases accuracy for tasks that cover multiple parts of a document.

  - Frameworks like LangChain, LlamaIndex, and Haystack already use early versions of this idea, retrieving small chunks to score relevance and sending larger parent sections to the LLM when needed. As models gain larger context windows, query-aware chunking becomes less about overcoming limits and more about optimizing relevance, cost, and reasoning quality.
