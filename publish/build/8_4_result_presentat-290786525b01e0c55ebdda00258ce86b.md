# Result Presentation

After the query is transformed, the retriever and reranker work together to produce a short list of relevant results. Instead of just showing that list, it is more effective to give a direct answer taken from or built from those results. This saves time and gives a clearer, more immediate response that better meets the user's needs. Choosing an extractive reader, a generative reader, or a synthesizer depends on the question, the available context, and the user's intent.

  - An extractive reader is best when you need exact facts from a few documents or short passages. It selects the exact words from the source, so its answers match the original text and are factual. For example, if the question is "Who won the 2025 Nobel Prize in Physics?", the extractive reader returns "John Clarke, Michel Devoret, and John Martinis". This method is fast, reliable, and especially suitable when accuracy is essential and the context is small.

  - A generative reader excels when a question requires combining information from multiple sources. For example, if a user asks "What contributions led to the 2025 Nobel Prize in Physics?" a generative reader can combine details from several documents and merge facts about the winners' research into a clear, natural answer. This makes it a good choice for complex, open-ended questions or to provide a concise, fluent summary. However, that flexibility can lead the model to hallucinate, inventing details not present in the sources, so it needs careful control.

  - The synthesizer keeps the generative reader's strengths and adds a multi-step retrieval process. It is best for deep research or when an answer must be built from several layers of information. For example, answering "What teams and organizations were behind the 2025 Nobel-winning quantum computing research?" might require first identifying the winners, then finding their research areas, and finally tracing their affiliations. This step by step approach improves accuracy and completeness but increases complexity and processing time.

A system deciding between these approaches can leverage several factors:

  - Context Size and Quality: Extractive readers work best with small, highly relevant passages. Generative readers or synthesizers are better suited to broad, diverse information.

  - Question Complexity: Simple fact-based questions call for extractive methods. Complex, multi-faceted questions benefit from generative or synthesizing methods.

  - User Intent: If the user wants exact quotes or citations, an extractive reader is best. For explanations or summaries, a generative reader works better. For thorough research, the synthesizer works best.

  - Risk Tolerance: When exact factual accuracy is essential, as in legal or medical questions, extractive readers reduce the risk of hallucination. When interpretation is important, generative readers provide value despite some risk.

In practical systems, a specially trained model evaluates the query, chooses the best presentation strategy, and gives a plan based on the pipeline blueprint shown at the start of this chapter.

Guardrails are built into the pipeline to make sure answers from the generative reader and synthesizer stay accurate, safe, and relevant to their area of use. Instead of generating responses from everything the model has learned, these systems are guided to rely only on documents retrieved from approved sources. This matters because generative models can produce convincing but incorrect information, a problem known as hallucination.

  - To prevent this, guardrails make sure answers are based on retrieved evidence and stay within the application's scope. For example, if a user asks about insurance policies, the system must use only information from that specific insurance company's products, not from competitors or from general knowledge the model learned during training. This is enforced with prompt constraints, filtered retrieval sources, and checks that compare the generated answer to trusted documents.

  - In a synthesizer setup where multiple retrieval steps build a more complete answer, guardrails are even more critical. At each step the system checks that new information matches the domain and the user's original intent. If a reliable answer cannot be found, the system will indicate uncertainty instead of producing incorrect content.

An extractive reader based on BERT finds the exact span in a passage that answers a question. It does not generate new text. Instead, it predicts the start and end positions of the answer in the context. Below are the steps it follows when using models such as DistilBERT or RoBERTa trained for this task:

  - Input Format: The question and passage are joined into one sequence before being fed into BERT (see figure on the right). [CLS] marks the start of the input. [SEP] separates the question and the passage. This arrangement lets BERT apply self-attention across both the question and the passage.

  - Encoding: BERT processes the combined sequence with its transformer encoder. Like the bi-encoder, it produces a contextual embedding that captures each token's meaning relative to all other tokens in the sequence.

  - Prediction Head: On top of BERT, a simple prediction head assigns each token in the passage a start score and an end score. These scores indicate how likely the token is to be the beginning or the end of the answer. The final answer is the span with the highest sum of start and end probabilities.

  - Training with SQuAD (Supervised Fine-Tuning): The model is fine-tuned on datasets such as SQuAD (Stanford Question Answering Dataset). For each question and passage pair, SQuAD provides the correct start and end token positions. During training the model learns to reduce the error between its predicted positions and the true positions.

  - Inference: The model receives the question and the passage. The prediction head produces start and end scores for each token. The model extracts the answer by selecting the token span with the highest probability.

[CLS]Question tokens [SEP] Passage tokens [SEP]

BERT

start

end

prediction head

Passage

from transformers import pipeline

# Load an extractive question-answering pipeline

qa_pipeline = pipeline("question-answering",

model="distilbert-base-uncased-distilled-squad")

# Define the question and context

question = "Who developed the theory of relativity?"

context = """

Albert Einstein developed the theory of relativity in the    early 20th century. It transformed our understanding of    space, time, and gravity.

"""

# Run the extractive reader

result = qa_pipeline(question=question, context=context)

print(result)

⮡ {'score': 0.9970649480819702, 'start': 0, 'end': 15, 'answer': 'Albert Einstein'}

span relevant to question

A generative reader uses a large language model such as GPT or Qwen to create new text rather than extract it from the passage. After receiving the user's question and the retrieved documents, the model generates the answer one token at a time, based on patterns learned during training. This lets it produce natural, coherent responses instead of copying exact text spans.

Effective prompt engineering is key to improving RAG systems. Clear prompts guide the model to stay grounded in retrieved documents, answer within the intended domain, and avoid hallucinations:

  - Clear Task Instruction: Explain what the model should do: answer a question, summarize, or explain.

  - Context or Retrieved Evidence: Include the relevant passages or documents after the instruction. The model should treat these as the only source of truth.

  - Grounding Reminder: Explicitly tell the model to base its answer only on the provided context and not on prior knowledge or assumptions.

  - Domain or Scope Restriction: Useful for business or specialized applications. It keeps the model from drifting into unrelated knowledge.

  - Output Format Specification: Specify how the answer should be structured: short answer, paragraph, bullet list, JSON, or other.

  - Tone or Style Guidance (Optional): If needed, specify formality, voice, or audience.

  - Uncertainty Handling: Tell the model what to do if the answer is unclear or missing.

Implementing a synthesizer means coordinating multiple components in an iterative retrieval and reasoning loop so the final answer is supported and coherent. Here is how this is typically done:

  - The user’s question is first transformed or expanded to optimize retrieval.

  - A retriever/reranker fetches a set of relevant documents or passages based on the original query.

  - A reader or reasoning module processes these documents to extract important entities, concepts, or partial answers that can guide the next retrieval step.

  - The system uses the extracted information to create refined or follow-up queries that retrieve additional documents providing context or details missing from the first pass.

  - After collecting several document sets, a generative model combines the information, merges facts, and clears up contradictions to produce a coherent answer. Prompt engineering often guides the model to base its answers on the retrieved evidence.

  - Apply guardrails throughout the process to ensure the model uses only supported information. Detect and reject hallucinated or unsupported claims. Handle uncertainty gracefully, and when evidence is insufficient say I do not know.

  - Finally, the synthesizer produces the answer in the required format, matching the user's intent and the domain constraints.

Citations: The ability to provide citations is a key strength of Retrieval-Augmented Generation (RAG) systems compared to large language models alone. By linking answers to source documents, citation-aware RAG systems deliver more transparent, accurate, and trustworthy responses.

  - The citation process is integrated into the retrieval and reading pipeline to preserve source information. During retrieval, when relevant documents or passages are found, their metadata is kept. This metadata typically includes a unique passage ID for internal tracking, a source document identifier, the document URL or physical location, a retrieval relevance score, and the passage's original text.

  - As the reader component processes retrieved passages, either by extracting exact text spans or by generating synthesized answers, the system records which passages contributed to the final response. This tracing is essential for creating an auditable link between the answer and the underlying evidence.

  - The final output of a citation-aware RAG system usually includes the answer plus structured citation details (e.g., JSON format). For example, an answer naming the 2025 Nobel Prize in Physics laureates would include a citation to the official Nobel Prize website, the exact passage used, and the access date.

  - This architecture offers several key benefits. First, it enables source verification. Users can follow citations to confirm an answer and read the original material themselves. Second, citation transparency helps build user trust because answers are shown to be based on verifiable evidence rather than unsupported claims. Third, this traceability lets developers monitor and refine the retrieval and reading components step by step, so the system becomes more accurate and reliable over time.

        - {

        - "answer": "John Clarke, Michel H. Devoret, and John M. Martinis",

        - "confidence": 0.95,

        - "citation": {

        - "passage_id": "passage_001",

        - "source": "Nobel Prize Official Website",

        - "url": "https://nobelprize.org/prizes/physics/2025/",

        - "original_text": "The 2025 Nobel Prize in Physics was awarded to John Clarke,

        - Michel H. Devoret, and John M. Martinis for their groundbreaking

        - discovery of macroscopic quantum mechanical tunneling and energy

        - quantization in an electric circuit.",

        - "retrieved_date": "2025-10-07"

        - }

        - }
