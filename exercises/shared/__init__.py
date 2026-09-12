"""
Shared utilities for Multimedia Retrieval demos.

Import from any notebook in the demos/ folder:

    # Collections
    from shared.synthetic_collection import MINI, documents, text, ids
    from shared.library_collection import (
        LIBRARY, NEED, GRADES, RUNS,
        doc2text, texts, record, grade, is_relevant, relevant_ids
    )

    # Text processing
    from shared.text import tokenize, remove_stopwords, stem_tokens, pipeline, bag_of_words

    # Retrieval models
    from shared.retrieval import (
        document_frequencies, vocabulary, tfidf_vector,
        cosine_similarity, bm25_score, boolean_and, boolean_or,
        rank_collection_vsm, rank_collection_bm25
    )

    # Display
    from shared.display import print_table, display_md, print_wrapped

    # LLM
    from shared.llm import invoke_claude, AWSConnectionError

    # PDF
    from shared.pdf import extract_text_from_pdf, get_pdf_documents

    # Data
    from shared.data import load_gutenberg_book, get_cached_book_ids

    # Visualization
    from shared.plot import plot_pr_curve, plot_venn_diagram
"""
