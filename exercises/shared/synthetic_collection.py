"""
Synthetic document collections for retrieval demos.

Usage:
    from shared.synthetic_collection import MINI, documents, text, ids
"""

MINI = {
    # Animal adventures and lexical variants
    "b1": "The Cat and Dog in the Forest. A cat and a dog begin a woodland adventure.",
    "b2": "Cats of the Woodland. Wild cats begin an adventure beneath ancient trees.",
    "b3": "The Forest Hound. A loyal dog follows a woodland trail through ancient trees.",
    "b4": "Feline Detective. A clever cat solves mysteries with a canine companion.",

    # Ambiguous uses of forest
    "b5": "Random Forest for Pet Detection. A random forest model classifies cats and dogs in photographs.",
    "b6": "Forests of Search Trees. Algorithms explore a forest of binary trees and graph paths.",

    # Repetition for term-frequency saturation
    "b7": "Dog Dog Dog! A dog chases a ball, finds a bone, and wakes the neighbors.",
    "b8": "Forest Forest Forest! A forest guide names trees, flowers, rivers, birds, and hidden ruins.",

    # Matched terms at contrasting document lengths
    "b9": "Cat Dog Forest.",
    "b10": "Cat Dog Forest. A cat meets a dog in a forest beside rivers, mountains, castles, villages, bridges, and caves.",

    # Synonyms and a topical distractor
    "b11": "Woodland Companions. A kitten and a puppy share a moonlit adventure among old trees.",
    "b12": "The Pet Bakery. A cat, a dog, and a baker make cakes, bread, biscuits, pies, and coffee.",
}


def documents(collection=None):
    """Iterate (doc_id, text) pairs."""
    return list((collection or MINI).items())


def text(doc_id, collection=None):
    """Look up document text by ID."""
    return (collection or MINI)[doc_id]


def ids(collection=None):
    """All document IDs."""
    return list((collection or MINI).keys())
