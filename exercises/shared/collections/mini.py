"""
MINI collection — 12 hand-crafted documents for formula tracing.

Designed to expose: term overlap, repetition, length variation, polysemy,
synonyms, and lexical mismatch. Used as the running example throughout
Chapter 1 of the textbook.
"""

from shared.collections import Collection, register


MINI_DOCUMENTS = {
    "b1": "The Cat and Dog in the Forest. A cat and a dog begin a woodland adventure.",
    "b2": "Cats of the Woodland. Wild cats begin an adventure beneath ancient trees.",
    "b3": "The Forest Hound. A loyal dog follows a woodland trail through ancient trees.",
    "b4": "Feline Detective. A clever cat solves mysteries with a canine companion.",
    "b5": "Random Forest for Pet Detection. A random forest model classifies cats and dogs in photographs.",
    "b6": "Forests of Search Trees. Algorithms explore a forest of binary trees and graph paths.",
    "b7": "Dog Dog Dog! A dog chases a ball, finds a bone, and wakes the neighbors.",
    "b8": "Forest Forest Forest! A forest guide names trees, flowers, rivers, birds, and hidden ruins.",
    "b9": "Cat Dog Forest.",
    "b10": "Cat Dog Forest. A cat meets a dog in a forest beside rivers, mountains, castles, villages, bridges, and caves.",
    "b11": "Woodland Companions. A kitten and a puppy share a moonlit adventure among old trees.",
    "b12": "The Pet Bakery. A cat, a dog, and a baker make cakes, bread, biscuits, pies, and coffee.",
}


@register("mini")
class MiniCollection(Collection):
    """
    12 short documents — the textbook's running example.

    Each document is a single retrieval unit (no splitting needed).
    """

    description = "12 hand-crafted documents for formula tracing (Ch. 1 running example)"
    language = "en"

    def _load(self):
        for doc_id, text in MINI_DOCUMENTS.items():
            self._docs[doc_id] = {
                "id": doc_id,
                "source": "synthetic",
                "text": text,
            }
