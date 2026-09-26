"""Ch03 helper - a Naive Bayes intent classifier for customer-support utterances.

You implement three things:
  - extract_features: turn an utterance into feature tokens (your text pipeline);
  - train: estimate the Naive Bayes counts from labelled utterances;
  - predict: return the most probable intent for a new utterance.

The feature options and the prior are constructor flags, so you can compare pipelines
without rewriting the class.
"""
import math
import re
from collections import Counter, defaultdict

from nltk.corpus import stopwords as _stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

_STOPWORDS = set(_stopwords.words("english"))
_stem = PorterStemmer().stem
_lemmatize = WordNetLemmatizer().lemmatize


class IntentClassifier:
    """Multinomial Naive Bayes over bag-of-words features extracted from utterances."""

    def __init__(self, remove_stopwords=False, stemming=False,
                 lemmatization=False, prior="uniform"):
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.prior = prior          # "uniform" or "observed"

    def extract_features(self, text: str) -> list[str]:
        """Turn one utterance into a list of feature tokens, honouring the flags."""
        # YOUR CODE HERE
        raise NotImplementedError

    def train(self, data: list[tuple[str, str]]) -> "IntentClassifier":
        """Estimate class and token counts from (utterance, label) pairs."""
        # YOUR CODE HERE
        raise NotImplementedError

    def predict(self, text: str) -> str:
        """Return the most probable class for an utterance (log-space Naive Bayes)."""
        # YOUR CODE HERE
        raise NotImplementedError
