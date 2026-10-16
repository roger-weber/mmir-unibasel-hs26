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
        tokens = re.findall(r"[a-z]+", text.lower())
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if self.stemming:
            tokens = [_stem(t) for t in tokens]
        elif self.lemmatization:
            tokens = [_lemmatize(_lemmatize(t, "v"), "n") for t in tokens]
        return tokens

    def train(self, data: list[tuple[str, str]]) -> "IntentClassifier":
        """Estimate class and token counts from (utterance, label) pairs."""
        self.class_docs = Counter()
        self.token_counts = defaultdict(Counter)
        self.class_total = Counter()
        self.vocab = set()
        for text, label in data:
            self.class_docs[label] += 1
            for token in self.extract_features(text):
                self.token_counts[label][token] += 1
                self.class_total[label] += 1
                self.vocab.add(token)
        self.vocab_size = len(self.vocab)
        self.n_docs = sum(self.class_docs.values())
        self.classes = sorted(self.class_docs)
        return self

    def predict(self, text: str) -> str:
        """Return the most probable class for an utterance (log-space Naive Bayes)."""
        features = self.extract_features(text)
        best, best_score = None, None
        for c in self.classes:
            if self.prior == "observed":
                score = math.log(self.class_docs[c] / self.n_docs)
            else:
                score = 0.0
            denom = self.class_total[c] + self.vocab_size
            counts = self.token_counts[c]
            for token in features:
                score += math.log((counts[token] + 1) / denom)
            if best_score is None or score > best_score:
                best_score, best = score, c
        return best
