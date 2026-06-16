"""TF-IDF cosine similarity + fuzzy string matching, as standalone scorers.

Both return, per query, the best candidate from `reference_names` and a
score in [0, 1] (fuzzy is rescaled from its native 0-100 range) so they can
be combined as features downstream instead of chained if/elif fallbacks.
"""
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CosineMatcher:
    """Nearest-neighbor matcher over a corpus of example texts.

    `corpus_texts` is what gets vectorized and compared against (e.g. messy
    training examples); `corpus_labels` is what gets returned for the best
    match (e.g. each example's canonical hospital name). Matching against
    many noisy *examples* instead of a deduped canonical list is what gives
    TF-IDF/fuzzy enough surface-form variety to work -- see
    docs/accuracy_analysis.md for why matching directly against canonical
    names collapses accuracy.
    """

    def __init__(self, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)

    def fit(self, corpus_texts, corpus_labels=None):
        self.corpus_texts = list(corpus_texts)
        self.corpus_labels = list(corpus_labels) if corpus_labels is not None else self.corpus_texts
        self.corpus_matrix = self.vectorizer.fit_transform(self.corpus_texts)
        return self

    def predict(self, queries, exclude_indices=None):
        query_matrix = self.vectorizer.transform(queries)
        sims = cosine_similarity(query_matrix, self.corpus_matrix)
        if exclude_indices is not None:
            for row, idx in enumerate(exclude_indices):
                if idx is not None:
                    sims[row, idx] = -1.0
        top_idx = sims.argmax(axis=1)
        top_score = sims.max(axis=1)
        preds = [self.corpus_labels[i] for i in top_idx]
        return preds, top_score


def fuzzy_predict(queries, corpus_texts, corpus_labels=None, scorer=fuzz.token_sort_ratio, exclude_indices=None):
    """exclude_indices: optional per-query corpus index to skip (leave-one-out
    calibration -- a query that's also a corpus row shouldn't trivially match itself)."""
    corpus_texts = list(corpus_texts)
    corpus_labels = list(corpus_labels) if corpus_labels is not None else corpus_texts
    if exclude_indices is None:
        exclude_indices = [None] * len(queries)
    preds, scores = [], []
    for q, exclude_idx in zip(queries, exclude_indices):
        if exclude_idx is None:
            _, score, idx = process.extractOne(q, corpus_texts, scorer=scorer)
        else:
            results = process.extract(q, corpus_texts, scorer=scorer, limit=2)
            _, score, idx = next(r for r in results if r[2] != exclude_idx)
        preds.append(corpus_labels[idx])
        scores.append(score / 100.0)
    return preds, np.array(scores)
