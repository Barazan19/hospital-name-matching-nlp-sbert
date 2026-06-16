"""Reranking ensemble over TF-IDF cosine, fuzzy and SBERT candidates.

The original notebooks chained the three methods with a fixed if/elif
priority order (cosine -> fuzzy -> SBERT), so a method later in the chain
could never override an earlier method's wrong-but-confident guess. Here
every method proposes a candidate + a raw score, a small logistic
regression (fit on the training set) converts each method's raw score
into a calibrated P(this candidate is correct), and the candidate with
the highest calibrated probability wins. If that probability is below
`review_threshold`, the row is routed to NEEDS_REVIEW instead of forcing
a guess -- which is what actually lets the *resolved* accuracy approach
100%, with the remainder going to a human queue (see docs/accuracy_analysis.md).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_matcher import CosineMatcher, fuzzy_predict
from preprocessing import clean_name
from sbert_matcher import SBERTMatcher

NEEDS_REVIEW = "NEEDS_REVIEW"


class EnsembleMatcher:
    def __init__(self, sbert_model_path, review_threshold=0.5):
        self.cosine_matcher = CosineMatcher()
        self.sbert_matcher = SBERTMatcher(sbert_model_path)
        self.review_threshold = review_threshold
        self.rerankers = {}  # method_name -> fitted LogisticRegression

    def fit_reference(self, corpus_texts, corpus_labels):
        """corpus_texts: messy training examples. corpus_labels: their canonical
        names. Matching against many noisy examples (not a deduped canonical
        list) is what gives TF-IDF/fuzzy/SBERT enough surface-form coverage to
        work -- see docs/accuracy_analysis.md."""
        self.cosine_matcher.fit(corpus_texts, corpus_labels)
        self.sbert_matcher.fit(corpus_texts, corpus_labels)
        return self

    def _raw_candidates(self, queries, exclude_indices=None):
        cosine_preds, cosine_scores = self.cosine_matcher.predict(queries, exclude_indices=exclude_indices)
        fuzzy_preds, fuzzy_scores = fuzzy_predict(
            queries, self.cosine_matcher.corpus_texts, self.cosine_matcher.corpus_labels,
            exclude_indices=exclude_indices,
        )
        sbert_preds, sbert_scores = self.sbert_matcher.predict(queries, exclude_indices=exclude_indices)
        return {
            "cosine": (cosine_preds, cosine_scores),
            "fuzzy": (fuzzy_preds, fuzzy_scores),
            "sbert": (sbert_preds, sbert_scores),
        }

    def fit_reranker(self, train_clean_names, train_true_names, exclude_indices=None):
        """exclude_indices: positions of these same rows inside the fitted
        corpus, so calibration doesn't see queries trivially matching
        themselves (see evaluate.py for how these positions are derived)."""
        candidates = self._raw_candidates(train_clean_names, exclude_indices=exclude_indices)
        true_arr = np.asarray(train_true_names)

        for method, (preds, scores) in candidates.items():
            preds_arr = np.asarray(preds)
            is_correct = (preds_arr == true_arr).astype(int)
            X = np.asarray(scores).reshape(-1, 1)
            if is_correct.sum() == 0 or is_correct.sum() == len(is_correct):
                clf = None
            else:
                clf = LogisticRegression()
                clf.fit(X, is_correct)
            self.rerankers[method] = clf
        return self

    def _calibrated_prob(self, method, scores):
        clf = self.rerankers.get(method)
        scores = np.asarray(scores).reshape(-1, 1)
        if clf is None:
            return scores.ravel()
        return clf.predict_proba(scores)[:, 1]

    def predict(self, queries, exclude_indices=None):
        candidates = self._raw_candidates(queries, exclude_indices=exclude_indices)
        n = len(queries)

        probs = {method: self._calibrated_prob(method, scores) for method, (_, scores) in candidates.items()}

        final_preds, final_probs, final_methods = [], [], []
        for i in range(n):
            best_method = max(probs, key=lambda m: probs[m][i])
            best_prob = probs[best_method][i]
            best_pred = candidates[best_method][0][i]

            if best_prob < self.review_threshold:
                final_preds.append(NEEDS_REVIEW)
            else:
                final_preds.append(best_pred)
            final_probs.append(best_prob)
            final_methods.append(best_method)

        result = pd.DataFrame({
            "query": list(queries),
            "cosine_pred": candidates["cosine"][0],
            "cosine_score": candidates["cosine"][1],
            "fuzzy_pred": candidates["fuzzy"][0],
            "fuzzy_score": candidates["fuzzy"][1],
            "sbert_pred": candidates["sbert"][0],
            "sbert_score": candidates["sbert"][1],
            "chosen_method": final_methods,
            "confidence": final_probs,
            "final_pred": final_preds,
        })
        return result


def prepare_clean(series):
    return series.apply(clean_name)
