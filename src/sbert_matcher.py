"""SBERT-based nearest-neighbor matcher against a corpus of example texts.

Same corpus_texts/corpus_labels split as CosineMatcher (see
baseline_matcher.py) -- matching against many noisy training examples
instead of a deduped canonical list.
"""
from sentence_transformers import SentenceTransformer, util


class SBERTMatcher:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)

    def fit(self, corpus_texts, corpus_labels=None):
        self.corpus_texts = list(corpus_texts)
        self.corpus_labels = list(corpus_labels) if corpus_labels is not None else self.corpus_texts
        self.corpus_emb = self.model.encode(
            self.corpus_texts, convert_to_tensor=True, show_progress_bar=False
        )
        return self

    def predict(self, queries, batch_size=64, exclude_indices=None):
        query_emb = self.model.encode(
            list(queries), convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size
        )
        sims = util.cos_sim(query_emb, self.corpus_emb)
        if exclude_indices is not None:
            for row, idx in enumerate(exclude_indices):
                if idx is not None:
                    sims[row, idx] = -1.0
        top_idx = sims.argmax(dim=1)
        top_score = sims.max(dim=1).values
        preds = [self.corpus_labels[i] for i in top_idx.tolist()]
        return preds, top_score.cpu().numpy()
