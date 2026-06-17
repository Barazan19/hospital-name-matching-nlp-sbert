"""Production cascade matcher for office-laptop (CPU-only) deployment.

Designed so the cheap, dependency-light stages do the bulk of the work
and the expensive SBERT stage only runs on the small residual of rows
the lexical stages weren't confident about. SBERT is loaded lazily -- if
every row resolves earlier, the model is never even loaded, so cold
start is fast.

Stages (each row exits at the first one that's confident):
    0. noise gate     -- claim-note garbage with no hospital keyword ->
                         NEEDS_REVIEW (don't force a match)
    1. exact match    -- normalized text already a known name -> instant
    2. lexical        -- char-ngram + word cosine + fuzzy, calibrated;
                         the CPU-only workhorse (char-ngram is strongest)
    3. sbert          -- only for low-confidence residual; recovers
                         semantic cases (cross-lingual synonyms, etc.)
    4. review         -- still unsure -> human queue

Usage:
    from pipeline import HospitalMatcher
    m = HospitalMatcher.from_training_csvs()      # fits in a few seconds
    m.predict_one("RSSTELISABETH SEMARANG")
    m.predict_batch(["...", "..."])               # -> DataFrame
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_matcher import CosineMatcher, fuzzy_predict
from preprocessing import clean_name, resolve_label_conflicts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SBERT = ROOT / "models" / "sbert-hospital-matcher"
DEFAULT_CACHE = ROOT / "models" / "corpus_sbert_emb.npy"
DEFAULT_ARTIFACT = ROOT / "models" / "pipeline_lexical.joblib"

REVIEW = "NEEDS_REVIEW"

HOSPITAL_KEYWORDS = {
    "RS", "RSU", "RSUD", "RSIA", "RSUP", "RSAL", "RSAU", "RUMAH", "SAKIT",
    "KLINIK", "CLINIC", "HOSPITAL", "APOTEK", "APOTIK", "LAB", "LABORATORIUM",
    "MEDICAL", "MEDIKA", "CENTRE", "CENTER", "PUSKESMAS", "DR", "DRG",
    "SPECIALIST", "SPESIALIS", "PRAKTEK", "PRAKTIK", "POLI", "PMI",
    "DENTAL", "EYE", "MATA", "THT", "BEDAH", "PRODIA", "BIDAN", "PUSAT",
}


class HospitalMatcher:
    def __init__(
        self,
        corpus_texts,
        corpus_labels,
        sbert_model_path=DEFAULT_SBERT,
        sbert_cache=DEFAULT_CACHE,
        lexical_threshold=0.6,
        sbert_threshold=0.6,
        noise_min_words=12,
        calibration_sample=2000,
        random_state=42,
        use_sbert=True,
    ):
        self.corpus_texts = list(corpus_texts)
        self.corpus_labels = list(corpus_labels)
        self.sbert_model_path = str(sbert_model_path)
        self.sbert_cache = Path(sbert_cache)
        self.lexical_threshold = lexical_threshold
        self.sbert_threshold = sbert_threshold
        self.noise_min_words = noise_min_words
        # use_sbert=False runs a pure-lexical pipeline (stages 0-2): no SBERT
        # weights needed at all, low-confidence rows go straight to review.
        self.use_sbert = use_sbert

        self.exact = {}
        for t, l in zip(self.corpus_texts, self.corpus_labels):
            self.exact.setdefault(t, l)

        self.word_matcher = CosineMatcher(ngram_range=(1, 2), analyzer="word")
        self.char_matcher = CosineMatcher(ngram_range=(3, 5), analyzer="char_wb")
        self.word_matcher.fit(self.corpus_texts, self.corpus_labels)
        self.char_matcher.fit(self.corpus_texts, self.corpus_labels)

        self._sbert = None  # lazy
        self._fit_lexical_calibrators(calibration_sample, random_state)

    # ---- construction helpers -------------------------------------------------
    @classmethod
    def from_training_csvs(cls, paths=None, **kwargs):
        paths = paths or [
            ROOT / "data" / "raw" / "Hospital_Train.csv",
            ROOT / "data" / "raw" / "Hospital_Train_new.csv",
        ]
        frames = []
        for p in paths:
            df = pd.read_csv(p).dropna(subset=["Hospital_Name (clean)", "Hospital Name rev 2"])
            df = pd.DataFrame({
                "clean": df["Hospital_Name (clean)"].apply(clean_name),
                "true": df["Hospital Name rev 2"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            })
            frames.append(df)
        data = pd.concat(frames, ignore_index=True).drop_duplicates()
        data = data[(data["clean"] != "") & (data["true"] != "")]
        data = resolve_label_conflicts(data)
        return cls(data["clean"].tolist(), data["true"].tolist(), **kwargs)

    @classmethod
    def load_or_build(cls, artifact=DEFAULT_ARTIFACT, use_sbert=None, **kwargs):
        """Production entrypoint: load the pre-fit lexical artifact in
        seconds if it exists, otherwise build once (~minutes) and save it.
        SBERT is still loaded lazily on first escalation. Pass
        use_sbert=False to run a pure-lexical pipeline with no SBERT weights."""
        artifact = Path(artifact)
        if artifact.exists():
            obj = cls.load(artifact)
            if use_sbert is not None:
                obj.use_sbert = use_sbert
            return obj
        if use_sbert is not None:
            kwargs["use_sbert"] = use_sbert
        obj = cls.from_training_csvs(**kwargs)
        obj.save(artifact)
        return obj

    def save(self, artifact=DEFAULT_ARTIFACT):
        """Persist the fitted lexical state (vectorizers, corpus, calibrators).
        SBERT weights/embeddings are cached separately and not duplicated here."""
        artifact = Path(artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "corpus_texts": self.corpus_texts,
            "corpus_labels": self.corpus_labels,
            "word_matcher": self.word_matcher,
            "char_matcher": self.char_matcher,
            "lex_calibrators": self.lex_calibrators,
            "exact": self.exact,
            "config": {
                "sbert_model_path": self.sbert_model_path,
                "sbert_cache": str(self.sbert_cache),
                "lexical_threshold": self.lexical_threshold,
                "sbert_threshold": self.sbert_threshold,
                "noise_min_words": self.noise_min_words,
                "use_sbert": self.use_sbert,
            },
        }, artifact)
        return artifact

    @classmethod
    def load(cls, artifact=DEFAULT_ARTIFACT):
        state = joblib.load(artifact)
        obj = cls.__new__(cls)
        obj.corpus_texts = state["corpus_texts"]
        obj.corpus_labels = state["corpus_labels"]
        obj.word_matcher = state["word_matcher"]
        obj.char_matcher = state["char_matcher"]
        obj.lex_calibrators = state["lex_calibrators"]
        obj.exact = state["exact"]
        cfg = state["config"]
        obj.sbert_model_path = cfg["sbert_model_path"]
        obj.sbert_cache = Path(cfg["sbert_cache"])
        obj.lexical_threshold = cfg["lexical_threshold"]
        obj.sbert_threshold = cfg["sbert_threshold"]
        obj.noise_min_words = cfg["noise_min_words"]
        obj.use_sbert = cfg.get("use_sbert", True)
        obj._sbert = None
        return obj

    # ---- calibration ----------------------------------------------------------
    def _fit_lexical_calibrators(self, sample_n, random_state):
        n = len(self.corpus_texts)
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=min(sample_n, n), replace=False)
        queries = [self.corpus_texts[i] for i in idx]
        truth = np.array([self.corpus_labels[i] for i in idx])
        exclude = list(idx)  # leave-one-out so a query can't match itself

        self.lex_calibrators = {}
        wp, ws = self.word_matcher.predict(queries, exclude_indices=exclude)
        cp, cs = self.char_matcher.predict(queries, exclude_indices=exclude)
        fp, fs = fuzzy_predict(queries, self.corpus_texts, self.corpus_labels, exclude_indices=exclude)
        for name, preds, scores in [("cosine", wp, ws), ("char", cp, cs), ("fuzzy", fp, fs)]:
            self.lex_calibrators[name] = self._fit_one(preds, scores, truth)

    @staticmethod
    def _fit_one(preds, scores, truth):
        correct = (np.asarray(preds) == truth).astype(int)
        if correct.sum() in (0, len(correct)):
            return None
        return LogisticRegression().fit(np.asarray(scores).reshape(-1, 1), correct)

    @staticmethod
    def _prob(clf, scores):
        scores = np.asarray(scores).reshape(-1, 1)
        return scores.ravel() if clf is None else clf.predict_proba(scores)[:, 1]

    # ---- lazy SBERT -----------------------------------------------------------
    def _ensure_sbert(self):
        if self._sbert is not None:
            return
        from sbert_matcher import SBERTMatcher  # imported only when needed
        self._sbert = SBERTMatcher(self.sbert_model_path)
        if self.sbert_cache.exists():
            import torch
            emb = np.load(self.sbert_cache)
            self._sbert.corpus_texts = self.corpus_texts
            self._sbert.corpus_labels = self.corpus_labels
            self._sbert.corpus_emb = torch.tensor(emb)
        else:
            self._sbert.fit(self.corpus_texts, self.corpus_labels)
            self.sbert_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.sbert_cache, self._sbert.corpus_emb.cpu().numpy())
        # calibrate SBERT once, leave-one-out, reusing the cached embeddings
        rng = np.random.default_rng(0)
        idx = rng.choice(len(self.corpus_texts), size=min(2000, len(self.corpus_texts)), replace=False)
        queries = [self.corpus_texts[i] for i in idx]
        truth = np.array([self.corpus_labels[i] for i in idx])
        sp, ss = self._sbert.predict(queries, exclude_indices=list(idx))
        self.sbert_calibrator = self._fit_one(sp, ss, truth)

    # ---- noise gate -----------------------------------------------------------
    def _is_noise(self, cleaned):
        tokens = cleaned.split()
        if not tokens:
            return True
        if any(tok in HOSPITAL_KEYWORDS for tok in tokens):
            return False
        return len(tokens) >= self.noise_min_words

    # ---- prediction -----------------------------------------------------------
    def predict_batch(self, names):
        cleaned = [clean_name(x) for x in names]
        n = len(names)
        out = [None] * n
        pending = []  # indices needing lexical matching

        for i, c in enumerate(cleaned):
            if self._is_noise(c):
                out[i] = self._row(names[i], c, REVIEW, 0.0, "noise_gate", True)
            elif c in self.exact:
                out[i] = self._row(names[i], c, self.exact[c], 1.0, "exact", False)
            else:
                pending.append(i)

        if pending:
            self._resolve_lexical(names, cleaned, pending, out)

        return pd.DataFrame(out)

    def _resolve_lexical(self, names, cleaned, pending, out):
        q = [cleaned[i] for i in pending]
        cands = {}
        wp, ws = self.word_matcher.predict(q)
        cp, cs = self.char_matcher.predict(q)
        fp, fs = fuzzy_predict(q, self.corpus_texts, self.corpus_labels)
        cands["cosine"] = (wp, self._prob(self.lex_calibrators["cosine"], ws))
        cands["char"] = (cp, self._prob(self.lex_calibrators["char"], cs))
        cands["fuzzy"] = (fp, self._prob(self.lex_calibrators["fuzzy"], fs))

        residual = []  # (pending_pos, global_idx)
        for k, gi in enumerate(pending):
            best = max(cands, key=lambda m: cands[m][1][k])
            prob = cands[best][1][k]
            pred = cands[best][0][k]
            if prob >= self.lexical_threshold:
                out[gi] = self._row(names[gi], cleaned[gi], pred, float(prob), best, False)
            else:
                residual.append((k, gi, best, prob, pred))

        if residual:
            if self.use_sbert:
                self._resolve_sbert(names, cleaned, residual, out)
            else:
                # pure-lexical mode: keep best lexical guess but flag for review
                for (_, gi, lex_best, lex_prob, lex_pred) in residual:
                    out[gi] = self._row(
                        names[gi], cleaned[gi], REVIEW, float(lex_prob), lex_best, True, suggestion=lex_pred
                    )

    def _resolve_sbert(self, names, cleaned, residual, out):
        self._ensure_sbert()
        q = [cleaned[gi] for (_, gi, _, _, _) in residual]
        sp, ss = self._sbert.predict(q)
        sprob = self._prob(self.sbert_calibrator, ss)
        for j, (_, gi, lex_best, lex_prob, lex_pred) in enumerate(residual):
            if sprob[j] >= self.sbert_threshold and sprob[j] >= lex_prob:
                out[gi] = self._row(names[gi], cleaned[gi], sp[j], float(sprob[j]), "sbert", False)
            elif lex_prob >= self.lexical_threshold:
                out[gi] = self._row(names[gi], cleaned[gi], lex_pred, float(lex_prob), lex_best, False)
            else:
                # keep the best available guess but flag for review
                best_pred, best_prob, best_m = (
                    (sp[j], float(sprob[j]), "sbert") if sprob[j] >= lex_prob else (lex_pred, float(lex_prob), lex_best)
                )
                out[gi] = self._row(names[gi], cleaned[gi], REVIEW, best_prob, best_m, True, suggestion=best_pred)

    @staticmethod
    def _row(inp, cleaned, pred, conf, stage, needs_review, suggestion=None):
        return {
            "input": inp,
            "cleaned": cleaned,
            "prediction": pred,
            "confidence": round(float(conf), 4),
            "stage": stage,
            "needs_review": needs_review,
            "suggestion": suggestion if suggestion is not None else (pred if not needs_review else None),
        }

    def predict_one(self, name):
        return self.predict_batch([name]).iloc[0].to_dict()


if __name__ == "__main__":
    m = HospitalMatcher.load_or_build()  # builds + saves on first run, loads after
    demo = [
        "RSSTELISABETH SEMARANG",
        "BANDUNG ADVENTIST HOSPITAL",
        "KLINIK MATA NUSANTARA KEMAYORAN",
        "ADVICENYA KEMBALI PADA FORM B PERTAMA PADA HAL VESTIBULO NEUROARTITIS DIKONSULKAN",
    ]
    print(m.predict_batch(demo).to_string())
