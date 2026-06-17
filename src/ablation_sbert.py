"""Ablation: how much does SBERT actually add over TF-IDF cosine + fuzzy?

Two questions this answers, both written to results/ablation_sbert.json:

1. Aggregate: does dropping SBERT from the ensemble change accuracy?
   (Spoiler from the data: barely -- the calibrated cosine+fuzzy reranker
   flags the same low-confidence rows for review that SBERT would have
   needed to fix.)

2. Niche: on which rows does SBERT *uniquely* get the right answer while
   BOTH cosine and fuzzy get it wrong? Those are the rows where matching
   needed semantic similarity, not surface/lexical overlap -- the honest
   place to point to when claiming SBERT earned its keep. Each such row is
   logged with a character-level similarity between the query and the true
   answer, to show they really are lexically dissimilar (low overlap =
   SBERT did something fuzzy/TF-IDF structurally cannot).

Run from repo root:
    python src/ablation_sbert.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_matcher import CosineMatcher, fuzzy_predict
from ensemble_matcher import EnsembleMatcher
from evaluate import leave_one_out_indices, load_and_clean
from preprocessing import resolve_label_conflicts

ROOT = Path(__file__).resolve().parents[1]
TEST_PATHS = {
    "test_original": "data/raw/Hospital_Test.csv",
    "test_new": "data/raw/Hospital_Test_new.csv",
}


def fit_reranker_scores(preds, scores, true_arr):
    correct = (np.asarray(preds) == true_arr).astype(int)
    if correct.sum() in (0, len(correct)):
        return None
    return LogisticRegression().fit(np.asarray(scores).reshape(-1, 1), correct)


def calibrated(clf, scores):
    scores = np.asarray(scores).reshape(-1, 1)
    return scores.ravel() if clf is None else clf.predict_proba(scores)[:, 1]


def main():
    train = pd.concat(
        [load_and_clean("data/raw/Hospital_Train.csv"), load_and_clean("data/raw/Hospital_Train_new.csv")],
        ignore_index=True,
    ).drop_duplicates()
    train = train[(train["clean"] != "") & (train["true"] != "")].reset_index(drop=True)
    train = resolve_label_conflicts(train)

    matcher = EnsembleMatcher(sbert_model_path=str(ROOT / "models" / "sbert-hospital-matcher"))
    matcher.fit_reference(train["clean"], train["true"])

    # Calibrate all three rerankers on leave-one-out corpus rows.
    calib = train.sample(n=min(5000, len(train)), random_state=42)
    excl = list(calib.index)
    cand = matcher._raw_candidates(calib["clean"], exclude_indices=excl)
    true_c = calib["true"].values
    clfs = {m: fit_reranker_scores(p, s, true_c) for m, (p, s) in cand.items()}

    report = {}
    for label, path in TEST_PATHS.items():
        test = load_and_clean(path)
        excl_t = leave_one_out_indices(test["clean"], matcher.cosine_matcher.corpus_texts)
        cand_t = matcher._raw_candidates(test["clean"], exclude_indices=excl_t)
        true_t = test["true"].values

        probs = {m: calibrated(clfs[m], s) for m, (p, s) in cand_t.items()}
        preds = {m: np.asarray(p) for m, (p, s) in cand_t.items()}

        def ensemble_pick(methods):
            chosen_pred, chosen_conf = [], []
            for i in range(len(test)):
                best = max(methods, key=lambda m: probs[m][i])
                chosen_pred.append(preds[best][i])
                chosen_conf.append(probs[best][i])
            return np.asarray(chosen_pred), np.asarray(chosen_conf)

        lexical = [m for m in ("cosine", "char", "fuzzy") if m in preds]
        full_pred, full_conf = ensemble_pick(lexical + ["sbert"])
        nosb_pred, nosb_conf = ensemble_pick(lexical)

        def acc_at(pred, conf, t):
            mask = conf >= t
            n = len(pred)
            n_res = int(mask.sum())
            n_cor = int(((pred == true_t) & mask).sum())
            return {
                "threshold": t,
                "coverage": n_res / n,
                "accuracy_on_resolved": (n_cor / n_res) if n_res else None,
            }

        # Where does SBERT uniquely win? cosine AND fuzzy both wrong, sbert right.
        cos_ok = preds["cosine"] == true_t
        fuz_ok = preds["fuzzy"] == true_t
        sb_ok = preds["sbert"] == true_t
        unique_sbert = (~cos_ok) & (~fuz_ok) & sb_ok

        unique_rows = []
        for i in np.where(unique_sbert)[0]:
            q = test["clean"].iloc[i]
            ans = true_t[i]
            unique_rows.append({
                "query": q,
                "true_answer": ans,
                "sbert_pred": preds["sbert"][i],
                "cosine_pred": preds["cosine"][i],
                "fuzzy_pred": preds["fuzzy"][i],
                "char_similarity_query_vs_answer": round(fuzz.token_sort_ratio(q, ans), 1),
            })

        report[label] = {
            "standalone_accuracy": {
                m: round(accuracy_score(true_t, preds[m]), 4) for m in preds
            },
            "ensemble_with_sbert": [acc_at(full_pred, full_conf, t) for t in (0.5, 0.6, 0.7)],
            "ensemble_without_sbert": [acc_at(nosb_pred, nosb_conf, t) for t in (0.5, 0.6, 0.7)],
            "n_rows_sbert_uniquely_correct": int(unique_sbert.sum()),
            "sbert_unique_wins": unique_rows,
        }

    outpath = ROOT / "results" / "ablation_sbert.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
