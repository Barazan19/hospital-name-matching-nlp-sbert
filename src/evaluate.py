"""End-to-end evaluation: baselines vs the fine-tuned-SBERT ensemble.

Run from the repo root:
    python src/evaluate.py --sbert-model models/sbert-hospital-matcher
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_matcher import CosineMatcher, fuzzy_predict
from ensemble_matcher import EnsembleMatcher
from preprocessing import clean_name, resolve_label_conflicts

ROOT = Path(__file__).resolve().parents[1]


def load_and_clean(path):
    df = pd.read_csv(ROOT / path)
    df = df.dropna(subset=["Hospital_Name (clean)", "Hospital Name rev 2"])
    df["clean"] = df["Hospital_Name (clean)"].apply(clean_name)
    df["true"] = df["Hospital Name rev 2"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df[["clean", "true"]]


def leave_one_out_indices(query_texts, corpus_texts):
    """Position in corpus_texts to exclude for each query, when that exact
    text is also a corpus row -- otherwise a query that happens to be
    verbatim-duplicated in the corpus trivially "matches itself" and the
    measured accuracy stops reflecting genuine generalization (see
    docs/accuracy_analysis.md for how this leak was found)."""
    text_to_pos = {}
    for i, t in enumerate(corpus_texts):
        text_to_pos.setdefault(t, i)
    return [text_to_pos.get(t) for t in query_texts]


def baseline_accuracy(test_df, corpus_texts, corpus_labels):
    corpus_texts = list(corpus_texts)
    exclude = leave_one_out_indices(test_df["clean"], corpus_texts)
    cosine_matcher = CosineMatcher().fit(corpus_texts, corpus_labels)
    cosine_preds, _ = cosine_matcher.predict(test_df["clean"], exclude_indices=exclude)
    fuzzy_preds, _ = fuzzy_predict(test_df["clean"], corpus_texts, corpus_labels, exclude_indices=exclude)
    return {
        "cosine_only": accuracy_score(test_df["true"], cosine_preds),
        "fuzzy_only": accuracy_score(test_df["true"], fuzzy_preds),
    }


def metrics_at_threshold(result, threshold):
    n = len(result)
    resolved = result[result["confidence"] >= threshold]
    n_resolved = len(resolved)
    n_correct = (resolved["chosen_pred"] == resolved["true"]).sum()
    return {
        "threshold": threshold,
        "n_total": n,
        "n_resolved": n_resolved,
        "n_needs_review": n - n_resolved,
        "coverage": n_resolved / n if n else 0.0,
        "accuracy_on_resolved": n_correct / n_resolved if n_resolved else 0.0,
        "forced_accuracy_all_rows": n_correct / n if n else 0.0,
        "effective_accuracy_with_human_review": (n_correct + (n - n_resolved)) / n if n else 0.0,
    }


def ensemble_eval(matcher, test_df, label, outdir):
    exclude = leave_one_out_indices(test_df["clean"], matcher.cosine_matcher.corpus_texts)
    # Predict with threshold=0 so every row's chosen candidate is recorded;
    # NEEDS_REVIEW routing is then evaluated for several thresholds below
    # without re-running the (expensive) matchers.
    saved_threshold = matcher.review_threshold
    matcher.review_threshold = 0.0
    result = matcher.predict(test_df["clean"], exclude_indices=exclude)
    matcher.review_threshold = saved_threshold

    result["true"] = test_df["true"].values
    result["chosen_pred"] = result["final_pred"]  # threshold=0 -> final_pred is always the chosen candidate

    outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(outdir / f"ensemble_predictions_{label}.csv", index=False)

    sweep = [metrics_at_threshold(result, t) for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    headline = metrics_at_threshold(result, saved_threshold)
    return {"headline": headline, "threshold_sweep": sweep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbert-model", default=str(ROOT / "models" / "sbert-hospital-matcher"))
    parser.add_argument("--reranker-sample-size", type=int, default=5000)
    parser.add_argument("--review-threshold", type=float, default=0.6)
    args = parser.parse_args()

    train = pd.concat([
        load_and_clean("data/raw/Hospital_Train.csv"),
        load_and_clean("data/raw/Hospital_Train_new.csv"),
    ], ignore_index=True).drop_duplicates()
    train = train[(train["clean"] != "") & (train["true"] != "")].reset_index(drop=True)
    print(f"Training corpus (deduped messy->canonical pairs): {len(train)}")
    train = resolve_label_conflicts(train)
    print(f"Training corpus after resolving conflicting labels per input: {len(train)}")

    test_sets = {
        "test_original": load_and_clean("data/raw/Hospital_Test.csv"),
        "test_new": load_and_clean("data/raw/Hospital_Test_new.csv"),
    }

    report = {}
    for label, test_df in test_sets.items():
        report[label] = {"baseline": baseline_accuracy(test_df, train["clean"], train["true"])}

    matcher = EnsembleMatcher(sbert_model_path=args.sbert_model, review_threshold=args.review_threshold)
    print(f"Fitting matchers on the full corpus ({len(train)} rows)...")
    matcher.fit_reference(train["clean"], train["true"])

    # Calibrate on a sample of corpus rows used as queries, leave-one-out
    # style (each row's own position is excluded from its own candidate
    # search) so the reranker doesn't just learn "score ~1.0 = correct".
    calib = train.sample(n=min(args.reranker_sample_size, len(train)), random_state=42)
    print(f"Fitting reranker on {len(calib)} leave-one-out calibration rows...")
    matcher.fit_reranker(calib["clean"], calib["true"], exclude_indices=list(calib.index))

    outdir = ROOT / "results"
    for label, test_df in test_sets.items():
        print(f"Evaluating ensemble on {label} ({len(test_df)} rows)...")
        report[label]["ensemble"] = ensemble_eval(matcher, test_df, label, outdir)

    with open(outdir / "metrics_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
