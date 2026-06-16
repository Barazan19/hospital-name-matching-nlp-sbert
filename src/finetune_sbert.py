"""Domain-adapt SBERT on (messy_name -> canonical_name) pairs.

The off-the-shelf 'paraphrase-MiniLM-L6-v2' used in the original notebook
is English-only and was never shown a single hospital name during
pretraining, so its embedding space has no notion of "RSUD" ~ "RSU" ~
"RUMAH SAKIT UMUM DAERAH" being related. Fine-tuning with
MultipleNegativesRankingLoss pulls each messy variant's embedding toward
its correct canonical name and (via in-batch negatives) away from every
other canonical name seen in the same batch -- directly optimizing for
the retrieval task this project actually needs.

Usage:
    python src/finetune_sbert.py --epochs 2 --batch-size 64
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import clean_name

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_pairs(train_files):
    frames = [pd.read_csv(ROOT / f) for f in train_files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Hospital_Name (clean)", "Hospital Name rev 2"])

    anchors = df["Hospital_Name (clean)"].apply(clean_name)
    positives = df["Hospital Name rev 2"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    pairs = pd.DataFrame({"anchor": anchors, "positive": positives})
    pairs = pairs[(pairs["anchor"] != "") & (pairs["positive"] != "")]
    pairs = pairs.drop_duplicates()
    return pairs.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-files", nargs="+", default=[
        "data/raw/Hospital_Train.csv", "data/raw/Hospital_Train_new.csv",
    ])
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "sbert-hospital-matcher"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pairs = build_pairs(args.train_files)
    print(f"Training pairs: {len(pairs)}")

    train_dataset = Dataset.from_pandas(pairs[["anchor", "positive"]])

    model = SentenceTransformer(args.base_model)
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(ROOT / "models" / "_trainer_ckpts"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
