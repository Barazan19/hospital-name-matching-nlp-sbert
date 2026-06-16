# SBERT Architecture

This document explains the SBERT-based component of the matching pipeline:
what SBERT is, why it was chosen, how it is fine-tuned for hospital names
specifically, and how it fits into the full ensemble.

## 1. Why SBERT instead of plain BERT

A vanilla BERT encodes a sentence into per-token vectors; to compare two
hospital names you'd have to run BERT once per *pair* (cross-encoder),
which is O(n²) and far too slow for matching one messy name against a
reference list of thousands of canonical names.

**Sentence-BERT (SBERT)** instead fine-tunes BERT with a pooling layer so
that **one fixed-size vector represents the whole sentence**, and that
vector space is shaped so cosine similarity between two vectors reflects
semantic similarity between the two sentences. That means:

- Reference names are embedded **once** and cached.
- A new query name is embedded once and compared to all cached reference
  vectors with simple cosine similarity (vector math, no model
  inference per pair).

This is what makes SBERT usable as a real-time nearest-neighbor matcher
instead of a one-off classifier.

## 2. Base architecture

```mermaid
flowchart TB
    subgraph SBERT["SBERT Bi-Encoder"]
        direction TB
        IN["Input text\n e.g. 'RS HERMINA HOSPITALS MADIUN'"]
        TOK["Tokenizer\n(WordPiece / SentencePiece)"]
        TRF["Transformer encoder\n(multilingual MiniLM-L12, 12 layers)"]
        POOL["Mean pooling over token embeddings"]
        NORM["L2 normalization"]
        VEC["Sentence embedding\n(384-dim vector)"]
        IN --> TOK --> TRF --> POOL --> NORM --> VEC
    end
```

Base model: `paraphrase-multilingual-MiniLM-L12-v2`
(swapped in from the original `paraphrase-MiniLM-L6-v2`, which is
English-only — most reference names here are Indonesian/Singaporean).

## 3. Domain fine-tuning (this is the part that moves accuracy)

Off-the-shelf SBERT has never seen "RSUD", "RSIA" or claim-system noise
phrases — it has no reason to know that `"RS HERMINA HOSPITALS MADIUN"`
and `"RS HERMINA MADIUN"` should land near each other in vector space.
We fine-tune it on this project's own (messy_name → canonical_name)
pairs using **contrastive learning**:

```mermaid
flowchart LR
    A["Anchor:\nmessy name\n'RSSTELISABETH SEMARANG'"] --> ENC1["SBERT encoder\n(shared weights)"]
    P["Positive:\ncanonical name\n'RS ST ELISABETH SEMARANG'"] --> ENC2["SBERT encoder\n(shared weights)"]
    N1["In-batch negatives:\nevery other canonical name\nin the same training batch"] --> ENC3["SBERT encoder\n(shared weights)"]
    ENC1 --> VA["anchor vector"]
    ENC2 --> VP["positive vector"]
    ENC3 --> VN["negative vectors"]
    VA & VP & VN --> LOSS["MultipleNegativesRankingLoss\npull anchor↔positive together,\npush anchor↔negatives apart"]
    LOSS --> UPDATE["Backprop updates\nencoder weights"]
```

- **Anchor** = a raw/messy hospital name as it appears in claims data.
- **Positive** = its correct canonical reference name.
- **Negatives** = every *other* canonical name that happens to be in the
  same training batch (no manual negative mining needed).
- **Loss**: `MultipleNegativesRankingLoss` — a standard retrieval loss
  that directly optimizes "the correct match should have the highest
  cosine similarity among all candidates in the batch," which is
  exactly the task this project needs at inference time.

Training script: [`src/finetune_sbert.py`](../src/finetune_sbert.py).
Output: a fine-tuned model directory at `models/sbert-hospital-matcher/`
(not committed to git — regenerate it locally with the script).

## 4. Inference / matching flow

```mermaid
flowchart TB
    REF["Reference list:\nunique canonical hospital names"] --> EMB_REF["Encode once,\ncache embeddings"]
    Q["Incoming messy name"] --> CLEAN["src/preprocessing.py\n(normalize, strip claim-system noise)"]
    CLEAN --> EMB_Q["Encode with fine-tuned SBERT"]
    EMB_Q --> SIM["Cosine similarity vs\nall cached reference embeddings"]
    EMB_REF --> SIM
    SIM --> TOP["Top-1 match + similarity score"]
```

Implementation: [`src/sbert_matcher.py`](../src/sbert_matcher.py).

## 5. Where SBERT sits in the full pipeline

SBERT is one of three scorers feeding the reranking ensemble — see
[`src/ensemble_matcher.py`](../src/ensemble_matcher.py) and the system
diagram in the main [README](../README.md#architecture). It is the
scorer most robust to **word reordering, synonyms, and abbreviation
variants** (e.g. "ortho" vs "orthopedi"), which is precisely the failure
mode where TF-IDF cosine and character-level fuzzy matching struggle —
see [accuracy_analysis.md](accuracy_analysis.md) for the mismatch
breakdown that motivated this design.
