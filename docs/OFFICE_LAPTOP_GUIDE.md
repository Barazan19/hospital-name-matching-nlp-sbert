# Running this repo on an office laptop (CPU, no GPU)

A step-by-step guide to deploy the hospital-name matcher on a plain
office laptop. Two modes:

- **Mode A — lexical only** (recommended start): noise gate + exact match
  + char-ngram/word-cosine/fuzzy. No SBERT, no large model download.
- **Mode B — lexical + SBERT**: adds the deep semantic stage for the
  hardest rows. Needs the fine-tuned SBERT weights (~470 MB).

You can start with Mode A and switch to Mode B later by flipping one
flag — no code changes.

---

## 0. Prerequisites

- **Python 3.10 or 3.11** installed (`python --version`).
- Internet access for `pip install` (first time only).
- ~2 GB free disk (more if you use SBERT).

---

## 1. Get the code

```bash
git clone https://github.com/Barazan19/hospital-name-matching-nlp-sbert.git
cd hospital-name-matching-nlp-sbert
```

(Or copy the folder via USB — see §6 for the minimal file list.)

---

## 2. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

For **Mode A** you can install a lighter set (no torch / SBERT):

```bash
pip install pandas numpy scikit-learn rapidfuzz joblib openpyxl
```

---

## 3. Run — Mode A (lexical only, recommended start)

```bash
# match one or more names
python run_demo.py --no-sbert "RS ELISABET SEMARNG" "GYNAE ONCO PARTNERS"

# interactive prompt
python run_demo.py --no-sbert
```

Expected output:

```
  input      : RS ELISABET SEMARNG
  prediction : RS ELISABETH SEMARANG
  confidence : 0.62   (resolved at stage: fuzzy)

  input      : GYNAE ONCO PARTNERS
  prediction : NEEDS_REVIEW  [NEEDS REVIEW]
  suggestion : GYNAE ONCOLOGY CENTRE PTE LTD SINGAPORE
  confidence : 0.28   (resolved at stage: char)
```

- **First run** builds and caches the matcher (~90 s). **Later runs load
  in under a second.**
- No SBERT weights are needed in this mode. Rows the lexical stages
  aren't confident about come back as `NEEDS_REVIEW` with a best-guess
  `suggestion` for a human to confirm.

---

## 4. Run — Mode B (lexical + SBERT, maximum accuracy)

SBERT weights are **not** in the GitHub repo (too large, gitignored). Get
them one of two ways:

**Option 1 — copy the prebuilt weights (fast).** Copy the folder
`models/sbert-hospital-matcher/` from the machine where it was trained
onto this laptop, into the same path. ~470 MB.

**Option 2 — regenerate on this laptop (slow, ~50 min on CPU):**

```bash
python src/finetune_sbert.py --epochs 1 --batch-size 32
```

Then run with SBERT enabled (the default):

```bash
python run_demo.py "GYNAE ONCO PARTNERS" "BANDUNG ADVENTIST HOSPITAL"
```

SBERT is **loaded lazily** — only when a row actually reaches the SBERT
stage. The first such call also caches the corpus embeddings to
`models/corpus_sbert_emb.npy` so subsequent runs are fast.

---

## 5. Use it from your own Python code

```python
import sys; sys.path.insert(0, "src")
from pipeline import HospitalMatcher

# Mode A:
m = HospitalMatcher.load_or_build(use_sbert=False)
# Mode B:
# m = HospitalMatcher.load_or_build()          # use_sbert=True by default

print(m.predict_one("RS ELISABET SEMARNG"))
# {'input': 'RS ELISABET SEMARNG', 'prediction': 'RS ELISABETH SEMARANG',
#  'confidence': 0.62, 'stage': 'fuzzy', 'needs_review': False, ...}

import pandas as pd
df = m.predict_batch(["APOTIK ASEAN JAYA", "GYNAE ONCO PARTNERS"])
df.to_excel("matched.xlsx", index=False)   # route NEEDS_REVIEW rows to a human
```

**Tunables** (pass to `load_or_build` / constructor):

| Param | Default | Effect |
|---|---|---|
| `lexical_threshold` | 0.6 | raise → fewer auto-answers, higher accuracy, more review |
| `sbert_threshold` | 0.6 | same, for the SBERT stage |
| `noise_min_words` | 12 | min word count for the noise gate to flag a keyword-less row |
| `use_sbert` | True | `False` = pure-lexical, no SBERT weights needed |

---

## 6. Minimal file list (if copying by USB instead of git clone)

**Mode A (lexical only):**

```
run_demo.py
requirements.txt
src/pipeline.py
src/baseline_matcher.py
src/preprocessing.py
src/sbert_matcher.py          # imported only if SBERT is ever enabled
data/raw/Hospital_Train.csv
data/raw/Hospital_Train_new.csv
```

**Mode B** additionally needs `models/sbert-hospital-matcher/` (the
fine-tuned weights).

Not needed at runtime: `notebooks/`, `docs/`, `results/`,
`src/evaluate.py`, `src/ablation_sbert.py`, `src/ensemble_matcher.py`,
`src/finetune_sbert.py` (only for retraining).

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: sentence_transformers` in Mode B | `pip install sentence-transformers torch` (or run Mode A with `--no-sbert`) |
| SBERT stage errors about missing model | the `models/sbert-hospital-matcher/` folder isn't present — use Mode A, or get the weights (§4) |
| First run feels stuck | it's building the cache (~90 s); subsequent runs are instant |
| Want to rebuild the cache | delete `models/pipeline_lexical.joblib` and `models/corpus_sbert_emb.npy`, then run again |
| Unicode error in console | already handled (`run_demo.py` prints ASCII only); if you see it in your own scripts, avoid emoji in `print` |

---

## 8. What "accuracy" to expect

This is honest, leak-free performance (see
[`accuracy_analysis.md`](accuracy_analysis.md)): roughly **64–77%** on the
rows the system answers confidently, with the rest routed to review.
There is a measured hard ceiling (~70–85%) because some inputs simply
don't contain enough signal to recover the right name — those are meant
to land in the `NEEDS_REVIEW` queue, not be force-guessed.
