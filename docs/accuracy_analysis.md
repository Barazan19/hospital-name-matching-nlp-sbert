# Can this hit 100% accuracy? Analysis & recommendations

## TL;DR

**The 88-90% numbers in the original notebooks are not real.** Every
notebook except one decides its final prediction by checking each
candidate against the test set's ground-truth answer (`y_test`) before
choosing it — that's test-set leakage baked directly into the decision
logic, not a measurement of how the model would perform on unseen data.
SBERT itself never even ran successfully in the "SBERT" notebook (a
`ModuleNotFoundError` crashed that cell). Section 1 below shows exactly
where the leak is and what the *honest* numbers are.

After removing the leak and fixing two more measurement bugs this repo's
own pipeline introduced along the way (§2), the honest, leak-free
results are:

| | TF-IDF cosine alone | Fuzzy alone | Ensemble (fine-tuned SBERT + calibrated reranker) |
|---|---|---|---|
| `Hospital_Test.csv` | 49.5% | 48.1% | **63.4%** resolved @ 70% coverage (**74.3%** effective with human review) |
| `Hospital_Test_new.csv` | 54.2% | 50.3% | **66.7%** resolved @ 74% coverage (**75.5%** effective with human review) |

Run `python src/evaluate.py` to regenerate
[`results/metrics_summary.json`](../results/metrics_summary.json), which
also includes a full confidence-threshold sweep (§4) — at a stricter
threshold the ensemble reaches **~75-82% accuracy** on the subset it's
confident enough to answer, trading off coverage.

100% is still not a realistic target for the model alone — see §3 — but
the *system* (model + human review for low-confidence cases) can
realistically approach it.

## 1. The leakage bug in the original notebooks

Look at the decision logic in `notebooks/03_cosine_fuzzy_sbert.ipynb`
(and identically in `01_cosine_fuzzy_baseline.ipynb` and
`04_tfidf_cosine_logreg_hybrid_test.ipynb`):

```python
if cosine_preds[i] == actual:          # actual = y_test.iloc[i] -- the answer!
    final_preds.append(cosine_preds[i])
elif has_zero:
    ...
    if new_cosine_pred == actual:      # checks the answer again
        final_preds.append(new_cosine_pred)
    else:
        final_preds.append(cleaned_input)
elif fuzzy_match == actual:            # checks the answer a third time
    final_preds.append(fuzzy_match)
else:
    final_preds.append(cleaned_input)
```

Every branch except the final `else` explicitly compares a candidate to
`actual` (the ground truth) **before** deciding whether to use it. That
is not a real decision rule — a production system never has `y_test` to
peek at. It mathematically guarantees the reported accuracy is at least
`max(cosine_accuracy, fuzzy_accuracy, secondary_cosine_accuracy)`, and in
practice inflates it far above any single method's real accuracy.

Reproducing the exact same code (`docs/accuracy_analysis.md` numbers
were verified by re-running this logic with branch counters) confirms it
hits the reported **0.8883** — but breaks down as:

| Branch | Rows | Why it's not a real prediction |
|---|---|---|
| `cosine_preds[i] == actual` | 109 | Only taken when already correct |
| `fuzzy→cosine == actual` | 11 | Only taken when already correct |
| `fuzzy_match == actual` | 4 | Only taken when already correct |
| fallback (`cleaned_input`) | 82 | The only branch not peeking — and it isn't a prediction at all, just the raw cleaned text, which happens to already equal the answer for 59 of those 82 rows |

`notebooks/04_tfidf_cosine_logreg_hybrid_test.ipynb` (90.3%, the
"highest" number) has the identical pattern with `cosine_preds[i] ==
y_test.iloc[i]` / `logreg_preds[i] == y_test.iloc[i]`.

**The one notebook that doesn't do this** —
`notebooks/02_logreg_cosine_fuzzy.ipynb` — makes its decision from score
*thresholds* (`cosine_scores[i] >= cosine_thresh`, etc.), never from
`y_test` directly. It also happens to report the lowest accuracy
(61.3%), on the hardest/noisiest test set. That isn't a coincidence:
it's the only one of the four that was ever telling the truth.

And separately: the SBERT cell in `03_cosine_fuzzy_sbert.ipynb` crashes
with `ModuleNotFoundError: No module named 'sentence_transformers'`
right after `pip install sentence-transformers` — almost certainly
because the kernel was never restarted after the install. **SBERT's
re-ranking loop never executed in that notebook.** The saved "SBERT"
accuracy is identical cosine+fuzzy logic with leakage, not SBERT.

## 2. Two more leaks this repo's own pipeline had to fix

Building a leak-free replacement surfaced two more measurement bugs —
both worth recording since they're easy to reintroduce:

**Matching against a deduped canonical list instead of raw examples.**
The first version of `src/ensemble_matcher.py` built its TF-IDF/fuzzy/SBERT
reference set from the *unique canonical names* (~7,200 of them). That
collapsed accuracy to ~55-65%, for two reasons: (a) it threw away the
huge surface-form diversity of real training examples that TF-IDF/fuzzy
actually need, and (b) the canonical-name column itself isn't fully
consistent — e.g. `"RS BUN"` and `"RS BUNDA JAKARTA"` both exist as
separate "canonical" entries for the same hospital, so a correct match
on the right hospital can still land on the textually "wrong" variant.
Fix: match against the full corpus of *example* (messy_name →
canonical_name) pairs and return the matched example's label — the same
paradigm the original notebooks used, just without the leak.

**`Hospital_Train_new.csv` contains 100% of `Hospital_Test.csv` verbatim.**
Combining both training files into one corpus (for more coverage) means
every row of the original test set is also sitting in the corpus — so
evaluating against it measures memorization, not generalization, and
where the train-side label disagrees with the test-side label for the
identical text (it does, for 58/206 rows), the model is unfairly marked
wrong despite a "perfect" match. Fix: `evaluate.py`'s
`leave_one_out_indices()` excludes, per query, any corpus row whose
cleaned text is identical to that query, before computing the
similarity argmax — so a query can never trivially match its own
duplicate in the corpus, regardless of which file it came from.

## 3. Why 100% is still not realistic, even now

Some inputs have no recoverable signal in the text itself:

- `"KLINIK AP DAN AP"` → actual `"LABORATORIUM DAN KLINIK PRAMITA"` —
  zero shared tokens; only knowable from a business-specific aliasing
  table.
- Free-text claim notes accidentally left in the name column
  (`"REVIEW DOR 22 -DOKUMEN LAPORAN OP SUDAH DIKIRIM... RS BETHESDA"`)
  sometimes bury or delete the actual hospital name after noise-stripping.
- The canonical-label column is itself inconsistently curated (§2) —
  for a meaningful fraction of inputs, *the training data doesn't agree
  with the test data* on what the "right" answer even is. No model can
  learn an answer that contradicts every example it was shown.

The honest framing is the same as before, just with honest numbers
behind it: route what the model is confident about automatically, and
send the rest to a human queue. §4 shows that confidence threshold is a
real, tunable lever, not a hand-wave.

## 4. Confidence threshold sweep (coverage vs. accuracy)

From `results/metrics_summary.json` → `threshold_sweep`, on
`Hospital_Test.csv`:

| Confidence ≥ | Coverage | Accuracy on resolved | Effective (resolved + human-reviewed rest) |
|---|---|---|---|
| 0.3 | 88.3% | 58.2% | 63.1% |
| 0.4 | 80.6% | 60.2% | 68.0% |
| 0.5 | 70.4% | 63.4% | 74.3% |
| 0.6 | 59.2% | 71.3% | 83.0% |
| 0.7 | 36.4% | 74.7% | 90.8% |
| 0.8 | 13.1% | 81.5% | 97.6% |

This is a real, usable dial: a team with more human-review capacity can
set a high threshold and get >80% accuracy automatically with the
remainder reviewed; a team that wants to minimize manual review can
accept a lower threshold and a lower resolved-accuracy. `Hospital_Test_new.csv`
shows the same monotonic pattern (see the full JSON) — confirming the
calibrated probability is actually tracking real correctness likelihood,
not just noise.

## 5. Does SBERT actually help? (ablation)

This is the question that matters most for a project whose headline is
"SBERT." Honest answer, from `src/ablation_sbert.py` →
[`results/ablation_sbert.json`](../results/ablation_sbert.json):

**Standalone, SBERT is the single best matcher** — it beats both lexical
methods on its own:

| Test set | cosine-only | fuzzy-only | **SBERT-only** |
|---|---|---|---|
| `Hospital_Test.csv` | 49.5% | 48.1% | **52.9%** |
| `Hospital_Test_new.csv` | 54.2% | 50.3% | **58.7%** |

**But in the full calibrated ensemble, removing SBERT barely changes the
headline accuracy:**

| Test set @ conf≥0.6 | with SBERT | without SBERT (cosine+fuzzy only) |
|---|---|---|
| `Hospital_Test.csv` | 71.3% acc / 59.2% coverage | 71.8% acc / 53.4% coverage |
| `Hospital_Test_new.csv` | 68.6% acc / 67.7% coverage | 68.0% acc / 66.5% coverage |

SBERT adds a few points of **coverage** (more rows auto-resolved at the
same accuracy) but not headline accuracy, because the calibrated
cosine+fuzzy reranker already routes the rows SBERT would fix into the
human-review queue.

**Where SBERT *uniquely* earns its place** is the small set of rows where
BOTH lexical methods are wrong and only SBERT is right — **11 rows across
the two test sets**. These are exactly the cases lexical matching is
structurally incapable of, with low character-overlap to the answer:

| Query | True answer | What lexical methods picked | Why only SBERT got it |
|---|---|---|---|
| `BANDUNG ADVENTIST HOSPITAL` | `RS ADVENT BANDUNG` | both picked **PENANG** Adventist Hospital | needed "Adventist"≈"Advent" + that *Bandung*, not Penang, is the anchor |
| `HOSPITAL PAKAR DAMANSARA` | `DAMANSARA SPESIALIST HOSPITAL MALAYSIA` | `CMH HOSPITAL PAKAR` | "Pakar" = "Specialist" (Malay) — a cross-lingual synonym |
| `GYNAE ONCO PARTNERS` | `GYNAE ONCOLOGY CENTRE PTE LTD SINGAPORE` | `ONCOCARE CANCER CENTRE` / `CARE COLLAB PARTNERS` | "Onco" → "Oncology" |
| `DEMAM SAMPAI 378 ... RS SANTO BORROMEUS BANDUNG` (claim note) | `RS ST BORROMEUS` | a lab / unrelated RS | hospital name buried in free text — **char-similarity 12.6** |

So the defensible, non-inflated claim is: *SBERT was the strongest single
matcher and uniquely recovered semantically-hard cases (cross-lingual
synonyms, abbreviation expansion, names buried in free-text notes) that
TF-IDF and fuzzy matching cannot reach — though for this particular,
lexical-error-dominated dataset, classical methods plus calibration were
competitive on headline accuracy.* Knowing that distinction — and
measuring it instead of assuming it — is the actual engineering result.

## 6. Reproducing the numbers

```bash
python src/finetune_sbert.py          # fine-tunes and saves models/sbert-hospital-matcher
python src/evaluate.py                 # leak-free baselines + ensemble accuracy + threshold sweep
python src/ablation_sbert.py           # with-vs-without SBERT + SBERT's unique wins
cat results/metrics_summary.json results/ablation_sbert.json
```

## 7. Further levers if more headroom is needed

- **More fine-tuning epochs / hard-negative mining**: the current SBERT
  fine-tune is 1 epoch with in-batch negatives only; training loss was
  still ~1.0 (not converged) when it stopped. More epochs and explicit
  hard negatives (confusable hospitals in the same city/chain) likely
  have headroom left.
- **Canonical label cleanup**: a one-time pass to merge near-duplicate
  canonical names (`"RS BUN"` / `"RS BUNDA JAKARTA"`, `"PRODIA"` /
  `"PRODIA TANGERANG"`) would remove some of the irreducible-looking
  errors in §3 that are actually just inconsistent labeling.
- **Aliasing dictionary** for business-specific abbreviations no
  similarity model can infer (e.g. "AP", "MHJS").
- **Active learning**: every `NEEDS_REVIEW` row a human resolves is a
  free new training pair for the next `finetune_sbert.py` run.
