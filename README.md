# RSNA Knee Abnormality Detection — 2.5D CNN

Human-readable Kaggle notebook and supporting utilities for the
[RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

## Tested notebook

The source-controlled notebook is:

- `notebooks/rsna-knee-2-5d-baseline.ipynb`

It was edited and executed in the Kaggle browser environment against the attached
competition data. The verified full supervised run used:

- Tesla T4 accelerator with PyTorch 2.10.0 + CUDA 12.8
- all 58 studies containing official labels (49 train / 9 validation)
- up to one series from each anatomical plane
- orientation-aware DICOM ordering and center-slice triplets
- 160 × 160 inputs, three epochs, and a compact GroupNorm CNN
- exact 12-target study-level submission validation

The remaining 4,349 training metadata rows do not contain official target values,
so they are excluded from supervised loss rather than assigned invented labels.

## Notebook flow

1. **Stage 1 — Contract check:** validates the explicit competition paths, package
   environment, and exact 13-column submission schema without recursively scanning
   the approximately 570 GB input tree.
2. **Stage 2 — Model pipeline:** selects series, decodes and normalizes DICOMs,
   trains the compact 2.5D model, performs study-level inference, and atomically
   writes `/kaggle/working/submission.csv`.
3. **Stage 3 — Independent verification:** reloads the submission and temporary
   checkpoint and rechecks shape, order, numeric types, finiteness, and probability
   bounds.

The visible Kaggle test set contains three studies. In a saved competition run,
Kaggle replaces it with the hidden test set; the notebook derives row IDs and order
from the runtime `sample_submission.csv`.

## Run on Kaggle

1. Create or open a Kaggle notebook for the competition.
2. Attach **RSNA Knee Abnormality Detection** as the competition input.
3. Turn internet off and select a GPU accelerator.
4. Import `notebooks/rsna-knee-2-5d-baseline.ipynb`.
5. Run all cells and require both `STAGE 2 PASSED` and `STAGE 3 PASSED`.
6. Confirm `/kaggle/working/submission.csv` exists before saving a version.

Saving a Kaggle version, publishing the notebook, and submitting to the competition
are deliberately separate actions. A competition submission can consume a daily
submission slot and should be confirmed immediately before clicking Submit.

## Repository hygiene

The repository intentionally excludes competition CSVs, DICOMs, reports,
identifiers, credentials, generated submissions, model checkpoints, and archives.
Only synthetic sample rows are tracked under `data/`.

To prepare a downloaded Kaggle notebook for source control:

```powershell
python scripts/sanitize_notebook.py `
  path\to\downloaded.ipynb `
  notebooks\rsna-knee-2-5d-baseline.ipynb
```

The sanitizer removes outputs, execution counts, transient Kaggle metadata, and
empty cells, then validates the notebook structure.

## Important limitations

- This is a research competition baseline, not a clinical diagnostic system.
- The public three-study test run does not measure hidden-test leaderboard quality.
- No external pretrained weights or internet downloads are used.
- Do not print, commit, upload, or redistribute patient/study identifiers, reports,
  DICOM data, or generated submission contents.
