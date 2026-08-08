# RSNA Knee Abnormality Detection — Report-Supervised DINOv2 MIL

Human-readable Kaggle notebook and supporting utilities for the
[RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

## Verified Kaggle baseline

The source-controlled baseline notebook is
`notebooks/rsna-knee-2-5d-baseline.ipynb`. It is a sanitized, code-only copy of
the private Kaggle notebook: outputs, execution counts, transient metadata,
patient data, predictions, checkpoints, and cached features are not tracked.

[Kaggle Version 3](https://www.kaggle.com/code/surajkumar8642/rsna-knee-2-5d-cnn-smoke-and-baseline/log?scriptVersionId=340880172)
completed successfully in 1 hour 32 minutes 29 seconds on T4 x2 with:

- all 4,407 training studies encoded;
- frozen DINOv2 ViT-S/14 multi-plane features;
- confidence-weighted report supervision plus official labels;
- leakage-safe five-fold target-attention MIL training;
- exactly 58 out-of-fold predictions and all 12 targets scorable;
- OOF macro AUC `0.6280`, compared with `0.5833` for the earlier CPU CNN;
- zero train or test fallback studies;
- an exact 3-row x 13-column visible-test submission;
- independent schema, finiteness, probability-range, nonce, checkpoint, and
  checksum validation.

Version 3 is the authoritative completed run. Version 4 is a quick-saved source
and documentation snapshot and did not consume another full training run. The
Kaggle draft session is stopped, the accelerator is set to **None**, and the
competition submission has not been made.

## Version 5 cached-head laboratory

The browser-authored Version 5 notebook is mirrored as both
`notebooks/rsna-knee-v5-cached-head-lab.ipynb` and the easier-to-review
`notebooks/rsna-knee-v5-cached-head-lab.py`. Both are source-only exports of
[private Kaggle Version 3](https://www.kaggle.com/code/surajkumar8642/rsna-knee-v5-cached-head-lab?scriptVersionId=340945234);
they contain no cell outputs or row-level data.

Version 5 reused the exact verified Version 3 feature cache and pinned folds.
Its browser-only contract suite covers ordering, fold-local supervision,
cache provenance, report parsing, cached MIL behavior, deterministic training,
atomic checkpoint identity, exact-once OOF assembly, and paired study-level
bootstrap comparison. The committed Kaggle run completed successfully in 53
seconds on CPU with the experiment execution guard idle.

The bounded T4 experiments produced:

- exact H0 reproduction: OOF macro AUC `0.6280362353`;
- H1 fold-gated supervision: `0.6447939998` (`+0.0167577645`), which failed the
  common screen because the paired study bootstrap probability was `0.7485`
  (required `0.75`) and two targets fell by more than `0.10`;
- H2 trusted-gold weight 2: `0.6176430711`, which was rejected.

H3 and later factors were not run because the predeclared continuation gates
failed. Version 3 remains the confirmed candidate; H1 is recorded as a rejected
high-macro experiment, not a promoted candidate. T4 was used for about ten
minutes across the bounded experiments and corrected audit rerun, then the
session was stopped and the accelerator reset to **None**. No competition
submission was created.

## Notebook flow

1. Validate the current-run environment, competition files, IDs, and exact
   12-target schema without recursively scanning the image tree.
2. Parse reports into high-precision positive, negative, or unknown soft labels
   while printing only aggregate diagnostics.
3. Deterministically select sagittal, coronal, and axial series, physically order
   DICOM slices, normalize intensities, and sample central anatomy.
4. Extract reusable 384-dimensional slice features with the attached offline
   DINOv2 ViT-S/14 model.
5. Train a weighted target-query attention MIL head on five leakage-safe folds.
6. Require exactly one OOF prediction for each of the 58 officially labelled
   studies and compute macro ROC AUC only from official labels.
7. Rank-average fold predictions and independently validate the submission.

## Run on Kaggle

1. Import `notebooks/rsna-knee-2-5d-baseline.ipynb` into a private Kaggle
   competition notebook.
2. Attach the **RSNA Knee Abnormality Detection** competition input and the
   Kaggle-hosted DINOv2 ViT-S/14 weights expected by the notebook.
3. Keep internet off. Use a compatible T4 accelerator for smoke and full feature
   extraction; leave the accelerator off when no run is active.
4. Run contract mode, then a bounded smoke fold, before a full run.
5. Accept a model change only when identical-fold leakage-safe OOF improves on
   `0.6280` and all data, cache, inference, and submission contracts pass.
6. Save the validated notebook version. Publishing code and submitting to the
   competition remain separate explicit actions.

The visible Kaggle test set has three studies. In a saved competition run Kaggle
replaces it with the hidden test set, so the notebook derives identifiers, row
order, and target order from the runtime `sample_submission.csv`.

## Local validation

Run the repository checks with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_pipeline.ps1
```

To prepare a downloaded Kaggle notebook for source control:

```powershell
python scripts/sanitize_notebook.py `
  path\to\downloaded.ipynb `
  notebooks\rsna-knee-2-5d-baseline.ipynb
```

## Repository safety

Do not commit competition CSVs, reports, DICOMs, patient or study identifiers,
derived per-study labels, embeddings, model checkpoints, generated submissions,
or credentials. Only code, documentation, synthetic examples, and aggregate
metrics belong in this public repository.

This is a research competition system, not a clinical diagnostic system. A high
leaderboard rank is an objective, not a guarantee; decisions are gated by
leakage-safe OOF evidence rather than repeated public-leaderboard probing.
