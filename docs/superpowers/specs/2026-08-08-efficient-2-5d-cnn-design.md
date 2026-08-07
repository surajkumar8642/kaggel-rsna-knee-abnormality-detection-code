# Efficient 2.5D CNN Baseline Design

## Purpose

Build a reliable, human-readable machine-learning baseline for the RSNA Knee Abnormality Detection competition. The solution will train on knee MRI DICOM series, predict all twelve competition targets, run in Kaggle's notebook-only environment with internet disabled, and finish within the nine-hour runtime limit.

The first goal is a complete and trustworthy workflow rather than a leaderboard-grade ensemble. A reader should understand what the model does, why each step exists, and whether the generated submission is valid.

## Competition requirements

- Input studies contain several MRI series stored as DICOM slices.
- Series metadata identifies anatomical plane, fluid sensitivity, and fat suppression.
- Training labels may be missing for some studies or conditions.
- Required targets are `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, and `Fracture`.
- Evaluation uses macro-averaged ROC AUC across the twelve targets.
- The scoring notebook must create `/kaggle/working/submission.csv` with the exact sample-submission rows and columns.
- Internet access must be disabled and CPU or GPU runtime must not exceed nine hours.
- Kaggle replaces the visible example test data with different hidden data during scoring.
- The dataset is approximately 570 GB, so the baseline cannot read every slice.
- A maximum of five submissions per day is allowed, with up to two final submissions selected for judging.

## Rules and license safeguards

The implementation will follow the competition Rules, Kaggle's Code Competition guidance, and the RSNA MIRA Dataset Research Use Agreement.

- Competition data, DICOM images, reports, CSV rows, identifiers, or derived copies will not be committed to GitHub, placed in the code dataset, or shared with non-participants.
- The workflow will not attempt to identify or re-identify any person.
- The dataset is for research use and is not FDA reviewed. Outputs are not for diagnosis, clinical decision-making, or patient care.
- Required RSNA notices will not be removed or altered.
- Only synthetic data and the small hand-written schema fixture will be used in public automated tests.
- Publicly shared competition code must also be shared through the competition's Kaggle notebook or forum. The tested notebook will therefore be the Kaggle companion to the GitHub repository.
- Dependencies must use licenses compatible with the competition's open-source and commercial-use conditions. The first version will use permissively licensed Python, NumPy, pandas, PyTorch, pydicom, scikit-learn, and Pillow components.
- External data or model weights will not be used in the first version. A later version may use them only if they are free, reasonably accessible to all competitors, rule-compliant, documented, and available while notebook internet is disabled.
- No submission will be sent automatically. Saving a tested notebook version and submitting it are separate actions.
- If the work becomes prize-eligible, delivery must include training code, inference code, model weights, environment details, reproducible instructions, and the required winner license and documentation.

## Recommended architecture

### 1. Repository structure

The single-file workflow will become a small package with clear responsibilities:

- `src/rsna_knee/constants.py` owns target names and stable competition constants.
- `src/rsna_knee/config.py` owns validated, serializable training and inference settings.
- `src/rsna_knee/data.py` loads CSV metadata, discovers series, samples slices, decodes DICOM data, and prepares tensors.
- `src/rsna_knee/model.py` defines the compact 2.5D CNN and masked multi-label loss.
- `src/rsna_knee/training.py` creates deterministic splits, trains the model, saves checkpoints, and reports study-level metrics.
- `src/rsna_knee/inference.py` performs study-level prediction and creates a validated submission.
- `src/submission_workflow.py` remains as a compatible command-line entry point for simple build and validation tasks.
- `notebooks/rsna-knee-2-5d-baseline.ipynb` is the human-readable Kaggle runner and is versioned in GitHub.
- `tests/` contains fast tests using synthetic metadata and tiny generated DICOM files.

The notebook will import tested package code instead of containing a second implementation. Kaggle will receive the same repository snapshot used by local tests.

### 2. DICOM and series preprocessing

For each selected series, the loader will:

1. Sort slices using spatial position when available, then instance number, then filename as a deterministic fallback.
2. Select evenly spaced center-region slice triplets instead of reading the entire series.
3. Decode pixel data with `pydicom`, apply rescale slope and intercept, and invert `MONOCHROME1` images.
4. Replace non-finite values, clip intensities to robust percentiles, normalize to `[0, 1]`, and resize to a configurable square image size.
5. Stack three adjacent slices as the input channels, providing limited through-plane context with efficient 2D convolutions.

Unreadable or unsupported DICOM files will be counted and skipped. A series with too few valid slices will repeat the nearest valid slice. A study with no readable images will receive a fallback prediction instead of crashing the run.

### 3. Study sampling

The baseline will cap work per study:

- Prefer a balanced selection of sagittal, coronal, and axial series.
- Use fluid-sensitive and fat-suppression metadata as preference signals, not hard requirements.
- Limit series and slice triplets through configuration.
- Provide a small smoke configuration and a larger baseline configuration.

Sampling will be deterministic for validation and inference. Training will use seeded light augmentation such as horizontal flips, small rotations, and intensity variation.

### 4. Model

The first model will be a compact CNN trained from scratch so it does not depend on internet downloads or privately cached weights. It will contain:

- A small convolutional feature extractor for 2.5D three-channel inputs.
- Global average pooling.
- A twelve-output classification head.
- Sigmoid probabilities at inference time.

Training will use masked binary cross-entropy so missing labels do not affect the loss. Positive-class weights will be calculated from the training split, clipped to a safe range, and used only when both classes are present.

Slice-triplet logits will be averaged to produce one probability per target per study.

### 5. Training and validation

The workflow will use a deterministic, study-level, multi-label-aware train/validation split with no study overlap. It will report label counts and prevalence in both splits.

Training will include:

- Reproducible seeds for Python, NumPy, and PyTorch.
- Automatic GPU use when CUDA is available and CPU fallback otherwise.
- Mixed precision on supported GPUs.
- Configurable study limits, batch size, epochs, and worker count.
- Early stopping based on validation macro AUC when enough positive and negative samples exist.
- Checkpointing of the best model and its complete configuration.

Validation will calculate AUC per target only when the target contains both classes. The macro score will average valid target AUCs and list targets that cannot be scored.

### 6. Small-data-first execution

Notebook development will use two explicit modes:

- `smoke`: a small number of real studies, one short epoch, minimal series and slices, and full structural checks.
- `baseline`: the larger approved study limit and training schedule chosen from measured smoke-mode throughput.

The baseline mode will run only after smoke mode passes DICOM loading, tensor-shape, loss, checkpoint, inference, and submission validation checks. Runtime measurements will leave a safety margin below nine hours because hidden data can differ from visible example data.

### 7. Inference and submission

Inference will discover `test.csv`, `test_series.csv`, and the test DICOM tree at runtime. It will not hardcode the three example test identifiers.

The workflow will start with `sample_submission.csv`, preserve its row order and exact columns, and replace only target values. If a study cannot be decoded, it will use clipped per-target training prevalence as a safe fallback.

Before writing `submission.csv`, validation will confirm:

- Exact columns and order match `sample_submission.csv`.
- Every sample-submission study appears once and in the expected order.
- No identifiers are missing or duplicated.
- Predictions are numeric, finite, and between zero and one.
- No index column is written.
- The final path is exactly `/kaggle/working/submission.csv` on Kaggle.

The notebook will print a compact report and small preview instead of recursively listing hundreds of thousands of files.

## Human-readable notebook

The generic Kaggle starter cell will be replaced. The GitHub-versioned notebook will contain short Markdown sections:

1. Goal, research-only notice, and competition requirements.
2. Configuration and smoke/baseline mode.
3. Environment, GPU, input-path, and schema checks.
4. How MRI slices become model inputs.
5. Model summary.
6. Training progress.
7. Validation results.
8. Test inference.
9. Submission checks.
10. Artifact locations and next steps.

Visible output will stay concise. Diagnostics will use counts, representative examples, elapsed time, and actionable messages.

## Testing strategy

### Local tests

- Target constants exactly match the twelve live targets.
- Configuration rejects invalid sizes, counts, modes, and probabilities.
- Slice sorting remains deterministic with missing metadata.
- DICOM normalization handles slope, intercept, inversion, constant images, and non-finite values.
- Triplet sampling returns the expected tensor shape.
- Masked loss ignores missing labels and remains finite.
- The model returns a batch-by-twelve tensor.
- Study aggregation returns stable probabilities in `[0, 1]`.
- The train/validation split is deterministic and has no overlap.
- Submission validation catches missing or extra columns, wrong row order, duplicates, non-finite values, and out-of-range predictions.
- A tiny end-to-end test trains and infers on generated DICOM studies.
- Failure of an earlier pipeline step stops the command and cannot be hidden by validation of an old output.

### Kaggle smoke tests

- Confirm accepted rules, competition input paths, and twelve columns.
- Confirm internet is disabled for the saved version.
- Detect the available accelerator and report resource limits.
- Run smoke mode on a small number of real studies.
- Confirm DICOM decoder coverage and report skipped transfer syntaxes.
- Measure data-loading and inference throughput.
- Create and validate `/kaggle/working/submission.csv` for the visible example studies.

### Kaggle baseline tests

- Select study limits and epochs using smoke-mode timing with a safety margin.
- Train and save the best checkpoint.
- Report per-target and macro validation AUC where defined.
- Run complete example-test inference.
- Save and run a notebook version end to end with internet disabled.
- Verify the notebook output contains `submission.csv` and no restricted data artifacts.

The final competition submission remains a separately confirmed action because failed submissions consume the daily limit.

## Error handling and hidden-data robustness

Expected recoverable issues will produce counts and examples rather than full trace dumps. These include unreadable DICOMs, missing optional metadata, incomplete labels, empty series, and studies with no usable images.

Configuration errors, incorrect schemas, empty training sets, output-shape mismatches, and invalid submissions will stop the run with plain-English messages. File access will always consider missing paths. The workflow will never silently validate an old output after a failed build.

Hidden-data robustness will be tested by changing synthetic study counts, slice counts, series combinations, label missingness, and row order. The notebook will derive all test behavior from the runtime sample submission rather than visible example-data assumptions.

## GitHub and Kaggle delivery

Changes will be committed in focused units after tests pass. The README will explain local testing, Kaggle setup, notebook execution, expected runtime, smoke versus baseline mode, licensing safeguards, and the difference between saving a version and submitting.

Generated competition data, copied reports, DICOMs, identifiers, model checkpoints, credentials, submissions, and Kaggle outputs will remain excluded from Git.

After local and Kaggle verification, the tested code and notebook will be pushed to the existing `origin` repository without rewriting history. Because the repository is public, the matching tested notebook will also be shared through the competition's Kaggle Code area in accordance with the public-code-sharing rule. Public sharing and competition submission will each receive an action-time confirmation.

## Explicit non-goals for the first version

- No 3D transformer or multi-model ensemble.
- No report-language model or weak-label generation.
- No external data or pretrained weights.
- No publication of competition data or derived dataset copies.
- No automatic public dataset or notebook publication.
- No automatic competition submission.
- No claim of medical validation or clinical suitability.

These may be reconsidered only after the reliable baseline provides measured runtime and validation evidence.
