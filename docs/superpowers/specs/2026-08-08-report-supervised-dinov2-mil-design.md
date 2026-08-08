# Report-Supervised DINOv2 MIL Design

## Objective

Replace the 58-study compact-CNN baseline with a substantially stronger but still competition-safe pipeline for the RSNA Knee Abnormality Detection competition. The new pipeline will use the 4,349 previously unused training reports as weak supervision, extract reusable multi-plane MRI features with a public DINOv2 encoder, train a five-fold attention multiple-instance-learning ensemble, and create an exactly validated 12-target submission within Kaggle's nine-hour notebook limit.

This is a research competition system, not a clinical diagnostic product. A high leaderboard position is an objective, not a guarantee. Decisions will be based on leakage-safe out-of-fold evidence and measured runtime rather than repeated public-leaderboard probing.

## Evidence and constraints

- The competition metric is macro ROC AUC across twelve binary targets, so every target has equal weight and prediction ordering matters more than probability calibration.
- `train.csv` has 4,407 studies but only 58 rows with official target labels. The other 4,349 rows contain reports and are the largest available supervision source.
- The verified Version 2 baseline used only the 58 official labels and reached an unstable 0.5833 validation macro AUC on nine validation studies.
- The hidden test set is approximately 1,300 studies and the competition input is approximately 570 GB. The notebook cannot decode every slice from every series.
- Public Kaggle Code evidence shows a DINOv2-based notebook with a 0.835 public score and a 4h50m dual-T4 runtime. This supports DINOv2 as a competitive feature encoder but also sets a warning against expensive repeated end-to-end runs.
- Saved competition notebooks must run with internet disabled and finish in nine hours. GPU availability is limited to 30 hours per week. TPU will not be used because the PyTorch pipeline is not implemented for XLA.
- Kaggle's current PyTorch build cannot execute on a P100 `sm_60`. The existing architecture-aware CUDA guard remains mandatory. Full feature extraction will start only on a compatible T4 allocation.
- Competition data, reports, DICOMs, study identifiers, derived per-study labels, embeddings, checkpoints, and submissions must not enter Git or a public artifact. Code and aggregate metrics may be source controlled later, but the user has paused GitHub synchronization for this phase.

## Considered approaches

### A. Report labels plus frozen multi-plane DINOv2 MIL — selected

Generate positive, negative, or unknown soft labels from reports, extract DINOv2 ViT-S/14 features for a bounded set of slices, and train five small target-aware attention heads. The encoder runs once, while fold models train on the cached features. This provides the best expected gain per GPU-hour and keeps five-fold ensembling affordable.

### B. End-to-end plane-specific visual experts

Fine-tune separate backbones for sagittal, coronal, and axial images and route targets to anatomical experts. This may achieve a higher ceiling, but three encoders and five folds are likely to consume most or all of the weekly GPU budget and risk the nine-hour inference limit.

### C. Self-supervised MRI pretraining and a large ensemble

Pretrain on all unlabeled slices, then train several MIL or 3D-lite models and distill them. This is the most ambitious route but is not the right first full run under the current 30-hour quota. It becomes a later experiment only if Approach A establishes trustworthy OOF gains and measured spare runtime.

## Data contracts

### Target schema

The exact targets remain:

1. `ACL`
2. `MCL`
3. `Medial Meniscus`
4. `Lateral Meniscus`
5. `Medial OA`
6. `Lateral OA`
7. `PF OA`
8. `Effusion`
9. `Synovitis`
10. `Baker's`
11. `Contusion`
12. `Fracture`

The notebook will derive the identifier column, target order, test row order, and final output shape from `sample_submission.csv`. It will not assume the visible three-study test size.

### Report-derived labels

Each unlabeled study-target pair becomes one of four states:

- confident positive with soft value `0.95` and loss weight `0.70`;
- probable positive with soft value `0.80` and loss weight `0.40`;
- confident negative with soft value `0.05` and loss weight `0.50`;
- unknown with no loss contribution.

Official labels remain hard `0/1` values with loss weight `1.0`.

The report parser will normalize Unicode and case, inspect finding and impression text without printing it, and use target-specific concepts plus bounded context rules for negation, uncertainty, prior history, and postoperative wording. Examples of invalid positive evidence include phrases such as “no tear,” “cannot exclude,” “history of repair,” and mentions outside the current finding context. Later positive or negative clauses override earlier general statements only when they address the same target.

The parser will report aggregate coverage, positive prevalence, conflicts, and agreement on official labels. It will never emit report text or study identifiers. A target with implausible coverage or prevalence will be disabled rather than silently generating broad labels.

### Study and series selection

For each study, select at most one deterministic best series from each available anatomical plane. Rank series by recognized plane, usable DICOM count, fluid sensitivity, fat suppression, and stable series identifier. Sample eight uniformly spaced slices from the central 90% of each selected series, for a maximum of 24 images per study.

Each slice is decoded with slope/intercept handling, `MONOCHROME1` inversion, finite-value replacement, robust percentile clipping, square padding, and resize to 224 × 224. The normalized grayscale slice is repeated into three channels for DINOv2. Training-time feature extraction uses only mild intensity changes and no geometry that can destroy anatomy.

## Model architecture

### Frozen encoder

Use the publicly accessible DINOv2 ViT-S/14 weights attached as a Kaggle Model input. The notebook must resolve the attached weight path explicitly and verify the expected model name and embedding dimension before GPU work. Internet stays disabled.

The encoder is frozen for the first implementation. Automatic mixed precision is enabled only on a compatible CUDA device. Two T4s may use data parallelism for feature extraction; fold-head training does not need multiple GPUs.

For every sampled slice, cache a float16 embedding with its plane index and validity mask in `/kaggle/working`. Cache files are private run artifacts and must not be published or copied into Git.

### Attention MIL head

Each study supplies up to 24 embeddings. Add a learned plane embedding, then apply layer normalization and a compact projection. Twelve learned target queries attend independently over valid slice embeddings. Each target query produces an attention-pooled representation followed by a two-layer MLP and one logit.

This target-specific pooling lets ACL, meniscal, osteoarthritis, effusion, and fracture predictions focus on different slices while sharing the expensive visual encoder. Missing planes are handled through masks rather than fabricated images.

Training uses confidence-weighted masked binary cross-entropy. A clipped target-wise positive weight is computed from the current training fold using only non-unknown labels. Early stopping selects the lowest validation BCE only as a fallback; the primary selection statistic is gold-label macro AUC aggregated from out-of-fold predictions.

## Leakage-safe cross-validation

- Build five deterministic folds over the 58 official-label studies using multi-label stratification where possible.
- If a stable patient identifier exists, all studies for one patient must remain in one fold. Otherwise group strictly by study and state that patient grouping was unavailable.
- For fold `k`, exclude held-out gold studies entirely from training, including any label derived from their reports.
- All other officially labeled studies use weight `1.0`; all eligible unlabeled studies use the fixed report parser's soft labels and confidence weights.
- Extract visual embeddings without labels once, then train five independently seeded MIL heads.
- Concatenate the 58 held-out predictions into one OOF table and compute per-target AUC plus macro AUC only after every fold completes.
- Do not tune report rules, series selection, or model structure against public-leaderboard results. A new variant must first improve leakage-safe OOF evidence.

Final hidden-test predictions are the per-target rank average of the five fold models. Rank averaging is appropriate for ROC AUC and reduces scale differences between folds. The current compact CNN is blended only if its OOF predictions are available on the identical 58-study folds and the blend improves macro AUC.

## Compute-efficient execution

The notebook will expose explicit run stages and stop conditions:

1. `contract`: CPU only. Validate files, schema, report parser invariants, DINOv2 input path, and CUDA compatibility.
2. `smoke`: compatible T4 only. Process 24 studies and at most two slices per plane, train one fold for two epochs, run visible-test inference, and validate the submission.
3. `full`: compatible T4 for one feature pass over all train and runtime test studies, then five small fold heads. Stop the interactive session immediately after artifacts and metrics are inspected.

The full configuration uses 24 slices per study, feature batches chosen from measured memory, float16 cached embeddings, pinned-memory prefetch on CUDA, and a conservative worker count. The initial target is at most six hours total on T4 x2, leaving a three-hour hidden-data safety margin. The notebook will print elapsed time and projected completion after fixed study intervals. It will abort before quota waste if the projection exceeds 7.5 hours.

The CPU fallback remains for contract checks and tiny smoke diagnostics. It must refuse the full DINOv2 run on CPU with an actionable message. P100 remains an unsupported full-run accelerator for the installed PyTorch build.

## Validation gates

### Report parser gate

- All twelve targets have configured concepts.
- No official labels are overwritten.
- Every output is finite or explicitly unknown.
- Aggregate coverage, prevalence, conflict rate, and gold-label agreement are printed without identifiers or text.
- Any target outside configured coverage/prevalence bounds is disabled and clearly reported.

### Image and feature gate

- Real DICOM decode, normalization, and slice-selection checks pass.
- A real study produces embeddings of the expected dimension with finite values.
- Plane indices and masks align with embeddings.
- An unsupported GPU selects the safe path before the first model kernel.
- Smoke extraction and inference contain no fallback study unless the reason is counted and reported.

### Model gate

- One optimizer step on synthetic embeddings reduces or preserves a deterministic test loss.
- Unknown labels contribute zero loss.
- Gold and weak-label weights are applied exactly.
- Five folds have no study overlap.
- OOF predictions contain each of the 58 gold studies exactly once.
- Per-target and macro AUC are computed only for targets with both classes.

### Submission gate

- Exact columns and column order match `sample_submission.csv`.
- Exact row identifiers and order match the runtime sample.
- Predictions are numeric, finite, and within `[0, 1]`.
- No duplicate identifiers or index column exist.
- The file is atomically written to `/kaggle/working/submission.csv`.
- The notebook prints only aggregate statistics and a checksum, not patient identifiers.

## Failure handling

Unreadable DICOMs, absent planes, unsupported transfer syntaxes, and empty series are counted. A partially usable study proceeds with masked instances. A study with no usable image uses fold-ensemble target prevalence and increments the fallback counter. Incorrect schemas, missing DINOv2 weights, all-unknown targets, non-finite embeddings, fold leakage, output mismatch, or full mode on unsupported compute stop the run immediately.

No failed cell may be followed by validation of an older checkpoint or submission. Run-specific artifact names contain a random execution nonce, and Stage 3 verifies that the nonce matches the current process.

## Delivery sequence

1. Add report parsing and aggregate diagnostics in the private Kaggle notebook and run CPU contract tests.
2. Attach and verify DINOv2 model weights with internet disabled.
3. Add multi-plane feature extraction and run the 24-study T4 smoke test.
4. Add the weighted attention-MIL head, one-fold smoke training, inference, and strict submission checks.
5. Run the full five-fold pipeline once, inspect OOF/runtime/fallback evidence, and stop compute.
6. Save a private Kaggle version only after the full run succeeds.
7. Do not publish, synchronize GitHub, or submit to the competition until those separate actions are requested at action time.

## Success criteria

The implementation is ready for a competition submission decision only when:

- all contract, parser, feature, model, fold, and submission gates pass in one saved private Kaggle version;
- the run finishes under seven and a half hours on the visible workload;
- all 4,349 report-only studies are either assigned justified partial labels or explicitly counted as unknown;
- OOF macro AUC on the 58 gold studies materially exceeds 0.5833 and at least ten targets are scorable;
- hidden-test inference is projected to remain below nine hours;
- the accelerator session is stopped after inspection; and
- no restricted artifacts leave Kaggle.
