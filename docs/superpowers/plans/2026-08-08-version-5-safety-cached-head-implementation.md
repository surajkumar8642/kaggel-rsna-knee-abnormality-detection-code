# Version 5 Browser-Only Safety and Cached-Head Implementation Plan

> **Execution rule:** All code authoring, functional tests, fixes, smoke runs, and experiments happen in the signed-in Kaggle browser notebook. Local storage is used only for plans, downloaded/sanitized notebook snapshots, aggregate evidence, and Git history.

**Goal:** Repair hidden-test ordering and fold-local supervision, reproduce Version 3 from its pinned feature cache, and evaluate sequential cached-head improvements without another DICOM/DINO pass.

**Authoritative environment:** Private Kaggle notebook, Python 3.12, NumPy, pandas, scikit-learn, PyTorch, competition input, exact Version 3 output, Internet off, accelerator None unless a measured promoted run needs T4.

---

## Task 1: Create an isolated private Version 5 notebook

- [ ] In Chrome, create or copy a private notebook named RSNA Knee V5 Cached Head Lab.
- [ ] Attach the RSNA competition input and exact Version 3 output (scriptVersionId 340880172).
- [ ] Keep Internet off and accelerator None.
- [ ] Add a Markdown overview stating that this notebook is OOF-only and must not create submission.csv.
- [ ] Add RUN_STAGE with allowed values contract, repro, screen, confirm and default contract.
- [ ] Print only aggregate counts, shapes, hashes, metrics, and runtimes.
- [ ] Save a quick source version only after the initial contract cells parse.

## Task 2: Browser TDD for immutable run contracts

- [ ] Add a contract-test cell before the implementation cell.
- [ ] Run it and require the expected NameError for the not-yet-defined contract objects.
- [ ] Implement the exact 12-target tuple, source version 340880172, expected cache dimensions (4407,24,384), fold seed 20260808, and registered H0-H9 experiment configs.
- [ ] Rerun the test cell and require every assertion to pass.
- [ ] Reject unknown experiment names and mutable ad hoc overrides.
- [ ] Record the aggregate marker V5 CONTRACT OBJECTS PASSED.

## Task 3: Browser TDD for sample-driven hidden-test ordering

- [ ] Add synthetic in-memory sample/test frames whose ID orders differ.
- [ ] Call align_test_to_sample before defining it and require NameError.
- [ ] Implement non-null uniqueness checks, exact ID-set equality, one-to-one left merge from sample IDs, and final NumPy order equality.
- [ ] Implement validate_predictions for exact columns, row count, ID order, uniqueness, numeric types, finiteness, and [0,1].
- [ ] Prove duplicate IDs, missing IDs, extra IDs, reversed output order, NaN, infinity, and out-of-range values fail.
- [ ] Apply the function to runtime test.csv and sample_submission.csv before any feature extraction.
- [ ] Assert exact ID equality again immediately before any future atomic write.
- [ ] Do not print IDs or prediction rows.

## Task 4: Browser TDD for fold-local supervision

- [ ] Add a six-row/two-target synthetic fixture.
- [ ] Call build_fold_supervision before defining it and require NameError.
- [ ] Implement a fresh supervision bundle per fold using copies of parser arrays.
- [ ] Gate weak labels only on non-gold rows.
- [ ] Set disabled weak values to unknown and weights to zero.
- [ ] Override non-held-out official cells with exact 0/1 and the configured trusted-gold weight.
- [ ] Zero all held-out weights defensively and exclude held-out rows from the loader.
- [ ] Compute sampler weights and positive weights from fold-training rows only.
- [ ] Mutate every held-out parser and official value, rebuild, and require byte-identical enabled masks, training indices, values, weights, sampler weights, and positive weights.
- [ ] Require raw weak arrays to remain unchanged.
- [ ] Record the aggregate marker FOLD SUPERVISION REGRESSION PASSED.

## Task 5: Pin and validate the Version 3 cache

- [ ] Search only under the explicitly attached Version 3 input root for train_full_8.npz.
- [ ] Require exactly one match; zero or multiple matches abort.
- [ ] Compute and print only file size and SHA-256.
- [ ] Load with allow_pickle=False.
- [ ] Require features float16 (4407,24,384), planes int8 (4407,24), masks bool (4407,24).
- [ ] Require finite features, active planes in 0..2, inactive planes -1, inactive features zero, every bag nonempty, and fallback_studies zero.
- [ ] Hash the current train.csv ID order without printing it and record the legacy row-order provenance limitation.
- [ ] Add a negative synthetic cache test for wrong dtype, wrong shape, NaN, invalid plane, nonzero padding, empty bag, and nonzero fallback.
- [ ] Record the aggregate marker V5 CACHE CONTRACT PASSED.

## Task 6: Reproduce exact folds and OOF accounting

- [ ] Add synthetic tests before the fold implementation and require NameError.
- [ ] Implement only the deterministic greedy multi-label split used by Version 3; do not conditionally change algorithms.
- [ ] Require 58 official-label studies, fold sizes 12/12/12/11/11, no overlap, one validation assignment per study, and a stable assignment SHA-256.
- [ ] Implement an OOF accumulator of shape (58,12) with NaN initialization and integer assignment counts.
- [ ] Reject duplicate writes, missing rows, wrong shapes, and non-finite predictions.
- [ ] Average all predeclared seeds inside each fold before one OOF write.
- [ ] Recompute per-target and macro AUC from official labels and require stored/displayed equality within 1e-12.
- [ ] Implement a fixed-seed 2,000-resample paired study bootstrap.
- [ ] Record the aggregate marker V5 FOLD AND OOF CONTRACT PASSED.

## Task 7: Reproduce the Version 3 MIL head

- [ ] Add CPU synthetic tests before the model implementation and require NameError.
- [ ] Implement the Version 3 target-attention MIL behavior exactly.
- [ ] Require output shape [batch,12], finite logits and gradients, missing-plane support, invalid-plane rejection, padding mutation invariance, and deterministic tiny-batch loss reduction.
- [ ] Implement the same weighted masked BCE, AdamW settings, LR 3e-4, weight decay 1e-3, batch 32, 20 epochs, patience 4, gradient clip 1.0, and seed formula.
- [ ] Run only CPU synthetic tests first.
- [ ] Record the aggregate marker V5 MODEL CONTRACT PASSED.

## Task 8: H0 reproduction gate

- [ ] Run H0 with the exact cache, folds, raw Version 3 supervision, head, and seeds.
- [ ] Start on CPU; enable T4 only if measured projection materially benefits.
- [ ] Require exactly 58 OOF rows, 12 scorable targets, finite predictions, matching cache/fold/config hashes, and macro AUC in [0.6230,0.6330].
- [ ] If H0 fails, stop experiments and compare cache SHA, fold digest, raw supervision, model hyperparameters, and seed behavior one at a time.
- [ ] Do not run H1 until H0 passes.
- [ ] Stop T4 immediately if used.

## Task 9: Sequential one-factor cached experiments

Run every experiment against the current champion. Never stack a failed variant.

- [ ] H1: correct fold-local gating only.
- [ ] H2: trusted-gold multiplier 2.0 only.
- [ ] H3: multiplier 3.0 only if H2 is promising but inconclusive.
- [ ] H4: three fixed seeds averaged without seed selection.
- [ ] H5: full/even/odd bag-mask inference with fixed 0.50/0.25/0.25 blend.
- [ ] H6: fixed predeclared EMA.
- [ ] H7: residual-statistics MIL using attention plus masked mean/max.
- [ ] H8: hierarchical plane-aware MIL.
- [ ] H9: a second deterministic CV repetition only for a promoted candidate.

For each candidate require correctness gates, macro delta at least 0.005, paired-bootstrap probability at least 0.75 with at least 1,500 valid samples, at least 7/12 targets non-worse within 0.01, and no early-screen target drop above 0.10.

## Task 10: Champion confirmation

- [ ] Confirm only candidates that pass their single-factor screen.
- [ ] Require rank-averaged OOF macro at least 0.6450.
- [ ] Require mean single-seed macro at least 0.6380 and seed standard deviation at most 0.015.
- [ ] Require at least 7/12 target improvements and no more than two target drops above 0.03.
- [ ] Freeze exactly one champion config hash, or retain Version 3 if none passes.
- [ ] Save only private aggregate manifests and private checkpoints.
- [ ] Do not create submission.csv.

## Task 11: Independent browser-run review

- [ ] A safety reviewer checks notebook cells, executed outputs, ordering, held-out immutability, hashes, OOF counts, and privacy.
- [ ] A model reviewer checks identical comparisons, target deltas, bootstrap validity, promotion logic, and runtime.
- [ ] Fix any issue in the Kaggle browser, rerun the failing regression there, then rerun affected downstream cells.
- [ ] Save a private Kaggle version only after both reviews approve.
- [ ] Stop the session and set accelerator None.

## Task 12: Local storage and Git handoff

- [ ] Download the exact reviewed Kaggle notebook.
- [ ] Sanitize outputs, execution counts, transient metadata, and empty cells for storage.
- [ ] Store the sanitized notebook, this plan, and aggregate evidence only.
- [ ] Never store reports, IDs, DICOMs, labels, features, checkpoints, credentials, or submission files locally.
- [ ] Review Git diff for privacy and unintended files.
- [ ] Commit and push the code-only snapshot to the existing draft PR.
- [ ] Record Kaggle version, runtime, aggregate metrics, accelerator-off state, and next phase in CODEX_STATUS.md.

The report-supervision implementation plan is written only after the cached-head champion is frozen. The imaging plan is written only after the report-supervision champion is frozen.
