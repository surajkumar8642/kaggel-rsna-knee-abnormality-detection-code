# Version 5 Safety and Cached-Head Laboratory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Repair hidden-test ordering and fold-local supervision, reproduce the Version 3 OOF result from its pinned feature cache, and evaluate one-factor cached-head improvements without another DICOM/DINO pass.

**Architecture:** Reusable, synthetic-testable modules live under src/rsna_v5. A deterministic builder inlines those modules into a private Kaggle notebook so local and browser code cannot silently diverge. The notebook attaches the exact Version 3 output, validates provenance, runs an immutable registry, writes only private aggregate experiment evidence, and never creates a competition submission.

**Tech Stack:** Python 3.12, NumPy, pandas, scikit-learn, PyTorch, pytest, nbformat, Kaggle T4 only for promoted head runs, and signed-in Chrome for Kaggle execution.

---

## File map

- Modify requirements-dev.txt: add pytest.
- Create src/rsna_v5/contracts.py: immutable targets, cache, and experiment configs.
- Create src/rsna_v5/ordering.py: sample-driven test ordering and prediction validation.
- Create src/rsna_v5/supervision.py: pure fold-local supervision.
- Create src/rsna_v5/folds.py: pinned deterministic folds and digest.
- Create src/rsna_v5/cache.py: cache provenance and array checks.
- Create src/rsna_v5/data.py: cached-bag dataset and loaders.
- Create src/rsna_v5/models.py: V3, residual-statistics, and plane-aware MIL.
- Create src/rsna_v5/metrics.py: exact OOF, AUC, and paired bootstrap.
- Create src/rsna_v5/training.py: common trainer and seed averaging.
- Create src/rsna_v5/runner.py: immutable registry execution.
- Create scripts/build_v5_cached_notebook.py: deterministic notebook builder.
- Create notebooks/rsna-knee-v5-cached-head-experiments.ipynb: generated notebook.
- Create tests/rsna_v5/: focused synthetic tests.
- Modify scripts/test_pipeline.ps1, README.md, and CODEX_STATUS.md.

## Task 1: Test environment and immutable contracts

**Files:**
- Modify requirements-dev.txt
- Create src/rsna_v5/__init__.py
- Create src/rsna_v5/contracts.py
- Create tests/rsna_v5/test_contracts.py

- [ ] **Step 1: Add the test dependency**

Append pytest>=8.3 to requirements-dev.txt, create .venv, and install requirements-dev.txt.

- [ ] **Step 2: Write failing contract tests**

~~~python
from dataclasses import FrozenInstanceError
import pytest
from src.rsna_v5.contracts import EXPERIMENTS, TARGETS, CacheContract, get_experiment

def test_exact_targets():
    assert TARGETS == (
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    )

def test_contract_is_frozen():
    contract = CacheContract(path="train_full_8.npz", sha256="a" * 64)
    with pytest.raises(FrozenInstanceError):
        contract.train_rows = 1

def test_ad_hoc_experiment_is_rejected():
    assert "v3_repro" in EXPERIMENTS
    with pytest.raises(KeyError, match="unregistered experiment"):
        get_experiment("ad_hoc")
~~~

- [ ] **Step 3: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_contracts.py -q
~~~

Expected: import failure because src.rsna_v5.contracts does not exist.

- [ ] **Step 4: Implement minimal contracts**

Define frozen CacheContract and ExperimentConfig, exact TARGETS, SOURCE_SCRIPT_VERSION=340880172, expected cache dimensions (4407,24,384), and H0-H9 from the design. get_experiment must accept only registered names.

- [ ] **Step 5: Run GREEN and commit**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_contracts.py -q
git add requirements-dev.txt src/rsna_v5 tests/rsna_v5/test_contracts.py
git commit -m "test: define Version 5 experiment contracts"
~~~

Expected: 3 passed.

## Task 2: Sample-driven test ordering

**Files:**
- Create src/rsna_v5/ordering.py
- Create tests/rsna_v5/test_ordering.py

- [ ] **Step 1: Write failing tests**

~~~python
import numpy as np
import pandas as pd
import pytest
from src.rsna_v5.ordering import align_test_to_sample, validate_predictions

def test_rows_follow_sample_order():
    sample = pd.DataFrame({"id": ["s2", "s1"], "A": [0.0, 0.0]})
    test = pd.DataFrame({"id": ["s1", "s2"], "value": [1, 2]})
    aligned = align_test_to_sample(test, sample, "id")
    assert aligned["id"].tolist() == ["s2", "s1"]
    assert aligned["value"].tolist() == [2, 1]

@pytest.mark.parametrize("bad", [["s1", "s1"], ["s1", "s3"]])
def test_duplicate_or_mismatched_ids_fail(bad):
    sample = pd.DataFrame({"id": ["s1", "s2"], "A": [0.0, 0.0]})
    with pytest.raises(ValueError):
        align_test_to_sample(pd.DataFrame({"id": bad}), sample, "id")

def test_prediction_validation_rejects_nan():
    sample = pd.DataFrame({"id": ["s2", "s1"], "A": [0.0, 0.0]})
    pred = pd.DataFrame({"id": ["s2", "s1"], "A": [np.nan, 0.8]})
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(pred, sample, "id", ("A",))
~~~

- [ ] **Step 2: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_ordering.py -q
~~~

- [ ] **Step 3: Implement exact ordering**

align_test_to_sample checks non-null uniqueness and set equality, then performs sample[[id]].merge(test, how="left", validate="one_to_one", sort=False). validate_predictions checks exact columns, row count, ID order/uniqueness, numeric dtype, finiteness, and [0,1] without printing rows.

- [ ] **Step 4: Run GREEN and commit**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_ordering.py -q
git add src/rsna_v5/ordering.py tests/rsna_v5/test_ordering.py
git commit -m "fix: enforce sample-driven test ordering"
~~~

## Task 3: Fold-local supervision and held-out immutability

**Files:**
- Create src/rsna_v5/supervision.py
- Create tests/rsna_v5/test_supervision.py

- [ ] **Step 1: Write failing supervision tests**

Build a six-row/two-target fixture. Assert disabled non-gold weights are zero and values NaN, training-gold official labels override weak values with the configured multiplier, and held-out rows have zero weight. Add a mutation test that changes every held-out parser and official value, rebuilds, and requires exact equality of enabled mask, training indices, target/weight arrays, sampler weights, and positive weights.

- [ ] **Step 2: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_supervision.py -q
~~~

- [ ] **Step 3: Implement the pure API**

Implement build_fold_supervision with positional NumPy-array parameters named
weak_values, weak_weights, official_values, training_gold_positions,
heldout_positions, enabled_targets, and gold_multiplier, returning a frozen
SupervisionBundle. It copies inputs, gates only non-gold weak cells, applies exact
official overrides to training-gold known cells, zeros held-out weights, derives
usable rows from positive supervision mass, and computes bounded positive weights
only from training rows.

- [ ] **Step 4: Run GREEN and commit**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_supervision.py -q
git add src/rsna_v5/supervision.py tests/rsna_v5/test_supervision.py
git commit -m "fix: build leakage-safe fold supervision"
~~~

## Task 4: Pinned folds and exact OOF metrics

**Files:**
- Create src/rsna_v5/folds.py
- Create src/rsna_v5/metrics.py
- Create tests/rsna_v5/test_folds_metrics.py

- [ ] **Step 1: Write failing fold tests**

Use synthetic multi-label gold rows. Two calls must return identical assignments and SHA-256; no fold may overlap; all validation assignment counts must equal one.

- [ ] **Step 2: Write failing OOF tests**

OOFAccumulator(58,12) must reject duplicate writes, missing rows, wrong shape, and non-finite predictions. Three seed arrays must be averaged inside a fold before one assignment is recorded.

- [ ] **Step 3: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_folds_metrics.py -q
~~~

- [ ] **Step 4: Implement fixed Version 3-compatible folds**

Port the deterministic greedy multi-label study split and expose build_gold_folds(labels, ids, seed=20260808). Hash assignment bytes and seed. Never switch algorithm based on optional packages.

- [ ] **Step 5: Implement metrics**

OOFAccumulator stores float64 NaNs plus integer counts. finalize requires all counts one. auc_report returns per-target, scorable count, and macro. paired_bootstrap_probability performs 2,000 fixed-seed study resamples and reports probability plus valid count.

- [ ] **Step 6: Run GREEN and commit**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_folds_metrics.py -q
git add src/rsna_v5/folds.py src/rsna_v5/metrics.py tests/rsna_v5/test_folds_metrics.py
git commit -m "feat: pin folds and OOF metrics"
~~~

## Task 5: Exact feature-cache provenance

**Files:**
- Create src/rsna_v5/cache.py
- Create tests/rsna_v5/test_cache.py

- [ ] **Step 1: Write failing cache tests**

Create tiny temporary NPZ files. Cover valid load, wrong SHA, dtype/shape mismatch, non-finite features, invalid active plane, nonzero padding, empty bag, fallback count, and zero/two filename matches.

- [ ] **Step 2: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_cache.py -q
~~~

- [ ] **Step 3: Implement exact resolution and loading**

resolve_exact_cache requires exactly one explicit filename below the attached Version 3 root. load_cache hashes before np.load(allow_pickle=False), validates the eight known arrays, disables write flags, and returns FeaturePayload. It rejects cache discovery by newest-time or first-match behavior.

- [ ] **Step 4: Run GREEN and commit**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_cache.py -q
git add src/rsna_v5/cache.py tests/rsna_v5/test_cache.py
git commit -m "feat: validate pinned DINO cache"
~~~

## Task 6: Cached data and three bounded MIL heads

**Files:**
- Create src/rsna_v5/data.py
- Create src/rsna_v5/models.py
- Create tests/rsna_v5/test_data_models.py

- [ ] **Step 1: Write failing tests**

Test batch shapes, deterministic loaders, output [B,12], finite gradients, missing planes, invalid plane rejection, padded-feature mutation invariance, and a 16-example synthetic overfit whose BCE falls by at least 50%.

- [ ] **Step 2: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_data_models.py -q
~~~

- [ ] **Step 3: Implement dataset**

Items contain features, planes, mask, targets, and weights only. Convert float16 features to float32 per item. Replace inactive plane -1 with 0 only after preserving its mask.

- [ ] **Step 4: Implement models**

TargetAttentionMILV3 preserves Version 3 behavior. ResidualStatisticsMIL combines target attention with masked global mean/max. HierarchicalPlaneMIL attends per plane, masks absent planes, learns target-plane gates, and adds a global mean residual. All expose forward(features, planes, mask).

- [ ] **Step 5: Run GREEN and commit**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_data_models.py -q
git add src/rsna_v5/data.py src/rsna_v5/models.py tests/rsna_v5/test_data_models.py
git commit -m "feat: add cached MIL heads"
~~~

## Task 7: Common trainer and immutable runner

**Files:**
- Create src/rsna_v5/training.py
- Create src/rsna_v5/runner.py
- Create tests/rsna_v5/test_training_runner.py

- [ ] **Step 1: Write failing tests**

Cover deterministic repeats, held-out exclusion, finite loss/gradient, checkpoint hash match, fixed seed list, seed averaging before OOF write, and rejection of runtime registry overrides.

- [ ] **Step 2: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_training_runner.py -q
~~~

- [ ] **Step 3: Implement Version 3 training**

Use AdamW, LR 3e-4, weight decay 1e-3, batch 32, 20 epochs, patience 4, gradient clip 1.0, and seed 20260808 + 1000*seed_index + fold_index. Preserve the Version 3 macro-AUC/validation-loss selection rule.

- [ ] **Step 4: Implement runner**

For each fold, build supervision, train all registered seeds, average held-out seed predictions, write once to OOF, then return aggregate config/fold/cache hashes, per-target/macro AUC, seed variability, paired bootstrap, and runtime. Never write a submission CSV.

- [ ] **Step 5: Run GREEN and commit**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_training_runner.py -q
git add src/rsna_v5/training.py src/rsna_v5/runner.py tests/rsna_v5/test_training_runner.py
git commit -m "feat: run immutable cached experiments"
~~~

## Task 8: Deterministic private Kaggle notebook

**Files:**
- Create scripts/build_v5_cached_notebook.py
- Create tests/rsna_v5/test_notebook_builder.py
- Create notebooks/rsna-knee-v5-cached-head-experiments.ipynb

- [ ] **Step 1: Write failing builder tests**

Require one overview cell, ordered module cells, zero outputs, null execution counts, no credentials/UID literals, AST-valid code, and byte-identical rebuilds.

- [ ] **Step 2: Run RED**

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_notebook_builder.py -q
~~~

- [ ] **Step 3: Implement builder**

Use nbformat, embed exact UTF-8 module sources in dependency order, then append Kaggle configuration and runner cells. RUN_STAGE defaults to "contract" and accepts only contract, repro, screen, confirm.

- [ ] **Step 4: Build, run GREEN, and commit**

~~~powershell
.\.venv\Scripts\python.exe scripts/build_v5_cached_notebook.py
.\.venv\Scripts\python.exe -m pytest tests/rsna_v5/test_notebook_builder.py -q
git add scripts/build_v5_cached_notebook.py tests/rsna_v5/test_notebook_builder.py notebooks/rsna-knee-v5-cached-head-experiments.ipynb
git commit -m "feat: build Version 5 head laboratory"
~~~

## Task 9: Full local verification and documentation

**Files:**
- Modify scripts/test_pipeline.ps1
- Modify README.md
- Modify CODEX_STATUS.md

- [ ] **Step 1: Extend the test entrypoint**

After existing synthetic submission validation, invoke the selected Python with -m pytest tests/rsna_v5 -q and preserve immediate nonzero exit behavior.

- [ ] **Step 2: Document boundaries and gates**

Document the pinned Version 3 dependency, experiment names, no-submission boundary, local command, and promotion thresholds.

- [ ] **Step 3: Run fresh verification**

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/test_pipeline.ps1
.\.venv\Scripts\python.exe scripts/build_v5_cached_notebook.py
git diff --exit-code -- notebooks/rsna-knee-v5-cached-head-experiments.ipynb
git diff --check
git status -sb
~~~

Expected: existing synthetic submission valid, all tests pass, notebook build idempotent, and diff check clean.

- [ ] **Step 4: Commit and push**

~~~powershell
git add scripts/test_pipeline.ps1 README.md CODEX_STATUS.md
git commit -m "docs: verify Version 5 cached workflow"
git push
~~~

## Task 10: Kaggle cache contract and reproduction

**Files:**
- Modify source only through TDD and deterministic rebuilds.
- Update CODEX_STATUS.md.

- [ ] **Step 1: Create a new private Kaggle notebook**

Import the generated notebook, attach competition data and exact Version 3 output, keep internet off, accelerator None.

- [ ] **Step 2: Run contract stage**

Record aggregate filename, size, SHA-256, shapes, dtypes, fallback count, fold digest, and source version. Pin the observed SHA through a failing local test, then rebuild.

- [ ] **Step 3: Require the contract marker**

~~~text
V5 CONTRACT PASSED: cache=4407x24x384, folds=5, gold=58, targets=12
~~~

Any mismatch blocks reproduction.

- [ ] **Step 4: Run H0**

Run repro on CPU first; enable T4 only if measured head runtime warrants it. Require 58 OOF rows, 12 scorable targets, matching hashes, and macro AUC from 0.6230 through 0.6330.

- [ ] **Step 5: Diagnose before continuing**

If H0 fails, compare fold digest, raw supervision, cache SHA, model hyperparameters, and seeds. Do not run H1 until H0 passes.

## Task 11: One-factor screens and confirmation

**Files:**
- Update CODEX_STATUS.md.
- Modify code only through test-first local changes.

- [ ] **Step 1: Run registry order**

Run H1 then H2. Run H3 only under its registered condition. Freeze one supervision recipe before H4-H9. Never stack a failed candidate.

- [ ] **Step 2: Apply screen gate**

Require delta at least 0.005, bootstrap probability at least 0.75 with 1,500 valid resamples, at least 7/12 targets non-worse within 0.01, and no screen target drop above 0.10.

- [ ] **Step 3: Confirm promoted candidate**

Require macro at least 0.6450, mean single-seed at least 0.6380, seed standard deviation at most 0.015, at least 7/12 target improvements, and no more than two drops over 0.03.

- [ ] **Step 4: Save and stop**

Save only private aggregate manifests/checkpoints, stop the session, set accelerator None, and do not create, publish, or submit submission.csv.

## Task 12: Independent review and phase handoff

**Files:**
- Modify CODEX_STATUS.md and README.md.
- Update draft PR description if evidence materially changes.

- [ ] **Step 1: Run two independent reviews**

Reviewer one checks spec, leakage, ordering, hashes, OOF, and tests. Reviewer two checks comparison fairness, target deltas, bootstrap, runtime, and promotion.

- [ ] **Step 2: Verify locally again**

Run the complete suite, inspect Git diff, verify local and remote SHA, and verify Kaggle session stopped with accelerator None.

- [ ] **Step 3: Freeze one decision**

If no candidate passes, retain Version 3 and record rejection reasons. If one passes, name exactly one cached-head champion and freeze its config hash.

- [ ] **Step 4: Commit and push handoff**

~~~powershell
git add CODEX_STATUS.md README.md
git commit -m "docs: record Version 5 cached-head results"
git push
~~~

Write the report-supervision implementation plan only after this evidence-backed handoff.
