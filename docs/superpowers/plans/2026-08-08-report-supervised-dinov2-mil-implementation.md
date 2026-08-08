# Report-Supervised DINOv2 MIL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a private Kaggle notebook that converts report-only studies into confidence-weighted supervision, extracts multi-plane DINOv2 features once, trains a leakage-safe five-fold target-attention ensemble, and writes an exact competition submission.

**Architecture:** The existing owned Kaggle notebook remains the runner and retains its contract/submission/device checks. New cells add a deterministic report parser, bounded three-plane slice sampling, a frozen DINOv2 feature cache, a small target-query attention MIL head, five-fold OOF training, rank-averaged inference, and runtime guards. All executable model code is authored and tested in the private Kaggle browser notebook; local files contain only this design and plan until the user resumes GitHub synchronization.

**Tech Stack:** Kaggle Python 3.12, PyTorch 2.10, pandas, NumPy, pydicom, scikit-learn, public Kaggle-hosted DINOv2 ViT-S/14 weights, Chrome browser control.

---

## File and cell map

- Modify in Kaggle browser: private notebook `surajkumar8642/rsna-knee-2-5d-cnn-smoke-and-baseline`.
- Preserve locally for later synchronization: `notebooks/rsna-knee-2-5d-baseline.ipynb`; do not edit it during this browser-only phase.
- Create later only after a successful saved run: a sanitized local copy of the Kaggle notebook. This is outside the present execution scope because GitHub synchronization is paused.
- Update for coordination: `C:\Work\suraj\github\Kaggle.com\rsna-knee-abnormality-detection-code\CODEX_STATUS.md`.
- Never create locally: report exports, DICOMs, identifiers, pseudo-label tables, embeddings, checkpoints, or submissions.

### Task 1: Reframe the notebook and lock run contracts

**Files:**
- Modify in browser: Kaggle notebook Markdown and first code cell.
- Update: `C:\Work\suraj\github\Kaggle.com\rsna-knee-abnormality-detection-code\CODEX_STATUS.md`.

- [ ] **Step 1: Add a failing configuration contract**

Insert a code cell after imports and before data access:

```python
TARGETS = list(sample_submission.columns[1:])
assert TARGETS == [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

RUN_MODE = "contract"
assert RUN_MODE in {"contract", "smoke", "full"}
assert len(TARGETS) == 12
assert train_df["StudyInstanceUID"].is_unique
assert sample_submission["StudyInstanceUID"].is_unique
```

- [ ] **Step 2: Run the contract cell and require the expected failure**

Expected first failure: `NameError` for `sample_submission` or `train_df`, proving the assertions are not validating stale objects.

- [ ] **Step 3: Move CSV loading before the contract and add a run nonce**

```python
from pathlib import Path
import secrets

DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
train_df = pd.read_csv(DATA_ROOT / "train.csv")
train_series_df = pd.read_csv(DATA_ROOT / "train_series.csv")
test_df = pd.read_csv(DATA_ROOT / "test.csv")
test_series_df = pd.read_csv(DATA_ROOT / "test_series.csv")
sample_submission = pd.read_csv(DATA_ROOT / "sample_submission.csv")
RUN_NONCE = secrets.token_hex(8)
```

- [ ] **Step 4: Run the cell and require `CONTRACT PASSED`**

The output must show shapes and target count only. It must not print identifiers, report text, or recursive input paths.

- [ ] **Step 5: Record the browser cell ownership and passing contract in `CODEX_STATUS.md`**

Do not commit or push; GitHub synchronization remains paused.

### Task 2: Build the report weak-label parser test-first

**Files:**
- Modify in browser: new notebook section `Report supervision`.

- [ ] **Step 1: Add parser unit cases that fail before implementation**

```python
PARSER_CASES = [
    ("complete acl tear", "ACL", "positive"),
    ("no acl tear", "ACL", "negative"),
    ("cannot exclude acl tear", "ACL", "unknown"),
    ("status post acl repair", "ACL", "unknown"),
    ("large joint effusion", "Effusion", "positive"),
    ("no joint effusion", "Effusion", "negative"),
    ("degenerative medial meniscus tear", "Medial Meniscus", "positive"),
    ("no acute fracture", "Fracture", "negative"),
    ("marrow contusion lateral femoral condyle", "Contusion", "positive"),
    ("small baker cyst", "Baker's", "positive"),
]

for text, target, expected in PARSER_CASES:
    assert classify_target_mention(text, target).state == expected
```

Expected first failure: `NameError: classify_target_mention is not defined`.

- [ ] **Step 2: Implement typed parser results and bounded context rules**

```python
from dataclasses import dataclass
import re
import unicodedata

@dataclass(frozen=True)
class WeakLabel:
    state: str
    value: float
    weight: float

UNKNOWN = WeakLabel("unknown", np.nan, 0.0)
CONFIDENT_POSITIVE = WeakLabel("positive", 0.95, 0.70)
PROBABLE_POSITIVE = WeakLabel("positive", 0.80, 0.40)
CONFIDENT_NEGATIVE = WeakLabel("negative", 0.05, 0.50)

NEGATION = re.compile(r"\b(no|without|negative for|absent)\b")
UNCERTAINTY = re.compile(r"\b(possible|possibly|may represent|cannot exclude|question of)\b")
HISTORY = re.compile(r"\b(history of|status post|postoperative|reconstruction|repair)\b")

def normalize_report(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = unicodedata.normalize("NFKD", value).lower()
    return re.sub(r"\s+", " ", value).strip()

def mention_context(text: str, start: int, end: int, radius: int = 64) -> str:
    return text[max(0, start - radius):min(len(text), end + radius)]
```

Define a literal `TARGET_CONCEPTS` dictionary for all twelve targets. Each target must have positive concepts and target-specific negative-safe aliases. `classify_target_mention` must return unknown for absent concepts, uncertainty, or history-only mentions; negative when a negation occurs in the bounded context; probable positive for hedged but affirmative chronic findings; and confident positive otherwise.

- [ ] **Step 3: Run parser cases and verify they pass**

Expected output: `PARSER UNIT TESTS PASSED: 10`.

- [ ] **Step 4: Generate in-memory weak labels without exporting rows**

```python
weak_values = np.full((len(train_df), len(TARGETS)), np.nan, dtype=np.float32)
weak_weights = np.zeros_like(weak_values)
report_column = next(c for c in train_df.columns if c.lower() in {"report", "reporttext", "report_text"})

for row_pos, report in enumerate(train_df[report_column].tolist()):
    normalized = normalize_report(report)
    for target_pos, target in enumerate(TARGETS):
        parsed = classify_target_mention(normalized, target)
        weak_values[row_pos, target_pos] = parsed.value
        weak_weights[row_pos, target_pos] = parsed.weight
```

- [ ] **Step 5: Enforce aggregate parser safety gates**

For each target compute coverage, positive prevalence among covered labels, conflict count, and agreement on official labels. Disable a target's weak supervision when coverage is below 1%, above 98%, positive prevalence is below 0.1%, above 80%, or official agreement is below 60%. Print aggregate percentages only.

Expected marker: `REPORT SUPERVISION PASSED: 12 targets checked` plus the number of enabled targets. No report fragment or identifier may appear in output.

### Task 3: Add deterministic multi-plane slice selection

**Files:**
- Modify in browser: new notebook section `MRI sampling`.

- [ ] **Step 1: Add failing sampling tests**

```python
assert uniform_positions(30, 8).tolist() == [1, 5, 9, 13, 16, 20, 24, 28]
assert uniform_positions(3, 8).shape == (8,)
assert set(uniform_positions(3, 8).tolist()) <= {0, 1, 2}

synthetic = pd.DataFrame({
    "SeriesInstanceUID": ["sag", "cor", "ax"],
    "Anatomical_Plane": ["Sagittal", "Coronal", "Axial"],
    "Fluid_Sensitive": [1, 1, 0],
    "Fat_Suppression": [1, 0, 1],
})
chosen = choose_plane_series(synthetic)
assert set(chosen["Anatomical_Plane"]) == {"Sagittal", "Coronal", "Axial"}
assert len(chosen) == 3
```

Expected first failure: one or both functions are undefined.

- [ ] **Step 2: Implement bounded deterministic sampling**

```python
PLANE_TO_INDEX = {"Sagittal": 0, "Coronal": 1, "Axial": 2}

def uniform_positions(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    lo = 0.05 * (length - 1)
    hi = 0.95 * (length - 1)
    return np.rint(np.linspace(lo, hi, count)).astype(np.int64).clip(0, length - 1)

def choose_plane_series(rows: pd.DataFrame) -> pd.DataFrame:
    ranked = rows.copy()
    ranked["plane_rank"] = ranked["Anatomical_Plane"].map(PLANE_TO_INDEX).fillna(99)
    ranked["quality"] = (
        ranked["Fluid_Sensitive"].fillna(0).astype(int) * 2
        + ranked["Fat_Suppression"].fillna(0).astype(int)
    )
    ranked = ranked.sort_values(
        ["plane_rank", "quality", "SeriesInstanceUID"],
        ascending=[True, False, True],
        kind="stable",
    )
    return ranked[ranked["plane_rank"] < 99].drop_duplicates("plane_rank")
```

- [ ] **Step 3: Reuse and strengthen the existing real-DICOM checks**

Require 224 × 224 float32 normalized output, finite values, range `[0,1]`, correct `MONOCHROME1` inversion, and deterministic spatial ordering. Expected marker: `MRI SAMPLING PASSED`.

### Task 4: Attach DINOv2 and test frozen feature extraction

**Files:**
- Modify in browser: notebook Input panel and section `Frozen visual encoder`.

- [ ] **Step 1: Attach the public Kaggle DINOv2 ViT-S/14 model input**

Use the same Kaggle-hosted DINOv2 resource visible on the leading public notebook. Keep internet disabled. Record the exact input slug and license in notebook Markdown.

- [ ] **Step 2: Add a failing weight-resolution test before loading CUDA**

```python
dino_candidates = sorted(Path("/kaggle/input").glob("**/*dinov2*vits14*"))
assert dino_candidates, "DINOv2 ViT-S/14 weights are not attached"
```

Expected first result before attachment: assertion failure. After attachment, require exactly one selected weight file and print only its resource-relative name.

- [ ] **Step 3: Keep the architecture-aware CUDA guard and add full-mode refusal**

```python
if RUN_MODE == "full" and not USE_CUDA:
    raise RuntimeError("Full DINOv2 extraction requires a compatible T4 CUDA device")
```

- [ ] **Step 4: Load and freeze the encoder**

Construct DINOv2 ViT-S/14 with the attached weights, call `eval()`, set every parameter's `requires_grad` to `False`, and assert the feature dimension is 384.

- [ ] **Step 5: Run a real-study feature smoke test**

Process one study with two slices per available plane. Require shape `(instances, 384)`, valid plane indices, finite float16 cache values, and no gradients. Expected marker: `DINO FEATURE TEST PASSED`.

- [ ] **Step 6: Stop the GPU immediately if this test fails**

Do not proceed to dataset extraction after any weight, device, shape, finiteness, or runtime assertion failure.

### Task 5: Implement weighted target-attention MIL with TDD

**Files:**
- Modify in browser: notebook section `Target-aware MIL`.

- [ ] **Step 1: Add deterministic failing loss and shape tests**

```python
torch.manual_seed(17)
fake_x = torch.randn(2, 6, 384)
fake_plane = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2]])
fake_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
fake_y = torch.randint(0, 2, (2, 12)).float()
fake_w = torch.ones_like(fake_y)
fake_w[0, 3] = 0

mil = TargetAttentionMIL(384, 192, 12, 3)
logits = mil(fake_x, fake_plane, fake_mask)
assert logits.shape == (2, 12)
loss_a = weighted_masked_bce(logits, fake_y, fake_w)
fake_y[0, 3] = 1 - fake_y[0, 3]
loss_b = weighted_masked_bce(logits, fake_y, fake_w)
assert torch.allclose(loss_a, loss_b)
```

Expected first failure: model or loss function undefined.

- [ ] **Step 2: Implement the MIL head**

Use a 3-entry learned plane embedding of width 32, concatenate it to 384-D features, project to 192-D, layer-normalize, and compute scaled dot-product attention from twelve learned target queries. Mask invalid instances with the minimum finite value before softmax. Apply a `192 → 96 → 1` MLP independently to each target-pooled vector.

- [ ] **Step 3: Implement exact weighted masked BCE**

```python
def weighted_masked_bce(logits, targets, weights, pos_weight=None):
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none", pos_weight=pos_weight
    )
    denom = weights.sum().clamp_min(1.0)
    return (raw * weights).sum() / denom
```

- [ ] **Step 4: Run tests and one optimizer-step regression**

Train on the fixed synthetic batch for 20 steps at learning rate `1e-2`. Require final loss no greater than initial loss and all logits finite. Expected marker: `MIL UNIT TESTS PASSED`.

### Task 6: Build leakage-safe folds and cached-feature training

**Files:**
- Modify in browser: notebook section `Cross-validation`.

- [ ] **Step 1: Add fold invariants before training**

```python
gold_mask = train_df[TARGETS].notna().any(axis=1)
gold_rows = train_df.loc[gold_mask].copy()
assert len(gold_rows) == 58

seen_validation = set()
for fold_id, (train_idx, valid_idx) in enumerate(folds):
    train_ids = set(gold_rows.iloc[train_idx]["StudyInstanceUID"])
    valid_ids = set(gold_rows.iloc[valid_idx]["StudyInstanceUID"])
    assert train_ids.isdisjoint(valid_ids)
    assert seen_validation.isdisjoint(valid_ids)
    seen_validation.update(valid_ids)
assert len(seen_validation) == len(gold_rows)
```

Expected first failure: `folds` undefined.

- [ ] **Step 2: Create five deterministic multi-label folds**

Use iterative multi-label stratification when available in the Kaggle image. If unavailable, greedily assign studies in descending label-rarity order to the fold with the lowest per-target positive count and lowest size. Seed all tie-breaking with `20260808`.

- [ ] **Step 3: Extract features once**

For smoke mode use 24 studies and two slices per plane. For full mode use every training study and eight slices per plane. Save only run-private float16 arrays under `/kaggle/working/cache_<RUN_NONCE>/`. Print aggregate counts, decode failures, missing-plane counts, throughput, elapsed time, and projected total runtime every 100 studies.

- [ ] **Step 4: Add the runtime abort**

After at least 200 studies, estimate full extraction duration from median per-study time. Raise an error and stop if projected notebook runtime exceeds 7.5 hours.

- [ ] **Step 5: Train one smoke fold**

Train two epochs with AdamW, learning rate `3e-4`, weight decay `1e-3`, batch size 32 studies, and gradient clipping at 1.0. Require finite loss and a valid prediction for every held-out study.

- [ ] **Step 6: Train all five full folds only after smoke passes**

Train at most 20 epochs per fold with patience 4. Save only the best state dict for each fold under the nonce directory. At completion, require exactly 58 unique OOF predictions and compute all scorable target AUCs plus macro AUC.

- [ ] **Step 7: Apply the performance gate**

Require OOF macro AUC above 0.5833 and at least ten scorable targets before hidden-test submission preparation. If the gate fails, retain logs, stop compute, and do not save a competition-ready version.

### Task 7: Rank-average inference and independently validate the submission

**Files:**
- Modify in browser: notebook sections `Inference` and `Independent verification`.

- [ ] **Step 1: Extract test embeddings once and run all five heads**

Derive test studies and order from `sample_submission.csv`. Use the identical series and slice pipeline. Count fallback studies and require zero fallbacks in smoke mode unless the source data is genuinely unreadable.

- [ ] **Step 2: Implement deterministic rank averaging**

```python
def rank_average(fold_predictions: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(fold_predictions, axis=0)
    ranked = np.empty_like(stacked, dtype=np.float64)
    for fold in range(stacked.shape[0]):
        for target in range(stacked.shape[2]):
            ranked[fold, :, target] = pd.Series(
                stacked[fold, :, target]
            ).rank(method="average", pct=True).to_numpy()
    return ranked.mean(axis=0).clip(1e-5, 1 - 1e-5)
```

- [ ] **Step 3: Write the nonce atomically**

Write `submission_<RUN_NONCE>.csv` and `run_<RUN_NONCE>.json`, then atomically replace `/kaggle/working/submission.csv`. The manifest contains only the nonce, shapes, aggregate metrics, runtime, fallback counts, configuration, and submission SHA-256.

- [ ] **Step 4: Validate from a fresh read**

Reload the CSV and manifest. Require matching nonce, exact sample columns/order/identifiers, numeric finite probabilities in `[0,1]`, no duplicates, no index column, and current-file modification time later than the run start.

- [ ] **Step 5: Print safe final evidence**

Print shape, prediction min/max/mean, fallback count, elapsed time, OOF macro AUC, scorable targets, and SHA-256. Do not print the submission rows or any identifier.

### Task 8: Browser smoke run, full run, and compute shutdown

**Files:**
- Modify in browser: `RUN_MODE` only.
- Update: `C:\Work\suraj\github\Kaggle.com\rsna-knee-abnormality-detection-code\CODEX_STATUS.md`.

- [ ] **Step 1: Run `contract` with accelerator None**

Require all schema, parser, sampling, synthetic MIL, device-selection, and stale-output tests to pass. Stop the CPU session after inspecting output.

- [ ] **Step 2: Select T4 x2 and run `smoke`**

Confirm the UI reports two T4 devices and the notebook reports supported architectures before DINOv2 loading. Require all stage markers, a valid visible-test submission, and a measured runtime projection below 7.5 hours.

- [ ] **Step 3: Stop the smoke session immediately**

Inspect logs and output, then turn off the interactive session before editing or discussing results.

- [ ] **Step 4: Set `RUN_MODE = "full"` and use Save & Run All once**

Keep internet disabled. Do not start a second full run while the first is active. Monitor aggregate progress and cancel only for a failed gate, non-finite values, unsupported compute, or a projection above 7.5 hours.

- [ ] **Step 5: Verify the saved log and artifact**

Require successful completion, five folds, 58 OOF rows, macro AUC above 0.5833, at least ten scorable targets, exact submission validation, and runtime below 7.5 hours.

- [ ] **Step 6: Stop all compute**

Confirm no Draft Session, GPU session, TPU session, or running saved version remains. Record exact runtime and remaining GPU hours in `CODEX_STATUS.md`.

- [ ] **Step 7: Keep the notebook private and stop before external actions**

Do not publish the notebook, synchronize GitHub, or click the competition Submit button. Those actions remain separate and require current evidence plus action-time authorization.

## Self-review

- Spec coverage: every design requirement maps to Tasks 1–8, including report supervision, DINOv2 input validation, multi-plane sampling, weighted MIL, five-fold leakage controls, rank averaging, runtime aborts, stale-output protection, privacy, and GPU shutdown.
- Placeholder scan: the plan contains no unspecified implementation gaps; literal target names, functions, assertions, configurations, expected failures, and pass markers are provided.
- Type consistency: report outputs use `WeakLabel`; feature arrays are study × instance × 384 plus plane/mask arrays; MIL outputs are batch × 12; fold predictions are fold × study × 12; rank averaging returns study × 12.
- Authorization consistency: browser notebook edits and private Kaggle runs are authorized. Publishing, GitHub synchronization, and competition submission remain excluded.
