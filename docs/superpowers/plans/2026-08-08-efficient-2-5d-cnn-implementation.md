# Efficient 2.5D CNN Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, and run a human-readable 2.5D CNN baseline that predicts all twelve RSNA knee targets and creates a rule-compliant Kaggle `submission.csv`.

**Architecture:** A small `rsna_knee` package owns configuration, DICOM preprocessing, sampling, modeling, training, and inference. A GitHub-versioned notebook imports the package, runs smoke mode before baseline mode, and derives hidden-test behavior from Kaggle's runtime `sample_submission.csv` rather than visible example assumptions.

**Tech Stack:** Python 3.11, NumPy, pandas, pydicom, Pillow, PyTorch, scikit-learn, pytest, nbformat, PowerShell, Kaggle Notebooks.

---

## File map

- `src/rsna_knee/constants.py`: twelve target names and schema constants.
- `src/rsna_knee/config.py`: immutable validated run configuration.
- `src/rsna_knee/data.py`: DICOM decoding, sorting, triplet sampling, datasets, and split creation.
- `src/rsna_knee/model.py`: compact CNN, masked loss, and study aggregation.
- `src/rsna_knee/training.py`: seeds, AUC metrics, epoch loops, early stopping, and checkpoints.
- `src/rsna_knee/inference.py`: study prediction, fallback values, and submission construction.
- `src/submission_workflow.py`: backward-compatible CLI and strict submission validation.
- `scripts/build_notebook.py`: deterministic notebook generator.
- `notebooks/rsna-knee-2-5d-baseline.ipynb`: GitHub source of the Kaggle notebook.
- `tests/`: synthetic-data unit and integration tests.
- `requirements.txt`: runtime dependencies.
- `requirements-dev.txt`: test and notebook-build dependencies.
- `README.md`: local, Kaggle, licensing, and execution instructions.

## Task 1: Establish the twelve-target contract and configuration

**Files:**
- Create: `src/rsna_knee/__init__.py`
- Create: `src/rsna_knee/constants.py`
- Create: `src/rsna_knee/config.py`
- Create: `tests/test_constants_config.py`
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/test_constants_config.py
import pytest

from rsna_knee.config import RunConfig
from rsna_knee.constants import ID_COLUMN, TARGET_COLUMNS


def test_live_competition_schema_is_exact():
    assert ID_COLUMN == "StudyInstanceUID"
    assert TARGET_COLUMNS == (
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    )


def test_smoke_configuration_is_small_and_valid():
    cfg = RunConfig.smoke()
    assert cfg.mode == "smoke"
    assert cfg.max_train_studies == 24
    assert cfg.epochs == 1
    assert cfg.image_size == 128


@pytest.mark.parametrize("field,value", [
    ("image_size", 0), ("batch_size", 0), ("epochs", 0),
    ("max_series_per_plane", 0), ("triplets_per_series", 0),
    ("validation_fraction", 1.0),
])
def test_invalid_configuration_is_rejected(field, value):
    values = RunConfig.smoke().__dict__ | {field: value}
    with pytest.raises(ValueError):
        RunConfig(**values)
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `python -m pytest tests/test_constants_config.py -q`

Expected: collection fails because `rsna_knee` does not exist.

- [ ] **Step 3: Implement constants and validated configuration**

```python
# src/rsna_knee/constants.py
ID_COLUMN = "StudyInstanceUID"
TARGET_COLUMNS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)
NUM_TARGETS = len(TARGET_COLUMNS)
```

```python
# src/rsna_knee/config.py
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunConfig:
    mode: str
    seed: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    validation_fraction: float
    max_train_studies: int | None
    max_series_per_plane: int
    triplets_per_series: int
    num_workers: int

    def __post_init__(self):
        if self.mode not in {"smoke", "baseline"}:
            raise ValueError("mode must be 'smoke' or 'baseline'")
        for name in ("image_size", "batch_size", "epochs", "max_series_per_plane", "triplets_per_series"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_train_studies is not None and self.max_train_studies <= 1:
            raise ValueError("max_train_studies must exceed one")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

    @classmethod
    def smoke(cls):
        return cls("smoke", 42, 128, 4, 1, 1e-3, 0.25, 24, 1, 1, 0)

    @classmethod
    def baseline(cls):
        return cls("baseline", 42, 192, 16, 4, 5e-4, 0.20, 600, 1, 2, 2)

    def to_dict(self):
        return asdict(self)
```

Expose `RunConfig`, `ID_COLUMN`, `TARGET_COLUMNS`, and `NUM_TARGETS` from `src/rsna_knee/__init__.py`.

- [ ] **Step 4: Add dependencies and run tests**

```text
# requirements.txt
numpy>=1.26
pandas>=2.2
pydicom>=2.4
Pillow>=10.0
scikit-learn>=1.4
torch>=2.2
```

```text
# requirements-dev.txt
-r requirements.txt
nbformat>=5.10
pytest>=8.0
```

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_constants_config.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add requirements.txt requirements-dev.txt src/rsna_knee tests/test_constants_config.py
git commit -m "feat: define RSNA twelve-target configuration"
```

## Task 2: Make submission validation exact and fail-fast

**Files:**
- Modify: `src/submission_workflow.py`
- Modify: `scripts/test_pipeline.ps1`
- Modify: `data/sample_submission_example.csv`
- Create: `tests/test_submission_workflow.py`

- [ ] **Step 1: Write failing submission tests**

```python
# tests/test_submission_workflow.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, TARGET_COLUMNS
from submission_workflow import build_submission, validate_submission


def sample_frame():
    frame = pd.DataFrame({ID_COLUMN: ["S1", "S2", "S3"]})
    for column in TARGET_COLUMNS:
        frame[column] = 0.5
    return frame


def test_build_preserves_exact_sample_schema(tmp_path: Path):
    sample = tmp_path / "sample.csv"
    output = tmp_path / "submission.csv"
    sample_frame().to_csv(sample, index=False)
    build_submission(sample, output, method="constant")
    result = pd.read_csv(output)
    assert list(result.columns) == [ID_COLUMN, *TARGET_COLUMNS]
    assert result[ID_COLUMN].tolist() == ["S1", "S2", "S3"]


@pytest.mark.parametrize("mutation,expected", [
    (lambda df: df.drop(columns=["ACL"]), "Missing columns"),
    (lambda df: df.assign(extra=1), "Unexpected extra columns"),
    (lambda df: df.iloc[::-1], "Identifier order mismatch"),
    (lambda df: df.assign(ACL=np.nan), "non-finite"),
    (lambda df: df.assign(ACL=1.1), "outside [0,1]"),
])
def test_validator_rejects_invalid_submissions(tmp_path, mutation, expected):
    sample = sample_frame()
    submission = mutation(sample.copy())
    sample_path = tmp_path / "sample.csv"
    submission_path = tmp_path / "submission.csv"
    sample.to_csv(sample_path, index=False)
    submission.to_csv(submission_path, index=False)
    report = validate_submission(submission_path, sample_path)
    assert not report["ok"]
    assert any(expected in error for error in report["errors"])
```

- [ ] **Step 2: Verify failures against the old three-target assumptions**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_submission_workflow.py -q`

Expected: identifier-order and non-finite checks fail.

- [ ] **Step 3: Implement strict validation**

Update `validate_submission` to compare exact columns, row counts, duplicate IDs, identifier order, numeric conversion, finite values, and range. Build output by copying the sample frame and replacing only inferred target columns.

```python
expected_columns = list(sample_df.columns)
if list(submission_df.columns) != expected_columns:
    missing = [c for c in expected_columns if c not in submission_df]
    extra = [c for c in submission_df if c not in expected_columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected extra columns: {', '.join(extra)}")
    if not missing and not extra:
        errors.append("Column order mismatch")
if ID_COLUMN in submission_df and submission_df[ID_COLUMN].duplicated().any():
    errors.append("Duplicate identifiers")
if ID_COLUMN in submission_df and submission_df[ID_COLUMN].tolist() != sample_df[ID_COLUMN].tolist():
    errors.append("Identifier order mismatch")
for target in TARGET_COLUMNS:
    if target not in submission_df:
        continue
    values = pd.to_numeric(submission_df[target], errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        errors.append(f"{target}: non-finite values")
    elif ((values < 0) | (values > 1)).any():
        errors.append(f"{target}: values outside [0,1]")
```

- [ ] **Step 4: Make PowerShell stop after a failed build**

```powershell
& $python src\submission_workflow.py build --sample-submission $Sample --submission $Submission --method constant
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python src\submission_workflow.py validate --sample-submission $Sample --submission $Submission
exit $LASTEXITCODE
```

Replace `data/sample_submission_example.csv` with three synthetic IDs and all twelve target columns set to `0.5`.

- [ ] **Step 5: Run tests and helper script**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_submission_workflow.py -q`

Run: `powershell -ExecutionPolicy Bypass -File scripts/test_pipeline.ps1 -Submission C:\Users\suraj2\.codex\visualizations\2026\08\07\019fde62-7bbe-7861-a1b8-db7875685b49\submission.csv`

Expected: tests pass and helper reports a valid 12-target submission.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/submission_workflow.py scripts/test_pipeline.ps1 data/sample_submission_example.csv tests/test_submission_workflow.py
git commit -m "fix: enforce live Kaggle submission schema"
```

## Task 3: Implement deterministic DICOM preprocessing

**Files:**
- Create: `src/rsna_knee/data.py`
- Create: `tests/conftest.py`
- Create: `tests/test_dicom_data.py`

- [ ] **Step 1: Add a synthetic DICOM fixture and failing tests**

```python
# tests/conftest.py
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def write_dicom(path: Path, pixels: np.ndarray, instance: int, monochrome1=False):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.InstanceNumber = instance
    ds.RescaleSlope = 2
    ds.RescaleIntercept = -10
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
```

```python
# tests/test_dicom_data.py
import numpy as np
from rsna_knee.data import decode_normalized_slice, sample_triplet_indices, sort_dicom_paths


def test_sort_uses_instance_number(tmp_path):
    from conftest import write_dicom
    for instance in (3, 1, 2):
        write_dicom(tmp_path / f"{instance}.dcm", np.full((8, 8), instance), instance)
    ordered = sort_dicom_paths(list(tmp_path.glob("*.dcm")))
    assert [path.name for path in ordered] == ["1.dcm", "2.dcm", "3.dcm"]


def test_decode_normalizes_and_inverts_monochrome1(tmp_path):
    from conftest import write_dicom
    path = tmp_path / "slice.dcm"
    write_dicom(path, np.arange(64).reshape(8, 8), 1, monochrome1=True)
    image = decode_normalized_slice(path, image_size=16)
    assert image.shape == (16, 16)
    assert np.isfinite(image).all()
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert image[0, 0] > image[-1, -1]


def test_triplet_indices_are_deterministic_and_in_bounds():
    first = sample_triplet_indices(20, triplets=3)
    assert first == sample_triplet_indices(20, triplets=3)
    assert all(0 <= index < 20 for triplet in first for index in triplet)
```

- [ ] **Step 2: Run tests and observe missing functions**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_dicom_data.py -q`

Expected: import failure for DICOM functions.

- [ ] **Step 3: Implement sorting, decoding, and triplet selection**

```python
# src/rsna_knee/data.py
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image


def _sort_key(path: Path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        position = tuple(float(x) for x in getattr(ds, "ImagePositionPatient", ()))
        if position:
            return (0, position[-1], str(path))
        return (1, int(getattr(ds, "InstanceNumber", 0)), str(path))
    except Exception:
        return (2, 0, str(path))


def sort_dicom_paths(paths):
    return sorted((Path(path) for path in paths), key=_sort_key)


def decode_normalized_slice(path: Path, image_size: int):
    ds = pydicom.dcmread(path, force=True)
    pixels = ds.pixel_array.astype(np.float32)
    pixels = pixels * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(pixels, [1.0, 99.0])
    if high <= low:
        normalized = np.zeros_like(pixels, dtype=np.float32)
    else:
        normalized = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        normalized = 1.0 - normalized
    image = Image.fromarray((normalized * 255).astype(np.uint8)).resize((image_size, image_size))
    return np.asarray(image, dtype=np.float32) / 255.0


def sample_triplet_indices(slice_count: int, triplets: int):
    if slice_count <= 0:
        return []
    centers = np.linspace(0, slice_count - 1, num=triplets + 2, dtype=int)[1:-1]
    return [tuple(min(max(center + offset, 0), slice_count - 1) for offset in (-1, 0, 1)) for center in centers]
```

- [ ] **Step 4: Run DICOM tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_dicom_data.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/rsna_knee/data.py tests/conftest.py tests/test_dicom_data.py
git commit -m "feat: add deterministic DICOM preprocessing"
```

## Task 4: Build study sampling, labels, and deterministic splits

**Files:**
- Modify: `src/rsna_knee/data.py`
- Create: `tests/test_study_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

```python
# tests/test_study_dataset.py
import numpy as np
import pandas as pd
import torch

from rsna_knee.constants import ID_COLUMN, TARGET_COLUMNS
from rsna_knee.data import StudyDataset, split_studies


def test_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame({ID_COLUMN: [f"S{i}" for i in range(30)]})
    for index, target in enumerate(TARGET_COLUMNS):
        frame[target] = [(row + index) % 3 == 0 for row in range(30)]
    train_a, val_a = split_studies(frame, validation_fraction=0.2, seed=42)
    train_b, val_b = split_studies(frame, validation_fraction=0.2, seed=42)
    assert train_a[ID_COLUMN].tolist() == train_b[ID_COLUMN].tolist()
    assert val_a[ID_COLUMN].tolist() == val_b[ID_COLUMN].tolist()
    assert set(train_a[ID_COLUMN]).isdisjoint(val_a[ID_COLUMN])


def test_dataset_returns_triplet_and_mask(synthetic_study):
    dataset = StudyDataset(**synthetic_study, image_size=32, training=False)
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["target"].shape == (12,)
    assert item["mask"].dtype == torch.bool
```

Add this fixture to `tests/conftest.py` before running the dataset tests:

```python
import pandas as pd
import pytest
from rsna_knee.constants import ID_COLUMN, TARGET_COLUMNS


@pytest.fixture
def synthetic_study(tmp_path):
    study_id = "S1"
    series_id = "Q1"
    series_dir = tmp_path / study_id / series_id
    series_dir.mkdir(parents=True)
    for instance in (1, 2, 3):
        pixels = np.arange(64, dtype=np.uint16).reshape(8, 8) + instance
        write_dicom(series_dir / f"{instance}.dcm", pixels, instance)
    studies = pd.DataFrame({ID_COLUMN: [study_id]})
    for index, target in enumerate(TARGET_COLUMNS):
        studies[target] = [1.0 if index == 0 else np.nan]
    series = pd.DataFrame({
        ID_COLUMN: [study_id],
        "SeriesInstanceUID": [series_id],
        "Anatomical_Plane": ["Sagittal"],
        "Fluid_Sensitive": [1],
        "Fat_Suppression": [0],
    })
    return {"studies": studies, "series": series, "dicom_root": tmp_path}
```

- [ ] **Step 2: Confirm tests fail**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_study_dataset.py -q`

Expected: missing `StudyDataset` and `split_studies`.

- [ ] **Step 3: Implement study-level sampling and masks**

`StudyDataset` will precompute `(study_id, series_id, triplet)` samples from `train_series.csv`, prefer one series per plane, decode the triplet, return a three-channel float tensor, repeat the study targets, and set `mask = isfinite(targets)`. It will catch per-file decoding errors, try the next valid slice, and record skipped counts. `split_studies` will use a seeded greedy assignment that balances each observed target count while preserving study-level separation.

```python
def _labels(row):
    values = row.loc[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
    mask = np.isfinite(values)
    return np.nan_to_num(values, nan=0.0), mask


def split_studies(frame, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    desired = max(1, round(len(frame) * validation_fraction))
    label_matrix = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
    observed = np.isfinite(label_matrix)
    positive = np.nan_to_num(label_matrix, nan=0.0)
    rarity = (positive / np.maximum(positive.sum(axis=0), 1)).sum(axis=1)
    ranked = sorted(order, key=lambda index: (-rarity[index], index))
    val_indices = sorted(ranked[:desired])
    train_indices = sorted(set(range(len(frame))) - set(val_indices))
    return frame.iloc[train_indices].reset_index(drop=True), frame.iloc[val_indices].reset_index(drop=True)
```

- [ ] **Step 4: Run dataset tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_study_dataset.py -q`

Expected: all tests pass and no study overlap.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/rsna_knee/data.py tests/conftest.py tests/test_study_dataset.py
git commit -m "feat: add MRI study sampling and splits"
```

## Task 5: Implement compact CNN, masked loss, and aggregation

**Files:**
- Create: `src/rsna_knee/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing model tests**

```python
# tests/test_model.py
import torch
from rsna_knee.model import KneeCNN, masked_bce_with_logits, aggregate_study_probabilities


def test_model_output_shape():
    assert KneeCNN()(torch.rand(2, 3, 64, 64)).shape == (2, 12)


def test_masked_loss_ignores_missing_targets():
    logits = torch.zeros(2, 12, requires_grad=True)
    targets = torch.zeros(2, 12)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, :2] = True
    loss = masked_bce_with_logits(logits, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0


def test_aggregation_averages_each_study():
    probabilities = torch.tensor([[0.2] * 12, [0.6] * 12, [0.9] * 12])
    result = aggregate_study_probabilities(["A", "A", "B"], probabilities)
    assert result["A"][0] == 0.4
    assert result["B"][0] == 0.9
```

- [ ] **Step 2: Verify failures**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_model.py -q`

Expected: import failure for model functions.

- [ ] **Step 3: Implement model and loss**

```python
# src/rsna_knee/model.py
import torch
from torch import nn
from torch.nn import functional as F
from .constants import NUM_TARGETS


class KneeCNN(nn.Module):
    def __init__(self, num_targets=NUM_TARGETS):
        super().__init__()
        channels = (3, 32, 64, 128, 192)
        blocks = []
        for source, target in zip(channels, channels[1:]):
            blocks.extend([
                nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(target), nn.SiLU(),
                nn.Conv2d(target, target, 3, padding=1, groups=target, bias=False),
                nn.BatchNorm2d(target), nn.SiLU(),
            ])
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[-1], num_targets)

    def forward(self, images):
        return self.head(self.pool(self.features(images)).flatten(1))


def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    selected = losses[mask]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    return selected.mean()


def aggregate_study_probabilities(study_ids, probabilities):
    grouped = {}
    for study_id, values in zip(study_ids, probabilities.detach().cpu()):
        grouped.setdefault(study_id, []).append(values)
    return {study_id: torch.stack(rows).mean(0).numpy() for study_id, rows in grouped.items()}
```

- [ ] **Step 4: Run model tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_model.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/rsna_knee/model.py tests/test_model.py
git commit -m "feat: add compact knee MRI classifier"
```

## Task 6: Implement training, AUC metrics, and checkpoints

**Files:**
- Create: `src/rsna_knee/training.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write failing training tests**

```python
# tests/test_training.py
import numpy as np
import torch
from rsna_knee.training import compute_auc_report, seed_everything, save_checkpoint


def test_auc_skips_targets_without_both_classes():
    targets = np.array([[0, 1], [1, 1], [0, 1]], dtype=float)
    predictions = np.array([[0.1, 0.7], [0.9, 0.8], [0.2, 0.9]])
    mask = np.ones_like(targets, dtype=bool)
    report = compute_auc_report(targets, predictions, mask, ("scored", "constant"))
    assert report["per_target"]["scored"] == 1.0
    assert report["per_target"]["constant"] is None
    assert report["macro_auc"] == 1.0


def test_checkpoint_contains_configuration(tmp_path):
    from rsna_knee.config import RunConfig
    from rsna_knee.model import KneeCNN
    path = tmp_path / "model.pt"
    save_checkpoint(path, KneeCNN(), RunConfig.smoke(), epoch=1, macro_auc=0.6)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["config"]["mode"] == "smoke"
    assert saved["epoch"] == 1
```

- [ ] **Step 2: Confirm failures**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_training.py -q`

Expected: missing training helpers.

- [ ] **Step 3: Implement deterministic training helpers**

```python
# src/rsna_knee/training.py
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_auc_report(targets, predictions, mask, target_names):
    per_target = {}
    for index, name in enumerate(target_names):
        selected = mask[:, index]
        truth = targets[selected, index]
        score = predictions[selected, index]
        per_target[name] = float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else None
    valid = [value for value in per_target.values() if value is not None]
    return {"per_target": per_target, "macro_auc": float(np.mean(valid)) if valid else None}


def save_checkpoint(path, model, config, epoch, macro_auc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch, "macro_auc": macro_auc}, path)
```

Add `train_one_epoch` and `evaluate` loops using `torch.autocast`, gradient scaling when CUDA is active, masked loss, and study-level aggregation. Add early stopping after two epochs without improvement.

- [ ] **Step 4: Run training tests and a tiny synthetic epoch**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_training.py -q`

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_study_dataset.py tests/test_model.py tests/test_training.py -q`

Expected: all tests pass in CPU mode.

- [ ] **Step 5: Commit Task 6**

```powershell
git add src/rsna_knee/training.py tests/test_training.py
git commit -m "feat: add reproducible MRI model training"
```

## Task 7: Implement inference and fallback-safe submission creation

**Files:**
- Create: `src/rsna_knee/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write failing inference tests**

```python
# tests/test_inference.py
import numpy as np
import pandas as pd
from rsna_knee.constants import ID_COLUMN, TARGET_COLUMNS
from rsna_knee.inference import build_prediction_frame, training_prevalence


def test_prevalence_ignores_missing_and_clips():
    train = pd.DataFrame({target: [0.0, 1.0, np.nan] for target in TARGET_COLUMNS})
    values = training_prevalence(train)
    assert np.allclose(values, 0.5)


def test_prediction_frame_starts_from_sample_order():
    sample = pd.DataFrame({ID_COLUMN: ["B", "A"]})
    for target in TARGET_COLUMNS:
        sample[target] = 0.5
    predicted = {"A": np.full(12, 0.8)}
    fallback = np.full(12, 0.3)
    result = build_prediction_frame(sample, predicted, fallback)
    assert result[ID_COLUMN].tolist() == ["B", "A"]
    assert result.loc[0, "ACL"] == 0.3
    assert result.loc[1, "ACL"] == 0.8
```

- [ ] **Step 2: Verify failures**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_inference.py -q`

Expected: missing inference module.

- [ ] **Step 3: Implement prevalence and sample-first construction**

```python
# src/rsna_knee/inference.py
import numpy as np
from .constants import ID_COLUMN, TARGET_COLUMNS


def training_prevalence(train_frame, minimum=0.05, maximum=0.95):
    values = train_frame.loc[:, TARGET_COLUMNS].mean(skipna=True).fillna(0.5).to_numpy(float)
    return np.clip(values, minimum, maximum)


def build_prediction_frame(sample_frame, predicted_by_study, fallback):
    output = sample_frame.copy()
    rows = []
    for study_id in output[ID_COLUMN].astype(str):
        rows.append(np.asarray(predicted_by_study.get(study_id, fallback), dtype=float))
    values = np.clip(np.vstack(rows), 0.0, 1.0)
    for index, target in enumerate(TARGET_COLUMNS):
        output[target] = values[:, index]
    return output
```

Add `predict_loader(model, loader, device)` that returns study-level averaged probabilities and logs fallback study IDs without exposing competition data in Git artifacts.

- [ ] **Step 4: Run inference and complete suite**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_inference.py -q`

Run: `$env:PYTHONPATH='src'; python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 7**

```powershell
git add src/rsna_knee/inference.py tests/test_inference.py
git commit -m "feat: add fallback-safe study inference"
```

## Task 8: Create the human-readable GitHub notebook

**Files:**
- Create: `scripts/build_notebook.py`
- Create: `notebooks/rsna-knee-2-5d-baseline.ipynb`
- Create: `tests/test_notebook.py`

- [ ] **Step 1: Write failing notebook-content tests**

```python
# tests/test_notebook.py
import nbformat


def test_notebook_is_readable_and_rule_compliant():
    notebook = nbformat.read("notebooks/rsna-knee-2-5d-baseline.ipynb", as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "Research use only" in text
    assert "RUN_MODE = \"smoke\"" in text
    assert "sample_submission.csv" in text
    assert "/kaggle/working/submission.csv" in text
    assert "os.walk('/kaggle/input')" not in text
    assert "validate_submission" in text
    headings = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    assert any("## Training" in heading for heading in headings)
    assert any("## Submission checks" in heading for heading in headings)
```

- [ ] **Step 2: Verify the notebook is absent**

Run: `python -m pytest tests/test_notebook.py -q`

Expected: file-not-found failure.

- [ ] **Step 3: Build the notebook deterministically**

`scripts/build_notebook.py` will create cells with `nbformat` and write the notebook with stable metadata. The code cells will:

```python
RUN_MODE = "smoke"
COMPETITION_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
PROJECT_CANDIDATES = sorted(Path("/kaggle/input").glob("**/src/rsna_knee"))
if PROJECT_CANDIDATES:
    sys.path.insert(0, str(PROJECT_CANDIDATES[0].parents[1]))
else:
    sys.path.insert(0, "../src")
config = RunConfig.smoke() if RUN_MODE == "smoke" else RunConfig.baseline()
```

The input-check cell will locate CSV files, confirm the twelve columns, print only file counts and a short schema summary, and raise a plain-English error for missing paths. Training and inference cells will call package functions. The final cell will validate and write `/kaggle/working/submission.csv`.

- [ ] **Step 4: Generate and validate the notebook**

Run: `python scripts/build_notebook.py`

Run: `python -m pytest tests/test_notebook.py -q`

Run: `python -m jupyter nbconvert --to notebook --execute notebooks/rsna-knee-2-5d-baseline.ipynb --output-dir C:\Users\suraj2\.codex\visualizations\2026\08\07\019fde62-7bbe-7861-a1b8-db7875685b49 --ExecutePreprocessor.timeout=300`

Expected: structure tests pass. Local execution stops at the explicit missing-Kaggle-input check with the documented message; pure package tests remain green.

- [ ] **Step 5: Commit Task 8**

```powershell
git add scripts/build_notebook.py notebooks/rsna-knee-2-5d-baseline.ipynb tests/test_notebook.py
git commit -m "feat: add readable Kaggle MRI notebook"
```

## Task 9: Documentation, packaging, and data-safety checks

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Modify: `scripts/bootstrap_env.ps1`
- Create: `tests/test_repository_safety.py`

- [ ] **Step 1: Write failing repository-safety tests**

```python
# tests/test_repository_safety.py
from pathlib import Path


def test_gitignore_blocks_sensitive_and_generated_artifacts():
    text = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in ("*.dcm", "train.csv", "train_series.csv", "test.csv", "test_series.csv", "submission.csv", "*.pt", "*.pth", "kaggle.json"):
        assert pattern in text


def test_readme_documents_required_workflow():
    text = Path("README.md").read_text(encoding="utf-8")
    for phrase in ("Research use only", "smoke mode", "baseline mode", "internet disabled", "Save & Run All", "submission.csv"):
        assert phrase.lower() in text.lower()
```

- [ ] **Step 2: Verify documentation tests fail**

Run: `python -m pytest tests/test_repository_safety.py -q`

Expected: missing safety patterns and workflow sections.

- [ ] **Step 3: Update ignore rules and README**

Add exact ignore patterns for DICOMs, competition CSVs, checkpoints, notebook output, submissions, Kaggle credentials, caches, and temporary archives. Document installation, local test commands, package layout, notebook regeneration, smoke-first execution, baseline promotion criteria, Kaggle GPU/internet settings, Save & Run All, license safeguards, and the final submission confirmation boundary.

Update `scripts/bootstrap_env.ps1` to install `requirements-dev.txt`, run a package import check, and report Kaggle authentication without printing credentials.

- [ ] **Step 4: Run documentation and safety checks**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_repository_safety.py -q`

Run: `git status --short`

Expected: tests pass and no generated DICOM, checkpoint, credential, or submission files appear.

- [ ] **Step 5: Commit Task 9**

```powershell
git add .gitignore README.md scripts/bootstrap_env.ps1 tests/test_repository_safety.py
git commit -m "docs: document safe Kaggle ML workflow"
```

## Task 10: Complete local verification

**Files:**
- Modify only if a test exposes a defect in Tasks 1-9.

- [ ] **Step 1: Run formatting-neutral compile checks**

Run: `python -X pycache_prefix=C:\Users\suraj2\.codex\visualizations\2026\08\07\019fde62-7bbe-7861-a1b8-db7875685b49\pycache -m compileall -q src scripts tests`

Expected: exit code 0.

- [ ] **Step 2: Run the full local suite**

Run: `$env:PYTHONPATH='src'; python -m pytest -q`

Expected: all tests pass with no warnings caused by project code.

- [ ] **Step 3: Run the submission helper in a writable artifact directory**

Run: `powershell -ExecutionPolicy Bypass -File scripts/test_pipeline.ps1 -Submission C:\Users\suraj2\.codex\visualizations\2026\08\07\019fde62-7bbe-7861-a1b8-db7875685b49\submission.csv`

Expected: valid 3-row, 12-target output.

- [ ] **Step 4: Review the diff and repository contents**

Run: `git diff --check`

Run: `git status --short --branch`

Run: `git ls-files | rg "(\.dcm$|submission\.csv$|kaggle\.json$|\.pt$|\.pth$)"`

Expected: no whitespace errors, a clean tracked state, and no restricted/generated artifacts.

## Task 11: Run Kaggle smoke mode on real competition data

**Files:**
- Kaggle draft notebook created from `notebooks/rsna-knee-2-5d-baseline.ipynb`.
- Kaggle private code dataset containing only tracked project files.

- [ ] **Step 1: Package only tracked files**

Run: `git archive --format=zip --output=C:\Users\suraj2\.codex\visualizations\2026\08\07\019fde62-7bbe-7861-a1b8-db7875685b49\rsna-knee-code.zip HEAD`

Run: `tar -tf C:\Users\suraj2\.codex\visualizations\2026\08\07\019fde62-7bbe-7861-a1b8-db7875685b49\rsna-knee-code.zip`

Expected: source, tests, notebook, and docs are present; no competition data, credentials, submissions, or checkpoints are present.

- [ ] **Step 2: Update the private Kaggle code dataset**

Use Chromium's Kaggle Upload Data dialog, keep visibility private, upload the tracked archive, title it `RSNA Knee 2.5D CNN Source`, and attach it to the notebook. Creating or replacing the cloud dataset requires action-time confirmation.

- [ ] **Step 3: Import the GitHub-versioned notebook**

Replace the generic Kaggle starter with `notebooks/rsna-knee-2-5d-baseline.ipynb`. Keep `RUN_MODE = "smoke"`, enable a GPU accelerator if quota allows, and leave internet disabled for the saved run.

- [ ] **Step 4: Run smoke mode and inspect each checkpoint**

Expected notebook evidence:

- all twelve targets found;
- 24 or fewer training studies selected;
- DICOM success/skip counts printed without full identifiers;
- tensors have shape `batch x 3 x 128 x 128`;
- one epoch completes with finite loss;
- checkpoint loads successfully;
- validation report lists defined and undefined target AUCs;
- three example test studies receive predictions;
- `/kaggle/working/submission.csv` passes exact validation.

- [ ] **Step 5: Record smoke timing and decoder coverage**

Copy only aggregate timing, file counts, skipped-decoder counts, GPU name, and memory use into the task notes. Do not copy DICOMs, reports, or identifiers into GitHub.

## Task 12: Promote to baseline mode, verify, publish code, and push GitHub

**Files:**
- Modify: `src/rsna_knee/config.py` only if smoke timing requires smaller safe limits.
- Regenerate: `notebooks/rsna-knee-2-5d-baseline.ipynb` if configuration changes.
- Modify: `README.md` with measured aggregate runtime.

- [ ] **Step 1: Calculate safe baseline limits**

Use smoke elapsed time per study and choose `max_train_studies`, epochs, and triplets so estimated training plus hidden-test inference is below 7.5 hours, leaving at least 1.5 hours of safety margin.

- [ ] **Step 2: Apply any timing-driven configuration change with TDD**

Update `test_smoke_configuration_is_small_and_valid` and add an assertion for the chosen baseline limits before changing `RunConfig.baseline()`. Run the test red, make the minimal config change, regenerate the notebook, and run the suite green.

- [ ] **Step 3: Run Kaggle baseline mode**

Set `RUN_MODE = "baseline"`, choose Save Version -> Save & Run All, keep internet disabled, and monitor for the notebook-only constraints. A saved version is an external cloud action and requires action-time confirmation.

- [ ] **Step 4: Verify saved-version artifacts**

Confirm the saved version finished under nine hours, `/kaggle/working/submission.csv` exists, schema validation passes, values are finite and non-constant where model inference succeeded, and no restricted training data is present in outputs.

- [ ] **Step 5: Commit measured configuration and documentation**

```powershell
git add src/rsna_knee/config.py notebooks/rsna-knee-2-5d-baseline.ipynb README.md
git commit -m "perf: tune RSNA baseline from Kaggle smoke timing"
```

- [ ] **Step 6: Run final verification before push**

Run: `$env:PYTHONPATH='src'; python -m pytest -q`

Run: `git diff --check`

Run: `git status --short --branch`

Expected: all tests pass and the branch is clean and ahead of `origin/master` only by intentional commits.

- [ ] **Step 7: Share matching code on Kaggle and push GitHub**

After action-time confirmation, make the tested Kaggle notebook public in the competition Code area so public competition code is shared with competitors. Push `master` to the existing `origin` without force. Verify the remote commit and notebook version match.

- [ ] **Step 8: Keep competition submission separate**

Present the saved-version validation evidence and request explicit confirmation before clicking Kaggle's Submit action. A failed submission consumes one of the five daily submissions, so no automatic submission is allowed.
