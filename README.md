# RSNA Knee Abnormality Detection

Competition link: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

This repository is intended to hold notebooks, scripts, and experiments for this Kaggle competition.

## Layout
- `notebooks/`: exploratory notebooks
- `src/`: training/inference code
- `data/`: manifests, metadata, and sample files (no raw/private data)
- `scripts/`: shell helpers for testing and submission
- `outputs/`: generated submission files

## Quick start: generate a valid submission

1. Place Kaggle `sample_submission.csv` in `data/` (or provide your own path).
2. Build predictions:

```powershell
py src/submission_workflow.py build `
  --sample-submission data/sample_submission.csv `
  --submission outputs/submission.csv `
  --method zero
```

3. Validate before submit:

```powershell
py src/submission_workflow.py validate `
  --sample-submission data/sample_submission.csv `
  --submission outputs/submission.csv
```

4. (Optional) Submit from CLI:

```powershell
py src/submission_workflow.py submit `
  --submission outputs/submission.csv `
  --competition rsna-knee-abnormality-detection `
  --message "baseline"
```

## Helper scripts

- `scripts/test_pipeline.ps1` builds and validates against `data/sample_submission_example.csv`.
- `scripts/submit.ps1` is a thin wrapper over `src/submission_workflow.py submit`.

You can run the local smoke test now:

```powershell
.\scripts\test_pipeline.ps1
```

## Notes

- The repository does not include Kaggle `raw` data.
- The default generator creates a zero baseline for all targets; replace with a real model output when ready.

## Quick DICOM check

The provided file is `.dcm` (DICOM), which is a standard medical imaging format.  
To inspect it:

```powershell
$python = 'C:\\Users\\suraj2\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'
& $python -c "import pydicom; ds=pydicom.dcmread(r'D:\\Downloads\\<your_file>.dcm'); print(ds.Modality, ds.BodyPartExamined, ds.Rows, ds.Columns, ds.PatientID)"
```

A valid `.dcm` should read without exceptions and expose tags like `SOPInstanceUID` and image fields (`Rows`, `Columns`, `PixelSpacing`).

## Environment bootstrap

`powershell
.\scripts\bootstrap_env.ps1
`

This installs numpy/pandas/pydicom/matplotlib/kaggle and reports Kaggle auth status.

