"""
Submission workflow helpers for the RSNA Knee Abnormality Detection competition.

Features:
- Build a submission file from a Kaggle sample_submission template.
- Validate a generated submission against the same template.
- Submit through Kaggle CLI when available.

Usage:
- Build:
  python src/submission_workflow.py build \
    --sample-submission data/sample_submission.csv \
    --submission outputs/submission.csv

- Validate:
  python src/submission_workflow.py validate \
    --submission outputs/submission.csv \
    --sample-submission data/sample_submission.csv

- Submit:
  python src/submission_workflow.py submit \
    --submission outputs/submission.csv \
    --competition rsna-knee-abnormality-detection \
    --message "baseline"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


KNOWN_ID_COLUMNS = {
    "row_id",
    "id",
    "study_id",
    "image_id",
    "prediction_id",
    "patient_id",
    "series_id",
    "study_instance_uid",
    "accession_number",
}


def _normalize(values: Optional[str]) -> List[str]:
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _infer_columns(df: pd.DataFrame, id_columns: Iterable[str] = ()) -> List[str]:
    if id_columns:
        missing = [c for c in id_columns if c not in df.columns]
        if missing:
            raise ValueError(f"ID columns not in sample_submission: {', '.join(missing)}")
        return [c for c in df.columns if c not in id_columns]

    detected_ids = [c for c in df.columns if c.lower() in KNOWN_ID_COLUMNS]
    if detected_ids:
        return [c for c in df.columns if c not in detected_ids]

    if len(df.columns) > 1:
        return list(df.columns[1:])

    return []


def _id_columns(df: pd.DataFrame, id_columns: Iterable[str] = ()) -> List[str]:
    if id_columns:
        missing = [c for c in id_columns if c not in df.columns]
        if missing:
            raise ValueError(f"ID columns not in sample_submission: {', '.join(missing)}")
        return list(id_columns)

    detected_ids = [c for c in df.columns if c.lower() in KNOWN_ID_COLUMNS]
    if detected_ids:
        return detected_ids

    return [df.columns[0]]


def build_submission(
    sample_submission: Path,
    output: Path,
    method: str = "zero",
    id_columns: Optional[List[str]] = None,
    seed: int = 42,
    clip: bool = True,
) -> Path:
    df = _read_csv(sample_submission)
    id_columns = id_columns or []
    targets = _infer_columns(df, id_columns)
    if not targets:
        raise ValueError(
            "Could not detect prediction columns. Provide --id-columns and/or --target-columns."
        )

    if method == "random":
        rng = np.random.default_rng(seed)
        pred = rng.random(size=(len(df), len(targets)))
    elif method == "one":
        pred = np.ones((len(df), len(targets)))
    elif method == "constant":
        pred = np.full((len(df), len(targets)), 0.5)
    elif method == "zero":
        pred = np.zeros((len(df), len(targets)))
    else:
        raise ValueError(
            "method must be one of: zero, one, random, constant"
        )

    pred_df = pd.DataFrame(pred, columns=targets)
    if clip:
        pred_df = pred_df.clip(0.0, 1.0)

    out = df.copy()
    if not targets:
        # no-op; keep template
        return output
    for col in targets:
        out[col] = pred_df[col]

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return output


def validate_submission(submission: Path, sample_submission: Path) -> dict:
    sample_df = _read_csv(sample_submission)
    submission_df = _read_csv(submission)
    ids = _id_columns(sample_df)
    targets = _infer_columns(sample_df, ids)
    missing = [c for c in sample_df.columns if c not in submission_df.columns]
    extra = [c for c in submission_df.columns if c not in sample_df.columns]
    errors = []

    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected extra columns: {', '.join(extra)}")
    if len(submission_df) != len(sample_df):
        errors.append(
            f"Row count mismatch: submission={len(submission_df)} sample={len(sample_df)}"
        )

    for target in targets:
        if target in submission_df:
            non_numeric = pd.to_numeric(submission_df[target], errors="coerce").isna().sum()
            if non_numeric:
                errors.append(f"{target}: {int(non_numeric)} non-numeric values")
            below = (submission_df[target] < 0).sum()
            above = (submission_df[target] > 1).sum()
            if below or above:
                errors.append(
                    f"{target}: values outside [0,1] ({int(below)} below, {int(above)} above)"
                )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "id_columns": ids,
        "target_columns": targets,
        "rows": len(submission_df),
    }


def submit_submission(submission: Path, competition: str, message: str) -> int:
    try:
        result = subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                competition,
                "-f",
                str(submission),
                "-m",
                message,
            ],
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "kaggle CLI not found. Install it and authenticate first."
        )
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["build", "validate", "submit"],
        help="Workflow command to run.",
    )
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--submission", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--id-columns", type=str, default="")
    parser.add_argument("--method", choices=["zero", "one", "random", "constant"], default="zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--message", type=str, default="baseline submission")
    parser.add_argument("--competition", type=str, default="rsna-knee-abnormality-detection")
    parser.add_argument("--clip", action="store_true", default=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    id_columns = _normalize(args.id_columns)

    if args.command == "build":
        output = build_submission(
            sample_submission=args.sample_submission,
            output=args.submission,
            method=args.method,
            id_columns=id_columns,
            seed=args.seed,
            clip=args.clip,
        )
        print(f"Submission created: {output}")
        return 0

    if args.command == "validate":
        report = validate_submission(
            submission=args.submission,
            sample_submission=args.sample_submission,
        )
        if report["ok"]:
            print(
                f"Submission is valid. rows={report['rows']}, "
                f"id_columns={report['id_columns']}, "
                f"target_columns={report['target_columns']}"
            )
            return 0
        print("Validation failed:")
        for err in report["errors"]:
            print(f"- {err}")
        return 1

    if args.command == "submit":
        code = submit_submission(
            submission=args.submission,
            competition=args.competition,
            message=args.message,
        )
        print(f"Submit command exited with code {code}")
        return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
