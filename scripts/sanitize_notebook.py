"""Prepare a browser-tested Kaggle notebook for safe source control."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


TRANSIENT_NOTEBOOK_METADATA = {
    "execution",
    "kaggle",
    "papermill",
    "varInspector",
    "widgets",
}
TRANSIENT_CELL_METADATA = {
    "collapsed",
    "execution",
    "scrolled",
    "trusted",
}


def sanitize_notebook(source: Path, destination: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    cells = [cell for cell in notebook.cells if cell.source.strip()]

    overview_cells = [
        cell
        for cell in cells
        if cell.cell_type == "markdown"
        and cell.source.lstrip().startswith("# RSNA Knee 2.5D CNN")
    ]
    if overview_cells:
        overview = overview_cells[0]
        cells = [overview, *[cell for cell in cells if cell is not overview]]

    for cell in cells:
        for key in TRANSIENT_CELL_METADATA:
            cell.metadata.pop(key, None)
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    for key in TRANSIENT_NOTEBOOK_METADATA:
        notebook.metadata.pop(key, None)
    notebook.cells = cells

    nbformat.validate(notebook)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sanitize_notebook(args.source, args.destination)


if __name__ == "__main__":
    main()
