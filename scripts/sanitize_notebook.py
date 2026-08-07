"""Prepare a browser-tested Kaggle notebook for safe source control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import nbformat
except ModuleNotFoundError:  # Keep the sanitizer usable in a minimal Python install.
    nbformat = None


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
    if nbformat is not None:
        notebook = nbformat.read(source, as_version=4)
        cells = [cell for cell in notebook.cells if cell.source.strip()]
        get_source = lambda cell: cell.source
        get_type = lambda cell: cell.cell_type
        get_metadata = lambda cell: cell.metadata
    else:
        notebook = json.loads(source.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4
        cells = [
            cell
            for cell in notebook.get("cells", [])
            if "".join(cell.get("source", [])).strip()
        ]
        get_source = lambda cell: "".join(cell.get("source", []))
        get_type = lambda cell: cell.get("cell_type")
        get_metadata = lambda cell: cell.setdefault("metadata", {})

    overview_cells = [
        cell
        for cell in cells
        if get_type(cell) == "markdown"
        and get_source(cell).lstrip().startswith("# RSNA Knee 2.5D CNN")
    ]
    if overview_cells:
        overview = overview_cells[0]
        cells = [overview, *[cell for cell in cells if cell is not overview]]

    for cell in cells:
        for key in TRANSIENT_CELL_METADATA:
            get_metadata(cell).pop(key, None)
        if get_type(cell) == "code":
            if nbformat is not None:
                cell.execution_count = None
                cell.outputs = []
            else:
                cell["execution_count"] = None
                cell["outputs"] = []

    for key in TRANSIENT_NOTEBOOK_METADATA:
        if nbformat is not None:
            notebook.metadata.pop(key, None)
        else:
            notebook.setdefault("metadata", {}).pop(key, None)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if nbformat is not None:
        notebook.cells = cells
        nbformat.validate(notebook)
        nbformat.write(notebook, destination)
    else:
        notebook["cells"] = cells
        serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        json.loads(serialized)
        destination.write_text(serialized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sanitize_notebook(args.source, args.destination)


if __name__ == "__main__":
    main()
