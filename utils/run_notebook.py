"""execute a notebook headlessly (nbclient, python3 kernel), optionally skipping code cells, and save the executed copy.

  python3 utils/run_notebook.py ensemble_eval.ipynb
  python3 utils/run_notebook.py 0846A_eval.ipynb --keep-going --skip '^quick_patch_outputs ='

--skip REGEX                   drop every code cell whose source matches (repeatable)
--keep-going                   run the remaining cells after a failing one; the exit code is still 1
The executed notebook goes to output/notebook_runs/<name>_<time>.ipynb (the source notebook is not modified).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# utils/platform.py would shadow the stdlib module when this file is run as a script
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook")
    parser.add_argument("--skip", action="append", default=[], help="regex: drop code cells whose source matches")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failing cell (exit code 1 at the end)")
    args = parser.parse_args()

    source = Path(args.notebook).resolve()
    notebook = nbformat.read(source, as_version=4)
    patterns = [re.compile(pattern, re.MULTILINE) for pattern in args.skip]
    kept = []
    for cell in notebook.cells:
        if cell.cell_type == "code" and any(pattern.search(cell.source) for pattern in patterns):
            print(f"[notebook] skip: {cell.source.splitlines()[0] if cell.source else ''}", flush=True)
            continue
        kept.append(cell)
    notebook.cells = kept
    out_dir = REPO / "output" / "notebook_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source.stem}_{time.strftime('%Y%m%d_%H%M%S')}.ipynb"
    client = NotebookClient(notebook, kernel_name="python3", timeout=None, allow_errors=args.keep_going,
                            resources={"metadata": {"path": str(REPO)}})
    failed = False
    started = time.time()
    try:
        client.execute()
    except CellExecutionError as error:
        failed = True
        print(f"[notebook] cell failed: {error}", flush=True)
    finally:
        nbformat.write(notebook, target)
    errors = [cell for cell in notebook.cells if cell.cell_type == "code"
              and any(output.get("output_type") == "error" for output in cell.get("outputs", []))]
    for cell in errors:
        print(f"[notebook] ERROR in cell: {cell.source.splitlines()[0] if cell.source else ''}", flush=True)
    print(f"[notebook] {source.name}: {len(kept)} cells in {(time.time() - started) / 60:.1f} min, "
          f"{len(errors)} failed; executed copy {target}", flush=True)
    sys.exit(1 if failed or errors else 0)


if __name__ == "__main__":
    main()
