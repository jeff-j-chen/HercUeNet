"""one organised folder for the figures of the leave-all-but-one runs (utils/single_sheet_runs.py), plus every model's own sheet.

For each run (numbered in corpus order) the model's final.pth renders the one PHerc0846A sheet it trained on through
0846A_eval.ipynb (prediction | composite, and the inklabel overlay | asf cue of a trained sheet); the prize-sheet figure the
runs already made is copied next to it:

  output/single_sheet_results/prize_sheet/<NN>_<alias>_<sheet name>.png     the model on the prize sheet
  output/single_sheet_results/trained_sheet/<NN>_<alias>_<sheet name>.png   the model on the sheet it trained on

usage: python3 -m utils.single_sheet_figures [--only 20261012073246,...] [--wait-for single_sheet]
  tmux new -d -s single_sheet_figures 'python3 -m utils.single_sheet_figures --wait-for single_sheet'
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils import mix6  # noqa: E402
from utils.single_sheet_runs import FIGURE_ROOT, LOG_ROOT, render_prize_sheet, result_dir, sheets, stream  # noqa: E402

RESULTS = REPO / "output" / "single_sheet_results"


def sheet_name(alias: int) -> str:
    return mix6.SHEETS_0846A[alias][1]


def render_trained_sheet(alias: int, checkpoint: Path, target: Path) -> int:
    """0846A_eval.ipynb with only its setup cells and render(<the run's sheet>) -> target png."""
    import nbformat

    notebook = nbformat.read(REPO / "0846A_eval.ipynb", as_version=4)
    helpers = next(i for i, cell in enumerate(notebook.cells) if "def render_remaining_quick_patches" in cell.source)
    notebook.cells = [*notebook.cells[:helpers + 1], nbformat.v4.new_code_cell(f"render({sheet_name(alias)!r})")]
    work = REPO / "_ves_tmp" / f"single_sheet_render_{alias}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    path = work / "render.ipynb"
    nbformat.write(notebook, path)
    env = dict(os.environ, VESUVIUS_CHECKPOINT=str(checkpoint), VESUVIUS_OUT_DIR=str(work / "out"))
    code = stream([sys.executable, "utils/run_notebook.py", str(path)], LOG_ROOT / f"only_{alias}_trained_sheet_figure.log", env)
    produced = work / "out" / f"{sheet_name(alias)}.png"
    if code == 0 and produced.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), target)
    elif code == 0:
        code = 1
    shutil.rmtree(work, ignore_errors=True)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated sheet aliases (default: all 12)")
    parser.add_argument("--wait-for", default="", help="tmux session to wait for (the training queue) before starting")
    args = parser.parse_args()
    os.chdir(REPO)
    while args.wait_for and subprocess.run(["tmux", "has-session", "-t", f"={args.wait_for}"], capture_output=True).returncode == 0:
        time.sleep(60)
    aliases = [int(item) for item in args.only.split(",") if item.strip()] or sheets()
    LOG_ROOT.mkdir(exist_ok=True)
    failed = []
    for index, alias in enumerate(sheets(), start=1):
        if alias not in aliases:
            continue
        stem = f"{index:02d}_{alias}_{sheet_name(alias)}"
        folder = result_dir(alias)
        if folder is None:
            print(f"[figures] {alias}: no trained model (models/runs_single_sheet/only_{alias}*/final.pth), skipped", flush=True)
            failed.append(alias)
            continue
        prize = FIGURE_ROOT / folder.name / f"{mix6.PRIZE_SHEET}.png"
        if not prize.is_file() and render_prize_sheet(folder) != 0:
            failed.append(alias)
        if prize.is_file():
            (RESULTS / "prize_sheet").mkdir(parents=True, exist_ok=True)
            shutil.copy2(prize, RESULTS / "prize_sheet" / f"{stem}.png")
        trained = RESULTS / "trained_sheet" / f"{stem}.png"
        if trained.is_file():
            print(f"[figures] {alias}: trained-sheet figure exists already", flush=True)
            continue
        started = time.time()
        code = render_trained_sheet(alias, folder / "final.pth", trained)
        print(f"[figures] {alias}: trained-sheet figure {'ok' if code == 0 else f'FAILED (exit {code})'} "
              f"after {(time.time() - started) / 60:.1f} min -> {trained}", flush=True)
        if code != 0:
            failed.append(alias)
    if failed:
        raise SystemExit(f"[figures] failed: {sorted(set(failed))}")
    print(f"[figures] done: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
