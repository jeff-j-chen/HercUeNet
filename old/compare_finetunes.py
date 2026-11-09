"""Prize-sheet predictions of every saved checkpoint of the reader-v3 / v8-in corpus comparison (mix22 vs 0846A only).

  python3 compare_finetunes.py --kind reader --corpus mix22     # predict (cached) + one PNG per checkpoint
  python3 compare_finetunes.py --kind v8in --corpus 0846a
  python3 compare_finetunes.py --grids                          # convergence grids + summary.csv from every cached prediction

Checkpoints: reader-v3 reader-v3/runs/<corpus>_4000/ckpt_00N000.pth (every 1000 steps), v8-in the trained-epoch<N>.ckpt saved after each
epoch of nader-fork/runs/v8in_<corpus>_5ep. The sheet is fiber_only_20261029d_flatboi (never trained or validated on), read as in
ensemble_eval.ipynb. Output: output/finetune_compare/ (png/ per checkpoint, grid_*.png, summary.csv; cache/ holds the maps).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "old"))
Image.MAX_IMAGE_PIXELS = None

from utils import predict_models  # noqa: E402
from utils.fast_figure import colorize, orient_display, save_panel_grid  # noqa: E402

SHEET = "fiber_only_20261029d_flatboi"
ZARR = REPO / "ves_zarrs2" / f"{SHEET}.zarr"
MASK = REPO / "masks" / "extras" / f"{SHEET}.png"
OUT = REPO / "output" / "finetune_compare"
CORPORA = {"mix22": "mix22", "0846a": "0846A only"}
KINDS = {"reader": "reader-v3", "v8in": "v8-in"}
STRETCH = (1.0, 99.5)
BAD = (0.78, 0.86, 0.95)


def run_dir(kind: str, corpus: str) -> Path:
    return REPO / ("reader-v3" if kind == "reader" else "nader-fork") / "runs" / (f"{corpus}_4000" if kind == "reader" else f"v8in_{corpus}_5ep")


def checkpoints(kind: str, corpus: str) -> list[tuple[int, Path]]:
    """[(step for reader-v3 / epoch for v8-in, checkpoint)] of every saved model of the run, in order."""
    folder = run_dir(kind, corpus)
    found = {}
    if kind == "reader":
        for path in sorted(folder.glob("ckpt_*.pth")):
            found[int(path.stem.split("_")[-1])] = path
    else:
        for path in sorted(folder.glob("runs/*/checkpoints/trained-epoch*.ckpt"), key=lambda p: p.stat().st_mtime):
            found[int(re.search(r"trained-epoch(\d+)", path.name).group(1)) + 1] = path  # the latest run folder wins
    return sorted(found.items())


def label(kind: str, number: int) -> str:
    return f"step {number}" if kind == "reader" else f"epoch {number}"


def predict(kind: str, corpus: str, number: int, checkpoint: Path) -> np.ndarray:
    cache = OUT / "cache" / kind / corpus
    if kind == "reader":
        return predict_models.predict_reader(ZARR, checkpoint, cache / f"step_{number:06d}.tif", batch_size=4)
    return predict_models.predict_v8in(ZARR, checkpoint, cache / f"epoch_{number}", batch_size=int(os.environ.get("VESUVIUS_V8IN_BATCH", 16)))


def to_frame(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, np.nan, np.float32)
    rows, cols = min(mask.shape[0], array.shape[0]), min(mask.shape[1], array.shape[1])
    out[:rows, :cols] = array[:rows, :cols]
    out[~mask] = np.nan
    return out


def stretch(frame: np.ndarray, mask: np.ndarray):
    low, high = np.percentile(frame[mask & np.isfinite(frame)], STRETCH)
    return np.where(np.isfinite(frame), np.clip((frame - low) / max(float(high - low), 1e-6), 0, 1), np.nan), float(low), float(high)


def render(kind: str, corpus: str) -> None:
    """predict every saved checkpoint of one run (cached) and save one stretched PNG each."""
    mask = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE) > 0
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    saved = checkpoints(kind, corpus)
    if not saved:
        raise SystemExit(f"no saved checkpoints in {run_dir(kind, corpus)}")
    for number, checkpoint in saved:
        frame = to_frame(predict(kind, corpus, number, checkpoint), mask)
        stretched, low, high = stretch(frame, mask)
        rgb = colorize(orient_display(stretched, SHEET), "gray_r", BAD)
        path = OUT / "png" / f"{kind}_{corpus}_{label(kind, number).replace(' ', '_')}.png"
        cv2.imwrite(str(path), rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        print(f"[{kind} {corpus} {label(kind, number)}] p{STRETCH[0]:g} {low:.4f} p{STRETCH[1]:g} {high:.4f} -> {path}", flush=True)


def grids() -> None:
    """convergence grids (rows = corpus, columns = checkpoints; stretched and fixed 0..1 scale) and summary.csv."""
    import pandas as pd

    mask = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE) > 0
    rows_csv = []
    for kind, name in KINDS.items():
        maps = {}
        for corpus in CORPORA:
            for number, checkpoint in checkpoints(kind, corpus):
                maps[(corpus, number)] = to_frame(predict(kind, corpus, number, checkpoint), mask)
        if not maps:
            continue
        numbers = sorted({number for _, number in maps})
        for scale in ("stretched", "fixed"):
            rows = []
            for corpus, corpus_name in CORPORA.items():
                row = []
                for number in numbers:
                    frame = maps.get((corpus, number))
                    if frame is None:
                        continue
                    if scale == "stretched":
                        shown, low, high = stretch(frame, mask)
                        note = f"p{STRETCH[0]:g}-p{STRETCH[1]:g} {low:.3f}..{high:.3f}"
                    else:
                        shown, note = frame, "fixed 0..1"
                    rgb = colorize(orient_display(shown, SHEET), "gray_r", BAD)
                    row.append((f"{name} | {corpus_name} | {label(kind, number)} | {note}", rgb, cv2.INTER_AREA, None))
                if row:
                    rows.append(row)
            if rows:
                cell_hw = rows[0][0][1].shape[:2]
                path = OUT / f"grid_{kind}_{scale}.png"
                save_panel_grid(rows, cell_hw, str(path), gap_px=24)
                print(f"Saved {path}", flush=True)
        for (corpus, number), frame in maps.items():
            valid = mask & np.isfinite(frame)
            values = frame[valid]
            other = [c for c in CORPORA if c != corpus][0]
            twin = maps.get((other, number))
            final = maps[(corpus, max(n for c, n in maps if c == corpus))]
            rows_csv.append({
                "model": name, "corpus": CORPORA[corpus], "checkpoint": number, "mean": float(values.mean()), "p50": float(np.median(values)),
                "p99.5": float(np.percentile(values, 99.5)), "frac_above_0.5": float((values > 0.5).mean()),
                "corr_with_other_corpus_same_checkpoint": float(np.corrcoef(values, twin[valid])[0, 1]) if twin is not None else np.nan,
                "corr_with_own_final": float(np.corrcoef(values, final[valid])[0, 1]),
            })
    if rows_csv:
        frame = pd.DataFrame(rows_csv)
        frame.to_csv(OUT / "summary.csv", index=False)
        print(frame.round(3).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=tuple(KINDS))
    parser.add_argument("--corpus", choices=tuple(CORPORA))
    parser.add_argument("--grids", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.kind and args.corpus:
        render(args.kind, args.corpus)
    if args.grids:
        grids()
    if not (args.grids or (args.kind and args.corpus)):
        parser.error("give --kind and --corpus, or --grids")


if __name__ == "__main__":
    main()
