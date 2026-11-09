"""stitch each run's two figures (output/single_sheet_results/, utils/single_sheet_figures.py) into one 2x2 grid:

  top     the model on the sheet it trained on   (prediction | composite: the first row of its trained_sheet figure)
  bottom  the same model on the prize sheet       (prediction | composite)

The two figures differ in width, so the narrower one is centred on white. Output: output/single_sheet_results/stitched_2x2/<NN>_<alias>_<sheet>.png

usage: python3 -m utils.single_sheet_stitch
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
RESULTS = Path(__file__).resolve().parent.parent / "output" / "single_sheet_results"


def first_row(image: np.ndarray) -> np.ndarray:
    """rows of a 2-row panel grid down to the end of its first panel row (the first long blank band between the rows)."""
    blank = (image == 255).all(axis=(1, 2))
    runs, start = [], None
    for index, value in enumerate(blank):
        if value and start is None:
            start = index
        if not value and start is not None:
            runs.append((start, index - 1))
            start = None
    gaps = [run for run in runs if run[0] > 0 and run[1] - run[0] > 100]
    return image[:gaps[0][0] + 60]


def pad_width(image: np.ndarray, width: int) -> np.ndarray:
    left = (width - image.shape[1]) // 2
    return np.pad(image, ((0, 0), (left, width - image.shape[1] - left), (0, 0)), constant_values=255)


def main() -> None:
    out = RESULTS / "stitched_2x2"
    out.mkdir(parents=True, exist_ok=True)
    for trained in sorted((RESULTS / "trained_sheet").glob("*.png")):
        prize = RESULTS / "prize_sheet" / trained.name
        if not prize.is_file():
            print(f"[stitch] {trained.name}: no prize-sheet figure, skipped", flush=True)
            continue
        top = first_row(np.asarray(Image.open(trained).convert("RGB")))
        bottom = np.asarray(Image.open(prize).convert("RGB"))
        width = max(top.shape[1], bottom.shape[1])
        grid = np.vstack([pad_width(top, width), pad_width(bottom, width)])
        Image.fromarray(grid).save(out / trained.name, compress_level=3)
        print(f"[stitch] {trained.name}: {grid.shape[1]}x{grid.shape[0]} (top {top.shape[1]}x{top.shape[0]}, bottom {bottom.shape[1]}x{bottom.shape[0]})", flush=True)
    print(f"[stitch] done: {out}", flush=True)


if __name__ == "__main__":
    main()
