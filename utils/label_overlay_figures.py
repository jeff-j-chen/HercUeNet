"""plain 2x2 figures of every training sheet for relabelling, in output/label_figures/<sheet>.png:

  z-score composite (reader-v3 + HercUeNet + v8-in, ensemble_eval.ipynb) | v8-in epoch 1
  the same two with the red inklabel overlay (dilated_inklabels/<alias>.png, as 0846A_eval.ipynb draws it)

[ENSEMBLE_SUBDIR=<output/ensemble folder of an ENSEMBLE_SHEET=train0846a run>] python3 -m utils.label_overlay_figures [sheet name ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils import mix6
from utils.fast_figure import colorize, orient_display, red_overlay, save_panel_grid

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parent.parent
ENSEMBLE = REPO / "output" / "ensemble" / os.environ.get("ENSEMBLE_SUBDIR", "mix28_ps3_c4_r2000_v1")
OUT = REPO / "output" / "label_figures"
BAD = (0.78, 0.86, 0.95)
STRETCH = (1.0, 99.5)
V8IN_FLOOR = 50.0  # the notebook shows v8-in with its median as 0


def render(alias: int, render_id: str, name: str) -> Path:
    folder = ENSEMBLE / name
    z_path, v8in_path = folder / "ensemble_z_score.npy", folder / "cache" / "v8_in__mix28_fine_tune__epoch_1" / "v8in_prediction.npy"
    if not (z_path.is_file() and v8in_path.is_file()):
        raise FileNotFoundError(f"{name}: ensemble not finished ({z_path.name}, v8in cache)")
    mask = cv2.imread(str(REPO / "masks" / f"{render_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
    with Image.open(REPO / "dilated_inklabels" / f"{alias}.png") as image:
        label = np.asarray(image.convert("L")) > 0
    z_score = np.load(z_path).astype(np.float32)
    v8in = np.full(mask.shape, np.nan, np.float32)
    raw = np.load(v8in_path).astype(np.float32)
    rows, cols = min(mask.shape[0], raw.shape[0]), min(mask.shape[1], raw.shape[1])
    v8in[:rows, :cols] = raw[:rows, :cols]
    v8in[~mask] = np.nan
    if label.shape != mask.shape or z_score.shape != mask.shape:
        raise ValueError(f"{name}: mask {mask.shape}, label {label.shape}, z-score {z_score.shape} differ")

    valid = np.isfinite(z_score)
    low, high = np.percentile(z_score[valid], STRETCH)
    z_rgb = colorize(orient_display(z_score, name), "gray_r", BAD, vmin=float(low), vmax=float(high))
    valid = np.isfinite(v8in)
    low, high = np.percentile(v8in[valid], (V8IN_FLOOR, STRETCH[1]))
    shown = np.where(valid, np.clip((v8in - low) / max(float(high - low), 1e-6), 0, 1), np.nan)
    v8in_rgb = colorize(orient_display(shown, name), "gray_r", BAD)
    display_label = orient_display(label, name)
    cell_hw = z_rgb.shape[:2]
    rows_out = [
        [("z-score composite", z_rgb, cv2.INTER_AREA, None), ("v8-in epoch 1", v8in_rgb, cv2.INTER_AREA, None)],
        [("z-score composite | red inklabel overlay", red_overlay(z_rgb, display_label), cv2.INTER_AREA, None),
         ("v8-in epoch 1 | red inklabel overlay", red_overlay(v8in_rgb, display_label), cv2.INTER_AREA, None)],
    ]
    path = OUT / f"{name}.png"
    save_panel_grid(rows_out, cell_hw, str(path), gap_px=24)
    return path


def main() -> None:
    wanted = set(sys.argv[1:])
    failed = []
    for alias in mix6.CUE_TRAIN_0846A:
        render_id, name = mix6.SHEETS_0846A[int(alias)]
        if wanted and name not in wanted:
            continue
        try:
            print(f"Saved {render(int(alias), str(render_id), name)}", flush=True)
        except Exception as error:
            failed.append(name)
            print(f"SKIPPED {name}: {error}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
